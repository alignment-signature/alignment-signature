"""All-layers-simultaneous single-site ablation generation driver.

For one aligned model, this script:
  1. Loads the model in bf16.
  2. Loads `data/analysis_results/per_layer/directions/<model>/v_cross_L<L>.pt` for every L.
  3. Registers one forward post-hook on `blocks[L]` per L, each ablating that
     layer's own `v_cross[L]` direction. ALL hooks fire in a single generation pass.
  4. Generates `n_prefixes_per_domain` prompts per domain (selected via
     `pasta.prompts.select_prefix_ids` with the canonical hash-seeded selection).
  5. Writes one bundle per domain to
     `data/analysis_results/per_layer/generations/<model>/all_layers_ablated/<domain>.json`.

Single condition per model — no per-layer sweep here: this ablates every layer
at once in a single pass, rather than one layer at a time across n_layers passes.

Usage:
  python run_all_layers.py --model llama-3.1-8b-instruct --gpu 0
  python run_all_layers.py --model gemma-2-9b-it --gpu 1 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

from _all_layers_hooks import build_all_layers_single_site_hooks
from _batched_generation import chat_prompt_ids_batch, generate_once_batch

from pasta.generation import build_generation_record, chat_prompt_ids, generate_once
from pasta.io import write_record_incremental
from pasta.models import load_model
from pasta.prompts import build_strategy_A_prompt, load_prefixes

DATA_ROOT = REPO_ROOT / "data" / "analysis_results" / "per_layer"
DIRECTIONS_ROOT = DATA_ROOT / "directions"
GENERATIONS_ROOT = DATA_ROOT / "generations"
CONDITION_TAG = "all_layers_ablated"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="Model alias (must match configs/models.yaml).")
    p.add_argument("--gpu", type=int, default=None,
                   help="Single CUDA device id (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--use-auto-device-map", action="store_true",
                   help="Use device_map='auto' (for >32B models that don't fit one GPU).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--config", default=str(HERE / "configs" / "models.yaml"))
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a (model, domain) cell if its output already has the expected number of generations.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None and not args.use_auto_device_map:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = yaml.safe_load(open(args.config))
    spec = next((m for m in cfg["models"] if m["alias"] == args.model), None)
    if spec is None:
        raise SystemExit(f"unknown model alias '{args.model}'. "
                         f"Known: {[m['alias'] for m in cfg['models']]}")

    domains = cfg["domains"]
    n_per_domain = cfg["n_prefixes_per_domain"]
    gen_cfg = cfg["generation"]

    direction_dir = DIRECTIONS_ROOT / spec["alias"]
    out_root = GENERATIONS_ROOT / spec["alias"] / CONDITION_TAG
    if not direction_dir.exists():
        raise SystemExit(f"missing directions dir: {direction_dir}")

    print(f"[run] model={spec['alias']} ({spec['id']})  n_layers={spec['n_layers']}  "
          f"hooks=ALL layers simultaneously  n_per_domain={n_per_domain} "
          f"(first-{n_per_domain}-by-id)  bs={args.batch_size}", flush=True)

    prompts_subdir = cfg.get("prompts_subdir")
    domain_entries: dict[str, list[dict]] = {}
    for domain in domains:
        all_prefixes = load_prefixes(domain, override_subdir=prompts_subdir)
        sel = sorted([p for p in all_prefixes if p["id"] < n_per_domain],
                     key=lambda p: p["id"])
        domain_entries[domain] = sel

    t_load = time.time()
    device_str = "auto" if args.use_auto_device_map else "cuda:0"
    loaded = load_model(spec["id"], device=device_str)
    if loaded.n_layers != spec["n_layers"]:
        raise SystemExit(f"config n_layers={spec['n_layers']} but model has "
                         f"{loaded.n_layers}; update configs/models.yaml")
    print(f"[run] loaded in {time.time()-t_load:.1f}s "
          f"(d_model={loaded.d_model})", flush=True)

    fwd_pre, fwd_post = build_all_layers_single_site_hooks(loaded, direction_dir)
    print(f"[run] registered {len(fwd_post)} hooks "
          f"(one per layer in [0, {loaded.n_layers}))", flush=True)

    bundle_meta_template = {
        "model_type": "instruct",
        "model_id": spec["id"],
        "strategy_id": "A",
        "strategy_name": "naturalistic",
        "quantization": "bf16",
        "generation_params": {
            "temperature":    gen_cfg["temperature"],
            "top_p":          gen_cfg["top_p"],
            "max_new_tokens": gen_cfg["max_new_tokens"],
            "batched":        args.batch_size > 1,
            "batch_size":     args.batch_size,
        },
        "intervention": {
            "mode": "all_layers_output_ablate",
            "layer": "all",
            "alpha": 1.0,
            "tag": CONDITION_TAG,
            "directions_dir": str(direction_dir),
            "n_hooks": len(fwd_post),
        },
    }

    t_global = time.time()
    grand_total = 0
    for domain in domains:
        entries = domain_entries[domain]
        if not entries:
            continue
        out_fp = out_root / f"{domain}.json"
        done_ids: set = set()
        if args.skip_existing and out_fp.exists():
            try:
                out_data = json.load(open(out_fp))
                done_ids = {g.get("original_prefix_id")
                            for g in out_data.get("generations", [])}
            except Exception:
                done_ids = set()

        missing = [e for e in entries if e["id"] not in done_ids]
        if not missing:
            print(f"[{domain}] all {len(entries)} done — skip", flush=True)
            continue
        print(f"[{domain}] {len(done_ids)} done, {len(missing)} to go", flush=True)
        bundle_meta = {**bundle_meta_template, "domain": domain}
        t_dom = time.time()

        if args.batch_size <= 1:
            for entry in missing:
                messages = build_strategy_A_prompt(domain, entry)["aligned_messages"]
                input_ids = chat_prompt_ids(loaded, messages)
                try:
                    result = generate_once(
                        loaded=loaded, input_ids=input_ids,
                        max_new_tokens=gen_cfg["max_new_tokens"],
                        temperature=gen_cfg["temperature"],
                        top_p=gen_cfg["top_p"],
                        fwd_pre_hooks=fwd_pre, fwd_hooks=fwd_post,
                        seed=gen_cfg["seed"] + entry["id"],
                    )
                except Exception as e:
                    print(f"  ERROR id={entry['id']}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                record = build_generation_record(
                    strategy_id="A", domain=domain, model_type="instruct",
                    model_id=spec["id"], sample_id=entry["id"],
                    original_prefix_id=entry["id"], prompt_used=messages,
                    raw_text=result["raw_text"],
                    finish_reason=result["finish_reason"],
                    usage=result["usage"],
                )
                write_record_incremental(out_root, domain, record, bundle_meta)
                grand_total += 1
        else:
            bs = args.batch_size
            for batch_start in range(0, len(missing), bs):
                chunk = missing[batch_start: batch_start + bs]
                messages_list = [build_strategy_A_prompt(domain, e)["aligned_messages"]
                                 for e in chunk]
                input_ids, attn = chat_prompt_ids_batch(loaded, messages_list)
                batch_seed = (gen_cfg["seed"] + batch_start
                              + sum(e["id"] for e in chunk))
                try:
                    batch_results = generate_once_batch(
                        loaded=loaded, input_ids=input_ids, attention_mask=attn,
                        max_new_tokens=gen_cfg["max_new_tokens"],
                        temperature=gen_cfg["temperature"],
                        top_p=gen_cfg["top_p"],
                        fwd_pre_hooks=fwd_pre, fwd_hooks=fwd_post,
                        seed=batch_seed,
                    )
                except Exception as e:
                    print(f"  ERROR batch start={batch_start}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                for entry, res in zip(chunk, batch_results):
                    record = build_generation_record(
                        strategy_id="A", domain=domain, model_type="instruct",
                        model_id=spec["id"], sample_id=entry["id"],
                        original_prefix_id=entry["id"],
                        prompt_used=build_strategy_A_prompt(domain, entry)["aligned_messages"],
                        raw_text=res["raw_text"],
                        finish_reason=res["finish_reason"],
                        usage=res["usage"],
                    )
                    write_record_incremental(out_root, domain, record, bundle_meta)
                    grand_total += 1

        el = time.time() - t_dom
        print(f"  [{domain}] done {len(missing)}  "
              f"elapsed={el:.0f}s  ~{el/max(len(missing),1):.2f}s/sample",
              flush=True)

    print(f"\n[done] {grand_total} samples in {time.time()-t_global:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
