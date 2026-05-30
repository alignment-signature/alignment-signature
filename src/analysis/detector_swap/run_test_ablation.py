"""Step 2 of detector-swap: at the swap-detector's best L (per model), generate
the 125-prompt test set with `pasta.hooks.build_all_ablation_hooks` ablating
v_cross[L*_<swap-detector>].

For one (model alias, swap detector) pair, reads the best_L.json produced by the
corresponding score_train_sweep_<detector>.py and runs ablation generation on
the 25 test prompts (Path A canonical, ids [0..24]) of each of the 5 domains —
125 total per (model, swap-detector).

The swap detector is one of {fdg, binoculars, imbd}. Pangram is the canonical
selector and is not a valid value here — its test bundle comes from the
canonical PASTA pipeline (scaled_ablation/alpha_1.00).

Output:
    data/analysis_results/detector_swap/generations/<alias>/<swap-detector>_L<L*>.json
        (125-record bundle; filename encodes both which detector picked L and which L it picked)

Usage:
    python analysis/detector_swap/run_test_ablation.py --alias llama-3.1-8b-instruct --swap-detector fdg        --gpu 0 --batch-size 4
    python analysis/detector_swap/run_test_ablation.py --alias gemma-2-9b-it         --swap-detector binoculars --gpu 0 --batch-size 4
    python analysis/detector_swap/run_test_ablation.py --alias qwen2.5-7b-instruct   --swap-detector imbd       --gpu 0 --batch-size 4
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

from _batched_generation import chat_prompt_ids_batch, generate_once_batch

from pasta.generation import build_generation_record
from pasta.hooks import build_all_ablation_hooks
from pasta.models import load_model
from pasta.prompts import build_strategy_A_prompt, load_prefixes


SWAP_DETECTORS = ("fdg", "binoculars", "imbd")
_BEST_L_KEYS = {
    "fdg": "fdg_best_L",
    "binoculars": "binoculars_best_L",
    "imbd": "imbd_best_L",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "configs" / "models.yaml"))
    p.add_argument("--alias", required=True)
    p.add_argument("--swap-detector", default="fdg", choices=SWAP_DETECTORS,
                   help="Which non-Pangram detector's L*-sweep picks the layer "
                        "to ablate (also determines the output filename "
                        "<swap_detector>_L<L*>.json). Default: fdg.")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--layer", type=int, default=None,
                   help="Override which L to ablate (default: from best_L.json "
                        "of the chosen --swap-detector).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def _select_test_prefix_ids(domain: str, n_test: int) -> list[int]:
    """Path A canonical: first n_test ids per domain (i.e., [0..n_test-1])."""
    prefixes = load_prefixes(domain)
    all_ids = sorted(p["id"] for p in prefixes)
    return all_ids[:n_test]


def main() -> int:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = yaml.safe_load(open(args.config))
    all_models = {m["alias"]: m for m in cfg["models"]}
    if args.alias not in all_models:
        raise SystemExit(f"unknown alias '{args.alias}'. Known: {list(all_models)}")
    spec = all_models[args.alias]
    domains = cfg["domains"]
    n_test = cfg["n_test_per_domain"]
    gen_cfg = cfg["generation"]

    sweep_key = f"{args.swap_detector}_train_sweep_subdir"
    if sweep_key not in cfg["output"]:
        raise SystemExit(
            f"missing 'output.{sweep_key}' in {args.config}; expected one per swap detector"
        )
    sweep_root = REPO_ROOT / cfg["output"][sweep_key] / args.alias
    if args.layer is None:
        best_fp = sweep_root / "best_L.json"
        if not best_fp.exists():
            raise SystemExit(
                f"missing {best_fp}; run score_train_sweep_{args.swap_detector}.py first"
            )
        L_star = json.load(open(best_fp))[_BEST_L_KEYS[args.swap_detector]]
    else:
        L_star = args.layer

    print(f"[run-test] alias={args.alias}  swap_detector={args.swap_detector}  "
          f"L*_{args.swap_detector}={L_star}  Pangram-best L={spec['pangram_best_L']}",
          flush=True)

    direction_dir = REPO_ROOT / spec["direction_dir"]
    direction_fp = direction_dir / f"v_cross_L{L_star}.pt"
    if not direction_fp.exists():
        raise SystemExit(f"missing direction tensor: {direction_fp}")
    direction = torch.load(direction_fp, map_location="cpu")
    if direction.dtype != torch.float32:
        direction = direction.float()

    out_root = REPO_ROOT / cfg["output"]["test_generations_subdir"] / args.alias
    out_fp = out_root / f"{args.swap_detector}_L{L_star}.json"
    if args.skip_existing and out_fp.exists():
        try:
            existing = json.load(open(out_fp))
            if existing.get("num_generations", 0) >= n_test * len(domains):
                print(f"[run-test] {out_fp} already complete; skip")
                return 0
        except Exception:
            pass

    t_load = time.time()
    loaded = load_model(spec["id"], device="cuda:0")
    print(f"[run-test] loaded in {time.time()-t_load:.1f}s "
          f"(n_layers={loaded.n_layers}, d_model={loaded.d_model})", flush=True)
    if loaded.n_layers != spec["n_layers"]:
        raise SystemExit(f"layer count mismatch: cfg={spec['n_layers']} loaded={loaded.n_layers}")

    fwd_pre, fwd_post = build_all_ablation_hooks(loaded, direction)

    bundle_meta = {
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
            "mode": f"all_layer_ablate_at_{args.swap_detector}_best_L",
            "direction_path": str(direction_fp),
            "swap_detector": args.swap_detector,
            f"{args.swap_detector}_best_L": L_star,
            "pangram_best_L_reference": spec["pangram_best_L"],
            "tag": f"v_cross_L{L_star}_{args.swap_detector}_best",
        },
        "n_test_per_domain": n_test,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    if not out_fp.exists():
        with open(out_fp, "w") as f:
            json.dump({**bundle_meta, "num_generations": 0,
                       "num_degenerate": 0, "generations": []},
                      f, indent=2, ensure_ascii=False)

    bundle = json.load(open(out_fp))
    done_ids = {(g["domain"], g["original_prefix_id"])
                for g in bundle.get("generations", [])}

    t_global = time.time()
    grand_total = 0
    for d in domains:
        all_prefixes = load_prefixes(d)
        by_id = {p["id"]: p for p in all_prefixes}
        test_ids = _select_test_prefix_ids(d, n_test)
        entries = [by_id[i] for i in test_ids if (d, i) not in done_ids]
        if not entries:
            print(f"[run-test/{d}] all done — skip", flush=True)
            continue
        print(f"[run-test/{d}] todo {len(entries)}", flush=True)
        t_dom = time.time()
        bs = args.batch_size
        for batch_start in range(0, len(entries), bs):
            chunk = entries[batch_start: batch_start + bs]
            messages_list = [build_strategy_A_prompt(d, e)["aligned_messages"] for e in chunk]
            input_ids, attn = chat_prompt_ids_batch(loaded, messages_list)
            batch_seed = (gen_cfg["seed"] + L_star * 100000 + batch_start
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
                print(f"  ERROR batch start={batch_start}: {type(e).__name__}: {e}",
                      flush=True)
                continue
            bundle = json.load(open(out_fp))
            for entry, res in zip(chunk, batch_results):
                record = build_generation_record(
                    strategy_id="A", domain=d, model_type="instruct",
                    model_id=spec["id"], sample_id=entry["id"],
                    original_prefix_id=entry["id"],
                    prompt_used=build_strategy_A_prompt(d, entry)["aligned_messages"],
                    raw_text=res["raw_text"], finish_reason=res["finish_reason"],
                    usage=res["usage"],
                )
                bundle["generations"].append(record)
                grand_total += 1
            bundle["num_generations"] = len(bundle["generations"])
            bundle["num_degenerate"] = sum(1 for g in bundle["generations"]
                                            if g.get("is_degenerate"))
            with open(out_fp, "w") as f:
                json.dump(bundle, f, indent=2, ensure_ascii=False)
        print(f"[run-test/{d}] done in {time.time()-t_dom:.0f}s", flush=True)

    print(f"\n[run-test {args.alias} done] {grand_total} samples in "
          f"{time.time()-t_global:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
