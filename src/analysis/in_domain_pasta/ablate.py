"""Per-domain ablation generation (test set, or the train sweep).

For one (model, domain) pair and a list of source layers L:

  - load ``v_cross_L<L>.pt`` for that (model, domain),
  - register a 3-site all-layer ablation via ``build_all_ablation_hooks``,
  - generate that domain's prompts (``--split test`` or ``--split train``) under
    the intervention,
  - write ``generations[_train_sweep]/<alias>/<domain>/L<L>.json``.

``--split test`` generates the canonical Path-A test ids ``[0..n_test-1]`` (the
comparison slice vs. the main PASTA direction); ``--split train`` generates the
hash-seeded train ids the direction was extracted from (the per-layer sweep used
to pick each domain's best layer L*_D). The two splits are disjoint by
construction and land in separate output roots.

Per-sample seed is deterministic: single-sample uses ``generation.seed +
original_prefix_id``; batched uses ``generation.seed + L*100000 + batch_start +
sum(ids in batch)`` so a given layer/batch is reproducible.

Each per-layer bundle is persisted incrementally with
``pasta.io.write_record_incremental`` (one append per sample), keyed by the
filename stem ``L<L>``, so a crashed run leaves a resumable partial file and
``--skip-existing`` can pick up the already-generated prefix ids.

Usage:
  python ablate.py --model llama-3.1-8b-instruct --domain college_essays --layers all --gpu 0 --batch-size 4
  python ablate.py --model llama-3.1-8b-instruct --domain college_essays --layers 0..15 --gpu 0
  python ablate.py --model llama-3.1-8b-instruct --domain college_essays --layers 0,5,10 --gpu 0 --skip-existing
"""
from __future__ import annotations

import argparse
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
from _domain_selection import select_test_prefix_ids, select_train_prefix_ids

from pasta.generation import build_generation_record, chat_prompt_ids, generate_once
from pasta.hooks import build_all_ablation_hooks
from pasta.io import read_json, write_record_incremental
from pasta.models import load_model
from pasta.prompts import build_strategy_A_prompt, load_prefixes

DATA_ROOT = REPO_ROOT / "data" / "analysis_results" / "in_domain_pasta"
DIRECTIONS_ROOT = DATA_ROOT / "directions"
GENERATIONS_ROOT = DATA_ROOT / "generations"
GENERATIONS_TRAIN_SWEEP_ROOT = DATA_ROOT / "generations_train_sweep"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "configs" / "models.yaml"))
    p.add_argument("--model", required=True,
                   help="Model alias (must match one of `models:` in --config).")
    p.add_argument("--domain", required=True,
                   help="Domain whose direction is extracted from and tested on.")
    p.add_argument("--gpu", type=int, default=None,
                   help="Single GPU id (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--split", choices=["test", "train"], default="test",
                   help="'test' = the [0..n_test-1] comparison slice; "
                        "'train' = the hash-seeded train ids (the per-layer L*_D sweep).")
    p.add_argument("--layers", default="all",
                   help="'all', '0..15', or '0,5,10'.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def _pick_model(cfg: dict, alias: str) -> dict:
    """Return the `models:` entry whose alias matches, or exit listing the choices."""
    spec = next((m for m in cfg["models"] if m["alias"] == alias), None)
    if spec is None:
        raise SystemExit(f"unknown model alias '{alias}'. "
                         f"Known: {[m['alias'] for m in cfg['models']]}")
    return spec


def parse_layer_spec(spec: str, n_layers: int) -> list[int]:
    """Expand 'all' / 'a..b' / 'a,b,c' into an explicit list of layer indices."""
    s = spec.strip()
    if s == "all":
        return list(range(n_layers))
    if ".." in s:
        a, b = s.split("..", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _build_bundle_meta(spec: dict, domain: str, gen_cfg: dict, batch_size: int,
                       direction_path: Path, n_train: int, layer: int,
                       train_ids: list[int], test_ids: list[int]) -> dict:
    """Bundle header for one (model, domain, layer); matches the legacy shape."""
    return {
        "model_type": "instruct",
        "model_id": spec["id"],
        "strategy_id": "A",
        "strategy_name": "naturalistic",
        "quantization": "bf16",
        "domain": domain,
        "generation_params": {
            "temperature":    gen_cfg["temperature"],
            "top_p":          gen_cfg["top_p"],
            "max_new_tokens": gen_cfg["max_new_tokens"],
            "batched":        batch_size > 1,
            "batch_size":     batch_size,
        },
        "intervention": {
            "mode": "all_layer_ablate_per_domain",
            "direction_path": str(direction_path),
            "extracted_from_domain": domain,
            "extracted_from_n_train": n_train,
            "layer_extracted_from": layer,
            "tag": f"v_cross_{domain}_L{layer}_all_layer_ablated",
        },
        "split": {
            "train_ids": train_ids,
            "test_ids":  test_ids,
            "n_train":   len(train_ids),
            "n_test":    len(test_ids),
        },
    }


def _done_ids(out_fp: Path, skip_existing: bool) -> set:
    """Prefix ids already present in a partial bundle, for --skip-existing resume."""
    if not (skip_existing and out_fp.exists()):
        return set()
    try:
        return {g["original_prefix_id"]
                for g in read_json(out_fp).get("generations", [])}
    except Exception:
        return set()


def _generate_single(loaded, domain, model_id, gen_cfg, fwd_pre, fwd_post,
                     missing, out_root, stem, bundle_meta) -> int:
    """Single-sample path via pasta.generation.generate_once."""
    n = 0
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
            print(f"  ERROR id={entry['id']}: {type(e).__name__}: {e}", flush=True)
            continue
        record = build_generation_record(
            strategy_id="A", domain=domain, model_type="instruct",
            model_id=model_id, sample_id=entry["id"],
            original_prefix_id=entry["id"], prompt_used=messages,
            raw_text=result["raw_text"], finish_reason=result["finish_reason"],
            usage=result["usage"],
        )
        write_record_incremental(out_root, stem, record, bundle_meta)
        n += 1
    return n


def _generate_batched(loaded, domain, model_id, gen_cfg, fwd_pre, fwd_post,
                      missing, batch_size, layer, out_root, stem, bundle_meta) -> int:
    """Batched path via _batched_generation.chat_prompt_ids_batch + generate_once_batch."""
    n = 0
    for batch_start in range(0, len(missing), batch_size):
        chunk = missing[batch_start: batch_start + batch_size]
        messages_list = [build_strategy_A_prompt(domain, e)["aligned_messages"]
                         for e in chunk]
        input_ids, attn = chat_prompt_ids_batch(loaded, messages_list)
        batch_seed = (gen_cfg["seed"] + layer * 100000 + batch_start
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
                model_id=model_id, sample_id=entry["id"],
                original_prefix_id=entry["id"],
                prompt_used=build_strategy_A_prompt(domain, entry)["aligned_messages"],
                raw_text=res["raw_text"], finish_reason=res["finish_reason"],
                usage=res["usage"],
            )
            write_record_incremental(out_root, stem, record, bundle_meta)
            n += 1
    return n


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = yaml.safe_load(open(args.config))
    spec = _pick_model(cfg, args.model)
    if args.domain not in cfg["domains"]:
        raise SystemExit(f"unknown domain '{args.domain}'. choices: {cfg['domains']}")

    n_train = cfg["n_train_per_domain"]
    n_test = cfg["n_test_per_domain"]
    gen_cfg = cfg["generation"]
    prompts_subdir = cfg.get("prompts_subdir")

    train_ids = select_train_prefix_ids(args.domain, n_train, prompts_subdir=prompts_subdir)
    test_ids = select_test_prefix_ids(args.domain, n_test, prompts_subdir=prompts_subdir)
    assert not (set(train_ids) & set(test_ids)), "train and test overlap!"
    target_ids = test_ids if args.split == "test" else train_ids
    print(f"[ablate/{args.domain}] split={args.split}  "
          f"target_ids={target_ids[:5]}…(n={len(target_ids)})", flush=True)

    direction_dir = DIRECTIONS_ROOT / spec["alias"] / args.domain
    if not direction_dir.exists():
        raise SystemExit(f"missing directions for {args.domain}: {direction_dir}. "
                         f"Run extract.py first.")

    layers = parse_layer_spec(args.layers, spec["n_layers"])
    gen_root = GENERATIONS_ROOT if args.split == "test" else GENERATIONS_TRAIN_SWEEP_ROOT
    out_root = gen_root / spec["alias"] / args.domain

    print(f"[ablate/{args.domain}] loading aligned model…", flush=True)
    t_load = time.time()
    loaded = load_model(spec["id"], device="cuda:0")
    print(f"[ablate/{args.domain}] loaded in {time.time()-t_load:.1f}s "
          f"(n_layers={loaded.n_layers}, d_model={loaded.d_model})", flush=True)
    if loaded.n_layers != spec["n_layers"]:
        raise SystemExit(f"config n_layers={spec['n_layers']} but model has {loaded.n_layers}")

    all_prefixes = load_prefixes(args.domain, override_subdir=prompts_subdir)
    by_id = {p["id"]: p for p in all_prefixes}
    target_entries = [by_id[i] for i in target_ids]

    t_global = time.time()
    grand_total = 0
    for L in layers:
        direction_path = direction_dir / f"v_cross_L{L}.pt"
        direction = torch.load(direction_path, map_location="cpu")
        if direction.dtype != torch.float32:
            direction = direction.float()
        fwd_pre, fwd_post = build_all_ablation_hooks(loaded, direction)

        stem = f"L{L}"
        out_fp = out_root / f"{stem}.json"
        missing = [e for e in target_entries
                   if e["id"] not in _done_ids(out_fp, args.skip_existing)]
        if not missing:
            print(f"[ablate/{args.domain}/L{L}] all {len(target_entries)} done — skip",
                  flush=True)
            continue

        bundle_meta = _build_bundle_meta(
            spec, args.domain, gen_cfg, args.batch_size, direction_path,
            n_train, L, train_ids, test_ids,
        )
        t_layer = time.time()
        if args.batch_size > 1:
            grand_total += _generate_batched(
                loaded, args.domain, spec["id"], gen_cfg, fwd_pre, fwd_post,
                missing, args.batch_size, L, out_root, stem, bundle_meta,
            )
        else:
            grand_total += _generate_single(
                loaded, args.domain, spec["id"], gen_cfg, fwd_pre, fwd_post,
                missing, out_root, stem, bundle_meta,
            )
        print(f"[ablate/{args.domain}/L{L}] {len(missing)} done in "
              f"{time.time()-t_layer:.0f}s", flush=True)

    print(f"\n[ablate {args.domain} done] {grand_total} samples in "
          f"{time.time()-t_global:.0f}s", flush=True)


if __name__ == "__main__":
    main()
