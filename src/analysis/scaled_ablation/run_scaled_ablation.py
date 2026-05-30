"""Scaled-ablation generation: scale the v_cross[L*] projection by alpha.

For each (model, alpha) combination, ablate fraction `alpha` of the projection
of every residual-stream activation onto v_cross[L*] (best layer from the prior
multi-model sweep). alpha=0 -> identity (aligned baseline), alpha=1 -> full
ablation (canonical pasta_ablated output), alpha in (0,1) -> partial,
alpha>1 -> over-ablation along the same direction.

This loads the aligned model once, then sweeps over the requested alphas in
sequence (cheap: only the hook factories rebuild between alphas).

Usage:
  python run_scaled_ablation.py --model llama-3.1-8b-instruct --gpu 0
  python run_scaled_ablation.py --model gemma-2-9b-it --gpu 1 \
      --alphas 0.25,0.5,0.75,1.25,1.5,2.0 --prefix-id-cap 25

Outputs:
  data/analysis_results/scaled_ablation/generations/alpha_{a:.2f}/<source>/<domain>.json
where <source> matches the canonical aligned naming so detector buckets line up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

from _hooks import build_scaled_ablation_hooks
from pasta.generation import build_generation_record, chat_prompt_ids, generate_once
from pasta.io import write_record_incremental
from pasta.models import load_model
from pasta.prompts import build_strategy_A_prompt, load_prefixes

DATA_DIR = REPO_ROOT / "data" / "analysis_results" / "scaled_ablation"
GEN_ROOT = DATA_DIR / "generations"
DIRECTIONS_DIR = DATA_DIR / "directions"

DOMAINS = [
    "college_essays", "creative_fiction", "news_articles",
    "opinion_pieces", "scientific_abstracts",
]


def _direction_for(model_name: str, layer: int) -> Path:
    return DIRECTIONS_DIR / model_name / f"v_cross_L{layer}.pt"


MODEL_SPECS = {
    "llama-3.1-8b-instruct": {
        "out": "llama-3.1-8b-instruct",
        "dir": _direction_for("llama-3.1-8b-instruct", 1),
        "id":  "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "L":   1,
        "gpus": 1,
    },
    "gemma-2-9b-it": {
        "out": "gemma-2-9b-it",
        "dir": _direction_for("gemma-2-9b-it", 2),
        "id":  "google/gemma-2-9b-it",
        "L":   2,
        "gpus": 1,
    },
    "qwen2.5-7b-instruct": {
        "out": "qwen2.5-7b-instruct",
        "dir": _direction_for("qwen2.5-7b-instruct", 14),
        "id":  "Qwen/Qwen2.5-7B-Instruct",
        "L":   14,
        "gpus": 1,
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    p.add_argument("--gpu", type=int, default=None,
                   help="Single CUDA device id (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--alphas", default="0.25,0.5,0.75,1.25,1.5,2.0",
                   help="Comma-separated alpha values to sweep.")
    p.add_argument("--prefix-id-cap", type=int, default=25,
                   help="Only generate for prefix IDs in [0, cap).")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed-base", type=int, default=123,
                   help="seed for prefix_id 0; per-sample seed = seed_base + prefix_id.")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (alpha,domain,prefix_id) combos already on disk.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    spec = MODEL_SPECS[args.model]
    instruct_id = spec["id"]
    best_layer  = spec["L"]
    direction_fp = spec["dir"]
    out_dirname  = spec["out"]
    if not direction_fp.exists():
        raise SystemExit(f"missing direction: {direction_fp}")

    direction = torch.load(direction_fp, map_location="cpu")
    if direction.dtype != torch.float32:
        direction = direction.float()

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    print(f"[{args.model}] alphas={alphas}  prefix_cap={args.prefix_id_cap}  "
          f"L={best_layer}  dir_norm={direction.norm().item():.4f}", flush=True)

    t_load = time.time()
    loaded = load_model(instruct_id, device="cuda:0")
    print(f"[{args.model}] loaded in {time.time()-t_load:.1f}s "
          f"(n_layers={loaded.n_layers}, d_model={loaded.d_model})", flush=True)

    domain_prefixes = {}
    for domain in DOMAINS:
        all_prefixes = load_prefixes(domain)
        filt = sorted([p for p in all_prefixes if p["id"] < args.prefix_id_cap],
                      key=lambda p: p["id"])
        domain_prefixes[domain] = filt
        if not filt:
            print(f"[{args.model}/{domain}] WARN: no prefixes with id < "
                  f"{args.prefix_id_cap}", flush=True)

    t_global = time.time()
    grand_total = 0

    for alpha in alphas:
        alpha_tag = f"alpha_{alpha:.2f}"
        out_dir_alpha = GEN_ROOT / out_dirname / alpha_tag
        out_dir_alpha.mkdir(parents=True, exist_ok=True)

        bundle_meta_template = {
            "model_type": "instruct",
            "model_id": instruct_id,
            "strategy_id": "A",
            "strategy_name": "naturalistic",
            "quantization": "bf16",
            "generation_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
            },
            "intervention": {
                "mode": "partial_ablate",
                "direction_path": str(direction_fp),
                "layer": best_layer,
                "alpha": alpha,
                "tag": f"v_cross_L{best_layer}_partial_ablate_alpha{alpha:.2f}",
            },
        }

        fwd_pre_hooks, fwd_hooks = build_scaled_ablation_hooks(loaded, direction, alpha)

        for domain in DOMAINS:
            entries = domain_prefixes[domain]
            if not entries:
                continue

            out_fp = out_dir_alpha / f"{domain}.json"
            done_ids: set = set()
            if out_fp.exists() and args.skip_existing:
                try:
                    out_data = json.load(open(out_fp))
                    done_ids = {g.get("original_prefix_id")
                                for g in out_data.get("generations", [])}
                except Exception:
                    done_ids = set()

            missing = [e for e in entries if e["id"] not in done_ids]
            if not missing:
                print(f"[{args.model}/a={alpha:.2f}/{domain}] all {len(entries)} done — skip",
                      flush=True)
                continue

            print(f"[{args.model}/a={alpha:.2f}/{domain}] "
                  f"{len(done_ids)} done, {len(missing)} to go",
                  flush=True)
            bundle_meta = {**bundle_meta_template, "domain": domain}
            t_dom = time.time()

            for i, entry in enumerate(missing, 1):
                prompts = build_strategy_A_prompt(domain, entry)
                messages = prompts["aligned_messages"]
                input_ids = chat_prompt_ids(loaded, messages)

                sample_id = entry["id"]

                try:
                    result = generate_once(
                        loaded=loaded,
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        fwd_pre_hooks=fwd_pre_hooks,
                        fwd_hooks=fwd_hooks,
                        seed=args.seed_base + entry["id"],
                    )
                except Exception as e:
                    print(f"  [{args.model}/a={alpha:.2f}/{domain}] "
                          f"ERROR id={entry['id']}: {type(e).__name__}: {e}",
                          flush=True)
                    continue

                record = build_generation_record(
                    strategy_id="A", domain=domain, model_type="instruct",
                    model_id=instruct_id, sample_id=sample_id,
                    original_prefix_id=entry["id"], prompt_used=messages,
                    raw_text=result["raw_text"],
                    finish_reason=result["finish_reason"], usage=result["usage"],
                )
                write_record_incremental(out_dir_alpha, domain, record, bundle_meta)
                grand_total += 1

                if i % args.save_every == 0 or i == len(missing):
                    el = time.time() - t_dom
                    print(f"  [{args.model}/a={alpha:.2f}/{domain}] "
                          f"{i}/{len(missing)}  elapsed={el:.0f}s  "
                          f"~{el/i:.2f}s/sample",
                          flush=True)

    print(f"\n[{args.model}] all done — generated {grand_total} samples in "
          f"{time.time()-t_global:.0f}s", flush=True)


if __name__ == "__main__":
    main()
