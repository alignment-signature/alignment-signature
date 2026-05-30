"""Pangram scoring for the all-layers-simultaneous condition.

Thin wrapper that reuses the canonical Pangram evaluator's `evaluate_domain`
function (from `scripts/detection/evaluate_pangram.py`) but resolves the
generation/results directories for this experiment's output layout:

    generations:  data/analysis_results/per_layer/generations/<model>/all_layers_ablated/<domain>.json
    results:      data/analysis_results/per_layer/detection_results/pangram/<model>/<domain>.json

The results layout is flat (no per-condition subdir) since this experiment
has only one condition per model — `all_layers_ablated`.

The results layout mirrors the canonical Pangram output schema, so downstream
readers pick the scores up uniformly.

`evaluate_domain` already enforces:
  - the 1 req/sec Pangram rate limit (via PangramClient.rate_limit_delay)
  - skipping degenerate samples
  - identical record schema to the canonical Pangram outputs

This wrapper additionally:
  - hard-skips a (model, domain) cell if the per-domain results JSON already
    exists (so re-running never burns extra Pangram credits)
  - parses configs/models.yaml to discover the model alias + domain list,
    matching how `run_all_layers.py` consumes the same config

Usage (from repo root):
    python analysis/per_layer/score_pangram.py --model llama-3.1-8b-instruct
    python analysis/per_layer/score_pangram.py            # all configured models
    python analysis/per_layer/score_pangram.py --dry-run  # cost-free preview
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pasta.vendor.pangram_client import PangramClient

CONDITION_TAG = "all_layers_ablated"
DATA_ROOT = REPO_ROOT / "data" / "analysis_results" / "per_layer"
GENERATIONS_ROOT = DATA_ROOT / "generations"
RESULTS_ROOT = DATA_ROOT / "detection_results" / "pangram"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "configs" / "models.yaml"))
    p.add_argument("--model", default=None,
                   help="Model alias from configs/models.yaml. Default: all configured.")
    p.add_argument("--domain", default="all")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't call the API; just report what would be scored.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "detection"))
    from evaluate_pangram import evaluate_domain, save_results

    cfg = yaml.safe_load(open(args.config))
    aliases = [m["alias"] for m in cfg["models"]]
    if args.model:
        if args.model not in aliases:
            sys.exit(f"unknown model alias '{args.model}'. Known: {aliases}")
        aliases = [args.model]
    domains = cfg["domains"] if args.domain == "all" else [args.domain]

    api_key = os.environ.get("PANGRAM_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: PANGRAM_API_KEY not set (check .env).")
        return 1
    client = PangramClient(api_key) if not args.dry_run else None

    for alias in aliases:
        gen_dir = GENERATIONS_ROOT / alias / CONDITION_TAG
        results_dir = RESULTS_ROOT / alias
        bucket_label = f"per_layer/{alias}"
        print("=" * 60)
        print(f"Pangram eval ({'DRY RUN' if args.dry_run else 'live'})")
        print(f"  model:        {alias}")
        print(f"  generations:  {gen_dir}")
        print(f"  results:      {results_dir}")
        print(f"  domains:      {domains}")
        print("=" * 60)

        for d in domains:
            out_path = results_dir / f"{d}.json"
            if out_path.exists() and not args.dry_run:
                print(f"[skip] {out_path} already exists — not re-scoring "
                      f"(protects Pangram credits).")
                continue
            try:
                res = evaluate_domain(client, gen_dir, bucket_label, d, args.dry_run)
            except FileNotFoundError as e:
                print(f"  ERROR: {e}")
                continue
            if not args.dry_run:
                p_out = save_results(results_dir, d, res)
                print(f"  -> {p_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
