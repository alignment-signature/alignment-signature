"""End-to-end driver for the in-domain PASTA analysis (generation phases).

For each model alias in configs/models.yaml, runs the three generation phases:

  1. extract.py  --phase all     -> baselines + per-domain v_cross_D directions
  2. ablate.py   --split test     -> test-set ablation per (domain, layer)
  3. ablate.py   --split train     -> train-sweep ablation per (domain, layer)

Detector scoring is a SEPARATE step (it needs detector API
keys and/or a GPU, and benefits from being its own pass):

  score.py --detector pangram --split train   # score the train sweep
  score.py --detector <d> --split test         # score the test set per detector

Usage (from the repo root):
  python analysis/in_domain_pasta/run_analysis.py
  python analysis/in_domain_pasta/run_analysis.py --models llama-3.1-8b-instruct --gpus 0 1
  python analysis/in_domain_pasta/run_analysis.py --skip-extract     # directions already on disk
  python analysis/in_domain_pasta/run_analysis.py --skip-train       # test split only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PYBIN = sys.executable
DEFAULT_CONFIG = HERE / "configs" / "models.yaml"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--models", nargs="+", default=None,
                   help="Model aliases to run (default: all in --config).")
    p.add_argument("--gpus", nargs="+", type=int, default=[0],
                   help="GPU ids; the first two are used for the parallel "
                        "aligned/base extraction (default: [0]).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-test", action="store_true",
                   help="Skip the test-split ablation.")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip the train-sweep ablation.")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Force re-running completed cells.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands without running them.")
    return p.parse_args()


def run(cmd: list[str], dry: bool) -> int:
    print("  $ " + " ".join(cmd), flush=True)
    return 0 if dry else subprocess.call(cmd, cwd=str(REPO_ROOT))


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    all_aliases = [m["alias"] for m in cfg["models"]]
    aliases = args.models or all_aliases
    unknown = [a for a in aliases if a not in all_aliases]
    if unknown:
        sys.exit(f"[fatal] unknown model alias(es): {unknown}. Known: {all_aliases}")

    domains = cfg["domains"]
    g0 = args.gpus[0]
    g1 = args.gpus[1] if len(args.gpus) > 1 else args.gpus[0]
    skip = [] if args.no_skip_existing else ["--skip-existing"]

    rc = 0
    for alias in aliases:
        print(f"\n=== {alias} ===", flush=True)
        if not args.skip_extract:
            rc |= run([PYBIN, str(HERE / "extract.py"), "--config", args.config,
                       "--model", alias, "--phase", "all",
                       "--gpu-aligned", str(g0), "--gpu-base", str(g1)] + skip, args.dry_run)
        for split, skip_split in (("test", args.skip_test), ("train", args.skip_train)):
            if skip_split:
                continue
            for domain in domains:
                rc |= run([PYBIN, str(HERE / "ablate.py"), "--config", args.config,
                           "--model", alias, "--domain", domain, "--split", split,
                           "--layers", "all", "--gpu", str(g0),
                           "--batch-size", str(args.batch_size)] + skip, args.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
