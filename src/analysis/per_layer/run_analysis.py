"""Main entry point for the all-layers-simultaneous single-site analysis.

Each model is run as ONE generation pass with per-layer single-site hooks
active at every layer simultaneously (each hook ablates that layer's own
v_cross[L] from the block's output). Discovers visible GPUs and queues one
model per GPU; extras queue and run as GPUs free.

NOTE: Detector scoring is NOT part of this script — invoke
`scripts/detection/evaluate_*.py` separately on the generations after this
script finishes.

Usage (from the repo root):
    # Default: all 3 configured models, auto-detect every visible CUDA device
    python analysis/per_layer/run_analysis.py

    # Pin specific GPUs
    python analysis/per_layer/run_analysis.py --devices cuda:0 cuda:1

    # Subset of models
    python analysis/per_layer/run_analysis.py --models llama-3.1-8b-instruct

    # Sequential on 1 GPU
    python analysis/per_layer/run_analysis.py --devices cuda:0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
LOG_DIR = REPO_ROOT / "data" / "analysis_results" / "per_layer" / "logs"
PYBIN = sys.executable


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--devices", nargs="+", default=None,
                    help="GPU device strings (e.g. cuda:0 cuda:1). "
                         "If omitted, auto-detects via torch.cuda.device_count().")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Model aliases to run (default: all in configs/models.yaml).")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="Force re-running completed cells.")
    ap.add_argument("--config", default=str(HERE / "configs" / "models.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def discover_devices() -> list[str]:
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    if n == 0:
        print("[warn] No CUDA devices detected; falling back to cuda:0 "
              "(generation will be very slow without a GPU).")
        return ["cuda:0"]
    return [f"cuda:{i}" for i in range(n)]


def stage_generate(models: list[str], devices: list[str], cfg_path: Path,
                   batch_size: int, skip_existing: bool, dry_run: bool) -> int:
    """Queue (model, GPU) jobs; one model per GPU at a time."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    queue: deque[str] = deque(models)
    free_devices: deque[str] = deque(devices)
    running: dict[subprocess.Popen, tuple[str, str, Path, object, float]] = {}
    rc_overall = 0

    print(f"\n[generate] {len(models)} model(s) across {len(devices)} GPU(s)")

    while queue or running:
        while queue and free_devices:
            model = queue.popleft()
            dev = free_devices.popleft()
            log_path = LOG_DIR / f"run_{model}.log"
            cmd = [PYBIN, str(HERE / "run_all_layers.py"),
                   "--model", model, "--config", str(cfg_path),
                   "--batch-size", str(batch_size)]
            if dev.startswith("cuda:"):
                cmd += ["--gpu", dev.split(":", 1)[1]]
            if skip_existing:
                cmd.append("--skip-existing")
            if dry_run:
                print(f"[dry] {model} on {dev}: {' '.join(cmd)}")
                free_devices.append(dev)
                continue
            log_fp = open(log_path, "w")
            print(f"[launch] {model} on {dev}  → {log_path}")
            p = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT,
                                 cwd=str(REPO_ROOT))
            running[p] = (model, dev, log_path, log_fp, time.time())

        if dry_run:
            return 0
        if not running:
            break

        while True:
            done = []
            for proc, (model, dev, log_path, log_fp, t0) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                log_fp.close()
                el = time.time() - t0
                if rc == 0:
                    print(f"[done]  {model} on {dev}  rc=0  ({el:.0f}s)")
                else:
                    print(f"[FAIL]  {model} on {dev}  rc={rc}  ({el:.0f}s) — see {log_path}")
                    rc_overall |= 1
                free_devices.append(dev)
                done.append(proc)
            for p in done:
                del running[p]
            if done:
                break
            time.sleep(15)

    return rc_overall


def main():
    args = parse_args()

    cfg = yaml.safe_load(open(args.config))
    all_models = [m["alias"] for m in cfg["models"]]
    models = args.models if args.models else all_models
    unknown = [m for m in models if m not in all_models]
    if unknown:
        sys.exit(f"[fatal] unknown model alias(es): {unknown}. Known: {all_models}")
    devices = args.devices or discover_devices()
    print(f"[setup] models={models}  devices={devices}")

    rc = stage_generate(
        models, devices, Path(args.config),
        args.batch_size,
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run,
    )
    if rc:
        print(f"[warn] generate stage had failures (rc={rc}) — continuing")
    return rc


if __name__ == "__main__":
    sys.exit(main())
