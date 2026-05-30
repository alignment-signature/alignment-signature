"""Main entry point for the random-direction ablation analysis.

End-to-end driver. Discovers available GPUs, samples random directions
for the requested models, then runs all 7 ablation conditions for each
(no-intervention aligned model + PASTA + 5 random-k) — one model per
GPU, queueing as GPUs free.

NOTE: The full headline run is already on disk under
`data/analysis_results/random_direction_ablation/`; the detection results for the 4 headline
models × 7 conditions × 4 detectors are committed. Re-running this script
only makes sense if you're regenerating from scratch (e.g. on a new HF
checkpoint) or extending to additional models.

Detector scoring is NOT part of this script — invoke
`scripts/detection/evaluate_*.py` separately on the generations after
this script finishes (each detector has its own rate-limit / GPU /
credit concerns and benefits from being its own step).

Usage (from the repo root):
    # Default: 4 headline configs, auto-detect every visible CUDA device
    python analysis/random_direction_ablation/run_analysis.py

    # Pin specific GPUs
    python analysis/random_direction_ablation/run_analysis.py --devices cuda:0 cuda:1

    # Subset of configs (any matching configs/<name>.yaml in this analysis)
    python analysis/random_direction_ablation/run_analysis.py \\
        --configs config_gemma_2_9b config_llama_3_1_8b

    # Sequential on 1 GPU
    python analysis/random_direction_ablation/run_analysis.py --devices cuda:0

    # Re-use existing v_random_*.pt (skip Algorithm 1)
    python analysis/random_direction_ablation/run_analysis.py --no-sample
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CONFIGS_DIR = HERE / "configs"
LOG_DIR = REPO_ROOT / "data" / "analysis_results" / "random_direction_ablation" / "logs"
PYBIN = sys.executable

DEFAULT_CONFIGS = [
    "config_gemma_2_9b",
    "config_llama_3_1_8b",
    "config_qwen2_5_7b",
    "config_qwen2_5_1_5b",
]


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--devices", nargs="+", default=None,
        help="GPU device strings (e.g. cuda:0 cuda:1). "
             "If omitted, auto-detects via torch.cuda.device_count() and "
             "uses every visible device. Falls back to cuda:0 when CUDA is "
             "unavailable so the script still launches (CPU = slow).",
    )
    ap.add_argument(
        "--configs", nargs="+", default=DEFAULT_CONFIGS,
        help="Config stems to run (matches configs/<stem>.yaml). "
             f"Default: {DEFAULT_CONFIGS}",
    )
    ap.add_argument(
        "--no-sample", action="store_true",
        help="Skip Algorithm 1 (re-use existing v_random_k*.pt on disk).",
    )
    ap.add_argument(
        "--no-skip-existing", action="store_true",
        help="Force re-running completed (condition, domain) cells.",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def discover_devices() -> list[str]:
    """Return ['cuda:0', 'cuda:1', ...] for visible CUDA devices, else ['cuda:0']."""
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


def resolve_configs(stems: list[str]) -> list[Path]:
    paths = []
    for s in stems:
        name = s if s.endswith(".yaml") else f"{s}.yaml"
        p = CONFIGS_DIR / name
        if not p.exists():
            sys.exit(f"[fatal] config not found: {p}")
        paths.append(p)
    return paths


def stage_sample(configs: list[Path], dry_run: bool) -> int:
    """Algorithm 1: sample 5 random directions per model. CPU; serial is fine."""
    rc = 0
    for cfg in configs:
        cmd = [PYBIN, str(HERE / "sample_random_directions.py"), "--config", str(cfg)]
        print(f"\n[sample] {cfg.stem}")
        if dry_run:
            print(f"  $ {' '.join(cmd)}"); continue
        rc |= subprocess.call(cmd, cwd=str(REPO_ROOT))
    return rc


def stage_generate(
    configs: list[Path],
    devices: list[str],
    skip_existing: bool,
    dry_run: bool,
) -> int:
    """Algorithm 2: queue (config, GPU) jobs; each GPU runs one model at a time."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    queue: deque[Path] = deque(configs)
    free_devices: deque[str] = deque(devices)
    running: dict[subprocess.Popen, tuple[Path, str, Path, object, float]] = {}
    rc_overall = 0

    print(f"\n[generate] {len(configs)} configs across {len(devices)} GPU(s)")

    while queue or running:
        while queue and free_devices:
            cfg = queue.popleft()
            dev = free_devices.popleft()
            log_path = LOG_DIR / f"run_{cfg.stem}.log"
            cmd = [PYBIN, str(HERE / "run_all_conditions_one_model.py"),
                   "--config", str(cfg), "--device", dev]
            if skip_existing:
                cmd.append("--skip-existing")
            if dry_run:
                print(f"[dry] {cfg.stem} on {dev}: {' '.join(cmd)}")
                free_devices.append(dev)
                continue
            log_fp = open(log_path, "w")
            print(f"[launch] {cfg.stem} on {dev}  → {log_path}")
            p = subprocess.Popen(
                cmd, stdout=log_fp, stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
            )
            running[p] = (cfg, dev, log_path, log_fp, time.time())

        if dry_run:
            return 0

        if not running:
            break

        while True:
            done = []
            for proc, (cfg, dev, log_path, log_fp, t0) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                log_fp.close()
                el = time.time() - t0
                if rc == 0:
                    print(f"[done]  {cfg.stem} on {dev}  rc=0  ({el:.0f}s)")
                else:
                    print(f"[FAIL]  {cfg.stem} on {dev}  rc={rc}  ({el:.0f}s) — see {log_path}")
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

    configs = resolve_configs(args.configs)
    devices = args.devices or discover_devices()
    print(f"[setup] configs={[c.stem for c in configs]}  devices={devices}")

    rc = 0
    if not args.no_sample:
        rc |= stage_sample(configs, args.dry_run)
        if rc:
            print(f"[fatal] sample stage rc={rc}")
            return rc

    rc |= stage_generate(
        configs,
        devices,
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run,
    )
    if rc:
        print(f"[warn] generate stage had failures (rc={rc}) — continuing")

    return rc


if __name__ == "__main__":
    sys.exit(main())
