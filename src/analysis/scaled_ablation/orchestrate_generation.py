"""Orchestrate scaled-ablation generation across multiple GPUs.

Launches one subprocess per (model, gpu) assignment. Each subprocess is
`run_scaled_ablation.py --model <m> --gpu <g> --alphas <list>` and sweeps
all alphas for its model in sequence, loading the model once.

Usage:
  python orchestrate_generation.py --gpus 0,1
  python orchestrate_generation.py --gpus 0,1,2,3 --skip-existing
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "analysis_results" / "scaled_ablation"
EXP_DIR = Path(__file__).resolve().parent
SCRIPT = EXP_DIR / "run_scaled_ablation.py"
LOG_DIR = DATA_DIR / "logs"

MODELS = ["llama-3.1-8b-instruct", "gemma-2-9b-it", "qwen2.5-7b-instruct"]
DEFAULT_ALPHAS = "0.25,0.5,0.75,1.25,1.5,2.0"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default="0,1",
                   help="Comma-separated GPU IDs to use as a worker pool.")
    p.add_argument("--alphas", default=DEFAULT_ALPHAS,
                   help="Comma-separated alpha values to sweep per model.")
    p.add_argument("--prefix-id-cap", type=int, default=25)
    p.add_argument("--models", default=None,
                   help="Comma-separated subset of models (default: all 3).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Pass through to per-model worker.")
    p.add_argument("--python", default=None,
                   help="Override python interpreter (default: this interpreter).")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    models = [m.strip() for m in (args.models.split(",") if args.models else MODELS) if m.strip()]
    py = args.python or sys.executable

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    free_gpus = deque(gpus)
    queue = deque(models)
    running: dict[subprocess.Popen, tuple[str, int, Path, float]] = {}

    print(f"[orchestrate] gpus={gpus}  models={models}  alphas={args.alphas}", flush=True)
    if args.dry_run:
        for m in queue:
            cmd = [py, "-u", str(SCRIPT), "--model", m, "--gpu", "<gpu>",
                   "--alphas", args.alphas, "--prefix-id-cap", str(args.prefix_id_cap)]
            if args.skip_existing:
                cmd.append("--skip-existing")
            print("  $", " ".join(cmd))
        return

    t0 = time.time()
    failures: list[tuple[str, int]] = []

    while queue or running:
        while queue and free_gpus:
            model = queue.popleft()
            gpu = free_gpus.popleft()
            log_fp = LOG_DIR / f"{model}__gpu{gpu}.log"
            cmd = [py, "-u", str(SCRIPT),
                   "--model", model,
                   "--gpu", str(gpu),
                   "--alphas", args.alphas,
                   "--prefix-id-cap", str(args.prefix_id_cap)]
            if args.skip_existing:
                cmd.append("--skip-existing")
            log = open(log_fp, "w", buffering=1)
            env = os.environ.copy()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[proc] = (model, gpu, log_fp, time.time())
            print(f"[orchestrate] launched model={model} gpu={gpu} pid={proc.pid} "
                  f"log={log_fp}", flush=True)

        if running:
            done = []
            while not done:
                for proc in list(running):
                    rc = proc.poll()
                    if rc is None:
                        continue
                    model, gpu, log_fp, t_start = running.pop(proc)
                    el = time.time() - t_start
                    print(f"[orchestrate] DONE model={model} gpu={gpu} rc={rc} "
                          f"elapsed={el:.0f}s log={log_fp}", flush=True)
                    free_gpus.append(gpu)
                    if rc != 0:
                        failures.append((model, rc))
                    done.append(proc)
                if not done:
                    time.sleep(5)

    print(f"\n[orchestrate] all done in {time.time()-t0:.0f}s "
          f"({'no failures' if not failures else f'{len(failures)} failures: {failures}'})",
          flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
