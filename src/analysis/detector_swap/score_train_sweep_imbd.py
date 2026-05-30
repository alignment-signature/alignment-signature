"""Step 1 (ImBD variant) of detector-swap: score the canonical PASTA per-layer
ablation train sweep with ImBD to find ImBD-best L per model.

Mirrors the FDG version (`score_train_sweep_fdg.py`) — same algorithm,
different detector. Reuses the canonical `ImBDDetector` class from
`scripts/detection/evaluate_imbd.py` directly (no vendoring).

For each model M:
    For each layer L = 0..n_layers-1:
        Read existing canonical PASTA train sweep generations, filter to the
        25 canonical PASTA train ids (5 per domain via
        select_prefix_ids(domain, n=5, seed=42)) across all 5 domains, and
        score each with ImBD (gpt-neo-2.7B + ImBD-inference LoRA).
    L*_imbd = argmin_L mean(ImBD prob_machine over 25 prompts)

Outputs:
    data/analysis_results/detector_swap/imbd_train_sweep/<alias>/L<L>.json
    data/analysis_results/detector_swap/imbd_train_sweep/<alias>/best_L.json

Requires the ImBD checkpoints under `vendor/ImBD/models/`:
    vendor/ImBD/models/ImBD-inference/      (LoRA adapter)
    vendor/ImBD/models/gpt-neo-2.7B/        (base model)

Download instructions (run once before this script):
    cd vendor/ImBD
    uv run --no-project huggingface-cli download xyzhu1225/ImBD-inference \\
        --local-dir models/ImBD-inference
    uv run --no-project huggingface-cli download EleutherAI/gpt-neo-2.7B \\
        --exclude '*.ot' '*.msgpack' '*.h5' --local-dir models/gpt-neo-2.7B

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/detector_swap/score_train_sweep_imbd.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "detection"))

from evaluate_imbd import ImBDDetector, IMBD_CKPT, IMBD_REF, IMBD_ROOT

from _canonical_pasta_train_ids import canonical_pasta_train_ids


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "configs" / "models.yaml"))
    p.add_argument("--aliases", default=None,
                   help="Comma-separated aliases (default: all in config).")
    p.add_argument("--task", default="generate",
                   help="ImBD calibration task. 'generate' matches free-form LLM output.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def _check_checkpoints():
    if not IMBD_CKPT.exists():
        raise SystemExit(
            f"ImBD checkpoint not found at {IMBD_CKPT}.\n"
            f"Download it: cd vendor/ImBD && uv run --no-project huggingface-cli "
            f"download xyzhu1225/ImBD-inference --local-dir models/ImBD-inference"
        )
    if not (IMBD_ROOT / "models" / "gpt-neo-2.7B").exists():
        raise SystemExit(
            f"gpt-neo-2.7B base not found at {IMBD_ROOT / 'models' / 'gpt-neo-2.7B'}.\n"
            f"Download it: cd vendor/ImBD && uv run --no-project huggingface-cli "
            f"download EleutherAI/gpt-neo-2.7B --exclude '*.ot' '*.msgpack' '*.h5' "
            f"--local-dir models/gpt-neo-2.7B"
        )


def main() -> int:
    args = parse_args()
    _check_checkpoints()

    cfg = yaml.safe_load(open(args.config))
    all_models = {m["alias"]: m for m in cfg["models"]}
    aliases = ([a.strip() for a in args.aliases.split(",")]
               if args.aliases else list(all_models))
    domains = cfg["domains"]

    out_root = REPO_ROOT / "data" / "analysis_results" / "detector_swap" / "imbd_train_sweep"

    print(f"[score-imbd-sweep] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
          f"task={args.task}  device={args.device}", flush=True)
    detector = ImBDDetector(IMBD_CKPT, IMBD_REF, args.task, args.device)

    train_id_sets = {d: canonical_pasta_train_ids(d) for d in domains}

    for alias in aliases:
        spec = all_models[alias]
        n_layers = spec["n_layers"]
        sweep_root = REPO_ROOT / spec["train_sweep_root"]
        sweep_tpl = spec["train_sweep_dir_tpl"]
        out_dir = out_root / alias
        out_dir.mkdir(parents=True, exist_ok=True)

        per_layer_scores: dict[int, dict] = {}
        t_alias = time.time()
        for L in range(n_layers):
            out_fp = out_dir / f"L{L}.json"
            if args.skip_existing and out_fp.exists():
                try:
                    per_layer_scores[L] = json.load(open(out_fp))
                    continue
                except Exception:
                    pass

            sweep_dir = sweep_root / sweep_tpl.format(L=L)
            scores: list[float] = []
            sample_records: list[dict] = []
            for d in domains:
                gen_fp = sweep_dir / f"{d}.json"
                if not gen_fp.exists():
                    continue
                bundle = json.load(open(gen_fp))
                wanted = train_id_sets[d]
                for g in bundle.get("generations", []):
                    pid = g.get("original_prefix_id")
                    if pid not in wanted:
                        continue
                    text = g.get("processed_text") or ""
                    if g.get("is_degenerate") or not text.strip():
                        sample_records.append({
                            "id": g.get("id"), "domain": d,
                            "status": "skipped",
                            "reason": "degenerate" if g.get("is_degenerate") else "empty",
                        })
                        continue
                    try:
                        det = detector.score(text)
                        if not det.get("success"):
                            sample_records.append({
                                "id": g.get("id"), "domain": d, "status": "error",
                                "error": det.get("error", "unknown"),
                            })
                            continue
                        sample_records.append({
                            "id": g.get("id"), "domain": d,
                            "status": "ok", "crit": det["crit"],
                            "prob_machine": det["prob_machine"],
                        })
                        scores.append(det["prob_machine"])
                    except Exception as e:
                        sample_records.append({
                            "id": g.get("id"), "domain": d,
                            "status": "error", "error": f"{type(e).__name__}: {e}",
                        })
            mean = sum(scores) / len(scores) if scores else None
            payload = {
                "alias": alias, "layer": L,
                "n_samples": len(sample_records),
                "n_scored": len(scores),
                "mean_prob_machine": mean,
                "results": sample_records,
                "source_dir": str(sweep_dir),
            }
            with open(out_fp, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            per_layer_scores[L] = payload
            mean_str = f"{mean:.4f}" if mean is not None else "na"
            print(f"[score-imbd-sweep] {alias} L{L:2d}  mean={mean_str}  "
                  f"n_scored={len(scores)}/{len(sample_records)}", flush=True)

        valid = [(L, p["mean_prob_machine"]) for L, p in per_layer_scores.items()
                 if p["mean_prob_machine"] is not None]
        valid.sort(key=lambda t: t[1])
        if not valid:
            print(f"[score-imbd-sweep] ALIAS {alias}: no valid scores!", flush=True)
            continue
        best_L, best_mean = valid[0]
        worst_L, worst_mean = valid[-1]
        best_payload = {
            "alias": alias,
            "imbd_best_L": int(best_L),
            "imbd_best_mean_prob_machine": float(best_mean),
            "imbd_worst_L": int(worst_L),
            "imbd_worst_mean_prob_machine": float(worst_mean),
            "all_layer_curve": [
                {"layer": L, "mean_prob_machine": m}
                for L, m in sorted(valid, key=lambda t: t[0])
            ],
            "pangram_best_L_reference": spec["pangram_best_L"],
        }
        with open(out_dir / "best_L.json", "w") as f:
            json.dump(best_payload, f, indent=2)
        print(f"\n[score-imbd-sweep] {alias}: ImBD-best L = {best_L} (mean={best_mean:.4f})  "
              f"vs Pangram-best L = {spec['pangram_best_L']}  "
              f"in {time.time()-t_alias:.0f}s", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
