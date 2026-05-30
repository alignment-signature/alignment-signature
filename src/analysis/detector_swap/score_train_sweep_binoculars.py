"""Step 1 (Binoculars variant) of detector-swap: score the canonical PASTA
per-layer ablation train sweep with Binoculars to find Binoculars-best L per
model.

Mirrors the FDG version (`score_train_sweep_fdg.py`) — same algorithm,
different detector. Reuses the canonical `Binoculars` class from
`scripts/detection/evaluate_binoculars.py` directly (no vendoring).

For each model M:
    For each layer L = 0..n_layers-1:
        Read existing canonical PASTA train sweep generations, filter to the
        25 canonical PASTA train ids (5 per domain via
        select_prefix_ids(domain, n=5, seed=42)) across all 5 domains, and
        score each with Binoculars (falcon-7b observer / falcon-7b-instruct
        performer).
    L*_bino = argmin_L mean(Binoculars score over 25 prompts)
    (Binoculars is "lower = AI", so argmin = best ablation under Binoculars.)

Outputs:
    data/analysis_results/detector_swap/binoculars_train_sweep/<alias>/L<L>.json
    data/analysis_results/detector_swap/binoculars_train_sweep/<alias>/best_L.json

Binoculars uses both GPUs by design (observer on cuda:0, performer on cuda:1).
Don't override CUDA_VISIBLE_DEVICES.

Usage:
    python analysis/detector_swap/score_train_sweep_binoculars.py
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

from evaluate_binoculars import Binoculars

from _canonical_pasta_train_ids import canonical_pasta_train_ids


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "configs" / "models.yaml"))
    p.add_argument("--aliases", default=None,
                   help="Comma-separated aliases (default: all in config).")
    p.add_argument("--observer", default="tiiuae/falcon-7b")
    p.add_argument("--performer", default="tiiuae/falcon-7b-instruct")
    p.add_argument("--mode", choices=["low-fpr", "accuracy"], default="low-fpr")
    p.add_argument("--max-token-observed", type=int, default=512)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    all_models = {m["alias"]: m for m in cfg["models"]}
    aliases = ([a.strip() for a in args.aliases.split(",")]
               if args.aliases else list(all_models))
    domains = cfg["domains"]

    out_root = REPO_ROOT / "data" / "analysis_results" / "detector_swap" / "binoculars_train_sweep"

    print(f"[score-bino-sweep] observer={args.observer}  performer={args.performer}  "
          f"mode={args.mode}", flush=True)
    detector = Binoculars(
        observer_id=args.observer,
        performer_id=args.performer,
        mode=args.mode,
        max_token_observed=args.max_token_observed,
    )

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
                        score = float(detector.compute_score(text))
                        det = {"status": "ok", "score": score,
                               "predicted_label": detector.predict_label(score)}
                    except Exception as e:
                        det = {"status": "error", "error": str(e)}
                    sample_records.append({"id": g.get("id"), "domain": d, **det})
                    if det.get("status") == "ok":
                        scores.append(score)
            mean = sum(scores) / len(scores) if scores else None
            payload = {
                "alias": alias, "layer": L,
                "n_samples": len(sample_records),
                "n_scored": len(scores),
                "mean_score": mean,
                "binoculars_threshold": detector.threshold,
                "binoculars_mode": detector.mode,
                "results": sample_records,
                "source_dir": str(sweep_dir),
            }
            with open(out_fp, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            per_layer_scores[L] = payload
            mean_str = f"{mean:.4f}" if mean is not None else "na"
            print(f"[score-bino-sweep] {alias} L{L:2d}  mean_score={mean_str}  "
                  f"n_scored={len(scores)}/{len(sample_records)}", flush=True)

        valid = [(L, p["mean_score"]) for L, p in per_layer_scores.items()
                 if p["mean_score"] is not None]
        valid.sort(key=lambda t: t[1])
        if not valid:
            print(f"[score-bino-sweep] ALIAS {alias}: no valid scores!", flush=True)
            continue
        worst_L, worst_mean = valid[0]
        best_L, best_mean = valid[-1]
        best_payload = {
            "alias": alias,
            "binoculars_best_L": int(best_L),
            "binoculars_best_mean_score": float(best_mean),
            "binoculars_worst_L": int(worst_L),
            "binoculars_worst_mean_score": float(worst_mean),
            "all_layer_curve": [
                {"layer": L, "mean_score": m}
                for L, m in sorted(valid, key=lambda t: t[0])
            ],
            "pangram_best_L_reference": spec["pangram_best_L"],
            "binoculars_threshold": detector.threshold,
            "binoculars_mode": detector.mode,
        }
        with open(out_dir / "best_L.json", "w") as f:
            json.dump(best_payload, f, indent=2)
        print(f"\n[score-bino-sweep] {alias}: Binoculars-best L = {best_L} "
              f"(mean={best_mean:.4f})  vs Pangram-best L = "
              f"{spec['pangram_best_L']}  in {time.time()-t_alias:.0f}s", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
