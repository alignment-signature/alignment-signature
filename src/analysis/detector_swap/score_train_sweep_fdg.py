"""Step 1 of detector-swap: FDG-score the canonical PASTA per-layer ablation
train sweep generations to find FDG-best L per model.

For each model M:
    For each layer L = 0..n_layers-1:
        Read existing canonical-PASTA train sweep generations:
          {train_sweep_root}/{train_sweep_dir_tpl.format(L=L)}/<domain>.json   for each domain
        Filter to the 25 canonical-PASTA train ids (5 per domain via
          select_prefix_ids(domain, n=5, seed=42)) and FDG-score each.
    L*_fdg = argmin_L mean(FDG prob_machine over 25 prompts)

Outputs:
    data/analysis_results/detector_swap/fdg_train_sweep/<alias>/L<L>.json     (per-layer scores)
    data/analysis_results/detector_swap/fdg_train_sweep/<alias>/best_L.json   (chosen L*_fdg + curve)

Imports the canonical Fast-DetectGPT machinery from
`scripts/detection/evaluate_fast_detect_gpt.py` (`build_detector` +
`score_one`) — no local vendoring.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/detector_swap/score_train_sweep_fdg.py
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

from evaluate_fast_detect_gpt import build_detector, score_one

from _canonical_pasta_train_ids import canonical_pasta_train_ids


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "configs" / "models.yaml"))
    p.add_argument("--aliases", default=None,
                   help="Comma-separated aliases (default: all in config).")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="Force re-scoring layers that already have a JSON output.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    all_models = {m["alias"]: m for m in cfg["models"]}
    aliases = ([a.strip() for a in args.aliases.split(",")]
               if args.aliases else list(all_models))
    domains = cfg["domains"]

    out_root = REPO_ROOT / cfg["output"]["fdg_train_sweep_subdir"]

    print(f"[score-fdg-sweep] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)
    build_detector()

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
                        det = score_one(text)
                    except Exception as e:
                        det = {"status": "error", "error": str(e)}
                    sample_records.append({"id": g.get("id"), "domain": d, **det})
                    if det.get("status") == "ok":
                        scores.append(det.get("prob_machine"))
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
            print(f"[score-fdg-sweep] {alias} L{L:2d}  mean={mean_str}  "
                  f"n_scored={len(scores)}/{len(sample_records)}", flush=True)

        valid = [(L, p["mean_prob_machine"]) for L, p in per_layer_scores.items()
                 if p["mean_prob_machine"] is not None]
        valid.sort(key=lambda t: t[1])
        if not valid:
            print(f"[score-fdg-sweep] ALIAS {alias}: no valid scores!", flush=True)
            continue
        best_L, best_mean = valid[0]
        worst_L, worst_mean = valid[-1]
        best_payload = {
            "alias": alias,
            "fdg_best_L": int(best_L),
            "fdg_best_mean_prob_machine": float(best_mean),
            "fdg_worst_L": int(worst_L),
            "fdg_worst_mean_prob_machine": float(worst_mean),
            "all_layer_curve": [
                {"layer": L, "mean_prob_machine": m}
                for L, m in sorted(valid, key=lambda t: t[0])
            ],
            "pangram_best_L_reference": spec["pangram_best_L"],
        }
        with open(out_dir / "best_L.json", "w") as f:
            json.dump(best_payload, f, indent=2)
        print(f"\n[score-fdg-sweep] {alias}: FDG-best L = {best_L} (mean={best_mean:.4f})  "
              f"vs Pangram-best L = {spec['pangram_best_L']}  "
              f"in {time.time()-t_alias:.0f}s", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
