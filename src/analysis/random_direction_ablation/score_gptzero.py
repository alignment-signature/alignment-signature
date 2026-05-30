"""Score GPTZero on the random-direction-ablation generation tree.

Walks every (condition, domain) under
  data/analysis_results/random_direction_ablation/generations/<model>/<condition>/<domain>.json
and writes mirrored detection results to
  data/analysis_results/random_direction_ablation/detection_results/gptzero/<model>/<condition>/<domain>.json

Output schema matches the existing pangram outputs in this analysis so the
detection-result schema is uniform across detectors:
  {"strategy_dir", "model_type", "domain", "total_samples", "processed",
   "errors", "skipped", "dry_run",
   "results": [{"id", "text_length", "is_degenerate",
                "detection": {"success", "fraction_ai", "fraction_human",
                              "prediction_short", "raw_response"}}, ...]}

Resume-safe per file: a (model, condition, domain) cell is skipped if it
already exists with all 25 records scored (success or hard-failure marked).
For partial files, only the missing records are scored and the file is
re-written. To re-score from scratch, delete the target file first.

Usage (from repo root):
  # Score one (model) — covers all 7 conditions × 5 domains = 875 records
  python analysis/random_direction_ablation/score_gptzero.py --model gemma-2-9b

  # Score one (model, condition) — 5 domains × 25 = 125 records. Use this
  # for finer-grained parallelism: a process holding (model, condition)
  # owns 5 disjoint output files, so multiple processes can run without
  # racing on the same file.
  python analysis/random_direction_ablation/score_gptzero.py \\
      --model gemma-2-9b --condition v_random_k0_ablated

  # Score all four models sequentially
  python analysis/random_direction_ablation/score_gptzero.py --all

  # Parallel scheme — fan out one process per (model, condition):
  for m in gemma-2-9b llama-3.1-8b qwen2.5-1.5b qwen2.5-7b; do
    for c in $(ls data/analysis_results/random_direction_ablation/generations/$m); do
      python analysis/random_direction_ablation/score_gptzero.py \\
          --model $m --condition $c &
    done
  done
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "analysis_results" / "random_direction_ablation"
GEN_ROOT = DATA_ROOT / "generations"
DET_ROOT = DATA_ROOT / "detection_results" / "gptzero"

MODELS = ["gemma-2-9b", "llama-3.1-8b", "qwen2.5-1.5b", "qwen2.5-7b"]
DOMAINS = ["college_essays", "creative_fiction", "news_articles",
           "opinion_pieces", "scientific_abstracts"]


def _conditions_for(model: str) -> list[str]:
    """List of condition directory names available for the given model.

    Includes `aligned`, `v_cross_L<L*>_ablated`, and `v_random_k0..4_ablated`.
    Discovered dynamically from the on-disk layout so we don't hard-code L*.
    """
    p = GEN_ROOT / model
    if not p.exists():
        return []
    return sorted(c.name for c in p.iterdir() if c.is_dir())


def _load_api_key() -> str:
    key = os.environ.get("GPTZERO_API_KEY")
    if key:
        return key
    from dotenv import dotenv_values
    for p in [
        REPO_ROOT / ".env",
    ]:
        if p.exists():
            v = dotenv_values(p).get("GPTZERO_API_KEY")
            if v:
                return v
    raise SystemExit("GPTZERO_API_KEY not found in env or any known .env file")


def _normalize_detection(det: dict) -> dict:
    """Map raw GPTZero client output onto the canonical
    {fraction_ai, fraction_human, prediction_short} shape used in this
    analysis. Drops the bulky `raw_response.documents[*].sentences`
    payload to keep output files small."""
    if not det or not det.get("success"):
        return det
    cls = det.get("class_probabilities") or {}
    frac = cls.get("ai")
    if frac is None:
        frac = det.get("completely_generated_prob")
    if frac is None:
        frac = det.get("average_generated_prob")
    if frac is not None:
        frac = float(frac)
        det["fraction_ai"] = frac
        det["fraction_human"] = float(cls.get("human", 1.0 - frac))
        det["prediction_short"] = "AI" if frac >= 0.5 else "Human"
    if "raw_response" in det and isinstance(det["raw_response"], dict):
        doc = (det["raw_response"].get("documents") or [{}])[0]
        det["raw_response"] = {
            "version": det["raw_response"].get("version"),
            "completely_generated_prob": doc.get("completely_generated_prob"),
            "average_generated_prob": doc.get("average_generated_prob"),
            "predicted_class": doc.get("predicted_class"),
            "class_probabilities": doc.get("class_probabilities"),
            "confidence_category": doc.get("confidence_category"),
        }
    return det


def _already_done(out_path: Path) -> set[str]:
    """Return record ids already scored in `out_path` (success only).
    Records with errors/skipped are not memoized so they get retried."""
    if not out_path.exists():
        return set()
    try:
        existing = json.load(open(out_path))
    except Exception:
        return set()
    done: set[str] = set()
    for r in existing.get("results", []):
        det = r.get("detection") or {}
        if det.get("success"):
            done.add(r["id"])
    return done


def _score_domain(client, model: str, condition: str, domain: str) -> dict:
    in_path = GEN_ROOT / model / condition / f"{domain}.json"
    out_path = DET_ROOT / model / condition / f"{domain}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.load(open(in_path))
    gens = data.get("generations", [])
    total = len(gens)

    done = _already_done(out_path)
    todo = [g for g in gens if g["id"] not in done]
    if not todo and out_path.exists():
        return json.load(open(out_path))

    out_results = []
    if out_path.exists():
        try:
            existing = json.load(open(out_path))
            out_results = list(existing.get("results", []))
        except Exception:
            out_results = []
    keyed = {r["id"]: r for r in out_results}

    print(f"  [{model}/{condition}/{domain}] total={total} done={len(done)} todo={len(todo)}", flush=True)

    n_proc, n_err = 0, 0
    for i, g in enumerate(todo, 1):
        text = g.get("processed_text") or g.get("raw_text") or ""
        text_length = len(text)
        is_degenerate = bool(g.get("is_degenerate"))
        rec_id = g["id"]
        if is_degenerate or not text.strip():
            det = {"success": False, "error": "skipped (degenerate or empty)"}
            keyed[rec_id] = {
                "id": rec_id,
                "text_length": text_length,
                "is_degenerate": is_degenerate,
                "detection": det,
            }
            continue
        try:
            det = client.check_text(text)
            det = _normalize_detection(det)
            time.sleep(getattr(client, "rate_limit_delay", 1.0))
        except Exception as e:
            det = {"success": False, "error": str(e)}
        keyed[rec_id] = {
            "id": rec_id,
            "text_length": text_length,
            "is_degenerate": is_degenerate,
            "detection": det,
        }
        if det.get("success"):
            n_proc += 1
        else:
            n_err += 1
        if i % 5 == 0:
            _write_out(out_path, data, model, domain, gens, keyed)

    _write_out(out_path, data, model, domain, gens, keyed)
    return json.load(open(out_path))


def _write_out(out_path: Path, gen_data: dict, model: str, domain: str,
               gens: list[dict], keyed: dict) -> None:
    results = [keyed[g["id"]] for g in gens if g["id"] in keyed]
    processed = sum(1 for r in results
                    if (r.get("detection") or {}).get("success"))
    errors = sum(1 for r in results
                 if not (r.get("detection") or {}).get("success")
                 and not r.get("is_degenerate"))
    skipped = sum(1 for r in results if r.get("is_degenerate"))
    payload = {
        "strategy_dir": gen_data.get("strategy_id") or gen_data.get("strategy_name"),
        "model_type": gen_data.get("model_type"),
        "domain": domain,
        "total_samples": len(gens),
        "processed": processed,
        "errors": errors,
        "skipped": skipped,
        "dry_run": False,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False)


def score_model(model: str, only_condition: str | None = None) -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "detection"))
    from evaluate_gptzero import GPTZeroClient
    client = GPTZeroClient(api_key=_load_api_key())

    conditions = _conditions_for(model)
    if only_condition is not None:
        if only_condition not in conditions:
            print(f"  [{model}] condition {only_condition!r} not present under "
                  f"{GEN_ROOT / model}; available={conditions}", flush=True)
            return
        conditions = [only_condition]
    if not conditions:
        print(f"  [{model}] no conditions on disk under {GEN_ROOT / model}", flush=True)
        return

    t0 = time.time()
    print(f"[{model}] {len(conditions)} conditions × {len(DOMAINS)} domains", flush=True)
    for cond in conditions:
        for d in DOMAINS:
            in_path = GEN_ROOT / model / cond / f"{d}.json"
            if not in_path.exists():
                continue
            _score_domain(client, model, cond, d)
    print(f"[{model}] done in {time.time()-t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, help="Score one model")
    ap.add_argument("--condition", default=None,
                    help="Restrict to a single condition (e.g. v_random_k0_ablated). "
                         "Requires --model. Use this for fine-grained parallelism "
                         "where each process owns a disjoint set of output files.")
    ap.add_argument("--all", action="store_true",
                    help="Score all models sequentially (use background "
                         "shell processes per (model, condition) for max parallelism)")
    args = ap.parse_args()

    if args.condition and not args.model:
        ap.error("--condition requires --model")

    if args.all:
        for m in MODELS:
            score_model(m)
    elif args.model:
        score_model(args.model, only_condition=args.condition)
    else:
        ap.error("specify either --model <name> [--condition <cond>] or --all")


if __name__ == "__main__":
    main()
