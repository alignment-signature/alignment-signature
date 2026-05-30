"""Subset existing α=0 (aligned) and α=1 (best-L ablated) Pangram results
from the n=100 runs down to first 25 samples (sample_id 0..24) and copy them
into the scaled-ablation experiment results tree.

This gives the dose-response curve anchored endpoints without re-running
generation or detection. Output layout matches the new α∈{0.25..2.0} files:

  detection_results/pangram/<source>/alpha_0.00/<domain>.json
  detection_results/pangram/<source>/alpha_1.00/<domain>.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "analysis_results" / "scaled_ablation"
OUT_PANG = DATA_DIR / "detection_results" / "pangram"

DOMAINS = [
    "college_essays", "creative_fiction", "news_articles",
    "opinion_pieces", "scientific_abstracts",
]

SOURCES = [
    ("llama-3.1-8b-instruct",   "llama3.1-8b", "llama-3.1-8b-instruct_best_L1"),
    ("gemma-2-9b-it",           "gemma-2-9b",  "gemma-2-9b-it_best_L2"),
    ("qwen2.5-7b-instruct",     "qwen2.5-7b",  "qwen2.5-7b-instruct_best_L14"),
]

A0_PANG_ROOT = Path(os.environ.get("SCALED_ABLATION_A0_PANG_ROOT", ""))
A1_PANG_ROOT = Path(os.environ.get("SCALED_ABLATION_A1_PANG_ROOT", ""))

LIMIT = 25


def _id_suffix(rec_id: str) -> int | None:
    """Parse the trailing numeric sample-id from 'strategy_A-<domain>-<mt>-<N>'."""
    try:
        return int(rec_id.rsplit("-", 1)[-1])
    except Exception:
        return None


def _filter_records(records: list[dict], limit: int) -> list[dict]:
    out = []
    for r in records:
        sid = r.get("sample_id")
        if sid is None:
            sid = _id_suffix(r.get("id", ""))
        if sid is None or sid >= limit:
            continue
        out.append(r)
    return out


def subset_pangram(in_fp: Path, out_fp: Path, alpha: float, source: str, domain: str):
    if not in_fp.exists():
        print(f"  [miss] {in_fp}")
        return False
    d = json.load(open(in_fp))
    records = d.get("results", [])
    sub = _filter_records(records, LIMIT)
    payload = {
        "source": str(in_fp),
        "domain": domain,
        "alpha": alpha,
        "source_model": source,
        "intervention": d.get("intervention"),
        "model_id": d.get("model_id"),
        "total_samples": len(sub),
        "results": sub,
    }
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  [ok]   {out_fp}  (n={len(sub)})")
    return True


def main():
    if not A0_PANG_ROOT.exists() or not A1_PANG_ROOT.exists():
        sys.exit(
            "Set SCALED_ABLATION_A0_PANG_ROOT and SCALED_ABLATION_A1_PANG_ROOT to the\n"
            "external α=0 / α=1 Pangram source dirs before running."
        )
    n_ok = 0
    n_miss = 0
    for model_dir, a0_alias, a1_alias in SOURCES:
        print(f"\n## {model_dir}")
        for d in DOMAINS:
            in_fp  = A0_PANG_ROOT / a0_alias / f"{d}.json"
            out_fp = OUT_PANG / model_dir / "alpha_0.00" / f"{d}.json"
            ok = subset_pangram(in_fp, out_fp, 0.0, model_dir, d)
            n_ok += ok; n_miss += (not ok)

            in_fp  = A1_PANG_ROOT / a1_alias / f"{d}.json"
            out_fp = OUT_PANG / model_dir / "alpha_1.00" / f"{d}.json"
            ok = subset_pangram(in_fp, out_fp, 1.0, model_dir, d)
            n_ok += ok; n_miss += (not ok)

    print(f"\n[done] ok={n_ok}  miss={n_miss}")


if __name__ == "__main__":
    main()
