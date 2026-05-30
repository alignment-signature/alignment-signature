"""Detector-quality metrics over the shipped AI-detector scores.

Two quantities per detector, computed from the per-record ``detection.fraction_ai``
scores under ``data/analysis_results/detector_quality/`` (normalised so higher =
more AI; records with ``detection.success`` false are skipped):

1. ``detector_accuracy(detector)`` — how well the detector tells **aligned-model
   text** (label = AI) apart from **human text** (label = human), at a 0.5
   threshold. Reads the ``delta_aligned_human/`` tree (aligned vs. human) and
   returns balanced accuracy ``(TPR + TNR) / 2`` along with its two rates:
     - ``tpr`` = fraction of aligned texts flagged AI (``fraction_ai >= 0.5``)
     - ``tnr`` = fraction of human texts left human (``fraction_ai < 0.5``)

2. ``aligned_base_delta(detector)`` — the mean
   ``Δ = fraction_ai(aligned) − fraction_ai(base)`` over matched
   ``(aligned-model, domain, prefix-id)`` pairs. Reads the
   ``delta_aligned_base/`` tree (aligned vs. base).

Run directly to print both for all six detectors (no API keys or GPU needed):
    python src/analysis/detector_quality/detector_quality.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "analysis_results" / "detector_quality"
QUALITY_TREE = DATA_ROOT / "delta_aligned_human" / "ai-detection"
DELTA_TREE = DATA_ROOT / "delta_aligned_base" / "ai-detection"

DETECTORS = ["pangram", "gptzero", "imbd", "originality", "binoculars", "fast_detectgpt"]

DOMAINS = ["college_essays", "creative_fiction", "news_articles", "opinion_pieces"]

MODEL_PAIRS = {
    "meta-llama_meta-llama-3.1-8b-instruct": "meta-llama_meta-llama-3.1-8b",
    "allenai_olmo-3-7b-instruct":            "allenai_olmo-3-1025-7b",
    "google_gemma-2-9b-it":                  "google_gemma-2-9b",
    "mistralai_mistral-7b-instruct-v0.3":    "mistralai_mistral-7b-v0.3",
    "qwen_qwen2.5-14b-instruct":             "qwen_qwen2.5-14b",
    "qwen_qwen2.5-3b-instruct":              "qwen_qwen2.5-3b",
    "qwen_qwen2.5-7b-instruct":              "qwen_qwen2.5-7b",
    "allenai_llama-3.1-tulu-3-8b":           "meta-llama_meta-llama-3.1-8b",
}

THRESHOLD = 0.5


def _scores_by_prefix(path: Path) -> dict[int, float]:
    """Map ``original_prefix_id -> fraction_ai`` for successful records in one jsonl."""
    scores: dict[int, float] = {}
    if not path.exists():
        return scores
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            det = rec.get("detection") or {}
            if det.get("success") and det.get("fraction_ai") is not None:
                scores[rec["original_prefix_id"]] = float(det["fraction_ai"])
    return scores


def detector_accuracy(detector: str, domains: list[str] = DOMAINS) -> dict:
    """Balanced accuracy @0.5 separating aligned-model text (AI) from human text.

    Returns ``{accuracy, tpr, tnr, n_aligned, n_human}`` pooled over all aligned
    models and the given domains.
    """
    root = QUALITY_TREE / detector
    aligned_vals: list[float] = []
    human_vals: list[float] = []
    for domain in domains:
        human_vals += _scores_by_prefix(root / "human" / f"{domain}.jsonl").values()
        for model_dir in sorted((root / "aligned").glob("*")):
            aligned_vals += _scores_by_prefix(model_dir / f"{domain}.jsonl").values()

    tpr = sum(v >= THRESHOLD for v in aligned_vals) / len(aligned_vals) if aligned_vals else float("nan")
    tnr = sum(v < THRESHOLD for v in human_vals) / len(human_vals) if human_vals else float("nan")
    return {
        "accuracy": 0.5 * (tpr + tnr),
        "tpr": tpr,
        "tnr": tnr,
        "n_aligned": len(aligned_vals),
        "n_human": len(human_vals),
    }


def aligned_base_delta(detector: str, domains: list[str] = DOMAINS) -> dict:
    """Mean Δ = fraction_ai(aligned) − fraction_ai(base) over matched pairs.

    Pairs are matched by ``(aligned-model, domain, prefix-id)`` using the
    aligned↔base mapping in ``MODEL_PAIRS``. Returns ``{delta, n_pairs}``.
    """
    root = DELTA_TREE / detector
    deltas: list[float] = []
    for aligned_dir, base_dir in MODEL_PAIRS.items():
        for domain in domains:
            aligned = _scores_by_prefix(root / "aligned" / aligned_dir / f"{domain}.jsonl")
            base = _scores_by_prefix(root / "base" / base_dir / f"{domain}.jsonl")
            deltas += [aligned[pid] - base[pid] for pid in aligned.keys() & base.keys()]
    delta = sum(deltas) / len(deltas) if deltas else float("nan")
    return {"delta": delta, "n_pairs": len(deltas)}


def main() -> None:
    header = f"{'detector':<16}{'accuracy':>10}{'tpr':>8}{'tnr':>8}{'Δ(aln−base)':>14}{'n_pairs':>9}"
    print(header)
    print("-" * len(header))
    for detector in DETECTORS:
        acc = detector_accuracy(detector)
        dlt = aligned_base_delta(detector)
        print(
            f"{detector:<16}{acc['accuracy']:>10.3f}{acc['tpr']:>8.3f}{acc['tnr']:>8.3f}"
            f"{dlt['delta']:>14.3f}{dlt['n_pairs']:>9d}"
        )


if __name__ == "__main__":
    main()
