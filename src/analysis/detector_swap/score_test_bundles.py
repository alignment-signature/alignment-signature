"""Cross-detector scoring of detector-swap TEST bundles.

For each (model, layer) cell in the 4-detector cross-comparison, score the
125-prompt Path A test slice with the requested detector. Output schema
mirrors the existing `detection_results/pangram/<alias>/L<L>.json` pattern.

The cells scored per model are the **union** of best-L picked by each
detector's Step 1:

    Pangram-best L  +  FDG-best L  +  Binoculars-best L  +  ImBD-best L

When two detectors agree on a layer, the bundle is scored once and reused
(the output filename L<L>.json is keyed on the layer, not the detector).

Usage:
    python analysis/detector_swap/score_test_bundles.py --detector fdg
    python analysis/detector_swap/score_test_bundles.py --detector binoculars
    python analysis/detector_swap/score_test_bundles.py --detector imbd
    python analysis/detector_swap/score_test_bundles.py --detector pangram
    python analysis/detector_swap/score_test_bundles.py --detector fdg \
        --aliases llama-3.1-8b-instruct gemma-2-9b-it

The Pangram path costs ~125 API calls per (model, new-layer) and is rate-limited
to 1 req/sec; the local detectors (FDG, Binoculars, ImBD) are free and fast.
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

from _test_bundle_cells import (
    best_Ls, load_test_records, unique_cells,
)


class _PangramAdapter:
    name = "pangram"
    rate_limited = True

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
        api_key = os.environ.get("PANGRAM_API_KEY")
        if not api_key:
            raise SystemExit("PANGRAM_API_KEY not set (check .env).")
        from pasta.vendor.pangram_client import PangramClient
        self.client = PangramClient(api_key)

    def score(self, text: str) -> dict:
        det = self.client.check_text(text)
        time.sleep(self.client.rate_limit_delay)
        return {"detection": det, "success": det.get("success", False),
                "fraction_ai": det.get("fraction_ai") if det.get("success") else None}


class _FDGAdapter:
    name = "fast_detect_gpt"
    rate_limited = False

    def __init__(self):
        from evaluate_fast_detect_gpt import build_detector, score_one
        build_detector()
        self._score_one = score_one

    def score(self, text: str) -> dict:
        det = self._score_one(text)
        return {"detection": det, "success": det.get("status") == "ok",
                "prob_machine": det.get("prob_machine")}


class _BinocularsAdapter:
    name = "binoculars"
    rate_limited = False

    def __init__(self, observer="tiiuae/falcon-7b",
                 performer="tiiuae/falcon-7b-instruct", mode="low-fpr"):
        from evaluate_binoculars import Binoculars
        self.detector = Binoculars(observer_id=observer, performer_id=performer, mode=mode)

    def score(self, text: str) -> dict:
        try:
            s = float(self.detector.compute_score(text))
            return {"detection": {"score": s,
                                  "predicted_label": self.detector.predict_label(s),
                                  "status": "ok"},
                    "success": True, "score": s}
        except Exception as e:
            return {"detection": {"status": "error", "error": f"{type(e).__name__}: {e}"},
                    "success": False, "score": None}


class _ImBDAdapter:
    name = "imbd"
    rate_limited = False

    def __init__(self, task="generate", device="cuda:0"):
        from evaluate_imbd import ImBDDetector, IMBD_CKPT, IMBD_REF
        self.detector = ImBDDetector(IMBD_CKPT, IMBD_REF, task, device)

    def score(self, text: str) -> dict:
        det = self.detector.score(text)
        return {"detection": det, "success": det.get("success", False),
                "prob_machine": det.get("prob_machine") if det.get("success") else None,
                "crit": det.get("crit") if det.get("success") else None}


ADAPTERS = {
    "pangram":    _PangramAdapter,
    "fdg":        _FDGAdapter,
    "binoculars": _BinocularsAdapter,
    "imbd":       _ImBDAdapter,
}
DET_SUBDIR = {"pangram": "pangram", "fdg": "fast_detect_gpt",
              "binoculars": "binoculars", "imbd": "imbd"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", required=True, choices=list(ADAPTERS))
    p.add_argument("--config", default=str(HERE / "configs" / "models.yaml"))
    p.add_argument("--aliases", nargs="+", default=None,
                   help="Subset of model aliases (default: all in config).")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    all_aliases = [m["alias"] for m in cfg["models"]]
    aliases = args.aliases or all_aliases
    bad = [a for a in aliases if a not in all_aliases]
    if bad:
        raise SystemExit(f"unknown alias(es): {bad}.  Known: {all_aliases}")

    out_root = REPO_ROOT / "data" / "analysis_results" / "detector_swap" / "detection_results" / DET_SUBDIR[args.detector]

    plan: list[tuple[str, int]] = []
    for alias in aliases:
        for L in unique_cells(alias):
            plan.append((alias, L))
    print(f"[score-test/{args.detector}] {len(plan)} (alias, layer) cells to score:")
    for alias, L in plan:
        print(f"  • {alias}  L={L}")

    print(f"\n[load detector] {args.detector}")
    adapter = ADAPTERS[args.detector]()

    for alias, L in plan:
        out_fp = out_root / alias / f"L{L}.json"
        if args.skip_existing and out_fp.exists():
            print(f"[skip] {out_fp.relative_to(REPO_ROOT)} already exists")
            continue
        try:
            records = load_test_records(alias, L)
        except FileNotFoundError as e:
            print(f"[skip] no test bundle for {alias} L={L}: {e}")
            continue
        bs = best_Ls(alias)
        meta_layer_picked_by = [d for d, l in bs.items() if l == L]
        results = []
        n_ok = n_err = n_skip = 0
        t0 = time.time()
        for r in records:
            sid = r.get("id")
            text = r.get("processed_text") or r.get("raw_text") or ""
            if r.get("is_degenerate") or not text.strip():
                results.append({"id": sid, "skipped": True,
                                "reason": "degenerate" if r.get("is_degenerate") else "empty"})
                n_skip += 1
                continue
            try:
                res = adapter.score(text)
            except Exception as e:
                results.append({"id": sid, "detection": {"status": "error", "error": str(e)}})
                n_err += 1
                continue
            entry = {"id": sid,
                     "sample_id": r.get("sample_id"),
                     "original_prefix_id": r.get("original_prefix_id"),
                     "domain": r.get("domain"),
                     "text_length": len(text),
                     **res}
            results.append(entry)
            if res.get("success"):
                n_ok += 1
            else:
                n_err += 1
        payload = {
            "alias": alias, "layer": L, "detector": adapter.name,
            "n_records": len(records),
            "processed": n_ok, "errors": n_err, "skipped": n_skip,
            "layer_picked_by_detectors": meta_layer_picked_by,
            "results": results,
        }
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        with open(out_fp, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[score-test/{args.detector}] {alias}/L{L}  "
              f"ok={n_ok} err={n_err} skip={n_skip}  ({time.time()-t0:.0f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
