"""Pangram scorer for the scaled-ablation experiment generation tree.

Walks
    data/analysis_results/scaled_ablation/generations/alpha_{a}/<source>/<domain>.json
and writes per-domain Pangram results to
    data/analysis_results/scaled_ablation/detection_results/pangram/alpha_{a}/<source>/<domain>.json

Reuses the vendored PangramClient. API rate-limit handled by the client's
internal sleep. Results are written incrementally (one .json per domain).

Usage:
  PANGRAM_API_KEY=... python score_pangram.py
  python score_pangram.py --alphas 0.25,0.5
  python score_pangram.py --models llama-3.1-8b-instruct_best_L1 --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "analysis_results" / "scaled_ablation"
GEN_ROOT = DATA_DIR / "generations"
OUT_ROOT = DATA_DIR / "detection_results" / "pangram"
sys.path.insert(0, str(REPO_ROOT / "src"))
from pasta.vendor.pangram_client import PangramClient

DOMAINS = [
    "college_essays", "creative_fiction", "news_articles",
    "opinion_pieces", "scientific_abstracts",
]
DEFAULT_ALPHAS = [0.25, 0.5, 0.75, 1.25, 1.5, 2.0]
DEFAULT_SOURCES = [
    "llama-3.1-8b-instruct",
    "gemma-2-9b-it",
    "qwen2.5-7b-instruct",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alphas", default=None,
                   help="Comma-separated alpha values (default: 0.25,0.5,0.75,1.25,1.5,2.0).")
    p.add_argument("--models", default=None,
                   help="Comma-separated source dir names (default: 3 target models).")
    p.add_argument("--domains", default="all")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def evaluate_domain(client: PangramClient | None, gen_path: Path, dry_run: bool) -> dict:
    data = json.load(open(gen_path))
    generations = data.get("generations", [])
    results = []
    n_proc = 0
    n_err = 0
    for i, gen in enumerate(generations, 1):
        sid = gen["id"]
        text = gen.get("processed_text") or ""
        is_deg = bool(gen.get("is_degenerate", False))
        if is_deg or not text.strip():
            results.append({"id": sid, "skipped": True, "reason": "degenerate" if is_deg else "empty"})
            continue
        if dry_run:
            results.append({"id": sid, "text_length": len(text), "dry_run": True})
            continue
        det = client.check_text(text)
        time.sleep(client.rate_limit_delay)
        if det.get("success"):
            n_proc += 1
        else:
            n_err += 1
        results.append({
            "id": sid,
            "sample_id": gen.get("sample_id"),
            "original_prefix_id": gen.get("original_prefix_id"),
            "text_length": len(text),
            "is_degenerate": is_deg,
            "detection": det,
        })
    return {
        "source": str(gen_path),
        "domain": data.get("domain"),
        "model_id": data.get("model_id"),
        "intervention": data.get("intervention"),
        "total_samples": len(generations),
        "processed": n_proc,
        "errors": n_err,
        "skipped": sum(1 for r in results if r.get("skipped")),
        "results": results,
    }


def main():
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    if not args.dry_run and not os.environ.get("PANGRAM_API_KEY"):
        print("ERROR: PANGRAM_API_KEY not set (check .env)")
        return 1
    alphas = [float(a) for a in (args.alphas.split(",") if args.alphas else [str(a) for a in DEFAULT_ALPHAS])]
    models = [m.strip() for m in (args.models.split(",") if args.models else DEFAULT_SOURCES)]
    domains = DOMAINS if args.domains == "all" else [d.strip() for d in args.domains.split(",")]

    client = PangramClient(os.environ["PANGRAM_API_KEY"]) if not args.dry_run else None

    t0 = time.time()
    n_files = 0
    for alpha in alphas:
        atag = f"alpha_{alpha:.2f}"
        for src in models:
            for dom in domains:
                gen_fp = GEN_ROOT / src / atag / f"{dom}.json"
                if not gen_fp.exists():
                    print(f"[miss] {gen_fp}")
                    continue
                out_fp = OUT_ROOT / src / atag / f"{dom}.json"
                if args.skip_existing and out_fp.exists():
                    print(f"[skip] {out_fp}")
                    continue
                print(f"[score] {atag} / {src} / {dom}", flush=True)
                t_dom = time.time()
                res = evaluate_domain(client, gen_fp, args.dry_run)
                out_fp.parent.mkdir(parents=True, exist_ok=True)
                with open(out_fp, "w") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
                print(f"  -> processed={res['processed']} err={res['errors']} "
                      f"skip={res['skipped']} ({time.time()-t_dom:.0f}s)", flush=True)
                n_files += 1
    print(f"\n[done] {n_files} files in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main() or 0)
