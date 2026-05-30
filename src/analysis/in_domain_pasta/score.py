"""Unified AI-text-detector scorer for the in-domain PASTA ablation experiment.

This module merges the seven per-detector scorers that used to live beside it
(`score_pangram.py`, `score_gptzero.py`, `score_fdg.py`, `score_binoculars.py`,
`score_imbd.py`, `score_originality_at_new_Lstar.py`) into ONE
detector-parameterized scorer. A small `Detector` registry holds one backend
per detector; each backend lazily builds its client/model (API detectors read
their key from the environment or `REPO_ROOT/.env`; GPU models load once) and
exposes `score(text) -> dict` normalized to the common detection schema:

    {"success", "fraction_ai", "fraction_human", "prediction_short", ...}

A single cell-walk drives every detector: for the chosen `(alias, domain(s),
layer(s))` it walks

    data/analysis_results/in_domain_pasta/generations/<alias>/<domain>/L<L>.json

scores each generation's `processed_text` (falling back to `raw_text`), and
writes

    data/analysis_results/in_domain_pasta/detection_results/<detector>/<alias>/<domain>/L<L>.json

matching the per-cell record schema the legacy per-detector scorers emitted.
The walk is resume-safe via `--skip-existing` (a cell whose output already has
one record per generation is left untouched).

Fast-DetectGPT is special: the canonical `evaluate_fast_detect_gpt.run()` owns
its own cell-walk and output writing, so for `--detector fast_detect_gpt` this
module delegates to `run(bucket, source, layers, skip_existing=...)` per
`(alias, domain)` exactly as the old `score_fdg.py` did, instead of going
through the per-text `Detector.score` path.

All `evaluate_*` imports are lazy (inside each backend's builder), so this
module imports cleanly even when `scripts/detection/` is absent — those
modules are only required at run time.

Usage:
    # Pangram (API key: PANGRAM_API_KEY)
    python analysis/in_domain_pasta/score.py --detector pangram \\
        --model llama-3.1-8b-instruct

    # GPTZero (API key: GPTZERO_API_KEY)
    python analysis/in_domain_pasta/score.py --detector gptzero \\
        --model qwen2.5-7b-instruct --workers 16

    # Originality (API key: ORIGINALITY_API_KEY)
    python analysis/in_domain_pasta/score.py --detector originality \\
        --model llama-3.1-8b-instruct --domains news_articles

    # Binoculars (GPU; pin a card first)
    CUDA_VISIBLE_DEVICES=0 python analysis/in_domain_pasta/score.py \\
        --detector binoculars --model llama-3.1-8b-instruct \\
        --observer tiiuae/falcon-7b --performer tiiuae/falcon-7b-instruct

    # ImBD (GPU)
    CUDA_VISIBLE_DEVICES=1 python analysis/in_domain_pasta/score.py \\
        --detector imbd --model llama-3.1-8b-instruct --task generate

    # Fast-DetectGPT (GPU; delegates to evaluate_fast_detect_gpt.run)
    CUDA_VISIBLE_DEVICES=0 python analysis/in_domain_pasta/score.py \\
        --detector fast_detect_gpt --model llama-3.1-8b-instruct

    # Cost-free preview of the cells a run would touch
    python analysis/in_domain_pasta/score.py --detector pangram \\
        --model llama-3.1-8b-instruct --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

DATA_ROOT = REPO_ROOT / "data" / "analysis_results" / "in_domain_pasta"
GENERATIONS_ROOT = DATA_ROOT / "generations"
GENERATIONS_TRAIN_SWEEP_ROOT = DATA_ROOT / "generations_train_sweep"
DETECTION_RESULTS_ROOT = DATA_ROOT / "detection_results"
DETECTION_RESULTS_TRAIN_SWEEP_ROOT = DATA_ROOT / "detection_results_train_sweep"

DETECTOR_NAMES = (
    "pangram",
    "gptzero",
    "originality",
    "binoculars",
    "fast_detect_gpt",
    "imbd",
)


def _evaluate_module_path() -> None:
    """Make `scripts/detection/evaluate_*` importable. Called lazily by backends
    so the module imports fine when that directory is absent."""
    p = str(REPO_ROOT / "scripts" / "detection")
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_api_key(env_var: str) -> str | None:
    """Read an API key from the environment, falling back to `REPO_ROOT/.env`."""
    load_dotenv(REPO_ROOT / ".env")
    return os.environ.get(env_var)


def _text_of(record: dict) -> str:
    return record.get("processed_text") or record.get("raw_text") or ""


def _base_record(record: dict, text: str) -> dict:
    return {
        "id": record.get("id"),
        "sample_id": record.get("sample_id"),
        "original_prefix_id": record.get("original_prefix_id"),
        "text_length": len(text),
        "is_degenerate": record.get("is_degenerate"),
    }


class Detector:
    """A single AI-text detector with lazy client/model construction.

    `name` keys into the registry of backends below. `kwargs` carries
    detector-specific knobs (observer/performer/mode/max_token_observed for
    Binoculars; task/device for ImBD; api_key + rate limiting for the HTTP
    detectors; workers for GPTZero concurrency).

    `score(text)` returns a dict normalized to the common schema
    `{success, fraction_ai, fraction_human, prediction_short, ...}`. The client
    or GPU model is built once on first use and cached on the instance.
    """

    THRESHOLD = 0.5

    def __init__(self, name: str, **kwargs):
        if name not in DETECTOR_NAMES:
            raise ValueError(f"unknown detector {name!r}; known: {DETECTOR_NAMES}")
        self.name = name
        self.kwargs = kwargs
        self._client = None
        self._builder = getattr(self, f"_build_{name}")
        self._scorer = getattr(self, f"_score_{name}")
        self.config: dict = {}

    @property
    def client(self):
        if self._client is None:
            self._client = self._builder()
        return self._client

    def score(self, text: str) -> dict:
        return self._scorer(self.client, text)

    def _build_pangram(self):
        from pasta.vendor.pangram_client import PangramClient

        api_key = self.kwargs.get("api_key") or _load_api_key("PANGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("PANGRAM_API_KEY not set (check env or .env)")
        client = PangramClient(api_key)
        self.config = {"threshold": self.THRESHOLD,
                       "rate_limit_delay": client.rate_limit_delay}
        return client

    def _score_pangram(self, client, text: str) -> dict:
        det = client.check_text(text)
        time.sleep(getattr(client, "rate_limit_delay", 1.0))
        return det

    def _build_originality(self):
        _evaluate_module_path()
        from evaluate_originality import OriginalityClient

        api_key = self.kwargs.get("api_key") or _load_api_key("ORIGINALITY_API_KEY")
        if not api_key:
            raise RuntimeError("ORIGINALITY_API_KEY not set (check env or .env)")
        client = OriginalityClient(api_key)
        self.config = {"threshold": self.THRESHOLD,
                       "rate_limit_delay": getattr(client, "rate_limit_delay", 0.0)}
        return client

    def _score_originality(self, client, text: str) -> dict:
        det = client.check_text(text)
        time.sleep(getattr(client, "rate_limit_delay", 0.0))
        return det

    def _build_gptzero(self):
        _evaluate_module_path()
        from evaluate_gptzero import GPTZeroClient

        api_key = self.kwargs.get("api_key") or _load_api_key("GPTZERO_API_KEY")
        if not api_key:
            raise RuntimeError("GPTZERO_API_KEY not set (check env or .env)")
        client = GPTZeroClient(api_key)
        client.rate_limit_delay = 0.0
        self.config = {"threshold": self.THRESHOLD,
                       "max_workers": self.kwargs.get("workers", 16),
                       "retry_on_429": True, "max_retries": 5}
        return client

    @staticmethod
    def _normalize_gptzero(det: dict) -> dict:
        """Map raw GPTZero output onto the canonical {fraction_ai,
        fraction_human, prediction_short} shape and trim the bulky
        `raw_response.documents[*].sentences` payload down to a per-document
        summary (captured verbatim from the canonical evaluator)."""
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

    def _score_gptzero(self, client, text: str) -> dict:
        max_retries = self.config.get("max_retries", 5)
        delay = 1.0
        for _ in range(max_retries):
            det = client.check_text(text)
            if det.get("success"):
                return self._normalize_gptzero(det)
            err = (det.get("error") or "").lower()
            if "rate limit" in err or " 429" in err or "retry_after" in det:
                wait = float(det.get("retry_after", delay)) + random.uniform(0, 0.5)
                time.sleep(wait)
                delay = min(delay * 2, 30)
                continue
            return det
        return {"success": False,
                "error": f"max retries exceeded after {max_retries} attempts"}

    def _build_binoculars(self):
        _evaluate_module_path()
        from evaluate_binoculars import Binoculars

        detector = Binoculars(
            observer_id=self.kwargs.get("observer", "tiiuae/falcon-7b"),
            performer_id=self.kwargs.get("performer", "tiiuae/falcon-7b-instruct"),
            mode=self.kwargs.get("mode", "low-fpr"),
            max_token_observed=self.kwargs.get("max_token_observed", 512),
            use_bfloat16=True,
        )
        self.config = {
            "observer_id": self.kwargs.get("observer", "tiiuae/falcon-7b"),
            "performer_id": self.kwargs.get("performer", "tiiuae/falcon-7b-instruct"),
            "mode": self.kwargs.get("mode", "low-fpr"),
            "max_token_observed": self.kwargs.get("max_token_observed", 512),
            "threshold": float(detector.threshold),
        }
        return detector

    def _score_binoculars(self, detector, text: str) -> dict:
        """Binoculars emits a raw score where LOWER = more machine-like, with a
        decision `threshold`. Map it to a probability via the logistic
        `fraction_ai = 1 / (1 + exp(score - threshold))`, so `fraction_ai`
        crosses 0.5 exactly at the threshold and rises as the score falls below
        it. `prediction_short` is taken from the detector's own
        `predict_label`."""
        try:
            score = float(detector.compute_score(text))
        except Exception as e:
            return {"success": False, "error": str(e)}
        threshold = float(detector.threshold)
        label = detector.predict_label(score)
        fraction_ai = 1.0 / (1.0 + math.exp(score - threshold))
        return {
            "success": True,
            "score": score,
            "threshold": threshold,
            "predicted_label": label,
            "is_ai": label == "AI",
            "fraction_ai": fraction_ai,
            "fraction_human": 1.0 - fraction_ai,
            "prediction_short": "AI" if label == "AI" else "Human",
        }

    def _build_imbd(self):
        _evaluate_module_path()
        from evaluate_imbd import ImBDDetector, IMBD_CKPT, IMBD_REF

        task = self.kwargs.get("task", "generate")
        detector = ImBDDetector(
            ckpt_dir=IMBD_CKPT, ref_path=IMBD_REF,
            task=task, device=self.kwargs.get("device", "cuda"),
        )
        self.config = {"task": task, "threshold": self.THRESHOLD}
        return detector

    def _score_imbd(self, detector, text: str) -> dict:
        try:
            s = detector.score(text)
        except Exception as e:
            return {"success": False, "error": str(e)}
        prob_machine = float(s["prob_machine"])
        return {
            "success": True,
            "crit": s["crit"],
            "prob_machine": prob_machine,
            "task": s.get("task", self.kwargs.get("task", "generate")),
            "is_ai": prob_machine >= 0.5,
            "fraction_ai": prob_machine,
            "fraction_human": 1.0 - prob_machine,
            "prediction_short": "AI" if prob_machine >= 0.5 else "Human",
        }

    def _build_fast_detect_gpt(self):
        raise RuntimeError(
            "fast_detect_gpt is scored via evaluate_fast_detect_gpt.run(); "
            "it does not expose a per-text Detector.score path")

    def _score_fast_detect_gpt(self, client, text: str) -> dict:
        raise RuntimeError("fast_detect_gpt is scored via run(); see _run_fdg")


def _cell_record(detector: Detector, record: dict) -> dict:
    """Score one generation, mirroring the legacy per-detector record schema:
    pass-through identity fields, skip degenerate/empty, else `status`/scores."""
    text = _text_of(record)
    rec = _base_record(record, text)
    if record.get("is_degenerate"):
        rec["status"] = "skipped_degenerate"
        return rec
    if not text.strip():
        rec["status"] = "skipped_empty"
        return rec
    det = detector.score(text)
    if det.get("success"):
        rec["status"] = "ok"
        rec["detection"] = det
        if det.get("fraction_ai") is not None:
            rec["is_ai"] = float(det["fraction_ai"]) >= Detector.THRESHOLD
    else:
        rec["status"] = "error"
        rec["error"] = det.get("error")
    return rec


def _cell_is_complete(existing: dict, n_samples: int) -> bool:
    """A cell counts as fully scored iff its output holds one non-error,
    non-missing record per generation (ok or a legitimate skip)."""
    results = existing.get("results") or []
    if len(results) != n_samples:
        return False
    for rec in results:
        if rec is None:
            return False
        if rec.get("status") not in ("ok", "skipped_degenerate", "skipped_empty"):
            return False
    return True


def _score_cell(detector: Detector, gen_path: Path, out_path: Path,
                alias: str, domain: str, layer: int,
                skip_existing: bool, workers: int) -> dict:
    gen_data = json.load(open(gen_path))
    samples = gen_data.get("generations", [])

    if skip_existing and out_path.exists():
        try:
            existing = json.load(open(out_path))
            if _cell_is_complete(existing, len(samples)):
                return {"status": "skipped", "layer": layer, "n": len(samples)}
        except Exception:
            pass

    out = {
        "detector": detector.name,
        "bucket": "in_domain_pasta",
        "source": f"{alias}/{domain}",
        "domain": f"L{layer}",
        "model_type": gen_data.get("model_type"),
        "model_id": gen_data.get("model_id"),
        "config": detector.config,
        "n_samples": len(samples),
        "results": [None] * len(samples),
    }

    t0 = time.time()
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_cell_record, detector, r): i
                       for i, r in enumerate(samples)}
            for fut in futures:
                out["results"][futures[fut]] = fut.result()
    else:
        for i, r in enumerate(samples):
            out["results"][i] = _cell_record(detector, r)
    out["elapsed_sec"] = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return {"status": "ok", "layer": layer, "elapsed_sec": out["elapsed_sec"],
            "n": len(samples)}


def _run_fdg(alias: str, domains: list[str], layers: list[int],
             skip_existing: bool, dry_run: bool) -> int:
    """Delegate to the canonical Fast-DetectGPT evaluator, which owns its own
    cell-walk and output writing (one process loads the scoring model once)."""
    layer_names = [f"L{L}" for L in layers]
    if dry_run:
        for d in domains:
            print(f"[score:fast_detect_gpt] DRY RUN {alias}/{d}: "
                  f"would run layers {layer_names}", flush=True)
        return 0
    _evaluate_module_path()
    from evaluate_fast_detect_gpt import run

    for d in domains:
        source = f"{alias}/{d}"
        print(f"[score:fast_detect_gpt] {source}: scoring {len(layer_names)} layers",
              flush=True)
        run("in_domain_pasta", source, layer_names, skip_existing=skip_existing)
        print(f"[score:fast_detect_gpt] {source}: DONE", flush=True)
    return 0


def _resolve_spec(cfg: dict, model: str) -> dict:
    spec = next((m for m in cfg["models"] if m["alias"] == model), None)
    if spec is None:
        sys.exit(f"unknown model alias {model!r}. "
                 f"Known: {[m['alias'] for m in cfg['models']]}")
    return spec


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", required=True, choices=list(DETECTOR_NAMES))
    p.add_argument("--config", default=str(HERE / "configs" / "models.yaml"))
    p.add_argument("--model", required=True,
                   help="Model alias from configs/models.yaml.")
    p.add_argument("--domains", nargs="*", default=None,
                   help="Domains to score. Default: all in config.")
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="Layer indices to score. Default: all (0..n_layers-1).")
    p.add_argument("--split", choices=["test", "train"], default="test",
                   help="'test' scores generations/…; 'train' scores "
                        "generations_train_sweep/… (used to pick L*_D).")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip cells already fully scored (default on).")
    p.add_argument("--no-skip-existing", dest="skip_existing",
                   action="store_false",
                   help="Re-score cells even if their output already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't build clients or call detectors; just list cells.")
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent in-flight requests per cell "
                        "(GPTZero default 16; GPU/local detectors stay at 1).")
    p.add_argument("--observer", default="tiiuae/falcon-7b",
                   help="Binoculars observer model id.")
    p.add_argument("--performer", default="tiiuae/falcon-7b-instruct",
                   help="Binoculars performer model id.")
    p.add_argument("--mode", choices=["low-fpr", "accuracy"], default="low-fpr",
                   help="Binoculars decision mode.")
    p.add_argument("--max-token-observed", type=int, default=512,
                   help="Binoculars max observed tokens.")
    p.add_argument("--task", default="generate",
                   choices=["polish", "generate", "rewrite", "expand", "all"],
                   help="ImBD task conditioning.")
    p.add_argument("--device", default="cuda", help="ImBD device.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    spec = _resolve_spec(cfg, args.model)
    alias = spec["alias"]
    n_layers = spec["n_layers"]
    domains = args.domains if args.domains else cfg["domains"]
    layers = args.layers if args.layers is not None else list(range(n_layers))

    gen_root = GENERATIONS_ROOT if args.split == "test" else GENERATIONS_TRAIN_SWEEP_ROOT
    det_root = DETECTION_RESULTS_ROOT if args.split == "test" else DETECTION_RESULTS_TRAIN_SWEEP_ROOT

    print("=" * 60)
    print(f"score ({'DRY RUN' if args.dry_run else 'live'})")
    print(f"  detector:  {args.detector}")
    print(f"  model:     {alias}  (n_layers={n_layers})")
    print(f"  split:     {args.split}")
    print(f"  domains:   {domains}")
    print(f"  layers:    {layers}")
    print("=" * 60)

    if args.detector == "fast_detect_gpt":
        if args.split == "train":
            sys.exit("fast_detect_gpt supports --split test only "
                     "(the train sweep uses pangram for L*_D selection).")
        return _run_fdg(alias, domains, layers, args.skip_existing, args.dry_run)

    workers = args.workers
    if args.detector == "gptzero" and workers <= 1:
        workers = 16

    detector = None
    if not args.dry_run:
        detector = Detector(
            args.detector,
            workers=workers,
            observer=args.observer,
            performer=args.performer,
            mode=args.mode,
            max_token_observed=args.max_token_observed,
            task=args.task,
            device=args.device,
        )
        _ = detector.client

    results_root = det_root / args.detector
    total = len(domains) * len(layers)
    done = 0
    for d in domains:
        gen_dir = gen_root / alias / d
        if not gen_dir.exists():
            print(f"[score:{args.detector}] no generations dir: {gen_dir} — skip",
                  flush=True)
            continue
        for L in layers:
            gen_path = gen_dir / f"L{L}.json"
            out_path = results_root / alias / d / f"L{L}.json"
            if not gen_path.exists():
                print(f"[score:{args.detector}] {alias}/{d}/L{L}: "
                      f"MISSING gen file — skip", flush=True)
                continue
            if args.dry_run:
                print(f"[score:{args.detector}] DRY RUN would score "
                      f"{gen_path} -> {out_path}", flush=True)
                done += 1
                continue
            res = _score_cell(detector, gen_path, out_path, alias, d, L,
                              args.skip_existing, workers)
            done += 1
            if res["status"] == "ok":
                print(f"[score:{args.detector}] {alias}/{d}/L{L}: ok "
                      f"n={res['n']} ({res['elapsed_sec']:.1f}s)  "
                      f"[{done}/{total}]", flush=True)
            else:
                print(f"[score:{args.detector}] {alias}/{d}/L{L}: "
                      f"{res['status']} n={res.get('n', 0)}  [{done}/{total}]",
                      flush=True)
    print(f"[score:{args.detector}] DONE — {done} cells processed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
