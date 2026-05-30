"""Shared helper: enumerate the (model, L) test-bundle cells the detector-swap
cross-scoring needs.

For each model, we score test bundles at the layer chosen by each detector's
own Step 1 (Pangram-best L, FDG-best L, Binoculars-best L, ImBD-best L).
When two detectors agree on a layer, that cell is shared. The function returns
the **unique** (model, L) pairs needed plus a `bundle_for(alias, L)` resolver
that returns:

  - For test bundles produced by `run_test_ablation.py`
    (data/analysis_results/detector_swap/generations/<alias>/<detector>_L<L>.json
    in the new naming; data/analysis_results/detector_swap/generations/<alias>/L<L>.json
    in the legacy naming — bundle_for() probes both):
        these contain the full Path A test set (125 prompts) in one file.
        Returned as ('combined', path).

  - For canonical PASTA pasta_ablated best-layer bundles
    (data/generations/pasta_ablated/<model_dir>/v_cross_L<L>_ablated/<domain>.json):
        these contain the full 100-prompt corpus; the caller must filter to
        the Path A test set ids [0..24].  Returned as ('per_domain', [path,..]).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

PASTA_ABLATED_MODEL_DIR = {
    "llama-3.1-8b-instruct":  "meta-llama_meta-llama-3.1-8b-instruct",
    "gemma-2-9b-it":          "google_gemma-2-9b-it",
    "qwen2.5-7b-instruct":    "qwen_qwen2.5-7b-instruct",
}

PATH_A_TEST_IDS = set(range(25))

DOMAINS = ["college_essays", "creative_fiction", "news_articles",
           "opinion_pieces", "scientific_abstracts"]


def best_Ls(alias: str) -> dict[str, int]:
    """Return {'pangram':L, 'fdg':L, 'binoculars':L, 'imbd':L_or_None}.

    Pangram-best L comes from the saved analysis/detector_swap/configs/models.yaml.
    FDG and Binoculars come from data/analysis_results/detector_swap/<det>_train_sweep/<alias>/best_L.json.
    ImBD comes from data/analysis_results/detector_swap/imbd_train_sweep/<alias>/best_L.json
      (None if that sweep hasn't been run yet).
    """
    import yaml
    cfg = yaml.safe_load(open(HERE / "configs" / "models.yaml"))
    spec = next(m for m in cfg["models"] if m["alias"] == alias)
    out = {"pangram": int(spec["pangram_best_L"])}

    base = REPO_ROOT / "data" / "analysis_results" / "detector_swap"
    for det, key in [("fdg", "fdg_best_L"),
                     ("binoculars", "binoculars_best_L"),
                     ("imbd", "imbd_best_L")]:
        fp = base / f"{det}_train_sweep" / alias / "best_L.json"
        if fp.exists():
            out[det] = int(json.load(open(fp))[key])
        else:
            out[det] = None
    return out


def unique_cells(alias: str) -> list[int]:
    """Return the sorted unique layers we need test scores at for one alias."""
    bs = best_Ls(alias)
    return sorted({L for L in bs.values() if L is not None})


def find_combined_bundle(alias: str, layer: int) -> Path | None:
    """Search the detector_swap generations dir for a 125-record combined bundle
    at (alias, layer). Prefers the new <detector>_L<L>.json naming and falls
    back to the legacy L<L>.json. When several <detector>_L<L>.json files exist
    at the same L (one per swap detector that picked that L), their content is
    identical by construction (same canonical PASTA hook footprint + same seed
    at the same L), so we return any one of them — the first match by sorted
    filename. Returns None if no combined bundle is found at all.
    """
    gen_dir = REPO_ROOT / "data" / "analysis_results" / "detector_swap" / "generations" / alias
    if not gen_dir.exists():
        return None
    new_matches = sorted(gen_dir.glob(f"*_L{layer}.json"))
    if new_matches:
        return new_matches[0]
    legacy = gen_dir / f"L{layer}.json"
    if legacy.exists():
        return legacy
    return None


def bundle_for(alias: str, layer: int) -> tuple[str, list[Path]]:
    """Locate the test-bundle source for (alias, layer).

    Returns ('combined', [Path]) if a single 125-prompt detector_swap bundle
    exists (typical for layers picked by FDG/Binoculars/ImBD), or
    ('per_domain', [Path, ...]) pointing at the 5 canonical pasta_ablated
    bundles whose `original_prefix_id`s need to be filtered to [0..24]
    (typical for the Pangram-best layer — those bundles carry the FULL
    100-prompt corpus, so filtering to [0..24] yields a real 125-row Path A
    test slice).

    A 'per_domain' bundle is only accepted if every domain JSON contains the
    full 100-prompt corpus. The per-layer ablation TRAIN sweep bundles
    (5 prompts/domain = canonical PASTA train ids) are NOT a valid Path A
    test source even though they live in the same path tree — and we refuse
    them here, raising FileNotFoundError so the caller knows to regenerate.
    """
    combined = find_combined_bundle(alias, layer)
    if combined is not None:
        data = json.load(open(combined))
        n = data.get("num_generations") or len(data.get("generations", []))
        if n < 125:
            raise FileNotFoundError(
                f"Combined bundle {combined} only has n={n} (expected 125). "
                f"Likely a run_test_ablation.py write in progress; retry once finished."
            )
        return "combined", [combined]
    mdir = PASTA_ABLATED_MODEL_DIR[alias]
    per_domain = [REPO_ROOT / "data" / "generations" / "pasta_ablated" / mdir
                  / f"v_cross_L{layer}_ablated" / f"{d}.json" for d in DOMAINS]
    if all(p.exists() for p in per_domain):
        for p in per_domain:
            data = json.load(open(p))
            n = data.get("num_generations") or len(data.get("generations", []))
            if n < 100:
                raise FileNotFoundError(
                    f"No valid Path A test bundle for alias={alias} layer={layer}.  "
                    f"Found per_domain bundles but they hold only n={n} (the canonical "
                    f"PASTA TRAIN sample, not the full corpus).  Generate the Path A "
                    f"test bundle first via run_test_ablation.py."
                )
        return "per_domain", per_domain
    gen_dir = REPO_ROOT / "data" / "analysis_results" / "detector_swap" / "generations" / alias
    raise FileNotFoundError(
        f"No test bundle found for alias={alias} layer={layer}. Tried:\n"
        f"  combined (new naming):    {gen_dir}/*_L{layer}.json\n"
        f"  combined (legacy naming): {gen_dir}/L{layer}.json\n"
        f"  per_domain:               {per_domain[0]} ..."
    )


def load_test_records(alias: str, layer: int) -> list[dict]:
    """Return the 125-prompt Path A test slice for (alias, layer), regardless of
    whether the source is a combined detector_swap bundle or 5 per-domain
    pasta_ablated bundles.
    """
    kind, paths = bundle_for(alias, layer)
    records: list[dict] = []
    if kind == "combined":
        data = json.load(open(paths[0]))
        for g in data.get("generations", []):
            if g.get("original_prefix_id") in PATH_A_TEST_IDS:
                records.append(g)
        return records
    for fp in paths:
        data = json.load(open(fp))
        for g in data.get("generations", []):
            if g.get("original_prefix_id") in PATH_A_TEST_IDS:
                records.append(g)
    return records
