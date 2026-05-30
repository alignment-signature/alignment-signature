"""Project-root resolution.

PASTA scripts and library code resolve all on-disk paths relative to a single
project root — the directory that contains `pyproject.toml`, `configs/`, and
`outputs/`. We resolve it lazily so library code can be imported from anywhere
(notebooks, tests, ad-hoc scripts) without a chdir.

Resolution order:
    1. $PASTA_PROJECT_ROOT environment variable, if set.
    2. Walk up from the importing file until a directory containing
       `pyproject.toml` is found.
    3. Fall back to CWD.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    env = os.environ.get("PASTA_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "pasta").exists():
            return parent
    return Path.cwd().resolve()


def resolve(rel: str | os.PathLike) -> Path:
    """Resolve `rel` against project_root() if it's not already absolute."""
    p = Path(rel)
    return p if p.is_absolute() else project_root() / p


# ---------------------------------------------------------------------------
# Canonical generation / detection bucket layout
#
#   data/generations/aligned/<aligned-model>/<domain>.json
#   data/generations/base/<base-model>/<domain>.json
#   data/generations/pasta_ablated/<target-model>/<tag>/<domain>.json
#   data/detection_results/pangram/<bucket-relative-path>
#
# where <*-model> is the HF id lowercased with '/' replaced by '_' (see
# _model_dirname) and <tag> encodes the intervention (e.g. v_cross_L1_ablated).
# All callers (scripts, extraction, analysis) should go through these helpers
# rather than concatenating path strings inline.
# ---------------------------------------------------------------------------


def _output(cfg: dict) -> dict:
    return cfg["output"]


def _generations_root(cfg: dict) -> Path:
    return resolve(_output(cfg).get("generations_root", "data/generations"))


def _pangram_root(cfg: dict) -> Path:
    return resolve(_output(cfg).get("pangram_root", "data/detection_results/pangram"))


def _model_dirname(model_id: str) -> str:
    """Canonical on-disk dir for a model: HF id lowercased, '/' → '_'."""
    return model_id.lower().replace("/", "_")


def aligned_baseline_dir(cfg: dict) -> Path:
    """Where aligned-model no-intervention generations live.

    Layout: data/generations/aligned/<lowercased(cfg.models.instruct)>/<domain>.json
    e.g. data/generations/aligned/meta-llama_meta-llama-3.1-8b-instruct/...
    """
    return _generations_root(cfg) / "aligned" / _model_dirname(cfg["models"]["instruct"])


def base_baseline_dir(cfg: dict) -> Path:
    """Where base-model no-intervention generations live.

    Layout: data/generations/base/<lowercased(cfg.models.base)>/<domain>.json
    e.g. data/generations/base/meta-llama_meta-llama-3.1-8b/...
    """
    return _generations_root(cfg) / "base" / _model_dirname(cfg["models"]["base"])


def baseline_dir_for(cfg: dict, model_type: str) -> Path:
    """Dispatch to aligned/base baseline dir by side string."""
    if model_type == "instruct":
        return aligned_baseline_dir(cfg)
    if model_type == "base":
        return base_baseline_dir(cfg)
    raise ValueError(f"Unknown model_type {model_type!r}; expected 'base' or 'instruct'.")


def pasta_ablated_dir(cfg: dict, target_model: str, tag: str) -> Path:
    """Where pasta-ablated generations land.

    Layout: data/generations/pasta_ablated/<lowercased(cfg.models[target])>/<tag>/<domain>.json
    e.g. data/generations/pasta_ablated/meta-llama_meta-llama-3.1-8b-instruct/v_cross_L1_ablated/...

    `target_model` is "instruct" (the default — ablating the aligned model) or
    "base" (the unusual case — ablating the base model). The dir name encodes
    which model the hooks attached to; the tag encodes the intervention details
    (layer, ablate vs steer vs random vs scaled).
    """
    target_id = cfg["models"][target_model]
    return _generations_root(cfg) / "pasta_ablated" / _model_dirname(target_id) / tag


def pangram_mirror(cfg: dict, generations_dir: Path) -> Path:
    """Mirror a generations path under the pangram results root.

    Given an absolute path inside `<generations_root>/...`, return the
    corresponding path under `<pangram_root>/...`.
    """
    gen_root = _generations_root(cfg).resolve()
    abs_gen = generations_dir.resolve()
    try:
        rel = abs_gen.relative_to(gen_root)
    except ValueError as e:
        raise ValueError(
            f"{abs_gen} is not under generations_root={gen_root}; cannot mirror."
        ) from e
    return _pangram_root(cfg) / rel
