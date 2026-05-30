"""Code-callable discovery + loading for everything under `data/`.

This module is the single API a script should use to ask "what's actually in
the data dir?" — instead of grepping prose READMEs or recomputing path
conventions. The README files document the JSON contract for human readers;
this module does the same job for code.

Public surface:
    prompts_for(domain)              -> list of prompt entries (pre-built)
    prefix_entries(domain)           -> list of raw prefix entries (with human_continuation)
    human_continuation_for(domain, prefix_id) -> str | None
    load_human(cfg, domain)          -> list of independent human-corpus entries, or None
    aligned_bundles(cfg)             -> {domain: Path} for every present file
    base_bundles(cfg)                -> {domain: Path} for every present file
    pasta_ablated_runs(cfg)          -> list of dicts describing each run on disk
    list_subsets()                   -> list of dicts (subset_name + manifest)
    register_subset(manifest)        -> Path to the new subset dir
    summarize(cfg)                   -> nested dict of what's present
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from pasta.io import read_json, write_json
from pasta.paths import (
    aligned_baseline_dir,
    base_baseline_dir,
    pasta_ablated_dir,
    project_root,
    resolve,
)
from pasta.prompts import load_prefixes


# ---------------------------------------------------------------------------
# Prompts (canonical: data/prompts/<domain>.json — pre-built schema)
# ---------------------------------------------------------------------------
def prompts_for(domain: str) -> list[dict]:
    """Return the list of prompt entries for `domain`.

    Thin wrapper around `pasta.prompts.load_prefixes` to keep this module the
    single entry point for data lookups. Raises SystemExit if the prompt file
    is missing for `domain`.

    NOTE: pre-built entries (`data/prompts/<domain>.json`) carry only the
    prompt-side fields (`aligned_messages`, `base_prompt`, `aligned_user_text`).
    They do NOT include `human_continuation`. For prompt-matched human
    continuations, use `prefix_entries(domain)` and `human_continuation_for`.
    """
    return load_prefixes(domain)


# ---------------------------------------------------------------------------
# Raw prefixes (data/prefixes/<domain>.json — includes human_continuation)
# ---------------------------------------------------------------------------
def prefix_entries(domain: str) -> list[dict]:
    """Return the raw prefix entries for `domain`.

    Reads `data/prefixes/<domain>.json` directly (bypassing the prompts/
    auto-detection in `pasta.prompts`) and returns the list under the
    `prefixes` key. Each entry is expected to have `id`, `prefix`,
    `instruction`, `human_continuation`, plus optional `source_meta`.

    Raises FileNotFoundError if the raw file is not present — the user must
    populate `data/prefixes/<domain>.json` from the upstream source dataset.
    """
    path = resolve("data/prefixes") / f"{domain}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw prefixes for {domain!r} at {path}. "
            f"Pre-built data/prompts/<domain>.json does NOT carry "
            f"human_continuation; populate data/prefixes/ from the upstream "
            f"source if you need prompt-matched human text."
        )
    data = read_json(path)
    return list(data.get("prefixes", []))


def human_continuation_for(domain: str, prefix_id: int) -> str | None:
    """Return the human continuation paired with `(domain, prefix_id)`.

    Looks up the raw prefix entry and returns its `human_continuation` field.
    Returns None if the raw prefixes file does not exist or the id is not
    found. Use this when you need to compare a model's generation against
    the actual human text that followed the same prefix in the source doc.
    """
    try:
        entries = prefix_entries(domain)
    except FileNotFoundError:
        return None
    for e in entries:
        if e.get("id") == prefix_id:
            return e.get("human_continuation")
    return None


# ---------------------------------------------------------------------------
# Human reference text (data/generations/human/<domain>.json)
# ---------------------------------------------------------------------------
def human_dir() -> Path:
    return resolve("data/generations/human")


def load_human(cfg: dict, domain: str) -> dict | None:
    """Return the human-corpus generation bundle for `domain`, or None if missing.

    The returned dict is shaped like a model generation bundle
    (`{model_type: "human", model_id: <dataset>, generations: [{id, raw_text,
    processed_text, ...}, ...]}`). `prefix_used` is `null` on each entry —
    these samples are an independent human corpus, NOT paired to any prompt.
    For prompt-matched human continuations, use `human_continuation_for`.
    """
    path = human_dir() / f"{domain}.json"
    if not path.exists():
        return None
    return read_json(path)


# ---------------------------------------------------------------------------
# Generation buckets
# ---------------------------------------------------------------------------
def _present_per_domain(directory: Path, domains: list[str]) -> dict[str, Path]:
    """Return {domain: path} only for domains whose JSON exists under `directory`."""
    if not directory.exists():
        return {}
    return {d: directory / f"{d}.json" for d in domains if (directory / f"{d}.json").exists()}


def aligned_bundles(cfg: dict) -> dict[str, Path]:
    """Map domain -> existing path under data/generations/aligned/<pair>/<t>/.
    Domains with no file are omitted (returns {} when nothing is present)."""
    return _present_per_domain(aligned_baseline_dir(cfg), cfg["domains"])


def base_bundles(cfg: dict) -> dict[str, Path]:
    """Map domain -> existing path under data/generations/base/<pair>/<t>/."""
    return _present_per_domain(base_baseline_dir(cfg), cfg["domains"])


@dataclass
class PastaAblatedRun:
    """One concrete on-disk pasta-ablated run."""
    target: str
    tag: str
    temperature_tag: str
    path: Path
    domains_present: list[str]


def pasta_ablated_runs(cfg: dict) -> list[PastaAblatedRun]:
    """Enumerate every pasta-ablated run on disk for cfg.pair.id.

    Walks `data/generations/pasta_ablated/<pair>/<target>/<tag>/<t>/` and
    returns one entry per (target, tag, t) directory found, recording which
    domains have JSON files present.
    """
    pair_root = resolve("data/generations") / "pasta_ablated" / cfg["pair"]["id"]
    if not pair_root.exists():
        return []
    runs: list[PastaAblatedRun] = []
    for target_dir in sorted(p for p in pair_root.iterdir() if p.is_dir()):
        for tag_dir in sorted(p for p in target_dir.iterdir() if p.is_dir()):
            for t_dir in sorted(p for p in tag_dir.iterdir() if p.is_dir()):
                domains_here = sorted(
                    p.stem for p in t_dir.glob("*.json") if p.is_file()
                )
                runs.append(PastaAblatedRun(
                    target=target_dir.name,
                    tag=tag_dir.name,
                    temperature_tag=t_dir.name,
                    path=t_dir,
                    domains_present=domains_here,
                ))
    return runs


# ---------------------------------------------------------------------------
# Subsets (data/subsets/<name>/, indexed via data/subsets/index.json)
# ---------------------------------------------------------------------------
def subsets_root() -> Path:
    return resolve("data/subsets")


def _subsets_index_path() -> Path:
    return subsets_root() / "index.json"


def list_subsets() -> list[dict]:
    """Read every subset manifest under data/subsets/<name>/manifest.json.

    Authoritative source is the on-disk dirs (each must have a manifest).
    `data/subsets/index.json` is a cached summary updated by `register_subset`;
    if it drifts from the directories, the directories win. This function
    reconciles by globbing.
    """
    root = subsets_root()
    if not root.exists():
        return []
    out = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest_path = sub / "manifest.json"
        if not manifest_path.exists():
            continue
        out.append(read_json(manifest_path))
    return out


def register_subset(manifest: dict) -> Path:
    """Create a new subset dir from a manifest dict and update the index.

    Validates required fields, writes `data/subsets/<name>/manifest.json`, and
    appends a stub to `data/subsets/index.json`. Does NOT copy generation
    bundles; the caller is responsible for materializing per-domain JSONs
    inside the new dir.
    """
    required = {"subset_name", "selection", "source_buckets", "domains"}
    missing = required - set(manifest.keys())
    if missing:
        raise ValueError(f"manifest missing required fields: {sorted(missing)}")
    name = manifest["subset_name"]
    sub_dir = subsets_root() / name
    if sub_dir.exists():
        raise FileExistsError(f"subset {name!r} already exists at {sub_dir}; "
                              "create a new <subset-name> rather than overwriting.")
    sub_dir.mkdir(parents=True)
    write_json(sub_dir / "manifest.json", manifest)

    idx_path = _subsets_index_path()
    idx = read_json(idx_path) if idx_path.exists() else {"subsets": []}
    idx["subsets"].append({
        "subset_name": name,
        "detector": manifest.get("detector"),
        "n_per_domain": manifest.get("selection", {}).get("n_per_domain"),
    })
    write_json(idx_path, idx)
    return sub_dir


# ---------------------------------------------------------------------------
# Top-level summary (used by scripts/inspect_data.py)
# ---------------------------------------------------------------------------
def summarize(cfg: dict) -> dict:
    """Return a single dict with what is actually present under data/."""
    domains = cfg["domains"]

    # Prompts (pre-built schema, data/prompts/)
    prompts_present: dict[str, int] = {}
    for d in domains:
        try:
            prompts_present[d] = len(prompts_for(d))
        except SystemExit:
            prompts_present[d] = 0

    # Raw prefixes (data/prefixes/) — required for human_continuation lookup
    prefixes_present: dict[str, int | None] = {}
    for d in domains:
        try:
            prefixes_present[d] = len(prefix_entries(d))
        except FileNotFoundError:
            prefixes_present[d] = None

    # Human
    human_present: dict[str, int | None] = {}
    for d in domains:
        bundle = load_human(cfg, d)
        human_present[d] = None if bundle is None else len(bundle.get("generations", []))

    # Generations
    aligned = {d: str(p) for d, p in aligned_bundles(cfg).items()}
    base = {d: str(p) for d, p in base_bundles(cfg).items()}
    runs = [
        {**asdict(r), "path": str(r.path)}
        for r in pasta_ablated_runs(cfg)
    ]

    # Subsets
    sub = list_subsets()

    return {
        "project_root": str(project_root()),
        "pair_id": cfg["pair"]["id"],
        "temperature_tag": cfg.get("output", {}).get("temperature_tag", "t1"),
        "domains": domains,
        "prompts": prompts_present,
        "prefixes": prefixes_present,
        "human": human_present,
        "generations": {
            "aligned": aligned,
            "base": base,
            "pasta_ablated_runs": runs,
        },
        "subsets": sub,
    }
