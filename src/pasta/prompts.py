"""Prompt loading and Strategy-A prompt construction.

Two on-disk schemas are supported, auto-detected by file shape:

  1. RAW PREFIXES — `data/prefixes/<domain>.json`:
       { "domain": ..., "prefixes": [ {id, prefix, instruction, ...}, ... ] }
     Each entry is passed through `pasta.vendor.strategy_templates` to
     construct the base + aligned prompts at runtime.

  2. PRE-BUILT STRATEGY-A — `data/prompts/<domain>.json`:
       { "domain": ..., "prompts": [
            {id, domain, aligned_messages, aligned_user_text, base_prompt}, ...
       ] }
     Used as-is. This is the schema produced by the host repo's
     `build_strategy_A_prompts.py`.

If both exist for a domain, `data/prompts/` wins; pin a single location
explicitly with `prompts_subdir:` in the config.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from pasta.io import read_json
from pasta.paths import resolve
from pasta.vendor.strategy_templates import build_strategy_A, derive_fields

PREFIX_SUBDIRS = ("data/prompts", "data/prefixes")


def _resolve_prompts_path(domain: str, override_subdir: str | None = None) -> tuple[Path, str]:
    """Find the per-domain prompt file. Returns (path, schema) where schema is
    "prebuilt" (entries already have aligned_messages + base_prompt) or "raw"
    (entries have prefix + instruction)."""
    candidates = (override_subdir,) if override_subdir else PREFIX_SUBDIRS
    for sub in candidates:
        path = resolve(sub) / f"{domain}.json"
        if path.exists():
            data = read_json(path)
            if "prompts" in data:
                return path, "prebuilt"
            if "prefixes" in data:
                return path, "raw"
            raise SystemExit(f"{path}: expected top-level 'prompts' or 'prefixes' key")
    locs = " or ".join(str(resolve(s) / f"{domain}.json") for s in candidates)
    raise SystemExit(f"Missing prompts file for domain '{domain}'. Looked in: {locs}")


def load_prefixes(domain: str, override_subdir: str | None = None) -> list[dict]:
    """Load entries for `domain`. Returned dicts include a `_schema` key
    ("raw" or "prebuilt") so downstream code knows how to build prompts."""
    path, schema = _resolve_prompts_path(domain, override_subdir)
    data = read_json(path)
    entries = data["prompts"] if schema == "prebuilt" else data["prefixes"]
    return [{**e, "_schema": schema} for e in entries]


def _stable_hash(s: str) -> int:
    """Deterministic cross-process hash. Python's builtin hash() is salted per process."""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "big")


def select_prefix_ids(domain: str, n: int, seed: int, override_subdir: str | None = None) -> list[int]:
    """Deterministically pick `n` prefix IDs for `domain`.

    Sorts by id, seeds, then random.sample. Same (domain, n, seed) → same ids
    across machines/processes. Caps n at len(available) and warns on cap."""
    prefixes = load_prefixes(domain, override_subdir)
    ids = sorted(p["id"] for p in prefixes)
    if n > len(ids):
        print(f"[warn] domain '{domain}': requested n={n} > available {len(ids)}; using all.")
        n = len(ids)
    rng = random.Random(seed + _stable_hash(domain))
    return sorted(rng.sample(ids, n))


def build_strategy_A_prompt(domain: str, prefix_entry: dict) -> dict:
    """Return {aligned_messages, base_prompt} for one entry.

    For prebuilt entries returns the stored prompts directly; for raw prefixes
    runs the vendor template builder."""
    if prefix_entry.get("_schema") == "prebuilt" or (
        "aligned_messages" in prefix_entry and "base_prompt" in prefix_entry
    ):
        return {
            "aligned_messages": prefix_entry["aligned_messages"],
            "base_prompt": prefix_entry["base_prompt"],
        }
    fields = derive_fields(domain, prefix_entry)
    return build_strategy_A(domain, fields)
