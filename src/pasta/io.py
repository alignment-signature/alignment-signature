"""JSON I/O helpers.

`read_json` / `write_json` are the single chokepoint for on-disk JSON so every
caller gets the same behaviour — UTF-8 encoding, 2-space indentation, non-ASCII
preserved (not escaped), and no leaked file handles.

`write_bundle` / `write_record_incremental` persist per-domain generation
bundles. Long-running generation jobs must persist after every sample rather
than only at the end, so that a crashed run leaves a resumable partial file;
`write_record_incremental` appends one record to the bundle on each call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    return path


def write_bundle(out_dir: Path, domain: str, bundle: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_json(out_dir / f"{domain}.json", bundle)


def write_record_incremental(out_dir: Path, domain: str, record: dict, bundle_meta: dict) -> Path:
    """Append a single generation to the domain bundle, writing after every sample."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{domain}.json"
    if out_path.exists():
        bundle = read_json(out_path)
    else:
        bundle = {**bundle_meta, "num_generations": 0, "num_degenerate": 0, "generations": []}
    bundle["generations"].append(record)
    bundle["num_generations"] = len(bundle["generations"])
    bundle["num_degenerate"] = sum(1 for g in bundle["generations"] if g.get("is_degenerate"))
    return write_json(out_path, bundle)
