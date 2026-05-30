"""Per-pair YAML config loader.

The single source of truth for a PASTA run is one config file under
`configs/`. The loader returns the parsed dict as-is; consumers reach into the
expected keys (`pair`, `models`, `devices`, `domains`, `generation`, `output`,
`phase_A`). See `configs/llama3.1-8b.yaml` for the canonical schema.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from pasta.paths import project_root

DEFAULT_CONFIG_REL = "configs/llama3.1-8b.yaml"


def default_config_path() -> Path:
    return project_root() / DEFAULT_CONFIG_REL


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else default_config_path()
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
