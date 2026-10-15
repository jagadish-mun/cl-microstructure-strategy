"""Load YAML configuration with optional dotted-key overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

Config = dict[str, Any]


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Read a YAML config and apply ``{"a.b.c": value}`` style overrides.

    Args:
        path: YAML file; defaults to ``configs/default.yaml``.
        overrides: dotted keys mapped to replacement values.
    """
    with open(path or DEFAULT_CONFIG) as fh:
        cfg: Config = yaml.safe_load(fh)
    for key, value in (overrides or {}).items():
        set_dotted(cfg, key, value)
    return cfg


def set_dotted(cfg: Config, key: str, value: Any) -> None:
    """Set ``cfg[a][b][c] = value`` for ``key == "a.b.c"``; missing keys are an error."""
    *parents, leaf = key.split(".")
    node = cfg
    for part in parents:
        node = node[part]
    if leaf not in node:
        raise KeyError(f"unknown config key: {key}")
    node[leaf] = value


def with_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Return a deep copy of ``cfg`` with dotted overrides applied."""
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        set_dotted(out, key, value)
    return out


def data_path(cfg: Config, *parts: str) -> Path:
    """Path under the configured data directory."""
    return Path(cfg["paths"]["data_dir"]).joinpath(*parts)


def reports_path(cfg: Config, *parts: str) -> Path:
    """Path under the configured reports directory (created if missing)."""
    root = Path(cfg["paths"]["reports_dir"])
    root.mkdir(parents=True, exist_ok=True)
    return root.joinpath(*parts)
