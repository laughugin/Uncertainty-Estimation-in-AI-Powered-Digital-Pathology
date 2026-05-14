"""Load, merge, and persist YAML configuration for the pipeline."""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_REL = Path("configs/uncertainty_lab_default.yaml")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load YAML; merge ``configs/uncertainty_lab_default.yaml`` when it exists."""
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    default_path = repo_root / DEFAULT_CONFIG_REL
    defaults = load_yaml(default_path) if default_path.exists() else {}
    if path is None:
        base = copy.deepcopy(defaults)
    else:
        base = deep_merge(defaults, load_yaml(path))
    if overrides:
        base = deep_merge(base, overrides)
    base.setdefault("run", {})
    if base["run"].get("repo_root") in (None, "null"):
        base["run"]["repo_root"] = str(repo_root)
    return base


def save_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def stamp_run_dir(base_dir: Path, name: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = (name or "run").replace("/", "-").replace(" ", "_")[:80]
    return base_dir / f"{ts}_{slug}"
