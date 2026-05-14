"""Resolve torch.device from pipeline config (``run.device``: auto | cpu | cuda)."""
from __future__ import annotations

from typing import Any

import torch


def resolve_device(config: dict[str, Any]) -> torch.device:
    choice = str(config.get("run", {}).get("device", "auto")).lower().strip()
    if choice in ("cpu",):
        return torch.device("cpu")
    if choice in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
