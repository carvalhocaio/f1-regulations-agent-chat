"""Shared environment-variable parsing helpers."""

from __future__ import annotations

import os
from pathlib import Path


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def get_package_dir(pkg: object) -> Path:
    """Resolve the directory of an installed or local package."""
    pkg_file = getattr(pkg, "__file__", None)
    if pkg_file is not None:
        return Path(pkg_file).parent
    return Path(getattr(pkg, "__path__")[0])
