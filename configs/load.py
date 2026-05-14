"""Загрузка YAML из ``configs/``."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _config_base() -> Path:
    raw = os.environ.get("PHASE1_CONFIG_DIR")
    if raw:
        return Path(raw).resolve()
    return Path(__file__).resolve().parent


def _read_from(base: Path, name: str) -> dict[str, Any]:
    path = base / name
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def get_phase1_config() -> dict[str, Any]:
    base = _config_base()
    cam = _read_from(base, "camera_defaults.yaml")
    mod = _read_from(base, "model_defaults.yaml")
    fus = _read_from(base, "fusion_defaults.yaml")
    return {
        "camera": dict(cam.get("camera") or {}),
        "detector": dict(mod.get("detector") or {}),
        "tracker": dict(mod.get("tracker") or {}),
        "fusion": dict(fus.get("fusion") or {}),
    }


@lru_cache(maxsize=1)
def get_analytics_config() -> dict[str, Any]:
    base = _config_base()
    return _read_from(base, "analytics_defaults.yaml")


def clear_phase1_config_cache() -> None:
    for fn in (get_phase1_config, get_analytics_config):
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
