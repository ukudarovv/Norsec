"""Параметры трекинга из ``configs/analytics_defaults.yaml``."""

from __future__ import annotations

from typing import Any


def tracking_params(analytics_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = analytics_cfg or {}
    tr = dict(cfg.get("tracking") or {})
    return {
        "max_positions": int(tr.get("max_positions", 120)),
        "track_ttl_sec": float(tr.get("track_ttl_sec", 30.0)),
        "cleanup_interval_sec": float(tr.get("cleanup_interval_sec", 5.0)),
    }
