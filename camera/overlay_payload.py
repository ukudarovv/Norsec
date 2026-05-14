"""Формат overlay-события для WebSocket и UI."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_overlay_payload(
    camera_id: str,
    *,
    people: list[dict[str, Any]],
    poses: list[dict[str, Any]] | None = None,
    signals: list[dict[str, Any]] | None = None,
    risk_score: float,
    risk_level: str,
    timestamp_iso: str | None = None,
    analytics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "camera_id": camera_id,
        "timestamp": timestamp_iso or utc_iso_z(),
        "people": list(people or []),
        "poses": list(poses or []),
        "signals": list(signals or []),
        "risk": {"score": float(risk_score), "level": str(risk_level)},
    }
    if analytics:
        out["analytics"] = dict(analytics)
    return out
