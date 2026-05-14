"""Статусы потока камеры (runtime)."""

from __future__ import annotations

CAMERA_STATUSES: list[str] = [
    "offline",
    "connecting",
    "online",
    "analyzing",
    "error",
]


def normalize_status(value: str) -> str:
    v = (value or "").strip().lower()
    return v if v in CAMERA_STATUSES else "offline"
