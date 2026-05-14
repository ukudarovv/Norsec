"""Потокобезопасное состояние live-камеры: overlay, риск, метрики (Phase 1)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from camera.stream_status import normalize_status

logger = logging.getLogger(__name__)


class CameraLiveState:
    """latest_overlay / latest_risk / статус / ошибки / FPS (один объект на камеру)."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = str(camera_id)
        self._lock = threading.Lock()
        self._seq = 0
        self._overlay: dict[str, Any] | None = None
        self._jpeg: bytes | None = None
        self._camera_status = "offline"
        self._last_error: str | None = None
        self._last_frame_time: float | None = None
        self._dropped_frames = 0
        self._fps_ewma = 0.0
        self._last_proc_mono: float | None = None
        self._last_update = 0.0
        self._latest_risk: dict[str, float | str] = {"score": 0.0, "level": "green"}

    def set_camera_status(self, status: str) -> None:
        with self._lock:
            self._camera_status = normalize_status(status)
            self._last_update = time.time()

    def set_last_error(self, message: str | None) -> None:
        with self._lock:
            self._last_error = message
            self._last_update = time.time()

    def record_frame_received(self) -> None:
        with self._lock:
            self._last_frame_time = time.time()
            self._last_update = time.time()

    def record_dropped_frame(self) -> None:
        with self._lock:
            self._dropped_frames += 1
            self._last_update = time.time()

    def record_processed_frame(self, mono_now: float) -> None:
        with self._lock:
            if self._last_proc_mono is not None:
                dt = mono_now - self._last_proc_mono
                if dt > 1e-6:
                    inst = 1.0 / dt
                    self._fps_ewma = 0.25 * inst + 0.75 * self._fps_ewma if self._fps_ewma > 0 else inst
            self._last_proc_mono = mono_now
            self._last_update = time.time()

    def publish_overlay(self, payload: dict[str, Any]) -> None:
        with self._lock:
            risk = payload.get("risk") or {}
            try:
                sc = float(risk.get("score", 0.0))
            except (TypeError, ValueError):
                sc = 0.0
            self._latest_risk = {"score": sc, "level": str(risk.get("level", "green"))}
            self._overlay = dict(payload)
            self._seq += 1
            self._last_update = time.time()

    def publish_jpeg(self, data: bytes) -> None:
        with self._lock:
            self._jpeg = bytes(data)
            self._last_update = time.time()

    def snapshot(self) -> tuple[int, dict[str, Any] | None, bytes | None]:
        with self._lock:
            return self._seq, (dict(self._overlay) if self._overlay is not None else None), self._jpeg

    def snapshot_metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "camera_status": self._camera_status,
                "last_error": self._last_error,
                "last_frame_time": self._last_frame_time,
                "dropped_frames": self._dropped_frames,
                "fps_estimate": round(self._fps_ewma, 2),
                "last_update": self._last_update,
                "latest_risk": dict(self._latest_risk),
                "overlay_seq": self._seq,
            }

    def reset_counters(self) -> None:
        with self._lock:
            self._dropped_frames = 0
            self._fps_ewma = 0.0
            self._last_proc_mono = None
