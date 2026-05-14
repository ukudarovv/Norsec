"""Хаб потоков: по камере — ``CameraLiveState`` (overlay + MJPEG + метрики)."""

from __future__ import annotations

import threading

from camera.live_state import CameraLiveState

# Совместимость с этапом 10
CameraStreamSink = CameraLiveState


class StreamHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sinks: dict[str, CameraLiveState] = {}

    def sink(self, camera_id: str) -> CameraLiveState:
        with self._lock:
            if camera_id not in self._sinks:
                self._sinks[camera_id] = CameraLiveState(camera_id)
            return self._sinks[camera_id]

    def clear_sink(self, camera_id: str) -> None:
        with self._lock:
            self._sinks.pop(camera_id, None)
