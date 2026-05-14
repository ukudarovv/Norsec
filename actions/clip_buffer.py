"""FIFO-буфер кропов по track_id для temporal action recognition."""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ClipBuffer:
    """
    Хранит последние ``clip_length`` кадров (RGB crop) на трек.
    Очистка «забытых» треков по разнице индексов кадров.
    """

    def __init__(
        self,
        clip_length: int = 16,
        max_tracks: int = 128,
        stale_frame_gap: int = 120,
    ) -> None:
        self.clip_length = max(2, int(clip_length))
        self.max_tracks = max(1, int(max_tracks))
        self.stale_frame_gap = max(1, int(stale_frame_gap))
        self._buf: Dict[int, Deque[np.ndarray]] = {}
        self._last_seen: Dict[int, int] = {}

    def _cleanup(self, frame_index: int) -> None:
        dead = [t for t, last in self._last_seen.items() if frame_index - last > self.stale_frame_gap]
        for t in dead:
            self._buf.pop(t, None)
            self._last_seen.pop(t, None)
        if len(self._buf) > self.max_tracks:
            sorted_ids = sorted(self._last_seen.keys(), key=lambda k: self._last_seen[k])
            for t in sorted_ids[: len(self._buf) - self.max_tracks]:
                self._buf.pop(t, None)
                self._last_seen.pop(t, None)

    def update(self, track_id: int, frame_crop: np.ndarray, frame_index: int) -> None:
        if frame_crop is None or frame_crop.size == 0:
            return
        if frame_crop.ndim != 3 or frame_crop.shape[2] != 3:
            logger.warning("ClipBuffer: expected HxWx3 RGB crop, got shape %s", getattr(frame_crop, "shape", None))
            return
        tid = int(track_id)
        if tid not in self._buf:
            self._buf[tid] = deque(maxlen=self.clip_length)
        self._buf[tid].append(np.asarray(frame_crop, dtype=np.uint8).copy())
        self._last_seen[tid] = int(frame_index)
        self._cleanup(int(frame_index))

    def is_ready(self, track_id: int) -> bool:
        q = self._buf.get(int(track_id))
        return q is not None and len(q) >= self.clip_length

    def get_clip(self, track_id: int) -> Optional[np.ndarray]:
        """``(T, H, W, 3)`` uint8 или ``None``."""
        q = self._buf.get(int(track_id))
        if not q or len(q) < self.clip_length:
            return None
        return np.stack(list(q), axis=0)

    def clear_track(self, track_id: int) -> None:
        self._buf.pop(int(track_id), None)
        self._last_seen.pop(int(track_id), None)

    def reset(self) -> None:
        self._buf.clear()
        self._last_seen.clear()
