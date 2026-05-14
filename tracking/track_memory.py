"""Хранение истории треков для анализа траекторий (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tracking.tracking_config import tracking_params


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass
class TrackRecord:
    track_id: int
    first_seen: float
    last_seen: float
    positions: list[tuple[float, float]] = field(default_factory=list)
    bbox_history: list[tuple[float, float, float, float]] = field(default_factory=list)
    velocity_history: list[tuple[float, float]] = field(default_factory=list)
    lost_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "positions": [list(p) for p in self.positions],
            "bbox_history": [list(b) for b in self.bbox_history],
            "velocity_history": [list(v) for v in self.velocity_history],
            "lost_frames": self.lost_frames,
        }


class TrackMemory:
    """Хранит по ``track_id`` историю позиций, bbox и скоростей."""

    def __init__(self, analytics_cfg: dict[str, Any] | None = None) -> None:
        self._cfg = tracking_params(analytics_cfg)
        self._tracks: dict[int, TrackRecord] = {}

    def update(self, track_id: int, bbox: tuple[float, float, float, float], timestamp_sec: float) -> None:
        max_pos = self._cfg["max_positions"]
        cx, cy = _bbox_center(bbox)
        rec = self._tracks.get(track_id)
        if rec is None:
            self._tracks[track_id] = TrackRecord(
                track_id=track_id,
                first_seen=timestamp_sec,
                last_seen=timestamp_sec,
                positions=[(cx, cy)],
                bbox_history=[bbox],
                velocity_history=[],
                lost_frames=0,
            )
            return

        prev_t = rec.last_seen
        dt = max(timestamp_sec - prev_t, 1e-6)
        rec.lost_frames = 0
        if rec.positions:
            px, py = rec.positions[-1]
            vx = (cx - px) / dt
            vy = (cy - py) / dt
            rec.velocity_history.append((vx, vy))
            if len(rec.velocity_history) > max_pos:
                rec.velocity_history = rec.velocity_history[-max_pos:]
        rec.positions.append((cx, cy))
        rec.bbox_history.append(bbox)
        rec.last_seen = timestamp_sec
        if len(rec.positions) > max_pos:
            rec.positions = rec.positions[-max_pos:]
        if len(rec.bbox_history) > max_pos:
            rec.bbox_history = rec.bbox_history[-max_pos:]

    def mark_lost(self, track_id: int) -> None:
        if track_id in self._tracks:
            self._tracks[track_id].lost_frames += 1

    def get_track(self, track_id: int) -> TrackRecord | None:
        return self._tracks.get(track_id)

    def get_active_tracks(self) -> list[int]:
        return sorted(self._tracks.keys())

    def cleanup_old_tracks(self, current_time: float) -> list[int]:
        ttl = self._cfg["track_ttl_sec"]
        removed: list[int] = []
        for tid, rec in list(self._tracks.items()):
            if current_time - rec.last_seen > ttl:
                del self._tracks[tid]
                removed.append(tid)
        return removed

    def all_records(self) -> dict[int, TrackRecord]:
        return dict(self._tracks)
