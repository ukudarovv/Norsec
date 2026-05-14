"""Сводные признаки по треку для аналитики и обучения (Phase 2)."""

from __future__ import annotations

import math
from typing import Any

from tracking.track_memory import TrackMemory
from tracking import trajectory_analyzer as ta


def track_feature_vector(memory: TrackMemory, track_id: int) -> dict[str, Any]:
    """Плоский словарь признаков для логирования / future training."""
    v = ta.compute_velocity(memory, track_id)
    speed = ta.compute_speed(memory, track_id)
    direction = ta.compute_direction(memory, track_id)
    return {
        "track_id": track_id,
        "speed_px_per_sec": speed,
        "velocity_x": v[0] if v else None,
        "velocity_y": v[1] if v else None,
        "direction_rad": direction,
        "sharp_acceleration": ta.sharp_acceleration(memory, track_id),
        "stopped_after_motion": ta.stopped_after_motion(memory, track_id),
    }
