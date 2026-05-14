"""Анализ траекторий по ``TrackMemory`` (Phase 2)."""

from __future__ import annotations

import math
from typing import Any

from tracking.track_memory import TrackMemory


def normalize_vec(vx: float, vy: float) -> tuple[float, float] | None:
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return None
    return vx / n, vy / n


def compute_velocity(memory: TrackMemory, track_id: int) -> tuple[float, float] | None:
    rec = memory.get_track(track_id)
    if rec is None or not rec.velocity_history:
        return None
    return rec.velocity_history[-1]


def compute_speed(memory: TrackMemory, track_id: int) -> float | None:
    v = compute_velocity(memory, track_id)
    if v is None:
        return None
    return math.hypot(v[0], v[1])


def compute_direction(memory: TrackMemory, track_id: int) -> float | None:
    """Направление движения в радианах (0 = вправо по оси x)."""
    v = compute_velocity(memory, track_id)
    if v is None:
        return None
    u = normalize_vec(v[0], v[1])
    if u is None:
        return None
    return math.atan2(u[1], u[0])


def sharp_acceleration(
    memory: TrackMemory,
    track_id: int,
    *,
    threshold_px_per_sec2: float = 400.0,
) -> bool:
    rec = memory.get_track(track_id)
    if rec is None or len(rec.velocity_history) < 2:
        return False
    v1 = rec.velocity_history[-2]
    v2 = rec.velocity_history[-1]
    # приближённо: |Δv| за один шаг истории (без точного dt между шагами)
    dvx, dvy = v2[0] - v1[0], v2[1] - v1[1]
    return math.hypot(dvx, dvy) >= threshold_px_per_sec2


def is_moving_towards(
    memory: TrackMemory,
    track_a: int,
    track_b: int,
    *,
    min_cosine: float = 0.35,
) -> bool:
    """``track_a`` движется в сторону центра ``track_b`` (по последней скорости)."""
    va = compute_velocity(memory, track_a)
    rec_a = memory.get_track(track_a)
    rec_b = memory.get_track(track_b)
    if va is None or rec_a is None or rec_b is None:
        return False
    if not rec_a.positions or not rec_b.positions:
        return False
    ax, ay = rec_a.positions[-1]
    bx, by = rec_b.positions[-1]
    dx, dy = bx - ax, by - ay
    u = normalize_vec(dx, dy)
    uv = normalize_vec(va[0], va[1])
    if u is None or uv is None:
        return False
    dot = uv[0] * u[0] + uv[1] * u[1]
    return dot >= min_cosine


def is_following(
    memory: TrackMemory,
    track_a: int,
    track_b: int,
    *,
    min_sec: float = 4.0,
    max_lateral_px: float = 80.0,
    min_align_cos: float = 0.4,
) -> bool:
    """
    Упрощённо: ``a`` следует за ``b``, если оба движутся вдоль одной линии,
    ``a`` позади ``b`` по направлению движения ``b``, и пересечение по времени ≥ ``min_sec``.
    """
    rec_a = memory.get_track(track_a)
    rec_b = memory.get_track(track_b)
    if rec_a is None or rec_b is None:
        return False
    if not rec_a.positions or not rec_b.positions:
        return False
    vb = compute_velocity(memory, track_b)
    if vb is None:
        return False
    ub = normalize_vec(vb[0], vb[1])
    if ub is None:
        return False
    bx, by = rec_b.positions[-1]
    ax, ay = rec_a.positions[-1]
    # проекция (a-b) на направление vb: отрицательная => a сзади b
    relx, rely = ax - bx, ay - by
    along = relx * ub[0] + rely * ub[1]
    if along > 20.0:
        return False
    lateral = abs(relx * ub[1] - rely * ub[0])
    if lateral > max_lateral_px:
        return False
    va = compute_velocity(memory, track_a)
    if va is None:
        return False
    ua = normalize_vec(va[0], va[1])
    if ua is None:
        return False
    align = ua[0] * ub[0] + ua[1] * ub[1]
    if align < min_align_cos:
        return False
    overlap = min(rec_a.last_seen, rec_b.last_seen) - max(rec_a.first_seen, rec_b.first_seen)
    return overlap >= min_sec


def stopped_after_motion(
    memory: TrackMemory,
    track_id: int,
    *,
    still_speed_px_per_sec: float = 15.0,
    min_prior_speed: float = 40.0,
) -> bool:
    """Резкая остановка после заметного движения (эвристика «остановка после падения»)."""
    rec = memory.get_track(track_id)
    if rec is None or len(rec.velocity_history) < 3:
        return False
    recent = rec.velocity_history[-3:]
    speeds = [math.hypot(v[0], v[1]) for v in recent]
    if max(speeds[:-1]) < min_prior_speed:
        return False
    return speeds[-1] < still_speed_px_per_sec


def distance_between_tracks(memory: TrackMemory, a: int, b: int) -> float | None:
    ra, rb = memory.get_track(a), memory.get_track(b)
    if ra is None or rb is None or not ra.positions or not rb.positions:
        return None
    ax, ay = ra.positions[-1]
    bx, by = rb.positions[-1]
    return math.hypot(ax - bx, ay - by)


def bbox_overlap_fraction(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    return inter / aa


def movement_vector_from_history(
    memory: TrackMemory, track_id: int, window: int = 5
) -> tuple[float, float] | None:
    rec = memory.get_track(track_id)
    if rec is None or len(rec.positions) < 2:
        return None
    pts = rec.positions[-window:]
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    return (x1 - x0, y1 - y0)


def bbox_from_people(people: list[dict[str, Any]], track_id: int) -> tuple[float, float, float, float] | None:
    for p in people:
        if int(p.get("track_id", -1)) != track_id:
            continue
        bb = p.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            return float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    return None
