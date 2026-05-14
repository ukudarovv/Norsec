"""Социальные сигналы по трекам и траекториям (Phase 2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from tracking.track_memory import TrackMemory
from tracking import trajectory_analyzer as ta

SOCIAL_SIGNALS = [
    "rapid_approach",
    "following",
    "group_surrounding",
    "crowding",
    "blocking_path",
    "isolation_pressure",
    "person_down_crowding",
]


@dataclass
class SocialSignal:
    signal_type: str
    severity: float
    track_ids: list[int]
    timestamp_sec: float
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "severity": self.severity,
            "track_ids": list(self.track_ids),
            "timestamp_sec": self.timestamp_sec,
            "description": self.description,
            "evidence": dict(self.evidence),
        }


def _social_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    c = dict(cfg or {})
    return dict(c.get("social") or {})


def _dist_centers(memory: TrackMemory, a: int, b: int) -> float | None:
    return ta.distance_between_tracks(memory, a, b)


def _rapid_approach(
    memory: TrackMemory,
    timestamp_sec: float,
    sc: dict[str, Any],
) -> list[SocialSignal]:
    out: list[SocialSignal] = []
    thr = float(sc.get("rapid_approach_px_per_sec", 180))
    ids = memory.get_active_tracks()
    for i, ia in enumerate(ids):
        for ib in ids[i + 1 :]:
            ra, rb = memory.get_track(ia), memory.get_track(ib)
            if ra is None or rb is None or len(ra.positions) < 2 or len(rb.positions) < 2:
                continue
            d_now = ta.distance_between_tracks(memory, ia, ib)
            d_prev = math.hypot(
                ra.positions[-2][0] - rb.positions[-2][0],
                ra.positions[-2][1] - rb.positions[-2][1],
            )
            if d_now is None:
                continue
            fps = float(sc.get("assumed_fps", 30))
            dt = max(1.0 / fps, 1e-3)
            # приближение относительно предыдущего кадра: скорость сближения (px/сек)
            approach_rate = (d_prev - d_now) / dt
            if approach_rate > thr:
                sev = min(1.0, approach_rate / (thr * 2))
                out.append(
                    SocialSignal(
                        signal_type="rapid_approach",
                        severity=float(sev),
                        track_ids=[ia, ib],
                        timestamp_sec=float(timestamp_sec),
                        description="Risk signal detected: rapid decrease in interpersonal distance. Requires review.",
                        evidence={
                            "approach_rate_px_per_sec": approach_rate,
                            "distance_px": d_now,
                            "threshold_px_per_sec": thr,
                        },
                    )
                )
    return out


def _following(memory: TrackMemory, timestamp_sec: float, sc: dict[str, Any]) -> list[SocialSignal]:
    out: list[SocialSignal] = []
    min_sec = float(sc.get("following_min_sec", 4))
    ids = memory.get_active_tracks()
    for i, ia in enumerate(ids):
        for ib in ids[i + 1 :]:
            if ta.is_following(memory, ia, ib, min_sec=min_sec):
                out.append(
                    SocialSignal(
                        signal_type="following",
                        severity=0.65,
                        track_ids=[ia, ib],
                        timestamp_sec=float(timestamp_sec),
                        description="Risk signal detected: one track remains behind another along motion. Requires review.",
                        evidence={"min_sec": min_sec, "leader": ib, "follower": ia},
                    )
                )
            elif ta.is_following(memory, ib, ia, min_sec=min_sec):
                out.append(
                    SocialSignal(
                        signal_type="following",
                        severity=0.65,
                        track_ids=[ib, ia],
                        timestamp_sec=float(timestamp_sec),
                        description="Risk signal detected: one track remains behind another along motion. Requires review.",
                        evidence={"min_sec": min_sec, "leader": ia, "follower": ib},
                    )
                )
    return out


def _group_surrounding(
    memory: TrackMemory,
    timestamp_sec: float,
    sc: dict[str, Any],
) -> list[SocialSignal]:
    out: list[SocialSignal] = []
    close_px = float(sc.get("close_distance_px", 120))
    min_p = int(sc.get("surrounding_min_people", 3))
    ids = memory.get_active_tracks()
    for center_tid in ids:
        others = [t for t in ids if t != center_tid]
        near = [t for t in others if (d := _dist_centers(memory, center_tid, t)) is not None and d <= close_px]
        if len(near) >= min_p:
            out.append(
                SocialSignal(
                    signal_type="group_surrounding",
                    severity=min(1.0, 0.5 + 0.1 * len(near)),
                    track_ids=[center_tid, *near[:6]],
                    timestamp_sec=float(timestamp_sec),
                    description="Risk signal detected: multiple people within close range of one track. Requires review.",
                    evidence={"close_distance_px": close_px, "neighbor_count": len(near)},
                )
            )
    return out


def _crowding(
    memory: TrackMemory,
    timestamp_sec: float,
    sc: dict[str, Any],
) -> list[SocialSignal]:
    out: list[SocialSignal] = []
    radius = float(sc.get("crowding_zone_radius_px", 200))
    min_people = int(sc.get("crowding_min_people", 4))
    ids = memory.get_active_tracks()
    if len(ids) < min_people:
        return out
    # якорь — медиана по первому треку как центр зоны (упрощение MVP)
    anchor = ids[0]
    rec0 = memory.get_track(anchor)
    if rec0 is None or not rec0.positions:
        return out
    cx, cy = rec0.positions[-1]
    in_zone = []
    for tid in ids:
        r = memory.get_track(tid)
        if r is None or not r.positions:
            continue
        x, y = r.positions[-1]
        if math.hypot(x - cx, y - cy) <= radius:
            in_zone.append(tid)
    if len(in_zone) >= min_people:
        out.append(
            SocialSignal(
                signal_type="crowding",
                severity=min(1.0, 0.4 + 0.08 * len(in_zone)),
                track_ids=in_zone[:12],
                timestamp_sec=float(timestamp_sec),
                description="Risk signal detected: elevated local density. Requires review.",
                evidence={"radius_px": radius, "count": len(in_zone)},
            )
        )
    return out


def _blocking_path(
    memory: TrackMemory,
    people: list[dict[str, Any]],
    timestamp_sec: float,
    sc: dict[str, Any],
) -> list[SocialSignal]:
    out: list[SocialSignal] = []
    angle_deg = float(sc.get("blocking_angle_deg", 35))
    cos_thr = math.cos(math.radians(angle_deg))
    ids = memory.get_active_tracks()
    for mover in ids:
        mv = ta.movement_vector_from_history(memory, mover, window=6)
        if mv is None or math.hypot(mv[0], mv[1]) < 10.0:
            continue
        u = ta.normalize_vec(mv[0], mv[1])
        if u is None:
            continue
        bb_m = ta.bbox_from_people(people, mover)
        if bb_m is None:
            continue
        mx = (bb_m[0] + bb_m[2]) * 0.5
        my = (bb_m[1] + bb_m[3]) * 0.5
        for other in ids:
            if other == mover:
                continue
            bb_o = ta.bbox_from_people(people, other)
            if bb_o is None:
                continue
            ox = (bb_o[0] + bb_o[2]) * 0.5
            oy = (bb_o[1] + bb_o[3]) * 0.5
            vx, vy = ox - mx, oy - my
            w = math.hypot(vx, vy)
            if w < 1e-6:
                continue
            cos_align = abs((vx * u[0] + vy * u[1]) / w)
            if cos_align < cos_thr:
                continue
            frac = ta.bbox_overlap_fraction(bb_m, bb_o)
            if frac > 0.15:
                out.append(
                    SocialSignal(
                        signal_type="blocking_path",
                        severity=min(1.0, 0.55 + frac),
                        track_ids=[mover, other],
                        timestamp_sec=float(timestamp_sec),
                        description="Risk signal detected: overlap along motion corridor. Requires review.",
                        evidence={"overlap_fraction": frac, "angle_deg": angle_deg},
                    )
                )
    return out


def _isolation_pressure(
    memory: TrackMemory,
    timestamp_sec: float,
    sc: dict[str, Any],
) -> list[SocialSignal]:
    out: list[SocialSignal] = []
    close_px = float(sc.get("close_distance_px", 120))
    ids = memory.get_active_tracks()
    for target in ids:
        rt = memory.get_track(target)
        if rt is None or len(rt.velocity_history) < 2:
            continue
        v = rt.velocity_history[-1]
        if math.hypot(v[0], v[1]) > 25.0:
            continue
        group = [t for t in ids if t != target]
        movers = []
        for g in group:
            vg = ta.compute_velocity(memory, g)
            if vg is None:
                continue
            if math.hypot(vg[0], vg[1]) < 20.0:
                continue
            if ta.is_moving_towards(memory, g, target) and (_dist_centers(memory, g, target) or 999) < close_px * 3:
                movers.append(g)
        if len(movers) >= 2:
            out.append(
                SocialSignal(
                    signal_type="isolation_pressure",
                    severity=0.7,
                    track_ids=[target, *movers[:5]],
                    timestamp_sec=float(timestamp_sec),
                    description="Risk signal detected: group motion toward a comparatively static person. Requires review.",
                    evidence={"moving_toward_count": len(movers)},
                )
            )
    return out


def _person_down_crowding(
    memory: TrackMemory,
    timestamp_sec: float,
    sc: dict[str, Any],
    pose_flags: dict[int, list[str]] | None,
) -> list[SocialSignal]:
    out: list[SocialSignal] = []
    pf = pose_flags or {}
    close_px = float(sc.get("close_distance_px", 120))
    for tid, flags in pf.items():
        if not any(x in flags for x in ("person_on_ground", "person_falling")):
            continue
        others = [t for t in memory.get_active_tracks() if t != tid]
        near = [t for t in others if (d := _dist_centers(memory, tid, t)) is not None and d <= close_px * 1.5]
        if len(near) >= 2:
            out.append(
                SocialSignal(
                    signal_type="person_down_crowding",
                    severity=0.75,
                    track_ids=[tid, *near[:6]],
                    timestamp_sec=float(timestamp_sec),
                    description="Risk signal detected: person low to ground with nearby group. Requires review.",
                    evidence={"pose_flags": flags, "neighbor_count": len(near)},
                )
            )
    return out


def detect_social_signals(
    memory: TrackMemory,
    people: list[dict[str, Any]],
    timestamp_sec: float,
    analytics_cfg: dict[str, Any] | None = None,
    *,
    pose_flags_by_track: dict[int, list[str]] | None = None,
) -> list[SocialSignal]:
    sc = _social_cfg(analytics_cfg)
    signals: list[SocialSignal] = []
    signals.extend(_rapid_approach(memory, timestamp_sec, sc))
    signals.extend(_following(memory, timestamp_sec, sc))
    signals.extend(_group_surrounding(memory, timestamp_sec, sc))
    signals.extend(_crowding(memory, timestamp_sec, sc))
    signals.extend(_blocking_path(memory, people, timestamp_sec, sc))
    signals.extend(_isolation_pressure(memory, timestamp_sec, sc))
    signals.extend(_person_down_crowding(memory, timestamp_sec, sc, pose_flags_by_track))
    return signals
