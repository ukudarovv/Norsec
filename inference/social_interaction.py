"""Эвристики социальных рисков по трекам и траекториям (этап 3, не диагноз буллинга)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from inference.person_tracker import TrackedPerson

logger = logging.getLogger(__name__)

SIGNAL_RAPID_APPROACH = "rapid_approach"
SIGNAL_FOLLOWING = "following"
SIGNAL_GROUP_SURROUNDING = "group_surrounding"
SIGNAL_CROWDING = "crowding"
SIGNAL_ISOLATION_PRESSURE = "isolation_pressure"


@dataclass
class SocialSignal:
    signal_type: str
    severity: float
    track_ids: List[int]
    description: str
    timestamp_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "severity": round(float(self.severity), 4),
            "track_ids": list(self.track_ids),
            "description": self.description,
            "timestamp_sec": round(float(self.timestamp_sec), 4),
        }


def _bbox_center(tp: TrackedPerson) -> Tuple[float, float]:
    x1, y1, x2, y2 = tp.bbox
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def compute_distance_matrix(tracked_people: List[TrackedPerson]) -> dict[str, float]:
    """Попарные евклидовы расстояния между центрами bbox; ключи вида ``\"1-2\"`` (меньший id первым)."""
    out: dict[str, float] = {}
    n = len(tracked_people)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tracked_people[i], tracked_people[j]
            ca = _bbox_center(a)
            cb = _bbox_center(b)
            d = float(math.hypot(ca[0] - cb[0], ca[1] - cb[1]))
            i1, i2 = sorted((a.track_id, b.track_id))
            out[f"{i1}-{i2}"] = round(d, 4)
    return out


def _normalize_trajectories(trajectories: Dict[Any, list]) -> dict[int, list]:
    out: dict[int, list] = {}
    if not trajectories:
        return out
    for k, v in trajectories.items():
        try:
            ki = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, list):
            out[ki] = v
    return out


def _history_in_window(
    hist: list[dict[str, Any]], timestamp_sec: float, window_sec: float
) -> list[dict[str, Any]]:
    t0 = timestamp_sec - window_sec
    return [p for p in hist if float(p.get("timestamp_sec", 0.0)) >= t0]


def _velocity_from_history(hist: list[dict[str, Any]]) -> Tuple[float, float] | None:
    if len(hist) < 2:
        return None
    p0, p1 = hist[-2], hist[-1]
    t0 = float(p0["timestamp_sec"])
    t1 = float(p1["timestamp_sec"])
    dt = t1 - t0
    if dt < 1e-6:
        return None
    vx = (float(p1["center_x"]) - float(p0["center_x"])) / dt
    vy = (float(p1["center_y"]) - float(p0["center_y"])) / dt
    return vx, vy


def _norm2(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


class SocialInteractionAnalyzer:
    """Эвристики: резкое сближение, преследование, окружение, скопление, давление группы."""

    def __init__(
        self,
        close_distance_px: float = 120.0,
        rapid_approach_px_per_sec: float = 180.0,
        following_min_duration_sec: float = 4.0,
        surrounding_min_people: int = 3,
        history_window_sec: float = 6.0,
    ) -> None:
        self.close_distance_px = float(close_distance_px)
        self.rapid_approach_px_per_sec = float(rapid_approach_px_per_sec)
        self.following_min_duration_sec = float(following_min_duration_sec)
        self.surrounding_min_people = int(surrounding_min_people)
        self.history_window_sec = float(history_window_sec)
        self._pair_last: dict[tuple[int, int], tuple[float, float]] = {}
        self._following_accum: dict[tuple[int, int], float] = {}
        self._following_emitted: set[tuple[int, int]] = set()
        self._isolation_still_since: dict[int, float | None] = {}
        self._last_global_ts: float | None = None

    def reset(self) -> None:
        self._pair_last.clear()
        self._following_accum.clear()
        self._following_emitted.clear()
        self._isolation_still_since.clear()
        self._last_global_ts = None

    def update(
        self,
        tracked_people: List[TrackedPerson],
        trajectories: Dict[int, list],
        timestamp_sec: float,
    ) -> List[SocialSignal]:
        signals: List[SocialSignal] = []
        if not tracked_people:
            return signals

        traj = _normalize_trajectories(trajectories)

        ts = float(timestamp_sec)
        prev_ts = self._last_global_ts
        self._last_global_ts = ts
        dt_global = (ts - prev_ts) if prev_ts is not None else 0.12
        dt_global = max(1e-3, min(dt_global, 2.0))

        centers = {tp.track_id: _bbox_center(tp) for tp in tracked_people}
        dist_mat = compute_distance_matrix(tracked_people)

        signals.extend(self._signals_rapid_approach(dist_mat, ts))
        signals.extend(self._signals_following(tracked_people, traj, centers, ts, dt_global))
        signals.extend(self._signals_group_surrounding(tracked_people, centers, ts))
        signals.extend(self._signals_crowding(tracked_people, dist_mat, ts))
        signals.extend(self._signals_isolation_pressure(tracked_people, traj, centers, dist_mat, ts))

        return signals

    def _signals_rapid_approach(self, dist_mat: dict[str, float], ts: float) -> List[SocialSignal]:
        out: List[SocialSignal] = []
        for key, d_now in dist_mat.items():
            parts = key.split("-")
            if len(parts) != 2:
                continue
            a, b = int(parts[0]), int(parts[1])
            pair = (min(a, b), max(a, b))
            prev = self._pair_last.get(pair)
            self._pair_last[pair] = (ts, d_now)
            if prev is None:
                continue
            t_prev, d_prev = prev
            dt = ts - t_prev
            if dt <= 1e-6:
                continue
            closing = d_prev - d_now
            if closing > self.rapid_approach_px_per_sec * dt:
                rate = closing / dt
                sev = min(1.0, rate / max(self.rapid_approach_px_per_sec, 1e-6))
                out.append(
                    SocialSignal(
                        signal_type=SIGNAL_RAPID_APPROACH,
                        severity=float(sev),
                        track_ids=[a, b],
                        description=f"Social risk signal detected: Track {a} rapidly approached Track {b}",
                        timestamp_sec=ts,
                    )
                )
        return out

    def _signals_following(
        self,
        tracked_people: List[TrackedPerson],
        traj: dict[int, list],
        centers: dict[int, Tuple[float, float]],
        ts: float,
        dt_global: float,
    ) -> List[SocialSignal]:
        out: List[SocialSignal] = []
        ids = [tp.track_id for tp in tracked_people]
        seen_pairs: set[tuple[int, int]] = set()
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1 :]:
                ha = _history_in_window(traj.get(id_a, []), ts, self.history_window_sec)
                hb = _history_in_window(traj.get(id_b, []), ts, self.history_window_sec)
                va = _velocity_from_history(ha)
                vb = _velocity_from_history(hb)
                if va is None or vb is None:
                    self._following_accum.pop((id_a, id_b), None)
                    self._following_accum.pop((id_b, id_a), None)
                    continue
                na, nb = _norm2(va), _norm2(vb)
                if na < 8.0 or nb < 8.0:
                    self._following_accum.pop((id_a, id_b), None)
                    self._following_accum.pop((id_b, id_a), None)
                    continue
                cos_align = _dot(va, vb) / (na * nb)
                if cos_align < 0.55:
                    self._following_accum.pop((id_a, id_b), None)
                    self._following_accum.pop((id_b, id_a), None)
                    continue
                ca, cb = centers[id_a], centers[id_b]
                rab = (cb[0] - ca[0], cb[1] - ca[1])
                rba = (ca[0] - cb[0], ca[1] - cb[1])
                behind_ab = _dot(rab, vb) < -5.0
                behind_ba = _dot(rba, va) < -5.0
                d = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
                if not (self.close_distance_px * 0.35 < d < self.close_distance_px * 2.2):
                    self._following_accum.pop((id_a, id_b), None)
                    self._following_accum.pop((id_b, id_a), None)
                    self._following_emitted.discard((id_a, id_b))
                    self._following_emitted.discard((id_b, id_a))
                    continue
                if behind_ab:
                    pair = (id_a, id_b)
                elif behind_ba:
                    pair = (id_b, id_a)
                else:
                    self._following_accum.pop((id_a, id_b), None)
                    self._following_accum.pop((id_b, id_a), None)
                    self._following_emitted.discard((id_a, id_b))
                    self._following_emitted.discard((id_b, id_a))
                    continue
                follower, leader = pair
                seen_pairs.add((follower, leader))
                prev = self._following_accum.get(pair, 0.0)
                acc = prev + dt_global
                self._following_accum[pair] = acc
                if acc >= self.following_min_duration_sec and pair not in self._following_emitted:
                    self._following_emitted.add(pair)
                    out.append(
                        SocialSignal(
                            signal_type=SIGNAL_FOLLOWING,
                            severity=min(1.0, acc / max(self.following_min_duration_sec * 1.5, 1e-6)),
                            track_ids=[follower, leader],
                            description=(
                                f"Social risk signal detected: Track {follower} may be following Track {leader}"
                            ),
                            timestamp_sec=ts,
                        )
                    )
        for key in list(self._following_accum.keys()):
            if key not in seen_pairs:
                self._following_accum.pop(key, None)
                self._following_emitted.discard(key)
        return out

    def _signals_group_surrounding(
        self,
        tracked_people: List[TrackedPerson],
        centers: dict[int, Tuple[float, float]],
        ts: float,
    ) -> List[SocialSignal]:
        out: List[SocialSignal] = []
        if len(tracked_people) < self.surrounding_min_people:
            return out
        pts = [(tid, centers[tid]) for tid in centers]
        gx = sum(p[1][0] for p in pts) / len(pts)
        gy = sum(p[1][1] for p in pts) / len(pts)
        inner = min(pts, key=lambda p: math.hypot(p[1][0] - gx, p[1][1] - gy))
        target_id, tc = inner[0], inner[1]
        others = [(tid, c) for tid, c in pts if tid != target_id]
        if len(others) < self.surrounding_min_people - 1:
            return out
        close_others: list[tuple[int, Tuple[float, float]]] = []
        for oid, oc in others:
            if math.hypot(oc[0] - tc[0], oc[1] - tc[1]) <= self.close_distance_px * 1.1:
                close_others.append((oid, oc))
        if len(close_others) < self.surrounding_min_people - 1:
            return out
        angles = []
        for oid, oc in close_others:
            angles.append(math.atan2(oc[1] - tc[1], oc[0] - tc[0]))
        angles.sort()
        if len(angles) >= 2:
            spans = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
            spans.append(angles[0] + 2 * math.pi - angles[-1])
            max_gap = max(spans) if spans else 2 * math.pi
        else:
            max_gap = 2 * math.pi
        if max_gap > 2 * math.pi * 0.55:
            return out
        involved = [target_id] + [o[0] for o in close_others]
        ring_ids = sorted(set(involved) - {target_id})
        sev = min(1.0, len(close_others) / max(self.surrounding_min_people, 1))
        out.append(
            SocialSignal(
                signal_type=SIGNAL_GROUP_SURROUNDING,
                severity=float(sev),
                track_ids=sorted(set(involved)),
                description=(
                    f"Potential group pressure: Track {target_id} appears surrounded by tracks "
                    f"{', '.join(str(x) for x in ring_ids)}"
                ),
                timestamp_sec=ts,
            )
        )
        return out

    def _signals_crowding(
        self,
        tracked_people: List[TrackedPerson],
        dist_mat: dict[str, float],
        ts: float,
    ) -> List[SocialSignal]:
        out: List[SocialSignal] = []
        if len(tracked_people) < 3:
            return out
        close_pairs = 0
        for d in dist_mat.values():
            if d < self.close_distance_px * 0.75:
                close_pairs += 1
        max_pairs = len(tracked_people) * (len(tracked_people) - 1) // 2
        if max_pairs == 0:
            return out
        ratio = close_pairs / max_pairs
        if ratio >= 0.45 and close_pairs >= 2:
            ids = sorted(tp.track_id for tp in tracked_people)
            out.append(
                SocialSignal(
                    signal_type=SIGNAL_CROWDING,
                    severity=min(1.0, ratio),
                    track_ids=ids,
                    description="Social risk signal detected: multiple people crowding in one zone",
                    timestamp_sec=ts,
                )
            )
        return out

    def _signals_isolation_pressure(
        self,
        tracked_people: List[TrackedPerson],
        traj: dict[int, list],
        centers: dict[int, Tuple[float, float]],
        dist_mat: dict[str, float],
        ts: float,
    ) -> List[SocialSignal]:
        out: List[SocialSignal] = []
        if len(tracked_people) < 2:
            return out
        for tp in tracked_people:
            tid = tp.track_id
            hist = _history_in_window(traj.get(tid, []), ts, self.history_window_sec)
            if len(hist) < 2:
                continue
            p_first, p_last = hist[0], hist[-1]
            disp = math.hypot(
                float(p_last["center_x"]) - float(p_first["center_x"]),
                float(p_last["center_y"]) - float(p_first["center_y"]),
            )
            span = float(p_last["timestamp_sec"]) - float(p_first["timestamp_sec"])
            if span < 1.0:
                continue
            speed = disp / span
            if speed > 25.0:
                self._isolation_still_since.pop(tid, None)
                continue
            if self._isolation_still_since.get(tid) is None:
                self._isolation_still_since[tid] = ts
            t_still = self._isolation_still_since[tid]
            if t_still is None or (ts - t_still) < 2.0:
                continue
            others_approaching = 0
            min_d_now = 1e9
            for oid in centers:
                if oid == tid:
                    continue
                k = f"{min(tid, oid)}-{max(tid, oid)}"
                d = dist_mat.get(k)
                if d is None:
                    continue
                min_d_now = min(min_d_now, d)
                oh = _history_in_window(traj.get(oid, []), ts, self.history_window_sec)
                if len(oh) < 2:
                    continue
                o0, o1 = oh[0], oh[-1]
                d_old = math.hypot(
                    float(o0["center_x"]) - float(p_first["center_x"]),
                    float(o0["center_y"]) - float(p_first["center_y"]),
                )
                d_new = math.hypot(
                    float(o1["center_x"]) - float(p_last["center_x"]),
                    float(o1["center_y"]) - float(p_last["center_y"]),
                )
                if d_new < d_old - 15.0:
                    others_approaching += 1
            if others_approaching >= 2 and min_d_now < self.close_distance_px * 1.8:
                involved = [tid] + [o for o in centers if o != tid][:8]
                out.append(
                    SocialSignal(
                        signal_type=SIGNAL_ISOLATION_PRESSURE,
                        severity=min(1.0, 0.4 + 0.2 * others_approaching),
                        track_ids=sorted(set(involved)),
                        description=(
                            f"Potential group pressure: nearly stationary Track {tid} while "
                            f"multiple others reduce distance"
                        ),
                        timestamp_sec=ts,
                    )
                )
        return out


def draw_social_signals(
    frame_rgb: np.ndarray,
    tracked_people: List[TrackedPerson],
    signals: List[SocialSignal],
) -> np.ndarray:
    """Линии между участниками сигнала, подпись типа и severity."""
    import cv2

    img = frame_rgb.copy()
    h, w = img.shape[:2]
    cmap = {
        SIGNAL_RAPID_APPROACH: (0, 140, 255),
        SIGNAL_FOLLOWING: (180, 0, 255),
        SIGNAL_GROUP_SURROUNDING: (0, 0, 200),
        SIGNAL_CROWDING: (0, 165, 255),
        SIGNAL_ISOLATION_PRESSURE: (80, 80, 255),
    }
    id_to_center: dict[int, Tuple[int, int]] = {}
    for tp in tracked_people:
        cx, cy = _bbox_center(tp)
        ix = int(max(0, min(round(cx), w - 1)))
        iy = int(max(0, min(round(cy), h - 1)))
        id_to_center[tp.track_id] = (ix, iy)

    y_off = 24
    for sig in signals:
        color = cmap.get(sig.signal_type, (200, 200, 200))
        tids = sig.track_ids
        pts = [id_to_center[t] for t in tids if t in id_to_center]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)
        if len(pts) >= 3:
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                cv2.line(img, pts[i], pts[j], color, 1, cv2.LINE_AA)
        anchor = pts[0] if pts else (20, y_off)
        label = f"{sig.signal_type} sev={sig.severity:.2f}"
        cv2.putText(
            img,
            label,
            (anchor[0] + 4, max(anchor[1] - 6, y_off)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        y_off += 18
    return img
