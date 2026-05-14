"""Эвристики риска по позе (этап 4); формулировки — pose risk signal, не буллинг."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from inference.pose_estimator import PersonPose, PoseKeypoint

logger = logging.getLogger(__name__)

SIGNAL_RAISED_HAND = "raised_hand"
SIGNAL_FAST_ARM_MOTION = "fast_arm_motion"
SIGNAL_PERSON_FALLING = "person_falling"
SIGNAL_PERSON_ON_GROUND = "person_on_ground"
SIGNAL_AGGRESSIVE_BODY_LEAN = "aggressive_body_lean"
SIGNAL_CLOSE_HAND_TO_OTHER = "close_hand_to_other_person"


@dataclass
class PoseSignal:
    signal_type: str
    severity: float
    track_id: int
    timestamp_sec: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "severity": round(float(self.severity), 4),
            "track_id": int(self.track_id),
            "timestamp_sec": round(float(self.timestamp_sec), 4),
            "description": self.description,
        }


def _kp(pose: PersonPose, name: str, min_c: float) -> PoseKeypoint | None:
    k = pose.keypoints.get(name)
    if k is None or k.confidence < min_c:
        return None
    return k


def _bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])


def _point_in_bbox(px: float, py: float, bbox: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


class PoseSignalAnalyzer:
    def __init__(
        self,
        min_keypoint_confidence: float = 0.35,
        raised_hand_margin_px: float = 30.0,
        on_ground_aspect_ratio: float = 1.35,
        fast_motion_px_per_sec: float = 250.0,
        lean_angle_deg: float = 38.0,
        close_hand_px: float = 70.0,
        falling_hip_speed_px_per_sec: float = 380.0,
    ) -> None:
        self.min_keypoint_confidence = float(min_keypoint_confidence)
        self.raised_hand_margin_px = float(raised_hand_margin_px)
        self.on_ground_aspect_ratio = float(on_ground_aspect_ratio)
        self.fast_motion_px_per_sec = float(fast_motion_px_per_sec)
        self.lean_angle_deg = float(lean_angle_deg)
        self.close_hand_px = float(close_hand_px)
        self.falling_hip_speed_px_per_sec = float(falling_hip_speed_px_per_sec)
        self._wrist_prev: dict[int, dict[str, tuple[float, float, float]]] = {}
        self._hip_prev: dict[int, tuple[float, float]] = {}

    def reset(self) -> None:
        self._wrist_prev.clear()
        self._hip_prev.clear()

    def update(self, poses: list[PersonPose], timestamp_sec: float) -> list[PoseSignal]:
        ts = float(timestamp_sec)
        out: list[PoseSignal] = []
        if not poses:
            return out
        mc = self.min_keypoint_confidence

        for pose in poses:
            tid = pose.track_id
            x1, y1, x2, y2 = pose.bbox
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            ar = bw / bh
            if ar > self.on_ground_aspect_ratio:
                sev = min(1.0, (ar - self.on_ground_aspect_ratio) / 1.5)
                out.append(
                    PoseSignal(
                        signal_type=SIGNAL_PERSON_ON_GROUND,
                        severity=float(sev),
                        track_id=tid,
                        timestamp_sec=ts,
                        description=f"Pose risk signal: Track {tid} may be on the ground (wide bbox)",
                    )
                )

            lw = _kp(pose, "left_wrist", mc)
            rw = _kp(pose, "right_wrist", mc)
            ls = _kp(pose, "left_shoulder", mc)
            rs = _kp(pose, "right_shoulder", mc)
            lh = _kp(pose, "left_hip", mc)
            rh = _kp(pose, "right_hip", mc)

            if lw and ls and lw.y < ls.y - self.raised_hand_margin_px:
                out.append(
                    PoseSignal(
                        signal_type=SIGNAL_RAISED_HAND,
                        severity=0.45,
                        track_id=tid,
                        timestamp_sec=ts,
                        description=f"Pose risk signal: Track {tid} has raised hand",
                    )
                )
            if rw and rs and rw.y < rs.y - self.raised_hand_margin_px:
                out.append(
                    PoseSignal(
                        signal_type=SIGNAL_RAISED_HAND,
                        severity=0.45,
                        track_id=tid,
                        timestamp_sec=ts,
                        description=f"Pose risk signal: Track {tid} has raised hand",
                    )
                )

            if lh and rh:
                _hmx = 0.5 * (lh.x + rh.x)
                hmy = 0.5 * (lh.y + rh.y)
                prev = self._hip_prev.get(tid)
                self._hip_prev[tid] = (ts, hmy)
                if prev is not None:
                    pt, py0 = prev
                    dt = ts - pt
                    if dt > 1e-4 and dt >= 0.03:
                        vy = (hmy - py0) / dt
                        if vy > self.falling_hip_speed_px_per_sec:
                            out.append(
                                PoseSignal(
                                    signal_type=SIGNAL_PERSON_FALLING,
                                    severity=min(1.0, vy / max(self.falling_hip_speed_px_per_sec * 1.2, 1.0)),
                                    track_id=tid,
                                    timestamp_sec=ts,
                                    description=f"Pose risk signal: Track {tid} rapid downward body motion",
                                )
                            )

            for side in ("left", "right"):
                wn = f"{side}_wrist"
                w = _kp(pose, wn, mc * 0.5)
                if w is None:
                    continue
                key = f"{tid}:{side}"
                prev_w = self._wrist_prev.get(key)
                self._wrist_prev[key] = (ts, w.x, w.y)
                if prev_w is not None:
                    pt, px0, py0 = prev_w
                    dt = ts - pt
                    if dt > 1e-4 and dt >= 0.03:
                        dist = math.hypot(w.x - px0, w.y - py0)
                        speed = dist / dt
                        if speed > self.fast_motion_px_per_sec:
                            out.append(
                                PoseSignal(
                                    signal_type=SIGNAL_FAST_ARM_MOTION,
                                    severity=min(1.0, speed / max(self.fast_motion_px_per_sec * 1.5, 1.0)),
                                    track_id=tid,
                                    timestamp_sec=ts,
                                    description=f"Pose risk signal: Track {tid} fast arm motion",
                                )
                            )

            if ls and rs and lh and rh:
                sx = 0.5 * (ls.x + rs.x)
                sy = 0.5 * (ls.y + rs.y)
                hx = 0.5 * (lh.x + rh.x)
                hy = 0.5 * (lh.y + rh.y)
                vx, vy = hx - sx, hy - sy
                norm = math.hypot(vx, vy)
                if norm > 5.0:
                    ang = abs(math.atan2(abs(vx), max(abs(vy), 1e-6)))
                    if ang > math.radians(self.lean_angle_deg):
                        out.append(
                            PoseSignal(
                                signal_type=SIGNAL_AGGRESSIVE_BODY_LEAN,
                                severity=min(1.0, ang / math.radians(70.0)),
                                track_id=tid,
                                timestamp_sec=ts,
                                description=f"Pose risk signal: Track {tid} strong body lean",
                            )
                        )

        for i, pa in enumerate(poses):
            for pb in poses[i + 1 :]:
                for wname in ("left_wrist", "right_wrist"):
                    w = _kp(pa, wname, mc * 0.5)
                    if w is None:
                        continue
                    ob = pb.bbox
                    oc = _bbox_center(ob)
                    d = math.hypot(w.x - oc[0], w.y - oc[1])
                    inside = _point_in_bbox(w.x, w.y, ob)
                    if inside or d < self.close_hand_px:
                        out.append(
                            PoseSignal(
                                signal_type=SIGNAL_CLOSE_HAND_TO_OTHER,
                                severity=0.75 if inside else max(0.3, 1.0 - d / max(self.close_hand_px, 1.0)),
                                track_id=pa.track_id,
                                timestamp_sec=ts,
                                description=(
                                    f"Pose risk signal: Track {pa.track_id} hand near Track {pb.track_id}"
                                ),
                            )
                        )

        return out
