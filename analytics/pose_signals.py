"""Поза / ключевые точки — эвристики риска (Phase 2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

POSE_SIGNALS = [
    "raised_hand",
    "fast_arm_motion",
    "defensive_pose",
    "person_falling",
    "person_on_ground",
    "close_hand_to_person",
    "push_like_motion",
    "kick_like_motion",
]

# COCO-подобный порядок (0..16)
KP_NOSE = 0
KP_L_SH = 5
KP_R_SH = 6
KP_L_EL = 7
KP_R_EL = 8
KP_L_WR = 9
KP_R_WR = 10
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KN = 13
KP_R_KN = 14
KP_L_AN = 15
KP_R_AN = 16


@dataclass
class PoseSignal:
    signal_type: str
    severity: float
    track_id: int
    timestamp_sec: float
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "severity": self.severity,
            "track_id": self.track_id,
            "timestamp_sec": self.timestamp_sec,
            "description": self.description,
            "evidence": dict(self.evidence),
        }


def _pose_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    c = dict(cfg or {})
    return dict(c.get("pose") or {})


def _kp(
    keypoints: Sequence[Sequence[float]],
    idx: int,
    min_conf: float,
) -> tuple[float, float] | None:
    if idx < 0 or idx >= len(keypoints):
        return None
    row = keypoints[idx]
    if len(row) < 2:
        return None
    conf = float(row[2]) if len(row) > 2 else 1.0
    if conf < min_conf:
        return None
    return float(row[0]), float(row[1])


def _torso_height(keypoints: Sequence[Sequence[float]], min_conf: float) -> float | None:
    ls = _kp(keypoints, KP_L_SH, min_conf)
    rs = _kp(keypoints, KP_R_SH, min_conf)
    lh = _kp(keypoints, KP_L_HIP, min_conf)
    rh = _kp(keypoints, KP_R_HIP, min_conf)
    pts = [p for p in (ls, rs, lh, rh) if p is not None]
    if len(pts) < 2:
        return None
    ys = [p[1] for p in pts]
    return max(ys) - min(ys)


def detect_pose_signals(
    track_id: int,
    keypoints: Sequence[Sequence[float]],
    timestamp_sec: float,
    analytics_cfg: dict[str, Any] | None = None,
    *,
    prev_wrist: tuple[float, float] | None = None,
    other_person_wrists: Sequence[tuple[float, float]] | None = None,
) -> list[PoseSignal]:
    """Детекция по одному набору ключевых точек + опционально предыдущее положение кисти для скорости."""
    pc = _pose_cfg(analytics_cfg)
    min_conf = float(pc.get("keypoint_confidence", 0.35))
    fast_thr = float(pc.get("fast_arm_px_per_sec", 250))
    margin = float(pc.get("raised_hand_margin_px", 30))
    out: list[PoseSignal] = []

    nose = _kp(keypoints, KP_NOSE, min_conf)
    lw = _kp(keypoints, KP_L_WR, min_conf)
    rw = _kp(keypoints, KP_R_WR, min_conf)
    ls = _kp(keypoints, KP_L_SH, min_conf)
    rs = _kp(keypoints, KP_R_SH, min_conf)
    le = _kp(keypoints, KP_L_EL, min_conf)
    re = _kp(keypoints, KP_R_EL, min_conf)
    lh = _kp(keypoints, KP_L_HIP, min_conf)
    rh = _kp(keypoints, KP_R_HIP, min_conf)
    lk = _kp(keypoints, KP_L_KN, min_conf)
    rk = _kp(keypoints, KP_R_KN, min_conf)
    la = _kp(keypoints, KP_L_AN, min_conf)
    ra = _kp(keypoints, KP_R_AN, min_conf)

    torso = _torso_height(keypoints, min_conf)

    # raised_hand
    for side, wrist, shoulder in (("left", lw, ls), ("right", rw, rs)):
        if wrist and shoulder and nose:
            if wrist[1] + margin <= min(shoulder[1], nose[1]):
                out.append(
                    PoseSignal(
                        signal_type="raised_hand",
                        severity=0.6,
                        track_id=track_id,
                        timestamp_sec=float(timestamp_sec),
                        description="Risk signal detected: elevated wrist relative to shoulder. Requires review.",
                        evidence={"side": side},
                    )
                )
                break

    # fast_arm_motion
    if prev_wrist and lw:
        sp = math.hypot(lw[0] - prev_wrist[0], lw[1] - prev_wrist[1]) * 30.0  # ~1 кадр @30fps
        if sp > fast_thr:
            out.append(
                PoseSignal(
                    signal_type="fast_arm_motion",
                    severity=min(1.0, sp / (fast_thr * 1.5)),
                    track_id=track_id,
                    timestamp_sec=float(timestamp_sec),
                    description="Risk signal detected: fast arm displacement. Requires review.",
                    evidence={"speed_px_per_sec_est": sp},
                )
            )

    # defensive_pose — обе кисти ближе к корпусу и выше бедра
    if lw and rw and ls and rs and lh and rh:
        mid_sh_y = (ls[1] + rs[1]) * 0.5
        if lw[1] < mid_sh_y + 40 and rw[1] < mid_sh_y + 40:
            out.append(
                PoseSignal(
                    signal_type="defensive_pose",
                    severity=0.55,
                    track_id=track_id,
                    timestamp_sec=float(timestamp_sec),
                    description="Risk signal detected: compact upper-body pose. Requires review.",
                    evidence={},
                )
            )

    # person_falling / person_on_ground
    if nose and lh and rh:
        hip_y = (lh[1] + rh[1]) * 0.5
        if nose[1] > hip_y + (torso or 40) * 0.9:
            out.append(
                PoseSignal(
                    signal_type="person_falling",
                    severity=0.72,
                    track_id=track_id,
                    timestamp_sec=float(timestamp_sec),
                    description="Risk signal detected: head below hip line. Requires review.",
                    evidence={"nose_y": nose[1], "hip_y": hip_y},
                )
            )
    ankles = [p for p in (la, ra) if p]
    shoulders = [p for p in (ls, rs) if p]
    if ankles and shoulders:
        ankle_y = sum(p[1] for p in ankles) / len(ankles)
        shoulder_y = sum(p[1] for p in shoulders) / len(shoulders)
        if ankle_y + 20 < shoulder_y and torso and (shoulder_y - ankle_y) < torso * 0.55:
            out.append(
                PoseSignal(
                    signal_type="person_on_ground",
                    severity=0.78,
                    track_id=track_id,
                    timestamp_sec=float(timestamp_sec),
                    description="Risk signal detected: body low to ground. Requires review.",
                    evidence={},
                )
            )

    # close_hand_to_person
    if other_person_wrists and (lw or rw):
        for wx, wy in other_person_wrists:
            for wrist in (w for w in (lw, rw) if w):
                d = math.hypot(wrist[0] - wx, wrist[1] - wy)
                if d < 45.0:
                    out.append(
                        PoseSignal(
                            signal_type="close_hand_to_person",
                            severity=0.68,
                            track_id=track_id,
                            timestamp_sec=float(timestamp_sec),
                            description="Risk signal detected: hand near another person. Requires review.",
                            evidence={"distance_px": d},
                        )
                    )
                    break

    # push_like_motion — вытянутая рука вперёд, локоть почти выпрямлен
    if lw and le and ls:
        v1 = (lw[0] - le[0], lw[1] - le[1])
        v2 = (le[0] - ls[0], le[1] - ls[1])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 > 1e-3 and n2 > 1e-3:
            cosang = abs((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2))
            if cosang > 0.92 and n1 > (torso or 30) * 0.35:
                out.append(
                    PoseSignal(
                        signal_type="push_like_motion",
                        severity=0.62,
                        track_id=track_id,
                        timestamp_sec=float(timestamp_sec),
                        description="Risk signal detected: extended arm alignment. Requires review.",
                        evidence={"side": "left", "cos": cosang},
                    )
                )

    # kick_like_motion — лодыжка выше колена по y (упрощённо «высокая нога»)
    if la and lk and lh:
        if la[1] < lk[1] - 15:
            out.append(
                PoseSignal(
                    signal_type="kick_like_motion",
                    severity=0.58,
                    track_id=track_id,
                    timestamp_sec=float(timestamp_sec),
                    description="Risk signal detected: elevated ankle relative to knee. Requires review.",
                    evidence={"side": "left"},
                )
            )

    return out
