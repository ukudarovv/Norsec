"""Сигналы действий и отрисовка подписей (этап 5)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from inference.person_tracker import TrackedPerson

logger = logging.getLogger(__name__)

_SEVERITY_BY_ACTION: dict[str, float] = {
    "normal": 0.1,
    "push": 0.65,
    "punch": 0.9,
    "kick": 0.95,
    "grab": 0.8,
    "chase": 0.55,
    "escape": 0.45,
    "fall": 0.7,
    "block": 0.35,
    "defend": 0.4,
}


def action_severity(action: str) -> float:
    return float(_SEVERITY_BY_ACTION.get(action, 0.35))


@dataclass
class ActionSignal:
    track_id: int
    action_type: str
    confidence: float
    severity: float
    timestamp_sec: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "action_type": self.action_type,
            "confidence": round(float(self.confidence), 4),
            "severity": round(float(self.severity), 4),
            "timestamp_sec": round(float(self.timestamp_sec), 4),
            "description": self.description,
        }


def build_action_signal(
    track_id: int,
    action: str,
    confidence: float,
    timestamp_sec: float,
) -> ActionSignal:
    sev = action_severity(action)
    return ActionSignal(
        track_id=track_id,
        action_type=action,
        confidence=confidence,
        severity=sev,
        timestamp_sec=timestamp_sec,
        description=f"Action signal: track {track_id} → {action} (pose risk / motion proxy, not bullying verdict)",
    )


def detect_pair_interactions(signals: List[ActionSignal]) -> List[dict[str, Any]]:
    """Если два трека рядом по времени с punch/push/kick и defend/block — ``possible_conflict``."""
    aggressive = {"punch", "kick", "push", "grab"}
    passive = {"defend", "block", "escape"}
    out: list[dict[str, Any]] = []
    by_time: dict[float, list[ActionSignal]] = {}
    for s in signals:
        t = round(s.timestamp_sec, 2)
        by_time.setdefault(t, []).append(s)
    for t, group in by_time.items():
        ag = [s for s in group if s.action_type in aggressive]
        pv = [s for s in group if s.action_type in passive]
        for a in ag:
            for p in pv:
                if a.track_id == p.track_id:
                    continue
                out.append(
                    {
                        "interaction": "possible_conflict",
                        "timestamp_sec": t,
                        "track_ids": sorted([a.track_id, p.track_id]),
                        "description": (
                            f"Tracks {a.track_id} ({a.action_type}) vs {p.track_id} ({p.action_type}) "
                            f"at t≈{t}s — possible conflict (action signal only)"
                        ),
                    }
                )
    return out


def draw_action_labels(
    frame_rgb: np.ndarray,
    tracked_people: List[TrackedPerson],
    labels: Dict[int, Tuple[str, float]],
) -> np.ndarray:
    """Подписи вида ``ID 3 | PUSH 0.82`` у bbox."""
    import cv2

    img = frame_rgb.copy()
    h, w = img.shape[:2]
    for tp in tracked_people:
        if tp.track_id not in labels:
            continue
        act, conf = labels[tp.track_id]
        act_u = str(act).upper()
        x1, y1, x2, y2 = tp.bbox
        x1 = int(max(0, min(x1, w - 1)))
        y1 = int(max(0, min(y1, h - 1)))
        text = f"ID {tp.track_id} | {act_u} {conf:.2f}"
        cv2.putText(
            img,
            text,
            (x1 + 2, max(y1 - 22, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return img
