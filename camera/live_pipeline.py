"""Обработка одного кадра для live: overlay + стабилизированный fusion (Phase 1–2)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Protocol

import cv2
import numpy as np

from fusion.fusion_engine import FusionEngine
from fusion.fusion_stability import StableLiveFusion
from fusion.incident_candidate import IncidentCandidate
from fusion.risk_levels import risk_level_from_score

from camera.overlay_payload import build_overlay_payload

logger = logging.getLogger(__name__)


class LiveFramePipeline(Protocol):
    def process(
        self,
        frame_bgr: np.ndarray,
        timestamp_sec: float,
        camera_id: str,
    ) -> tuple[dict[str, Any], np.ndarray | None, IncidentCandidate | None]:
        """overlay dict, annotated frame (optional), incident candidate (optional)."""
        ...


def _bbox_around_center(cx: float, cy: float, bw: float, bh: float) -> tuple[float, float, float, float]:
    return (cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5)


def _stub_keypoints_raised_hand(cx: float, cy: float, *, fast_wrist: bool, frame_n: int) -> list[list[float]]:
    """COCO-like 17×3 для демо: поднятая левая кисть; при ``fast_wrist`` — смещение для скорости."""
    kps: list[list[float]] = [[0.0, 0.0, 0.05] for _ in range(17)]
    ox = 40.0 if fast_wrist and (frame_n % 2 == 0) else -40.0
    kps[0] = [cx, cy - 90.0, 0.92]  # nose
    kps[5] = [cx - 45.0, cy - 40.0, 0.92]  # L shoulder
    kps[6] = [cx + 45.0, cy - 40.0, 0.92]
    kps[7] = [cx - 55.0, cy + 10.0, 0.9]
    kps[8] = [cx + 55.0, cy + 10.0, 0.9]
    kps[9] = [cx - 50.0 + ox, cy - 130.0, 0.92]  # L wrist (raised)
    kps[10] = [cx + 50.0, cy + 20.0, 0.9]
    kps[11] = [cx - 25.0, cy + 70.0, 0.9]
    kps[12] = [cx + 25.0, cy + 70.0, 0.9]
    kps[13] = [cx - 30.0, cy + 130.0, 0.88]
    kps[14] = [cx + 30.0, cy + 130.0, 0.88]
    kps[15] = [cx - 30.0, cy + 190.0, 0.85]
    kps[16] = [cx + 30.0, cy + 190.0, 0.85]
    return kps


class StubLivePipeline:
    """
    MVP: лёгкий конвейер; периодически сильные сигналы → ``StableLiveFusion`` (persistence + cooldown).
    Phase 2: ``TrackMemory`` + социальные/поза-сигналы + suppression в ``overlay["analytics"]``.
    """

    def __init__(self, fuse_every: int = 8) -> None:
        self._frame_n = 0
        self._fuse_every = max(1, int(fuse_every))
        self._engine = FusionEngine()
        self._gate: StableLiveFusion | None = None
        self._track_mem = None
        self._analytics_cfg: dict[str, Any] = {}
        self._prev_lwrist: tuple[float, float] | None = None
        self._last_fuse_mono = 0.0
        try:
            from configs.load import get_analytics_config, get_phase1_config

            self._analytics_cfg = get_analytics_config()
            from tracking.track_memory import TrackMemory

            self._track_mem = TrackMemory(self._analytics_cfg)
            self._gate = StableLiveFusion(self._engine, get_phase1_config().get("fusion", {}))
        except Exception:
            logger.warning("stable_live_fusion_or_analytics_disabled", exc_info=True)
        self._display_smooth = 0.1
        self._display_level = "green"

    def _build_people(self, w: int, h: int) -> list[dict[str, Any]]:
        p = self._frame_n * 5
        cx2 = w * 0.72
        cx1 = w * 0.22 + min(float(p), w * 0.48)
        cy = h * 0.52
        bw, bh = w * 0.09, h * 0.28
        b1 = _bbox_around_center(cx1, cy, bw, bh)
        b2 = _bbox_around_center(cx2, cy, bw * 0.95, bh)
        pe = w * 0.07
        b3 = (cx2 + pe - bw * 0.45, cy - bh * 0.25, cx2 + pe + bw * 0.45, cy + bh * 0.45)
        b4 = (cx2 - pe - bw * 0.45, cy - bh * 0.25, cx2 - pe + bw * 0.45, cy + bh * 0.45)
        return [
            {"track_id": 1, "bbox": [float(x) for x in b1], "confidence": 0.91},
            {"track_id": 2, "bbox": [float(x) for x in b2], "confidence": 0.9},
            {"track_id": 3, "bbox": [float(x) for x in b3], "confidence": 0.88},
            {"track_id": 4, "bbox": [float(x) for x in b4], "confidence": 0.87},
        ]

    def process(
        self,
        frame_bgr: np.ndarray,
        timestamp_sec: float,
        camera_id: str,
    ) -> tuple[dict[str, Any], np.ndarray | None, IncidentCandidate | None]:
        self._frame_n += 1
        h, w = frame_bgr.shape[:2]
        people = self._build_people(w, h)

        if self._track_mem is not None:
            for p in people:
                bb = p.get("bbox") or []
                if len(bb) == 4:
                    self._track_mem.update(
                        int(p["track_id"]),
                        (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])),
                        float(timestamp_sec),
                    )
            self._track_mem.cleanup_old_tracks(float(timestamp_sec))

        social_raw: list[Any] = []
        pose_raw: list[Any] = []
        pose_flags: dict[int, list[str]] = {}

        if self._track_mem is not None:
            from analytics.pose_signals import detect_pose_signals
            from analytics.social_signals import detect_social_signals

            cx1 = (people[0]["bbox"][0] + people[0]["bbox"][2]) * 0.5
            cy1 = (people[0]["bbox"][1] + people[0]["bbox"][3]) * 0.5
            kps = _stub_keypoints_raised_hand(cx1, cy1, fast_wrist=True, frame_n=self._frame_n)
            pose_raw = detect_pose_signals(
                1,
                kps,
                float(timestamp_sec),
                self._analytics_cfg,
                prev_wrist=self._prev_lwrist,
            )
            self._prev_lwrist = (float(kps[9][0]), float(kps[9][1]))
            for ps in pose_raw:
                pose_flags.setdefault(int(ps.track_id), []).append(str(ps.signal_type))

            social_raw = detect_social_signals(
                self._track_mem,
                people,
                float(timestamp_sec),
                self._analytics_cfg,
                pose_flags_by_track=pose_flags,
            )

        social_fusion = [
            {
                "signal_type": s.signal_type,
                "severity": float(s.severity),
                "timestamp_sec": float(timestamp_sec),
                "track_ids": list(s.track_ids),
            }
            for s in social_raw
        ]
        pose_fusion = [
            {
                "signal_type": p.signal_type,
                "severity": float(p.severity),
                "timestamp_sec": float(timestamp_sec),
                "track_id": int(p.track_id),
            }
            for p in pose_raw
        ]

        is_fuse = self._frame_n % self._fuse_every == 0
        if is_fuse:
            self._last_fuse_mono = time.monotonic()

        if is_fuse and not social_fusion:
            social_fusion = [
                {
                    "signal_type": "crowding",
                    "severity": 0.85,
                    "timestamp_sec": float(timestamp_sec),
                    "track_ids": [1, 2, 3],
                }
            ]
        if is_fuse and not pose_fusion:
            pose_fusion = [
                {
                    "signal_type": "fast_arm_motion",
                    "severity": 0.82,
                    "timestamp_sec": float(timestamp_sec),
                    "track_id": 1,
                }
            ]

        from analytics.suppression import assess_live_suppression

        confs = [float(p.get("confidence") or 0) for p in people]
        signal_age = max(0.0, time.monotonic() - self._last_fuse_mono) if not is_fuse else 5.0
        sup = assess_live_suppression(
            zone_tags=[],
            track_confidences=confs,
            signal_age_sec=signal_age,
            frame_spike=not is_fuse,
            analytics_cfg=self._analytics_cfg,
            crowd_density=min(1.0, len(people) / 8.0),
        )

        candidate: IncidentCandidate | None = None
        signals: list[dict[str, Any]] = []

        if is_fuse:
            action = [
                {
                    "action_type": "punch",
                    "severity": 0.92,
                    "timestamp_sec": float(timestamp_sec),
                    "track_id": 1,
                }
            ]
            if self._gate is not None:
                self._display_smooth, self._display_level, candidate = self._gate.tick(
                    camera_id=str(camera_id),
                    start_sec=max(0.0, float(timestamp_sec) - 2.0),
                    end_sec=float(timestamp_sec) + 1.0,
                    social_signals=social_fusion,
                    pose_signals=pose_fusion,
                    action_signals=action,
                    audio_signals=[],
                    context={"severity": 0.0},
                    now_mono=time.monotonic(),
                )
            else:
                candidate = self._engine.fuse_window(
                    camera_id=str(camera_id),
                    start_sec=max(0.0, float(timestamp_sec) - 2.0),
                    end_sec=float(timestamp_sec) + 1.0,
                    social_signals=social_fusion,
                    pose_signals=pose_fusion,
                    action_signals=action,
                    audio_signals=[],
                    context={"severity": 0.0},
                )
                if candidate is not None:
                    self._display_smooth = float(candidate.risk_score)
                    self._display_level = str(candidate.risk_level)
                else:
                    self._display_smooth = 0.45
                    self._display_level = risk_level_from_score(self._display_smooth)

            signals = [
                {"type": "social", "name": "crowding", "severity": 0.85},
                {"type": "pose", "name": "fast_arm_motion", "severity": 0.82},
                {"type": "action", "name": "punch", "severity": 0.92},
            ]

        risk_score = float(max(0.0, min(1.0, self._display_smooth * sup.risk_multiplier)))
        risk_level = str(risk_level_from_score(risk_score))

        traj_preview: list[dict[str, Any]] = []
        if self._track_mem is not None:
            for tid in self._track_mem.get_active_tracks():
                rec = self._track_mem.get_track(tid)
                if rec is None or not rec.positions:
                    continue
                traj_preview.append(
                    {
                        "track_id": tid,
                        "points": [list(map(float, pt)) for pt in rec.positions[-12:]],
                    }
                )

        analytics_block: dict[str, Any] = {
            "social": [s.to_dict() for s in social_raw],
            "pose": [p.to_dict() for p in pose_raw],
            "trajectory_preview": traj_preview,
            "suppression": sup.to_dict(),
            "risk_modifiers": list(sup.reasons),
        }

        overlay = build_overlay_payload(
            str(camera_id),
            people=people,
            poses=[],
            signals=signals,
            risk_score=risk_score,
            risk_level=risk_level,
            analytics=analytics_block,
        )
        ann = frame_bgr.copy()
        try:
            for p in people:
                bb = p.get("bbox") or []
                if len(bb) == 4:
                    x1i, y1i, x2i, y2i = map(int, bb)
                    cv2.rectangle(ann, (x1i, y1i), (x2i, y2i), (0, 200, 255), 2)
                    cv2.putText(
                        ann,
                        f"id={p.get('track_id')} {risk_level}",
                        (x1i, max(0, y1i - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 200, 255),
                        1,
                        cv2.LINE_AA,
                    )
        except Exception:
            logger.debug("annotate_failed", exc_info=True)

        return overlay, ann, candidate


def build_default_pipeline() -> LiveFramePipeline:
    try:
        every = int(os.environ.get("LIVE_STUB_FUSE_EVERY", "8"))
    except ValueError:
        every = 8
    return StubLivePipeline(fuse_every=max(1, every))
