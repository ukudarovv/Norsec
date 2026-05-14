"""
Полный проход по видео: social → pose → actions → audio → fusion (этап 7).
Итог — bullying risk candidate для оператора, не автоматический вердикт.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from audio.video_audio_analyzer import analyze_video_audio
from actions.video_action_analyzer import analyze_video_actions
from fusion.fusion_engine import FusionEngine
from fusion.incident_store import IncidentStore
from fusion.risk_levels import risk_level_from_score, risk_level_label
from inference.video_pose_analyzer import analyze_video_pose
from inference.video_social_analyzer import analyze_video_social

logger = logging.getLogger(__name__)


def _max_sev(signals: list[dict[str, Any]]) -> float:
    m = 0.0
    for s in signals:
        try:
            m = max(m, float(s.get("severity", 0.0)))
        except (TypeError, ValueError):
            continue
    return m


def _video_duration_sec(video_path: str) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 1e-6:
            fps = 30.0
        return float(n / fps) if n > 0 else 0.0
    finally:
        cap.release()


def analyze_video_fusion(
    video_path: str,
    camera_id: str = "demo_camera",
    *,
    incident_store_path: str | None = None,
    save_incident: bool = True,
    fusion_engine: FusionEngine | None = None,
    context: dict[str, Any] | None = None,
    sample_every_sec: float = 0.2,
    max_frames: int = 300,
    confidence_threshold: float = 0.35,
    detector_model: str = "yolov8n.pt",
    pose_model: str = "yolov8n-pose.pt",
    action_backend: str = "heuristic",
    clip_length: int = 16,
    device: str | None = None,
    asr_model_size: str = "small",
    asr_device: str = "cpu",
    language: str | None = "ru",
) -> tuple[dict[str, Any], np.ndarray | None]:
    vp = Path(video_path)
    duration = _video_duration_sec(str(vp)) if vp.is_file() else 0.0

    empty_summary = {
        "risk_score": 0.0,
        "risk_level": "green",
        "risk_level_meaning": risk_level_label("green"),
        "incident_created": False,
    }

    if not vp.is_file():
        return (
            {
                "error": f"файл не найден: {video_path}",
                "video_path": str(video_path),
                "summary": empty_summary,
                "incident": None,
                "incident_id": None,
            },
            None,
        )

    social_payload, social_preview = analyze_video_social(
        str(vp),
        sample_every_sec=sample_every_sec,
        max_frames=max_frames,
        confidence_threshold=confidence_threshold,
        model_name=detector_model,
        device=device,
    )
    pose_payload, pose_preview = analyze_video_pose(
        str(vp),
        sample_every_sec=sample_every_sec,
        max_frames=max_frames,
        confidence_threshold=confidence_threshold,
        detector_model=detector_model,
        pose_model=pose_model,
        device=device,
    )
    action_payload, action_preview = analyze_video_actions(
        str(vp),
        sample_every_sec=sample_every_sec,
        max_frames=max_frames,
        confidence_threshold=confidence_threshold,
        detector_model=detector_model,
        pose_model=pose_model,
        action_model_name=action_backend,
        clip_length=clip_length,
        device=device,
    )
    audio_payload = analyze_video_audio(
        str(vp),
        asr_model_size=asr_model_size,
        asr_device=asr_device,
        language=language,
    )

    social_signals = list(social_payload.get("signals") or [])
    pose_signals = list(pose_payload.get("pose_signals") or [])
    action_signals = list(action_payload.get("actions") or [])
    audio_signals = list(audio_payload.get("audio_signals") or [])

    engine = fusion_engine or FusionEngine()
    ctx = dict(context or {})
    if "severity" not in ctx:
        ctx["severity"] = 0.0

    min_candidate = 0.50
    try:
        from configs.load import get_phase1_config

        min_candidate = float(get_phase1_config().get("fusion", {}).get("offline_candidate_min_score", 0.5))
    except Exception:
        pass

    candidate = engine.fuse_window(
        camera_id=str(camera_id),
        start_sec=0.0,
        end_sec=max(duration, 1.0),
        social_signals=social_signals,
        pose_signals=pose_signals,
        action_signals=action_signals,
        audio_signals=audio_signals,
        context=ctx,
        min_candidate_score=min_candidate,
    )

    preview: np.ndarray | None = social_preview
    if preview is None:
        preview = pose_preview
    if preview is None:
        preview = action_preview

    if candidate is None:
        # агрегированный score для UI даже без кандидата
        base = (
            _max_sev(social_signals) * engine.social_weight
            + _max_sev(pose_signals) * engine.pose_weight
            + _max_sev(action_signals) * engine.action_weight
            + _max_sev(audio_signals) * engine.audio_weight
            + float(ctx.get("severity", 0.0)) * engine.context_weight
        )
        base = min(float(base), 1.0)
        lvl = risk_level_from_score(base)
        out: dict[str, Any] = {
            "video_path": str(vp.resolve()),
            "camera_id": str(camera_id),
            "summary": {
                "risk_score": round(base, 4),
                "risk_level": lvl,
                "risk_level_meaning": risk_level_label(lvl),
                "incident_created": False,
            },
            "incident": None,
            "incident_id": None,
            "preview_note": "Bullying risk candidate threshold not reached — monitor or re-run with clearer media.",
            "sub_analyses": {
                "social_error": social_payload.get("error"),
                "pose_error": pose_payload.get("error"),
                "action_error": action_payload.get("error"),
                "audio_error": audio_payload.get("error"),
            },
        }
        return out, preview

    incident_id: str | None = None
    if save_incident:
        if os.environ.get("DATABASE_URL"):
            try:
                from api.services.incident_write_bridge import persist_incident_candidate

                incident_id = persist_incident_candidate(candidate, str(camera_id))
            except Exception:
                logger.exception("DB incident persist failed; falling back to JSON store")
                store = IncidentStore(incident_store_path or "data/incidents.json")
                incident_id = store.save(candidate)
        else:
            store = IncidentStore(incident_store_path or "data/incidents.json")
            incident_id = store.save(candidate)

    inc_dict = candidate.to_dict()
    inc_dict["incident_id"] = incident_id or inc_dict.get("incident_id")

    result: dict[str, Any] = {
        "video_path": str(vp.resolve()),
        "camera_id": str(camera_id),
        "summary": {
            "risk_score": round(float(candidate.risk_score), 4),
            "risk_level": candidate.risk_level,
            "risk_level_meaning": risk_level_label(candidate.risk_level),
            "incident_created": True,
        },
        "incident": {
            "incident_id": inc_dict.get("incident_id"),
            "camera_id": candidate.camera_id,
            "start_sec": inc_dict["start_sec"],
            "end_sec": inc_dict["end_sec"],
            "risk_score": inc_dict["risk_score"],
            "risk_level": inc_dict["risk_level"],
            "signal_types": inc_dict["signal_types"],
            "involved_track_ids": inc_dict["involved_track_ids"],
            "explanation": inc_dict["explanation"],
            "evidence": inc_dict["evidence"],
        },
        "incident_id": inc_dict.get("incident_id"),
        "sub_analyses": {
            "social_error": social_payload.get("error"),
            "pose_error": pose_payload.get("error"),
            "action_error": action_payload.get("error"),
            "audio_error": audio_payload.get("error"),
        },
    }
    return result, preview
