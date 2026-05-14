"""Сборка ответов Phase 2: аналитика инцидента и live-камеры."""

from __future__ import annotations

from typing import Any

from analytics.pose_signals import POSE_SIGNALS
from analytics.social_signals import SOCIAL_SIGNALS
from analytics.suppression import SuppressionResult, assess_live_suppression
from api.db.models import Incident
from camera.camera_manager import get_camera_manager


def build_incident_analytics(inc: Incident) -> dict[str, Any]:
    ev = dict(inc.evidence or {})
    social = list(ev.get("social_signals") or [])
    pose = list(ev.get("pose_signals") or [])
    traj = ev.get("trajectory") or ev.get("trajectory_summary") or {}
    sup = ev.get("suppression") or ev.get("suppression_result")
    return {
        "incident_id": str(inc.id),
        "camera_id": str(inc.camera_id),
        "social_signals": social,
        "pose_signals": pose,
        "trajectory_summary": traj if isinstance(traj, dict) else {"raw": traj},
        "suppression": sup,
        "evidence": ev,
        "note": "Risk signal detected — requires human review. Not a diagnostic verdict.",
    }


def build_live_camera_analytics(camera_id: str) -> dict[str, Any]:
    mgr = get_camera_manager()
    seq, overlay, _jpeg = mgr.hub.sink(str(camera_id)).snapshot()
    overlay = overlay or {}
    analytics = dict(overlay.get("analytics") or {})
    risk = overlay.get("risk") or {}
    sup_raw = analytics.get("suppression") or {}
    modifiers = analytics.get("risk_modifiers") or sup_raw.get("reasons") or []
    return {
        "camera_id": str(camera_id),
        "overlay_seq": int(seq or 0),
        "risk": risk,
        "active_social_signals": analytics.get("social", []),
        "active_pose_signals": analytics.get("pose", []),
        "trajectory_preview": analytics.get("trajectory_preview", []),
        "risk_modifiers": modifiers,
        "suppression": sup_raw,
        "signal_catalog": {"social": list(SOCIAL_SIGNALS), "pose": list(POSE_SIGNALS)},
    }


def signal_catalog() -> dict[str, Any]:
    return {
        "social_signals": list(SOCIAL_SIGNALS),
        "pose_signals": list(POSE_SIGNALS),
        "suppression_rules": [
            "sports_mode",
            "playground_mode",
            "high_density_crowd",
            "single_frame_spike",
            "low_confidence_tracks",
            "low_light_or_blur",
        ],
    }


def demo_suppression_payload(
    *,
    zone_tags: list[str] | None = None,
    confidences: list[float] | None = None,
    analytics_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Утилита для тестов / отладки."""
    r: SuppressionResult = assess_live_suppression(
        zone_tags=zone_tags,
        track_confidences=confidences,
        signal_age_sec=0.1,
        frame_spike=True,
        analytics_cfg=analytics_cfg,
        crowd_density=0.9,
        low_light_score=None,
    )
    return r.to_dict()
