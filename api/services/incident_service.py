"""Инциденты и отзывы (PostgreSQL / SQLAlchemy)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from api.db.models import Camera, Incident, Review, User
from api.schemas.incident import IncidentPatch, IncidentResponse
from api.schemas.review import REVIEW_STATUSES, ReviewRequest
from api.services import audit_service, camera_service
from api.services import review_service as review_svc
from fusion.incident_candidate import IncidentCandidate


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def incident_to_response(inc: Incident, *, last_reviewer_email: str | None = None) -> IncidentResponse:
    cam = inc.camera
    ext = cam.external_key if cam else None
    return IncidentResponse(
        id=str(inc.id),
        camera_id=str(inc.camera_id),
        camera_external_key=ext,
        start_sec=float(inc.start_sec),
        end_sec=float(inc.end_sec),
        risk_score=float(inc.risk_score),
        risk_level=str(inc.risk_level),
        signal_types=list(inc.signal_types or []),
        explanation=list(inc.explanation or []),
        review_status=str(inc.review_status),
        evidence=dict(inc.evidence or {}),
        involved_track_ids=list(inc.involved_track_ids) if inc.involved_track_ids is not None else None,
        clip_path=inc.clip_path,
        created_at=inc.created_at.isoformat() if inc.created_at else "",
        last_reviewer_email=last_reviewer_email,
    )


def list_incidents(
    db: Session,
    *,
    camera_id: str | None = None,
    risk_level: str | None = None,
    review_status: str | None = None,
    signal_type: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> list[IncidentResponse]:
    q = select(Incident).options(joinedload(Incident.camera)).order_by(Incident.created_at.desc())
    rows = list(db.scalars(q).all())

    cam_uuid: uuid.UUID | None = None
    cam_ext: str | None = None
    if camera_id:
        try:
            cam_uuid = uuid.UUID(camera_id)
        except ValueError:
            cam_ext = camera_id

    after = _parse_dt(created_after)
    before = _parse_dt(created_before)

    filtered: list[Incident] = []
    for inc in rows:
        if cam_uuid is not None and inc.camera_id != cam_uuid:
            continue
        if cam_ext is not None and (inc.camera is None or (inc.camera.external_key or "") != cam_ext):
            continue
        if risk_level and inc.risk_level != risk_level:
            continue
        if review_status and inc.review_status != review_status:
            continue
        if after and inc.created_at and inc.created_at < after:
            continue
        if before and inc.created_at and inc.created_at > before:
            continue
        if signal_type:
            st = list(inc.signal_types or [])
            if signal_type not in st:
                continue
        filtered.append(inc)

    ids = [i.id for i in filtered]
    email_map = review_svc.latest_reviewer_emails(db, ids)
    return [incident_to_response(inc, last_reviewer_email=email_map.get(inc.id)) for inc in filtered]


def get_incident(db: Session, incident_id: uuid.UUID) -> Incident | None:
    return db.scalar(
        select(Incident).options(joinedload(Incident.camera)).where(Incident.id == incident_id)
    )


def patch_incident(db: Session, incident_id: uuid.UUID, data: IncidentPatch) -> Incident | None:
    inc = db.get(Incident, incident_id)
    if inc is None:
        return None
    if data.clip_path is not None:
        inc.clip_path = data.clip_path
    if data.review_status is not None:
        if data.review_status not in REVIEW_STATUSES:
            raise ValueError("invalid review_status")
        inc.review_status = data.review_status
    db.commit()
    db.refresh(inc)
    return inc


def delete_incident(db: Session, incident_id: uuid.UUID) -> bool:
    inc = db.get(Incident, incident_id)
    if inc is None:
        return False
    db.delete(inc)
    db.commit()
    return True


def add_review(db: Session, incident_id: uuid.UUID, reviewer: User, body: ReviewRequest) -> Incident | None:
    return review_svc.apply_review(db, incident_id, reviewer, body)


def list_reviews(db: Session, incident_id: uuid.UUID) -> list[Review]:
    return list(
        db.scalars(select(Review).where(Review.incident_id == incident_id).order_by(Review.created_at.asc())).all()
    )


def append_operator_note(db: Session, incident_id: uuid.UUID, user: User, comment: str) -> Incident | None:
    inc = get_incident(db, incident_id)
    if inc is None:
        return None
    ev = dict(inc.evidence or {})
    notes = list(ev.get("operator_notes") or [])
    notes.append(
        {
            "user_id": str(user.id),
            "email": user.email,
            "comment": comment.strip(),
            "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )
    ev["operator_notes"] = notes
    inc.evidence = ev
    db.commit()
    db.refresh(inc)
    return inc


def build_incident_detail(db: Session, incident_id: uuid.UUID, *, url_prefix: str = "") -> dict[str, Any] | None:
    inc = get_incident(db, incident_id)
    if inc is None:
        return None
    from api.schemas.review import ReviewOut
    from api.services import analytics_service

    ev = dict(inc.evidence or {})
    analytics_block = analytics_service.build_incident_analytics(inc)
    sup = ev.get("suppression") or (analytics_block.get("suppression") if isinstance(analytics_block, dict) else {})
    reasons: list[str] = []
    if isinstance(sup, dict):
        reasons = list(sup.get("reasons") or sup.get("active_rules") or [])

    analytics = {
        "social_signals": ev.get("social_signals") or analytics_block.get("social_signals") or [],
        "pose_signals": ev.get("pose_signals") or analytics_block.get("pose_signals") or [],
        "action_signals": ev.get("action_signals") or [],
        "audio_signals": ev.get("audio_signals") or [],
        "suppression_reasons": reasons,
    }
    iid = str(inc.id)
    pfx = url_prefix.rstrip("/")
    video_clip_url = f"{pfx}/api/incidents/{iid}/media/clip" if inc.clip_path else ""
    snapshot_path = ev.get("snapshot_path")
    snapshot_url = f"{pfx}/api/incidents/{iid}/media/snapshot" if snapshot_path else ""

    rev_rows = list_reviews(db, incident_id)
    reviews_out = [ReviewOut.model_validate(r) for r in rev_rows]

    email_map = review_svc.latest_reviewer_emails(db, [inc.id])
    return {
        "incident": incident_to_response(inc, last_reviewer_email=email_map.get(inc.id)).model_dump(),
        "evidence": ev,
        "reviews": [x.model_dump(mode="json") for x in reviews_out],
        "video_clip_url": video_clip_url,
        "snapshot_url": snapshot_url,
        "analytics": analytics,
    }


def get_or_create_camera_by_key(db: Session, camera_key: str) -> Camera:
    c = camera_service.get_by_external_key(db, camera_key)
    if c:
        return c
    c = Camera(
        name=camera_key,
        external_key=camera_key,
        status="auto_registered",
        is_active=True,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def create_from_candidate(db: Session, candidate: IncidentCandidate, camera_key: str) -> Incident:
    cam = get_or_create_camera_by_key(db, camera_key)
    inc = Incident(
        camera_id=cam.id,
        start_sec=float(candidate.start_sec),
        end_sec=float(candidate.end_sec),
        risk_score=float(candidate.risk_score),
        risk_level=str(candidate.risk_level),
        review_status="new",
        signal_types=list(candidate.signal_types),
        explanation=list(candidate.explanation),
        evidence=dict(candidate.evidence or {}),
        involved_track_ids=list(candidate.involved_track_ids),
        clip_path=None,
    )
    db.add(inc)
    db.commit()
    db.refresh(inc)
    return inc


def camera_status(db: Session, camera_id: uuid.UUID) -> dict[str, Any]:
    cam = db.get(Camera, camera_id)
    if cam is None:
        return {
            "camera_id": str(camera_id),
            "incidents_total": 0,
            "open_for_review": 0,
            "last_incident_at": None,
            "last_risk_level": None,
            "last_review_status": None,
            "live": {"worker_running": False, "overlay_seq": 0},
        }
    q = select(Incident).where(Incident.camera_id == camera_id).order_by(Incident.created_at.desc())
    incs = list(db.scalars(q).all())
    open_statuses = frozenset({"new", "needs_review"})
    open_cnt = sum(1 for i in incs if i.review_status in open_statuses)
    last = incs[0] if incs else None

    live: dict[str, Any] = {"worker_running": False, "overlay_seq": 0}
    try:
        from camera.camera_manager import get_camera_manager

        mgr = get_camera_manager()
        sk = mgr.hub.sink(str(camera_id))
        live = sk.snapshot_metrics()
        live["worker_running"] = mgr.is_running(str(camera_id))
    except Exception:
        live = {"worker_running": False, "overlay_seq": 0}

    return {
        "camera_id": str(camera_id),
        "camera_name": cam.name,
        "db_status": cam.status,
        "incidents_total": len(incs),
        "open_for_review": open_cnt,
        "last_incident_at": last.created_at.isoformat() if last and last.created_at else None,
        "last_risk_level": last.risk_level if last else None,
        "last_review_status": last.review_status if last else None,
        "live": live,
    }


def dashboard_stats(db: Session) -> dict[str, Any]:
    rows = list(db.scalars(select(Incident).options(joinedload(Incident.camera))).all())
    total = len(rows)
    by_status: dict[str, int] = {}
    by_level: dict[str, int] = {}
    by_day: dict[str, int] = {}
    by_camera: dict[str, int] = {}
    risk_sum = 0.0
    fp = 0
    reviewed = 0
    for inc in rows:
        by_status[inc.review_status] = by_status.get(inc.review_status, 0) + 1
        by_level[inc.risk_level] = by_level.get(inc.risk_level, 0) + 1
        risk_sum += float(inc.risk_score)
        cam_label = inc.camera.name if inc.camera else str(inc.camera_id)
        by_camera[cam_label] = by_camera.get(cam_label, 0) + 1
        if inc.created_at:
            day = inc.created_at.date().isoformat()
            by_day[day] = by_day.get(day, 0) + 1
        if inc.review_status == "false_positive":
            fp += 1
        if inc.review_status in ("confirmed", "false_positive", "training_candidate", "archived"):
            reviewed += 1
    avg_risk = (risk_sum / total) if total else 0.0
    fp_rate = (fp / reviewed) if reviewed else 0.0
    cams = list(db.scalars(select(Camera)).all())
    online = sum(1 for c in cams if (c.status or "").lower() in ("online", "auto_registered"))
    return {
        "totals": {
            "risk_candidates": total,
            "needs_review": by_status.get("needs_review", 0) + by_status.get("new", 0),
            "confirmed": by_status.get("confirmed", 0),
            "false_positives": by_status.get("false_positive", 0),
            "average_risk_score": round(avg_risk, 4),
            "false_positive_rate": round(fp_rate, 4),
        },
        "by_status": by_status,
        "by_risk_level": by_level,
        "by_day": dict(sorted(by_day.items())),
        "by_camera": dict(sorted(by_camera.items(), key=lambda x: -x[1])[:20]),
        "camera_health": {"total_cameras": len(cams), "online_or_active": online},
    }
