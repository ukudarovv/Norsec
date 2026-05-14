from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.core.permissions import require_admin, require_operator_or_admin, require_operator_plus, require_viewer_plus
from api.db.models import User
from api.db.session import get_db
from api.schemas.incident import IncidentPatch, IncidentResponse
from api.schemas.review import OperatorNoteRequest
from api.services import analytics_service, audit_service, incident_service, review_service

router = APIRouter()


def _clip_root() -> Path:
    raw = os.environ.get("CLIP_STORAGE_ROOT")
    if raw:
        return Path(raw).resolve()
    return (Path(__file__).resolve().parents[2] / "media" / "clips").resolve()


def _safe_file_under_root(candidate: str, root: Path) -> Path | None:
    try:
        p = Path(candidate).expanduser().resolve()
    except OSError:
        return None
    try:
        p.relative_to(root)
    except ValueError:
        return None
    return p if p.is_file() else None


@router.get("/{incident_id}/analytics")
def get_incident_analytics(
    incident_id: uuid.UUID,
    _: User = Depends(require_viewer_plus),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    inc = incident_service.get_incident(db, incident_id)
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return analytics_service.build_incident_analytics(inc)


@router.get("/{incident_id}/detail")
def get_incident_detail(
    request: Request,
    incident_id: uuid.UUID,
    user: User = Depends(require_viewer_plus),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    pfx = str(request.base_url).rstrip("/")
    body = incident_service.build_incident_detail(db, incident_id, url_prefix=pfx)
    if body is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    audit_service.log_action(
        db,
        user_id=user.id,
        action="incident_viewed",
        entity_type="incident",
        entity_id=str(incident_id),
        meta={},
    )
    return body


@router.post("/{incident_id}/notes", response_model=IncidentResponse)
def post_operator_note(
    incident_id: uuid.UUID,
    body: OperatorNoteRequest,
    user: User = Depends(require_operator_or_admin),
    db: Session = Depends(get_db),
):
    inc = incident_service.append_operator_note(db, incident_id, user, body.comment)
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    audit_service.log_action(
        db,
        user_id=user.id,
        action="operator_note_added",
        entity_type="incident",
        entity_id=str(incident_id),
        meta={"length": len(body.comment)},
    )
    m = review_service.latest_reviewer_emails(db, [inc.id])
    return incident_service.incident_to_response(inc, last_reviewer_email=m.get(inc.id))


@router.get("/{incident_id}/media/clip")
def get_incident_clip(
    incident_id: uuid.UUID,
    user: User = Depends(require_viewer_plus),
    db: Session = Depends(get_db),
):
    inc = incident_service.get_incident(db, incident_id)
    if inc is None or not inc.clip_path:
        raise HTTPException(status_code=404, detail="Clip not available")
    root = _clip_root()
    path = _safe_file_under_root(str(inc.clip_path), root)
    if path is None:
        raise HTTPException(status_code=404, detail="Clip not available")
    audit_service.log_action(
        db,
        user_id=user.id,
        action="clip_exported",
        entity_type="incident",
        entity_id=str(incident_id),
        meta={"path": str(path.name)},
    )
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.get("/{incident_id}/media/snapshot")
def get_incident_snapshot(
    incident_id: uuid.UUID,
    user: User = Depends(require_viewer_plus),
    db: Session = Depends(get_db),
):
    inc = incident_service.get_incident(db, incident_id)
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    ev = dict(inc.evidence or {})
    raw = ev.get("snapshot_path")
    if not raw or not isinstance(raw, str):
        raise HTTPException(status_code=404, detail="Snapshot not available")
    root = _clip_root().parent / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    path = _safe_file_under_root(raw, root.resolve())
    if path is None:
        raise HTTPException(status_code=404, detail="Snapshot not available")
    audit_service.log_action(
        db,
        user_id=user.id,
        action="snapshot_viewed",
        entity_type="incident",
        entity_id=str(incident_id),
        meta={},
    )
    return FileResponse(path, media_type="image/jpeg", filename=path.name)


@router.get("", response_model=list[IncidentResponse])
def list_incidents(
    camera_id: str | None = None,
    risk_level: str | None = None,
    review_status: str | None = None,
    signal_type: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    _: User = Depends(require_viewer_plus),
    db: Session = Depends(get_db),
):
    return incident_service.list_incidents(
        db,
        camera_id=camera_id,
        risk_level=risk_level,
        review_status=review_status,
        signal_type=signal_type,
        created_after=created_after,
        created_before=created_before,
    )


@router.get("/{incident_id}", response_model=IncidentResponse)
def get_incident(
    incident_id: uuid.UUID,
    _: User = Depends(require_viewer_plus),
    db: Session = Depends(get_db),
):
    inc = incident_service.get_incident(db, incident_id)
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    m = review_service.latest_reviewer_emails(db, [inc.id])
    return incident_service.incident_to_response(inc, last_reviewer_email=m.get(inc.id))


@router.patch("/{incident_id}", response_model=IncidentResponse)
def patch_incident(
    incident_id: uuid.UUID,
    body: IncidentPatch,
    user: User = Depends(require_operator_plus),
    db: Session = Depends(get_db),
):
    if user.role == "operator" and body.review_status is not None:
        raise HTTPException(status_code=403, detail="Operators cannot change review_status")
    if body.review_status is not None and user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only admin may PATCH review_status; reviewers use POST /review",
        )
    try:
        inc = incident_service.patch_incident(db, incident_id, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    audit_service.log_action(
        db,
        user_id=user.id,
        action="patch_incident",
        entity_type="incident",
        entity_id=str(inc.id),
        meta=body.model_dump(),
    )
    m = review_service.latest_reviewer_emails(db, [inc.id])
    return incident_service.incident_to_response(inc, last_reviewer_email=m.get(inc.id))


@router.delete("/{incident_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_incident(
    incident_id: uuid.UUID,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    ok = incident_service.delete_incident(db, incident_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Incident not found")
    audit_service.log_action(
        db, user_id=admin.id, action="delete_incident", entity_type="incident", entity_id=str(incident_id)
    )
