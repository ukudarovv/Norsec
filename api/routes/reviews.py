from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.core.permissions import require_reviewer_or_admin, require_viewer_plus
from api.db.models import User
from api.db.session import get_db
from api.schemas.incident import IncidentResponse
from api.schemas.review import ReviewOut, ReviewRequest
from api.services import audit_service, incident_service, review_service

router = APIRouter()


@router.post("/{incident_id}/review", response_model=IncidentResponse)
def submit_review(
    incident_id: uuid.UUID,
    body: ReviewRequest,
    user: User = Depends(require_reviewer_or_admin),
    db: Session = Depends(get_db),
):
    try:
        inc = incident_service.add_review(db, incident_id, user, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    audit_service.log_action(
        db,
        user_id=user.id,
        action="incident_reviewed",
        entity_type="incident",
        entity_id=str(inc.id),
        meta={"status": body.status, "tags": body.tags},
    )
    if body.status == "training_candidate":
        audit_service.log_action(
            db,
            user_id=user.id,
            action="incident_sent_to_training",
            entity_type="incident",
            entity_id=str(inc.id),
            meta={"tags": body.tags},
        )
    if body.status == "archived":
        audit_service.log_action(
            db,
            user_id=user.id,
            action="incident_archived",
            entity_type="incident",
            entity_id=str(inc.id),
            meta={},
        )
    m = review_service.latest_reviewer_emails(db, [inc.id])
    return incident_service.incident_to_response(inc, last_reviewer_email=m.get(inc.id))


@router.get("/{incident_id}/reviews", response_model=list[ReviewOut])
def list_reviews(
    incident_id: uuid.UUID,
    _: User = Depends(require_viewer_plus),
    db: Session = Depends(get_db),
):
    inc = incident_service.get_incident(db, incident_id)
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return [ReviewOut.model_validate(r) for r in incident_service.list_reviews(db, incident_id)]
