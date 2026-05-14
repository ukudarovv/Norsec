from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.core.permissions import require_viewer_plus
from api.db.models import User
from api.db.session import get_db
from api.schemas.incident import IncidentResponse
from api.services import incident_service, review_service

router = APIRouter()


@router.get("/queue", response_model=list[IncidentResponse])
def review_queue(
    limit: int = Query(100, ge=1, le=500),
    _: User = Depends(require_viewer_plus),
    db: Session = Depends(get_db),
):
    rows = review_service.list_review_queue(db, limit=limit)
    ids = [r.id for r in rows]
    emails = review_service.latest_reviewer_emails(db, ids)
    return [incident_service.incident_to_response(r, last_reviewer_email=emails.get(r.id)) for r in rows]
