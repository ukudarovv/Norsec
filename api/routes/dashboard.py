from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.core.permissions import require_viewer_plus
from api.db.models import User
from api.db.session import get_db
from api.services import incident_service

router = APIRouter()


@router.get("/stats")
def dashboard_stats(_: User = Depends(require_viewer_plus), db: Session = Depends(get_db)) -> dict[str, object]:
    return incident_service.dashboard_stats(db)
