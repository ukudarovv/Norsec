"""Каталог сигналов и справочники Phase 2."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.core.permissions import require_viewer_plus
from api.db.models import User
from api.services import analytics_service

router = APIRouter()


@router.get("/signals")
def list_signal_definitions(_: User = Depends(require_viewer_plus)) -> dict[str, object]:
    return analytics_service.signal_catalog()
