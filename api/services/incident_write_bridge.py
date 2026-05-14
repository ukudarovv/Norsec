"""Сохранение fusion-кандидата в БД (вызывается из fusion без загрузки FastAPI app)."""

from __future__ import annotations

import logging

from fusion.incident_candidate import IncidentCandidate

logger = logging.getLogger(__name__)


def persist_incident_candidate(candidate: IncidentCandidate, camera_key: str) -> str:
    """Возвращает UUID инцидента строкой."""
    from api.db.session import get_session_factory
    from api.services import incident_service as inc_svc

    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        inc = inc_svc.create_from_candidate(db, candidate, camera_key)
        return str(inc.id)
    except Exception:
        logger.exception("persist_incident_candidate failed")
        raise
    finally:
        db.close()
