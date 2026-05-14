"""Запись в audit_logs."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from api.db.models import AuditLog


def log_action(
    db: Session,
    *,
    user_id: uuid.UUID | None,
    action: str,
    entity_type: str,
    entity_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    row = AuditLog(
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        meta=meta or {},
    )
    db.add(row)
    db.commit()
