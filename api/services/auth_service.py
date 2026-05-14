"""Регистрация и вход."""

from __future__ import annotations

import os

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.core.security import create_access_token, hash_password, verify_password
from api.db.models import User
from api.schemas.auth import LoginRequest, RegisterRequest
from api.schemas.user import UserCreate
from api.services import audit_service, user_service


def _count_users(db: Session) -> int:
    return int(db.scalar(select(func.count()).select_from(User)) or 0)


def register(db: Session, body: RegisterRequest) -> User:
    n = _count_users(db)
    if n == 0:
        u = User(
            email=body.email.lower().strip(),
            full_name=body.full_name,
            hashed_password=hash_password(body.password),
            role="admin",
            is_active=True,
        )
        db.add(u)
        db.commit()
        db.refresh(u)
        audit_service.log_action(db, user_id=u.id, action="register_first_admin", entity_type="user", entity_id=str(u.id))
        return u

    allow_open = os.environ.get("ALLOW_OPEN_REGISTRATION", "").lower() in ("1", "true", "yes")
    if allow_open:
        role = (body.role or "viewer").strip()
        if role not in ("viewer", "operator", "reviewer", "admin"):
            role = "viewer"
        u = User(
            email=body.email.lower().strip(),
            full_name=body.full_name,
            hashed_password=hash_password(body.password),
            role=role,
            is_active=True,
        )
        db.add(u)
        db.commit()
        db.refresh(u)
        audit_service.log_action(db, user_id=None, action="register_open", entity_type="user", entity_id=str(u.id))
        return u

    raise PermissionError(
        "Registration closed: create the first admin via POST /api/auth/register when DB is empty, "
        "or set ALLOW_OPEN_REGISTRATION=1 for dev, or ask an admin to POST /api/users."
    )


def login(db: Session, body: LoginRequest) -> tuple[str, User] | None:
    u = user_service.get_by_email(db, body.email)
    if u is None or not u.is_active:
        audit_service.log_action(
            db,
            user_id=None,
            action="login_failed",
            entity_type="user",
            entity_id=None,
            meta={"email": body.email},
        )
        return None
    if not verify_password(body.password, u.hashed_password):
        audit_service.log_action(
            db,
            user_id=None,
            action="login_failed",
            entity_type="user",
            entity_id=str(u.id),
            meta={"reason": "bad_password"},
        )
        return None
    token = create_access_token(subject=str(u.id), extra_claims={"role": u.role})
    audit_service.log_action(db, user_id=u.id, action="login", entity_type="user", entity_id=str(u.id))
    return token, u


def user_to_token_payload(u: User) -> dict:
    return {"id": str(u.id), "email": u.email, "role": u.role}
