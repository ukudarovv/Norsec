"""Пользователи (CRUD)."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.core.permissions import ROLES
from api.core.security import hash_password
from api.db.models import User
from api.schemas.user import UserCreate, UserPatch


def list_users(db: Session) -> list[User]:
    return list(db.scalars(select(User).order_by(User.created_at.desc())).all())


def get_user(db: Session, user_id: uuid.UUID) -> User | None:
    return db.get(User, user_id)


def get_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.lower().strip()))


def create_user(db: Session, data: UserCreate) -> User:
    role = (data.role or "viewer").strip()
    if role not in ROLES:
        role = "viewer"
    u = User(
        email=data.email.lower().strip(),
        full_name=data.full_name,
        hashed_password=hash_password(data.password),
        role=role,
        is_active=True,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def patch_user(db: Session, user_id: uuid.UUID, data: UserPatch) -> User | None:
    u = db.get(User, user_id)
    if u is None:
        return None
    if data.full_name is not None:
        u.full_name = data.full_name
    if data.role is not None:
        nr = data.role.strip()
        if nr not in ROLES:
            raise ValueError("invalid role")
        u.role = nr
    if data.is_active is not None:
        u.is_active = data.is_active
    if data.password is not None:
        u.hashed_password = hash_password(data.password)
    db.commit()
    db.refresh(u)
    return u


def delete_user(db: Session, user_id: uuid.UUID) -> bool:
    u = db.get(User, user_id)
    if u is None:
        return False
    db.delete(u)
    db.commit()
    return True
