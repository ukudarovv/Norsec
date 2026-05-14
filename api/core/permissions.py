"""RBAC: роли и зависимости FastAPI (этап 9)."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.orm import Session

from api.core.security import decode_token, parse_uuid_sub
from api.db.models import User
from api.db.session import get_db

ROLES: list[str] = ["admin", "operator", "reviewer", "viewer"]

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user_optional(
    cred: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Session = Depends(get_db),
) -> User | None:
    if cred is None or cred.scheme.lower() != "bearer":
        return None
    try:
        payload = decode_token(cred.credentials)
        uid = parse_uuid_sub(payload)
        if uid is None:
            return None
        user = db.get(User, uid)
        if user is None or not user.is_active:
            return None
        return user
    except JWTError:
        return None


def get_current_user(
    user: Annotated[User | None, Depends(get_current_user_optional)],
) -> User:
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def require_roles(allowed: list[str]):
    """Зависимость: пользователь с ролью из ``allowed``."""

    allowed_set = frozenset(allowed)

    def _dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_set:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return _dep


def require_admin(user: User = Depends(require_roles(["admin"]))) -> User:
    return user


def require_reviewer_or_admin(user: User = Depends(require_roles(["admin", "reviewer"]))) -> User:
    return user


def require_operator_plus(user: User = Depends(require_roles(["admin", "operator", "reviewer"]))) -> User:
    return user


def require_viewer_plus(user: User = Depends(require_roles(["admin", "operator", "reviewer", "viewer"]))) -> User:
    return user


def require_operator_or_admin(user: User = Depends(require_roles(["admin", "operator"]))) -> User:
    return user


def bearer_or_query_token(
    cred: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    token_q: str | None = Query(None, alias="token"),
) -> str | None:
    """JWT из Authorization: Bearer или query ``token`` (MJPEG / WS в браузере)."""
    if cred is not None and cred.scheme.lower() == "bearer":
        return cred.credentials
    return token_q


def get_current_user_from_raw_token(
    raw: Annotated[str | None, Depends(bearer_or_query_token)],
    db: Session = Depends(get_db),
) -> User | None:
    if not raw:
        return None
    try:
        payload = decode_token(raw)
        uid = parse_uuid_sub(payload)
        if uid is None:
            return None
        user = db.get(User, uid)
        if user is None or not user.is_active:
            return None
        return user
    except JWTError:
        return None


def require_viewer_plus_token(
    user: Annotated[User | None, Depends(get_current_user_from_raw_token)],
) -> User:
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    if user.role not in frozenset(ROLES):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    return user
