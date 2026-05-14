from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.core.permissions import require_admin
from api.db.models import User
from api.db.session import get_db
from api.schemas.user import UserCreate, UserOut, UserPatch
from api.services import audit_service, user_service

router = APIRouter()


@router.get("", response_model=list[UserOut])
def list_users(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return [UserOut.model_validate(u) for u in user_service.list_users(db)]


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    body: UserCreate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if user_service.get_by_email(db, str(body.email)):
        raise HTTPException(status_code=400, detail="Email already registered")
    u = user_service.create_user(db, body)
    audit_service.log_action(db, user_id=admin.id, action="create_user", entity_type="user", entity_id=str(u.id))
    return UserOut.model_validate(u)


@router.patch("/{user_id}", response_model=UserOut)
def patch_user(
    user_id: uuid.UUID,
    body: UserPatch,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    u = user_service.patch_user(db, user_id, body)
    if u is None:
        raise HTTPException(status_code=404, detail="User not found")
    audit_service.log_action(db, user_id=admin.id, action="update_user", entity_type="user", entity_id=str(u.id))
    return UserOut.model_validate(u)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: uuid.UUID,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete self")
    ok = user_service.delete_user(db, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    audit_service.log_action(db, user_id=admin.id, action="delete_user", entity_type="user", entity_id=str(user_id))
