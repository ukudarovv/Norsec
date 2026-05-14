from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.core.permissions import get_current_user, get_current_user_optional
from api.db.models import User
from api.db.session import get_db
from api.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from api.schemas.user import UserOut
from api.services import auth_service

router = APIRouter()


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    try:
        u = auth_service.register(db, body)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    return UserOut.model_validate(u)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    out = auth_service.login(db, body)
    if out is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token, user = out
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=auth_service.user_to_token_payload(user),
    )


@router.get("/me", response_model=dict)
def me(user: User = Depends(get_current_user)):
    return {"id": str(user.id), "email": user.email, "role": user.role, "full_name": user.full_name}


@router.post("/logout")
def logout(
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    if user:
        from api.services import audit_service

        audit_service.log_action(db, user_id=user.id, action="logout", entity_type="user", entity_id=str(user.id))
    return {"ok": True}
