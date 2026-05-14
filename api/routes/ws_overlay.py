"""WebSocket: live overlay JSON по камере."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, WebSocket
from jose import JWTError

from api.core.permissions import ROLES
from api.core.security import decode_token, parse_uuid_sub
from api.db.models import User
from api.db.session import get_session_factory
from camera.camera_manager import get_camera_manager

router = APIRouter()


@router.websocket("/ws/cameras/{camera_id}/overlay")
async def camera_overlay_ws(websocket: WebSocket, camera_id: str) -> None:
    await websocket.accept()
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return
    try:
        uuid.UUID(camera_id)
    except ValueError:
        await websocket.close(code=1003)
        return

    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        try:
            payload = decode_token(token)
            uid = parse_uuid_sub(payload)
            if uid is None:
                await websocket.close(code=1008)
                return
            user = db.get(User, uid)
            if user is None or not user.is_active:
                await websocket.close(code=1008)
                return
            if user.role not in frozenset(ROLES):
                await websocket.close(code=1008)
                return
        except JWTError:
            await websocket.close(code=1008)
            return
    finally:
        db.close()

    mgr = get_camera_manager()
    last_seq = -1
    try:
        while True:
            seq, overlay, _ = mgr.hub.sink(camera_id).snapshot()
            if overlay is not None and seq > last_seq:
                last_seq = seq
                await websocket.send_json(overlay)
            await asyncio.sleep(0.04)
    except Exception:
        return
