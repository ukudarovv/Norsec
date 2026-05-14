from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.core.permissions import (
    require_admin,
    require_operator_or_admin,
    require_viewer_plus,
    require_viewer_plus_token,
)
from api.db.models import User
from api.db.session import get_db
from api.schemas.camera import CameraCreate, CameraOut, CameraPatch
from api.services import analytics_service, audit_service, camera_live_service, camera_service, incident_service
from camera.camera_manager import get_camera_manager

router = APIRouter()


def _camera_out(cam, user: User) -> CameraOut:
    data = CameraOut.model_validate(cam)
    if user.role == "viewer":
        return data.model_copy(update={"rtsp_url": None})
    return data


@router.get("", response_model=list[CameraOut])
def list_cameras(user: User = Depends(require_viewer_plus), db: Session = Depends(get_db)):
    return [_camera_out(c, user) for c in camera_service.list_cameras(db)]


@router.get("/running")
def list_running_cameras(_: User = Depends(require_viewer_plus)) -> dict[str, list[str]]:
    return {"cameras": get_camera_manager().list_running()}


@router.post("", response_model=CameraOut, status_code=status.HTTP_201_CREATED)
def create_camera(
    body: CameraCreate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    c = camera_service.create_camera(db, body)
    audit_service.log_action(db, user_id=admin.id, action="create_camera", entity_type="camera", entity_id=str(c.id))
    return _camera_out(c, admin)


@router.post("/{camera_id}/test-connection")
def test_camera_connection(
    camera_id: uuid.UUID,
    user: User = Depends(require_operator_or_admin),
    db: Session = Depends(get_db),
) -> dict[str, bool | str]:
    cam = camera_service.get_camera(db, camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not (cam.rtsp_url or "").strip():
        raise HTTPException(status_code=400, detail="Camera has no rtsp_url")
    ok, code = camera_live_service.test_rtsp_connection(str(cam.rtsp_url))
    new_status = "online" if ok else "error"
    camera_service.patch_camera(db, cam.id, CameraPatch(status=new_status))
    audit_service.log_action(
        db,
        user_id=user.id,
        action="test_camera_connection",
        entity_type="camera",
        entity_id=str(cam.id),
        meta={"ok": ok, "detail": code},
    )
    return {"ok": ok, "detail": code}


@router.post("/{camera_id}/start")
def start_camera_analysis(
    camera_id: uuid.UUID,
    user: User = Depends(require_operator_or_admin),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    cam = camera_service.get_camera(db, camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not (cam.rtsp_url or "").strip():
        raise HTTPException(status_code=400, detail="Camera has no rtsp_url")
    cam_id_str = str(cam.id)
    try:
        get_camera_manager().start_camera(db, cam)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    audit_service.log_action(
        db, user_id=user.id, action="start_camera_analysis", entity_type="camera", entity_id=cam_id_str
    )
    return {"running": True, "camera_id": cam_id_str}


@router.post("/{camera_id}/stop")
def stop_camera_analysis(
    camera_id: uuid.UUID,
    user: User = Depends(require_operator_or_admin),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    cam = camera_service.get_camera(db, camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam_id_str = str(cam.id)
    stopped = get_camera_manager().stop_camera(camera_id)
    audit_service.log_action(
        db,
        user_id=user.id,
        action="stop_camera_analysis",
        entity_type="camera",
        entity_id=cam_id_str,
        meta={"stopped": stopped},
    )
    return {"stopped": stopped, "camera_id": cam_id_str}


@router.post("/{camera_id}/restart")
def restart_camera_analysis(
    camera_id: uuid.UUID,
    user: User = Depends(require_operator_or_admin),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    cam = camera_service.get_camera(db, camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not (cam.rtsp_url or "").strip():
        raise HTTPException(status_code=400, detail="Camera has no rtsp_url")
    cam_id_str = str(cam.id)
    try:
        get_camera_manager().restart_camera(db, cam)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    audit_service.log_action(
        db, user_id=user.id, action="restart_camera_analysis", entity_type="camera", entity_id=cam_id_str
    )
    return {"running": True, "camera_id": cam_id_str}


@router.get("/{camera_id}/mjpeg")
async def camera_mjpeg_stream(
    camera_id: uuid.UUID,
    _: User = Depends(require_viewer_plus_token),
) -> StreamingResponse:
    mgr = get_camera_manager()
    boundary = b"frame"

    async def gen():
        while True:
            _, _, jpeg = mgr.hub.sink(str(camera_id)).snapshot()
            if jpeg:
                yield b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.08)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


@router.get("/{camera_id}/analytics/live")
def camera_analytics_live(
    camera_id: uuid.UUID,
    _: User = Depends(require_viewer_plus),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    cam = camera_service.get_camera(db, camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return analytics_service.build_live_camera_analytics(str(cam.id))


@router.get("/{camera_id}/status")
def camera_status(
    camera_id: uuid.UUID,
    _: User = Depends(require_viewer_plus),
    db: Session = Depends(get_db),
):
    return incident_service.camera_status(db, camera_id)


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(
    camera_id: uuid.UUID,
    user: User = Depends(require_viewer_plus),
    db: Session = Depends(get_db),
):
    c = camera_service.get_camera(db, camera_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return _camera_out(c, user)


@router.patch("/{camera_id}", response_model=CameraOut)
def patch_camera(
    camera_id: uuid.UUID,
    body: CameraPatch,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    c = camera_service.patch_camera(db, camera_id, body)
    if c is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    audit_service.log_action(
        db, user_id=admin.id, action="update_camera", entity_type="camera", entity_id=str(c.id), meta=body.model_dump()
    )
    return _camera_out(c, admin)


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    camera_id: uuid.UUID,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    get_camera_manager().stop_camera(camera_id)
    ok = camera_service.delete_camera(db, camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Camera not found")
    audit_service.log_action(
        db, user_id=admin.id, action="delete_camera", entity_type="camera", entity_id=str(camera_id)
    )

