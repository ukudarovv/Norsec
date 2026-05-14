"""Камеры."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.db.models import Camera
from api.schemas.camera import CameraCreate, CameraPatch


def list_cameras(db: Session) -> list[Camera]:
    return list(db.scalars(select(Camera).order_by(Camera.created_at.desc())).all())


def get_camera(db: Session, camera_id: uuid.UUID) -> Camera | None:
    return db.get(Camera, camera_id)


def get_by_external_key(db: Session, key: str) -> Camera | None:
    return db.scalar(select(Camera).where(Camera.external_key == key))


def create_camera(db: Session, data: CameraCreate) -> Camera:
    c = Camera(
        name=data.name,
        location=data.location,
        rtsp_url=data.rtsp_url,
        status=data.status,
        is_active=data.is_active,
        external_key=data.external_key,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def patch_camera(db: Session, camera_id: uuid.UUID, data: CameraPatch) -> Camera | None:
    c = db.get(Camera, camera_id)
    if c is None:
        return None
    if data.name is not None:
        c.name = data.name
    if data.location is not None:
        c.location = data.location
    if data.rtsp_url is not None:
        c.rtsp_url = data.rtsp_url
    if data.status is not None:
        c.status = data.status
    if data.is_active is not None:
        c.is_active = data.is_active
    if data.external_key is not None:
        c.external_key = data.external_key
    db.commit()
    db.refresh(c)
    return c


def delete_camera(db: Session, camera_id: uuid.UUID) -> bool:
    c = db.get(Camera, camera_id)
    if c is None:
        return False
    db.delete(c)
    db.commit()
    return True
