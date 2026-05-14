"""Схемы камеры."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class CameraCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    location: str | None = None
    rtsp_url: str | None = None
    status: str = "unknown"
    external_key: str | None = Field(None, max_length=255, description="Стабильный ключ для fusion / RTSP")
    is_active: bool = True


class CameraPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    location: str | None = None
    rtsp_url: str | None = None
    status: str | None = None
    is_active: bool | None = None
    external_key: str | None = None


class CameraOut(BaseModel):
    id: uuid.UUID
    name: str
    location: str | None
    rtsp_url: str | None = None
    status: str
    is_active: bool
    external_key: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
