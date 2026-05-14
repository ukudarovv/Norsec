"""Схемы инцидента."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IncidentResponse(BaseModel):
    id: str
    camera_id: str
    camera_external_key: str | None = None
    start_sec: float
    end_sec: float
    risk_score: float
    risk_level: str
    signal_types: list[str]
    explanation: list[str]
    review_status: str
    evidence: dict[str, Any]
    involved_track_ids: list[int] | None = None
    clip_path: str | None = None
    created_at: str
    last_reviewer_email: str | None = None


class IncidentPatch(BaseModel):
    clip_path: str | None = None
    review_status: str | None = None
