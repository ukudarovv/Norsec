"""Review-схемы."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

REVIEW_STATUSES: list[str] = [
    "new",
    "needs_review",
    "confirmed",
    "false_positive",
    "training_candidate",
    "archived",
]

SUGGESTED_REVIEW_TAGS: list[str] = [
    "rough_play",
    "sports",
    "real_conflict",
    "audio_unclear",
    "bad_camera_angle",
    "false_positive",
    "use_for_training",
]


class ReviewRequest(BaseModel):
    status: str = Field(..., description="One of REVIEW_STATUSES")
    comment: str | None = None
    tags: list[str] = Field(default_factory=list)


class ReviewOut(BaseModel):
    id: uuid.UUID
    incident_id: uuid.UUID
    reviewer_id: uuid.UUID
    status: str
    comment: str | None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime

    model_config = {"from_attributes": True}


class OperatorNoteRequest(BaseModel):
    comment: str = Field(..., min_length=1)
