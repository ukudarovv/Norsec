"""
Canonical record types for verbal / episode-level Hub datasets.
See docs/DATASET_SCHEMA_AND_PRIVACY_KZ_RU.md.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from hf_ml_verbal.label_config import LABEL_NAMES


class UtteranceRecord(BaseModel):
    example_id: str
    text: str
    lang_primary: str | None = None
    lang_tags: list[str] = Field(default_factory=list)
    source_id: str | None = None
    license_spdx: str | None = None
    split: str | None = Field(
        default=None, description="train|validation|test или кастомный fold"
    )
    scene_coarse: str | None = Field(
        default=None,
        description="Грубый контекст без ПДн (например corridor, classroom)",
    )
    chunk_start_ms: int | None = None
    chunk_end_ms: int | None = None
    # Multi-label verbal risk (0/1 per label name)
    labels: dict[str, int] = Field(
        default_factory=dict,
        description="Ключи должны входить в config/verbal_labels.yaml",
    )
    annotator_notes: str | None = Field(
        default=None, description="Только для gated; не публиковать с персоналиями"
    )
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("labels")
    @classmethod
    def _labels_keys(cls, v: dict[str, int]) -> dict[str, int]:
        for k in v:
            if k not in LABEL_NAMES:
                raise ValueError(f"Unknown label key: {k}")
        return v


class EpisodeRecord(BaseModel):
    """Агрегат по эпизоду; рекомендуется только gated."""

    episode_id: str
    utterance_example_ids: list[str]
    persistence_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Повторяемость давления во времени"
    )
    targeting_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Прицельность к одному субъекту"
    )
    imbalance_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Дисбаланс сил/ролей"
    )
    bullying_candidate: bool | None = Field(
        default=None,
        description="Итоговая эпизодная метка после политики; gated only",
    )
    meta: dict[str, Any] = Field(default_factory=dict)
