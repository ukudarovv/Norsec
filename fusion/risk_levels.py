"""Уровни риска по числовому score (этап 7)."""

from __future__ import annotations

RISK_LEVELS: dict[str, str] = {
    "green": "normal",
    "yellow": "low_suspicion",
    "orange": "high_suspicion",
    "red": "requires_human_review",
}


def risk_level_from_score(score: float) -> str:
    s = float(score)
    if s < 0.25:
        return "green"
    if s < 0.50:
        return "yellow"
    if s < 0.75:
        return "orange"
    return "red"


def risk_level_label(level: str) -> str:
    return RISK_LEVELS.get(level, level)
