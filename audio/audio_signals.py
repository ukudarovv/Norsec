"""Структурированные audio risk signals (этап 6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AudioSignal:
    signal_type: str
    severity: float
    start_sec: float
    end_sec: float
    text: Optional[str]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "severity": round(float(self.severity), 4),
            "start_sec": round(float(self.start_sec), 4),
            "end_sec": round(float(self.end_sec), 4),
            "text": self.text,
            "description": self.description,
        }


def label_to_signal_type(label: str) -> str | None:
    m = {
        "threat": "verbal_threat",
        "insult": "verbal_insult",
        "humiliation": "verbal_humiliation",
        "aggressive_command": "aggressive_command",
        "neutral": None,
    }
    return m.get(label)
