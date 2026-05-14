"""Кандидат на инцидент для ручной проверки оператором (не вердикт вины)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IncidentCandidate:
    camera_id: str
    start_sec: float
    end_sec: float
    risk_score: float
    risk_level: str
    signal_types: list[str]
    involved_track_ids: list[int]
    explanation: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)
    incident_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "camera_id": self.camera_id,
            "start_sec": round(float(self.start_sec), 4),
            "end_sec": round(float(self.end_sec), 4),
            "risk_score": round(float(self.risk_score), 4),
            "risk_level": self.risk_level,
            "signal_types": list(self.signal_types),
            "involved_track_ids": list(self.involved_track_ids),
            "explanation": list(self.explanation),
            "evidence": dict(self.evidence),
        }
        if self.incident_id:
            d["incident_id"] = self.incident_id
        return d
