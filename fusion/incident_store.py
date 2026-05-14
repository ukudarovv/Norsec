"""MVP хранение кандидатов инцидентов в JSON (этап 7)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fusion.incident_candidate import IncidentCandidate

logger = logging.getLogger(__name__)


class IncidentStore:
    def __init__(self, path: str = "data/incidents.json") -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"incidents": []}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("incidents"), list):
                return raw
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("IncidentStore load failed: %s", e)
        return {"incidents": []}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self, candidate: IncidentCandidate) -> str:
        iid = candidate.incident_id or str(uuid.uuid4())
        candidate.incident_id = iid
        blob = candidate.to_dict()
        blob["created_at"] = datetime.now(timezone.utc).isoformat()
        blob["review_status"] = "new"
        blob["review_history"] = []
        data = self._load()
        data["incidents"].append(blob)
        self._write(data)
        return iid

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._load().get("incidents", []))

    def get_by_id(self, incident_id: str) -> dict[str, Any] | None:
        for row in self.list_all():
            if row.get("incident_id") == incident_id:
                return row
        return None
