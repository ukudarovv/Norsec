"""Запись кандидатов в обучающий датасет (JSON Lines)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_path() -> Path:
    raw = os.environ.get("TRAINING_CANDIDATES_PATH")
    if raw:
        return Path(raw).resolve()
    return Path(__file__).resolve().parent.parent / "data" / "training_candidates.jsonl"


def record_candidate(payload: dict[str, Any]) -> Path:
    """
    Добавляет одну строку JSON в конец файла.

    Ожидаемые поля (минимум): ``incident_id``, ``status``, ``tags``, ``labels``,
    ``clip_path``, ``snapshot_path``.
    """
    path = _default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row.setdefault("created_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    return path


def read_last_n(n: int = 50) -> list[dict[str, Any]]:
    path = _default_path()
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
