"""Хранение траекторий центроидов по track_id (этап 2)."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

from inference.person_tracker import TrackedPerson

logger = logging.getLogger(__name__)


class TrajectoryStore:
    """История точек на трек; общий лимит точек на трек — max_history."""

    def __init__(self, max_history: int = 90) -> None:
        self.max_history = max(1, int(max_history))
        self._points: dict[int, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )

    def update(self, tracked_people: list[TrackedPerson], timestamp_sec: float) -> None:
        for tp in tracked_people:
            x1, y1, x2, y2 = tp.bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            pt = {
                "timestamp_sec": round(float(timestamp_sec), 4),
                "center_x": int(round(cx)),
                "center_y": int(round(cy)),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            }
            self._points[tp.track_id].append(pt)

    def get_track_history(self, track_id: int) -> list[dict[str, Any]]:
        q = self._points.get(track_id)
        if not q:
            return []
        return list(q)

    def get_all_histories(self) -> dict[str, list[dict[str, Any]]]:
        return {str(k): list(v) for k, v in sorted(self._points.items())}

    def reset(self) -> None:
        self._points.clear()
