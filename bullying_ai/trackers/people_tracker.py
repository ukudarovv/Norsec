"""
Multi-person tracking with ByteTrack (via supervision).

Keeps stable `person_id` (tracker id) and optional trajectory memory (centroids).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from bullying_ai.types import DetectionDict, TrackedPersonDict, TrajectoryPoint

logger = logging.getLogger(__name__)


def _to_xyxy_array(detections: list[DetectionDict]) -> tuple[np.ndarray, np.ndarray]:
    if not detections:
        z = np.zeros((0, 4), dtype=np.float32)
        return z, np.zeros((0,), dtype=np.float32)
    xyxy = np.asarray([d["bbox"] for d in detections], dtype=np.float32)
    conf = np.asarray([d["confidence"] for d in detections], dtype=np.float32)
    return xyxy, conf


class PeopleTracker:
    """ByteTrack wrapper with trajectory memory."""

    def __init__(
        self,
        trajectory_maxlen: int = 60,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: float = 30.0,
    ) -> None:
        self.trajectory_maxlen = trajectory_maxlen
        self._frame_idx = 0
        self._trajectories: dict[int, deque[TrajectoryPoint]] = {}
        self._byte_track: Any = None
        self._track_kwargs = {
            "lost_track_buffer": lost_track_buffer,
            "minimum_matching_threshold": minimum_matching_threshold,
            "frame_rate": frame_rate,
        }

    def _ensure_tracker(self) -> Any:
        if self._byte_track is None:
            import supervision as sv

            self._byte_track = sv.ByteTrack(**self._track_kwargs)
            logger.info("Initialized ByteTrack (%s)", self._track_kwargs)
        return self._byte_track

    def reset(self) -> None:
        """Clear tracker and trajectories (e.g. new video stream)."""
        self._byte_track = None
        self._trajectories.clear()
        self._frame_idx = 0

    def track_people(self, detections: list[DetectionDict]) -> list[TrackedPersonDict]:
        """
        Update tracks given current-frame person detections (same camera order as `detect_people`).

        Args:
            detections: Output list from `PersonDetector.detect_people` for this frame.

        Returns:
            Tracked dicts with persistent `person_id` (ByteTrack id) and bbox/confidence.
        """
        import supervision as sv

        tracker = self._ensure_tracker()
        xyxy, conf = _to_xyxy_array(detections)
        if len(xyxy) == 0:
            sv_dets = sv.Detections.empty()
        else:
            class_id = np.zeros(len(xyxy), dtype=int)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)

        sv_dets = tracker.update_with_detections(sv_dets)
        self._frame_idx += 1

        out: list[TrackedPersonDict] = []
        if sv_dets.xyxy is None or len(sv_dets.xyxy) == 0:
            return out
        tids = sv_dets.tracker_id
        if tids is None:
            return out

        for i in range(len(sv_dets.xyxy)):
            tid = tids[i]
            if tid is None:
                continue
            tid_int = int(tid)
            x1, y1, x2, y2 = map(float, sv_dets.xyxy[i].tolist())
            cf = float(sv_dets.confidence[i]) if sv_dets.confidence is not None else 0.0
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            self._record(tid_int, cx, cy)
            out.append(
                {
                    "person_id": tid_int,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": cf,
                }
            )
        return out

    def _record(self, track_id: int, cx: float, cy: float) -> None:
        if track_id not in self._trajectories:
            self._trajectories[track_id] = deque(maxlen=self.trajectory_maxlen)
        self._trajectories[track_id].append(
            {"frame_idx": self._frame_idx, "x": cx, "y": cy}
        )

    def get_trajectory(self, person_id: int) -> list[TrajectoryPoint]:
        """Return centroid history for a track id (may be empty)."""
        q = self._trajectories.get(person_id)
        if not q:
            return []
        return list(q)


def track_people(
    detections: list[DetectionDict],
    tracker: PeopleTracker | None = None,
) -> tuple[list[TrackedPersonDict], PeopleTracker]:
    """
    Functional API: updates `tracker`, or creates a new `PeopleTracker` if None.

    Returns:
        (tracked_people, tracker_instance) — keep `tracker_instance` across frames.
    """
    tr = tracker or PeopleTracker()
    return tr.track_people(detections), tr
