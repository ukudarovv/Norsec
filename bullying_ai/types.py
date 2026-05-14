"""Shared typed structures for bullying_ai."""

from __future__ import annotations

from typing import TypedDict


class DetectionDict(TypedDict):
    """Single person box from `detect_people` (local indices until tracking)."""

    person_id: int
    bbox: list[float]
    confidence: float


class TrackedPersonDict(TypedDict):
    """Output after ByteTrack assigns persistent IDs."""

    person_id: int
    bbox: list[float]
    confidence: float


class TrajectoryPoint(TypedDict):
    """One stored centroid for a track."""

    frame_idx: int
    x: float
    y: float


class KeypointDict(TypedDict):
    """One keypoint in full-frame pixel coordinates."""

    name: str
    x: float
    y: float
    confidence: float


class SkeletonDict(TypedDict):
    """`extract_skeleton` return value."""

    keypoints: list[KeypointDict]
    torso_heading_deg: float | None
