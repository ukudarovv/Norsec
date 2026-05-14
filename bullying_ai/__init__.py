"""
Real-Time Bullying Detection Platform — phase 1 (detection, tracking, pose).

Install optional deps: pip install -r requirements-bullying.txt
This package is independent from inference/; integrate behind a facade later.
"""

from __future__ import annotations

from bullying_ai.detectors.person_detector import PersonDetector, detect_people
from bullying_ai.pose.skeleton import PoseEstimator, extract_skeleton, torso_heading_from_coco17_xy
from bullying_ai.trackers.people_tracker import PeopleTracker, track_people
from bullying_ai.types import DetectionDict, KeypointDict, SkeletonDict, TrackedPersonDict, TrajectoryPoint

__all__ = [
    "DetectionDict",
    "KeypointDict",
    "TrackedPersonDict",
    "TrajectoryPoint",
    "SkeletonDict",
    "PersonDetector",
    "PeopleTracker",
    "PoseEstimator",
    "detect_people",
    "track_people",
    "extract_skeleton",
    "torso_heading_from_coco17_xy",
]
