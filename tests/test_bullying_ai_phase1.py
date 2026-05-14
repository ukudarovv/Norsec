"""
Tests for bullying_ai phase 1 (no Hub weights unless RUN_BULLYING_AI_HEAVY=1).
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from bullying_ai.trackers.people_tracker import PeopleTracker
from bullying_ai.pose.skeleton import torso_heading_from_coco17_xy


class TestTorsoHeading(unittest.TestCase):
    def test_vertical_torso_points_up(self):
        # Shoulders above hips — vector hip->shoulder points up (negative y in image coords)
        kp_xy = np.zeros((17, 2), dtype=np.float32)
        kp_cf = np.ones(17, dtype=np.float32)
        kp_xy[5] = [100.0, 50.0]
        kp_xy[6] = [120.0, 50.0]
        kp_xy[11] = [100.0, 150.0]
        kp_xy[12] = [120.0, 150.0]
        ang = torso_heading_from_coco17_xy(kp_xy, kp_cf)
        self.assertIsNotNone(ang)
        self.assertAlmostEqual(ang or 0.0, -90.0, delta=1e-3)


class TestTrajectoryMemory(unittest.TestCase):
    def test_get_trajectory_empty(self):
        tr = PeopleTracker(trajectory_maxlen=5)
        self.assertEqual(tr.get_trajectory(999), [])


class TestByteTrackEmpty(unittest.TestCase):
    def test_empty_frame_does_not_crash(self):
        try:
            import supervision  # noqa: F401
        except ImportError:
            self.skipTest("supervision not installed (pip install -r requirements-bullying.txt)")
        tr = PeopleTracker(trajectory_maxlen=3)
        self.assertEqual(tr.track_people([]), [])


@unittest.skipUnless(os.environ.get("RUN_BULLYING_AI_HEAVY"), "set RUN_BULLYING_AI_HEAVY=1 for HF/YOLO smoke")
class TestHeavyImports(unittest.TestCase):
    def test_import_ultralytics_supervision(self):
        import ultralytics  # noqa: F401
        import supervision  # noqa: F401

    def test_smoke_detector_on_blank_frame(self):
        from bullying_ai.detectors.person_detector import PersonDetector

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = PersonDetector(model_name="yolo11n.pt")
        dets = det.detect_people(frame)
        self.assertIsInstance(dets, list)
