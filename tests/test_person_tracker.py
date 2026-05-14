"""Тесты PersonTracker, отрисовки и analyze_video_tracking (этап 2)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from inference.person_detector import PersonDetection
from inference.person_tracker import PersonTracker, TrackedPerson, draw_tracked_people
from inference.video_tracking_analyzer import analyze_video_tracking


class PersonTrackerTests(unittest.TestCase):
    def test_empty_detections_does_not_crash(self) -> None:
        tr = PersonTracker()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = tr.update(frame, [])
        self.assertEqual(out, [])

    @patch("inference.person_tracker.sv.ByteTrack")
    def test_detections_forwarded_to_bytetrack(self, mock_bt_cls: MagicMock) -> None:
        instance = MagicMock()
        mock_bt_cls.return_value = instance
        import supervision as sv

        instance.update_with_detections.return_value = sv.Detections(
            xyxy=np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0], dtype=np.int32),
            tracker_id=np.array([3], dtype=np.int32),
        )
        tr = PersonTracker()
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        dets = [
            PersonDetection(bbox=(10, 20, 30, 40), confidence=0.9, class_id=0),
        ]
        out = tr.update(frame, dets)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].track_id, 3)
        instance.update_with_detections.assert_called_once()

    def test_draw_same_shape(self) -> None:
        frame = np.zeros((100, 80, 3), dtype=np.uint8)
        tracked = [
            TrackedPerson(1, (5, 5, 20, 40), 0.91),
        ]
        traj = {"1": [{"timestamp_sec": 0.0, "center_x": 10, "center_y": 20, "bbox": [5, 5, 20, 40]}]}
        out = draw_tracked_people(frame, tracked, trajectories=traj)
        self.assertEqual(out.shape, frame.shape)

    def test_missing_video_file(self) -> None:
        payload, preview = analyze_video_tracking("__no_such_file_xyz__.mp4", max_frames=5)
        self.assertIsNotNone(payload.get("error"))
        self.assertEqual(payload.get("frames_analyzed"), 0)
        self.assertIsNone(preview)


if __name__ == "__main__":
    unittest.main()
