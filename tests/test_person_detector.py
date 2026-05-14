"""Этап 1: детекция людей (YOLO) — юнит-тесты (YOLO мокается, без скачивания весов)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from inference.person_detector import PersonDetection, PersonDetector, analyze_live_frame_people, draw_person_boxes
from inference.video_person_analyzer import analyze_video_people


class TestPersonDetector(unittest.TestCase):
    @patch("ultralytics.YOLO")
    def test_empty_black_frame_returns_empty_list(self, mock_yolo: MagicMock) -> None:
        mock_model = MagicMock()
        mock_res = MagicMock()
        mock_res.boxes = None
        mock_model.predict.return_value = [mock_res]
        mock_yolo.return_value = mock_model
        det = PersonDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual(det.detect(frame), [])

    def test_draw_person_boxes_same_shape(self):
        frame = np.zeros((100, 80, 3), dtype=np.uint8)
        dets = [
            PersonDetection(bbox=(10, 10, 50, 60), confidence=0.91, class_id=0),
        ]
        out = draw_person_boxes(frame, dets)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_person_detection_to_dict(self):
        d = PersonDetection(bbox=(100, 120, 300, 500), confidence=0.91, class_id=0)
        blob = d.to_dict()
        self.assertEqual(blob["bbox"], [100, 120, 300, 500])
        self.assertAlmostEqual(blob["confidence"], 0.91)
        self.assertEqual(blob["label"], "person")

    def test_analyze_video_missing_file(self):
        payload, prev = analyze_video_people("__no_such_file_zz__.mp4")
        self.assertIn("error", payload)
        self.assertIsNone(prev)


class TestAnalyzeLiveWithoutWeights(unittest.TestCase):
    @patch("ultralytics.YOLO")
    def test_analyze_live_frame_people(self, mock_yolo: MagicMock) -> None:
        mock_model = MagicMock()
        mock_res = MagicMock()
        mock_res.boxes = None
        mock_model.predict.return_value = [mock_res]
        mock_yolo.return_value = mock_model
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        det = PersonDetector()
        vis, js = analyze_live_frame_people(frame, detector=det)
        self.assertEqual(vis.shape, frame.shape)
        self.assertEqual(js["people_count"], 0)
        self.assertEqual(js["detections"], [])
