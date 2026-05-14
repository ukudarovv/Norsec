"""Тесты этапа 5: буфер клипов, severity, пустой клип, JSON."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from actions.action_model import ActionRecognizer
from actions.action_signals import ActionSignal, action_severity, build_action_signal, detect_pair_interactions
from actions.clip_buffer import ClipBuffer


class ClipBufferTests(unittest.TestCase):
    def test_fifo_maxlen(self) -> None:
        buf = ClipBuffer(clip_length=4, stale_frame_gap=1000)
        c = np.zeros((8, 8, 3), dtype=np.uint8)
        for i in range(5):
            c[:] = i
            buf.update(1, c.copy(), i)
        self.assertTrue(buf.is_ready(1))
        clip = buf.get_clip(1)
        assert clip is not None
        self.assertEqual(clip.shape[0], 4)
        self.assertLess(abs(float(clip[-1].mean()) - 4.0), 0.6)
        self.assertLess(abs(float(clip[0].mean()) - 1.0), 0.6)

    def test_stale_track_cleanup(self) -> None:
        buf = ClipBuffer(clip_length=2, stale_frame_gap=50)
        c = np.ones((4, 4, 3), dtype=np.uint8) * 100
        buf.update(7, c, 0)
        buf.update(7, c, 1)
        self.assertTrue(buf.is_ready(7))
        buf.update(8, c, 200)
        self.assertFalse(buf.is_ready(7))


class ActionModelTests(unittest.TestCase):
    def test_empty_clip(self) -> None:
        ar = ActionRecognizer(confidence_threshold=0.0)
        out = ar.predict(torch.zeros((0, 3, 8, 8)))
        self.assertEqual(out["action"], "normal")
        self.assertLessEqual(out["confidence"], 0.01)

    def test_severity_mapping(self) -> None:
        self.assertAlmostEqual(action_severity("punch"), 0.9)
        self.assertAlmostEqual(action_severity("push"), 0.65)
        self.assertAlmostEqual(action_severity("defend"), 0.4)

    def test_build_action_signal_json_keys(self) -> None:
        s = build_action_signal(2, "push", 0.81, 4.2)
        d = s.to_dict()
        self.assertEqual(d["track_id"], 2)
        self.assertEqual(d["action_type"], "push")
        self.assertIn("confidence", d)
        self.assertIn("severity", d)
        self.assertIn("timestamp_sec", d)


class InteractionTests(unittest.TestCase):
    def test_possible_conflict(self) -> None:
        sigs = [
            ActionSignal(2, "punch", 0.9, 0.9, 1.0, "a"),
            ActionSignal(5, "defend", 0.5, 0.4, 1.0, "b"),
        ]
        inter = detect_pair_interactions(sigs)
        self.assertTrue(any(x.get("interaction") == "possible_conflict" for x in inter))


if __name__ == "__main__":
    unittest.main()
