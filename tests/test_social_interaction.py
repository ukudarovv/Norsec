"""Тесты social interaction (этап 3)."""

from __future__ import annotations

import unittest

import numpy as np

from inference.person_tracker import TrackedPerson
from inference.social_interaction import (
    SIGNAL_RAPID_APPROACH,
    SocialInteractionAnalyzer,
    SocialSignal,
    compute_distance_matrix,
    draw_social_signals,
)


class SocialInteractionTests(unittest.TestCase):
    def test_distance_matrix(self) -> None:
        a = TrackedPerson(1, (0, 0, 6, 8), 0.9)
        b = TrackedPerson(2, (6, 8, 12, 16), 0.9)
        m = compute_distance_matrix([a, b])
        self.assertIn("1-2", m)
        self.assertAlmostEqual(m["1-2"], 10.0, places=3)

    def test_rapid_approach_synthetic(self) -> None:
        an = SocialInteractionAnalyzer(rapid_approach_px_per_sec=100.0)
        far = [
            TrackedPerson(1, (0, 0, 10, 10), 0.9),
            TrackedPerson(2, (400, 0, 410, 10), 0.9),
        ]
        self.assertEqual(an.update(far, {}, 0.0), [])
        close = [
            TrackedPerson(1, (0, 0, 10, 10), 0.9),
            TrackedPerson(2, (80, 0, 90, 10), 0.9),
        ]
        sigs = an.update(close, {}, 0.5)
        types = [s.signal_type for s in sigs]
        self.assertIn("rapid_approach", types)

    def test_group_surrounding_synthetic(self) -> None:
        an = SocialInteractionAnalyzer(close_distance_px=150.0, surrounding_min_people=3)
        tracked = [
            TrackedPerson(5, (190, 190, 210, 210), 0.9),
            TrackedPerson(2, (250, 190, 270, 210), 0.9),
            TrackedPerson(3, (190, 250, 210, 270), 0.9),
            TrackedPerson(4, (130, 190, 150, 210), 0.9),
        ]
        sigs = an.update(tracked, {}, 1.0)
        types = [s.signal_type for s in sigs]
        self.assertIn("group_surrounding", types)

    def test_empty_tracked(self) -> None:
        an = SocialInteractionAnalyzer()
        self.assertEqual(an.update([], {"1": []}, 0.0), [])

    def test_draw_social_signals_shape(self) -> None:
        frame = np.zeros((64, 48, 3), dtype=np.uint8)
        tracked = [TrackedPerson(1, (5, 5, 20, 30), 0.9)]
        sig = SocialSignal(
            signal_type=SIGNAL_RAPID_APPROACH,
            severity=0.5,
            track_ids=[1],
            description="test",
            timestamp_sec=0.0,
        )
        out = draw_social_signals(frame, tracked, [sig])
        self.assertEqual(out.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
