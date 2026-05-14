"""Тесты TrajectoryStore (этап 2)."""

from __future__ import annotations

import unittest

from inference.person_tracker import TrackedPerson
from inference.trajectory_store import TrajectoryStore


class TrajectoryStoreTests(unittest.TestCase):
    def test_max_history_truncates(self) -> None:
        store = TrajectoryStore(max_history=3)
        for i in range(5):
            store.update(
                [TrackedPerson(track_id=1, bbox=(0, 0, 10, 10), confidence=0.9)],
                timestamp_sec=float(i),
            )
        h = store.get_track_history(1)
        self.assertEqual(len(h), 3)
        self.assertEqual(h[0]["timestamp_sec"], 2.0)
        self.assertEqual(h[-1]["timestamp_sec"], 4.0)

    def test_center_computed(self) -> None:
        store = TrajectoryStore(max_history=90)
        store.update(
            [
                TrackedPerson(
                    track_id=7,
                    bbox=(100, 80, 200, 120),
                    confidence=0.85,
                )
            ],
            timestamp_sec=1.0,
        )
        pt = store.get_track_history(7)[0]
        self.assertEqual(pt["center_x"], 150)
        self.assertEqual(pt["center_y"], 100)
        self.assertEqual(pt["bbox"], [100, 80, 200, 120])

    def test_get_all_histories_keys_are_strings(self) -> None:
        store = TrajectoryStore()
        store.update([TrackedPerson(1, (0, 0, 1, 1), 0.5)], 0.0)
        all_h = store.get_all_histories()
        self.assertIn("1", all_h)
        self.assertEqual(len(all_h["1"]), 1)


if __name__ == "__main__":
    unittest.main()
