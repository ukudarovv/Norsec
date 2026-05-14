"""Тесты Fusion Engine (этап 7)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fusion.fusion_engine import FusionEngine
from fusion.incident_store import IncidentStore
from fusion.risk_levels import risk_level_from_score
from fusion.video_fusion_analyzer import analyze_video_fusion


class RiskLevelTests(unittest.TestCase):
    def test_bins(self) -> None:
        self.assertEqual(risk_level_from_score(0.0), "green")
        self.assertEqual(risk_level_from_score(0.24), "green")
        self.assertEqual(risk_level_from_score(0.25), "yellow")
        self.assertEqual(risk_level_from_score(0.49), "yellow")
        self.assertEqual(risk_level_from_score(0.50), "orange")
        self.assertEqual(risk_level_from_score(0.74), "orange")
        self.assertEqual(risk_level_from_score(0.75), "red")
        self.assertEqual(risk_level_from_score(1.0), "red")


class FusionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FusionEngine()

    def test_empty_signals_returns_none(self) -> None:
        r = self.engine.fuse_window(
            "cam1",
            0.0,
            10.0,
            [],
            [],
            [],
            [],
            context=None,
        )
        self.assertIsNone(r)

    def test_low_risk_returns_none(self) -> None:
        social = [{"signal_type": "following", "severity": 0.1, "track_ids": [1], "timestamp_sec": 0.0}]
        pose = [{"signal_type": "raised_hand", "severity": 0.15, "track_id": 1, "timestamp_sec": 0.1}]
        r = self.engine.fuse_window("cam", 0.0, 5.0, social, pose, [], [], None)
        self.assertIsNone(r)

    def test_punch_and_fast_arm_motion_creates_incident(self) -> None:
        pose = [
            {
                "signal_type": "fast_arm_motion",
                "severity": 0.85,
                "track_id": 1,
                "timestamp_sec": 1.0,
            }
        ]
        action = [
            {
                "action_type": "punch",
                "severity": 0.9,
                "track_id": 1,
                "confidence": 0.8,
                "timestamp_sec": 1.0,
            }
        ]
        c = self.engine.fuse_window("cam", 0.0, 5.0, [], pose, action, [], None)
        self.assertIsNotNone(c)
        assert c is not None
        self.assertGreaterEqual(c.risk_score, 0.50)
        self.assertIn("punch", c.signal_types)
        self.assertTrue(any("physical action signal: punch" in x.lower() for x in c.explanation))
        self.assertTrue(any("fast_arm_motion" in x for x in c.explanation))

    def test_group_surrounding_verbal_threat_escalation(self) -> None:
        social = [
            {
                "signal_type": "group_surrounding",
                "severity": 0.7,
                "track_ids": [1, 2, 3],
                "timestamp_sec": 2.0,
            }
        ]
        audio = [
            {
                "signal_type": "verbal_threat",
                "severity": 0.85,
                "start_sec": 2.0,
                "end_sec": 3.0,
                "text": "test",
                "description": "verbal risk",
            }
        ]
        c1 = FusionEngine().fuse_window("cam", 0.0, 10.0, social, [], [], audio, None)
        self.assertIsNotNone(c1)
        assert c1 is not None
        self.assertTrue(any("group surrounding" in x.lower() for x in c1.explanation))

    def test_risk_score_capped_at_one(self) -> None:
        social = [{"signal_type": "crowding", "severity": 1.0, "track_ids": [1, 2], "timestamp_sec": 0.0}]
        pose = [
            {"signal_type": "person_on_ground", "severity": 1.0, "track_id": 3, "timestamp_sec": 0.0},
            {"signal_type": "fast_arm_motion", "severity": 1.0, "track_id": 1, "timestamp_sec": 0.0},
        ]
        action = [{"action_type": "kick", "severity": 1.0, "track_id": 1, "confidence": 1.0, "timestamp_sec": 0.0}]
        audio = [{"signal_type": "verbal_threat", "severity": 1.0, "start_sec": 0.0, "end_sec": 1.0, "text": None}]
        eng = FusionEngine()
        c = eng.fuse_window("cam", 0.0, 10.0, social, pose, action, audio, {"severity": 1.0})
        self.assertIsNotNone(c)
        assert c is not None
        self.assertLessEqual(c.risk_score, 1.0)

    def test_explanation_has_detected_lines(self) -> None:
        pose = [{"signal_type": "fast_arm_motion", "severity": 0.9, "track_id": 1, "timestamp_sec": 0.0}]
        action = [{"action_type": "punch", "severity": 0.9, "track_id": 1, "confidence": 0.9, "timestamp_sec": 0.0}]
        c = self.engine.fuse_window("cam", 0.0, 5.0, [], pose, action, [], None)
        self.assertIsNotNone(c)
        assert c is not None
        self.assertTrue(any("detected" in e.lower() for e in c.explanation))


class IncidentStoreTests(unittest.TestCase):
    def test_save_list_get(self) -> None:
        from fusion.incident_candidate import IncidentCandidate

        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / "inc.json")
            store = IncidentStore(p)
            cand = IncidentCandidate(
                camera_id="c1",
                start_sec=0.0,
                end_sec=1.0,
                risk_score=0.9,
                risk_level="red",
                signal_types=["punch"],
                involved_track_ids=[1],
                explanation=["test"],
                evidence={"a": 1},
            )
            iid = store.save(cand)
            self.assertTrue(iid)
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            self.assertEqual(len(data["incidents"]), 1)
            row = store.get_by_id(iid)
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["risk_score"], 0.9)


class VideoFusionMissingFileTests(unittest.TestCase):
    def test_missing_video(self) -> None:
        payload, prev = analyze_video_fusion("__no_such__.mp4", camera_id="x")
        self.assertIsNone(prev)
        self.assertIn("error", payload)
        self.assertFalse(payload["summary"]["incident_created"])


if __name__ == "__main__":
    unittest.main()
