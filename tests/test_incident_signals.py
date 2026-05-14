"""Юниты для сводки «драка / конфликт»."""

from __future__ import annotations

import unittest

from inference.incident_signals import (
    fuse_batch_incident,
    incident_escalation_note_ru,
    physical_proxy_from_visual_rows,
    verbal_conflict_score,
)


class TestVerbalConflictScore(unittest.TestCase):
    def test_aggregate_is_max_of_four(self):
        probs = {
            "neutral_conflict": 0.1,
            "insult_humiliation": 0.6,
            "explicit_threat": 0.2,
            "coercion_harassment": 0.15,
        }
        agg, comp = verbal_conflict_score(probs)
        self.assertAlmostEqual(agg, 0.6)
        self.assertAlmostEqual(comp["conflict_focus"], 0.6)
        self.assertAlmostEqual(comp["escalation_peak"], 0.2)

    def test_escalation_note_detects_threat(self):
        note = incident_escalation_note_ru(
            {
                "neutral_conflict": 0.1,
                "insult_humiliation": 0.1,
                "explicit_threat": 0.5,
                "coercion_harassment": 0.1,
            }
        )
        self.assertIn("угроза", note)

    def test_fuse_batch_incident_values(self):
        visual = {"enabled": True, "summary": {"max": 0.73}}
        probs = {
            "neutral_conflict": 0.5,
            "insult_humiliation": 0.1,
            "explicit_threat": 0.0,
            "coercion_harassment": 0.0,
        }
        inc = fuse_batch_incident(visual, probs)
        self.assertAlmostEqual(inc["incident_physical_proxy"], 0.73)
        self.assertGreater(inc["incident_verbal_conflict"], 0.4)

    def test_physical_proxy_from_live_rows(self):
        rows = [
            {"violence_probability": 0.3, "_sig": 1.0},
            {"violence_probability": 0.88, "_sig": 2.0},
            {"error": True, "violence_probability": None},
        ]
        self.assertAlmostEqual(physical_proxy_from_visual_rows(rows), 0.88)


if __name__ == "__main__":
    unittest.main()
