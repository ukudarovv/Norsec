"""Юниты для CLIP-прокси опасных предметов."""

from __future__ import annotations

import unittest

from inference.dangerous_objects import (
    LABEL_SCENE_SAFE,
    LABEL_SCENE_WEAPON,
    aggregate_clip_multilabel_outputs,
    weapon_proxy_score_from_topk,
)


class TestAggregateMultilabel(unittest.TestCase):
    def test_weapon_proxy_sums_weapon_prompts(self):
        w1 = "a photo showing a handgun or pistol clearly visible"
        w2 = "a photo showing a rifle, shotgun or long gun clearly visible"
        safe = LABEL_SCENE_SAFE
        fs = frozenset({w1, w2})
        out = aggregate_clip_multilabel_outputs(
            [
                {"label": safe, "score": 0.1},
                {"label": w1, "score": 0.55},
                {"label": w2, "score": 0.35},
            ],
            safe_label=safe,
            weapon_labels=fs,
        )
        self.assertAlmostEqual(out[0], 0.9, places=5)
        self.assertIn("handgun", out[1].lower())
        self.assertIn("0.550", out[2])

    def test_top_label_is_global_argmax(self):
        out = aggregate_clip_multilabel_outputs(
            [
                {"label": "winning", "score": 0.91},
                {"label": LABEL_SCENE_WEAPON, "score": 0.09},
            ],
            weapon_labels=frozenset({LABEL_SCENE_WEAPON}),
        )
        self.assertAlmostEqual(out[0], 0.09, places=5)
        self.assertEqual(out[1][:9], "winning")


class TestWeaponProxyScoreLegacy(unittest.TestCase):
    def test_prefers_weapon_label_score(self):
        out = weapon_proxy_score_from_topk(
            [
                {"label": LABEL_SCENE_WEAPON, "score": 0.35},
                {"label": "other", "score": 0.65},
            ]
        )
        self.assertAlmostEqual(out[0], 0.35, places=5)

    def test_top_label_is_highest_ranked(self):
        out = weapon_proxy_score_from_topk(
            [
                {"label": "FIRST", "score": 0.9},
                {"label": LABEL_SCENE_WEAPON, "score": 0.1},
            ]
        )
        self.assertEqual(out[1], "FIRST")
        self.assertAlmostEqual(out[0], 0.1, places=5)


if __name__ == "__main__":
    unittest.main()
