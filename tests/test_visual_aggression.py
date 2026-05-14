"""Юниты для violence-proxy; лёгкая проверка config на Hub (skip при оффлайне)."""

from __future__ import annotations

import json
import os
import unittest

from inference.visual_aggression import (
    DEFAULT_VISUAL_VIOLENCE_MODEL,
    aggression_label_probability,
    summarize_scores,
)


class TestAggressionLabels(unittest.TestCase):
    def test_locih_style_safe_top(self):
        preds = [{"label": "safe", "score": 0.92}, {"label": "unsafe", "score": 0.08}]
        prob, top = aggression_label_probability(preds)
        self.assertAlmostEqual(prob, 0.08, places=5)
        self.assertEqual(top, "safe")

    def test_locih_style_unsafe_top(self):
        preds = [{"label": "unsafe", "score": 0.88}, {"label": "safe", "score": 0.12}]
        prob, top = aggression_label_probability(preds)
        self.assertAlmostEqual(prob, 0.88, places=5)
        self.assertEqual(top, "unsafe")

    def test_violence_substring_label(self):
        preds = [{"label": "violence_detected", "score": 0.82}]
        prob, _ = aggression_label_probability(preds)
        self.assertAlmostEqual(prob, 0.82, places=5)

    def test_singleton_nonviolence_is_zero(self):
        preds = [{"label": "NON_VIOLENCE", "score": 0.97}]
        prob, _ = aggression_label_probability(preds)
        self.assertAlmostEqual(prob, 0.0, places=5)


class TestSummarizeScores(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(summarize_scores([]), {"max": 0.0, "mean": 0.0, "p90": 0.0})

    def test_basic(self):
        s = summarize_scores([0.0, 1.0, 0.5])
        self.assertAlmostEqual(s["max"], 1.0, places=5)
        self.assertAlmostEqual(s["mean"], 0.5, places=5)


class TestHFHubMetadata(unittest.TestCase):
    """Убеждаемся, что дефолтный репозиторий подходит для transformers (AutoConfig)."""

    def test_default_repo_config_has_model_type(self):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise unittest.SkipTest("huggingface_hub not installed") from None

        try:
            cfg_path = hf_hub_download(repo_id=DEFAULT_VISUAL_VIOLENCE_MODEL, filename="config.json")
        except Exception as exc:
            raise unittest.SkipTest(f"Hugging Face недоступен: {exc}") from exc

        with open(cfg_path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertTrue(
            data.get("model_type"),
            "config.json должен содержать model_type (иначе pipeline(preview) падает)",
        )
        self.assertIn("ViTForImageClassification", data.get("architectures", []))


@unittest.skipUnless(os.environ.get("RUN_HF_SLOW_TESTS") == "1", "set RUN_HF_SLOW_TESTS=1 — скачивает веса ViT")
class TestHFIntegrationSmoke(unittest.TestCase):
    def test_classify_zero_image_cpu(self):
        import numpy as np
        import torch
        from inference.visual_aggression import classify_image_rgb

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        prob, lbl = classify_image_rgb(
            img,
            model_id=DEFAULT_VISUAL_VIOLENCE_MODEL,
            torch_device=torch.device("cpu"),
        )
        self.assertIsInstance(lbl, str)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
