"""Эмоции по аудио: задел под Wav2Vec2 + MVP по громкости/пикам."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EMOTIONS = ["neutral", "anger", "fear", "sadness", "distress"]

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore[assignment]


class SpeechEmotionAnalyzer:
    def __init__(self) -> None:
        self._model: Any = None

    def predict(self, wav_path: str) -> dict[str, Any]:
        path = Path(wav_path)
        if not path.is_file() or sf is None:
            return {
                "emotion": "neutral",
                "confidence": 0.4,
                "note": "no file or soundfile missing — neutral fallback",
            }
        try:
            data, sr = sf.read(str(path), always_2d=False)
        except Exception:
            logger.exception("SpeechEmotionAnalyzer.read failed")
            return {"emotion": "neutral", "confidence": 0.3, "note": "read error"}

        if data.ndim > 1:
            data = np.mean(data, axis=1)
        x = np.asarray(data, dtype=np.float32)
        if len(x) < 16:
            return {"emotion": "neutral", "confidence": 0.5, "note": "too short"}

        rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
        peaks = float(np.mean(np.abs(x) > 0.85 * (np.max(np.abs(x)) + 1e-6)))

        if rms > 0.12 and peaks > 0.02:
            return {
                "emotion": "anger",
                "confidence": min(0.9, 0.45 + rms * 2.0),
                "note": "possible_anger_or_scream (MVP: high RMS + peak rate)",
            }
        if rms < 0.015 and len(x) / sr > 0.5:
            return {
                "emotion": "sadness",
                "confidence": 0.42,
                "note": "low energy sustained — weak proxy for distress/sadness",
            }
        if peaks > 0.08:
            return {
                "emotion": "distress",
                "confidence": 0.55,
                "note": "irregular peaks — possible distress voice",
            }
        return {"emotion": "neutral", "confidence": 0.6, "note": "baseline"}
