"""Тесты аудио-пайплайна этапа 6 (без тяжёлого ASR при необходимости)."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from audio.audio_signals import label_to_signal_type
from audio.text_aggression_classifier import TextAggressionClassifier
from audio.video_audio_analyzer import analyze_video_audio
from audio.vad import split_speech_segments


class TextAggressionTests(unittest.TestCase):
    def test_insult_ru(self) -> None:
        c = TextAggressionClassifier()
        r = c.classify("Ты идиот, уйди отсюда")
        self.assertEqual(r["label"], "insult")

    def test_threat_ru(self) -> None:
        c = TextAggressionClassifier()
        r = c.classify("Я тебя убью если не уйдёшь")
        self.assertEqual(r["label"], "threat")

    def test_label_to_signal_type(self) -> None:
        self.assertEqual(label_to_signal_type("threat"), "verbal_threat")
        self.assertIsNone(label_to_signal_type("neutral"))


class VideoAudioAnalyzerTests(unittest.TestCase):
    def test_missing_file(self) -> None:
        r = analyze_video_audio("__no_such_file__.mp4")
        self.assertIn("error", r)
        self.assertFalse(r["summary"]["has_audio"])

    @patch("audio.video_audio_analyzer.probe_has_audio_stream", return_value=False)
    def test_no_audio_stream(self, _m: MagicMock) -> None:
        r = analyze_video_audio(str(Path(__file__).resolve()))
        self.assertFalse(r["summary"]["has_audio"])
        self.assertEqual(r.get("transcript"), [])


class VADShapeTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("importlib").util.find_spec("soundfile") is not None,
        "soundfile required",
    )
    def test_split_returns_segments(self) -> None:
        import numpy as np
        import soundfile as sf
        import tempfile

        sr = 16000
        t = np.zeros(sr * 2, dtype=np.float32)
        t[4000:12000] = 0.3 * np.sin(np.linspace(0, 50 * 2 * np.pi, 8000))
        fd, path = tempfile.mkstemp(suffix=".wav")
        import os

        os.close(fd)
        try:
            sf.write(path, t, sr, subtype="PCM_16")
            segs = split_speech_segments(path)
            self.assertGreaterEqual(len(segs), 1)
            self.assertIn("wav_path", segs[0])
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
