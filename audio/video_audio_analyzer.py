"""
Пайплайн: видео → WAV 16k → VAD → ASR → текст/эмоция → audio signals (этап 6).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from audio.asr_service import ASRService
from audio.audio_extract import extract_audio_16khz
from audio.audio_signals import AudioSignal, label_to_signal_type
from audio.speech_emotion import SpeechEmotionAnalyzer
from audio.text_aggression_classifier import TextAggressionClassifier
from audio.vad import split_speech_segments
from inference.audio_extract import probe_has_audio_stream

logger = logging.getLogger(__name__)


def analyze_video_audio(
    video_path: str,
    asr_model_size: str = "small",
    asr_device: str = "cpu",
    language: str | None = "ru",
) -> dict[str, Any]:
    """
    Полный анализ аудиодорожки видео. Без аудио — мягкий результат с ``has_audio: false``.
    """
    empty_summary = {
        "has_audio": False,
        "segments_count": 0,
        "audio_signals_count": 0,
        "max_audio_severity": 0.0,
    }
    vp = Path(video_path)
    if not vp.is_file():
        return {
            "error": f"файл не найден: {video_path}",
            "video_path": str(video_path),
            "summary": empty_summary,
            "transcript": [],
            "audio_signals": [],
        }

    if not probe_has_audio_stream(vp):
        return {
            "video_path": str(vp.resolve()),
            "summary": {**empty_summary, "has_audio": False},
            "transcript": [],
            "audio_signals": [],
            "note": "Аудиодорожка не обнаружена — анализ речи пропущен (pipeline не прерван).",
        }

    transcript: list[dict[str, Any]] = []
    audio_signals: list[dict[str, Any]] = []

    try:
        with tempfile.TemporaryDirectory(prefix="audio6_") as tmp:
            wav_full = Path(tmp) / "full_16k.wav"
            extract_audio_16khz(str(vp), str(wav_full))

            segments = split_speech_segments(str(wav_full), work_dir=str(Path(tmp) / "vad"))
            if not segments:
                segments = [{"start_sec": 0.0, "end_sec": -1.0, "wav_path": str(wav_full)}]

            asr = ASRService(model_size=asr_model_size, device=asr_device, language=language)
            clf = TextAggressionClassifier()
            emo = SpeechEmotionAnalyzer()

            for seg in segments:
                seg_wav = seg["wav_path"]
                t_off = float(seg["start_sec"])
                t_end = float(seg["end_sec"])

                rows = asr.transcribe(seg_wav)
                emo_out = emo.predict(seg_wav)

                for row in rows:
                    txt = (row.get("text") or "").strip()
                    rs = float(row.get("start_sec", 0.0))
                    re = float(row.get("end_sec", -1.0))
                    abs_s = t_off + rs
                    if re >= 0:
                        abs_e = t_off + re
                    elif t_end > 0:
                        abs_e = t_end
                    else:
                        abs_e = abs_s + 2.0

                    transcript.append(
                        {
                            "start_sec": round(abs_s, 3),
                            "end_sec": round(float(abs_e), 3),
                            "text": txt,
                            "confidence": row.get("confidence"),
                        }
                    )
                    if not txt:
                        continue

                    agg = clf.classify(txt)
                    label = str(agg.get("label", "neutral"))
                    stype = label_to_signal_type(label)
                    if stype:
                        sev = float(agg.get("confidence", 0.5))
                        audio_signals.append(
                            AudioSignal(
                                signal_type=stype,
                                severity=min(0.95, sev),
                                start_sec=abs_s,
                                end_sec=float(abs_e),
                                text=txt,
                                description=(
                                    f"Verbal risk signal ({stype}) — aggressive speech signal; "
                                    "not a bullying verdict"
                                ),
                            ).to_dict()
                        )

                em = str(emo_out.get("emotion", "neutral"))
                if em in ("anger", "distress") and float(emo_out.get("confidence", 0)) > 0.5:
                    st = "scream_or_shout" if em == "anger" else "distress_voice"
                    te = t_end if t_end > 0 else t_off + 2.0
                    audio_signals.append(
                        AudioSignal(
                            signal_type=st,
                            severity=float(emo_out.get("confidence", 0.55)),
                            start_sec=t_off,
                            end_sec=float(te),
                            text=None,
                            description="Possible scream / high arousal voice (MVP) — not bullying verdict",
                        ).to_dict()
                    )

            max_sev = max((float(s.get("severity", 0.0)) for s in audio_signals), default=0.0)

            return {
                "video_path": str(vp.resolve()),
                "summary": {
                    "has_audio": True,
                    "segments_count": len(segments),
                    "audio_signals_count": len(audio_signals),
                    "max_audio_severity": round(float(max_sev), 4),
                },
                "transcript": transcript,
                "audio_signals": audio_signals,
            }
    except Exception as exc:
        logger.exception("analyze_video_audio failed")
        return {
            "error": str(exc),
            "video_path": str(video_path),
            "summary": empty_summary,
            "transcript": [],
            "audio_signals": [],
        }
