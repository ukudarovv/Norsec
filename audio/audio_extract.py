"""Извлечение моно WAV 16 kHz из видео (этап 6)."""

from __future__ import annotations

import logging
from pathlib import Path

from inference.audio_extract import extract_wav_mono_16k, probe_has_audio_stream, resolve_ffmpeg_executable

logger = logging.getLogger(__name__)


def extract_audio_16khz(video_path: str, output_wav_path: str) -> str:
    """
    Извлекает аудио: mono, 16 kHz, PCM WAV.

    Returns:
        Путь к ``output_wav_path`` (строка).

    Raises:
        FileNotFoundError: нет ffmpeg или файла видео.
        RuntimeError: нет аудиодорожки или ошибка ffmpeg.
    """
    vp = Path(video_path)
    if not vp.is_file():
        raise FileNotFoundError(f"видео не найдено: {video_path}")
    if not resolve_ffmpeg_executable():
        raise FileNotFoundError(
            "ffmpeg не найден (PATH или imageio-ffmpeg). Установите ffmpeg или: pip install imageio-ffmpeg"
        )
    if not probe_has_audio_stream(vp):
        raise RuntimeError("в медиафайле нет аудиодорожки — извлечение невозможно")

    out = Path(output_wav_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    extract_wav_mono_16k(vp, out)
    logger.info("extract_audio_16khz: wrote %s", out)
    return str(out.resolve())
