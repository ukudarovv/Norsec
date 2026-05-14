from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class TranscriptSegment:
    """One ASR slice with timings in seconds (relative to full video/audio)."""

    start_s: float
    end_s: float
    text: str


def load_wav_numpy(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path))
    if getattr(data, "ndim", 1) == 2:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), int(sr)


def transcribe_chunks(
    wav_path: Path,
    pipe,
    *,
    chunk_s: float = 12.0,
    stride_s: float = 4.0,
) -> tuple[list[TranscriptSegment], int]:
    """
    Chunk raw audio (~16 kHz mono) and run Whisper per chunk via HF pipeline dict input.

    `pipe` — transformers.pipeline("automatic-speech-recognition", model=..., ...)
    """
    wav_path = Path(wav_path).resolve()
    audio, sr = load_wav_numpy(wav_path)
    if sr != 16000:
        raise ValueError(f"Ожидалась частота 16 kHz WAV, получено sr={sr}. Переснимите ffmpeg с -ar 16000.")

    segments: list[TranscriptSegment] = []
    chunk = int(chunk_s * sr)
    stride = int(stride_s * sr)
    if stride <= 0 or stride >= chunk:
        stride = chunk

    offset = 0
    duration = audio.shape[-1]

    while offset < duration:
        end = min(offset + chunk, duration)
        slice_np = audio[offset:end]
        t0 = offset / sr
        t1 = end / sr
        if slice_np.std() < 1e-4:
            txt = ""
        else:
            out = pipe({"array": slice_np, "sampling_rate": sr})
            txt = out["text"].strip()

        segments.append(
            TranscriptSegment(start_s=float(t0), end_s=float(t1), text=txt)
        )

        offset += stride

    return segments, sr
