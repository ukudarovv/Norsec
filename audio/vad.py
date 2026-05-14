"""Простой energy-based VAD и нарезка сегментов в WAV (этап 6)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore[assignment]


def split_speech_segments(wav_path: str, work_dir: str | None = None) -> list[dict[str, Any]]:
    """
    Делит WAV на сегменты по энергии (RMS).

    Возвращает список ``{start_sec, end_sec, wav_path}``.
    При ошибке чтения — один сегмент на весь файл (если файл есть).
    """
    path = Path(wav_path)
    if not path.is_file():
        return []

    if sf is None:
        logger.warning("soundfile not installed; VAD returns whole file as one segment")
        return [{"start_sec": 0.0, "end_sec": -1.0, "wav_path": str(path.resolve())}]

    try:
        data, sr = sf.read(str(path), always_2d=False)
    except Exception:
        logger.exception("split_speech_segments: read failed")
        return [{"start_sec": 0.0, "end_sec": -1.0, "wav_path": str(path.resolve())}]

    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float32)
    if len(data) == 0:
        return []

    frame = max(1, int(0.025 * sr))
    hop = max(1, frame // 2)
    thr = float(np.percentile(np.abs(data), 70) * 0.35 + 1e-6)
    rms = []
    for i in range(0, len(data) - frame, hop):
        rms.append(float(np.sqrt(np.mean(data[i : i + frame] ** 2))))
    if not rms:
        return [{"start_sec": 0.0, "end_sec": len(data) / sr, "wav_path": str(path.resolve())}]

    rms_arr = np.array(rms, dtype=np.float32)
    active = rms_arr > max(thr, float(np.median(rms_arr)) * 0.5)
    segs: list[tuple[float, float]] = []
    in_seg = False
    start_i = 0
    for i, a in enumerate(active):
        t0 = i * hop / sr
        if a and not in_seg:
            in_seg = True
            start_i = i
        elif not a and in_seg:
            in_seg = False
            t1 = (i * hop + frame) / sr
            if t1 - (start_i * hop / sr) >= 0.25:
                segs.append((start_i * hop / sr, min(t1, len(data) / sr)))
    if in_seg:
        t1 = len(data) / sr
        if t1 - (start_i * hop / sr) >= 0.2:
            segs.append((start_i * hop / sr, t1))

    if not segs:
        dur = float(len(data) / sr)
        return [{"start_sec": 0.0, "end_sec": dur, "wav_path": str(path.resolve())}]

    root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="vad_"))
    root.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for idx, (t0, t1) in enumerate(segs):
        i0 = int(t0 * sr)
        i1 = int(t1 * sr)
        i0 = max(0, min(i0, len(data) - 1))
        i1 = max(i0 + 1, min(i1, len(data)))
        chunk = data[i0:i1]
        seg_path = root / f"segment_{idx:03d}.wav"
        sf.write(str(seg_path), chunk, sr, subtype="PCM_16")
        out.append(
            {
                "start_sec": round(float(t0), 3),
                "end_sec": round(float(t1), 3),
                "wav_path": str(seg_path.resolve()),
            }
        )
    return out
