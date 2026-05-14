"""Потоковый режим с микрофона: Whisper + текстовый risk-head по коротким окнам."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inference.verbal_head import load_verbal_model, verbal_scores_batch

_lock = threading.Lock()
_pipe_asr = None
_verbal_tok_model: tuple[Any, Any, torch.device] | None = None
_cfg_key: tuple[str, str, str] | None = None

# Онлайн: по умолчанию не давать Whisper «уезжать» в китайский/японский на шуме (типичные галлюцинации).
DEFAULT_LIVE_WHISPER_LANGUAGE = "russian"
_LIVE_RMS_FLOOR = float(os.environ.get("LIVE_MIC_RMS_FLOOR", "0.008"))


def _likely_script_hallucination(text: str) -> bool:
    """Длинный текст: в основном один повторяющийся символ / CJK без кириллицы — часто мусор Whisper."""
    s = text.strip()
    if len(s) < 18:
        return False
    cyr = sum(1 for ch in s if "а" <= ch.lower() <= "я" or ch.lower() == "ё")
    lat = sum(1 for ch in s if "a" <= ch.lower() <= "z")
    cjk = sum(
        1
        for ch in s
        if ("\u4e00" <= ch <= "\u9fff") or ("\u3040" <= ch <= "\u30ff") or ("\uac00" <= ch <= "\ud7a3")
    )
    if cjk / max(len(s), 1) > 0.35 and cyr / max(len(s), 1) < 0.03 and lat / max(len(s), 1) < 0.03:
        return True
    uniq = len(set(s))
    if uniq <= 3 and len(s) >= 24:
        return True
    return False


def _normalize_mono(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr


def _device_for(gradio_choice: str) -> torch.device:
    c = (gradio_choice or "auto").strip().lower()
    if c == "cpu":
        return torch.device("cpu")
    if c.startswith("cuda"):
        dev = torch.device(c)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Выбрана CUDA, но она недоступна.")
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pip_device_index(dt: torch.device) -> int:
    if dt.type != "cuda":
        return -1
    idx = dt.index
    return 0 if idx is None else int(idx)


def ensure_streaming_models(
    whisper_model: str,
    verbal_ckpt: Path,
    gradio_device: str,
) -> None:
    global _pipe_asr, _verbal_tok_model, _cfg_key

    root_dev = _device_for(gradio_device)
    key = (whisper_model, str(verbal_ckpt.resolve()), str(root_dev))

    with _lock:
        if key == _cfg_key and _pipe_asr is not None and _verbal_tok_model is not None:
            return

        from transformers import pipeline

        pip_idx = _pip_device_index(root_dev)

        _pipe_asr = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            device=pip_idx,
        )
        tm = load_verbal_model(verbal_ckpt, device=root_dev)
        _verbal_tok_model = tm
        _cfg_key = key


@dataclass
class LiveLine:
    t_wall: str
    text: str
    scores: dict[str, float]


def process_streaming_chunk(
    audio: tuple[int, np.ndarray] | None,
    prior_lines: list[dict[str, Any]],
    *,
    whisper_model: str,
    verbal_ckpt: str,
    gradio_device: str,
    window_s: float,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Вызывается из Gradio `Audio.stream` каждые stream_every секунд.

    `audio`: (sample_rate, mono/stereo float numpy) — накопленная запись с начала сессии.
    """
    if audio is None:
        return _lines_to_md(prior_lines), prior_lines

    sr, y = audio
    if sr <= 0 or y is None:
        return _lines_to_md(prior_lines), prior_lines

    y = _normalize_mono(np.asarray(y))
    if y.size < int(0.4 * sr):
        return _lines_to_md(prior_lines), prior_lines

    win = min(int(window_s * sr), y.size)
    clip = y[-win:]

    if float(np.sqrt(np.mean(np.square(clip)))) < _LIVE_RMS_FLOOR:
        return _lines_to_md(prior_lines), prior_lines

    ckpt = Path(verbal_ckpt).expanduser()
    if not ckpt.exists():
        return (
            f"### Ошибка\nНет чекпоинта: `{ckpt}`",
            prior_lines,
        )

    try:
        ensure_streaming_models(whisper_model, ckpt, gradio_device)
    except Exception as e:  # noqa: BLE001
        return f"### Ошибка загрузки моделей\n`{e}`", prior_lines

    with _lock:
        pipe = _pipe_asr
        pack = _verbal_tok_model

    assert pipe is not None and pack is not None
    model, tokenizer, vdev = pack

    try:
        lang = (os.environ.get("LIVE_WHISPER_LANGUAGE") or DEFAULT_LIVE_WHISPER_LANGUAGE).strip() or "russian"
        out = pipe(
            {"array": clip, "sampling_rate": int(sr)},
            generate_kwargs={"task": "transcribe", "language": lang},
        )
        text = (out.get("text") if isinstance(out, dict) else str(out)) or ""
        text = str(text).strip()
    except Exception as e:  # noqa: BLE001
        return f"### ASR ошибка\n`{e}`", prior_lines

    if _likely_script_hallucination(text):
        return _lines_to_md(prior_lines), prior_lines

    if not text:
        return _lines_to_md(prior_lines), prior_lines

    _probs, verbals = verbal_scores_batch(model, tokenizer, vdev, [text])
    score_map = verbals[0]

    from datetime import datetime

    line = LiveLine(
        t_wall=datetime.now().strftime("%H:%M:%S"),
        text=text,
        scores=score_map,
    )
    row = {
        "t": line.t_wall,
        "text": line.text,
        "scores": line.scores,
    }
    if prior_lines and prior_lines[-1].get("text") == row["text"]:
        return _lines_to_md(prior_lines), prior_lines

    new_lines = list(prior_lines) + [row]
    if len(new_lines) > 120:
        new_lines = new_lines[-120:]

    return _lines_to_md(new_lines), new_lines


def _lines_to_md(lines: list[dict[str, Any]]) -> str:
    if not lines:
        return (
            "## Онлайн (микрофон)\n\n"
            "Нажмите **запись** на компоненте аудио и говорите. Каждые несколько секунд обновляется распознавание и оценки.\n\n"
            "_Короткие фразы и шум снижают качество; для нормального WER — тихая комната. "
            "На тишине Whisper иногда выдаёт «левый» текст (в т.ч. иероглифы) — включён язык **русский** и отсев таких отрезков._"
        )

    parts = ["## Онлайн: последние фрагменты", ""]
    for row in lines[-40:]:
        s = row["scores"]
        parts.append(
            f"- **{row['t']}** — {row['text'][:300]}{'…' if len(row['text']) > 300 else ''}  \n"
            f"  `neutral`={s.get('neutral_conflict', 0):.2f} · `insult`={s.get('insult_humiliation', 0):.2f} · "
            f"`threat`={s.get('explicit_threat', 0):.2f} · `coercion`={s.get('coercion_harassment', 0):.2f}"
        )
    parts.append("")
    parts.append(
        "_Это не полноценный safety-мониторинг в реальном времени: задержка зависит от CPU/GPU и размера окна._"
    )
    return "\n".join(parts)


def format_verbal_log(lines: list[dict[str, Any]]) -> str:
    """Markdown для списка строк онлайн-распознавания (как внутри process_streaming_chunk)."""
    return _lines_to_md(lines)


def clear_live_log() -> tuple[str, list[dict[str, Any]]]:
    return _lines_to_md([]), []
