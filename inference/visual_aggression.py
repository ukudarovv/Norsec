"""
Прокси физической агрессии по выборке кадров: предобученная HF image-classification (violence/non-violence).

Дисклеймер: это не школьный буллинг и не юридическое заключение; возможны ложные срабатывания.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_VISUAL_VIOLENCE_MODEL = "locih/violence_classification"


def _pipeline_device(torch_device: torch.device | str | None) -> int:
    if torch_device is None:
        td = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        td = torch.device(torch_device)
    if td.type == "cuda":
        idx = td.index if td.index is not None else 0
        if not torch.cuda.is_available():
            return -1
        return int(idx)
    return -1


@lru_cache(maxsize=8)
def _cached_image_classifier(model_id: str, device_idx: int) -> Any:
    from transformers import pipeline

    try:
        return pipeline("image-classification", model=model_id, device=device_idx)
    except Exception as e:
        msg = str(e).lower()
        hint = ""
        if "model_type" in msg or "unrecognized model" in msg:
            hint = (
                " Репозиторий несовместим с текущими transformers (часто нет поля `model_type` в config.json). "
                f"Выберите другой HF repo или модель по умолчанию `{DEFAULT_VISUAL_VIOLENCE_MODEL}`."
            )
        raise RuntimeError(f"{e}{hint}") from e


def _normalized_label(name: str) -> str:
    return re.sub(r"[\s_-]+", "", str(name).strip().lower())


def classify_image_rgb(
    rgb: np.ndarray,
    *,
    model_id: str,
    torch_device: torch.device,
) -> tuple[float, str]:
    """
    Один кадр RGB uint8 (H,W,3) → violence proxy score и top label из top-k пайплайна.
    """
    from PIL import Image

    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return 0.0, ""
    rgb = rgb[:, :, :3]
    pil = Image.fromarray(rgb)

    pipe = _cached_image_classifier(model_id, _pipeline_device(torch_device))
    raw = pipe(pil, top_k=5)
    preds: list[dict[str, Any]]
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        preds = raw
    elif isinstance(raw, dict):
        preds = [raw]
    else:
        preds = []
    return aggression_label_probability(preds)


def aggression_label_probability(predictions: list[dict[str, Any]]) -> tuple[float, str]:
    """
    Ищем в top-k класс, соответствующий насилию по подстроке в id2label; избегаем non/nonviolent.
    Суммируем score только по классам-«агрессия» (обычно один класс).
    """
    if not predictions:
        return 0.0, ""

    def is_aggression_name(name: str) -> bool:
        n = _normalized_label(name)
        if (
            "nonviolence" in n
            or "nonfight" in n
            or n in ("peaceful", "normal", "safe", "neutral")
            or (n.startswith("no") and "viol" in n)
        ):
            return False
        hints = (
            "violenc",
            "violent",
            "fight",
            "fighting",
            "brawl",
            "brawling",
            "punch",
            "scuffle",
            "riot",
            "assault",
            "aggression",
            "aggressive",
            # locih/violence_classification: классы safe / unsafe
            "unsafe",
        )
        return any(h in n for h in hints)

    violence_score = 0.0
    top_l = str(predictions[0].get("label", ""))

    seen = False
    for p in predictions:
        lab = str(p.get("label", ""))
        sc = float(p.get("score", 0.0))
        if is_aggression_name(lab):
            violence_score += sc
            seen = True
    if seen:
        return min(1.0, violence_score), top_l

    # Fallback: берём топ-скор среди классов, не содержащих non/nonviolent
    best = 0.0
    for p in predictions:
        lab_raw = str(p.get("label", ""))
        n = _normalized_label(lab_raw)
        if "nonviol" in n or n in ("neutral", "safe"):
            continue
        best = max(best, float(p.get("score", 0.0)))
    if best > 0:
        return best, top_l

    # Один класс и явно безопасность / non-violence → агрессия 0 (не путать топ-скор с violence_proxy)
    if len(predictions) == 1:
        n0 = _normalized_label(str(predictions[0].get("label", "")))
        if (
            "nonviol" in n0
            or n0 in ("safe", "neutral", "peaceful", "normal")
            or (n0.startswith("no") and "viol" in n0)
        ):
            return 0.0, top_l

    return float(predictions[0].get("score", 0.0)), top_l


def sample_video_frames_evenly(
    video_path: Path | str,
    *,
    max_frames: int = 48,
    target_fps_sample: float = 1.0,
    default_fps_fallback: float = 25.0,
) -> list[tuple[float, Any]]:
    """
    Равномерно по времени (индекс кадра), затем Pillow RGB.
    Возвращает список (timestamp_s, PIL.Image).
    """
    import cv2
    from PIL import Image

    path = Path(video_path).resolve()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = default_fps_fallback
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total <= 0:
        frames: list[tuple[int, Any]] = []
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % max(1, int(round(fps / max(0.1, target_fps_sample)))) == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append((i, Image.fromarray(rgb)))
            i += 1
            if len(frames) >= max_frames:
                break
        cap.release()
        return [(fi / fps, im) for fi, im in frames]

    duration = total / fps
    n_fps = max(1, int(duration * target_fps_sample))
    n = max(1, min(max_frames, n_fps))
    n = min(n, total)
    indices = sorted({int(round(x)) for x in np.linspace(0, total - 1, num=n)})
    indices = [max(0, min(total - 1, ix)) for ix in indices]

    out: list[tuple[float, Any]] = []
    for ix in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ix)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out.append((ix / fps, Image.fromarray(rgb)))
    cap.release()
    return out


def summarize_scores(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"max": 0.0, "mean": 0.0, "p90": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def analyze_visual_aggression(
    video_path: Path | str,
    *,
    model_id: str = DEFAULT_VISUAL_VIOLENCE_MODEL,
    torch_device: torch.device | str | None = None,
    max_frames: int = 48,
    target_fps_sample: float = 1.0,
) -> dict[str, Any]:
    """
    Возвращает словарь под ключ visual в payload: summary, frames (усечённый per_frame для JSON).
    При ошибке: visual_error текст, enabled True.
    """
    path = Path(video_path).resolve()
    pipe_dev = _pipeline_device(torch_device)

    sampled = sample_video_frames_evenly(
        path,
        max_frames=max_frames,
        target_fps_sample=target_fps_sample,
    )
    if not sampled:
        return {
            "enabled": True,
            "model_id": model_id,
            "frames_used": 0,
            "summary": summarize_scores([]),
            "per_frame": [],
            "peaks": [],
            "visual_error": "Не удалось прочитать кадры (OpenCV).",
        }

    try:
        pipe = _cached_image_classifier(model_id, pipe_dev)
    except Exception as e:  # noqa: BLE001
        return {
            "enabled": True,
            "model_id": model_id,
            "frames_used": 0,
            "summary": summarize_scores([]),
            "per_frame": [],
            "peaks": [],
            "visual_error": f"Не удалось загрузить модель: {e}",
        }

    images = [im for _, im in sampled]

    try:
        raw = pipe(images, top_k=5)
    except Exception as e:  # noqa: BLE001
        return {
            "enabled": True,
            "model_id": model_id,
            "frames_used": len(images),
            "summary": summarize_scores([]),
            "per_frame": [],
            "peaks": [],
            "visual_error": f"classification: {e}",
        }

    # Один вход: часто [{label,score},...]; несколько входов: [[{...}], ...]
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        raw = [raw]

    per_frame: list[dict[str, Any]] = []
    scores: list[float] = []

    for i, (ts, _) in enumerate(sampled):
        preds = raw[i] if i < len(raw) else []
        if not isinstance(preds, list):
            preds = []
        vprob, top_lab = aggression_label_probability(preds)
        scores.append(vprob)
        per_frame.append(
            {
                "timestamp_s": round(ts, 3),
                "violence_probability": round(vprob, 4),
                "top_label": top_lab,
            }
        )

    summary = summarize_scores(scores)

    sorted_idx = np.argsort(scores)[::-1][: min(5, len(scores))]
    peaks = [per_frame[int(j)] for j in sorted_idx]

    return {
        "enabled": True,
        "model_id": model_id,
        "frames_used": len(images),
        "summary": {k: round(v, 4) for k, v in summary.items()},
        "per_frame": per_frame,
        "peaks": peaks,
        "visual_error": None,
    }


def visual_disclaimer_md() -> str:
    return (
        "_Визуальный слой_: **прокси физической агрессии** (выборка кадров + HF-модель violence/non-violence). "
        "**Не является** школьным буллингом; возможны ложные срабатывания (спорт, бег и т.д.)."
    )


def visual_results_markdown_section(visual: dict[str, Any]) -> list[str]:
    """Строки markdown (без ведущего ## — заголовок снаружи)."""
    lines: list[str] = []
    if not visual.get("enabled"):
        return lines

    lines.append("")
    lines.append(visual_disclaimer_md())
    lines.append("")

    err = visual.get("visual_error")
    if err:
        lines.append(f"**Ошибка визуального слоя:** {err}")
        return lines

    su = visual.get("summary") or {}
    lines.append(
        f"Выбрано кадров: **{visual.get('frames_used', 0)}**. "
        f"Модель: `{visual.get('model_id', '')}`."
    )
    lines.append("")
    lines.append(f"- **Агрессия (прокси) — max:** {su.get('max', 0):.4f}")
    lines.append(f"- **mean:** {su.get('mean', 0):.4f}")
    lines.append(f"- **p90:** {su.get('p90', 0):.4f}")
    lines.append("")
    lines.append("| t (с) | violence_prob | top label |")
    lines.append("|-----:|---------------:|-----------|")

    peaks = visual.get("peaks") or []
    if not peaks:
        lines.append("| — | — | _(нет данных)_ |")
        return lines

    for row in peaks:
        lines.append(
            f"| {row['timestamp_s']:.2f} | {row['violence_probability']} | "
            f"{str(row['top_label']).replace('|', '/')} |"
        )
    return lines
