"""
Лица и настроение по одному кадру: Haar-каскад OpenCV (число лиц) + ViT эмоций на крупнейшем лице.

Ограничения: каскад даёт грубую оценку (пропуски/ложные срабатывания); эмоции — по вырезанному лицу,
не «состояние тела» и не клиническая оценка.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import torch

DEFAULT_EMOTION_MODEL = "dima806/facial_emotions_image_detection"


def _pipeline_device_idx(torch_device: torch.device | str | None) -> int:
    if torch_device is None:
        td = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        td = torch.device(torch_device)
    if td.type == "cuda":
        return int(td.index if td.index is not None else 0) if torch.cuda.is_available() else -1
    return -1


@lru_cache(maxsize=6)
def _cached_emotion_classifier(model_id: str, device_idx: int) -> Any:
    from transformers import pipeline

    return pipeline("image-classification", model=model_id, device=device_idx)


def count_faces_and_largest_crop(rgb: np.ndarray) -> tuple[int, np.ndarray | None]:
    """
    Возвращает (число найденных лиц, вырезка RGB крупнейшего лица или None).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return 0, None
    faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return 0, None
    areas = np.asarray(faces[:, 2], dtype=np.int64) * np.asarray(faces[:, 3], dtype=np.int64)
    idx = int(np.argmax(areas))
    x, y, w, h = [int(v) for v in faces[idx]]
    pad = max(8, int(0.12 * max(w, h)))
    h2, w2 = rgb.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w2, x + w + pad)
    y1 = min(h2, y + h + pad)
    if x1 <= x0 + 8 or y1 <= y0 + 8:
        return len(faces), None
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    return len(faces), crop


def classify_emotion_topk(
    face_rgb: np.ndarray,
    *,
    model_id: str,
    torch_device: torch.device,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    from PIL import Image

    pil = Image.fromarray(np.clip(face_rgb, 0, 255).astype(np.uint8))
    mid = (model_id or "").strip() or DEFAULT_EMOTION_MODEL
    pipe = _cached_emotion_classifier(mid, _pipeline_device_idx(torch_device))
    raw = pipe(pil, top_k=int(top_k))
    if isinstance(raw, list):
        return raw
    return []


def analyze_people_mood(rgb: np.ndarray, *, emotion_model_id: str, torch_device: torch.device) -> dict[str, Any]:
    n, crop = count_faces_and_largest_crop(rgb)
    if crop is None or n == 0:
        return {
            "face_count": int(n),
            "emotion_primary": None,
            "emotion_confidence": None,
            "emotion_topk": [],
        }
    tops = classify_emotion_topk(crop, model_id=emotion_model_id, torch_device=torch_device)
    primary = tops[0] if tops else None
    return {
        "face_count": int(n),
        "emotion_primary": (primary or {}).get("label"),
        "emotion_confidence": round(float((primary or {}).get("score", 0.0)), 4) if primary else None,
        "emotion_topk": [{"label": str(x.get("label", "")), "score": round(float(x.get("score", 0.0)), 4)} for x in tops],
    }


def people_disclaimer_md() -> str:
    return (
        "_Лица и настроение_: число анфасных лиц (Haar) и эмоция по **крупнейшему** вырезу (ViT). "
        "**Не** подсчёт людей со спины и **не** анализ позы или здоровья."
    )

