"""Распознавание действий по клипу: эвристика MVP + опционально SlowFast/X3D (pytorchvideo)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

ACTIONS = [
    "normal",
    "push",
    "punch",
    "kick",
    "grab",
    "chase",
    "escape",
    "fall",
    "block",
    "defend",
]


def _clip_to_tensor_uint8(clip_hwc: np.ndarray) -> torch.Tensor:
    """(T,H,W,3) uint8 → (T,3,H,W) float32 [0,1]."""
    t = torch.from_numpy(np.asarray(clip_hwc, dtype=np.float32) / 255.0)
    if t.ndim != 4:
        raise ValueError("clip must be T,H,W,3")
    return t.permute(0, 3, 1, 2).contiguous()


def _heuristic_predict(clip_tchw: torch.Tensor) -> dict[str, Any]:
    """
    Лёгкий MVP без fine-tuned классов: оценка темпорального движения и геометрии кропа.
    Не заменяет обученный SlowFast; даёт стабильные action signals для пайплайна и UI.
    """
    if clip_tchw.numel() == 0 or clip_tchw.shape[0] < 2:
        return {"action": "normal", "confidence": 0.25}

    x = clip_tchw
    diff = (x[1:] - x[:-1]).abs()
    motion = float(diff.mean().item())
    h, w = x.shape[-2], x.shape[-1]
    aspect = (w / max(h, 1)) if h > 0 else 1.0

    if aspect > 1.45 and motion < 0.04:
        act, conf = "fall", min(0.85, 0.5 + aspect * 0.1)
    elif motion < 0.012:
        act, conf = "normal", 0.72
    elif motion < 0.022:
        act, conf = "defend", 0.4
    elif motion < 0.03:
        act, conf = "block", 0.42
    elif motion > 0.12:
        if aspect > 1.2:
            act, conf = "kick", min(0.9, 0.55 + motion * 2.0)
        else:
            act, conf = "punch", min(0.92, 0.5 + motion * 2.2)
    elif motion > 0.07:
        act, conf = "push", min(0.88, 0.45 + motion * 3.5)
    elif motion > 0.045:
        act, conf = "grab", min(0.82, 0.4 + motion * 4.0)
    elif motion > 0.035:
        act, conf = "chase", min(0.75, 0.42 + motion * 5.0)
    elif motion > 0.03:
        act, conf = "escape", min(0.7, 0.4 + motion * 5.0)
    else:
        act, conf = "normal", 0.55

    if act not in ACTIONS:
        act, conf = "normal", 0.3
    return {"action": act, "confidence": float(round(conf, 4))}


class ActionRecognizer:
    """
    ``model_name``: ``slowfast`` | ``x3d`` | ``heuristic`` (логируется; инференс MVP — эвристика по движению в кропе).
    Полноценный SlowFast/X3D через pytorchvideo можно подключить позже без смены API.
    """

    def __init__(
        self,
        model_name: str = "heuristic",
        confidence_threshold: float = 0.35,
        device: str | None = None,
    ) -> None:
        self.model_name = (model_name or "heuristic").strip().lower()
        self.confidence_threshold = float(confidence_threshold)
        self._device = device  # зарезервировано под NN-backend
        self._torch_model: Any = None
        self._backend = "heuristic"
        self._load_nn_if_requested()

    def _load_nn_if_requested(self) -> None:
        """MVP: эвристика по движению в кропе. SlowFast/X3D (pytorchvideo) — в следующих итерациях."""
        self._backend = "heuristic"
        self._torch_model = None
        if self.model_name not in ("heuristic", "clip", "lightweight"):
            logger.info(
                "ActionRecognizer: model_name=%s → motion heuristic (pytorchvideo SlowFast/X3D not wired in this MVP).",
                self.model_name,
            )

    def predict(self, clip_tensor: torch.Tensor) -> dict[str, Any]:
        """
        ``clip_tensor``: ``(T, 3, H, W)`` float 0..1.
        Возвращает ``{"action": str, "confidence": float}``.
        """
        if clip_tensor is None or clip_tensor.numel() == 0:
            return {"action": "normal", "confidence": 0.0}
        if clip_tensor.dim() != 4:
            raise ValueError("clip_tensor must be (T, C, H, W)")

        out = _heuristic_predict(clip_tensor)
        if out["confidence"] < self.confidence_threshold and out["action"] != "normal":
            return {"action": "normal", "confidence": float(out["confidence"])}
        return out
