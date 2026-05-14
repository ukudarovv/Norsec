"""Общий даунскейл RGB-кадра для ускорения инференса без смены моделей."""

from __future__ import annotations

import cv2
import numpy as np


def downscale_rgb_max_long_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Уменьшает кадр так, чтобы max(h, w) <= max_side; max_side <= 0 — без изменений."""
    if max_side <= 0:
        return rgb
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    scale = max_side / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(rgb[:, :, :3], (nw, nh), interpolation=cv2.INTER_AREA)
