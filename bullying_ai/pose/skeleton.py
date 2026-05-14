"""
2D pose / skeleton extraction.

Default backend: Ultralytics YOLO-pose (`yolo11n-pose.pt`) — portable substitute for
RTMPose/OpenPose; swap `PoseEstimator` internals later for MMPose without changing
callers of `extract_skeleton(frame, bbox)`.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from bullying_ai.types import KeypointDict, SkeletonDict

logger = logging.getLogger(__name__)

_COCO17_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

_LEFT_SHOULDER, _RIGHT_SHOULDER = 5, 6
_LEFT_HIP, _RIGHT_HIP = 11, 12


def _bbox_clip(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1c = int(max(0, min(w - 1, x1)))
    y1c = int(max(0, min(h - 1, y1)))
    x2c = int(max(0, min(w, x2)))
    y2c = int(max(0, min(h, y2)))
    if x2c <= x1c:
        x2c = min(w, x1c + 1)
    if y2c <= y1c:
        y2c = min(h, y1c + 1)
    return x1c, y1c, x2c, y2c


def torso_heading_from_coco17_xy(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    *,
    min_confidence: float = 0.2,
) -> float | None:
    """
    Angle of vector (hip midpoint -> shoulder midpoint) vs +x axis, degrees.

    Args:
        keypoints_xy: (17, 2) array, COCO 17 order.
        keypoints_conf: (17,) visibility/confidence per joint.
    """
    need = (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)
    for idx in need:
        if idx >= len(keypoints_conf) or keypoints_conf[idx] < min_confidence:
            return None
    ls = keypoints_xy[_LEFT_SHOULDER]
    rs = keypoints_xy[_RIGHT_SHOULDER]
    lh = keypoints_xy[_LEFT_HIP]
    rh = keypoints_xy[_RIGHT_HIP]
    sx = 0.5 * (ls[0] + rs[0])
    sy = 0.5 * (ls[1] + rs[1])
    hx = 0.5 * (lh[0] + rh[0])
    hy = 0.5 * (lh[1] + rh[1])
    vx = sx - hx
    vy = sy - hy
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return None
    return float(math.degrees(math.atan2(vy, vx)))


class PoseEstimator:
    """YOLO-pose on a person crop (full-frame keypoints returned)."""

    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
        conf_threshold: float = 0.25,
        crop_padding_ratio: float = 0.12,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.crop_padding_ratio = crop_padding_ratio
        self._device = device
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            if self._device:
                self._model.to(self._device)
            logger.info("Loaded pose model: %s", self.model_name)
        return self._model

    def extract_skeleton(self, frame: np.ndarray, bbox: list[float]) -> SkeletonDict:
        """
        Args:
            frame: HxWx3 uint8 (BGR or RGB — consistent with detector).
            bbox: [x1, y1, x2, y2] in frame pixels.

        Returns:
            Structured keypoints (full image coords) and optional `torso_heading_deg`.
        """
        if len(bbox) != 4:
            raise ValueError("bbox must have 4 elements")
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(float, bbox)
        bw, bh = x2 - x1, y2 - y1
        pad = self.crop_padding_ratio * max(bw, bh, 1.0)
        cx1, cy1, cx2, cy2 = _bbox_clip(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w, h)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return {"keypoints": [], "torso_heading_deg": None}

        model = self._ensure_model()
        results = model.predict(
            source=crop,
            conf=self.conf_threshold,
            verbose=False,
        )
        if not results:
            return {"keypoints": [], "torso_heading_deg": None}
        r0 = results[0]
        if r0.keypoints is None or r0.keypoints.data is None or len(r0.keypoints.data) == 0:
            return {"keypoints": [], "torso_heading_deg": None}

        # Pick instance: largest box area inside (or closest to target bbox)
        kp_all = r0.keypoints.data.cpu().numpy()
        # x,y,conf per keypoint
        best_i = 0
        if len(kp_all) > 1 and hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes) > 0:
            bx = r0.boxes.xyxy.cpu().numpy()
            areas = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
            best_i = int(np.argmax(areas))

        inst = kp_all[best_i]
        keypoints: list[KeypointDict] = []
        kp_xy = np.zeros((17, 2), dtype=np.float32)
        kp_cf = np.zeros(17, dtype=np.float32)
        for j in range(min(17, inst.shape[0])):
            px, py = float(inst[j, 0]), float(inst[j, 1])
            pc = float(inst[j, 2]) if inst.shape[1] > 2 else 1.0
            fx = px + cx1
            fy = py + cy1
            name = _COCO17_NAMES[j] if j < len(_COCO17_NAMES) else f"kpt_{j}"
            keypoints.append(
                {"name": name, "x": fx, "y": fy, "confidence": pc}
            )
            kp_xy[j, 0], kp_xy[j, 1], kp_cf[j] = fx, fy, pc

        heading = torso_heading_from_coco17_xy(kp_xy, kp_cf)
        return {"keypoints": keypoints, "torso_heading_deg": heading}


def extract_skeleton(
    frame: np.ndarray,
    bbox: list[float],
    model_name: str = "yolo11n-pose.pt",
    *,
    conf_threshold: float = 0.25,
    crop_padding_ratio: float = 0.12,
    device: str | None = None,
) -> SkeletonDict:
    """Stateless one-shot; for video prefer a single `PoseEstimator` instance."""
    est = PoseEstimator(
        model_name=model_name,
        conf_threshold=conf_threshold,
        crop_padding_ratio=crop_padding_ratio,
        device=device,
    )
    return est.extract_skeleton(frame, bbox)
