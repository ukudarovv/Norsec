"""
Person detection via Ultralytics YOLO (YOLOv8 / YOLO11).

COCO class 0 = person. Outputs local `person_id` as 0..N-1 detection index
(stable IDs come from `track_people`).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bullying_ai.types import DetectionDict

logger = logging.getLogger(__name__)

# COCO "person"
_PERSON_CLASS_ID = 0


class PersonDetector:
    """Loads a YOLO model and runs person-only detection."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._device = device
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            if self._device:
                self._model.to(self._device)
            logger.info("Loaded YOLO weights: %s", self.model_name)
        return self._model

    def detect_people(self, frame: np.ndarray) -> list[DetectionDict]:
        """
        Args:
            frame: BGR or RGB image, shape (H, W, 3), uint8.

        Returns:
            List of dicts with `person_id` (local index), `bbox` [x1,y1,x2,y2] float,
            and `confidence`.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be HxWx3")
        model = self._ensure_model()
        results = model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[_PERSON_CLASS_ID],
            verbose=False,
        )
        if not results:
            return []
        r0 = results[0]
        boxes = r0.boxes
        if boxes is None or len(boxes) == 0:
            return []
        out: list[DetectionDict] = []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i].tolist())
            c = float(conf[i])
            out.append(
                {
                    "person_id": i,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": c,
                }
            )
        return out


def detect_people(
    frame: np.ndarray,
    model_name: str = "yolo11n.pt",
    *,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.5,
    device: str | None = None,
) -> list[DetectionDict]:
    """One-shot detection (creates a detector each call — use `PersonDetector` in a loop)."""
    det = PersonDetector(
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )
    return det.detect_people(frame)
