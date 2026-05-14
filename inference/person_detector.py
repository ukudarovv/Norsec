"""
YOLO (ultralytics) person detection + bbox drawing for RGB frames.

Этап 1: только детекция людей, без трекинга и буллинга.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO
_PERSON_CLASS_ID = 0
_COCO_PERSON_LABEL = "person"


@dataclass
class PersonDetection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    label: str = "person"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3])],
            "confidence": float(round(self.confidence, 4)),
            "label": self.label,
            "class_id": int(self.class_id),
        }


class PersonDetector:
    """Ultralytics YOLO: только класс person."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._device = device
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO(self.model_name)
                if self._device:
                    self._model.to(self._device)
                logger.info("PersonDetector: loaded %s", self.model_name)
            except Exception:
                logger.exception("PersonDetector: failed to load model %s", self.model_name)
                raise
        return self._model

    def detect(self, frame_rgb: np.ndarray) -> List[PersonDetection]:
        """
        Детекция людей на RGB-кадре (HxWx3, uint8).

        Пустой или некорректный кадр → [].
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return []
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            logger.warning("PersonDetector: expected HxWx3 RGB, got shape %s", getattr(frame_rgb, "shape", None))
            return []

        try:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            model = self._ensure_model()
            results = model.predict(
                source=frame_bgr,
                conf=self.confidence_threshold,
                classes=[_PERSON_CLASS_ID],
                verbose=False,
            )
        except Exception:
            logger.exception("PersonDetector: predict failed")
            return []

        if not results:
            return []
        r0 = results[0]
        boxes = r0.boxes
        if boxes is None or len(boxes) == 0:
            return []

        out: List[PersonDetection] = []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else None

        h, w = frame_rgb.shape[:2]
        for i in range(len(xyxy)):
            c = float(conf[i])
            if c < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = xyxy[i]
            xi1 = int(max(0, min(w - 1, round(float(x1)))))
            yi1 = int(max(0, min(h - 1, round(float(y1)))))
            xi2 = int(max(0, min(w, round(float(x2)))))
            yi2 = int(max(0, min(h, round(float(y2)))))
            if xi2 <= xi1:
                xi2 = min(w, xi1 + 1)
            if yi2 <= yi1:
                yi2 = min(h, yi1 + 1)
            cid = int(cls_ids[i]) if cls_ids is not None else _PERSON_CLASS_ID
            lab = _COCO_PERSON_LABEL if cid == _PERSON_CLASS_ID else str(cid)
            out.append(
                PersonDetection(
                    bbox=(xi1, yi1, xi2, yi2),
                    confidence=c,
                    class_id=cid,
                    label=lab,
                )
            )
        return out


def draw_person_boxes(
    frame_rgb: np.ndarray,
    detections: List[PersonDetection],
) -> np.ndarray:
    """Копия кадра с нарисованными bbox (RGB для Gradio)."""
    if frame_rgb is None or frame_rgb.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    canvas = copy.deepcopy(frame_rgb)
    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        caption = f"{det.label} {det.confidence:.2f}"
        cv2.putText(
            bgr,
            caption,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def analyze_live_frame_people(
    frame_rgb: np.ndarray,
    detector: PersonDetector | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Один кадр → изображение с bbox + JSON (people_count, detections).

    detector: переиспользуйте один экземпляр между вызовами, чтобы не грузить веса заново.
    """
    det = detector or PersonDetector()
    dets = det.detect(frame_rgb)
    vis = draw_person_boxes(frame_rgb, dets)
    payload: dict[str, Any] = {
        "people_count": len(dets),
        "detections": [d.to_dict() for d in dets],
    }
    return vis, payload
