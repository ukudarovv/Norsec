"""YOLO pose (Ultralytics) + сопоставление с track_id по IoU (этап 4)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from inference.person_tracker import TrackedPerson

logger = logging.getLogger(__name__)

COCO_KEYPOINT_NAMES = [
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
]

COCO_SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


@dataclass
class PoseKeypoint:
    name: str
    x: float
    y: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class PersonPose:
    track_id: int
    bbox: Tuple[int, int, int, int]
    keypoints: Dict[str, PoseKeypoint]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": [int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3])],
            "keypoints": {k: v.to_dict() for k, v in sorted(self.keypoints.items())},
            "confidence": round(float(self.confidence), 4),
        }


def bbox_iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class PoseEstimator:
    """Ultralytics YOLO-pose; сопоставление детекций позы с ``TrackedPerson`` по IoU."""

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        confidence_threshold: float = 0.35,
        device: str | None = None,
        iou_match_threshold: float = 0.25,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = float(confidence_threshold)
        self._device = device
        self.iou_match_threshold = float(iou_match_threshold)
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO(self.model_name)
                if self._device:
                    self._model.to(self._device)
                logger.info("PoseEstimator: loaded %s", self.model_name)
            except Exception:
                logger.exception("PoseEstimator: failed to load %s", self.model_name)
                raise
        return self._model

    def estimate(
        self,
        frame_rgb: np.ndarray,
        tracked_people: list[TrackedPerson],
    ) -> list[PersonPose]:
        out: list[PersonPose] = []
        if not tracked_people:
            return out
        if frame_rgb is None or frame_rgb.size == 0 or frame_rgb.ndim != 3:
            for tp in tracked_people:
                out.append(PersonPose(tp.track_id, tp.bbox, {}, 0.0))
            return out

        candidates: list[tuple[tuple[float, float, float, float], dict[str, PoseKeypoint], float]] = []
        try:
            model = self._ensure_model()
            results = model.predict(
                frame_rgb,
                conf=self.confidence_threshold,
                verbose=False,
                imgsz=min(640, max(frame_rgb.shape[0], frame_rgb.shape[1])),
            )
            if not results:
                raise ValueError("empty results")
            r0 = results[0]
            if r0.keypoints is None or r0.boxes is None or len(r0.boxes) == 0:
                pass
            else:
                xy = r0.keypoints.xy
                cf = r0.keypoints.conf
                boxes = r0.boxes.xyxy.cpu().numpy()
                n = int(len(boxes))
                for i in range(n):
                    x1, y1, x2, y2 = boxes[i]
                    box_t = (float(x1), float(y1), float(x2), float(y2))
                    kdict: dict[str, PoseKeypoint] = {}
                    confs: list[float] = []
                    if xy is not None and len(xy) > i:
                        row = xy[i].cpu().numpy()
                        crow = cf[i].cpu().numpy() if cf is not None else None
                        for j, name in enumerate(COCO_KEYPOINT_NAMES):
                            if j >= row.shape[0]:
                                break
                            px, py = float(row[j, 0]), float(row[j, 1])
                            pc = float(crow[j]) if crow is not None and j < len(crow) else 1.0
                            kdict[name] = PoseKeypoint(name=name, x=px, y=py, confidence=pc)
                            confs.append(pc)
                    mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
                    candidates.append((box_t, kdict, mean_conf))
        except Exception:
            logger.exception("PoseEstimator.estimate failed")
            for tp in tracked_people:
                out.append(PersonPose(tp.track_id, tp.bbox, {}, 0.0))
            return out

        used: set[int] = set()
        for tp in tracked_people:
            best_j: int | None = None
            best_iou = 0.0
            tb = tuple(float(x) for x in tp.bbox)
            for j, (pbox, kdict, _) in enumerate(candidates):
                if j in used:
                    continue
                iou = bbox_iou(pbox, tb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j is not None and best_iou >= self.iou_match_threshold:
                used.add(best_j)
                _, kdict, pconf = candidates[best_j]
                out.append(
                    PersonPose(
                        track_id=tp.track_id,
                        bbox=tp.bbox,
                        keypoints=kdict,
                        confidence=float(pconf),
                    )
                )
            else:
                out.append(PersonPose(track_id=tp.track_id, bbox=tp.bbox, keypoints={}, confidence=0.0))
        return out


def draw_poses(frame_rgb: np.ndarray, poses: list[PersonPose]) -> np.ndarray:
    """Точки, рёбра COCO_SKELETON, подпись ``ID n pose``."""
    import cv2

    img = frame_rgb.copy()
    h, w = img.shape[:2]
    colors = [
        (0, 255, 180),
        (255, 128, 0),
        (200, 100, 255),
        (100, 220, 255),
    ]
    for pi, pose in enumerate(poses):
        col = colors[abs(pose.track_id) % len(colors)]
        pts = pose.keypoints
        for a, b in COCO_SKELETON:
            if a not in pts or b not in pts:
                continue
            pa, pb = pts[a], pts[b]
            if pa.confidence < 0.2 or pb.confidence < 0.2:
                continue
            p1 = (int(max(0, min(pa.x, w - 1))), int(max(0, min(pa.y, h - 1))))
            p2 = (int(max(0, min(pb.x, w - 1))), int(max(0, min(pb.y, h - 1))))
            cv2.line(img, p1, p2, col, 2, cv2.LINE_AA)
        for kp in pts.values():
            if kp.confidence < 0.2:
                continue
            cx = int(max(0, min(kp.x, w - 1)))
            cy = int(max(0, min(kp.y, h - 1)))
            cv2.circle(img, (cx, cy), 4, col, -1, cv2.LINE_AA)
        x1, y1, _, _ = pose.bbox
        label = f"ID {pose.track_id} pose"
        cv2.putText(
            img,
            label,
            (int(x1) + 2, max(int(y1) - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            1,
            cv2.LINE_AA,
        )
    return img
