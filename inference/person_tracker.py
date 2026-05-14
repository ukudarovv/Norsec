"""Трекинг людей поверх детекций YOLO через supervision ByteTrack (этап 2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import supervision as sv

from inference.person_detector import PersonDetection

logger = logging.getLogger(__name__)


@dataclass
class TrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str = "person"


class PersonTracker:
    """ByteTrack с конвертацией из PersonDetection в supervision.Detections."""

    def __init__(
        self,
        track_activation_threshold: float = 0.35,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        self.track_activation_threshold = float(track_activation_threshold)
        self.lost_track_buffer = int(lost_track_buffer)
        self.minimum_matching_threshold = float(minimum_matching_threshold)
        self.frame_rate = int(frame_rate)
        self._byte_track = sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            frame_rate=self.frame_rate,
        )

    def update(
        self,
        frame_rgb: np.ndarray,
        detections: List[PersonDetection],
    ) -> List[TrackedPerson]:
        _ = frame_rgb  # оставлено для расширения (ROI, аналитика по кадру)
        if not detections:
            sv_dets = sv.Detections.empty()
        else:
            xyxy = np.array(
                [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]] for d in detections],
                dtype=np.float32,
            )
            conf = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id = np.zeros(len(detections), dtype=np.int32)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)

        out = self._byte_track.update_with_detections(sv_dets)
        return _tracked_from_sv(out)


def _tracked_from_sv(out: sv.Detections) -> List[TrackedPerson]:
    result: List[TrackedPerson] = []
    if out.xyxy is None or len(out.xyxy) == 0:
        return result
    n = len(out.xyxy)
    conf_arr = out.confidence if out.confidence is not None else np.ones(n, dtype=np.float32)
    tids = out.tracker_id
    for i in range(n):
        if tids is None or tids[i] is None:
            continue
        x1, y1, x2, y2 = out.xyxy[i]
        bx = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        cf = float(conf_arr[i]) if i < len(conf_arr) else 1.0
        result.append(
            TrackedPerson(
                track_id=int(tids[i]),
                bbox=bx,
                confidence=cf,
            )
        )
    return result


def draw_tracked_people(
    frame_rgb: np.ndarray,
    tracked_people: List[TrackedPerson],
    trajectories: dict | None = None,
) -> np.ndarray:
    """Рисует bbox и подпись `ID n | person 0.xx`; опционально линии траектории (≤30 точек)."""
    img = frame_rgb.copy()
    h, w = img.shape[:2]
    colors = [
        (255, 100, 100),
        (100, 255, 100),
        (100, 100, 255),
        (255, 200, 0),
        (200, 0, 255),
        (0, 200, 200),
    ]

    traj = trajectories or {}
    cv2 = _cv2()
    drawn_lines: set[int] = set()
    for tp in tracked_people:
        tid = tp.track_id
        if tid in drawn_lines:
            continue
        drawn_lines.add(tid)
        c = colors[abs(tid) % len(colors)]
        pts = traj.get(str(tid)) or traj.get(tid)
        if pts and len(pts) >= 2:
            last = pts[-30:]
            poly = np.array(
                [[p["center_x"], p["center_y"]] for p in last],
                dtype=np.int32,
            )
            if len(poly) >= 2:
                for j in range(1, len(poly)):
                    p0, p1 = poly[j - 1], poly[j]
                    cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), c, 2)
    for tp in tracked_people:
        tid = tp.track_id
        c = colors[abs(tid) % len(colors)]
        x1, y1, x2, y2 = tp.bbox
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        label = f"ID {tid} | person {tp.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), c, -1)
        cv2.putText(
            img, label, (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return img


def _cv2():
    import cv2

    return cv2


def analyze_live_frame_tracking(
    frame_rgb: np.ndarray,
    state: dict,
    *,
    model_name: str = "yolov8n.pt",
    device: str | None = None,
    confidence: float = 0.35,
) -> tuple[np.ndarray, dict, dict]:
    """Один кадр веб-камеры: детекция + трекинг + траектории; state хранит detector/tracker/store."""
    from inference.person_detector import PersonDetector

    if state is None:
        state = {}
    st = dict(state)

    if st.get("detector") is None:
        st["detector"] = PersonDetector(
            model_name=model_name,
            device=device,
            confidence_threshold=confidence,
        )
    if st.get("tracker") is None:
        st["tracker"] = PersonTracker()
    if st.get("trajectory_store") is None:
        from inference.trajectory_store import TrajectoryStore

        st["trajectory_store"] = TrajectoryStore(max_history=90)
    frame_index = int(st.get("frame_index", 0))
    fps = float(st.get("fps", 30.0))
    timestamp_sec = frame_index / max(fps, 1e-6)
    st["frame_index"] = frame_index + 1

    det = st["detector"]
    trk = st["tracker"]
    store = st["trajectory_store"]

    dets = det.detect(frame_rgb)
    tracked = trk.update(frame_rgb, dets)
    store.update(tracked, timestamp_sec)

    trajectories = store.get_all_histories()
    vis = draw_tracked_people(frame_rgb, tracked, trajectories=trajectories)

    out_json = {
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 4),
        "tracked_people": [
            {"track_id": t.track_id, "bbox": list(t.bbox), "confidence": round(t.confidence, 4)}
            for t in tracked
        ],
        "trajectories": trajectories,
    }
    return vis, out_json, st
