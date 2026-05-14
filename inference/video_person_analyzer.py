"""
Выборка кадров из видеофайла + детекция людей (Этап 1, без трекинга).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inference.person_detector import PersonDetection, PersonDetector, draw_person_boxes

logger = logging.getLogger(__name__)


def analyze_video_people(
    video_path: str,
    sample_every_sec: float = 1.0,
    max_frames: int = 30,
    confidence_threshold: float = 0.35,
    model_name: str = "yolov8n.pt",
    device: str | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """
    Анализирует видео, равномерно (по времени) берёт до max_frames кадров.

    Возвращает ``(payload, preview_rgb)`` — payload JSON-сериализуемый; preview — один кадр
    с лучшим охватом людей (bbox) или None.

    При ошибке payload содержит ключ ``error``.
    """
    err_preview: np.ndarray | None = None
    path = Path(video_path)
    if not path.is_file():
        return (
            {
                "error": f"файл не найден: {video_path}",
                "video_path": str(video_path),
                "frames_analyzed": 0,
                "summary": {"max_people": 0, "avg_people": 0.0},
                "frames": [],
            },
            err_preview,
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return (
            {
                "error": f"не удалось открыть видео: {video_path}",
                "video_path": str(video_path),
                "frames_analyzed": 0,
                "summary": {"max_people": 0, "avg_people": 0.0},
                "frames": [],
            },
            err_preview,
        )

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-6:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        interval_frames = max(1, int(round(sample_every_sec * fps)))

        detector = PersonDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
        )

        frames_out: list[dict[str, Any]] = []
        analyzed = 0
        idx = 0
        max_people = 0
        total_people = 0

        while analyzed < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            dets = detector.detect(frame_rgb)
            n = len(dets)
            max_people = max(max_people, n)
            total_people += n
            timestamp_sec = idx / fps if fps > 0 else 0.0

            frames_out.append(
                {
                    "frame_index": int(idx),
                    "timestamp_sec": round(float(timestamp_sec), 4),
                    "people_count": n,
                    "detections": [d.to_dict() for d in dets],
                }
            )
            analyzed += 1
            idx += interval_frames
            if frame_count > 0 and idx >= frame_count:
                break

        avg_people = (total_people / analyzed) if analyzed else 0.0

        # Превью: кадр с максимальным числом людей (последний такой снимок)
        preview_idx = 0
        best_n = -1
        for i, block in enumerate(frames_out):
            if block["people_count"] >= best_n:
                best_n = block["people_count"]
                preview_idx = i
        preview_rgb: np.ndarray | None = None
        if frames_out:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames_out[preview_idx]["frame_index"])
            ok, fb = cap.read()
            if ok and fb is not None:
                fr = cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)
                dets_preview: list[PersonDetection] = []
                for block in frames_out[preview_idx]["detections"]:
                    bb = block["bbox"]
                    dets_preview.append(
                        PersonDetection(
                            bbox=(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])),
                            confidence=float(block["confidence"]),
                            class_id=int(block.get("class_id", 0)),
                            label=str(block.get("label", "person")),
                        )
                    )
                preview_rgb = draw_person_boxes(fr, dets_preview)

        return (
            {
                "video_path": str(path.resolve()),
                "frames_analyzed": analyzed,
                "summary": {
                    "max_people": int(max_people),
                    "avg_people": round(float(avg_people), 3),
                },
                "frames": frames_out,
            },
            preview_rgb,
        )
    except Exception as exc:
        logger.exception("analyze_video_people failed")
        return (
            {
                "error": str(exc),
                "video_path": str(video_path),
                "frames_analyzed": 0,
                "summary": {"max_people": 0, "avg_people": 0.0},
                "frames": [],
            },
            err_preview,
        )
    finally:
        cap.release()
