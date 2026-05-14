"""
Видеофайл: детекция + ByteTrack + траектории (этап 2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inference.person_detector import PersonDetector
from inference.person_tracker import PersonTracker, TrackedPerson, draw_tracked_people
from inference.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)


def analyze_video_tracking(
    video_path: str,
    sample_every_sec: float = 0.2,
    max_frames: int = 150,
    confidence_threshold: float = 0.35,
    model_name: str = "yolov8n.pt",
    device: str | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """
    Возвращает (payload, preview_rgb). payload — JSON-совместимый dict по спецификации этапа 2.
    """
    err_preview: np.ndarray | None = None
    path = Path(video_path)
    if not path.is_file():
        return (
            {
                "error": f"файл не найден: {video_path}",
                "video_path": str(video_path),
                "frames_analyzed": 0,
                "summary": {
                    "unique_people": 0,
                    "max_people_in_frame": 0,
                    "avg_people_in_frame": 0.0,
                },
                "tracks": {},
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
                "summary": {
                    "unique_people": 0,
                    "max_people_in_frame": 0,
                    "avg_people_in_frame": 0.0,
                },
                "tracks": {},
                "frames": [],
            },
            err_preview,
        )

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-6:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        interval_frames = max(1, int(round(float(sample_every_sec) * fps)))

        detector = PersonDetector(
            model_name=model_name,
            confidence_threshold=float(confidence_threshold),
            device=device,
        )
        tracker = PersonTracker()
        store = TrajectoryStore(max_history=5000)

        frames_out: list[dict[str, Any]] = []
        analyzed = 0
        idx = 0
        max_in_frame = 0
        total_in_frame = 0
        all_ids: set[int] = set()

        while analyzed < int(max_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            dets = detector.detect(frame_rgb)
            tracked = tracker.update(frame_rgb, dets)
            timestamp_sec = idx / fps if fps > 0 else 0.0
            store.update(tracked, float(timestamp_sec))

            for t in tracked:
                all_ids.add(t.track_id)

            n = len(tracked)
            max_in_frame = max(max_in_frame, n)
            total_in_frame += n

            frames_out.append(
                {
                    "frame_index": int(idx),
                    "timestamp_sec": round(float(timestamp_sec), 4),
                    "people_count": n,
                    "tracked_people": [
                        {
                            "track_id": t.track_id,
                            "bbox": [t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]],
                            "confidence": round(t.confidence, 4),
                        }
                        for t in tracked
                    ],
                }
            )
            analyzed += 1
            idx += interval_frames
            if frame_count > 0 and idx >= frame_count:
                break

        avg_in_frame = (total_in_frame / analyzed) if analyzed else 0.0
        tracks = store.get_all_histories()

        # Превью: кадр с максимальным people_count с отрисовкой треков на этом кадре
        preview_rgb: np.ndarray | None = None
        if frames_out:
            best_i = max(range(len(frames_out)), key=lambda i: frames_out[i]["people_count"])
            fi = frames_out[best_i]["frame_index"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fb = cap.read()
            if ok and fb is not None:
                fr = cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)
                # Пересчитать детекции и один проход трека для согласованности превью —
                # проще: взять сохранённые tracked из frames_out[best_i] и нарисовать bbox;
                # траектории — из store (уже накоплены). Для линий нужны центры из истории.
                tp_list: list[TrackedPerson] = []
                for item in frames_out[best_i]["tracked_people"]:
                    bb = item["bbox"]
                    tp_list.append(
                        TrackedPerson(
                            track_id=int(item["track_id"]),
                            bbox=(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])),
                            confidence=float(item["confidence"]),
                        )
                    )
                preview_rgb = draw_tracked_people(fr, tp_list, trajectories=tracks)

        payload: dict[str, Any] = {
            "video_path": str(path.resolve()),
            "frames_analyzed": analyzed,
            "summary": {
                "unique_people": int(len(all_ids)),
                "max_people_in_frame": int(max_in_frame),
                "avg_people_in_frame": round(float(avg_in_frame), 3),
            },
            "tracks": tracks,
            "frames": frames_out,
        }
        return payload, preview_rgb
    except Exception as exc:
        logger.exception("analyze_video_tracking failed")
        return (
            {
                "error": str(exc),
                "video_path": str(video_path),
                "frames_analyzed": 0,
                "summary": {
                    "unique_people": 0,
                    "max_people_in_frame": 0,
                    "avg_people_in_frame": 0.0,
                },
                "tracks": {},
                "frames": [],
            },
            err_preview,
        )
    finally:
        cap.release()
