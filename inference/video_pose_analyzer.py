"""
Видео: детекция → трекинг → поза → pose signals (этап 4).
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inference.person_detector import PersonDetector
from inference.person_tracker import PersonTracker, TrackedPerson, draw_tracked_people
from inference.pose_estimator import PoseEstimator, draw_poses
from inference.pose_signals import PoseSignalAnalyzer
from inference.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)


def analyze_video_pose(
    video_path: str,
    sample_every_sec: float = 0.2,
    max_frames: int = 300,
    confidence_threshold: float = 0.35,
    detector_model: str = "yolov8n.pt",
    pose_model: str = "yolov8n-pose.pt",
    device: str | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    err_preview: np.ndarray | None = None
    empty_summary: dict[str, Any] = {
        "frames_analyzed": 0,
        "pose_signals_count": 0,
        "max_pose_severity": 0.0,
        "signal_types": {},
    }
    path = Path(video_path)
    if not path.is_file():
        return (
            {
                "error": f"файл не найден: {video_path}",
                "video_path": str(video_path),
                "summary": empty_summary,
                "frames": [],
                "pose_signals": [],
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
                "summary": empty_summary,
                "frames": [],
                "pose_signals": [],
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
            model_name=detector_model,
            confidence_threshold=float(confidence_threshold),
            device=device,
        )
        tracker = PersonTracker()
        store = TrajectoryStore(max_history=5000)
        pose_est = PoseEstimator(
            model_name=pose_model,
            confidence_threshold=float(confidence_threshold),
            device=device,
        )
        pose_an = PoseSignalAnalyzer()

        all_pose_signals: list[dict[str, Any]] = []
        frames_out: list[dict[str, Any]] = []
        per_frame_meta: list[tuple[int, int, list[TrackedPerson], list, list]] = []
        analyzed = 0
        idx = 0

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
            poses = pose_est.estimate(frame_rgb, tracked)
            p_sigs = pose_an.update(poses, float(timestamp_sec))
            for s in p_sigs:
                all_pose_signals.append(s.to_dict())

            frames_out.append(
                {
                    "frame_index": int(idx),
                    "timestamp_sec": round(float(timestamp_sec), 4),
                    "tracked_people": [
                        {
                            "track_id": t.track_id,
                            "bbox": [t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]],
                            "confidence": round(t.confidence, 4),
                        }
                        for t in tracked
                    ],
                    "poses": [p.to_dict() for p in poses],
                    "pose_signals": [s.to_dict() for s in p_sigs],
                }
            )
            per_frame_meta.append((idx, len(p_sigs), tracked, poses, p_sigs))
            analyzed += 1
            idx += interval_frames
            if frame_count > 0 and idx >= frame_count:
                break

        type_counts: Counter[str] = Counter(s["signal_type"] for s in all_pose_signals)
        max_sev = max((float(s.get("severity", 0.0)) for s in all_pose_signals), default=0.0)

        preview_rgb: np.ndarray | None = None
        if per_frame_meta:
            best_idx, _, best_tracked, best_poses, _ = max(per_frame_meta, key=lambda x: x[1])
            if max(x[1] for x in per_frame_meta) == 0:
                best_idx, _, best_tracked, best_poses, _ = per_frame_meta[-1]
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
            ok, fb = cap.read()
            if ok and fb is not None:
                fr = cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)
                tracks_str = store.get_all_histories()
                vis = draw_tracked_people(fr, best_tracked, trajectories=tracks_str)
                preview_rgb = draw_poses(vis, best_poses)

        payload: dict[str, Any] = {
            "video_path": str(path.resolve()),
            "summary": {
                "frames_analyzed": analyzed,
                "pose_signals_count": len(all_pose_signals),
                "max_pose_severity": round(float(max_sev), 4),
                "signal_types": dict(sorted(type_counts.items())),
            },
            "frames": frames_out,
            "pose_signals": all_pose_signals,
        }
        return payload, preview_rgb
    except Exception as exc:
        logger.exception("analyze_video_pose failed")
        return (
            {
                "error": str(exc),
                "video_path": str(video_path),
                "summary": empty_summary,
                "frames": [],
                "pose_signals": [],
            },
            err_preview,
        )
    finally:
        cap.release()


def analyze_live_frame_pose(
    frame_rgb: np.ndarray,
    state: dict,
    *,
    detector_model: str = "yolov8n.pt",
    pose_model: str = "yolov8n-pose.pt",
    device: str | None = None,
    confidence: float = 0.35,
) -> tuple[np.ndarray, dict, dict]:
    if state is None:
        state = {}
    st = dict(state)

    if st.get("detector") is None:
        st["detector"] = PersonDetector(
            model_name=detector_model,
            device=device,
            confidence_threshold=confidence,
        )
    if st.get("tracker") is None:
        st["tracker"] = PersonTracker()
    if st.get("trajectory_store") is None:
        st["trajectory_store"] = TrajectoryStore(max_history=5000)
    if st.get("pose_estimator") is None:
        st["pose_estimator"] = PoseEstimator(
            model_name=pose_model,
            confidence_threshold=confidence,
            device=device,
        )
    if st.get("pose_signal_analyzer") is None:
        st["pose_signal_analyzer"] = PoseSignalAnalyzer()

    frame_index = int(st.get("frame_index", 0))
    fps = float(st.get("fps", 30.0))
    timestamp_sec = frame_index / max(fps, 1e-6)
    st["last_timestamp"] = float(timestamp_sec)
    st["frame_index"] = frame_index + 1

    det = st["detector"]
    trk = st["tracker"]
    store = st["trajectory_store"]
    pe = st["pose_estimator"]
    ps = st["pose_signal_analyzer"]

    dets = det.detect(frame_rgb)
    tracked = trk.update(frame_rgb, dets)
    store.update(tracked, float(timestamp_sec))
    poses = pe.estimate(frame_rgb, tracked)
    sigs = ps.update(poses, float(timestamp_sec))

    tracks_str = store.get_all_histories()
    vis = draw_tracked_people(frame_rgb, tracked, trajectories=tracks_str)
    vis = draw_poses(vis, poses)

    out_json: dict[str, Any] = {
        "frame_index": frame_index,
        "timestamp_sec": round(float(timestamp_sec), 4),
        "last_timestamp": round(float(st["last_timestamp"]), 4),
        "tracked_people": [
            {"track_id": t.track_id, "bbox": list(t.bbox), "confidence": round(t.confidence, 4)}
            for t in tracked
        ],
        "poses": [p.to_dict() for p in poses],
        "pose_signals": [s.to_dict() for s in sigs],
    }
    return vis, out_json, st
