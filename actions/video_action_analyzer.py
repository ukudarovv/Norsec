"""
Видео / live: детекция → трекинг → поза → буфер кропов → action recognition (этап 5).
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from actions.action_model import ActionRecognizer, _clip_to_tensor_uint8
from actions.action_signals import (
    ActionSignal,
    build_action_signal,
    detect_pair_interactions,
    draw_action_labels,
)
from actions.clip_buffer import ClipBuffer
from inference.person_detector import PersonDetector
from inference.person_tracker import PersonTracker, TrackedPerson, draw_tracked_people
from inference.pose_estimator import PoseEstimator, draw_poses

logger = logging.getLogger(__name__)


def _crop_track_rgb(
    frame_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad_ratio: float = 0.12,
    out_hw: Tuple[int, int] = (112, 112),
) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    h, w = frame_rgb.shape[:2]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, out_hw, interpolation=cv2.INTER_LINEAR)


def analyze_video_actions(
    video_path: str,
    sample_every_sec: float = 0.2,
    max_frames: int = 300,
    confidence_threshold: float = 0.35,
    detector_model: str = "yolov8n.pt",
    pose_model: str = "yolov8n-pose.pt",
    action_model_name: str = "heuristic",
    clip_length: int = 16,
    device: str | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    err_preview: np.ndarray | None = None
    empty = {
        "frames_analyzed": 0,
        "action_signals_count": 0,
        "max_action_severity": 0.0,
        "signal_types": {},
    }
    path = Path(video_path)
    if not path.is_file():
        return (
            {
                "error": f"файл не найден: {video_path}",
                "video_path": str(video_path),
                "summary": empty,
                "actions": [],
                "interactions": [],
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
                "summary": empty,
                "actions": [],
                "interactions": [],
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
            model_name=detector_model,
            confidence_threshold=float(confidence_threshold),
            device=device,
        )
        tracker = PersonTracker()
        pose_est = PoseEstimator(
            model_name=pose_model,
            confidence_threshold=float(confidence_threshold),
            device=device,
        )
        clip_buf = ClipBuffer(clip_length=int(clip_length))
        act_rec = ActionRecognizer(
            model_name=action_model_name,
            confidence_threshold=float(confidence_threshold),
            device=device,
        )

        all_actions: list[dict[str, Any]] = []
        frames_out: list[dict[str, Any]] = []
        per_meta: list[tuple[int, int, list[TrackedPerson], list, dict[int, Tuple[str, float]]]] = []
        analyzed = 0
        idx = 0
        last_labels: dict[int, Tuple[str, float]] = {}

        while analyzed < int(max_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            dets = detector.detect(frame_rgb)
            tracked = tracker.update(frame_rgb, dets)
            poses = pose_est.estimate(frame_rgb, tracked)
            ts = idx / fps if fps > 0 else 0.0

            frame_actions: list[dict[str, Any]] = []
            frame_sigs: list[ActionSignal] = []

            for tp in tracked:
                crop = _crop_track_rgb(frame_rgb, tp.bbox)
                if crop is not None:
                    clip_buf.update(tp.track_id, crop, idx)
                if clip_buf.is_ready(tp.track_id):
                    clip_hwc = clip_buf.get_clip(tp.track_id)
                    if clip_hwc is not None:
                        tens = _clip_to_tensor_uint8(clip_hwc)
                        pred = act_rec.predict(tens)
                        act = str(pred.get("action", "normal"))
                        conf = float(pred.get("confidence", 0.0))
                        last_labels[tp.track_id] = (act, conf)
                        if act != "normal" or conf >= float(confidence_threshold):
                            sig = build_action_signal(tp.track_id, act, conf, float(ts))
                            frame_sigs.append(sig)
                            all_actions.append(sig.to_dict())
                            frame_actions.append(sig.to_dict())

            interactions = detect_pair_interactions(frame_sigs)
            frames_out.append(
                {
                    "frame_index": int(idx),
                    "timestamp_sec": round(float(ts), 4),
                    "tracked_people": [
                        {
                            "track_id": t.track_id,
                            "bbox": [t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]],
                            "confidence": round(t.confidence, 4),
                        }
                        for t in tracked
                    ],
                    "poses": [p.to_dict() for p in poses],
                    "actions": frame_actions,
                    "interactions": interactions,
                }
            )
            per_meta.append((idx, len(frame_actions), tracked, poses, dict(last_labels)))
            analyzed += 1
            idx += interval_frames
            if frame_count > 0 and idx >= frame_count:
                break

        type_counts = Counter(a["action_type"] for a in all_actions)
        max_sev = max((float(a.get("severity", 0.0)) for a in all_actions), default=0.0)
        all_interactions: list[dict[str, Any]] = []
        for block in frames_out:
            all_interactions.extend(block.get("interactions") or [])

        preview_rgb: np.ndarray | None = None
        if per_meta:
            best_idx, _, best_tr, best_pos, labels = max(per_meta, key=lambda x: x[1])
            if max(x[1] for x in per_meta) == 0:
                best_idx, _, best_tr, best_pos, labels = per_meta[-1]
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
            ok, fb = cap.read()
            if ok and fb is not None:
                fr = cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)
                vis = draw_tracked_people(fr, best_tr, trajectories=None)
                vis = draw_poses(vis, best_pos)
                preview_rgb = draw_action_labels(vis, best_tr, labels)

        payload: dict[str, Any] = {
            "video_path": str(path.resolve()),
            "summary": {
                "frames_analyzed": analyzed,
                "action_signals_count": len(all_actions),
                "max_action_severity": round(float(max_sev), 4),
                "signal_types": dict(sorted(type_counts.items())),
            },
            "actions": all_actions,
            "interactions": all_interactions,
            "frames": frames_out,
        }
        return payload, preview_rgb
    except Exception as exc:
        logger.exception("analyze_video_actions failed")
        return (
            {
                "error": str(exc),
                "video_path": str(video_path),
                "summary": empty,
                "actions": [],
                "interactions": [],
                "frames": [],
            },
            err_preview,
        )
    finally:
        cap.release()


def analyze_live_frame_actions(
    frame_rgb: np.ndarray,
    state: dict,
    *,
    detector_model: str = "yolov8n.pt",
    pose_model: str = "yolov8n-pose.pt",
    action_model_name: str = "heuristic",
    clip_length: int = 16,
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
        from inference.trajectory_store import TrajectoryStore

        st["trajectory_store"] = TrajectoryStore(max_history=5000)
    if st.get("pose_estimator") is None:
        st["pose_estimator"] = PoseEstimator(
            model_name=pose_model,
            confidence_threshold=confidence,
            device=device,
        )
    if st.get("clip_buffer") is None:
        st["clip_buffer"] = ClipBuffer(clip_length=int(clip_length))
    if st.get("action_recognizer") is None:
        st["action_recognizer"] = ActionRecognizer(
            model_name=action_model_name,
            confidence_threshold=confidence,
            device=device,
        )

    frame_index = int(st.get("frame_index", 0))
    fps = float(st.get("fps", 30.0))
    timestamp_sec = frame_index / max(fps, 1e-6)
    st["last_timestamp"] = float(timestamp_sec)
    st["frame_index"] = frame_index + 1

    det = st["detector"]
    trk = st["tracker"]
    store = st["trajectory_store"]
    pe = st["pose_estimator"]
    buf = st["clip_buffer"]
    ar = st["action_recognizer"]
    last_labels: dict[int, Tuple[str, float]] = dict(st.get("last_action_labels") or {})

    dets = det.detect(frame_rgb)
    tracked = trk.update(frame_rgb, dets)
    store.update(tracked, float(timestamp_sec))
    poses = pe.estimate(frame_rgb, tracked)

    frame_sigs: list[ActionSignal] = []
    for tp in tracked:
        crop = _crop_track_rgb(frame_rgb, tp.bbox)
        if crop is not None:
            buf.update(tp.track_id, crop, frame_index)
        if buf.is_ready(tp.track_id):
            clip_hwc = buf.get_clip(tp.track_id)
            if clip_hwc is not None:
                pred = ar.predict(_clip_to_tensor_uint8(clip_hwc))
                act = str(pred.get("action", "normal"))
                conf = float(pred.get("confidence", 0.0))
                last_labels[tp.track_id] = (act, conf)
                if act != "normal" or conf >= confidence:
                    frame_sigs.append(build_action_signal(tp.track_id, act, conf, float(timestamp_sec)))

    st["last_action_labels"] = last_labels
    tracks_str = store.get_all_histories()
    vis = draw_tracked_people(frame_rgb, tracked, trajectories=tracks_str)
    vis = draw_poses(vis, poses)
    vis = draw_action_labels(vis, tracked, last_labels)

    interactions = detect_pair_interactions(frame_sigs)
    out_json: dict[str, Any] = {
        "frame_index": frame_index,
        "timestamp_sec": round(float(timestamp_sec), 4),
        "last_timestamp": round(float(st["last_timestamp"]), 4),
        "tracked_people": [
            {"track_id": t.track_id, "bbox": list(t.bbox), "confidence": round(t.confidence, 4)}
            for t in tracked
        ],
        "poses": [p.to_dict() for p in poses],
        "actions": [s.to_dict() for s in frame_sigs],
        "interactions": interactions,
    }
    return vis, out_json, st
