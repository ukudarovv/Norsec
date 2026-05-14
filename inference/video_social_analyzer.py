"""
Видео: детекция → трекинг → траектории → социальные сигналы (этап 3).
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
from inference.social_interaction import (
    SocialInteractionAnalyzer,
    compute_distance_matrix,
    draw_social_signals,
)
from inference.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)


def _histories_int_keys(store: TrajectoryStore) -> dict[int, list]:
    raw = store.get_all_histories()
    return {int(k): v for k, v in raw.items()}


def analyze_video_social(
    video_path: str,
    sample_every_sec: float = 0.2,
    max_frames: int = 300,
    confidence_threshold: float = 0.35,
    model_name: str = "yolov8n.pt",
    device: str | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """
    Полный pipeline по видео. Возвращает (payload, preview_rgb).

    payload: ``summary`` (signals_count, max_social_severity, signal_types), ``signals`` — список dict.
    """
    err_preview: np.ndarray | None = None
    empty_summary = {
        "signals_count": 0,
        "max_social_severity": 0.0,
        "signal_types": {},
    }
    path = Path(video_path)
    if not path.is_file():
        return (
            {
                "error": f"файл не найден: {video_path}",
                "video_path": str(video_path),
                "summary": empty_summary,
                "signals": [],
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
                "signals": [],
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
        social = SocialInteractionAnalyzer()

        all_signals: list[dict[str, Any]] = []
        per_frame_meta: list[tuple[int, int, list[TrackedPerson], dict[str, list], list]] = []
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
            traj_int = _histories_int_keys(store)
            frame_sigs = social.update(tracked, traj_int, float(timestamp_sec))
            for s in frame_sigs:
                all_signals.append(s.to_dict())
            tracks_str = store.get_all_histories()
            per_frame_meta.append((idx, len(frame_sigs), tracked, tracks_str, list(frame_sigs)))
            analyzed += 1
            idx += interval_frames
            if frame_count > 0 and idx >= frame_count:
                break

        type_counts: Counter[str] = Counter()
        max_sev = 0.0
        for s in all_signals:
            type_counts[s["signal_type"]] += 1
            max_sev = max(max_sev, float(s.get("severity", 0.0)))

        preview_rgb: np.ndarray | None = None
        if per_frame_meta:
            best_idx, _, best_tracked, best_tracks_str, best_sigs = max(
                per_frame_meta, key=lambda x: x[1]
            )
            if max(x[1] for x in per_frame_meta) == 0:
                best_idx, _, best_tracked, best_tracks_str, best_sigs = per_frame_meta[-1]
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
            ok, fb = cap.read()
            if ok and fb is not None:
                fr = cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)
                vis = draw_tracked_people(fr, best_tracked, trajectories=best_tracks_str)
                preview_rgb = draw_social_signals(vis, best_tracked, best_sigs)

        payload: dict[str, Any] = {
            "video_path": str(path.resolve()),
            "frames_analyzed": analyzed,
            "summary": {
                "signals_count": len(all_signals),
                "max_social_severity": round(float(max_sev), 4),
                "signal_types": dict(sorted(type_counts.items())),
            },
            "signals": all_signals,
        }
        return payload, preview_rgb
    except Exception as exc:
        logger.exception("analyze_video_social failed")
        return (
            {
                "error": str(exc),
                "video_path": str(video_path),
                "summary": empty_summary,
                "signals": [],
            },
            err_preview,
        )
    finally:
        cap.release()


def analyze_live_frame_social(
    frame_rgb: np.ndarray,
    state: dict,
    *,
    model_name: str = "yolov8n.pt",
    device: str | None = None,
    confidence: float = 0.35,
) -> tuple[np.ndarray, dict, dict]:
    """Один кадр: detector, tracker, trajectory_store, social_analyzer, frame_index, last_timestamp в state."""
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
        st["trajectory_store"] = TrajectoryStore(max_history=5000)
    if st.get("social_analyzer") is None:
        st["social_analyzer"] = SocialInteractionAnalyzer()

    frame_index = int(st.get("frame_index", 0))
    fps = float(st.get("fps", 30.0))
    timestamp_sec = frame_index / max(fps, 1e-6)
    st["last_timestamp"] = float(timestamp_sec)
    st["frame_index"] = frame_index + 1

    det = st["detector"]
    trk = st["tracker"]
    store = st["trajectory_store"]
    soc = st["social_analyzer"]

    dets = det.detect(frame_rgb)
    tracked = trk.update(frame_rgb, dets)
    store.update(tracked, float(timestamp_sec))
    traj_int = _histories_int_keys(store)
    signals = soc.update(tracked, traj_int, float(timestamp_sec))

    tracks_str = store.get_all_histories()
    vis = draw_tracked_people(frame_rgb, tracked, trajectories=tracks_str)
    vis = draw_social_signals(vis, tracked, signals)

    out_json = {
        "frame_index": frame_index,
        "timestamp_sec": round(float(timestamp_sec), 4),
        "last_timestamp": round(float(st["last_timestamp"]), 4),
        "tracked_people": [
            {"track_id": t.track_id, "bbox": list(t.bbox), "confidence": round(t.confidence, 4)}
            for t in tracked
        ],
        "social_signals": [s.to_dict() for s in signals],
        "distance_matrix": compute_distance_matrix(tracked),
    }
    return vis, out_json, st
