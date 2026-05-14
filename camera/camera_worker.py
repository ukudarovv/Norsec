"""Поток воркера: RTSP → pipeline → overlay / MJPEG / инциденты (Phase 1)."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from camera.live_pipeline import LiveFramePipeline, build_default_pipeline
from camera.live_state import CameraLiveState
from camera.rtsp_reader import RTSPReader
from camera.stream_status import normalize_status

logger = logging.getLogger(__name__)


class CameraWorker:
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        pipeline: LiveFramePipeline | None,
        sink: CameraLiveState,
        *,
        camera_key_for_incidents: str,
        target_fps: float = 4.0,
        reconnect_delay_sec: float = 5.0,
        max_rtsp_failures: int = 5,
        on_runtime_status: Callable[[str], None] | None = None,
        on_finished: Callable[[str], None] | None = None,
    ) -> None:
        self.camera_id = str(camera_id)
        self.rtsp_url = str(rtsp_url)
        self.pipeline = pipeline or build_default_pipeline()
        self.sink = sink
        self.camera_key_for_incidents = str(camera_key_for_incidents)
        self.target_fps = max(0.5, float(target_fps))
        self._on_runtime_status = on_runtime_status
        self._on_finished = on_finished

        def _on_rtsp_err(msg: str) -> None:
            if msg:
                sink.set_last_error(msg)

        self._reader = RTSPReader(
            self.rtsp_url,
            reconnect_delay_sec=float(reconnect_delay_sec),
            max_failures=int(max_rtsp_failures),
            on_error=_on_rtsp_err,
        )
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._stop = threading.Event()

    def is_running(self) -> bool:
        return self._running.is_set() and self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.sink.reset_counters()
        self._stop.clear()
        self._running.set()
        logger.info("camera_worker_starting", extra={"camera_id": self.camera_id})
        self._thread = threading.Thread(target=self._run_loop, name=f"cam-worker-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        logger.info("camera_worker_stopping", extra={"camera_id": self.camera_id})
        self._stop.set()
        self._reader.stop()
        if self._thread is not None:
            self._thread.join(timeout=20.0)
        self._thread = None
        self._running.clear()
        logger.info("camera_worker_stopped", extra={"camera_id": self.camera_id})

    def _emit_status(self, status: str) -> None:
        st = normalize_status(status)
        self.sink.set_camera_status(st)
        if self._on_runtime_status:
            try:
                self._on_runtime_status(st)
            except Exception:
                logger.exception(
                    "runtime_status_callback_failed",
                    extra={"camera_id": self.camera_id},
                )

    def _persist_candidate(self, candidate: Any) -> None:
        if not os.environ.get("DATABASE_URL"):
            return
        try:
            from api.services.incident_write_bridge import persist_incident_candidate

            persist_incident_candidate(candidate, self.camera_key_for_incidents)
        except Exception:
            logger.exception(
                "persist_incident_failed",
                extra={"camera_id": self.camera_id},
            )

    def _run_loop(self) -> None:
        self._emit_status("connecting")
        t0 = time.monotonic()
        last_proc = 0.0
        min_interval = 1.0 / self.target_fps
        had_frame = False

        try:
            for frame_bgr in self._reader.read_frames():
                if self._stop.is_set():
                    break
                had_frame = True
                self.sink.record_frame_received()
                self._emit_status("analyzing")
                now = time.monotonic()
                if now - last_proc < min_interval:
                    self.sink.record_dropped_frame()
                    continue
                last_proc = now
                self.sink.record_processed_frame(now)
                ts = now - t0
                try:
                    overlay, annotated, candidate = self.pipeline.process(frame_bgr, ts, self.camera_id)
                    self.sink.publish_overlay(overlay)
                    if annotated is not None:
                        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                        if ok:
                            self.sink.publish_jpeg(buf.tobytes())
                    if candidate is not None:
                        self._persist_candidate(candidate)
                except Exception as e:
                    self.sink.set_last_error(f"pipeline: {e!s}")
                    logger.error(
                        "pipeline_frame_failed",
                        extra={"camera_id": self.camera_id, "error": str(e)},
                        exc_info=True,
                    )
                    self._emit_status("error")

            if not had_frame and not self._stop.is_set():
                self._emit_status("error")
            elif self._stop.is_set():
                self._emit_status("online")
            else:
                self._emit_status("offline")
        except Exception as e:
            self.sink.set_last_error(f"worker_fatal: {e!s}")
            logger.error(
                "camera_worker_fatal",
                extra={"camera_id": self.camera_id, "error": str(e)},
                exc_info=True,
            )
            self._emit_status("error")
        finally:
            self._running.clear()
            if self._on_finished:
                try:
                    self._on_finished(self.camera_id)
                except Exception:
                    logger.exception("on_finished_failed", extra={"camera_id": self.camera_id})
