"""Старт/стоп live-воркеров камер (in-memory)."""

from __future__ import annotations

import logging
import threading
import uuid

from sqlalchemy.orm import Session

from api.db.models import Camera
from api.schemas.camera import CameraPatch
from api.services import camera_service
from camera.camera_worker import CameraWorker
from camera.live_pipeline import LiveFramePipeline, build_default_pipeline
from camera.stream_hub import StreamHub

logger = logging.getLogger(__name__)


class CameraManager:
    """Хранит активные ``CameraWorker`` в памяти."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_workers: dict[str, CameraWorker] = {}
        self.hub = StreamHub()

    def is_running(self, camera_id: str) -> bool:
        with self._lock:
            w = self.active_workers.get(str(camera_id))
        return bool(w and w.is_running())

    def list_running(self) -> list[str]:
        with self._lock:
            return sorted([cid for cid, w in self.active_workers.items() if w.is_running()])

    def get_worker(self, camera_id: str) -> CameraWorker | None:
        with self._lock:
            return self.active_workers.get(str(camera_id))

    def _patch_status(self, cam_uuid: uuid.UUID, status: str) -> None:
        from api.db.session import get_session_factory

        SessionLocal = get_session_factory()
        db = SessionLocal()
        try:
            camera_service.patch_camera(db, cam_uuid, CameraPatch(status=status))
        except Exception:
            logger.exception("DB camera status update failed")
        finally:
            db.close()

    def _on_worker_finished(self, camera_id: str) -> None:
        with self._lock:
            self.active_workers.pop(str(camera_id), None)

    def start_camera(
        self,
        db: Session,
        camera: Camera,
        *,
        pipeline: LiveFramePipeline | None = None,
        target_fps: float = 4.0,
    ) -> CameraWorker:
        cid = str(camera.id)
        cam_uuid = camera.id
        if not (camera.rtsp_url or "").strip():
            raise ValueError("camera has no rtsp_url")

        try:
            from configs.load import get_phase1_config

            cam_cfg = get_phase1_config().get("camera") or {}
            tf = float(cam_cfg.get("target_fps", target_fps))
            reconnect = float(cam_cfg.get("reconnect_delay_sec", 5.0))
            max_fail = int(cam_cfg.get("max_failures", 5))
        except Exception:
            tf, reconnect, max_fail = target_fps, 5.0, 5

        with self._lock:
            existing = self.active_workers.get(cid)
            if existing is not None and existing.is_running():
                return existing

            sink = self.hub.sink(cid)
            incident_key = (camera.external_key or cid).strip()

            def on_status(st: str) -> None:
                self._patch_status(cam_uuid, st)

            worker = CameraWorker(
                cid,
                str(camera.rtsp_url).strip(),
                pipeline or build_default_pipeline(),
                sink,
                camera_key_for_incidents=incident_key,
                target_fps=tf,
                reconnect_delay_sec=reconnect,
                max_rtsp_failures=max_fail,
                on_runtime_status=on_status,
                on_finished=self._on_worker_finished,
            )
            self.active_workers[cid] = worker

        logger.info("camera_started", extra={"camera_id": cid})
        try:
            camera_service.patch_camera(db, camera.id, CameraPatch(status="connecting"))
        except Exception:
            logger.exception("initial status patch failed")
        worker.start()
        return worker

    def stop_camera(self, camera_id: uuid.UUID) -> bool:
        cid = str(camera_id)
        with self._lock:
            w = self.active_workers.get(cid)
        if w is None:
            return False
        w.stop()
        with self._lock:
            self.active_workers.pop(cid, None)
        self._patch_status(camera_id, "online")
        return True

    def restart_camera(self, db: Session, camera: Camera, *, pipeline: LiveFramePipeline | None = None) -> CameraWorker:
        self.stop_camera(camera.id)
        return self.start_camera(db, camera, pipeline=pipeline)

    def shutdown_all(self) -> None:
        with self._lock:
            items = list(self.active_workers.items())
        for _, w in items:
            try:
                w.stop()
            except Exception:
                logger.exception("worker stop failed")
        with self._lock:
            self.active_workers.clear()


_manager: CameraManager | None = None
_manager_lock = threading.Lock()


def get_camera_manager() -> CameraManager:
    global _manager  # noqa: PLW0603
    with _manager_lock:
        if _manager is None:
            _manager = CameraManager()
        return _manager


def reset_camera_manager() -> None:
    """Только для тестов: остановить воркеры и сбросить singleton."""
    global _manager  # noqa: PLW0603
    with _manager_lock:
        if _manager is not None:
            _manager.shutdown_all()
            _manager = None
