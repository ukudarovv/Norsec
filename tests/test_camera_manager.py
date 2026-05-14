"""CameraManager: старт/стоп воркера (мок кадров RTSP)."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy.orm import Session

from api.schemas.camera import CameraCreate
from api.services import camera_service
from camera.camera_manager import CameraManager, reset_camera_manager
from camera.rtsp_reader import RTSPReader


def _fake_frames(self: RTSPReader):
    while not self.is_stopped():
        yield np.zeros((64, 96, 3), dtype=np.uint8)


@pytest.fixture()
def manager(monkeypatch: pytest.MonkeyPatch) -> CameraManager:
    reset_camera_manager()
    monkeypatch.setattr(RTSPReader, "read_frames", _fake_frames)
    monkeypatch.setenv("LIVE_STUB_FUSE_EVERY", "2")
    return CameraManager()


def test_start_creates_worker_and_stop_stops(db_session: Session, manager: CameraManager) -> None:
    cam = camera_service.create_camera(
        db_session,
        CameraCreate(name="c1", rtsp_url="rtsp://fake/stream", external_key="ext1"),
    )
    w = manager.start_camera(db_session, cam)
    assert w.is_running() is True
    assert manager.is_running(str(cam.id)) is True
    assert manager.stop_camera(cam.id) is True
    manager.shutdown_all()
