"""CameraWorker: graceful stop + dropped frames (мок RTSP)."""

from __future__ import annotations

import time as _t

import numpy as np
import pytest
from sqlalchemy.orm import Session

from api.schemas.camera import CameraCreate
from api.services import camera_service
from camera.camera_worker import CameraWorker
from camera.live_pipeline import StubLivePipeline
from camera.rtsp_reader import RTSPReader
from camera.stream_hub import StreamHub


def _fake_frames(self: RTSPReader):
    n = 0
    while not self.is_stopped() and n < 80:
        yield np.zeros((32, 48, 3), dtype=np.uint8)
        n += 1


def test_worker_stop_joins_and_updates_live_state(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RTSPReader, "read_frames", _fake_frames)
    cam = camera_service.create_camera(
        db_session,
        CameraCreate(name="w1", rtsp_url="rtsp://fake", external_key="w1"),
    )
    hub = StreamHub()
    sink = hub.sink(str(cam.id))
    pipe = StubLivePipeline(fuse_every=99)
    worker = CameraWorker(
        str(cam.id),
        str(cam.rtsp_url),
        pipe,
        sink,
        camera_key_for_incidents=str(cam.id),
        target_fps=0.5,
        reconnect_delay_sec=0.01,
        max_rtsp_failures=3,
    )
    worker.start()
    _t.sleep(0.2)
    assert sink.snapshot_metrics()["dropped_frames"] >= 5
    worker.stop()
    assert worker.is_running() is False
