"""Потокобезопасность ``CameraLiveState`` (Phase 1)."""

from __future__ import annotations

import threading
import time

from camera.live_state import CameraLiveState


def test_live_state_threadsafe_overlay_and_metrics() -> None:
    st = CameraLiveState("cam-1")
    errors: list[Exception] = []

    def writer():
        try:
            for i in range(200):
                st.publish_overlay(
                    {
                        "camera_id": "cam-1",
                        "timestamp": "t",
                        "people": [],
                        "poses": [],
                        "signals": [],
                        "risk": {"score": 0.1 + (i % 10) * 0.01, "level": "green"},
                    }
                )
                st.record_dropped_frame()
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(500):
                st.snapshot()
                st.snapshot_metrics()
                time.sleep(0.0005)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert not errors
    m = st.snapshot_metrics()
    assert m["dropped_frames"] == 200
    assert m["camera_id"] == "cam-1"
