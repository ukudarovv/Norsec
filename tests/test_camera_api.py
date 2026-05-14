"""HTTP/WebSocket для live-камер (этап 10)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from camera.rtsp_reader import RTSPReader


def _fake_frames(self: RTSPReader):
    while not self.is_stopped():
        yield np.zeros((48, 64, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _stub_rtsp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RTSPReader, "read_frames", _fake_frames)
    monkeypatch.setenv("LIVE_STUB_FUSE_EVERY", "1")


def test_viewer_cannot_start_stop(client: TestClient, admin_headers: dict[str, str], viewer_headers: dict[str, str]) -> None:
    r = client.post(
        "/api/cameras",
        headers=admin_headers,
        json={"name": "live1", "rtsp_url": "rtsp://example/stream"},
    )
    assert r.status_code == 201, r.text
    cid = r.json()["id"]
    s = client.post(f"/api/cameras/{cid}/start", headers=viewer_headers)
    assert s.status_code == 403
    p = client.post(f"/api/cameras/{cid}/stop", headers=viewer_headers)
    assert p.status_code == 403


def test_operator_can_start_list_running_stop(
    client: TestClient, admin_headers: dict[str, str], operator_headers: dict[str, str]
) -> None:
    r = client.post(
        "/api/cameras",
        headers=admin_headers,
        json={"name": "live2", "rtsp_url": "rtsp://example/stream2"},
    )
    assert r.status_code == 201, r.text
    cid = r.json()["id"]
    st = client.post(f"/api/cameras/{cid}/start", headers=operator_headers)
    assert st.status_code == 200, st.text
    time.sleep(0.35)
    ru = client.get("/api/cameras/running", headers=operator_headers)
    assert ru.status_code == 200
    assert cid in ru.json().get("cameras", [])
    sp = client.post(f"/api/cameras/{cid}/stop", headers=operator_headers)
    assert sp.status_code == 200


def test_ws_overlay_payload_shape(
    client: TestClient, admin_headers: dict[str, str], operator_headers: dict[str, str]
) -> None:
    r = client.post(
        "/api/cameras",
        headers=admin_headers,
        json={"name": "live3", "rtsp_url": "rtsp://example/stream3"},
    )
    cid = r.json()["id"]
    client.post(f"/api/cameras/{cid}/start", headers=operator_headers)
    time.sleep(0.25)
    tok = client.post("/api/auth/login", json={"email": "admin@example.com", "password": "secret12345"}).json()[
        "access_token"
    ]
    with client.websocket_connect(f"/ws/cameras/{cid}/overlay?token={tok}") as ws:
        msg = ws.receive_json()
        assert msg.get("camera_id") == cid
        assert isinstance(msg.get("people"), list)
        assert isinstance(msg.get("poses"), list)
        assert isinstance(msg.get("signals"), list)
        assert isinstance(msg.get("risk"), dict)
        assert "score" in msg["risk"] and "level" in msg["risk"]
    client.post(f"/api/cameras/{cid}/stop", headers=operator_headers)


def test_incident_persisted_when_db_enabled(
    client: TestClient,
    admin_headers: dict[str, str],
    operator_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fast_fusion_cfg() -> dict:
        return {
            "fusion": {
                "incident_threshold": 0.55,
                "min_persistence_sec": 0.05,
                "cooldown_sec": 30.0,
                "hysteresis_alpha": 1.0,
            },
            "camera": {"target_fps": 15},
        }

    monkeypatch.setattr("configs.load.get_phase1_config", _fast_fusion_cfg)

    r = client.post(
        "/api/cameras",
        headers=admin_headers,
        json={"name": "live4", "rtsp_url": "rtsp://example/stream4"},
    )
    cid = r.json()["id"]
    before = client.get("/api/incidents", headers=admin_headers).json()
    n0 = len(before) if isinstance(before, list) else 0
    client.post(f"/api/cameras/{cid}/start", headers=operator_headers)
    time.sleep(0.8)
    after = client.get("/api/incidents", headers=admin_headers).json()
    n1 = len(after) if isinstance(after, list) else 0
    assert n1 > n0
    client.post(f"/api/cameras/{cid}/stop", headers=operator_headers)


def test_camera_status_includes_live_metrics(
    client: TestClient, admin_headers: dict[str, str], operator_headers: dict[str, str]
) -> None:
    r = client.post(
        "/api/cameras",
        headers=admin_headers,
        json={"name": "live6", "rtsp_url": "rtsp://example/stream6"},
    )
    cid = r.json()["id"]
    client.post(f"/api/cameras/{cid}/start", headers=operator_headers)
    time.sleep(0.35)
    st = client.get(f"/api/cameras/{cid}/status", headers=operator_headers)
    assert st.status_code == 200
    live = st.json().get("live") or {}
    assert "fps_estimate" in live
    assert "dropped_frames" in live
    assert "worker_running" in live
    client.post(f"/api/cameras/{cid}/stop", headers=operator_headers)


def test_test_connection_fast_mock(
    client: TestClient,
    admin_headers: dict[str, str],
    operator_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.services import camera_live_service

    monkeypatch.setattr(camera_live_service, "test_rtsp_connection", lambda *_a, **_k: (False, "cannot_open"))

    r = client.post(
        "/api/cameras",
        headers=admin_headers,
        json={"name": "live5", "rtsp_url": "rtsp://bad"},
    )
    cid = r.json()["id"]
    t = client.post(f"/api/cameras/{cid}/test-connection", headers=operator_headers)
    assert t.status_code == 200
    body = t.json()
    assert body.get("ok") is False
