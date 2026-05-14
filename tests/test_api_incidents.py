"""Инциденты через HTTP + PostgreSQL/SQLite (этап 9)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.db.session import get_session_factory
from api.services import camera_service, incident_service
from api.schemas.camera import CameraCreate
from fusion.incident_candidate import IncidentCandidate


def _seed() -> str:
    s = get_session_factory()()
    try:
        camera_service.create_camera(s, CameraCreate(name="cam", external_key="cam_a"))
        cand = IncidentCandidate(
            camera_id="cam_a",
            start_sec=1.0,
            end_sec=5.0,
            risk_score=0.82,
            risk_level="red",
            signal_types=["punch"],
            involved_track_ids=[1],
            explanation=["Detected physical action signal: punch"],
            evidence={"action_signals": [], "social_signals": []},
        )
        inc = incident_service.create_from_candidate(s, cand, "cam_a")
        return str(inc.id)
    finally:
        s.close()


def test_list_incidents(client: TestClient, admin_headers: dict[str, str]) -> None:
    _seed()
    r = client.get("/api/incidents", headers=admin_headers)
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["risk_level"] == "red"


def test_list_filter_camera(client: TestClient, admin_headers: dict[str, str]) -> None:
    _seed()
    r = client.get("/api/incidents", headers=admin_headers, params={"camera_id": "cam_a"})
    assert len(r.json()) == 1
    r2 = client.get("/api/incidents", headers=admin_headers, params={"camera_id": "other"})
    assert r2.json() == []


def test_get_incident(client: TestClient, admin_headers: dict[str, str]) -> None:
    iid = _seed()
    r = client.get(f"/api/incidents/{iid}", headers=admin_headers)
    assert r.status_code == 200


def test_get_incident_404(client: TestClient, admin_headers: dict[str, str]) -> None:
    import uuid

    missing = str(uuid.uuid4())
    assert client.get(f"/api/incidents/{missing}", headers=admin_headers).status_code == 404


def test_post_review(client: TestClient, admin_headers: dict[str, str], reviewer_headers: dict[str, str]) -> None:
    iid = _seed()
    r = client.post(
        f"/api/incidents/{iid}/review",
        headers=reviewer_headers,
        json={"status": "false_positive", "comment": "looks ok"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["review_status"] == "false_positive"


def test_post_review_invalid_status(client: TestClient, admin_headers: dict[str, str], reviewer_headers: dict[str, str]) -> None:
    iid = _seed()
    r = client.post(
        f"/api/incidents/{iid}/review",
        headers=reviewer_headers,
        json={"status": "not_a_status"},
    )
    assert r.status_code == 400


def test_cameras(client: TestClient, admin_headers: dict[str, str]) -> None:
    _seed()
    r = client.get("/api/cameras", headers=admin_headers)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert any((c.get("external_key") == "cam_a") for c in data)


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}
