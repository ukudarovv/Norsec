from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from api.db.session import get_session_factory
from api.services import camera_service, incident_service
from api.schemas.camera import CameraCreate
from fusion.incident_candidate import IncidentCandidate


def _seed_incident() -> uuid.UUID:
    s = get_session_factory()()
    try:
        camera_service.create_camera(s, CameraCreate(name="cam1", external_key="cam_a"))
        cand = IncidentCandidate(
            camera_id="cam_a",
            start_sec=1.0,
            end_sec=5.0,
            risk_score=0.8,
            risk_level="red",
            signal_types=["punch"],
            involved_track_ids=[1],
            explanation=["test"],
            evidence={"a": 1},
        )
        inc = incident_service.create_from_candidate(s, cand, "cam_a")
        return inc.id
    finally:
        s.close()


def test_viewer_cannot_review(client: TestClient, viewer_headers: dict[str, str]) -> None:
    iid = _seed_incident()
    r = client.post(
        f"/api/incidents/{iid}/review",
        headers=viewer_headers,
        json={"status": "confirmed", "comment": "x"},
    )
    assert r.status_code == 403


def test_reviewer_can_review(client: TestClient, reviewer_headers: dict[str, str]) -> None:
    iid = _seed_incident()
    r = client.post(
        f"/api/incidents/{iid}/review",
        headers=reviewer_headers,
        json={"status": "confirmed", "comment": "ok"},
    )
    assert r.status_code == 200
    assert r.json()["review_status"] == "confirmed"


def test_admin_create_user(client: TestClient, admin_headers: dict[str, str]) -> None:
    r = client.post(
        "/api/users",
        headers=admin_headers,
        json={"email": "op@example.com", "password": "secret12345", "role": "operator"},
    )
    assert r.status_code == 201
    assert r.json()["role"] == "operator"
