from __future__ import annotations

from fastapi.testclient import TestClient

from api.db.session import get_session_factory
from api.services import camera_service, incident_service
from api.schemas.camera import CameraCreate
from fusion.incident_candidate import IncidentCandidate


def _make_incident() -> str:
    s = get_session_factory()()
    try:
        camera_service.create_camera(s, CameraCreate(name="c", external_key="k"))
        cand = IncidentCandidate(
            camera_id="k",
            start_sec=0.0,
            end_sec=1.0,
            risk_score=0.9,
            risk_level="red",
            signal_types=["x"],
            involved_track_ids=[],
            explanation=["e"],
            evidence={},
        )
        inc = incident_service.create_from_candidate(s, cand, "k")
        return str(inc.id)
    finally:
        s.close()


def test_list_reviews(client: TestClient, admin_headers: dict[str, str], reviewer_headers: dict[str, str]) -> None:
    iid = _make_incident()
    client.post(
        f"/api/incidents/{iid}/review",
        headers=reviewer_headers,
        json={"status": "false_positive", "comment": "no"},
    )
    r = client.get(f"/api/incidents/{iid}/reviews", headers=admin_headers)
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert r.json()[0]["status"] == "false_positive"
