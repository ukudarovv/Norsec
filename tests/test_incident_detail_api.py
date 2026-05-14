"""Phase 3: incident detail API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.db.session import get_session_factory
from api.services import camera_service, incident_service
from api.schemas.camera import CameraCreate
from fusion.incident_candidate import IncidentCandidate


def _make() -> str:
    s = get_session_factory()()
    try:
        camera_service.create_camera(s, CameraCreate(name="c_det", external_key="k_det"))
        cand = IncidentCandidate(
            camera_id="k_det",
            start_sec=1.0,
            end_sec=3.0,
            risk_score=0.6,
            risk_level="yellow",
            signal_types=["social"],
            involved_track_ids=[],
            explanation=["line"],
            evidence={"social_signals": [{"signal_type": "crowding", "severity": 0.5}]},
        )
        inc = incident_service.create_from_candidate(s, cand, "k_det")
        return str(inc.id)
    finally:
        s.close()


def test_incident_detail_has_evidence(client: TestClient, admin_headers: dict[str, str]) -> None:
    iid = _make()
    r = client.get(f"/api/incidents/{iid}/detail", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert "incident" in body and "evidence" in body
    assert body["analytics"]["social_signals"]
    assert isinstance(body["reviews"], list)
