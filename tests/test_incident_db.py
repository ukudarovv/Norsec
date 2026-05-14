from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from api.db.session import get_session_factory
from api.services import camera_service, incident_service
from api.schemas.camera import CameraCreate
from fusion.incident_candidate import IncidentCandidate


def test_incident_persisted_in_db(client: TestClient, admin_headers: dict[str, str]) -> None:
    s = get_session_factory()()
    try:
        camera_service.create_camera(s, CameraCreate(name="c-db", external_key="ext1"))
        cand = IncidentCandidate(
            camera_id="ext1",
            start_sec=0.0,
            end_sec=3.0,
            risk_score=0.66,
            risk_level="orange",
            signal_types=["a"],
            involved_track_ids=[2],
            explanation=["line"],
            evidence={"k": "v"},
        )
        inc = incident_service.create_from_candidate(s, cand, "ext1")
        iid = inc.id
    finally:
        s.close()

    r = client.get(f"/api/incidents/{iid}", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["risk_score"] == 0.66
    assert body["involved_track_ids"] == [2]


def test_audit_on_login(client: TestClient, admin_headers: dict[str, str]) -> None:
    from api.db.models import AuditLog
    from sqlalchemy import select

    s = get_session_factory()()
    try:
        n = len(list(s.scalars(select(AuditLog)).all()))
        assert n >= 1
    finally:
        s.close()
