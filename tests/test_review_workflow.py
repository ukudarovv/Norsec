"""Phase 3: review workflow, comments, tags, audit."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.db.session import get_session_factory
from api.services import camera_service, incident_service
from api.schemas.camera import CameraCreate
from fusion.incident_candidate import IncidentCandidate


def _make_incident() -> str:
    s = get_session_factory()()
    try:
        camera_service.create_camera(s, CameraCreate(name="c_rw", external_key="k_rw"))
        cand = IncidentCandidate(
            camera_id="k_rw",
            start_sec=0.0,
            end_sec=1.0,
            risk_score=0.9,
            risk_level="red",
            signal_types=["x"],
            involved_track_ids=[],
            explanation=["e"],
            evidence={},
        )
        inc = incident_service.create_from_candidate(s, cand, "k_rw")
        return str(inc.id)
    finally:
        s.close()


def test_reviewer_can_change_status(client: TestClient, reviewer_headers: dict[str, str]) -> None:
    iid = _make_incident()
    r = client.post(
        f"/api/incidents/{iid}/review",
        headers=reviewer_headers,
        json={"status": "false_positive", "comment": "ok", "tags": ["false_positive"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["review_status"] == "false_positive"


def test_viewer_cannot_review(client: TestClient, viewer_headers: dict[str, str]) -> None:
    iid = _make_incident()
    r = client.post(
        f"/api/incidents/{iid}/review",
        headers=viewer_headers,
        json={"status": "confirmed", "comment": "x"},
    )
    assert r.status_code == 403


def test_comment_required_for_confirmed(client: TestClient, reviewer_headers: dict[str, str]) -> None:
    iid = _make_incident()
    r = client.post(
        f"/api/incidents/{iid}/review",
        headers=reviewer_headers,
        json={"status": "confirmed"},
    )
    assert r.status_code == 400


def test_review_history_has_tags(client: TestClient, admin_headers: dict[str, str], reviewer_headers: dict[str, str]) -> None:
    iid = _make_incident()
    client.post(
        f"/api/incidents/{iid}/review",
        headers=reviewer_headers,
        json={"status": "needs_review", "tags": ["rough_play"]},
    )
    r = client.get(f"/api/incidents/{iid}/reviews", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()[-1]["tags"] == ["rough_play"]


def test_audit_on_review(client: TestClient, admin_headers: dict[str, str], reviewer_headers: dict[str, str], db_session) -> None:
    from api.db.models import AuditLog
    from sqlalchemy import select

    iid = _make_incident()
    client.post(
        f"/api/incidents/{iid}/review",
        headers=reviewer_headers,
        json={"status": "archived", "comment": "done"},
    )
    rows = list(db_session.scalars(select(AuditLog).where(AuditLog.entity_id == iid)).all())
    actions = {x.action for x in rows}
    assert "incident_reviewed" in actions
    assert "incident_archived" in actions
