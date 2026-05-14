"""HTTP endpoints Phase 2 analytics."""

from __future__ import annotations

import uuid


def test_incident_analytics_endpoint(client, admin_headers, db_session):
    from api.db.models import Camera, Incident

    cam = Camera(name="a", rtsp_url="rtsp://x", status="offline", is_active=False)
    db_session.add(cam)
    db_session.commit()
    db_session.refresh(cam)
    inc = Incident(
        camera_id=cam.id,
        start_sec=0.0,
        end_sec=1.0,
        risk_score=0.5,
        risk_level="yellow",
        signal_types=["social"],
        explanation=["test"],
        review_status="new",
        evidence={"social_signals": [{"signal_type": "crowding", "severity": 0.5}]},
    )
    db_session.add(inc)
    db_session.commit()
    r = client.get(f"/api/incidents/{inc.id}/analytics", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["incident_id"] == str(inc.id)
    assert len(body["social_signals"]) == 1


def test_analytics_signals_catalog(client, admin_headers):
    r = client.get("/api/analytics/signals", headers=admin_headers)
    assert r.status_code == 200
    j = r.json()
    assert "rapid_approach" in j["social_signals"]
    assert "raised_hand" in j["pose_signals"]


def test_camera_live_analytics_requires_camera(client, admin_headers):
    fake = uuid.uuid4()
    r = client.get(f"/api/cameras/{fake}/analytics/live", headers=admin_headers)
    assert r.status_code == 404
