"""Phase 3: dashboard + queue permissions."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_dashboard_stats_viewer_ok(client: TestClient, viewer_headers: dict[str, str]) -> None:
    r = client.get("/api/dashboard/stats", headers=viewer_headers)
    assert r.status_code == 200
    assert "totals" in r.json()


def test_review_queue_viewer_ok(client: TestClient, viewer_headers: dict[str, str]) -> None:
    r = client.get("/api/reviews/queue", headers=viewer_headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_operator_note_forbidden_for_viewer(client: TestClient, viewer_headers: dict[str, str]) -> None:
    import uuid

    fake = uuid.uuid4()
    r = client.post(
        f"/api/incidents/{fake}/notes",
        headers=viewer_headers,
        json={"comment": "nope"},
    )
    assert r.status_code == 403
