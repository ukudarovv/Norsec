"""Phase 3: training candidates JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from api.db.session import get_session_factory
from api.services import camera_service, incident_service
from api.schemas.camera import CameraCreate
from fusion.incident_candidate import IncidentCandidate


def _make_inc(tmp_path: Path) -> str:
    s = get_session_factory()()
    try:
        camera_service.create_camera(s, CameraCreate(name="c_tr", external_key="k_tr"))
        cand = IncidentCandidate(
            camera_id="k_tr",
            start_sec=0.0,
            end_sec=1.0,
            risk_score=0.8,
            risk_level="orange",
            signal_types=["t"],
            involved_track_ids=[],
            explanation=["e"],
            evidence={},
        )
        inc = incident_service.create_from_candidate(s, cand, "k_tr")
        return str(inc.id)
    finally:
        s.close()


def test_training_candidate_creates_jsonl(
    client: TestClient, reviewer_headers: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "tc.jsonl"
    monkeypatch.setenv("TRAINING_CANDIDATES_PATH", str(p))
    iid = _make_inc(tmp_path)
    r = client.post(
        f"/api/incidents/{iid}/review",
        headers=reviewer_headers,
        json={"status": "training_candidate", "comment": "for dataset", "tags": ["use_for_training"]},
    )
    assert r.status_code == 200, r.text
    assert p.is_file()
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    row = json.loads(lines[-1])
    assert row["incident_id"] == iid
    assert row["status"] == "training_candidate"
