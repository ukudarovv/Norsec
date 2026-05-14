"""Тесты ``TrackMemory``."""

from __future__ import annotations

from tracking.track_memory import TrackMemory


def test_track_memory_stores_history():
    m = TrackMemory({"tracking": {"max_positions": 5, "track_ttl_sec": 60}})
    m.update(1, (0.0, 0.0, 10.0, 20.0), 0.0)
    m.update(1, (1.0, 0.0, 11.0, 20.0), 0.1)
    r = m.get_track(1)
    assert r is not None
    assert len(r.positions) == 2
    assert len(r.velocity_history) == 1
    assert r.first_seen == 0.0
    assert r.last_seen == 0.1


def test_cleanup_old_tracks():
    m = TrackMemory({"tracking": {"track_ttl_sec": 1.0, "max_positions": 50}})
    m.update(2, (0.0, 0.0, 4.0, 4.0), 0.0)
    removed = m.cleanup_old_tracks(5.0)
    assert 2 in removed
    assert m.get_track(2) is None
