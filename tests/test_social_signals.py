"""Социальные сигналы."""

from __future__ import annotations

from analytics.social_signals import detect_social_signals
from tracking.track_memory import TrackMemory


def test_rapid_approach():
    m = TrackMemory()
    cfg = {"social": {"rapid_approach_px_per_sec": 50, "assumed_fps": 30}}
    for i in range(5):
        t = i * 0.05
        m.update(1, (10.0 + i * 30.0, 50.0, 40.0 + i * 30.0, 120.0), t)
        m.update(2, (300.0 - i * 40.0, 50.0, 330.0 - i * 40.0, 120.0), t)
    people = [
        {"track_id": 1, "bbox": [10.0 + 4 * 30.0, 50.0, 40.0 + 4 * 30.0, 120.0], "confidence": 0.9},
        {"track_id": 2, "bbox": [300.0 - 4 * 40.0, 50.0, 330.0 - 4 * 40.0, 120.0], "confidence": 0.9},
    ]
    sigs = detect_social_signals(m, people, 0.25, cfg)
    types = {s.signal_type for s in sigs}
    assert "rapid_approach" in types


def test_following():
    m = TrackMemory()
    for i in range(50):
        t = i * 0.1
        bx = 200.0 + i * 3.0
        ax = 120.0 + i * 3.0
        m.update(1, (ax, 40.0, ax + 20.0, 100.0), t)
        m.update(2, (bx, 40.0, bx + 20.0, 100.0), t)
    people = [
        {"track_id": 1, "bbox": [120.0 + 49 * 3.0, 40.0, 140.0 + 49 * 3.0, 100.0], "confidence": 0.9},
        {"track_id": 2, "bbox": [200.0 + 49 * 3.0, 40.0, 220.0 + 49 * 3.0, 100.0], "confidence": 0.9},
    ]
    sigs = detect_social_signals(m, people, 4.9, {"social": {"following_min_sec": 4.0}})
    assert any(s.signal_type == "following" for s in sigs)


def test_group_surrounding():
    m = TrackMemory()
    center = (200.0, 200.0, 230.0, 260.0)
    m.update(0, center, 0.0)
    for k, off in enumerate([(20, 0), (-20, 0), (0, 25)], start=1):
        x = center[0] + off[0]
        y = center[1] + off[1]
        m.update(k, (x, y, x + 25.0, y + 60.0), 0.0)
    people = []
    for i in range(4):
        rec = m.get_track(i)
        assert rec is not None
        people.append({"track_id": i, "bbox": list(rec.bbox_history[-1]), "confidence": 0.9})
    sigs = detect_social_signals(m, people, 0.0, {"social": {"close_distance_px": 150, "surrounding_min_people": 3}})
    assert any(s.signal_type == "group_surrounding" for s in sigs)
