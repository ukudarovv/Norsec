"""Тесты траекторного анализатора."""

from __future__ import annotations

from tracking.track_memory import TrackMemory
from tracking import trajectory_analyzer as ta


def test_is_moving_towards():
    m = TrackMemory()
    m.update(1, (0.0, 100.0, 20.0, 140.0), 0.0)
    m.update(1, (30.0, 100.0, 50.0, 140.0), 0.1)
    m.update(2, (200.0, 100.0, 220.0, 140.0), 0.0)
    m.update(2, (200.0, 100.0, 220.0, 140.0), 0.1)
    assert ta.is_moving_towards(m, 1, 2, min_cosine=0.2) is True


def test_is_following():
    m = TrackMemory()
    for i in range(50):
        t = i * 0.1
        bx = 100.0 + i * 4.0
        ax = 40.0 + i * 4.0
        m.update(10, (ax, 50.0, ax + 20.0, 90.0), t)
        m.update(11, (bx, 50.0, bx + 20.0, 90.0), t)
    assert ta.is_following(m, 10, 11, min_sec=4.0, max_lateral_px=120.0) is True


def test_compute_speed():
    m = TrackMemory()
    m.update(3, (0.0, 0.0, 10.0, 10.0), 0.0)
    m.update(3, (10.0, 0.0, 20.0, 10.0), 1.0)
    s = ta.compute_speed(m, 3)
    assert s is not None and s > 1.0
