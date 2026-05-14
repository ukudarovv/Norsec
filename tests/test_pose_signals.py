"""Поза-сигналы."""

from __future__ import annotations

from analytics.pose_signals import detect_pose_signals


def test_raised_hand():
    kps = [[0.0, 0.0, 0.1] for _ in range(17)]
    kps[0] = [100.0, 80.0, 0.95]
    kps[5] = [70.0, 130.0, 0.95]
    kps[6] = [130.0, 130.0, 0.95]
    kps[9] = [70.0, 50.0, 0.95]
    kps[10] = [130.0, 150.0, 0.9]
    kps[11] = [85.0, 200.0, 0.9]
    kps[12] = [115.0, 200.0, 0.9]
    out = detect_pose_signals(1, kps, 0.0, {"pose": {"keypoint_confidence": 0.35}})
    assert any(p.signal_type == "raised_hand" for p in out)


def test_fast_arm_motion():
    kps = [[0.0, 0.0, 0.1] for _ in range(17)]
    kps[5] = [100.0, 100.0, 0.95]
    kps[9] = [100.0, 100.0, 0.95]
    prev = (20.0, 100.0)
    out = detect_pose_signals(
        1,
        kps,
        0.0,
        {"pose": {"keypoint_confidence": 0.2, "fast_arm_px_per_sec": 100}},
        prev_wrist=prev,
    )
    assert any(p.signal_type == "fast_arm_motion" for p in out)
