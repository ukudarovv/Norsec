"""Fusion stability: persistence + cooldown (Phase 1)."""

from __future__ import annotations

from fusion.fusion_engine import FusionEngine
from fusion.fusion_stability import StableLiveFusion


def _strong_signals(ts: float):
    social = [{"signal_type": "crowding", "severity": 0.95, "timestamp_sec": ts, "track_ids": [1]}]
    pose = [{"signal_type": "fast_arm_motion", "severity": 0.92, "timestamp_sec": ts, "track_id": 1}]
    action = [{"action_type": "punch", "severity": 0.96, "timestamp_sec": ts, "track_id": 1}]
    return social, pose, action


def test_persistence_requires_two_ticks_when_min_persistence_zero() -> None:
    eng = FusionEngine()
    cfg = {"incident_threshold": 0.55, "min_persistence_sec": 0.0, "cooldown_sec": 100.0, "hysteresis_alpha": 1.0}
    gate = StableLiveFusion(eng, cfg)
    social, pose, action = _strong_signals(0.0)
    _, _, c0 = gate.tick(
        camera_id="c1",
        start_sec=0.0,
        end_sec=1.0,
        social_signals=social,
        pose_signals=pose,
        action_signals=action,
        audio_signals=[],
        context=None,
        now_mono=0.0,
    )
    assert c0 is None
    _, _, c1 = gate.tick(
        camera_id="c1",
        start_sec=0.0,
        end_sec=1.0,
        social_signals=social,
        pose_signals=pose,
        action_signals=action,
        audio_signals=[],
        context=None,
        now_mono=0.001,
    )
    assert c1 is not None


def test_cooldown_blocks_immediate_second_incident() -> None:
    eng = FusionEngine()
    cfg = {"incident_threshold": 0.55, "min_persistence_sec": 0.0, "cooldown_sec": 10.0, "hysteresis_alpha": 1.0}
    gate = StableLiveFusion(eng, cfg)
    social, pose, action = _strong_signals(0.0)
    gate.tick(
        camera_id="c1",
        start_sec=0.0,
        end_sec=1.0,
        social_signals=social,
        pose_signals=pose,
        action_signals=action,
        audio_signals=[],
        context=None,
        now_mono=0.0,
    )
    _, _, first = gate.tick(
        camera_id="c1",
        start_sec=0.0,
        end_sec=1.0,
        social_signals=social,
        pose_signals=pose,
        action_signals=action,
        audio_signals=[],
        context=None,
        now_mono=0.01,
    )
    assert first is not None
    _, _, second = gate.tick(
        camera_id="c1",
        start_sec=0.0,
        end_sec=1.0,
        social_signals=social,
        pose_signals=pose,
        action_signals=action,
        audio_signals=[],
        context=None,
        now_mono=0.02,
    )
    assert second is None
