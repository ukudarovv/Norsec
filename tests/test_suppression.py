"""Suppression."""

from __future__ import annotations

from analytics.suppression import (
    SuppressionResult,
    apply_severity_multiplier,
    assess_live_suppression,
    should_suppress_incident_candidate,
)


def test_single_frame_spike_reduces_multiplier():
    r = assess_live_suppression(
        zone_tags=None,
        track_confidences=[0.9],
        signal_age_sec=0.05,
        frame_spike=True,
        analytics_cfg={"suppression": {"single_frame_spike_penalty": 0.4}},
    )
    assert "single_frame_spike" in r.active_rules
    assert r.risk_multiplier < 1.0


def test_low_confidence_suppresses_incident():
    sup, res = should_suppress_incident_candidate(
        severities=[0.9],
        confidences=[0.1],
        analytics_cfg={"suppression": {"min_track_confidence": 0.35}},
    )
    assert sup is True
    assert "low_confidence_tracks" in res.active_rules


def test_apply_severity_multiplier():
    s = SuppressionResult(risk_multiplier=0.5)
    assert apply_severity_multiplier(0.8, s) == 0.4
