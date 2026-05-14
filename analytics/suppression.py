"""Подавление ложных срабатываний (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUPPRESSION_RULES = [
    "sports_mode",
    "playground_mode",
    "high_density_crowd",
    "single_frame_spike",
    "low_confidence_tracks",
    "low_light_or_blur",
]


def _sup_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    c = dict(cfg or {})
    return dict(c.get("suppression") or {})


@dataclass
class SuppressionResult:
    active_rules: list[str] = field(default_factory=list)
    risk_multiplier: float = 1.0
    reasons: list[str] = field(default_factory=list)
    adjusted_severity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_rules": list(self.active_rules),
            "risk_multiplier": self.risk_multiplier,
            "reasons": list(self.reasons),
            "adjusted_severity": self.adjusted_severity,
        }


def assess_live_suppression(
    *,
    zone_tags: list[str] | None,
    track_confidences: list[float] | None,
    signal_age_sec: float | None,
    frame_spike: bool,
    analytics_cfg: dict[str, Any] | None = None,
    crowd_density: float | None = None,
    low_light_score: float | None = None,
) -> SuppressionResult:
    """Снижает вклад сигналов в live-оверлей при слабых условиях."""
    sc = _sup_cfg(analytics_cfg)
    min_persist = float(sc.get("min_signal_persistence_sec", 2))
    min_conf = float(sc.get("min_track_confidence", 0.35))
    spike_pen = float(sc.get("single_frame_spike_penalty", 0.4))
    res = SuppressionResult(risk_multiplier=1.0)
    tags = {t.lower() for t in (zone_tags or [])}

    if "sports" in tags:
        res.active_rules.append("sports_mode")
        res.risk_multiplier *= 0.75
        res.reasons.append("Zone tagged as sports: risk contribution dampened.")
    if "playground" in tags:
        res.active_rules.append("playground_mode")
        res.risk_multiplier *= 0.8
        res.reasons.append("Zone tagged as playground: risk contribution dampened.")

    if crowd_density is not None and crowd_density > 0.85:
        res.active_rules.append("high_density_crowd")
        res.risk_multiplier *= 0.85
        res.reasons.append("High crowd density: motion ambiguity increased.")

    if frame_spike:
        res.active_rules.append("single_frame_spike")
        res.risk_multiplier *= spike_pen
        res.reasons.append("Single-frame spike: temporal persistence required.")

    if track_confidences and max(track_confidences) < min_conf:
        res.active_rules.append("low_confidence_tracks")
        res.risk_multiplier *= 0.5
        res.reasons.append("Low detection confidence: risk not escalated.")

    if signal_age_sec is not None and signal_age_sec < min_persist:
        res.active_rules.append("single_frame_spike")
        res.risk_multiplier *= 0.7
        res.reasons.append("Signal not yet persistent enough in time.")

    if low_light_score is not None and low_light_score > 0.6:
        res.active_rules.append("low_light_or_blur")
        res.risk_multiplier *= 0.75
        res.reasons.append("Low light or blur heuristic: confidence reduced.")

    res.risk_multiplier = max(0.0, min(1.0, res.risk_multiplier))
    return res


def should_suppress_incident_candidate(
    *,
    severities: list[float],
    confidences: list[float] | None,
    analytics_cfg: dict[str, Any] | None = None,
    zone_tags: list[str] | None = None,
) -> tuple[bool, SuppressionResult]:
    """Для кандидата инцидента: не создавать при слишком слабых сигналах."""
    sc = _sup_cfg(analytics_cfg)
    min_conf = float(sc.get("min_track_confidence", 0.35))
    res = assess_live_suppression(
        zone_tags=zone_tags,
        track_confidences=confidences,
        signal_age_sec=999.0,
        frame_spike=len(severities) == 1 and max(severities, default=0) > 0.9,
        analytics_cfg=analytics_cfg,
    )
    if confidences and max(confidences) < min_conf:
        return True, res
    if res.risk_multiplier < 0.35:
        res.reasons.append("Combined suppression too strong for incident creation.")
        return True, res
    return False, res


def apply_severity_multiplier(base: float, suppression: SuppressionResult) -> float:
    return max(0.0, min(1.0, float(base) * suppression.risk_multiplier))
