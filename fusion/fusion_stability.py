"""Стабилизация live-инцидентов: persistence, cooldown, hysteresis (Phase 1)."""

from __future__ import annotations

import logging
from typing import Any

from fusion.fusion_engine import FusionEngine
from fusion.incident_candidate import IncidentCandidate
from fusion.risk_levels import risk_level_from_score

logger = logging.getLogger(__name__)


class StableLiveFusion:
    """
    Сглаживание риска и эмиссия ``IncidentCandidate`` только при устойчивом высоком риске
    и с cooldown между инцидентами.
    """

    def __init__(self, engine: FusionEngine, fusion_cfg: dict[str, Any]) -> None:
        self._engine = engine
        self._suspect = float(fusion_cfg.get("suspect_threshold", 0.5))
        self._incident = float(fusion_cfg.get("incident_threshold", 0.75))
        self._persist = float(fusion_cfg.get("min_persistence_sec", 2.0))
        self._cooldown = float(fusion_cfg.get("cooldown_sec", 20.0))
        self._alpha = float(fusion_cfg.get("hysteresis_alpha", 0.35))
        self._smooth = 0.0
        self._above_since: float | None = None
        self._last_emit_mono = -1e12

    def tick(
        self,
        *,
        camera_id: str,
        start_sec: float,
        end_sec: float,
        social_signals: list,
        pose_signals: list,
        action_signals: list,
        audio_signals: list,
        context: dict[str, Any] | None,
        now_mono: float,
    ) -> tuple[float, str, IncidentCandidate | None]:
        wr = self._engine.compute_window(
            camera_id,
            start_sec,
            end_sec,
            social_signals,
            pose_signals,
            action_signals,
            audio_signals,
            context,
        )
        raw = float(wr.risk_score)
        a = self._alpha
        self._smooth = a * raw + (1.0 - a) * self._smooth
        level = risk_level_from_score(self._smooth)

        cand: IncidentCandidate | None = None
        if self._smooth >= self._incident:
            if self._above_since is None:
                self._above_since = now_mono
            elif now_mono - self._above_since >= self._persist:
                if now_mono - self._last_emit_mono >= self._cooldown and raw >= self._incident * 0.98:
                    cand = self._engine.candidate_from_result(wr)
                    self._last_emit_mono = now_mono
                    self._above_since = None
                    logger.info(
                        "fusion_incident_emitted",
                        extra={"camera_id": camera_id, "risk_score": raw, "smoothed": self._smooth},
                    )
        else:
            self._above_since = None

        _ = self._suspect  # зарезервировано для UI «подозрение» / следующие этапы
        return self._smooth, level, cand

    def reset(self) -> None:
        self._smooth = 0.0
        self._above_since = None
        self._last_emit_mono = -1e12
