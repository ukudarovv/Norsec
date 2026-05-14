"""Multimodal fusion: объединение сигналов в bullying risk candidate (не подтверждение вины)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fusion.incident_candidate import IncidentCandidate
from fusion.risk_levels import risk_level_from_score

logger = logging.getLogger(__name__)


def _max_severity(signals: list[Any]) -> float:
    m = 0.0
    for s in signals or []:
        if not isinstance(s, dict):
            continue
        try:
            m = max(m, float(s.get("severity", 0.0)))
        except (TypeError, ValueError):
            continue
    return m


def _action_types(actions: list[Any]) -> set[str]:
    out: set[str] = set()
    for a in actions or []:
        if isinstance(a, dict) and a.get("action_type"):
            out.add(str(a["action_type"]).lower().strip())
    return out


def _pose_types(signals: list[Any]) -> set[str]:
    out: set[str] = set()
    for s in signals or []:
        if isinstance(s, dict) and s.get("signal_type"):
            out.add(str(s["signal_type"]).lower().strip())
    return out


def _social_types(signals: list[Any]) -> set[str]:
    return _pose_types(signals)


def _audio_types(signals: list[Any]) -> set[str]:
    return _pose_types(signals)


def _collect_track_ids(
    social_signals: list[Any],
    pose_signals: list[Any],
    action_signals: list[Any],
) -> list[int]:
    ids: set[int] = set()
    for s in social_signals or []:
        if not isinstance(s, dict):
            continue
        for tid in s.get("track_ids") or []:
            try:
                ids.add(int(tid))
            except (TypeError, ValueError):
                continue
    for s in pose_signals or []:
        if isinstance(s, dict) and s.get("track_id") is not None:
            try:
                ids.add(int(s["track_id"]))
            except (TypeError, ValueError):
                continue
    for s in action_signals or []:
        if isinstance(s, dict) and s.get("track_id") is not None:
            try:
                ids.add(int(s["track_id"]))
            except (TypeError, ValueError):
                continue
    return sorted(ids)


def _signal_times(
    social_signals: list[Any],
    pose_signals: list[Any],
    action_signals: list[Any],
    audio_signals: list[Any],
) -> list[float]:
    ts: list[float] = []
    for s in social_signals or []:
        if isinstance(s, dict) and "timestamp_sec" in s:
            try:
                ts.append(float(s["timestamp_sec"]))
            except (TypeError, ValueError):
                pass
    for s in pose_signals or []:
        if isinstance(s, dict) and "timestamp_sec" in s:
            try:
                ts.append(float(s["timestamp_sec"]))
            except (TypeError, ValueError):
                pass
    for s in action_signals or []:
        if isinstance(s, dict) and "timestamp_sec" in s:
            try:
                ts.append(float(s["timestamp_sec"]))
            except (TypeError, ValueError):
                pass
    for s in audio_signals or []:
        if isinstance(s, dict):
            for k in ("start_sec", "end_sec"):
                if k in s:
                    try:
                        ts.append(float(s[k]))
                    except (TypeError, ValueError):
                        pass
    return ts


def _repeated_activity_boost(
    social_signals: list[Any],
    pose_signals: list[Any],
    action_signals: list[Any],
    audio_signals: list[Any],
    window_sec: float = 10.0,
    min_events: int = 4,
) -> bool:
    """Несколько сигналов в коротком окне — слабый индикатор эскалации (MVP)."""
    events: list[float] = []
    for lst in (social_signals, pose_signals, action_signals, audio_signals):
        for s in lst or []:
            if not isinstance(s, dict):
                continue
            t = None
            if "timestamp_sec" in s:
                try:
                    t = float(s["timestamp_sec"])
                except (TypeError, ValueError):
                    t = None
            elif "start_sec" in s:
                try:
                    t = float(s["start_sec"])
                except (TypeError, ValueError):
                    t = None
            if t is not None:
                events.append(t)
    events.sort()
    if len(events) < min_events:
        return False
    j = 0
    for i, t0 in enumerate(events):
        while j < len(events) and events[j] - t0 <= window_sec:
            j += 1
        if j - i >= min_events:
            return True
    return False


def _build_explanation(
    social_signals: list[Any],
    pose_signals: list[Any],
    action_signals: list[Any],
    audio_signals: list[Any],
) -> tuple[list[str], list[str]]:
    """Возвращает (explanation_lines, unique_signal_types)."""
    expl: list[str] = []
    types_order: list[str] = []

    seen_social: set[str] = set()
    for s in social_signals or []:
        if not isinstance(s, dict):
            continue
        st = str(s.get("signal_type", "")).strip()
        if st and st not in seen_social:
            seen_social.add(st)
            types_order.append(st)
            expl.append(f"Detected social signal: {st}")

    seen_pose: set[str] = set()
    for s in pose_signals or []:
        if not isinstance(s, dict):
            continue
        st = str(s.get("signal_type", "")).strip()
        if st and st not in seen_pose:
            seen_pose.add(st)
            types_order.append(st)
            expl.append(f"Detected pose risk signal: {st}")

    seen_act: set[str] = set()
    for s in action_signals or []:
        if not isinstance(s, dict):
            continue
        at = str(s.get("action_type", "")).strip()
        if at and at not in seen_act:
            seen_act.add(at)
            types_order.append(at)
            expl.append(f"Detected physical action signal: {at}")

    seen_aud: set[str] = set()
    for s in audio_signals or []:
        if not isinstance(s, dict):
            continue
        st = str(s.get("signal_type", "")).strip()
        if st and st not in seen_aud:
            seen_aud.add(st)
            types_order.append(st)
            if st.startswith("verbal_") or st == "aggressive_command":
                expl.append(f"Detected verbal risk signal: {st}")
            else:
                expl.append(f"Detected audio risk signal: {st}")

    return expl, types_order


@dataclass(frozen=True)
class FusionWindowResult:
    """Результат скоринга окна без порога «создать инцидент»."""

    camera_id: str
    risk_score: float
    risk_level: str
    explanation: tuple[str, ...]
    signal_types: tuple[str, ...]
    involved_track_ids: tuple[int, ...]
    evidence: dict[str, Any]
    start_sec: float
    end_sec: float


class FusionEngine:
    def __init__(
        self,
        social_weight: float = 0.20,
        pose_weight: float = 0.20,
        action_weight: float = 0.30,
        audio_weight: float = 0.20,
        context_weight: float = 0.10,
    ) -> None:
        self.social_weight = float(social_weight)
        self.pose_weight = float(pose_weight)
        self.action_weight = float(action_weight)
        self.audio_weight = float(audio_weight)
        self.context_weight = float(context_weight)
        tw = (
            self.social_weight
            + self.pose_weight
            + self.action_weight
            + self.audio_weight
            + self.context_weight
        )
        if abs(tw - 1.0) > 1e-6 and tw > 0:
            self.social_weight /= tw
            self.pose_weight /= tw
            self.action_weight /= tw
            self.audio_weight /= tw
            self.context_weight /= tw

    def compute_window(
        self,
        camera_id: str,
        start_sec: float,
        end_sec: float,
        social_signals: list,
        pose_signals: list,
        action_signals: list,
        audio_signals: list,
        context: dict | None = None,
    ) -> FusionWindowResult:
        ctx = context or {}
        social_score = _max_severity(social_signals)
        pose_score = _max_severity(pose_signals)
        action_score = _max_severity(action_signals)
        audio_score = _max_severity(audio_signals)
        try:
            context_score = float(ctx.get("severity", 0.0))
        except (TypeError, ValueError):
            context_score = 0.0

        risk_score = (
            social_score * self.social_weight
            + pose_score * self.pose_weight
            + action_score * self.action_weight
            + audio_score * self.audio_weight
            + context_score * self.context_weight
        )

        atypes = _action_types(action_signals)
        ptypes = _pose_types(pose_signals)
        stypes = _social_types(social_signals)

        boosts: list[tuple[float, str]] = []

        if atypes & {"punch", "kick"} and "fast_arm_motion" in ptypes:
            boosts.append((0.15, "Escalation: physical strike/kick combined with fast arm motion (requires human review)."))

        if "group_surrounding" in stypes and "verbal_threat" in _audio_types(audio_signals):
            boosts.append((0.20, "Escalation: group surrounding combined with verbal threat (bullying risk candidate)."))

        if "person_on_ground" in ptypes and "crowding" in stypes:
            boosts.append((0.25, "Escalation: person on ground with crowding (requires human review)."))

        if "chase" in atypes and "distress_voice" in _audio_types(audio_signals):
            boosts.append((0.20, "Escalation: chase combined with distress voice (bullying risk candidate)."))

        if _repeated_activity_boost(social_signals, pose_signals, action_signals, audio_signals):
            boosts.append((0.15, "Escalation: repeated multimodal signals within ~10 seconds (low precision heuristic)."))

        for b, msg in boosts:
            risk_score += b
            logger.debug("fusion boost +%.2f: %s", b, msg)

        risk_score = min(float(risk_score), 1.0)

        explanation, signal_types = _build_explanation(
            social_signals, pose_signals, action_signals, audio_signals
        )
        expl_list = list(explanation)
        for _, msg in boosts:
            expl_list.append(msg)

        involved = _collect_track_ids(social_signals, pose_signals, action_signals)
        times = _signal_times(social_signals, pose_signals, action_signals, audio_signals)
        t0 = float(start_sec)
        t1 = float(end_sec)
        if times:
            t0 = min(t0, min(times))
            t1 = max(t1, max(times))

        evidence = {
            "social_signals": list(social_signals or []),
            "pose_signals": list(pose_signals or []),
            "action_signals": list(action_signals or []),
            "audio_signals": list(audio_signals or []),
            "context": dict(ctx),
        }

        level = risk_level_from_score(risk_score)
        return FusionWindowResult(
            camera_id=str(camera_id),
            risk_score=float(risk_score),
            risk_level=level,
            explanation=tuple(expl_list),
            signal_types=tuple(sorted(set(signal_types))),
            involved_track_ids=tuple(involved),
            evidence=evidence,
            start_sec=float(t0),
            end_sec=float(t1),
        )

    def candidate_from_result(self, res: FusionWindowResult) -> IncidentCandidate:
        expl = list(res.explanation)
        expl.append("Risk signal candidate — requires human review (not a confirmed incident).")
        if not any(isinstance(x, str) and x.strip() for x in expl):
            expl.insert(
                0,
                "Elevated multimodal risk (automated assessment); human verification required.",
            )
        return IncidentCandidate(
            camera_id=str(res.camera_id),
            start_sec=float(res.start_sec),
            end_sec=float(res.end_sec),
            risk_score=float(res.risk_score),
            risk_level=str(res.risk_level),
            signal_types=list(res.signal_types),
            involved_track_ids=list(res.involved_track_ids),
            explanation=expl,
            evidence=dict(res.evidence),
        )

    def fuse_window(
        self,
        camera_id: str,
        start_sec: float,
        end_sec: float,
        social_signals: list,
        pose_signals: list,
        action_signals: list,
        audio_signals: list,
        context: dict | None = None,
        *,
        min_candidate_score: float | None = None,
    ) -> IncidentCandidate | None:
        min_s = 0.50 if min_candidate_score is None else float(min_candidate_score)
        res = self.compute_window(
            camera_id,
            start_sec,
            end_sec,
            social_signals,
            pose_signals,
            action_signals,
            audio_signals,
            context,
        )
        if res.risk_score < min_s:
            return None
        return self.candidate_from_result(res)
