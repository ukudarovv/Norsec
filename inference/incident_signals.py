"""
Сводные сигналы «драка / конфликт»: склейка визуального violence-proxy и вербальных меток без новых моделей.
"""

from __future__ import annotations

from typing import Any

from hf_ml_verbal.label_config import LABEL_NAMES


def verbal_conflict_score(probs: dict[str, float]) -> tuple[float, dict[str, float]]:
    """
    Кортеж агрегата и компонент (все значения >= 0):
    - conflict_focus — спор / оскорбление без обязательной угрозы;
    - escalation_peak — явная угроза или принуждение;
    - verbal_aggregate — max по четырём классам.
    """
    nc = float(probs.get("neutral_conflict", 0.0) or 0.0)
    ins = float(probs.get("insult_humiliation", 0.0) or 0.0)
    thr = float(probs.get("explicit_threat", 0.0) or 0.0)
    coer = float(probs.get("coercion_harassment", 0.0) or 0.0)

    conflict_focus = max(nc, ins)
    escalation_peak = max(thr, coer)
    verbal_aggregate = max(nc, ins, thr, coer)

    comps = {
        "neutral_conflict": nc,
        "insult_humiliation": ins,
        "explicit_threat": thr,
        "coercion_harassment": coer,
        "conflict_focus": conflict_focus,
        "escalation_peak": escalation_peak,
    }
    return verbal_aggregate, comps


def incident_escalation_note_ru(probs: dict[str, float]) -> str:
    """Короткое текстовое пояснение по доминирующим каналам (эвристика порогов)."""
    parts: list[str] = []
    if float(probs.get("explicit_threat", 0) or 0) >= 0.35:
        parts.append("явная угроза")
    if float(probs.get("coercion_harassment", 0) or 0) >= 0.35:
        parts.append("принуждение или прессинг")
    if float(probs.get("insult_humiliation", 0) or 0) >= 0.45:
        parts.append("оскорбление или унижение")
    if float(probs.get("neutral_conflict", 0) or 0) >= 0.45:
        parts.append("нейтральный конфликт (спор/разногласие)")
    if not parts:
        return "сигналы вербальной головы ниже порогов сводки"
    return "; ".join(parts)


def physical_proxy_from_visual_blob(visual: dict[str, Any] | None) -> float:
    if not visual or not visual.get("enabled"):
        return 0.0
    summary = visual.get("summary") or {}
    return float(summary.get("max", 0.0) or 0.0)


def physical_proxy_from_visual_rows(rows: list[dict[str, Any]]) -> float:
    best = 0.0
    for r in rows:
        if r.get("error"):
            continue
        v = r.get("violence_probability")
        if v is None:
            continue
        try:
            best = max(best, float(v))
        except (TypeError, ValueError):
            continue
    return best


def max_scores_from_segment_dicts(segments: list[dict[str, Any]]) -> dict[str, float]:
    out = {k: 0.0 for k in LABEL_NAMES}
    for seg in segments:
        vd = seg.get("verbal")
        if not isinstance(vd, dict):
            continue
        for k in LABEL_NAMES:
            try:
                out[k] = max(out[k], float(vd.get(k, 0.0)))
            except (TypeError, ValueError):
                pass
    return out


def max_scores_from_live_verbal(lines: list[dict[str, Any]]) -> dict[str, float]:
    out = {k: 0.0 for k in LABEL_NAMES}
    for row in lines:
        sc = row.get("scores")
        if not isinstance(sc, dict):
            continue
        for k in LABEL_NAMES:
            try:
                out[k] = max(out[k], float(sc.get(k, 0.0)))
            except (TypeError, ValueError):
                pass
    return out


def fuse_batch_incident(
    visual: dict[str, Any] | None,
    max_scores: dict[str, float],
    *,
    verbal_skipped_reason: str | None = None,
) -> dict[str, Any]:
    phys = physical_proxy_from_visual_blob(visual)
    verbal_agg, comp = verbal_conflict_score(max_scores)

    if verbal_skipped_reason:
        note = verbal_skipped_reason
        v_conflict = 0.0
        v_esc = 0.0
        v_agg_round = 0.0
    else:
        note = incident_escalation_note_ru(max_scores)
        v_conflict = float(comp["conflict_focus"])
        v_esc = float(comp["escalation_peak"])
        v_agg_round = round(verbal_agg, 4)

    return {
        "incident_physical_proxy": round(phys, 4),
        "incident_verbal_conflict": round(v_conflict, 4),
        "incident_verbal_escalation": round(v_esc, 4),
        "incident_verbal_aggregate": v_agg_round,
        "incident_escalation_note": note,
    }


def incident_batch_markdown_lines(incident: dict[str, Any], *, verbal_available: bool) -> list[str]:
    lines = [
        "### Драка и конфликт (сводка)",
        "",
        (
            f"- **Прокси физической драки/насилия по кадрам:** `{incident['incident_physical_proxy']}` "
            "(0–1, без bbox и без гарантии двух человек)."
        ),
    ]
    if verbal_available:
        lines.extend(
            [
                (
                    f"- **Вербальная часть:** агрегат по окнам `{incident['incident_verbal_aggregate']}`, "
                    f"фокус конфликта/спора `{incident['incident_verbal_conflict']}`, "
                    f"угрозы/принуждение `{incident['incident_verbal_escalation']}`."
                ),
                f"- **Преобладание признаков (речь):** {incident['incident_escalation_note']}.",
            ]
        )
    else:
        lines.append(
            "- **Речь:** не анализировалась — вербальный слой для этой дорожки отключён (нет аудио)."
        )

    lines.extend(
        [
            "",
            "_Это технические прокси (не юридическое заключение); на видео спорт и движение часто повышают ложные срабатывания._",
            "",
        ]
    )
    return lines


def online_incident_markdown_lines(
    verbal_lines: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
) -> list[str]:
    """Блок Markdown для вкладки «Онлайн» до разделительной линии с остальными слоями."""
    vm = max_scores_from_live_verbal(verbal_lines)
    verb_agg, _ = verbal_conflict_score(vm)
    phys = physical_proxy_from_visual_rows(visual_rows)

    faux_visual = {"enabled": True, "summary": {"max": phys}}
    incident = fuse_batch_incident(faux_visual, vm)

    lines = [
        "## Сводка: драка и конфликт (онлайн)",
        "",
        (
            f"- **Видеопрокси (последние кадры журнала):** max violence_proxy ≈ **`{incident['incident_physical_proxy']:.4f}`** "
            f"(нет bbox; при выключенном violence-слое будет 0)."
        ),
        (
            f"- **Речь (максимум по накопленным фразам):** общий **`{verb_agg:.3f}`**, "
            f"конфликт/спор **`{incident['incident_verbal_conflict']:.3f}`**, "
            f"угроза/принуждение **`{incident['incident_verbal_escalation']:.3f}`**."
        ),
        f"- **Пояснение по речи:** {incident['incident_escalation_note']}.",
        "",
        "---",
        "",
    ]
    return lines


def prepend_online_incident(
    verbal_lines: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
    body_md: str,
) -> str:
    prefix = "\n".join(online_incident_markdown_lines(verbal_lines, visual_rows)).rstrip()
    return f"{prefix}\n\n{body_md.lstrip()}"
