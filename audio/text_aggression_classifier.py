"""Классификация агрессии в тексте: MVP rule-based (RU) + задел под XLM-RoBERTa."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

LABELS = ("neutral", "insult", "threat", "humiliation", "aggressive_command")


class TextAggressionClassifier:
    def __init__(self, model_name: str = "xlm-roberta-base") -> None:
        self.model_name = model_name
        self._hf_classifier: Any = None

    def classify(self, text: str) -> dict[str, Any]:
        t = (text or "").strip().lower()
        if not t:
            return {
                "label": "neutral",
                "confidence": 0.95,
                "scores": {k: 1.0 if k == "neutral" else 0.0 for k in LABELS},
            }

        scores = {k: 0.0 for k in LABELS}
        scores["neutral"] = 0.35

        threat_kw = (
            "убью", "убьют", "убей", "зарежу", "уроню", "сломаю", "разобью", "набью",
            "отрежу", "застрел", "взорву", "kill you", "hurt you", "beat you",
        )
        insult_kw = (
            "дурак", "дура", "идиот", "кретин", "тупой", "тупица", "мразь", "сволочь",
            "урод", "дебил", "лох", "stupid", "idiot", "moron",
        )
        hum_kw = (
            "позор", "никто не любит", "никому не нужен", "униж", "стыд", "никчём",
            "loser", "worthless", "pathetic",
        )
        cmd_kw = (
            "заткнись", "замолчи", "убери руки", "сюда иди", "иди сюда", "сделай как",
            "слушай сюда", "shut up", "get out", "do what i say",
        )

        for kw in threat_kw:
            if kw in t:
                scores["threat"] = max(scores["threat"], 0.82)
        for kw in insult_kw:
            if kw in t:
                scores["insult"] = max(scores["insult"], 0.78)
        for kw in hum_kw:
            if kw in t:
                scores["humiliation"] = max(scores["humiliation"], 0.76)
        for kw in cmd_kw:
            if kw in t:
                scores["aggressive_command"] = max(scores["aggressive_command"], 0.72)

        if re.search(r"[!?]{2,}", t):
            scores["aggressive_command"] = max(scores["aggressive_command"], 0.45)

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if best == "neutral" or scores[best] < 0.4:
            best = "neutral"
            conf = 0.55 + 0.2 * (1.0 - max(scores[k] for k in LABELS if k != "neutral"))
        else:
            conf = min(0.95, scores[best] + 0.05)

        return {
            "label": best,
            "confidence": round(float(conf), 4),
            "scores": {k: round(float(scores[k]), 4) for k in LABELS},
        }
