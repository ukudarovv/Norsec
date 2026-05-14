"""Single source of truth for verbal classification label names."""

from __future__ import annotations

LABEL_NAMES = [
    "neutral_conflict",
    "insult_humiliation",
    "explicit_threat",
    "coercion_harassment",
]

LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}
