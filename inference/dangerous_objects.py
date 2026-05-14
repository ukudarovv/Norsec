"""
Прокси «опасные предметы» по одному кадру: zero-shot CLIP с несколькими контрастными подписями.

Один проход пайплайна: softmax по всем кандидатам; weapon_proxy = сумма вероятностей по «семейству» опасных промптов.

Это учебное приближение, не детекция с bounding box; возможны ложные срабатывания.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch

DEFAULT_DANGEROUS_OBJECTS_MODEL = "openai/clip-vit-base-patch32"

# Один «мирный» кандидат против множества узких опасных гипотез — softmax распределяет массу точнее, чем один generic weapon.
LABEL_SCENE_SAFE = (
    "a photo without any weapon, knife, firearm or blade, only harmless objects"
)

LABEL_SCENE_WEAPON = (
    "a photo showing a dangerous weapon such as knife, blade, handgun, rifle, gun or similar object clearly visible"
)

WEAPON_FAMILY_LABELS: tuple[str, ...] = (
    LABEL_SCENE_WEAPON,
    "a photo showing a handgun or pistol clearly visible",
    "a photo showing a rifle, shotgun or long gun clearly visible",
    "a photo showing a kitchen knife, folding knife or dagger clearly visible as a weapon",
    "a photo showing a large blade such as a sword or machete clearly visible",
    "a photo showing a blunt weapon such as a bat, club or stick used as a weapon",
    "a photo showing scissors or sharp tools held in a threatening way as a weapon",
    "a photo showing an explosive device, bomb or grenade clearly visible",
    "a photo showing pepper spray or a small self-defense aerosol canister clearly visible",
    "a photo showing a tactical or combat knife clearly visible",
)

WEAPON_LABEL_SET: frozenset[str] = frozenset(WEAPON_FAMILY_LABELS)


def default_clip_candidate_labels() -> list[str]:
    return [LABEL_SCENE_SAFE, *WEAPON_FAMILY_LABELS]


def _pipeline_device(torch_device: torch.device | str | None) -> int:
    if torch_device is None:
        td = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        td = torch.device(torch_device)
    if td.type == "cuda":
        idx = td.index if td.index is not None else 0
        return int(idx) if torch.cuda.is_available() else -1
    return -1


@lru_cache(maxsize=4)
def _cached_zero_shot(model_id: str, device_idx: int) -> Any:
    from transformers import pipeline

    try:
        return pipeline(
            "zero-shot-image-classification",
            model=model_id,
            device=device_idx,
        )
    except Exception as e:
        msg = str(e).lower()
        hint = ""
        if "model_type" in msg or "image_processor" in msg or "preprocessor" in msg:
            hint = (
                f" Попробуйте другой CLIP-hub id или оставьте `{DEFAULT_DANGEROUS_OBJECTS_MODEL}`."
            )
        raise RuntimeError(f"{e}{hint}") from e


def aggregate_clip_multilabel_outputs(
    outputs: list[dict[str, Any]],
    *,
    safe_label: str = LABEL_SCENE_SAFE,
    weapon_labels: frozenset[str] = WEAPON_LABEL_SET,
) -> tuple[float, str, str]:
    """
    По списку пар {label, score} от HF zero-shot-image-classification:
    weapon_proxy = сумма score по weapon_labels (при полном наборе кандидатов совпадает с 1 - score(safe) по смыслу softmax).
    Возвращает (weapon_proxy, топ-1 подпись по score, краткая строка «лучший weapon-промпт»).
    """
    if not outputs:
        return 0.0, "", ""
    rows = [r for r in outputs if isinstance(r, dict)]
    if not rows:
        return 0.0, "", ""
    by_label: dict[str, float] = {}
    for r in rows:
        lab = str(r.get("label", "") or "")
        by_label[lab] = float(r.get("score", 0.0))
    w_sum = sum(by_label.get(w, 0.0) for w in weapon_labels)
    top = max(rows, key=lambda r: float(r.get("score", 0.0)))
    top_lab = str(top.get("label", "") or "")[:220]
    best_w = ""
    best_ws = -1.0
    for w in weapon_labels:
        sc = by_label.get(w, 0.0)
        if sc > best_ws:
            best_ws = sc
            best_w = w
    detail = ""
    if best_w and best_ws > 1e-9:
        short = best_w[:72] + ("…" if len(best_w) > 72 else "")
        detail = f"{short} @{best_ws:.3f}"
    wp = float(min(1.0, max(0.0, w_sum)))
    return wp, top_lab, detail


def weapon_proxy_score_from_topk(outputs: list[dict[str, Any]]) -> tuple[float, str]:
    """
    Обратная совместимость для старых двухклассовых списков: ищет только LABEL_SCENE_WEAPON или суммирует семейство.
    """
    wp, top, _detail = aggregate_clip_multilabel_outputs(outputs)
    return wp, top[:120]


def classify_dangerous_objects_rgb(
    rgb: np.ndarray,
    *,
    model_id: str,
    torch_device: torch.device,
    candidate_labels: list[str] | None = None,
) -> tuple[float, str, str]:
    """
    RGB uint8 H×W×3 → (weapon_proxy 0–1, топ-1 подпись среди всех кандидатов, краткая подсказка по лучшему weapon-промпту).
    """
    from PIL import Image

    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return 0.0, "", ""
    pil = Image.fromarray(rgb[:, :, :3])

    mid = (model_id or "").strip() or DEFAULT_DANGEROUS_OBJECTS_MODEL
    pipe = _cached_zero_shot(mid, _pipeline_device(torch_device))

    cand = candidate_labels if candidate_labels is not None else default_clip_candidate_labels()
    outputs = pipe(pil, candidate_labels=cand)
    raw = outputs if isinstance(outputs, list) else []
    return aggregate_clip_multilabel_outputs(raw)


def danger_disclaimer_md() -> str:
    return (
        "_Опасные предметы (онлайн)_: zero-shot CLIP по **набору** англоязычных промптов (несколько типов оружия/опасных "
        "предметов против одной «безопасной» сцены); это **прокси**, не промышленный детектор и не замена охранному "
        "видеоанализу."
    )
