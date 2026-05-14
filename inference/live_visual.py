"""Онлайн-поток с веб-камеры: violence ViT по кадру + прокси опасных предметов (CLIP) + лица/эмоции (Haar + ViT)."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np

from inference.dangerous_objects import (
    DEFAULT_DANGEROUS_OBJECTS_MODEL,
    classify_dangerous_objects_rgb,
    danger_disclaimer_md,
)
from inference.frame_resize import downscale_rgb_max_long_side
from inference.live_mic import _device_for
from inference.people_mood import DEFAULT_EMOTION_MODEL, analyze_people_mood, people_disclaimer_md
from inference.visual_aggression import DEFAULT_VISUAL_VIOLENCE_MODEL, classify_image_rgb

MAX_VISUAL_STREAM_ROWS = 100
MAX_DANGER_STREAM_ROWS = 100
MAX_PEOPLE_STREAM_ROWS = 100
MAX_PERSON_DET_STREAM_ROWS = 100

_live_pd_key: tuple | None = None
_live_pd: Any = None


def _yolo_device_string(gradio_device: str) -> str:
    """Строка устройства для ultralytics (cpu / cuda:N)."""
    td = _device_for(gradio_device)
    if td.type == "cuda":
        return f"cuda:{td.index}" if td.index is not None else "cuda:0"
    return "cpu"


def _get_live_person_detector(model_name: str, confidence: float, gradio_device: str) -> Any:
    """Один экземпляр YOLO на (model, conf, device), чтобы не перезагружать веса каждый кадр."""
    global _live_pd_key, _live_pd
    from inference.person_detector import PersonDetector

    name = (model_name or "").strip() or "yolov8n.pt"
    dev = _yolo_device_string(gradio_device)
    key = (name, float(confidence), dev)
    if _live_pd is None or _live_pd_key != key:
        _live_pd = PersonDetector(
            model_name=name,
            confidence_threshold=float(confidence),
            device=dev,
        )
        _live_pd_key = key
    return _live_pd

# Даунскейл длинной стороны перед ViT/CLIP/Haar (ускорение). 0 — отключить. См. LIVE_CAMERA_MAX_SIDE.
_DEFAULT_LIVE_CAMERA_MAX_SIDE = 448


def _live_inference_max_side() -> int:
    raw = (os.environ.get("LIVE_CAMERA_MAX_SIDE") or str(_DEFAULT_LIVE_CAMERA_MAX_SIDE)).strip()
    try:
        v = int(raw, 10)
    except ValueError:
        v = _DEFAULT_LIVE_CAMERA_MAX_SIDE
    return max(0, min(v, 4096))


def _frame_signature(rgb: np.ndarray) -> float:
    g = rgb[::12, ::12, :].astype(np.float64)
    return float(g.mean())


def _preprocess_webcam_rgb(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim < 2 or arr.size < 64 or arr.ndim == 2:
        return None
    if arr.shape[-1] < 3:
        return None
    return np.clip(arr[..., :3], 0, 255).astype(np.uint8)


def process_online_camera_frame(
    image: Any,
    visual_rows: list[dict[str, Any]],
    danger_rows: list[dict[str, Any]],
    people_rows: list[dict[str, Any]],
    person_det_rows: list[dict[str, Any]],
    last_cam_sig: float | None,
    *,
    aggression_enabled: bool,
    aggression_model_id: str,
    danger_enabled: bool,
    danger_clip_model_id: str,
    people_enabled: bool,
    emotion_model_id: str,
    person_detect_enabled: bool = False,
    person_det_model: str = "yolov8n.pt",
    person_det_conf: float = 0.35,
    gradio_device: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    float | None,
]:
    """
    Один тик камеры: при смене кадра (по сигнатуре) обновляет один или несколько журналов.
    Возвращает (visual_rows, danger_rows, people_rows, person_det_rows, last_cam_sig).
    """
    if (
        not aggression_enabled
        and not danger_enabled
        and not people_enabled
        and not person_detect_enabled
    ):
        return visual_rows, danger_rows, people_rows, person_det_rows, last_cam_sig

    rgb = _preprocess_webcam_rgb(image)
    if rgb is None:
        return visual_rows, danger_rows, people_rows, person_det_rows, last_cam_sig

    sig = _frame_signature(rgb)
    if last_cam_sig is not None and abs(float(last_cam_sig) - sig) < 0.25:
        return visual_rows, danger_rows, people_rows, person_det_rows, last_cam_sig

    rgb_inf = downscale_rgb_max_long_side(rgb, _live_inference_max_side())

    td = _device_for(gradio_device)
    v_out = list(visual_rows)
    d_out = list(danger_rows)
    p_out = list(people_rows)
    pd_out = list(person_det_rows)
    did_any = False

    if aggression_enabled:
        mid = (aggression_model_id or "").strip() or DEFAULT_VISUAL_VIOLENCE_MODEL
        try:
            vprob, top_lab = classify_image_rgb(rgb_inf, model_id=mid, torch_device=td)
            row: dict[str, Any] = {
                "t": datetime.now().strftime("%H:%M:%S"),
                "violence_probability": round(float(vprob), 4),
                "top_label": str(top_lab)[:120],
                "_sig": sig,
            }
        except Exception as e:  # noqa: BLE001
            row = {
                "t": datetime.now().strftime("%H:%M:%S"),
                "violence_probability": None,
                "top_label": str(e)[:160],
                "_sig": sig,
                "error": True,
            }
        v_out.append(row)
        if len(v_out) > MAX_VISUAL_STREAM_ROWS:
            v_out = v_out[-MAX_VISUAL_STREAM_ROWS:]
        did_any = True

    if danger_enabled:
        did = (danger_clip_model_id or "").strip() or DEFAULT_DANGEROUS_OBJECTS_MODEL
        try:
            wprob, top_clip, weapon_hint = classify_dangerous_objects_rgb(
                rgb_inf, model_id=did, torch_device=td
            )
            drow: dict[str, Any] = {
                "t": datetime.now().strftime("%H:%M:%S"),
                "weapon_proxy": round(float(wprob), 4),
                "top_label": str(top_clip)[:120],
                "_sig": sig,
            }
            if weapon_hint:
                drow["weapon_hint"] = str(weapon_hint)[:200]
        except Exception as e:  # noqa: BLE001
            drow = {
                "t": datetime.now().strftime("%H:%M:%S"),
                "weapon_proxy": None,
                "top_label": str(e)[:160],
                "_sig": sig,
                "error": True,
            }
        d_out.append(drow)
        if len(d_out) > MAX_DANGER_STREAM_ROWS:
            d_out = d_out[-MAX_DANGER_STREAM_ROWS:]
        did_any = True

    if people_enabled:
        eid = (emotion_model_id or "").strip() or DEFAULT_EMOTION_MODEL
        try:
            am = analyze_people_mood(rgb_inf, emotion_model_id=eid, torch_device=td)
            top_txt = ""
            tk = am.get("emotion_topk") or []
            if tk:
                top_txt = " · ".join(f"{x.get('label', '')}:{x.get('score')}" for x in tk[:3])
            prow: dict[str, Any] = {
                "t": datetime.now().strftime("%H:%M:%S"),
                "face_count": int(am.get("face_count", 0)),
                "emotion": am.get("emotion_primary"),
                "emotion_score": am.get("emotion_confidence"),
                "top_labels": top_txt[:200],
                "_sig": sig,
            }
        except Exception as e:  # noqa: BLE001
            prow = {
                "t": datetime.now().strftime("%H:%M:%S"),
                "face_count": None,
                "emotion": None,
                "emotion_score": None,
                "top_labels": str(e)[:200],
                "_sig": sig,
                "error": True,
            }
        p_out.append(prow)
        if len(p_out) > MAX_PEOPLE_STREAM_ROWS:
            p_out = p_out[-MAX_PEOPLE_STREAM_ROWS:]
        did_any = True

    if person_detect_enabled:
        try:
            det = _get_live_person_detector(
                person_det_model, float(person_det_conf), gradio_device
            )
            dets = det.detect(rgb_inf)
            prow_det: dict[str, Any] = {
                "t": datetime.now().strftime("%H:%M:%S"),
                "people_count": len(dets),
                "detections": [d.to_dict() for d in dets],
                "_sig": sig,
            }
        except Exception as e:  # noqa: BLE001
            prow_det = {
                "t": datetime.now().strftime("%H:%M:%S"),
                "people_count": None,
                "detections": [],
                "top_label": str(e)[:200],
                "_sig": sig,
                "error": True,
            }
        pd_out.append(prow_det)
        if len(pd_out) > MAX_PERSON_DET_STREAM_ROWS:
            pd_out = pd_out[-MAX_PERSON_DET_STREAM_ROWS:]
        did_any = True

    new_sig = float(sig) if did_any else last_cam_sig
    return v_out, d_out, p_out, pd_out, new_sig


def format_visual_stream_markdown(rows: list[dict[str, Any]]) -> str:
    from inference.visual_aggression import visual_disclaimer_md

    lines = ["## Онлайн: визуальный прокси (агрессия)", ""]
    lines.append(visual_disclaimer_md())
    lines.append("")
    if not rows:
        lines.append(
            "_Слой выключен или камера не отдаёт кадры. Включите галочку violence-proxy и запись у компонента «Камера»._"
        )
        return "\n".join(lines)

    for r in rows[-30:]:
        if r.get("error"):
            lines.append(f"- **{r['t']}** — ошибка: `{r.get('top_label', '')}`")
        else:
            lines.append(
                f"- **{r['t']}** — violence_proxy={r.get('violence_probability')} · label=`{r.get('top_label', '')}`"
            )
    lines.append("")
    lines.append("_Задержка зависит от GPU/CPU и частоты снимков Gradio._")
    return "\n".join(lines)


def format_danger_stream_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["## Онлайн: прокси опасных предметов (CLIP)", ""]
    lines.append(danger_disclaimer_md())
    lines.append("")
    if not rows:
        lines.append(
            "_Слой выключен или нет кадров. Включите галочку «опасные предметы» и запись камеры; первый запуск скачает CLIP (~350 MB)._"
        )
        return "\n".join(lines)

    for r in rows[-30:]:
        if r.get("error"):
            lines.append(f"- **{r['t']}** — ошибка: `{r.get('top_label', '')}`")
        else:
            hint = r.get("weapon_hint")
            suf = f" · weapon_family=`{hint}`" if hint else ""
            lines.append(
                f"- **{r['t']}** — weapon_proxy={r.get('weapon_proxy')} · top=`{r.get('top_label', '')}`{suf}"
            )
    lines.append("")
    lines.append(
        "_Многоклассовый CLIP zero-shot; не гарантирует детекцию мелких или частично скрытых объектов._"
    )
    return "\n".join(lines)


def format_people_stream_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["## Онлайн: лица и настроение (прокси)", ""]
    lines.append(people_disclaimer_md())
    lines.append("")
    if not rows:
        lines.append(
            "_Слой выключен или нет кадров. Включите запись камеры и галочку «лица и настроение»; первый запуск скачает модель ViT._"
        )
        return "\n".join(lines)

    for r in rows[-30:]:
        if r.get("error"):
            lines.append(f"- **{r['t']}** — ошибка: `{r.get('top_labels', '')}`")
        else:
            emo = r.get("emotion")
            fc = r.get("face_count")
            if fc == 0 or emo is None:
                lines.append(
                    f"- **{r['t']}** — лиц (анфас): **{fc}** · эмоция по главному лицу: нет лица или обрез"
                )
            else:
                lines.append(
                    f"- **{r['t']}** — лиц: **{fc}** · настроение (главное лицо): `{emo}` · "
                    f"score={r.get('emotion_score')} · top3=`{r.get('top_labels', '')}`"
                )
    lines.append("")
    english_labels = "_Эмодели на HF обычно дают классы на английском (happy, sad …); это наблюдение, не диагноз._"
    lines.append(english_labels)
    return "\n".join(lines)


def format_person_det_stream_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["## Онлайн: детекция людей (YOLO, bbox)", ""]
    lines.append(
        "_Класс `person` (COCO), без трекинга ID. Те же кадры `LIVE_CAMERA_MAX_SIDE`, что и для ViT/CLIP._"
    )
    lines.append("")
    if not rows:
        lines.append(
            "_Слой выключен или нет кадров. Включите «Детекция людей (YOLO)» и запись камеры; первый запуск скачает веса._"
        )
        return "\n".join(lines)

    for r in rows[-25:]:
        if r.get("error"):
            lines.append(f"- **{r['t']}** — ошибка: `{r.get('top_label', '')}`")
        else:
            n = r.get("people_count", 0)
            dets = r.get("detections") or []
            short = ""
            if dets:
                parts = []
                for i, d in enumerate(dets[:5]):
                    bb = d.get("bbox", [])
                    cf = d.get("confidence")
                    parts.append(f"#{i+1} bbox={bb} conf={cf}")
                short = " · " + "; ".join(parts)
                if len(dets) > 5:
                    short += " …"
            lines.append(f"- **{r['t']}** — людей: **{n}**{short}")
    lines.append("")
    lines.append("_Подробные bbox выводятся кратко (до 5 объектов в строке лога)._")
    return "\n".join(lines)


def format_quad_live_markdown(
    verbal_md: str,
    visual_rows: list,
    danger_rows: list,
    people_rows: list,
    person_det_rows: list | None = None,
) -> str:
    if person_det_rows is None:
        person_det_rows = []
    v = format_visual_stream_markdown(visual_rows)
    d = format_danger_stream_markdown(danger_rows)
    p = format_people_stream_markdown(people_rows)
    y = format_person_det_stream_markdown(person_det_rows)
    return f"{verbal_md}\n\n---\n\n{v}\n\n---\n\n{d}\n\n---\n\n{p}\n\n---\n\n{y}"


def format_triple_live_markdown(verbal_md: str, visual_rows: list, danger_rows: list) -> str:
    return format_quad_live_markdown(verbal_md, visual_rows, danger_rows, [], [])


def format_combined_live_markdown(
    verbal_rows: list[dict[str, Any]],
    visual_rows: list,
    danger_rows: list | None = None,
    people_rows: list | None = None,
    person_det_rows: list | None = None,
) -> str:
    from inference.live_mic import format_verbal_log

    if danger_rows is None:
        danger_rows = []
    if people_rows is None:
        people_rows = []
    if person_det_rows is None:
        person_det_rows = []
    return format_quad_live_markdown(
        format_verbal_log(verbal_rows), visual_rows, danger_rows, people_rows, person_det_rows
    )


# Обратная совместимость: только violence (без danger-лога)
def process_streaming_visual_frame(
    image: np.ndarray | None,
    prior_rows: list[dict[str, Any]],
    *,
    enabled: bool,
    model_id: str,
    gradio_device: str,
) -> list[dict[str, Any]]:
    v, _, _, _, _ = process_online_camera_frame(
        image,
        prior_rows,
        [],
        [],
        [],
        None,
        aggression_enabled=enabled,
        aggression_model_id=model_id,
        danger_enabled=False,
        danger_clip_model_id="",
        people_enabled=False,
        emotion_model_id="",
        person_detect_enabled=False,
        person_det_model="yolov8n.pt",
        person_det_conf=0.35,
        gradio_device=gradio_device,
    )
    return v
