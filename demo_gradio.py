"""
Gradio: загрузите видео → извлечение аудио (ffmpeg) → Whisper → вербальная голова.

Запуск из корня проекта:
  python demo_gradio.py
"""

from __future__ import annotations

import json
import os
import sys
import threading

import numpy as np
import pandas as pd
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Gradio/ffmpy вызывает бинарь `ffmpeg`; imageio-ffmpeg кладёт другой файл — делаем shim в PATH.
from inference.audio_extract import (  # noqa: E402
    ensure_gradio_ffmpy_finds_ffmpeg,
    resolve_ffmpeg_executable,
)

_ffmpeg_shim = ensure_gradio_ffmpy_finds_ffmpeg()
_ffmpeg_raw = resolve_ffmpeg_executable()

import gradio as gr  # noqa: E402 — после правки PATH

from inference.analyze_media import analyze_video_upload  # noqa: E402
from inference.incident_signals import prepend_online_incident  # noqa: E402
from inference.live_mic import format_verbal_log, process_streaming_chunk  # noqa: E402
from inference.live_visual import (
    format_quad_live_markdown,
    process_online_camera_frame,
)
from inference.visual_aggression import DEFAULT_VISUAL_VIOLENCE_MODEL  # noqa: E402
from inference.dangerous_objects import DEFAULT_DANGEROUS_OBJECTS_MODEL  # noqa: E402
from inference.people_mood import DEFAULT_EMOTION_MODEL  # noqa: E402

from inference.person_detector import PersonDetector, analyze_live_frame_people  # noqa: E402
from inference.person_tracker import analyze_live_frame_tracking  # noqa: E402
from inference.video_person_analyzer import analyze_video_people  # noqa: E402
from inference.video_social_analyzer import analyze_live_frame_social, analyze_video_social  # noqa: E402
from inference.video_pose_analyzer import analyze_live_frame_pose, analyze_video_pose  # noqa: E402
from actions.video_action_analyzer import analyze_live_frame_actions, analyze_video_actions  # noqa: E402
from audio.video_audio_analyzer import analyze_video_audio  # noqa: E402
from fusion.video_fusion_analyzer import analyze_video_fusion  # noqa: E402

_live_online_lock = threading.Lock()

_stage1_detector_key: tuple | None = None
_stage1_detector: PersonDetector | None = None


def _live_cam_stream_every_seconds() -> float:
    """Интервал обновления кадра онлайна; можно задать LIVE_CAM_STREAM_SEC (сек.), по умолчанию 3."""
    raw = (os.environ.get("LIVE_CAM_STREAM_SEC") or "3").strip().replace(",", ".")
    try:
        x = float(raw)
    except ValueError:
        x = 3.0
    return max(0.5, min(x, 60.0))



def _coerce_upload_path(value) -> str | None:  # noqa: ANN401
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        s = str(value)
        return s if s else None
    if isinstance(value, (list, tuple)) and value:
        return _coerce_upload_path(value[0])
    if isinstance(value, dict):
        for k in ("name", "path", "video"):
            inner = value.get(k)
            if inner:
                return _coerce_upload_path(inner)
        return None
    return str(value)


def _run(
    webcam_clip,
    disk_file,
    whisper_model,
    verbal_ckpt_dir,
    chunk_s,
    stride_s,
    device_choice,
    visual_enable,
    visual_model_hf,
):
    chosen = _coerce_upload_path(webcam_clip) or _coerce_upload_path(disk_file)
    video_path = chosen
    if not video_path:
        return (
            "### Нет входа\n\nЗагрузите файл **или** запишите короткий ролик с **веб-камеры** (и остановите запись перед разбором).",
            "",
            None,
        )
    device = None
    dc = (device_choice or "auto").strip().lower()
    if dc == "cpu":
        device = "cpu"
    elif dc.startswith("cuda"):
        device = dc

    verbal_ckpt = None
    verbal_ckpt_txt = (verbal_ckpt_dir or "").strip()
    if verbal_ckpt_txt:
        verbal_ckpt = Path(verbal_ckpt_txt)

    try:
        md, payload = analyze_video_upload(
            video_path,
            whisper_model=whisper_model,
            verbal_ckpt=verbal_ckpt,
            chunk_s=float(chunk_s),
            stride_s=float(stride_s),
            asr_torch_device=device,
            enable_visual_aggression=bool(visual_enable),
            visual_model_id=(visual_model_hf or "").strip() or None,
        )
        txt = json.dumps(payload, ensure_ascii=False, indent=2)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as fp:
            fp.write(txt)
            fp.flush()
            json_path = fp.name
        return md, txt, json_path
    except (ValueError, FileNotFoundError) as exc:
        return f"# Ошибка\n\n{exc}", "", None
    except Exception as exc:  # noqa: BLE001
        err = f"# Ошибка\n\n```\n{exc}\n```\n\n{traceback.format_exc()}"
        return err, "", None


def _live_online_state_normalize(state):
    if state is None or not isinstance(state, dict):
        return {
            "verbal": [],
            "visual": [],
            "danger": [],
            "people": [],
            "person_det": [],
            "last_cam_sig": None,
        }
    verb = state.get("verbal")
    vis = state.get("visual")
    dan = state.get("danger")
    pep = state.get("people")
    pdt = state.get("person_det")
    lcs = state.get("last_cam_sig")
    return {
        "verbal": verb if isinstance(verb, list) else [],
        "visual": vis if isinstance(vis, list) else [],
        "danger": dan if isinstance(dan, list) else [],
        "people": pep if isinstance(pep, list) else [],
        "person_det": pdt if isinstance(pdt, list) else [],
        "last_cam_sig": float(lcs) if lcs is not None and isinstance(lcs, (int, float)) else None,
    }


def _live_stream_audio(audio, state, whisper_m, ckpt_txt, dev_choice, win_s, vis_enabled, visual_model_hf):
    st = _live_online_state_normalize(state)
    whisper_m = whisper_m or "openai/whisper-tiny"
    ckpt = (ckpt_txt or "").strip() or str(ROOT / "checkpoints" / "verbal-latest")

    with _live_online_lock:
        md_part, verbal_new = process_streaming_chunk(
            audio,
            st["verbal"],
            whisper_model=whisper_m,
            verbal_ckpt=ckpt,
            gradio_device=dev_choice or "auto",
            window_s=float(win_s),
        )
        st_out = dict(st)
        st_out["verbal"] = verbal_new
        if md_part.startswith("### "):
            verbal_md = md_part
        else:
            verbal_md = format_verbal_log(verbal_new)
        md_quad = format_quad_live_markdown(
            verbal_md,
            st_out["visual"],
            st_out["danger"],
            st_out["people"],
            st_out.get("person_det", []),
        )
        md = prepend_online_incident(st_out["verbal"], st_out["visual"], md_quad)
        return md, st_out


def _live_stream_camera(
    cam_image,
    state,
    vis_enabled,
    visual_model_hf,
    danger_enabled,
    danger_model_hf,
    people_enabled,
    emotion_model_hf,
    person_det_enabled,
    person_det_model,
    person_det_conf,
    dev_choice,
):
    st = _live_online_state_normalize(state)
    with _live_online_lock:
        st_out = dict(st)
        v, d, p, pdet, sig = process_online_camera_frame(
            cam_image,
            st_out["visual"],
            st_out["danger"],
            st_out["people"],
            st_out.get("person_det", []),
            st_out.get("last_cam_sig"),
            aggression_enabled=bool(vis_enabled),
            aggression_model_id=(visual_model_hf or "").strip() or DEFAULT_VISUAL_VIOLENCE_MODEL,
            danger_enabled=bool(danger_enabled),
            danger_clip_model_id=(danger_model_hf or "").strip() or DEFAULT_DANGEROUS_OBJECTS_MODEL,
            people_enabled=bool(people_enabled),
            emotion_model_id=(emotion_model_hf or "").strip() or DEFAULT_EMOTION_MODEL,
            person_detect_enabled=bool(person_det_enabled),
            person_det_model=(person_det_model or "").strip() or "yolov8n.pt",
            person_det_conf=float(person_det_conf),
            gradio_device=dev_choice or "auto",
        )
        st_out["visual"] = v
        st_out["danger"] = d
        st_out["people"] = p
        st_out["person_det"] = pdet
        st_out["last_cam_sig"] = sig
        md_quad = format_quad_live_markdown(
            format_verbal_log(st_out["verbal"]),
            st_out["visual"],
            st_out["danger"],
            st_out["people"],
            st_out.get("person_det", []),
        )
        md = prepend_online_incident(st_out["verbal"], st_out["visual"], md_quad)
        return md, st_out


def _stage1_torch_device(dev_choice: str | None) -> str | None:
    dc = (dev_choice or "auto").strip().lower()
    if dc == "auto":
        return None
    if dc == "cpu":
        return "cpu"
    if dc.startswith("cuda"):
        return dc
    return None


def _stage1_get_detector(model_name: str, conf: float, dev_choice: str | None) -> PersonDetector:
    global _stage1_detector_key, _stage1_detector
    name = (model_name or "").strip() or "yolov8n.pt"
    device = _stage1_torch_device(dev_choice)
    key = (name, float(conf), device or "")
    if _stage1_detector is None or _stage1_detector_key != key:
        _stage1_detector = PersonDetector(
            model_name=name,
            confidence_threshold=float(conf),
            device=device,
        )
        _stage1_detector_key = key
    return _stage1_detector


def _stage1_run_video(
    video_file,
    sample_every_sec,
    max_frames,
    conf,
    model_name,
    dev_choice,
):
    path_str = _coerce_upload_path(video_file)
    if not path_str:
        return None, "", "### Нет файла\n\nЗагрузите видео (mp4, webm, …)."
    try:
        payload, preview = analyze_video_people(
            path_str,
            sample_every_sec=float(sample_every_sec),
            max_frames=int(max_frames),
            confidence_threshold=float(conf),
            model_name=(model_name or "").strip() or "yolov8n.pt",
            device=_stage1_torch_device(dev_choice),
        )
    except Exception as exc:  # noqa: BLE001
        return None, json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), f"### Ошибка\n\n`{exc}`"
    txt = json.dumps(payload, ensure_ascii=False, indent=2)
    if payload.get("error"):
        summ = (
            "### Этап 1 — сводка\n"
            f"- **Ошибка:** {payload['error']}\n"
        )
        return preview, txt, summ
    summ = (
        "### Этап 1 — сводка\n"
        f"- Кадров обработано: **{payload.get('frames_analyzed', 0)}**\n"
        f"- Максимум людей на кадре: **{payload['summary']['max_people']}**\n"
        f"- Среднее число людей: **{payload['summary']['avg_people']}**\n"
    )
    return preview, txt, summ


def _stage1_run_webcam_snapshot(image, conf, model_name, dev_choice):
    if image is None:
        return None, '{"people_count": 0, "detections": [], "note": "нет кадра"}'
    try:
        det = _stage1_get_detector(str(model_name or "yolov8n.pt"), float(conf), dev_choice)
        vis, pl = analyze_live_frame_people(image, detector=det)
        return vis, json.dumps(pl, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        return image, json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2)


def _stage1_stream_online_frame(image, conf, model_name, dev_choice):
    """Поток с веб-камеры: каждый тик — детекция людей и отрисовка bbox (онлайн-видео)."""
    if image is None:
        return None, ""
    try:
        frame = np.asarray(image)
        det = _stage1_get_detector(str(model_name or "yolov8n.pt"), float(conf), dev_choice)
        vis, pl = analyze_live_frame_people(frame, detector=det)
        return vis, json.dumps(pl, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        return np.asarray(image), json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2)


def _stage2_run_video_tracking(
    video_file,
    sample_every_sec,
    max_frames,
    conf,
    model_name,
    dev_choice,
):
    path_str = _coerce_upload_path(video_file)
    if not path_str:
        return None, "", "### Нет файла\n\nЗагрузите видео."
    try:
        payload, preview = analyze_video_tracking(
            path_str,
            sample_every_sec=float(sample_every_sec),
            max_frames=int(max_frames),
            confidence_threshold=float(conf),
            model_name=(model_name or "").strip() or "yolov8n.pt",
            device=_stage1_torch_device(dev_choice),
        )
    except Exception as exc:  # noqa: BLE001
        return None, json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), f"### Ошибка\n\n`{exc}`"
    txt = json.dumps(payload, ensure_ascii=False, indent=2)
    if payload.get("error"):
        summ = (
            "### Этап 2 — сводка\n"
            f"- **Ошибка:** {payload['error']}\n"
        )
        return preview, txt, summ
    s = payload.get("summary", {})
    summ = (
        "### Этап 2 — сводка\n"
        f"- Кадров обработано: **{payload.get('frames_analyzed', 0)}**\n"
        f"- Уникальных треков (людей): **{s.get('unique_people', 0)}**\n"
        f"- Максимум людей на кадре: **{s.get('max_people_in_frame', 0)}**\n"
        f"- Среднее число людей на кадре: **{s.get('avg_people_in_frame', 0)}**\n"
    )
    return preview, txt, summ


def _stage2_stream_tracking_frame(image, state, conf, model_name, dev_choice):
    """Онлайн-поток: трекинг с сохранением state между кадрами."""
    if image is None:
        return None, "", "### Активные треки\n\n—", state or {}
    try:
        frame = np.asarray(image)
        st = dict(state or {})
        st.setdefault("fps", 30.0)
        vis, pl, st_out = analyze_live_frame_tracking(
            frame,
            st,
            model_name=str(model_name or "yolov8n.pt"),
            device=_stage1_torch_device(dev_choice),
            confidence=float(conf),
        )
        txt = json.dumps(pl, ensure_ascii=False, indent=2)
        ids = [str(x["track_id"]) for x in pl.get("tracked_people", [])]
        active_md = "### Активные треки\n\n" + ("`" + ", ".join(ids) + "`" if ids else "—")
        return vis, txt, active_md, st_out
    except Exception as exc:  # noqa: BLE001
        st = dict(state or {})
        return (
            np.asarray(image),
            json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2),
            f"### Ошибка\n\n`{exc}`",
            st,
        )


def _stage2_reset_tracking_state():
    return {}


def _social_signals_dataframe(social_signals: list | None) -> pd.DataFrame:
    cols = ["signal_type", "severity", "track_ids", "description", "timestamp_sec"]
    if not social_signals:
        return pd.DataFrame(columns=cols)
    rows = []
    for s in social_signals:
        if not isinstance(s, dict):
            continue
        rows.append(
            {
                "signal_type": s.get("signal_type", ""),
                "severity": s.get("severity", 0.0),
                "track_ids": str(s.get("track_ids", [])),
                "description": (s.get("description") or "")[:200],
                "timestamp_sec": s.get("timestamp_sec", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _stage3_run_video_social(
    video_file,
    sample_every_sec,
    max_frames,
    conf,
    model_name,
    dev_choice,
):
    path_str = _coerce_upload_path(video_file)
    if not path_str:
        return None, "", "### Нет файла\n\nЗагрузите видео."
    try:
        payload, preview = analyze_video_social(
            path_str,
            sample_every_sec=float(sample_every_sec),
            max_frames=int(max_frames),
            confidence_threshold=float(conf),
            model_name=(model_name or "").strip() or "yolov8n.pt",
            device=_stage1_torch_device(dev_choice),
        )
    except Exception as exc:  # noqa: BLE001
        return None, json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), f"### Ошибка\n\n`{exc}`"
    txt = json.dumps(payload, ensure_ascii=False, indent=2)
    if payload.get("error"):
        summ = "### Этап 3 — сводка\n" + f"- **Ошибка:** {payload['error']}\n"
        return preview, txt, summ
    sm = payload.get("summary", {})
    types_txt = ", ".join(f"{k}: **{v}**" for k, v in sorted((sm.get("signal_types") or {}).items()))
    summ = (
        "### Этап 3 — Social Interaction Analysis\n"
        f"- Кадров обработано: **{payload.get('frames_analyzed', 0)}**\n"
        f"- Всего сигналов (кадры суммарно): **{sm.get('signals_count', 0)}**\n"
        f"- Макс. severity: **{sm.get('max_social_severity', 0)}**\n"
        f"- По типам: {types_txt or '—'}\n"
        "\n*Только social risk signals; финальный буллинг не определяется.*\n"
    )
    return preview, txt, summ


def _stage3_stream_social_frame(image, state, conf, model_name, dev_choice):
    if image is None:
        empty_df = _social_signals_dataframe([])
        return None, empty_df, "", "### Сигналы на кадре\n\n—", state or {}
    try:
        frame = np.asarray(image)
        st = dict(state or {})
        st.setdefault("fps", 30.0)
        vis, pl, st_out = analyze_live_frame_social(
            frame,
            st,
            model_name=str(model_name or "yolov8n.pt"),
            device=_stage1_torch_device(dev_choice),
            confidence=float(conf),
        )
        txt = json.dumps(pl, ensure_ascii=False, indent=2)
        df = _social_signals_dataframe(pl.get("social_signals", []))
        note = "### Сигналы на кадре\n\n" + (
            "\n".join(
                f"- **{r['signal_type']}** (sev={float(r['severity']):.2f}): {str(r['description'])[:120]}"
                for _, r in df.iterrows()
            )
            if len(df)
            else "—"
        )
        return vis, df, txt, note, st_out
    except Exception as exc:  # noqa: BLE001
        st = dict(state or {})
        return (
            np.asarray(image),
            _social_signals_dataframe([]),
            json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2),
            f"### Ошибка\n\n`{exc}`",
            st,
        )


def _stage3_reset_social_state():
    return {}


def _pose_risk_signals_dataframe(signals: list | None) -> pd.DataFrame:
    cols = ["signal_type", "severity", "track_id", "description", "timestamp_sec"]
    if not signals:
        return pd.DataFrame(columns=cols)
    rows = []
    for s in signals:
        if not isinstance(s, dict):
            continue
        rows.append(
            {
                "signal_type": s.get("signal_type", ""),
                "severity": s.get("severity", 0.0),
                "track_id": s.get("track_id", ""),
                "description": (s.get("description") or "")[:220],
                "timestamp_sec": s.get("timestamp_sec", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _stage4_run_video_pose(
    video_file,
    sample_every_sec,
    max_frames,
    conf,
    det_model,
    pose_model,
    dev_choice,
):
    path_str = _coerce_upload_path(video_file)
    if not path_str:
        empty = _pose_risk_signals_dataframe([])
        return None, "", "### Нет файла\n\nЗагрузите видео.", empty
    try:
        payload, preview = analyze_video_pose(
            path_str,
            sample_every_sec=float(sample_every_sec),
            max_frames=int(max_frames),
            confidence_threshold=float(conf),
            detector_model=(det_model or "").strip() or "yolov8n.pt",
            pose_model=(pose_model or "").strip() or "yolov8n-pose.pt",
            device=_stage1_torch_device(dev_choice),
        )
    except Exception as exc:  # noqa: BLE001
        empty = _pose_risk_signals_dataframe([])
        return None, json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), f"### Ошибка\n\n`{exc}`", empty
    txt = json.dumps(payload, ensure_ascii=False, indent=2)
    sig_df = _pose_risk_signals_dataframe(payload.get("pose_signals", []))
    if payload.get("error"):
        summ = "### Этап 4 — сводка\n" + f"- **Ошибка:** {payload['error']}\n"
        return preview, txt, summ, sig_df
    sm = payload.get("summary", {})
    types_txt = ", ".join(f"{k}: **{v}**" for k, v in sorted((sm.get("signal_types") or {}).items()))
    summ = (
        "### Этап 4 — Pose Estimation\n"
        f"- Кадров: **{sm.get('frames_analyzed', 0)}**\n"
        f"- Сигналов позы (всего): **{sm.get('pose_signals_count', 0)}**\n"
        f"- Макс. severity: **{sm.get('max_pose_severity', 0)}**\n"
        f"- По типам: {types_txt or '—'}\n"
        "\n*Только pose risk signals; без финального bullying score.*\n"
    )
    return preview, txt, summ, sig_df


def _stage4_stream_pose_frame(image, state, conf, det_model, pose_model, dev_choice):
    if image is None:
        empty = _pose_risk_signals_dataframe([])
        return None, empty, "", "### Pose risk signals\n\n—", state or {}
    try:
        frame = np.asarray(image)
        st = dict(state or {})
        st.setdefault("fps", 30.0)
        vis, pl, st_out = analyze_live_frame_pose(
            frame,
            st,
            detector_model=str(det_model or "yolov8n.pt"),
            pose_model=str(pose_model or "yolov8n-pose.pt"),
            device=_stage1_torch_device(dev_choice),
            confidence=float(conf),
        )
        txt = json.dumps(pl, ensure_ascii=False, indent=2)
        df = _pose_risk_signals_dataframe(pl.get("pose_signals", []))
        sigs = pl.get("pose_signals") or []
        if sigs:
            note = "### Pose risk signals\n\n" + "\n".join(
                f"- **{s.get('signal_type')}** (track {s.get('track_id')}, sev={float(s.get('severity', 0)):.2f})"
                for s in sigs[:12]
            )
        else:
            note = "### Pose risk signals\n\n—"
        return vis, df, txt, note, st_out
    except Exception as exc:  # noqa: BLE001
        st = dict(state or {})
        return (
            np.asarray(image),
            _pose_risk_signals_dataframe([]),
            json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2),
            f"### Ошибка\n\n`{exc}`",
            st,
        )


def _stage4_reset_pose_state():
    return {}


def _action_events_dataframe(actions: list | None) -> pd.DataFrame:
    cols = ["track_id", "action_type", "confidence", "severity", "timestamp_sec", "description"]
    if not actions:
        return pd.DataFrame(columns=cols)
    rows = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        rows.append(
            {
                "track_id": a.get("track_id", ""),
                "action_type": a.get("action_type", ""),
                "confidence": a.get("confidence", 0.0),
                "severity": a.get("severity", 0.0),
                "timestamp_sec": a.get("timestamp_sec", 0.0),
                "description": (a.get("description") or "")[:200],
            }
        )
    return pd.DataFrame(rows)


def _stage5_run_video_actions(
    video_file,
    sample_every_sec,
    max_frames,
    conf,
    det_model,
    pose_model,
    action_backend,
    clip_len,
    dev_choice,
):
    path_str = _coerce_upload_path(video_file)
    if not path_str:
        empty = _action_events_dataframe([])
        return None, "", "### Нет файла\n\nЗагрузите видео.", empty
    try:
        payload, preview = analyze_video_actions(
            path_str,
            sample_every_sec=float(sample_every_sec),
            max_frames=int(max_frames),
            confidence_threshold=float(conf),
            detector_model=(det_model or "").strip() or "yolov8n.pt",
            pose_model=(pose_model or "").strip() or "yolov8n-pose.pt",
            action_model_name=(action_backend or "heuristic").strip().lower(),
            clip_length=int(clip_len),
            device=_stage1_torch_device(dev_choice),
        )
    except Exception as exc:  # noqa: BLE001
        empty = _action_events_dataframe([])
        return None, json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), f"### Ошибка\n\n`{exc}`", empty
    txt = json.dumps(payload, ensure_ascii=False, indent=2)
    df = _action_events_dataframe(payload.get("actions", []))
    if payload.get("error"):
        summ = "### Этап 5 — сводка\n" + f"- **Ошибка:** {payload['error']}\n"
        return preview, txt, summ, df
    sm = payload.get("summary", {})
    summ = (
        "### Этап 5 — Action Recognition\n"
        f"- Кадров: **{sm.get('frames_analyzed', 0)}**\n"
        f"- Сигналов действий: **{sm.get('action_signals_count', 0)}**\n"
        f"- Макс. severity: **{sm.get('max_action_severity', 0)}**\n"
        f"- Типы: **{sm.get('signal_types', {})}**\n"
        "\n*Только action signals; без финального bullying verdict.*\n"
    )
    return preview, txt, summ, df


def _stage5_stream_actions_frame(
    image, state, conf, det_model, pose_model, action_backend, clip_len, dev_choice
):
    if image is None:
        empty = _action_events_dataframe([])
        return None, empty, "", "### Action signals\n\n—", state or {}
    try:
        frame = np.asarray(image)
        st = dict(state or {})
        st.setdefault("fps", 30.0)
        vis, pl, st_out = analyze_live_frame_actions(
            frame,
            st,
            detector_model=str(det_model or "yolov8n.pt"),
            pose_model=str(pose_model or "yolov8n-pose.pt"),
            action_model_name=str(action_backend or "heuristic").strip().lower(),
            clip_length=int(clip_len),
            device=_stage1_torch_device(dev_choice),
            confidence=float(conf),
        )
        txt = json.dumps(pl, ensure_ascii=False, indent=2)
        acts = pl.get("actions") or []
        df = _action_events_dataframe(acts)
        if acts:
            note = "### Action signals\n\n" + "\n".join(
                f"- **{a.get('action_type')}** track {a.get('track_id')} ({float(a.get('confidence', 0)):.2f})"
                for a in acts[:14]
            )
        else:
            note = "### Action signals\n\n—"
        return vis, df, txt, note, st_out
    except Exception as exc:  # noqa: BLE001
        st = dict(state or {})
        return (
            np.asarray(image),
            _action_events_dataframe([]),
            json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2),
            f"### Ошибка\n\n`{exc}`",
            st,
        )


def _stage5_reset_actions_state():
    return {}


def _stage6_asr_device(dev_choice: str | None) -> str:
    """faster-whisper ожидает ``cpu`` или ``cuda`` (не ``cuda:0``)."""
    dc = (dev_choice or "auto").strip().lower()
    if dc.startswith("cuda"):
        return "cuda"
    if dc == "cpu":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _stage6_transcript_dataframe(transcript: list) -> pd.DataFrame:
    cols = ["start_sec", "end_sec", "text", "confidence"]
    if not transcript:
        return pd.DataFrame(columns=cols)
    rows = []
    for row in transcript:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "start_sec": row.get("start_sec", ""),
                "end_sec": row.get("end_sec", ""),
                "text": (row.get("text") or "")[:500],
                "confidence": row.get("confidence"),
            }
        )
    return pd.DataFrame(rows)


def _stage6_audio_signals_dataframe(signals: list) -> pd.DataFrame:
    cols = ["signal_type", "severity", "start_sec", "end_sec", "text", "description"]
    if not signals:
        return pd.DataFrame(columns=cols)
    rows = []
    for s in signals:
        if not isinstance(s, dict):
            continue
        rows.append(
            {
                "signal_type": s.get("signal_type", ""),
                "severity": s.get("severity", 0.0),
                "start_sec": s.get("start_sec", ""),
                "end_sec": s.get("end_sec", ""),
                "text": (s.get("text") or "")[:300] if s.get("text") else "",
                "description": (s.get("description") or "")[:240],
            }
        )
    return pd.DataFrame(rows)


def _stage6_run_video_audio(
    video_file,
    audio_file,
    asr_size: str,
    dev_choice: str,
    lang_choice: str,
):
    path_str = _coerce_upload_path(video_file) or _coerce_upload_path(audio_file)
    empty_t = _stage6_transcript_dataframe([])
    empty_s = _stage6_audio_signals_dataframe([])
    if not path_str:
        return empty_t, empty_s, "{}", "### Этап 6 — Audio + Speech Analysis\n\nЗагрузите **видео** или **аудио** (wav/mp3 и т.п.)."
    lang = (lang_choice or "").strip().lower()
    if lang in ("", "auto"):
        lang_val: str | None = None
    else:
        lang_val = lang
    try:
        payload = analyze_video_audio(
            path_str,
            asr_model_size=(asr_size or "small").strip().lower(),
            asr_device=_stage6_asr_device(dev_choice),
            language=lang_val,
        )
    except Exception as exc:  # noqa: BLE001
        return empty_t, empty_s, json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), f"### Ошибка\n\n`{exc}`"
    txt = json.dumps(payload, ensure_ascii=False, indent=2)
    summ = payload.get("summary") or {}
    note = (payload.get("note") or "").strip()
    if payload.get("error"):
        md = (
            "### Этап 6 — сводка\n"
            f"- **Ошибка:** `{payload['error']}`\n"
            + (f"- {note}\n" if note else "")
            + "\n*Только verbal / audio risk signals; без вердикта «буллинг».*\n"
        )
        return _stage6_transcript_dataframe(payload.get("transcript", [])), _stage6_audio_signals_dataframe(
            payload.get("audio_signals", [])
        ), txt, md
    md = (
        "### Этап 6 — Audio + Speech Analysis\n"
        f"- Аудио в файле: **{summ.get('has_audio', False)}**\n"
        f"- Сегментов речи (VAD): **{summ.get('segments_count', 0)}**\n"
        f"- Audio signals: **{summ.get('audio_signals_count', 0)}**\n"
        f"- Макс. severity: **{summ.get('max_audio_severity', 0)}**\n"
        + (f"- Примечание: {note}\n" if note else "")
        + "\n*Verbal risk / aggressive speech / possible scream — не итоговый вердикт буллинга.*\n"
    )
    return (
        _stage6_transcript_dataframe(payload.get("transcript", [])),
        _stage6_audio_signals_dataframe(payload.get("audio_signals", [])),
        txt,
        md,
    )


def _stage7_html_banner(summary: dict) -> str:
    lvl = str(summary.get("risk_level") or "green")
    border = {
        "green": "#2e7d32",
        "yellow": "#f9a825",
        "orange": "#ef6c00",
        "red": "#c62828",
    }.get(lvl, "#616161")
    score = summary.get("risk_score", 0)
    meaning = summary.get("risk_level_meaning", "")
    created = bool(summary.get("incident_created", False))
    note = (
        "Bullying risk candidate saved — requires human review. This is NOT a confirmed incident."
        if created
        else "No incident candidate persisted (score below fusion threshold). Sub-analyses may still be useful."
    )
    return (
        f'<div style="border-left:6px solid {border};padding:12px 14px;background:#fafafa;'
        f'border-radius:6px;max-width:960px;">'
        f"<strong>Risk score:</strong> {score} &nbsp;|&nbsp; "
        f'<strong>Risk level:</strong> <span style="color:{border};font-weight:600;">{lvl}</span> '
        f"({meaning})<br/><span style=\"font-size:0.95em;\">{note}</span></div>"
    )


def _stage7_explanation_markdown(payload: dict) -> str:
    inc = payload.get("incident")
    if not inc:
        return (
            "### Этап 7 — explanation\n\n"
            "— (no incident candidate)\n\n"
            "*Only operator-facing risk fusion; never «bullying confirmed».*"
        )
    lines = "\n".join(f"- {e}" for e in (inc.get("explanation") or []))
    iid = inc.get("incident_id") or payload.get("incident_id")
    tracks = inc.get("involved_track_ids", [])
    return (
        "### Этап 7 — Fusion explanation\n\n"
        f"{lines}\n\n"
        f"- **incident_id:** `{iid}`\n"
        f"- **involved_track_ids:** {tracks}\n"
        f"- **time window:** {inc.get('start_sec')} — {inc.get('end_sec')} s\n\n"
        "*Bullying risk candidate — requires human review.*"
    )


def _stage7_run_video_fusion(
    video_file,
    camera_id,
    sample_every_sec,
    max_frames,
    conf,
    det_model,
    pose_model,
    action_backend,
    clip_len,
    dev_choice,
    asr_size,
    asr_dev,
    lang_choice,
):
    path_str = _coerce_upload_path(video_file)
    empty_summ = {
        "risk_score": 0.0,
        "risk_level": "green",
        "risk_level_meaning": "normal",
        "incident_created": False,
    }
    if not path_str:
        return (
            None,
            _stage7_html_banner(empty_summ),
            "{}",
            "### Этап 7 — Fusion Risk Analysis\n\nЗагрузите видеофайл.",
        )
    lang = (lang_choice or "").strip().lower()
    lang_val: str | None = None if lang in ("", "auto") else lang
    try:
        payload, preview = analyze_video_fusion(
            path_str,
            camera_id=(camera_id or "demo_camera").strip() or "demo_camera",
            incident_store_path=str(ROOT / "data" / "incidents.json"),
            sample_every_sec=float(sample_every_sec),
            max_frames=int(max_frames),
            confidence_threshold=float(conf),
            detector_model=(det_model or "").strip() or "yolov8n.pt",
            pose_model=(pose_model or "").strip() or "yolov8n-pose.pt",
            action_backend=(action_backend or "heuristic").strip().lower(),
            clip_length=int(clip_len),
            device=_stage1_torch_device(dev_choice),
            asr_model_size=str(asr_size or "small").lower(),
            asr_device=_stage6_asr_device(asr_dev),
            language=lang_val,
        )
    except Exception as exc:  # noqa: BLE001
        err = {"error": str(exc)}
        return (
            None,
            _stage7_html_banner(empty_summ),
            json.dumps(err, ensure_ascii=False, indent=2),
            f"### Ошибка\n\n`{exc}`",
        )
    summ = payload.get("summary") or empty_summ
    html = _stage7_html_banner(summ)
    txt = json.dumps(payload, ensure_ascii=False, indent=2)
    md = _stage7_explanation_markdown(payload)
    if payload.get("error"):
        md = f"### Fusion\n\n**Ошибка:** `{payload['error']}`\n\n" + md
    return preview, html, txt, md


def _live_clear_online():
    st = {
        "verbal": [],
        "visual": [],
        "danger": [],
        "people": [],
        "person_det": [],
        "last_cam_sig": None,
    }
    body = format_quad_live_markdown(format_verbal_log([]), [], [], [], [])
    md = prepend_online_incident([], [], body)
    return md, st


WHISPER_CHOICES = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Видео / онлайн → ASR, текст, визуальный proxy") as demo:
        gr.Markdown(
            "Два режима: **батч** (файл или короткая запись) и **онлайн**: поток микрофона "
            "**и** поток превью с камеры параллельно (раз в несколько секунд — см. вкладку). "
            "В батче — полный разбор файла включая HF violence-proxy по кадрам при галочке."
        )

        # Явный `gr.Tabs()` нужен для Gradio 5.x — иначе Tab под Blocks могут не показать полоску вкладок.
        with gr.Tabs():
            with gr.Tab("Файл / веб-камера (батч)"):
                gr.Markdown(
                    "- **FFmpeg**: системный или `imageio-ffmpeg` (PATH пополняется при старте).\n"
                    "- Приоритет: **запись с веб-камеры**, иначе файл.\n"
                    "- Звук: при наличии дорожки — аудио → Whisper → текстовый чекпоинт; **только картинка** — отчёт без ASR.\n"
                    "- **Визуальный слой** (галочка): violence/non-violence по подвыборке кадров; не является полным «школьным буллингом»."
                )
                webcam = gr.Video(
                    sources=["webcam"],
                    label="Запись с камеры и микрофона (остановите запись перед «Запустить разбор»)",
                )
                vid = gr.File(
                    label="Либо: видеофайл с диска (mp4/mov/webm/mkv)",
                    file_count="single",
                    file_types=[".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"],
                )
                whisper = gr.Dropdown(choices=WHISPER_CHOICES, value="openai/whisper-tiny", label="Whisper (HF)")
                ckpt_box = gr.Textbox(
                    value=str(ROOT / "checkpoints" / "verbal-latest"),
                    label="Путь к чекпоинту текстовой головы",
                )
                chunk = gr.Slider(4, 45, value=12, step=1, label="Длина окна ASR, сек")
                stride = gr.Slider(2, 20, value=4, step=1, label="Шаг окна ASR, сек")
                device = gr.Radio(["auto", "cpu", "cuda:0"], value="auto", label="Torch device для ASR+текста+визуала")
    
                visual_enable = gr.Checkbox(value=True, label="Визуальный прокси агрессии (выборка кадров)")
                visual_model_hf = gr.Textbox(
                    value=DEFAULT_VISUAL_VIOLENCE_MODEL,
                    label="HF id модели классификации кадров (violence-proxy)",
                    max_lines=1,
                )
    
                run_btn = gr.Button("Запустить разбор")
    
                md_out = gr.Markdown()
                json_view = gr.Textbox(label="JSON (результат)", lines=16)
                json_file = gr.File(label="Скачать JSON")
    
                run_btn.click(
                    fn=_run,
                    inputs=[
                        webcam,
                        vid,
                        whisper,
                        ckpt_box,
                        chunk,
                        stride,
                        device,
                        visual_enable,
                        visual_model_hf,
                    ],
                    outputs=[md_out, json_view, json_file],
                )

            with gr.Tab("Этап 1 — Детекция людей"):
                gr.Markdown(
                    "Детекция людей (**Ultralytics YOLO**), без ASR и violence-proxy других вкладок. "
                    "Первый запуск может скачать `yolov8n.pt`.\n\n"
                    "**Онлайн-видео:** блок ниже — поток с камеры с **bbox на каждом кадре**; включите **запись** у компонента камеры. "
                    "Частота обновления — переменная окружения **`LIVE_CAM_STREAM_SEC`** (по умолчанию ~3 с), как на вкладке «Онлайн». "
                    "Полный мультимодальный лог (речь + ViT + CLIP + YOLO) — на вкладке **«Онлайн — микрофон + камера»**."
                )
                st1_model = gr.Textbox(value="yolov8n.pt", label="Имя/путь модели YOLO (Ultralytics)", max_lines=1)
                st1_conf = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="Порог confidence")
                st1_dev = gr.Radio(["auto", "cpu", "cuda:0"], value="auto", label="Устройство (Torch / YOLO)")

                with gr.Accordion("Онлайн-видео (поток с камеры, bbox)", open=True):
                    gr.Markdown(
                        "Включите **запись** в превью камеры. Справа — кадр с распознанными людьми; ниже — JSON последнего обработанного кадра."
                    )
                    with gr.Row():
                        st1_live_in = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            label="Камера (онлайн-поток)",
                        )
                        st1_live_out = gr.Image(label="Распознавание людей (bbox в реальном времени)")
                    st1_live_json = gr.Textbox(label="JSON последнего кадра потока", lines=10)
                    st1_live_in.stream(
                        fn=_stage1_stream_online_frame,
                        inputs=[st1_live_in, st1_conf, st1_model, st1_dev],
                        outputs=[st1_live_out, st1_live_json],
                        stream_every=_live_cam_stream_every_seconds(),
                        show_progress="hidden",
                    )

                with gr.Accordion("Видеофайл", open=True):
                    st1_vid = gr.File(
                        label="Загрузить видео",
                        file_count="single",
                        file_types=[".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"],
                    )
                    st1_sample = gr.Slider(0.25, 10.0, value=1.0, step=0.25, label="Интервал выборки кадров, сек")
                    st1_maxf = gr.Slider(5, 120, value=30, step=1, label="Максимум проанализированных кадров")
                    st1_btn_vid = gr.Button("Запустить детекцию по видео")
                    st1_preview = gr.Image(label="Превью (bbox)")
                    st1_json = gr.Textbox(label="JSON", lines=18)
                    st1_sum = gr.Markdown()
                    st1_btn_vid.click(
                        _stage1_run_video,
                        [st1_vid, st1_sample, st1_maxf, st1_conf, st1_model, st1_dev],
                        [st1_preview, st1_json, st1_sum],
                    )
    
                with gr.Accordion("Один кадр с веб-камеры", open=True):
                    st1_cam = gr.Image(sources=["webcam"], type="numpy", label="Кадр (снимок / веб-камера)")
                    st1_btn_cam = gr.Button("Детектировать людей на кадре")
                    st1_out = gr.Image(label="Кадр с прямоугольниками")
                    st1_json_cam = gr.Textbox(label="JSON", lines=10)
                    st1_btn_cam.click(
                        _stage1_run_webcam_snapshot,
                        [st1_cam, st1_conf, st1_model, st1_dev],
                        [st1_out, st1_json_cam],
                    )

            with gr.Tab("Этап 2 — Tracking людей"):
                gr.Markdown(
                    "Мультиперсон-трекинг (**ByteTrack** / `supervision`) поверх детекций YOLO. "
                    "Состояние трекера в потоке **сохраняется между кадрами**; кнопка **Сбросить трекинг** обнуляет ID и траектории."
                )
                st2_model = gr.Textbox(value="yolov8n.pt", label="Модель YOLO", max_lines=1)
                st2_conf = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="Порог confidence")
                st2_dev = gr.Radio(["auto", "cpu", "cuda:0"], value="auto", label="Устройство")

                with gr.Accordion("Загрузка видео (tracking)", open=True):
                    st2_vid = gr.File(
                        label="Видео",
                        file_count="single",
                        file_types=[".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"],
                    )
                    st2_sample = gr.Slider(0.1, 5.0, value=0.2, step=0.1, label="Интервал выборки кадров, сек")
                    st2_maxf = gr.Slider(10, 400, value=150, step=10, label="Максимум кадров")
                    st2_btn_vid = gr.Button("Запустить tracking по видео")
                    st2_preview = gr.Image(label="Превью (bbox + ID + траектории)")
                    st2_json = gr.Textbox(label="JSON", lines=18)
                    st2_sum = gr.Markdown()
                    st2_btn_vid.click(
                        _stage2_run_video_tracking,
                        [st2_vid, st2_sample, st2_maxf, st2_conf, st2_model, st2_dev],
                        [st2_preview, st2_json, st2_sum],
                    )

                with gr.Accordion("Live webcam (tracking)", open=True):
                    gr.Markdown(
                        "Включите **запись** у компонента камеры. Справа — кадр с **стабильными track_id** и линиями траекторий; JSON и список активных ID обновляются в потоке."
                    )
                    st2_state = gr.State({})
                    with gr.Row():
                        st2_live_in = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            label="Камера (tracking)",
                        )
                        st2_live_out = gr.Image(label="Tracking (ID + траектории)")
                    st2_live_json = gr.Textbox(label="JSON кадра", lines=12)
                    st2_active = gr.Markdown()
                    st2_reset = gr.Button("Сбросить состояние трекинга")
                    st2_reset.click(_stage2_reset_tracking_state, outputs=st2_state)
                    st2_live_in.stream(
                        fn=_stage2_stream_tracking_frame,
                        inputs=[st2_live_in, st2_state, st2_conf, st2_model, st2_dev],
                        outputs=[st2_live_out, st2_live_json, st2_active, st2_state],
                        stream_every=_live_cam_stream_every_seconds(),
                        show_progress="hidden",
                    )

            with gr.Tab("Этап 3 — Social Interaction Analysis"):
                gr.Markdown(
                    "Поверх **детекции + трекинга** — эвристики социального риска: резкое сближение, преследование, "
                    "окружение, скопление, давление группы на почти неподвижного. "
                    "**Social risk signal detected** / **Potential group pressure** — не финальный диагноз буллинга."
                )
                st3_model = gr.Textbox(value="yolov8n.pt", label="Модель YOLO", max_lines=1)
                st3_conf = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="Порог confidence")
                st3_dev = gr.Radio(["auto", "cpu", "cuda:0"], value="auto", label="Устройство")

                with gr.Accordion("Загрузка видео (social analysis)", open=True):
                    st3_vid = gr.File(
                        label="Видео",
                        file_count="single",
                        file_types=[".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"],
                    )
                    st3_sample = gr.Slider(0.1, 5.0, value=0.2, step=0.1, label="Интервал выборки кадров, сек")
                    st3_maxf = gr.Slider(10, 600, value=300, step=10, label="Максимум кадров")
                    st3_btn_vid = gr.Button("Запустить social analysis по видео")
                    st3_preview = gr.Image(label="Превью (bbox + ID + траектории + сигналы)")
                    st3_json = gr.Textbox(label="JSON", lines=18)
                    st3_sum = gr.Markdown()
                    st3_btn_vid.click(
                        _stage3_run_video_social,
                        [st3_vid, st3_sample, st3_maxf, st3_conf, st3_model, st3_dev],
                        [st3_preview, st3_json, st3_sum],
                    )

                with gr.Accordion("Live webcam (social analysis)", open=True):
                    gr.Markdown(
                        "Включите **запись** у камеры. Кадр: bbox, track_id, траектории и линии между участниками сигналов; "
                        "таблица и JSON обновляются в потоке."
                    )
                    st3_state = gr.State({})
                    with gr.Row():
                        st3_live_in = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            label="Камера",
                        )
                        st3_live_out = gr.Image(label="Кадр + social signals")
                    st3_signals_df = gr.Dataframe(
                        label="Таблица сигналов (текущий кадр)",
                        headers=["signal_type", "severity", "track_ids", "description", "timestamp_sec"],
                        interactive=False,
                    )
                    st3_live_json = gr.Textbox(label="JSON кадра", lines=12)
                    st3_signals_md = gr.Markdown()
                    st3_reset = gr.Button("Сбросить состояние (трекинг + social)")
                    st3_reset.click(_stage3_reset_social_state, outputs=st3_state)
                    st3_live_in.stream(
                        fn=_stage3_stream_social_frame,
                        inputs=[st3_live_in, st3_state, st3_conf, st3_model, st3_dev],
                        outputs=[st3_live_out, st3_signals_df, st3_live_json, st3_signals_md, st3_state],
                        stream_every=_live_cam_stream_every_seconds(),
                        show_progress="hidden",
                    )

            with gr.Tab("Этап 4 — Pose Estimation"):
                gr.Markdown(
                    "**Ultralytics YOLO-pose** (например `yolov8n-pose.pt` или `yolo11n-pose.pt`): скелет на каждом треке, "
                    "pose risk signals (поднятая рука, быстрое движение рук, падение, на земле, наклон корпуса, рука у другого). "
                    "Без идентификации личности и без вердикта «буллинг»."
                )
                st4_det = gr.Textbox(value="yolov8n.pt", label="Модель детекции людей", max_lines=1)
                st4_pose = gr.Textbox(value="yolov8n-pose.pt", label="Модель pose (YOLO-pose)", max_lines=1)
                st4_conf = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="Порог confidence")
                st4_dev = gr.Radio(["auto", "cpu", "cuda:0"], value="auto", label="Устройство")

                with gr.Accordion("Загрузка видео (pose analysis)", open=True):
                    st4_vid = gr.File(
                        label="Видео",
                        file_count="single",
                        file_types=[".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"],
                    )
                    st4_sample = gr.Slider(0.1, 5.0, value=0.2, step=0.1, label="Интервал выборки кадров, сек")
                    st4_maxf = gr.Slider(10, 600, value=300, step=10, label="Максимум кадров")
                    st4_btn_vid = gr.Button("Запустить pose analysis по видео")
                    st4_preview = gr.Image(label="Превью (bbox + ID + скелет)")
                    st4_json = gr.Textbox(label="JSON", lines=16)
                    st4_sum = gr.Markdown()
                    st4_sig_df = gr.Dataframe(
                        label="Таблица pose signals (весь ролик)",
                        headers=["signal_type", "severity", "track_id", "description", "timestamp_sec"],
                        interactive=False,
                    )
                    st4_btn_vid.click(
                        _stage4_run_video_pose,
                        [st4_vid, st4_sample, st4_maxf, st4_conf, st4_det, st4_pose, st4_dev],
                        [st4_preview, st4_json, st4_sum, st4_sig_df],
                    )

                with gr.Accordion("Live webcam (pose)", open=True):
                    gr.Markdown(
                        "Включите **запись** у камеры: bbox, track_id, траектории и скелет; JSON и список pose risk signals."
                    )
                    st4_state = gr.State({})
                    with gr.Row():
                        st4_live_in = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            label="Камера",
                        )
                        st4_live_out = gr.Image(label="Кадр + pose")
                    st4_live_sig_df = gr.Dataframe(
                        label="Pose signals (текущий кадр)",
                        headers=["signal_type", "severity", "track_id", "description", "timestamp_sec"],
                        interactive=False,
                    )
                    st4_live_json = gr.Textbox(label="JSON кадра", lines=12)
                    st4_live_sig_md = gr.Markdown()
                    st4_reset = gr.Button("Сбросить состояние (pose pipeline)")
                    st4_reset.click(_stage4_reset_pose_state, outputs=st4_state)
                    st4_live_in.stream(
                        fn=_stage4_stream_pose_frame,
                        inputs=[st4_live_in, st4_state, st4_conf, st4_det, st4_pose, st4_dev],
                        outputs=[st4_live_out, st4_live_sig_df, st4_live_json, st4_live_sig_md, st4_state],
                        stream_every=_live_cam_stream_every_seconds(),
                        show_progress="hidden",
                    )

            with gr.Tab("Этап 5 — Action Recognition"):
                gr.Markdown(
                    "Временной клип по каждому **track_id** → эвристическое распознавание действия (MVP). "
                    "Подписи на кадре: ``ID n | ACTION conf``. Зависимости **torchvision / pytorchvideo / av** в `requirements.txt` "
                    "для дальнейшего подключения SlowFast/X3D; сейчас по умолчанию **heuristic** backend. "
                    "**Только action signals** — не вердикт буллинга."
                )
                st5_det = gr.Textbox(value="yolov8n.pt", label="YOLO детекция", max_lines=1)
                st5_pose = gr.Textbox(value="yolov8n-pose.pt", label="YOLO pose", max_lines=1)
                st5_backend = gr.Dropdown(
                    choices=["heuristic", "slowfast", "x3d"],
                    value="heuristic",
                    label="Action backend (MVP: heuristic; slowfast/x3d — задел под pytorchvideo)",
                )
                st5_clip = gr.Slider(8, 32, value=16, step=8, label="Длина клипа (кадров) на трек")
                st5_conf = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="Порог confidence")
                st5_dev = gr.Radio(["auto", "cpu", "cuda:0"], value="auto", label="Устройство")

                with gr.Accordion("Загрузка видео (actions)", open=True):
                    st5_vid = gr.File(
                        label="Видео",
                        file_count="single",
                        file_types=[".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"],
                    )
                    st5_sample = gr.Slider(0.1, 5.0, value=0.2, step=0.1, label="Интервал выборки кадров, сек")
                    st5_maxf = gr.Slider(10, 600, value=300, step=10, label="Максимум кадров")
                    st5_btn = gr.Button("Запустить action recognition по видео")
                    st5_preview = gr.Image(label="Превью (bbox + скелет + action)")
                    st5_json = gr.Textbox(label="JSON", lines=14)
                    st5_sum = gr.Markdown()
                    st5_df = gr.Dataframe(
                        label="Таблица actions",
                        headers=["track_id", "action_type", "confidence", "severity", "timestamp_sec", "description"],
                        interactive=False,
                    )
                    st5_btn.click(
                        _stage5_run_video_actions,
                        [
                            st5_vid,
                            st5_sample,
                            st5_maxf,
                            st5_conf,
                            st5_det,
                            st5_pose,
                            st5_backend,
                            st5_clip,
                            st5_dev,
                        ],
                        [st5_preview, st5_json, st5_sum, st5_df],
                    )

                with gr.Accordion("Live webcam (actions)", open=True):
                    gr.Markdown("Запись с камеры: полный пайплайн этапов 1–5 на кадр; сброс обнуляет буферы и модели.")
                    st5_state = gr.State({})
                    with gr.Row():
                        st5_live_in = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            label="Камера",
                        )
                        st5_live_out = gr.Image(label="Live: bbox + pose + action")
                    st5_live_df = gr.Dataframe(
                        label="Actions (текущий кадр)",
                        headers=["track_id", "action_type", "confidence", "severity", "timestamp_sec", "description"],
                        interactive=False,
                    )
                    st5_live_json = gr.Textbox(label="JSON", lines=12)
                    st5_live_md = gr.Markdown()
                    st5_reset = gr.Button("Сбросить состояние (action pipeline)")
                    st5_reset.click(_stage5_reset_actions_state, outputs=st5_state)
                    st5_live_in.stream(
                        fn=_stage5_stream_actions_frame,
                        inputs=[
                            st5_live_in,
                            st5_state,
                            st5_conf,
                            st5_det,
                            st5_pose,
                            st5_backend,
                            st5_clip,
                            st5_dev,
                        ],
                        outputs=[st5_live_out, st5_live_df, st5_live_json, st5_live_md, st5_state],
                        stream_every=_live_cam_stream_every_seconds(),
                        show_progress="hidden",
                    )

            with gr.Tab("Этап 6 — Audio + Speech Analysis"):
                gr.Markdown(
                    "**Пайплайн:** видео/аудио → ffmpeg mono 16 kHz → energy VAD → **faster-whisper** (или HF Whisper) "
                    "→ rule-based текстовые метки → MVP эмоция по громкости/пикам → **audio risk signals**. "
                    "Без аудиодорожки отчёт мягкий, без падения. **Не вердикт буллинга** — только verbal / audio risk."
                )
                st6_vid = gr.File(
                    label="Видео (приоритет, если загружены оба)",
                    file_count="single",
                    file_types=[".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"],
                )
                st6_aud = gr.File(
                    label="Или только аудио",
                    file_count="single",
                    file_types=[".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"],
                )
                st6_asr = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium"],
                    value="small",
                    label="Размер Whisper (faster-whisper / HF)",
                )
                st6_dev = gr.Radio(["auto", "cpu", "cuda:0"], value="auto", label="Устройство ASR")
                st6_lang = gr.Dropdown(
                    choices=[("auto (детект)", "auto"), ("ru", "ru"), ("en", "en")],
                    value="ru",
                    label="Язык ASR",
                )
                st6_btn = gr.Button("Запустить анализ речи и аудио-рисков")
                st6_tr = gr.Dataframe(
                    label="Transcript",
                    headers=["start_sec", "end_sec", "text", "confidence"],
                    interactive=False,
                )
                st6_sig = gr.Dataframe(
                    label="Audio risk signals",
                    headers=["signal_type", "severity", "start_sec", "end_sec", "text", "description"],
                    interactive=False,
                )
                st6_json = gr.Textbox(label="JSON", lines=16)
                st6_sum = gr.Markdown()
                st6_btn.click(
                    _stage6_run_video_audio,
                    [st6_vid, st6_aud, st6_asr, st6_dev, st6_lang],
                    [st6_tr, st6_sig, st6_json, st6_sum],
                )

            with gr.Tab("Этап 7 — Fusion Risk Analysis"):
                gr.Markdown(
                    "Объединение сигналов этапов **3–6** (social, pose, action, audio) + опциональный **context**. "
                    "Результат — **bullying risk candidate** для оператора; формулировки *requires human review*, "
                    "без «bullying confirmed». Инциденты MVP пишутся в `data/incidents.json`."
                )
                st7_vid = gr.File(
                    label="Видео",
                    file_count="single",
                    file_types=[".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"],
                )
                st7_cam = gr.Textbox(value="demo_camera", label="camera_id", max_lines=1)
                st7_det = gr.Textbox(value="yolov8n.pt", label="YOLO детекция", max_lines=1)
                st7_pose = gr.Textbox(value="yolov8n-pose.pt", label="YOLO pose", max_lines=1)
                st7_backend = gr.Dropdown(
                    choices=["heuristic", "slowfast", "x3d"],
                    value="heuristic",
                    label="Action backend",
                )
                st7_clip = gr.Slider(8, 32, value=16, step=8, label="Длина клипа (кадров) на трек")
                st7_conf = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="Порог confidence (детекция/pose)")
                st7_sample = gr.Slider(0.1, 5.0, value=0.2, step=0.1, label="Интервал выборки кадров, сек")
                st7_maxf = gr.Slider(10, 600, value=300, step=10, label="Максимум кадров на этап")
                st7_dev = gr.Radio(["auto", "cpu", "cuda:0"], value="auto", label="Устройство (YOLO/pose/actions)")
                st7_asr = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium"],
                    value="small",
                    label="Whisper (этап 6 внутри fusion)",
                )
                st7_asr_dev = gr.Radio(["auto", "cpu", "cuda:0"], value="auto", label="Устройство ASR")
                st7_lang = gr.Dropdown(
                    choices=[("auto (детект)", "auto"), ("ru", "ru"), ("en", "en")],
                    value="ru",
                    label="Язык ASR",
                )
                st7_btn = gr.Button("Запустить multimodal fusion")
                st7_preview = gr.Image(label="Превью (social / pose / action)")
                st7_risk_html = gr.HTML(label="Risk level")
                st7_json = gr.Textbox(label="JSON", lines=18)
                st7_expl = gr.Markdown()
                st7_btn.click(
                    _stage7_run_video_fusion,
                    [
                        st7_vid,
                        st7_cam,
                        st7_sample,
                        st7_maxf,
                        st7_conf,
                        st7_det,
                        st7_pose,
                        st7_backend,
                        st7_clip,
                        st7_dev,
                        st7_asr,
                        st7_asr_dev,
                        st7_lang,
                    ],
                    [st7_preview, st7_risk_html, st7_json, st7_expl],
                )

            with gr.Tab("Онлайн — микрофон + камера"):
                gr.Markdown(
                    "**Распознавание в реальном времени (дискретно):**\n"
                    "- **Микрофон**: каждые **~2 c** последнее аудио-окно **N секунд** → Whisper → текстовые скоры.\n"
                    "- **Камера** каждые **~3 s** интервал задаётся переменной `LIVE_CAM_STREAM_SEC` (по умолчанию **3**) при включённой записи превью:\n"
                    "  • **violence-proxy** (ViT, галочка) — см. ограничения в блоке ниже;\n"
                    "  • **прокси опасных предметов** — CLIP zero-shot по набору англоязычных промптов (первый запуск ~350 MB);\n"
                    "  • **лица и настроение** — грубый подсчёт анфасных лиц (OpenCV Haar) и эмоция по самому крупному лицу (ViT, HF);\n"
                    "  • **детекция людей YOLO** — bbox класса `person`, без трекинга ID (Ultralytics; первый запуск скачает веса).\n"
                    "Разрешите доступ к устройствам в браузере. Лог ниже объединяет потоки; даунскейл перед ViT/CLIP задаётся "
                    "`LIVE_CAMERA_MAX_SIDE` (по умолчанию **448**, значение **0** — без уменьшения). На CPU возможны секунды задержки."
                )
                with gr.Row():
                    live_mic_in = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="Микрофон (поток)",
                        format="wav",
                        recording=True,
                    )
                    live_cam = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label="Камера — превью для онлайн-кадров (включите запись у компонента)",
                    )
                live_visual_en = gr.Checkbox(value=True, label="Онлайн violence-proxy по кадру с камеры (ViT)")
                live_visual_model = gr.Textbox(
                    value=DEFAULT_VISUAL_VIOLENCE_MODEL,
                    label="HF id модели aggression/violence для кадра",
                    max_lines=1,
                )
                live_danger_en = gr.Checkbox(value=True, label="Онлайн прокси опасных предметов по кадру (CLIP)")
                live_danger_model = gr.Textbox(
                    value=DEFAULT_DANGEROUS_OBJECTS_MODEL,
                    label="HF id CLIP-модели (zero-shot image classification)",
                    max_lines=1,
                )
                live_people_en = gr.Checkbox(value=True, label="Онлайн: лица и настроение по кадру (Haar + ViT)")
                live_people_model = gr.Textbox(
                    value=DEFAULT_EMOTION_MODEL,
                    label="HF id модели классификации эмоций по вырезу лица",
                    max_lines=1,
                )
                live_person_det_en = gr.Checkbox(
                    value=True,
                    label="Онлайн: детекция людей YOLO (bbox, класс person)",
                )
                live_person_det_model = gr.Textbox(
                    value="yolov8n.pt",
                    label="Ultralytics YOLO (люди в онлайн-потоке)",
                    max_lines=1,
                )
                live_person_det_conf = gr.Slider(
                    0.05,
                    0.95,
                    value=0.35,
                    step=0.05,
                    label="Порог confidence YOLO (онлайн)",
                )
                live_whisper = gr.Dropdown(
                    choices=WHISPER_CHOICES,
                    value="openai/whisper-tiny",
                    label="Whisper для онлайна",
                )
                live_ckpt = gr.Textbox(
                    value=str(ROOT / "checkpoints" / "verbal-latest"),
                    label="Чекпоинт текстовой головы",
                )
                live_dev = gr.Radio(
                    ["auto", "cpu", "cuda:0"],
                    value="auto",
                    label="Устройство (Torch: ASR, ViT, CLIP, эмоции, YOLO)",
                )
                live_win = gr.Slider(
                    2,
                    20,
                    value=6,
                    step=1,
                    label="Размер аудио-окна на один проход Whisper (сек)",
                )
                live_md = gr.Markdown(
                    value=prepend_online_incident(
                        [],
                        [],
                        format_quad_live_markdown(format_verbal_log([]), [], [], [], []),
                    ),
                )
                live_state = gr.State(
                    {
                        "verbal": [],
                        "visual": [],
                        "danger": [],
                        "people": [],
                        "person_det": [],
                        "last_cam_sig": None,
                    }
                )
                clear_btn = gr.Button("Очистить онлайн-лог (аудио + журналы камеры)")
                clear_btn.click(fn=_live_clear_online, outputs=[live_md, live_state])
                live_mic_in.stream(
                    fn=_live_stream_audio,
                    inputs=[live_mic_in, live_state, live_whisper, live_ckpt, live_dev, live_win, live_visual_en, live_visual_model],
                    outputs=[live_md, live_state],
                    stream_every=2.0,
                    show_progress="hidden",
                )
                live_cam.stream(
                    fn=_live_stream_camera,
                    inputs=[
                        live_cam,
                        live_state,
                        live_visual_en,
                        live_visual_model,
                        live_danger_en,
                        live_danger_model,
                        live_people_en,
                        live_people_model,
                        live_person_det_en,
                        live_person_det_model,
                        live_person_det_conf,
                        live_dev,
                    ],
                    outputs=[live_md, live_state],
                    stream_every=_live_cam_stream_every_seconds(),
                    show_progress="hidden",
                )

    return demo


if __name__ == "__main__":
    if not _ffmpeg_raw:
        print(
            "\n[ВНИМАНИЕ] Нет исполняемого ffmpeg ни в PATH, ни через `imageio-ffmpeg`.\n"
            "Выполните:  python -m pip install imageio-ffmpeg\n"
            "или поставьте FFmpeg в систему (winget install Gyan.FFmpeg) и откройте новое окно терминала.\n",
            flush=True,
        )
    else:
        print(f"[ffmpeg] исходный бинарь: {_ffmpeg_raw}")
        if _ffmpeg_shim and Path(_ffmpeg_raw).resolve() != Path(_ffmpeg_shim).resolve():
            print(f"[ffmpeg] для Gradio ffmpy (копия как ffmpeg.exe): {_ffmpeg_shim}")

    demo = build_ui()

    preferred = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    last_exc: BaseException | None = None

    for offset in range(30):
        p = preferred + offset
        try:
            print(
                f"Запуск Gradio: http://127.0.0.1:{p} "
                f"(можно принудительно: set GRADIO_SERVER_PORT={preferred})"
            )
            demo.launch(server_name="127.0.0.1", server_port=p)
            break
        except OSError as e:
            msg = str(e).lower()
            if "cannot find empty port" in msg or "address already in use" in msg:
                last_exc = e
                continue
            raise
    else:
        hint = (
            "Закройте процесс, занявший порт "
            f"(например: netstat -ano | findstr :{preferred}), или задайте GRADIO_SERVER_PORT."
        )
        raise SystemExit(f"Не нашли свободный порт с {preferred} по {preferred + 29}. {hint}") from last_exc
