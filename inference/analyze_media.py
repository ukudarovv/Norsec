"""Сборка пайплайна: видео → WAV → сегменты ASR → вербальный risk head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import soundfile as sf
import torch

from inference.audio_extract import (
    extract_wav_mono_16k,
    ffmpeg_available,
    probe_has_audio_stream,
)
from inference.transcribe_chunked import transcribe_chunks
from inference.verbal_head import LABEL_NAMES, load_verbal_model, verbal_scores_batch
from inference.incident_signals import fuse_batch_incident, incident_batch_markdown_lines
from inference.visual_aggression import (
    DEFAULT_VISUAL_VIOLENCE_MODEL,
    analyze_visual_aggression,
    visual_results_markdown_section,
)


@dataclass
class SegmentInsight:
    start_s: float
    end_s: float
    transcript: str
    audio_rms: float
    verbal: dict[str, float]


@dataclass
class VideoBrief:
    duration_s: float | None
    fps: float | None
    frames: float | None


def probe_video(path: Path | str) -> VideoBrief:
    path = Path(path).resolve()
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return VideoBrief(None, None, None)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fc = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        duration = fc / fps if fps > 0 else None
        return VideoBrief(
            duration_s=duration,
            fps=fps if fps > 0 else None,
            frames=fc if fc > 0 else None,
        )
    except Exception:
        return VideoBrief(None, None, None)


def chunked_rms(snippet: np.ndarray) -> float:
    if snippet.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(snippet.astype(np.float64)))))


def report_video_without_audio_track(
    video_path: Path,
    brief: VideoBrief,
    *,
    whisper_model: str,
    verbal_ckpt_resolved: Path | None,
    enable_visual_aggression: bool = True,
    visual_model_id: str | None = None,
    visual_max_frames: int = 48,
    visual_torch_device: torch.device | str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Видео без звука: метаданные + пустые сегменты (вербальный риск без ASR невозможен)."""

    zeros = dict.fromkeys(LABEL_NAMES, 0.0)
    md_lines: list[str] = []
    md_lines.append("## Отчёт по видео")
    md_lines.append("")
    md_lines.append("**Аудиодорожки нет** — распознавание речи и вербальный анализ по тексту **не выполнялись**. "
        "При необходимости анализа речи добавьте звук или вкладку с микрофоном.")
    md_lines.append("")
    if brief.duration_s is not None and brief.duration_s > 0:
        md_lines.append(
            f"**Метаданные видео:** длительность ≈ **{brief.duration_s:.1f}s**, "
            f"FPS≈ **{(brief.fps or 0.0):.2f}**, кадров={int(brief.frames or 0)}"
        )
    else:
        md_lines.append("**Метаданные видео:** длительность/FPS недоступны (OpenCV).")

    md_lines.append("")
    md_lines.append("### Вербальный слой по транскрипту")
    md_lines.append("_(нет аудиопотока — таблица сегментов ASR недоступна)._")

    if enable_visual_aggression:
        mid = (visual_model_id or "").strip() or DEFAULT_VISUAL_VIOLENCE_MODEL
        visual = analyze_visual_aggression(
            video_path,
            model_id=mid,
            torch_device=visual_torch_device,
            max_frames=int(visual_max_frames),
        )
    else:
        visual = {"enabled": False}

    incident_no_audio = fuse_batch_incident(
        visual if enable_visual_aggression else None,
        zeros,
        verbal_skipped_reason="Аудиодорожка отсутствует — классы neutral_conflict и др. по речи не оцениваются.",
    )
    md_lines.append("")
    md_lines.extend(
        incident_batch_markdown_lines(incident_no_audio, verbal_available=False),
    )
    md_lines.append("## Визуальный прокси агрессии (выборка кадров)")
    if enable_visual_aggression:
        md_lines.extend(visual_results_markdown_section(visual))
    else:
        md_lines.append("")
        md_lines.append("*(Выключено в параметрах запуска.)*")

    payload: dict[str, Any] = {
        "video": str(video_path),
        "audio_present": False,
        "segments": [],
        "max_scores": zeros,
        "alert_segments": [],
        "whisper_model": whisper_model,
        "verbal_checkpoint": str(verbal_ckpt_resolved) if verbal_ckpt_resolved else None,
        "video_meta": asdict(brief),
        "skipped_asr_reason": "no_audio_stream",
        "visual": visual,
        "incident": incident_no_audio,
    }
    return "\n".join(md_lines), payload


def analyze_video_upload(
    video_path: Path | str | None,
    *,
    whisper_model: str = "openai/whisper-tiny",
    verbal_ckpt: Path | str | None = None,
    chunk_s: float = 12.0,
    stride_s: float = 4.0,
    asr_torch_device: torch.device | str | None = None,
    enable_visual_aggression: bool = True,
    visual_model_id: str | None = None,
    visual_max_frames: int = 48,
) -> tuple[str, dict[str, Any]]:
    if video_path is None:
        raise ValueError("Файл видео не передан.")

    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    brief = probe_video(video_path)

    if not ffmpeg_available():
        raise FileNotFoundError(
            "Нужен ffmpeg (PATH или pip `imageio-ffmpeg`) — чтобы определить дорожки и при наличии звука извлечь WAV. "
            "Windows: `pip install imageio-ffmpeg` или `winget install Gyan.FFmpeg`."
        )

    if not probe_has_audio_stream(video_path):
        if asr_torch_device is not None:
            vd = torch.device(asr_torch_device)
            if vd.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Выбрана CUDA, но torch.cuda.is_available() == False.")
        else:
            vd = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        root_dir = Path(__file__).resolve().parents[1]
        ck_try = Path(verbal_ckpt or root_dir / "checkpoints/verbal-latest").resolve()
        verbal_opt = ck_try if ck_try.exists() else None
        return report_video_without_audio_track(
            video_path,
            brief,
            whisper_model=whisper_model,
            verbal_ckpt_resolved=verbal_opt,
            enable_visual_aggression=enable_visual_aggression,
            visual_model_id=visual_model_id,
            visual_max_frames=int(visual_max_frames),
            visual_torch_device=vd,
        )

    root_dir = Path(__file__).resolve().parents[1]
    verbal_ckpt_resolved = Path(verbal_ckpt or root_dir / "checkpoints/verbal-latest").resolve()
    if not verbal_ckpt_resolved.exists():
        raise FileNotFoundError(
            f"Чекпоинт текстовой головы не найден: {verbal_ckpt_resolved}. "
            "Обучите модель: python training/train_verbal_classifier.py ..."
        )

    wav_path: Path | None = None
    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = Path(tmp_wav.name)
        extract_wav_mono_16k(video_path, wav_path)

        data, sr = sf.read(str(wav_path))
        if getattr(data, "ndim", 1) == 2:
            full_audio = np.mean(data, axis=1).astype(np.float32)
        else:
            full_audio = data.astype(np.float32)

        if sr != 16000:
            raise ValueError(f"Ожидалось 16000 Hz после ffmpeg, получено {sr}")

        from transformers import pipeline

        if asr_torch_device is not None:
            verbal_device = torch.device(asr_torch_device)
            if verbal_device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Выбрана CUDA, но torch.cuda.is_available() == False.")
        else:
            verbal_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        verbal_model, tokenizer, verbal_device = load_verbal_model(
            verbal_ckpt_resolved, device=verbal_device
        )

        if verbal_device.type == "cuda":
            cuda_idx = verbal_device.index
            pipe_idx = 0 if cuda_idx is None else int(cuda_idx)
        else:
            pipe_idx = -1

        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            device=pipe_idx,
        )

        segs_raw, sr_ck = transcribe_chunks(
            wav_path,
            asr_pipe,
            chunk_s=float(chunk_s),
            stride_s=float(stride_s),
        )
        assert sr_ck == sr

        transcripts = [(s.text or "").strip() for s in segs_raw]
        _probs, verbal_dicts = verbal_scores_batch(verbal_model, tokenizer, verbal_device, transcripts)

        rows: list[SegmentInsight] = []
        alert_segments: list[int] = []

        thresholds = {
            "explicit_threat": 0.45,
            "coercion_harassment": 0.45,
            "insult_humiliation": 0.55,
        }

        for idx, seg in enumerate(segs_raw):
            i0 = max(0, int(seg.start_s * sr))
            i1 = min(len(full_audio), int(seg.end_s * sr))
            clip = np.asarray(full_audio[i0:i1], dtype=np.float32)
            rms = chunked_rms(clip)
            vd = dict(verbal_dicts[idx])
            if any(vd.get(k, 0.0) >= thr for k, thr in thresholds.items()):
                alert_segments.append(idx)
            rows.append(
                SegmentInsight(
                    start_s=seg.start_s,
                    end_s=seg.end_s,
                    transcript=seg.text,
                    audio_rms=rms,
                    verbal=vd,
                )
            )

        max_scores = {k: max((r.verbal[k] for r in rows), default=0.0) for k in LABEL_NAMES}

        if enable_visual_aggression:
            vmid = (visual_model_id or "").strip() or DEFAULT_VISUAL_VIOLENCE_MODEL
            visual_blob = analyze_visual_aggression(
                video_path,
                model_id=vmid,
                torch_device=verbal_device,
                max_frames=int(visual_max_frames),
            )
        else:
            visual_blob = {"enabled": False}

        incident = fuse_batch_incident(
            visual_blob if enable_visual_aggression else None,
            max_scores,
        )

        md_lines: list[str] = []
        md_lines.append("## Отчёт по видео (ASR + вербальный риск)")
        md_lines.append("")
        md_lines.append(
            "**Вербальный слой MVP:** текст после ASR и обученная многослойная голова. "
            "**Визуальный слой** ниже по кадрам (прокси физической агрессии — не школьный буллинг целиком)."
        )
        md_lines.append("")
        if brief.duration_s is not None and brief.duration_s > 0:
            md_lines.append(
                f"Метаданные: длительность ≈ **{brief.duration_s:.1f}s**, "
                f"FPS≈ **{(brief.fps or 0.0):.2f}**, кадров={int(brief.frames or 0)}"
            )
            md_lines.append("")
        md_lines.append("Максимумы по классам:")
        for k in LABEL_NAMES:
            md_lines.append(f"- **{k}**: {max_scores[k]:.4f}")
        md_lines.append("")
        md_lines.append(f"Окон с порогами {thresholds}: **{len(alert_segments)}**")
        md_lines.append("")
        md_lines.extend(
            incident_batch_markdown_lines(incident, verbal_available=True),
        )
        md_lines.append("| t0 | t1 | RMS | ASR текст | explicit | insult | coercion | neutral |")
        md_lines.append("|--:|--:|-----:|-----------|-------:|-------:|-------:|-------:|")

        for r in rows:
            excerpt = r.transcript.replace("|", "/")
            if len(excerpt) > 160:
                excerpt = excerpt[:160] + "…"
            v = r.verbal
            md_lines.append(
                f"| {r.start_s:.1f} | {r.end_s:.1f} | {r.audio_rms:.5f} | {excerpt} | "
                f"{v['explicit_threat']:.3f} | {v['insult_humiliation']:.3f} | "
                f"{v['coercion_harassment']:.3f} | {v['neutral_conflict']:.3f} |"
            )

        md_lines.append("")
        md_lines.append("## Визуальный прокси агрессии (выборка кадров)")
        if enable_visual_aggression:
            md_lines.extend(visual_results_markdown_section(visual_blob))
        else:
            md_lines.append("")
            md_lines.append("*(Выключено в параметрах запуска.)*")

        payload: dict[str, Any] = {
            "video": str(video_path),
            "audio_present": True,
            "segments": [asdict(r) for r in rows],
            "max_scores": max_scores,
            "alert_segments": alert_segments,
            "whisper_model": whisper_model,
            "verbal_checkpoint": str(verbal_ckpt_resolved),
            "video_meta": asdict(brief),
            "visual": visual_blob,
            "incident": incident,
        }
        return "\n".join(md_lines), payload
    finally:
        if wav_path is not None and wav_path.exists():
            wav_path.unlink(missing_ok=True)


def payload_to_download_json(payload: dict[str, Any]) -> bytes:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
