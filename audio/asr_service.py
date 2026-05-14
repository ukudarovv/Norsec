"""ASR через faster-whisper с откатом на transformers Whisper (этап 6)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ASRService:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        self.model_size = (model_size or "small").strip().lower()
        self.device = device or "cpu"
        self.compute_type = compute_type or "int8"
        self.language = language
        self._fw_model: Any = None
        self._hf_pipe: Any = None
        self._fw_ok = False
        self._hf_ok = False

    def _load_faster_whisper(self) -> bool:
        if self._fw_ok:
            return self._fw_model is not None
        try:
            from faster_whisper import WhisperModel

            import torch

            dev = self.device if self.device in ("cpu", "cuda", "auto") else "cpu"
            if dev in ("cuda", "auto"):
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            ct = self.compute_type
            if dev == "cpu" and ct not in ("int8", "int8_float32", "default"):
                ct = "int8"
            self._fw_model = WhisperModel(self.model_size, device=dev, compute_type=ct)
            self._fw_ok = True
            logger.info("ASRService: faster-whisper %s", self.model_size)
            return True
        except Exception:
            logger.warning("ASRService: faster-whisper not available (%s)", self.model_size, exc_info=True)
            self._fw_model = None
            self._fw_ok = True
            return False

    def _load_hf_whisper(self) -> bool:
        if self._hf_ok:
            return self._hf_pipe is not None
        try:
            import torch
            from transformers import pipeline

            allowed = ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3")
            mid = f"openai/whisper-{self.model_size}" if self.model_size in allowed else "openai/whisper-small"
            dev = 0 if str(self.device).startswith("cuda") and torch.cuda.is_available() else -1
            self._hf_pipe = pipeline("automatic-speech-recognition", model=mid, device=dev)
            self._hf_ok = True
            logger.info("ASRService: HuggingFace pipeline %s", mid)
            return True
        except Exception:
            logger.warning("ASRService: HF Whisper pipeline failed", exc_info=True)
            self._hf_pipe = None
            self._hf_ok = True
            return False

    def transcribe(self, wav_path: str) -> list[dict[str, Any]]:
        path = Path(wav_path)
        if not path.is_file():
            return []

        if self._load_faster_whisper() and self._fw_model is not None:
            try:
                segs_gen, _info = self._fw_model.transcribe(
                    str(path),
                    language=self.language,
                    vad_filter=False,
                )
                out: list[dict[str, Any]] = []
                for s in list(segs_gen):
                    conf = getattr(s, "avg_logprob", None)
                    out.append(
                        {
                            "start_sec": round(float(s.start), 3),
                            "end_sec": round(float(s.end), 3),
                            "text": (s.text or "").strip(),
                            "confidence": round(float(conf), 4) if conf is not None else None,
                        }
                    )
                if out:
                    return out
            except Exception:
                logger.exception("faster_whisper.transcribe failed")

        if self._load_hf_whisper() and self._hf_pipe is not None:
            try:
                r = self._hf_pipe(str(path), return_timestamps="chunk", chunk_length_s=30)
                chunks = r.get("chunks") or []
                rows: list[dict[str, Any]] = []
                for c in chunks:
                    ts = c.get("timestamp")
                    t0, t1 = 0.0, -1.0
                    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                        t0, t1 = float(ts[0]), float(ts[1])
                    elif isinstance(ts, dict):
                        t0 = float(ts.get("start", 0.0))
                        t1 = float(ts.get("end", -1.0))
                    rows.append(
                        {
                            "start_sec": round(t0, 3),
                            "end_sec": round(t1, 3) if t1 >= 0 else -1.0,
                            "text": str(c.get("text", "")).strip(),
                            "confidence": None,
                        }
                    )
                if rows:
                    return rows
                txt = (r.get("text") or "").strip()
                if txt:
                    return [{"start_sec": 0.0, "end_sec": -1.0, "text": txt, "confidence": None}]
            except Exception:
                logger.exception("HF whisper transcribe failed")

        return [{"start_sec": 0.0, "end_sec": -1.0, "text": "", "confidence": None}]
