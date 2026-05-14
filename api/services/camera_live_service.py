"""Проверка RTSP и вспомогательные утилиты live (этап 10)."""

from __future__ import annotations

import logging
import time

import cv2

logger = logging.getLogger(__name__)


def test_rtsp_connection(rtsp_url: str, *, timeout_sec: float = 4.0) -> tuple[bool, str]:
    """
    Пытается открыть поток и прочитать хотя бы один кадр.
    Возвращает (ok, reason_code).
    """
    url = (rtsp_url or "").strip()
    if not url:
        return False, "empty_url"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        logger.warning("test_rtsp_connection: cannot open %s", url[:120])
        return False, "cannot_open"
    t0 = time.time()
    seen = False
    try:
        while time.time() - t0 < float(timeout_sec):
            ok, frame = cap.read()
            if ok and frame is not None:
                seen = True
                break
            time.sleep(0.03)
    finally:
        cap.release()
    if not seen:
        logger.warning("test_rtsp_connection: no frames %s", url[:120])
        return False, "no_frames"
    return True, "ok"
