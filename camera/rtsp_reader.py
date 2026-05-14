"""Чтение кадров RTSP через OpenCV с переподключением (Phase 1)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RTSPReader:
    def __init__(
        self,
        rtsp_url: str,
        reconnect_delay_sec: float = 5.0,
        max_failures: int = 5,
        *,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.rtsp_url = str(rtsp_url).strip()
        self.reconnect_delay_sec = float(reconnect_delay_sec)
        self.max_failures = int(max_failures)
        self._stop = threading.Event()
        self._on_error = on_error
        self._last_error_lock = threading.Lock()
        self._last_error: str | None = None

    def stop(self) -> None:
        self._stop.set()

    def is_stopped(self) -> bool:
        return self._stop.is_set()

    def last_error(self) -> str | None:
        with self._last_error_lock:
            return self._last_error

    def _set_error(self, msg: str) -> None:
        with self._last_error_lock:
            self._last_error = msg
        if self._on_error:
            try:
                self._on_error(msg)
            except Exception:
                logger.exception("on_error callback failed")

    def read_frames(self) -> Iterator[np.ndarray]:
        consecutive_open_failures = 0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                consecutive_open_failures += 1
                msg = f"open_failed attempt={consecutive_open_failures}/{self.max_failures}"
                self._set_error(msg)
                logger.warning(
                    "rtsp_open_failed",
                    extra={
                        "camera_rtsp": self.rtsp_url[:120],
                        "attempt": consecutive_open_failures,
                        "max_failures": self.max_failures,
                    },
                )
                if consecutive_open_failures >= self.max_failures:
                    logger.error(
                        "rtsp_reader_stopped_max_failures",
                        extra={"camera_rtsp": self.rtsp_url[:120]},
                    )
                    return
                time.sleep(self.reconnect_delay_sec)
                continue

            consecutive_open_failures = 0
            self._set_error("")
            try:
                while not self._stop.is_set():
                    try:
                        ok, frame = cap.read()
                    except Exception as e:
                        self._set_error(f"read_exception: {e!s}")
                        logger.error(
                            "rtsp_read_failed",
                            extra={"camera_rtsp": self.rtsp_url[:120], "error": str(e)},
                            exc_info=True,
                        )
                        break
                    if not ok or frame is None:
                        self._set_error("read_failed_eof")
                        logger.warning(
                            "rtsp_frame_read_failed",
                            extra={"camera_rtsp": self.rtsp_url[:120]},
                        )
                        break
                    yield frame
            finally:
                try:
                    cap.release()
                except Exception:
                    logger.debug("cap.release failed", exc_info=True)

            if self._stop.is_set():
                return
            time.sleep(self.reconnect_delay_sec)
