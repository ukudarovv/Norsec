"""RTSPReader: устойчивость к невалидному URL (мок OpenCV)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from camera.rtsp_reader import RTSPReader


def test_rtsp_reader_invalid_url_exits_without_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    def fake_capture(*_a, **_k):
        return mock_cap

    monkeypatch.setattr("camera.rtsp_reader.cv2.VideoCapture", fake_capture)

    r = RTSPReader("rtsp://127.0.0.1:9/nope", reconnect_delay_sec=0.01, max_failures=3)
    gen = r.read_frames()
    frames = []
    for _ in range(5):
        try:
            frames.append(next(gen))
        except StopIteration:
            break
    r.stop()
    assert frames == []
