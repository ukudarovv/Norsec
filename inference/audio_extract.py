from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

_CACHED_FFMPEG: str | None = None


def resolve_ffmpeg_executable() -> str | None:
    """
    Сначала ffmpeg из PATH; при отсутствии — бинарь из pip-пакета `imageio-ffmpeg`
    (удобно на Windows без ручной установки).
    """
    global _CACHED_FFMPEG  # noqa: PLW0603
    if _CACHED_FFMPEG and Path(_CACHED_FFMPEG).exists():
        return _CACHED_FFMPEG

    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        resolved = str(Path(sys_ffmpeg).resolve())
        _CACHED_FFMPEG = resolved
        return resolved

    try:
        import imageio_ffmpeg  # noqa: WPS433 — опционально

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        if bundled.exists():
            _CACHED_FFMPEG = str(bundled)
            return _CACHED_FFMPEG
    except Exception:
        pass

    _CACHED_FFMPEG = None
    return None


def ffmpeg_available() -> bool:
    return resolve_ffmpeg_executable() is not None


def ensure_gradio_ffmpy_finds_ffmpeg() -> Path | None:
    """
    Gradio/ffmpy запускает исполняемый файл по имени `ffmpeg`. У imageio-ffmpeg
    файл называется иначе (напр. ffmpeg-win-x86_64-v7.1.exe) — нужна копия с именем
    ffmpeg / ffmpeg.exe в каталоге, который добавляется **в начало PATH**.
    """
    resolved = resolve_ffmpeg_executable()
    if not resolved:
        return None

    sep = os.pathsep

    def _prepend_dir(dir_path: Path) -> None:
        d = str(dir_path.resolve())
        cur = os.environ.get("PATH", "")
        if d not in cur.split(sep):
            os.environ["PATH"] = d + sep + cur

    src = Path(resolved).resolve()
    win = os.name == "nt"
    shim_name = "ffmpeg.exe" if win else "ffmpeg"

    if src.name.lower() == shim_name.lower():
        _prepend_dir(src.parent)
        return src

    shim_root = Path(tempfile.gettempdir()) / "search_system_gradio_ffmpeg_shim"
    shim_root.mkdir(parents=True, exist_ok=True)
    dst = shim_root / shim_name
    try:
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dst)
    except OSError:
        _prepend_dir(src.parent)
        return src
    if not win:
        m = dst.stat().st_mode
        dst.chmod(m | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    _prepend_dir(shim_root)
    return dst.resolve()


def probe_has_audio_stream(media_path: Path | str) -> bool:
    """
    True, если FFmpeg видит хотя бы одну дорожку Audio в контейнере.

    ffmpeg -i всегда завершается с ненулевым кодом без выходного файла — это ожидаемо,
    нужен только текст stderr/stdout со списком потоков.
    """
    ffmpeg_bin = resolve_ffmpeg_executable()
    if not ffmpeg_bin:
        return False

    media_path = Path(media_path).resolve()
    res = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", str(media_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    blob = f"{res.stderr or ''}\n{res.stdout or ''}"
    return bool(
        re.search(r"Stream\s+#\d+:\d+(?:\([^)]*\))?:\s*Audio:", blob, re.MULTILINE)
    )


def extract_wav_mono_16k(video_path: Path, out_wav: Path | None = None) -> Path:
    """
    Extract mono 16 kHz WAV (Whisper-friendly) using ffmpeg (PATH или imageio-ffmpeg).

    Raises:
        FileNotFoundError: if ffmpeg is not on PATH
        subprocess.CalledProcessError: ffmpeg failed
    """
    ffmpeg_bin = resolve_ffmpeg_executable()
    if not ffmpeg_bin:
        raise FileNotFoundError(
            "ffmpeg не найден: ни в PATH, ни через пакет imageio-ffmpeg. "
            "Установите: pip install imageio-ffmpeg или системный ffmpeg."
        )

    video_path = Path(video_path).resolve()
    if out_wav is None:
        fh = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        out_wav = Path(fh.name)
        fh.close()

    out_wav = Path(out_wav).resolve()

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(out_wav),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
        raise RuntimeError(
            "ffmpeg вернул ошибку при извлечении WAV. Проверьте дорожку звука в видео.\n"
            f"Подробнее: {stderr[:2000]}"
        ) from e
    return out_wav
