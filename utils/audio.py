"""Audio decoding and subtitle formatting helpers."""

from __future__ import annotations

import io
import shutil
import subprocess
from collections.abc import Iterable, Sequence

import numpy as np

SAMPLE_RATE = 16_000

# miniaudio decodes the first three. Everything else reaches ffmpeg, either
# through mlx-audio itself or through the fallback below.
UPLOAD_TYPES = ["wav", "mp3", "flac", "aiff", "m4a", "aac", "ogg", "opus", "webm"]


def _decode_with_ffmpeg(data: bytes) -> np.ndarray:
    """Decode a container mlx-audio cannot identify, to mono 16 kHz float32.

    mlx-audio picks a decoder from magic bytes and has no branch for AIFF
    (``FORM``/``AIFC``) or raw ADTS AAC (``0xFFF1``), so both raise before any
    decoder runs even though ffmpeg reads them fine.

    Asks ffmpeg for raw ``s16le`` rather than WAV: a pipe is not seekable, so
    ffmpeg cannot backfill the RIFF size field and emits a ``0xFFFFFFFF``
    placeholder instead. Raw PCM has no header to misparse.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "This audio format needs ffmpeg, which is not installed. "
            "Install it with `brew install ffmpeg`, or convert the file to "
            "WAV, MP3 or FLAC first."
        )

    # fmt: off
    command = [
        "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1",
    ]
    # fmt: on
    result = subprocess.run(command, input=data, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ValueError(
            f"ffmpeg could not decode this file: {detail[-1] if detail else 'unknown error'}"
        )

    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def decode_to_mono16k(file) -> tuple[np.ndarray, float]:
    """Decode an uploaded or recorded audio file to a mono 16 kHz waveform.

    Streamlit's ``UploadedFile`` is a ``BytesIO`` subclass and mlx-audio's reader
    accepts one directly, so the whole decode / downmix / resample happens in one
    call with no temporary file on disk. Formats its sniffer does not recognise
    fall back to ffmpeg.

    Returns the waveform and its duration in seconds.
    """
    from mlx_audio.audio_io import read as audio_read

    data = file.getvalue()
    try:
        audio, _ = audio_read(
            io.BytesIO(data), dtype="float32", sample_rate=SAMPLE_RATE, nchannels=1
        )
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    except ValueError:
        waveform = _decode_with_ffmpeg(data)

    if waveform.size == 0:
        raise ValueError("The audio file decoded to zero samples.")

    return waveform, waveform.size / SAMPLE_RATE


def format_duration(seconds: float) -> str:
    """Render a duration as m:ss, or h:mm:ss once it passes an hour."""
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _timestamp(seconds: float, separator: str) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _cues(segments: Iterable[dict]) -> list[dict]:
    return [seg for seg in segments or [] if (seg.get("text") or "").strip()]


def to_srt(segments: Sequence[dict]) -> str:
    """Build an SRT file from the model's chunk segments.

    Cohere Transcribe emits no word- or sentence-level timings, so each cue spans
    a whole decoder chunk (up to 35 seconds). Useful for navigation, too coarse
    for real subtitles.
    """
    blocks = []
    for index, seg in enumerate(_cues(segments), start=1):
        start = _timestamp(seg["start"], ",")
        end = _timestamp(seg["end"], ",")
        blocks.append(f"{index}\n{start} --> {end}\n{seg['text'].strip()}\n")
    return "\n".join(blocks)


def to_vtt(segments: Sequence[dict]) -> str:
    """Build a WebVTT file from the model's chunk segments."""
    blocks = ["WEBVTT\n"]
    for seg in _cues(segments):
        start = _timestamp(seg["start"], ".")
        end = _timestamp(seg["end"], ".")
        blocks.append(f"{start} --> {end}\n{seg['text'].strip()}\n")
    return "\n".join(blocks)
