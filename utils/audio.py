"""Audio decoding and subtitle formatting helpers."""

from __future__ import annotations

import io
from collections.abc import Iterable, Sequence

import numpy as np

SAMPLE_RATE = 16_000

# miniaudio handles the first four; mlx-audio shells out to ffmpeg for the rest.
UPLOAD_TYPES = ["wav", "mp3", "flac", "aiff", "m4a", "aac", "ogg", "opus", "webm"]


def decode_to_mono16k(file) -> tuple[np.ndarray, float]:
    """Decode an uploaded or recorded audio file to a mono 16 kHz waveform.

    Streamlit's ``UploadedFile`` is a ``BytesIO`` subclass and mlx-audio's reader
    accepts one directly, so the whole decode / downmix / resample happens in one
    call with no temporary file on disk.

    Returns the waveform and its duration in seconds.
    """
    from mlx_audio.audio_io import read as audio_read

    buffer = io.BytesIO(file.getvalue())
    audio, _ = audio_read(buffer, dtype="float32", sample_rate=SAMPLE_RATE, nchannels=1)

    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
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
