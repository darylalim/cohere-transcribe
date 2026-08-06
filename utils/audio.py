"""Audio decoding and subtitle formatting helpers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence

import numpy as np

SAMPLE_RATE = 16_000

# miniaudio decodes the first three. Everything else reaches ffmpeg, either
# through mlx-audio itself or through the fallback below.
UPLOAD_TYPES = ["wav", "mp3", "flac", "aiff", "m4a", "aac", "ogg", "opus", "webm"]


def _decode_with_ffmpeg(data: bytes) -> np.ndarray:
    """Decode audio mlx-audio could not handle, to mono 16 kHz float32.

    Covers two gaps. mlx-audio picks a decoder from magic bytes and has no
    branch for AIFF (``FORM``/``AIFC``) or raw ADTS AAC (``0xFFF1``), so both
    raise before any decoder runs. It also pipes MP4/M4A to ffmpeg on stdin,
    which fails whenever the ``moov`` index sits at the end of the file — the
    normal layout unless the encoder was told ``+faststart``.

    So write to a real file rather than piping: ffmpeg cannot seek backwards on
    a pipe, and on an unseekable MP4 it decodes nothing and exits 0, which is
    how an entirely valid recording turns into an empty array.

    Asks for raw ``s16le`` rather than WAV for the same reason in reverse: on
    unseekable *output* ffmpeg cannot backfill the RIFF size field and writes a
    ``0xFFFFFFFF`` placeholder. Raw PCM has no header to misparse.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "This audio format needs ffmpeg, which is not installed. "
            "Install it with `brew install ffmpeg`, or convert the file to "
            "WAV, MP3 or FLAC first."
        )

    with tempfile.NamedTemporaryFile(suffix=".audio") as source:
        source.write(data)
        source.flush()
        # fmt: off
        command = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", source.name,
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1",
        ]
        # fmt: on
        # -nostdin, and stdin closed: ffmpeg reads the terminal for interactive
        # keys otherwise, and `streamlit run` is normally started from one. A
        # stray keypress could abort a decode into a silently truncated result.
        result = subprocess.run(
            command, capture_output=True, check=False, stdin=subprocess.DEVNULL
        )

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ValueError(
            f"ffmpeg could not decode this file: {detail[-1] if detail else 'unknown error'}"
        )

    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def decode_to_mono16k(file) -> tuple[np.ndarray, float]:
    """Decode an uploaded or recorded audio file to a mono 16 kHz waveform.

    Streamlit's ``UploadedFile`` is a ``BytesIO`` subclass and mlx-audio's reader
    accepts one directly, so the common path does decode, downmix and resample in
    one call with no temporary file. Anything it cannot handle falls back to
    ffmpeg.

    Returns the waveform and its duration in seconds.
    """
    from mlx_audio.audio_io import read as audio_read

    file.seek(0)
    try:
        # Passed straight through rather than copied into a fresh BytesIO:
        # UploadedFile already is one, and uploads run to 1000 MB here.
        audio, _ = audio_read(
            file, dtype="float32", sample_rate=SAMPLE_RATE, nchannels=1
        )
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    except Exception:  # noqa: BLE001 - anything it raises, ffmpeg gets a turn
        # Deliberately broad. mlx-audio surfaces at least three unrelated types
        # for "cannot decode": ValueError from its magic-byte sniffer,
        # miniaudio.DecodeError from the wav/mp3/flac path, and RuntimeError
        # from its own ffmpeg wrapper. Narrowing this to ValueError meant the
        # fallback never ran for the latter two.
        waveform = np.empty(0, dtype=np.float32)

    # An empty result is a failure, not silence: mlx-audio returns one instead
    # of raising when its piped ffmpeg call cannot seek. Retry before giving up.
    if waveform.size == 0:
        waveform = _decode_with_ffmpeg(file.getvalue())

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
    """Segments worth writing as a subtitle cue.

    Checks every key the writers go on to index, not just ``text`` — a segment
    missing ``start`` would otherwise raise KeyError while rendering a result
    that has already succeeded, taking the transcript off screen with it.
    """
    return [
        seg
        for seg in segments or []
        if (seg.get("text") or "").strip()
        and seg.get("start") is not None
        and seg.get("end") is not None
    ]


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
