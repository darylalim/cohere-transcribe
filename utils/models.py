"""Checkpoint loading for Cohere Transcribe under mlx-audio."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
import streamlit as st

# The only checkpoint mlx-audio's Python runtime can load. mlx-audio remaps the
# upstream Hugging Face layout at load time; the community "-mlx-" conversions on
# the Hub target the Swift runtime and mlx-speech instead, and use a different
# parameter naming scheme (`decoder.core.*`, `bridge_proj.*` rather than
# `transf_decoder.decoder.*`, `encoder_decoder_proj.*`). See load_asr below.
DEFAULT_REPO = "CohereLabs/cohere-transcribe-03-2026"

# The 14 languages the model was trained on. It has no language detection, so
# one of these has to be picked explicitly for every transcription.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "ar": "Arabic",
    "zh": "Chinese (Mandarin)",
    "nl": "Dutch",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pl": "Polish",
    "pt": "Portuguese",
    "es": "Spanish",
    "vi": "Vietnamese",
}


@dataclass
class Transcript:
    """One finished transcription, held in session state across reruns."""

    source_key: str
    source_name: str
    text: str
    language: str
    duration_s: float
    elapsed_s: float
    segments: list[dict] = field(default_factory=list)

    @property
    def speedup(self) -> float:
        """Audio seconds transcribed per second of wall clock (RTFx)."""
        return self.duration_s / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def stem(self) -> str:
        return self.source_name.rsplit(".", 1)[0] or "transcript"


class ModelAccessError(RuntimeError):
    """The checkpoint could not be downloaded, usually because it is gated."""


class ModelWeightsError(RuntimeError):
    """The checkpoint loaded but its weights do not match the mlx-audio model."""


def _patch_vad_dtype() -> None:
    """Work around a VAD crash in mlx-audio 0.4.7.

    ``cohere_asr._segment_with_vad`` hands an ``mx.array`` to a Silero backend
    whose ``detect_speech`` calls ``waveform.astype(np.float32)``. MLX arrays only
    accept MLX dtypes, so every ``vad=True`` call raises TypeError. Coerce to
    numpy on the way in. Safe to delete once fixed upstream.
    """
    from mlx_audio.stt.models.cohere_asr import vad

    if getattr(vad.SileroMlxBackend, "_accepts_mx_array", False):
        return

    original = vad.SileroMlxBackend.detect_speech

    @functools.wraps(original)
    def detect_speech(self, waveform):
        return original(self, np.asarray(waveform, dtype=np.float32))

    vad.SileroMlxBackend.detect_speech = detect_speech  # ty: ignore[invalid-assignment]
    vad.SileroMlxBackend._accepts_mx_array = True  # ty: ignore[unresolved-attribute]


# max_entries=1 keeps a single multi-gigabyte model resident, so pointing the app
# at a different repo evicts the previous one instead of holding both in memory.
@st.cache_resource(max_entries=1, show_spinner=False)
def load_asr(repo_id: str):
    """Download (once) and load a Cohere Transcribe checkpoint for MLX.

    Loads with ``strict=True`` deliberately. mlx-audio defaults to ``strict=False``,
    which silently leaves any unmatched module randomly initialised — an
    incompatible checkpoint then transcribes fluent-looking multilingual nonsense
    instead of failing. Better to refuse to load.
    """
    from mlx_audio.stt import load

    try:
        model = load(repo_id, strict=True)
    except ValueError as exc:
        if "not in model" in str(exc) or "missing" in str(exc).lower():
            raise ModelWeightsError(
                f"`{repo_id}` is not compatible with mlx-audio's Python runtime — its "
                "weights do not match the model definition, and loading it anyway "
                "would produce nonsense text.\n\n"
                "The community `-mlx-` conversions of Cohere Transcribe on the Hub "
                "target the Swift runtime and mlx-speech, not mlx-audio. Use "
                f"`{DEFAULT_REPO}`, which mlx-audio converts on load."
            ) from exc
        raise
    except Exception as exc:
        message = str(exc).lower()
        if any(t in message for t in ("gated", "401", "restricted", "unauthorized")):
            raise ModelAccessError(
                f"`{repo_id}` is a gated repository.\n\n"
                f"1. Accept the terms at https://huggingface.co/{repo_id}\n"
                "2. Run `hf auth login` in this environment\n"
                "3. Reload this page"
            ) from exc
        raise

    _patch_vad_dtype()
    return model
