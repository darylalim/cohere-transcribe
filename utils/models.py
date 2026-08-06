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


def _cap_segment_length(
    segments: list, meta: list[dict], max_chunk_s: float, sample_rate: int
) -> tuple[list, list[dict]]:
    """Split any VAD segment longer than the model's chunk window.

    When Silero detects no speech at all, ``_segment_with_vad`` hands back the
    entire waveform as one chunk, and ``generate`` takes the VAD branch instead
    of ``_prepare_segments`` — so the 35 second window never applies. An hour of
    music or room tone with VAD on would reach the encoder as a single
    57-million-sample chunk and either stall or run out of memory, where the
    non-VAD path would have produced about a hundred.
    """
    limit = int(max_chunk_s * sample_rate)
    if limit <= 0 or all(len(segment) <= limit for segment in segments):
        return segments, meta

    capped_segments, capped_meta = [], []
    for segment, info in zip(segments, meta):
        if len(segment) <= limit:
            capped_segments.append(segment)
            capped_meta.append({**info, "chunk_idx": len(capped_meta)})
            continue

        start_s = float(info.get("start", 0.0) or 0.0)
        for offset in range(0, len(segment), limit):
            piece = segment[offset : offset + limit]
            capped_segments.append(piece)
            capped_meta.append(
                {
                    **info,
                    "chunk_idx": len(capped_meta),
                    "start": start_s + offset / sample_rate,
                    "end": start_s + (offset + len(piece)) / sample_rate,
                }
            )
    return capped_segments, capped_meta


def _patch_vad_dtype() -> None:
    """Work around the VAD path in mlx-audio 0.4.7, which assumes numpy.

    ``Model._segment_with_vad`` is written against numpy throughout — it reaches
    ``waveform.astype(np.float32)`` inside the Silero backend and
    ``waveform[a:b].copy()`` when it slices the detected runs out, and its own
    return annotation is ``List[np.ndarray]``. But ``generate`` hands it the
    ``mx.array`` that ``_to_mono`` produced, and MLX arrays have neither method,
    so every ``vad=True`` call raises.

    Coerce once at that boundary rather than patching each call inside, so a
    third numpy assumption in the same function cannot resurface. Safe to delete
    once fixed upstream.
    """
    from mlx_audio.stt.models.cohere_asr.cohere_asr import Model

    # _segment_with_vad is private and pyproject allows any mlx-audio >= 0.4.4,
    # including the release that fixes this. Bail out quietly if it is gone
    # rather than failing every load — this runs even when VAD is switched off.
    original = getattr(Model, "_segment_with_vad", None)
    if original is None or getattr(original, "_coerces_numpy", False):
        return

    @functools.wraps(original)
    def _segment_with_vad(self, waveform, *args, **kwargs):
        segments, meta = original(
            self, np.asarray(waveform, dtype=np.float32), *args, **kwargs
        )
        return _cap_segment_length(
            segments, meta, kwargs.get("max_chunk_s", 30.0), self.sample_rate
        )

    _segment_with_vad._coerces_numpy = True  # ty: ignore[unresolved-attribute]
    Model._segment_with_vad = _segment_with_vad  # ty: ignore[invalid-assignment]


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
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    from mlx_audio.stt import load

    try:
        model = load(repo_id, strict=True)
    except ValueError as exc:
        # The four messages mlx.nn.Module.load_weights(strict=True) can raise.
        # Shape and dtype mismatches are the same class of problem as missing
        # keys — a checkpoint built for a different runtime — and deserve the
        # same explanation rather than a bare ValueError.
        message = str(exc).lower()
        if any(
            marker in message
            for marker in ("not in model", "missing", "expected shape", "expected mx.")
        ):
            raise ModelWeightsError(
                f"`{repo_id}` is not compatible with mlx-audio's Python runtime — its "
                "weights do not match the model definition, and loading it anyway "
                "would produce nonsense text.\n\n"
                "The community `-mlx-` conversions of Cohere Transcribe on the Hub "
                "target the Swift runtime and mlx-speech, not mlx-audio. Use "
                f"`{DEFAULT_REPO}`, which mlx-audio converts on load."
            ) from exc
        raise
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        # Matched by type, not by substring. A "401" anywhere in an unrelated
        # message — a shard name like model-00401-of-00500 — used to be enough
        # to send a disk-full or corrupt-download error to `hf auth login`.
        raise ModelAccessError(
            f"`{repo_id}` is gated, or not visible to this account.\n\n"
            f"1. Accept the terms at https://huggingface.co/{repo_id}\n"
            "2. Run `hf auth login` in this environment\n"
            "3. Reload this page"
        ) from exc

    _patch_vad_dtype()
    return model
