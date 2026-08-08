"""Smoke test the real transcription path against known ground truth.

Run from the repository root, once `uv run hf auth login` has a valid token:

    uv run verify_transcription.py

Synthesizes speech with macOS `say`, so the reference text is exact and word
error rate is meaningful. A working checkpoint scores near 0%. The failure this
guards against -- a randomly initialised decoder loaded under strict=False --
returns fluent multilingual text and scores ~100% with non-Latin characters.

Unit tests cannot cover this: the broken checkpoint returned a confident,
non-empty, correctly-typed string. Only a comparison against text we authored
ourselves distinguishes it from a working one.

Requires macOS (`say`) and ffmpeg, and downloads ~4 GB on first run.
"""

from __future__ import annotations

import io
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")

from utils.audio import UPLOAD_TYPES, decode_to_mono16k, to_srt, to_vtt
from utils.models import DEFAULT_REPO, load_asr

# Named rather than inlined at the comparisons so tests/test_pure.py can import
# the actual numbers. Re-typing them there would have let a loosened threshold
# here pass a test written specifically to catch a loosened threshold.
MAX_WER = 0.15
MAX_NON_ASCII = 0.05

SHORT_TEXT = (
    "The quick brown fox jumps over the lazy dog while the ambitious "
    "researcher calibrates the spectrometer in the basement laboratory."
)

# Roughly 42 seconds at the default `say` rate, which is what puts it past the
# 35 second chunk window -- the only way to exercise long-form splitting and get
# more than one segment back.
LONG_TEXT = (
    "Marine biologists tracking humpback whales across the northern Pacific "
    "reported a significant change in migration timing this season. "
    "The animals arrived at their feeding grounds nearly three weeks earlier "
    "than the thirty year average recorded by the observatory. "
    "Researchers attribute the shift to warmer surface temperatures and an "
    "unusually early bloom of krill along the continental shelf. "
    "Similar patterns have appeared in the southern hemisphere, where "
    "populations now depart the breeding lagoons before the equinox. "
    "The team plans to deploy additional acoustic buoys next spring to "
    "measure whether the trend continues or reverses. "
    "Funding for the expanded survey comes from a consortium of universities "
    "and two national science agencies."
)


class Failure(Exception):
    """A check did not pass."""


def normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation, split to words."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by reference length."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1] / len(ref)


def non_ascii_ratio(text: str) -> float:
    """Share of non-ASCII characters -- the signature of a random decoder."""
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) > 127) / len(letters)


def synthesize(text: str, path: pathlib.Path) -> None:
    """Render text to an AIFF file with the macOS speech synthesizer."""
    subprocess.run(["say", "-o", str(path), text], check=True, capture_output=True)


class _Upload(io.BytesIO):
    """Mimic Streamlit's UploadedFile, which is what decode_to_mono16k expects."""

    def __init__(self, path: pathlib.Path):
        super().__init__(path.read_bytes())
        self.name = path.name


def check(label: str, reference: str, model, tmp: pathlib.Path, **kwargs) -> dict:
    path = tmp / f"{label}.aiff"
    synthesize(reference, path)

    waveform, duration_s = decode_to_mono16k(_Upload(path))
    started = time.perf_counter()
    output = model.generate(waveform, sample_rate=16_000, language="en", **kwargs)
    elapsed = time.perf_counter() - started

    text = (output.text or "").strip()
    wer = word_error_rate(reference, text)
    foreign = non_ascii_ratio(text)
    segments = output.segments or []

    print(f"\n--- {label} ---")
    print(
        f"audio       {duration_s:.1f}s   elapsed {elapsed:.1f}s   "
        f"RTFx {duration_s / elapsed:.0f}x"
    )
    print(f"WER         {wer:.1%}")
    print(f"non-ascii   {foreign:.1%}")
    print(f"segments    {len(segments)}")
    print(f"reference   {reference[:100]}...")
    print(f"transcript  {text[:100]}...")

    if not text:
        raise Failure(f"{label}: transcript is empty")
    if foreign > MAX_NON_ASCII:
        raise Failure(
            f"{label}: {foreign:.0%} non-ASCII characters -- this is the random "
            "decoder signature, the checkpoint is not loading correctly"
        )
    if wer > MAX_WER:
        raise Failure(
            f"{label}: WER {wer:.1%} exceeds the {MAX_WER:.0%} threshold.\n"
            f"  expected: {reference}\n"
            f"  got:      {text}"
        )

    return {"wer": wer, "segments": segments, "duration_s": duration_s}


def check_decoding(tmp: pathlib.Path) -> list[Failure]:
    """Every format the uploader advertises must decode to real samples.

    Needs no model, so it runs first. Includes an MP4 written *without*
    ``+faststart``: that puts the ``moov`` index at the end of the file, which
    is the normal layout, and it decoded to an empty array until the ffmpeg
    fallback stopped piping. Empty is the failure mode to watch -- it does not
    raise on its own.
    """
    source = tmp / "formats.aiff"
    synthesize(SHORT_TEXT, source)
    _, reference_s = decode_to_mono16k(_Upload(source))
    failures = []

    for ext in UPLOAD_TYPES:
        target = tmp / f"formats.{ext}"
        if ext != "aiff":
            # No -movflags for m4a on purpose: ffmpeg's default already leaves
            # the moov index at the end, which is the layout that broke.
            result = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), str(target)],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                # Never skip silently -- a missing row reads as a passing row.
                failures.append(
                    Failure(f"encode {ext}: ffmpeg could not produce the test file")
                )
                print(f"  {ext:5} SKIP (ffmpeg cannot encode it here)")
                continue
        else:
            target = source

        try:
            _, duration = decode_to_mono16k(_Upload(target))
            # Not just "> 0". The mode next door to an empty decode is a short
            # one: on an unseekable container ffmpeg can emit its 32 KB buffer
            # and stop, and 0.2s of a 7s file would otherwise print as a pass.
            drift = abs(duration - reference_s) / reference_s
            if drift > 0.05:
                raise ValueError(
                    f"decoded {duration:.2f}s against a {reference_s:.2f}s "
                    f"source ({drift:.0%} off)"
                )
            print(f"  {ext:5} {duration:5.1f}s")
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            failures.append(Failure(f"decode {ext}: {type(exc).__name__}: {exc}"))
            print(f"  {ext:5} FAIL {exc}")

    return failures


def run(tmp: pathlib.Path) -> int:
    for binary in ("say", "ffmpeg"):
        if shutil.which(binary) is None:
            print(f"FAIL  `{binary}` is not on PATH; this script needs it to run")
            return 1

    print("checking every advertised upload format decodes")
    failures = check_decoding(tmp)

    print(f"\nloading {DEFAULT_REPO} (first run downloads ~4 GB)")
    started = time.perf_counter()
    model = load_asr(DEFAULT_REPO)
    print(f"loaded in {time.perf_counter() - started:.1f}s")

    try:
        check("short", SHORT_TEXT, model, tmp)
    except Failure as exc:
        failures.append(exc)

    try:
        long_result = check("long-form", LONG_TEXT, model, tmp)

        # Long-form splitting is what SRT/VTT export depends on, so confirm it
        # actually produced multiple chunks and that they format cleanly.
        segments = long_result["segments"]
        if len(segments) < 2:
            failures.append(
                Failure(
                    f"long-form: {long_result['duration_s']:.0f}s of audio produced "
                    f"{len(segments)} segment(s); expected the 35s window to split it"
                )
            )
        else:
            srt, vtt = to_srt(segments), to_vtt(segments)
            print(f"\nSRT {len(srt)} chars, VTT {len(vtt)} chars")
            print(srt[:180])
            if "-->" not in srt or not vtt.startswith("WEBVTT"):
                failures.append(Failure("subtitle export malformed"))
    except Failure as exc:
        failures.append(exc)

    # The shimmed VAD path never runs in the unit tests against real weights.
    try:
        check("vad", SHORT_TEXT, model, tmp, vad=True, vad_merge_gap_s=1.0)
    except Failure as exc:
        failures.append(exc)
    except Exception as exc:  # noqa: BLE001 - the shim regressing looks like this
        failures.append(Failure(f"vad: {type(exc).__name__}: {exc}"))

    print("\n" + "=" * 60)
    if failures:
        for failure in failures:
            print(f"FAIL  {failure}")
        return 1
    print("PASS  transcription verified against real weights")
    return 0


def main() -> int:
    # TemporaryDirectory so the transcoded audio does not accumulate in the
    # system temp dir on every run, including the success path.
    with tempfile.TemporaryDirectory(prefix="cohere-verify-") as tmp:
        return run(pathlib.Path(tmp))


if __name__ == "__main__":
    sys.exit(main())
