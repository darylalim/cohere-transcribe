"""Unit tests for the pure functions -- no model, no audio, no network.

These do not replace `verify_transcription.py`, and they cannot: the bug that
motivated it returns a confident, non-empty, correctly-typed string, so only a
comparison against text we authored ourselves distinguishes a working checkpoint
from a broken one. This file covers what that test *cannot* reach:

1. Its own oracle. `word_error_rate` and `non_ascii_ratio` are the entire pass
   or fail decision over there, and a bug in either fails toward a false pass --
   a WER that drifts low reads as a *better* transcript. Nothing else checks
   them.
2. The branches that exist for inputs `say` cannot produce. `_cap_segment_length`
   fires only when Silero detects no speech, and `_cues` only on a malformed
   segment; every fixture over there is wall-to-wall synthesized speech, so
   neither branch has ever executed in a test.
3. Subtitle structure. `verify_transcription.py` asserts `"-->" in srt` and that
   the VTT starts with WEBVTT, which passes just as happily on indices starting
   at 0 or the two separators swapped.

Deliberately absent: anything that mocks `model.generate`. A stub returning a
plausible string asserts that the dataclass holds a string, which was never in
doubt, and is the exact shape of the false confidence `verify_transcription.py`
exists to defeat. Real decoding is also out -- `check_decoding` already covers it
against nine actual containers, which no fixture set can match.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.audio import (
    _cues,
    _timestamp,
    format_duration,
    to_srt,
    to_vtt,
)
from utils.models import Transcript, _cap_segment_length

# Importing verify_transcription runs its module-level warnings.filterwarnings
# ("ignore"), which then applies to the whole pytest session. Harmless here --
# nothing below asserts on a warning -- but it is why an unrelated DeprecationWarning
# will not surface in this run.
from verify_transcription import (
    MAX_NON_ASCII,
    MAX_WER,
    non_ascii_ratio,
    normalize,
    word_error_rate,
)

TEN_WORDS = "one two three four five six seven eight nine ten"


# --- The oracle -----------------------------------------------------------
# word_error_rate and non_ascii_ratio decide whether verify_transcription.py
# passes. Every case below is one that would let a broken checkpoint through.


def test_wer_identical_is_zero():
    assert word_error_rate(TEN_WORDS, TEN_WORDS) == 0.0


def test_wer_ignores_case_and_punctuation():
    """The reference is written with punctuation and the model may not emit it.

    If normalize under-stripped, a correct transcript would score a nonzero WER
    and the threshold would have to be loosened to compensate -- which is how a
    real regression later slips under it.
    """
    assert word_error_rate("Hello, world!", "hello world") == 0.0


@pytest.mark.parametrize(
    ("hypothesis", "expected"),
    [
        # One substitution, one deletion, one insertion: each is a single edit
        # against a ten word reference, so each must score exactly 0.1. A
        # transposed cost table or an off-by-one in the DP init shows up here.
        ("one two three four five six seven eight nine zero", 0.1),
        ("one two three four five six seven eight nine", 0.1),
        ("one two three four five six seven eight nine ten eleven", 0.1),
    ],
)
def test_wer_counts_one_edit_as_one(hypothesis, expected):
    assert word_error_rate(TEN_WORDS, hypothesis) == pytest.approx(expected)


def test_wer_all_wrong_is_one():
    """The floor case for a broken decoder that happens to stay in ASCII.

    non_ascii_ratio catches multilingual nonsense; this is what catches nonsense
    that stayed in the Latin alphabet.
    """
    assert word_error_rate("one two three", "alpha beta gamma") == 1.0


def test_wer_exceeds_one_when_the_hypothesis_runs_long():
    """Not normalised to 1.0, on purpose -- runaway generation should read as
    worse than merely wrong, and it is compared against a 0.15 threshold anyway."""
    assert word_error_rate("alpha beta", "one two three four five six") == 3.0


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("", "", 0.0),
        ("", "spurious text", 1.0),
        # An empty transcript is rejected before WER is consulted, but a division
        # by len(ref) here would raise rather than score.
        (TEN_WORDS, "", 1.0),
    ],
)
def test_wer_handles_empty_sides(reference, hypothesis, expected):
    assert word_error_rate(reference, hypothesis) == expected


def test_normalize_splits_on_stripped_punctuation():
    assert normalize("Hello, world -- again!") == ["hello", "world", "again"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0.0),
        ("   \n\t", 0.0),  # no letters at all, not a division by zero
        ("hello world", 0.0),
        ("日本語", 1.0),
        ("ab日", 1 / 3),
    ],
)
def test_non_ascii_ratio(text, expected):
    assert non_ascii_ratio(text) == pytest.approx(expected)


def test_non_ascii_ratio_excludes_whitespace_from_the_denominator():
    """Spaces in the denominator would dilute the ratio under the 5% threshold,
    which is the one number standing between a random decoder and a green run."""
    assert non_ascii_ratio("a 日") == pytest.approx(0.5)


def test_thresholds_separate_a_good_transcript_from_a_random_decoder():
    """Ties the oracle to the decision it gates.

    Imports MAX_WER and MAX_NON_ASCII rather than repeating 0.15 and 0.05, so
    loosening a threshold in verify_transcription.py is visible here. Re-typing
    them would have made this test blind to the one drift it exists to catch.
    """
    good = "the quick brown fox jumps over the lazy dog"
    near = "the quick brown fox jumped over the lazy dog"
    assert word_error_rate(good, near) <= MAX_WER
    assert non_ascii_ratio(good) <= MAX_NON_ASCII

    # The observed signature of a decoder loaded under strict=False.
    nonsense = "私はここにいます 그리고 여기에"
    assert word_error_rate(good, nonsense) > MAX_WER
    assert non_ascii_ratio(nonsense) > MAX_NON_ASCII


def test_the_thresholds_are_not_loosened_into_uselessness():
    """The constants are imported above, so this is the check that they still
    mean something: a threshold raised past these lands a random decoder in the
    passing range, and every other test in this file would stay green."""
    assert MAX_WER <= 0.25
    assert MAX_NON_ASCII <= 0.10


# --- Segment capping ------------------------------------------------------
# The no-speech path. Silero returns the whole waveform as one chunk, generate
# takes the VAD branch instead of _prepare_segments, and the 35 second window
# never applies -- so an hour of room tone reaches the encoder as one array.
# `say` always produces speech, so verify_transcription.py cannot reach this.


def test_cap_splits_an_oversized_segment_and_renumbers():
    segments, meta = _cap_segment_length(
        [np.zeros(40_000, dtype=np.float32)],
        [{"start": 0.0, "end": 2.5, "chunk_idx": 0}],
        max_chunk_s=1.0,
        sample_rate=16_000,
    )

    assert [len(s) for s in segments] == [16_000, 16_000, 8_000]
    # chunk_idx must be renumbered against the new list, not carried over from
    # the segment that was split -- three chunks all claiming index 0 would.
    assert [m["chunk_idx"] for m in meta] == [0, 1, 2]
    # Timings are recomputed from the offset, so the tail is not left claiming
    # the whole original span.
    assert [(m["start"], m["end"]) for m in meta] == [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 2.5),
    ]


def test_cap_offsets_from_the_original_start():
    """A segment that did not begin at zero must not have its pieces rebased
    there, or every cue after the first split lands in the wrong place."""
    _, meta = _cap_segment_length(
        [np.zeros(32_000, dtype=np.float32)],
        [{"start": 10.0, "end": 12.0}],
        max_chunk_s=1.0,
        sample_rate=16_000,
    )
    assert [(m["start"], m["end"]) for m in meta] == [(10.0, 11.0), (11.0, 12.0)]


@pytest.mark.parametrize("info", [{"start": None}, {}])
def test_cap_tolerates_a_startless_segment(info):
    """`info.get("start", 0.0) or 0.0` guards twice and both halves are load
    bearing: the default covers a meta dict with no `start` key at all, the `or`
    covers a key present but None. Either one reaching float() unguarded raises
    inside a working transcription. Parametrized because covering only the None
    case let the default be deleted with the whole suite still green.
    """
    _, meta = _cap_segment_length(
        [np.zeros(32_000, dtype=np.float32)],
        [info],
        max_chunk_s=1.0,
        sample_rate=16_000,
    )
    assert [m["start"] for m in meta] == [0.0, 1.0]


def test_cap_leaves_short_segments_untouched():
    """Returns the originals rather than copies: this runs on every VAD call,
    and the common case must not rebuild a list of multi-megabyte arrays."""
    original = np.zeros(100, dtype=np.float32)
    meta_in = [{"start": 0.0}]

    segments, meta = _cap_segment_length(
        [original], meta_in, max_chunk_s=1.0, sample_rate=16_000
    )

    assert segments[0] is original
    assert meta is meta_in


def test_cap_is_a_no_op_when_the_limit_is_not_positive():
    """A limit of 0 makes the step of `range(0, len(segment), limit)` zero, which
    raises ValueError. The guard returns the segments untouched instead."""
    original = np.zeros(40_000, dtype=np.float32)
    segments, _ = _cap_segment_length(
        [original], [{"start": 0.0}], max_chunk_s=0.0, sample_rate=16_000
    )
    assert segments[0] is original


def test_cap_splits_only_what_is_oversized():
    segments, meta = _cap_segment_length(
        [np.zeros(8_000, dtype=np.float32), np.zeros(32_000, dtype=np.float32)],
        [{"start": 0.0, "end": 0.5}, {"start": 0.5, "end": 2.5}],
        max_chunk_s=1.0,
        sample_rate=16_000,
    )
    assert [len(s) for s in segments] == [8_000, 16_000, 16_000]
    assert [m["chunk_idx"] for m in meta] == [0, 1, 2]


# --- Cue filtering --------------------------------------------------------


def test_cues_keeps_a_segment_starting_at_zero():
    """The regression a truthiness check would introduce. `start` is 0.0 for the
    first cue of every transcript, and `if seg.get("start")` would drop it."""
    assert _cues([{"start": 0.0, "end": 1.0, "text": "x"}]) == [
        {"start": 0.0, "end": 1.0, "text": "x"}
    ]


@pytest.mark.parametrize(
    "segment",
    [
        {"end": 1.0, "text": "x"},  # no start -- the KeyError _cues exists for
        {"start": 0.0, "text": "x"},  # no end
        {"start": 0.0, "end": 1.0},  # no text
        {"start": 0.0, "end": 1.0, "text": "   "},  # whitespace only
        {"start": 0.0, "end": 1.0, "text": None},
        {"start": None, "end": 1.0, "text": "x"},
    ],
)
def test_cues_drops_anything_the_writers_would_index_into(segment):
    assert _cues([segment]) == []


def test_writers_survive_a_malformed_segment():
    """The point of the filter: a bad segment must not take an already-successful
    transcript off screen while it renders."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "kept"},
        {"end": 2.0, "text": "dropped"},
    ]
    # Both directions for both writers. Asserting only that "dropped" is absent
    # would pass on a writer that emitted no cues at all.
    assert "kept" in to_srt(segments)
    assert "dropped" not in to_srt(segments)
    assert "kept" in to_vtt(segments)
    assert "dropped" not in to_vtt(segments)


# --- Subtitle output ------------------------------------------------------


def test_srt_is_byte_exact():
    """Indices from 1, comma before the milliseconds, and a blank line between
    blocks. Players reject the file over any of the three, and the `"-->" in srt`
    check in verify_transcription.py passes on all of them."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.0, "end": 2.0, "text": " World "},
    ]
    assert to_srt(segments) == (
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
        "\n"
        "2\n00:00:01,000 --> 00:00:02,000\nWorld\n"
    )


def test_vtt_is_byte_exact():
    """WEBVTT header, a blank line after it, a period before the milliseconds,
    and no cue indices."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.0, "end": 2.0, "text": "World"},
    ]
    assert to_vtt(segments) == (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.000\nHello\n"
        "\n"
        "00:00:01.000 --> 00:00:02.000\nWorld\n"
    )


def test_the_two_formats_do_not_share_a_separator():
    """Swapping them is a one character edit that leaves both files parsing as
    plausible text and neither playing."""
    segment = [{"start": 1.5, "end": 2.0, "text": "x"}]
    assert "00:00:01,500" in to_srt(segment)
    assert "00:00:01.500" in to_vtt(segment)


@pytest.mark.parametrize(
    "segments",
    [
        [],
        # The case that matters: segments exist, so streamlit_app.py calls both
        # writers, but nothing survives _cues. A no-speech transcript looks like
        # this, and the model transcribes silence readily enough to produce one.
        [{"start": 0.0, "end": 1.0, "text": "  "}],
    ],
)
def test_both_writers_are_falsy_with_no_cues(segments):
    """streamlit_app.py disables each download button on a falsy string. to_vtt
    used to return a truthy "WEBVTT\\n" here, so the VTT button stayed live and
    offered a header with no cues on exactly the transcripts whose SRT button
    had already greyed out."""
    assert to_srt(segments) == ""
    assert to_vtt(segments) == ""


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (61.0, "00:01:01,000"),
        (3661.25, "01:01:01,250"),
        (0.0014, "00:00:00,001"),  # sub-millisecond input still renders
        (36000.0, "10:00:00,000"),  # two digit hours are not truncated
    ],
)
def test_timestamp(seconds, expected):
    assert _timestamp(seconds, ",") == expected


# --- Duration formatting --------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0:00"),
        (5, "0:05"),
        (59, "0:59"),
        (60, "1:00"),
        (90, "1:30"),
        # The hour boundary in both directions. 3599 must stay in m:ss as 59:59
        # rather than rolling to 0:59:59, and 3600 must grow the field.
        (3599, "59:59"),
        (3600, "1:00:00"),
        (3661, "1:01:01"),
        (7325, "2:02:05"),
        (59.4, "0:59"),
        (59.6, "1:00"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


# --- Transcript -----------------------------------------------------------


def _transcript(
    source_name: str = "meeting.wav",
    duration_s: float = 60.0,
    elapsed_s: float = 6.0,
) -> Transcript:
    """Named keywords rather than **overrides merged into a dict: ty cannot narrow
    the merged mapping back to the dataclass fields and reports every argument."""
    return Transcript(
        source_key="key",
        source_name=source_name,
        text="hello",
        language="en",
        duration_s=duration_s,
        elapsed_s=elapsed_s,
    )


def test_speedup_is_audio_over_wall_clock():
    assert _transcript().speedup == 10.0


def test_speedup_does_not_divide_by_zero():
    """The metric renders before anything guarantees elapsed_s is positive, and
    a ZeroDivisionError here would take down a finished transcript."""
    assert _transcript(elapsed_s=0.0).speedup == 0.0


@pytest.mark.parametrize(
    ("source_name", "expected"),
    [
        ("meeting.wav", "meeting"),
        ("a.b.c.wav", "a.b.c"),  # rsplit, so only the extension goes
        ("recording", "recording"),  # st.audio_input supplies no extension
        (".wav", "transcript"),  # empty stem would name the download ".txt"
        ("", "transcript"),
    ],
)
def test_stem(source_name, expected):
    assert _transcript(source_name=source_name).stem == expected
