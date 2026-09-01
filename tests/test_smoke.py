"""Checks that need no checkpoint, and that neither other layer makes.

`tests/test_pure.py` covers the pure functions and the integration test's own
oracle. `verify_transcription.py` covers real decoding against real weights, at
the cost of macOS, ffmpeg, a Hugging Face token and a ~4 GB download. Two things
fall between them: both are cheap, both are invisible to every job that runs on
push, and both fail in production rather than in a test.
"""

from __future__ import annotations

import importlib.util
import pathlib

from utils.models import Transcript

APP = pathlib.Path(__file__).resolve().parent.parent / "streamlit_app.py"


def test_sentencepiece_is_installed() -> None:
    """The `[stt]` extra must keep supplying sentencepiece.

    `CohereAsrTokenizer.__init__` imports it unconditionally, from a
    ``post_load_hook`` that mlx-audio runs as the last step of a load -- *after*
    ``load_weights(strict=True)`` has already passed. So losing it fails neither
    at resolve time nor at import time: it fails four gigabytes in, on the first
    real transcription, past the one check this app treats as its integrity gate.

    Until mlx-audio 0.5.1 it arrived by accident, as a transitive of mlx-lm, so
    dropping `[stt]` from `pyproject.toml` is a one-token edit that reads as
    tidying. Nothing else here would notice. This file and `test_pure.py` never
    import `mlx_audio`; CI's `check` job runs `check_decoding`, which needs no
    model; and only `integration`, which is `workflow_dispatch` only, loads a
    checkpoint at all. `find_spec` rather than an import because the question is
    whether `uv sync` put it in the environment, not whether it initialises.
    """
    assert importlib.util.find_spec("sentencepiece") is not None, (
        "sentencepiece is missing. pyproject.toml must request `mlx-audio[stt]` -- "
        "that extra is the only thing naming it, and without it every load_asr() "
        "call raises ModuleNotFoundError after the checkpoint has downloaded."
    )


def _app():
    """An AppTest for the real script, imported lazily to keep collection cheap."""
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=60)


def test_app_script_runs() -> None:
    """`streamlit_app.py` must execute top to bottom on the pinned Streamlit.

    Nothing else runs it. `test_pure.py` imports only `utils`, ruff and ty are
    static, and `check_decoding` and `verify_transcription.py` both stop at
    `utils/`. So a Streamlit release that drops a keyword this app passes -- and
    it passes several recent ones deliberately, per CLAUDE.md's conventions --
    ships with every job green and is found by the first person to type
    `streamlit run`.

    Reaches no model. AppTest stops after the first render and never presses
    Transcribe, so `load_asr` is never called and nothing imports `mlx_audio`:
    `utils/` keeps those imports inside the functions that need them. That is
    what lets this sit in the ubuntu `test` job, which has no Apple Silicon, no
    ffmpeg and no token -- and it is why this is not the mocked-`generate` test
    `test_pure.py` rules out, since the run never reaches transcription.
    """
    app = _app().run()

    assert not app.exception, [str(e) for e in app.exception]
    # Not merely "no exception". A script that rendered nothing would satisfy
    # that too, and an empty page is the shape this failure takes.
    assert app.sidebar.selectbox, "the language picker did not render"


def test_app_renders_a_finished_result() -> None:
    """The result half must render too, which an empty run does not reach.

    `streamlit_app.py:28` reads `result` out of session state and everything
    below -- the metrics row, the three download buttons, the chunk table -- is
    gated on it. So an unseeded AppTest exercises only the input half, and a bad
    keyword in the result section passes it: measured, by mutating
    `st.dataframe(lazy=True)` to carry a nonexistent argument and watching
    `test_app_script_runs` stay green while this test turns red.

    Seeding a `Transcript` is enough, and needs nothing expensive: it is a plain
    dataclass, so no decode, no checkpoint and no network are involved. Two
    segments rather than one because the chunk expander is itself gated on
    `len(result.segments) > 1`, and that expander is where the recent Streamlit
    APIs are densest.
    """
    app = _app()
    app.session_state["result"] = Transcript(
        source_key="smoke",
        source_name="meeting.wav",
        text="the quick brown fox",
        language="en",
        duration_s=42.0,
        elapsed_s=1.5,
        segments=[
            {"start": 0.0, "end": 20.0, "text": "the quick brown"},
            {"start": 20.0, "end": 42.0, "text": "fox"},
        ],
    )
    app.run()

    assert not app.exception, [str(e) for e in app.exception]
    assert len(app.dataframe) == 1, "the chunk table did not render"
