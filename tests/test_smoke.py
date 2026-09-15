"""Checks that need no checkpoint, and that neither other layer makes.

`tests/test_pure.py` covers the pure functions and the integration test's own
oracle. `verify_transcription.py` covers real decoding against real weights, at
the cost of macOS, ffmpeg, a Hugging Face token and a ~4 GB download. Three
things fall between them: all are cheap, all are invisible to every job that
runs on push, and all fail in production rather than in a test.
"""

from __future__ import annotations

import importlib.util
import pathlib

import toml

from utils.models import Transcript

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = ROOT / "streamlit_app.py"
CONFIG = ROOT / ".streamlit" / "config.toml"


def test_sentencepiece_is_installed() -> None:
    """The `[stt]` extra must keep supplying sentencepiece.

    `CohereAsrTokenizer.__init__` imports it unconditionally, from a
    ``post_load_hook`` that mlx-audio runs as the last step of a load -- *after*
    ``load_weights(strict=True)`` has already passed. So losing it fails neither
    at resolve time nor at import time: it fails four gigabytes in, on the first
    real transcription, past the one check this app treats as its integrity gate.

    Until mlx-audio 0.5.0 it arrived by accident, as a transitive of mlx-lm, so
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

    `streamlit_app.py:36` reads `result` out of session state and everything
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


def _theme_tables(table: dict, section: str = "theme"):
    """Yield ``(section, keys)`` for the theme table and every nested one."""
    yield section, {k: v for k, v in table.items() if not isinstance(v, dict)}
    for name, sub in table.items():
        if isinstance(sub, dict):
            yield from _theme_tables(sub, f"{section}.{name}")


def test_theme_config_keeps_its_invariants() -> None:
    """`.streamlit/config.toml` must stay inside the three rules its header states.

    Each of these fails silently. A key in the flat `[theme]` table is inherited
    by both variants and, with no variant table beside it, pins every visitor to
    light regardless of their OS -- `baseFontSize` alone was enough, and it is
    how the theme came to be written under `[theme.dark]` in the first place. A
    `font = "Name:https://..."` line -- the form every bundled Streamlit theme
    template uses -- has the browser fetch Google on every cold load, which is
    the one request the emoji favicon was chosen to close. And a key Streamlit
    registers for a different table (`showSidebarBorder` and the font-size keys
    exist only flat) is logged once at startup and dropped, so the value the
    comment beside it argues for is never in effect.

    AppTest, which the two tests above run under, parses this file and catches
    none of the three: it has no frontend, so a flat key and a font URL are
    inert there, and the dropped key's warning is merely captured by pytest.
    The registry check goes through `get_options_for_section`, which is how
    `app_session.py` itself decides what reaches the proto, rather than through
    the parser's log line. Values are not checked here -- the ratios are the
    file's own argument, re-measured when a value changes, not a structure a
    test can hold.
    """
    from streamlit import config

    # `toml`, not `tomllib`: the latter is 3.11+, and `requires-python` claims
    # 3.10. `toml` is what `streamlit.config` reads this file with, so this is
    # also the parse Streamlit actually performs rather than a near one.
    theme = toml.loads(CONFIG.read_text(encoding="utf-8")).get("theme", {})

    assert theme.get("dark") or theme.get("light"), (
        "a variant table must carry at least one key, or Streamlit offers no "
        "light/dark switch and a flat key would pin every visitor to light"
    )
    for section, keys in _theme_tables(theme):
        if section == "theme":
            assert not keys, (
                f"flat [theme] keys {sorted(keys)} restyle both variants and, "
                "without a variant table, pin every visitor to light -- move "
                "them under [theme.dark] or [theme.light]"
            )
        registered = set(config.get_options_for_section(section))
        assert set(keys) <= registered, (
            f"[{section}] carries {sorted(set(keys) - registered)}, which "
            f"Streamlit registers for a different table: it is logged as not a "
            "valid config option at startup and dropped, so nothing it argues "
            "for is in effect"
        )
        assert "fontFaces" not in keys, f"[{section}] must not self-host fonts"
        for key in ("font", "headingFont", "codeFont"):
            assert "://" not in str(keys.get(key, "")), (
                f"[{section}] {key} names a stylesheet URL; the browser would "
                "fetch it on every cold load, and the only network calls this "
                "app makes are Hugging Face weight downloads"
            )
