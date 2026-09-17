"""Checks that need no checkpoint, and that neither other layer makes.

`tests/test_pure.py` covers the pure functions and the integration test's own
oracle. `verify_transcription.py` covers real decoding against real weights, at
the cost of macOS, ffmpeg, a Hugging Face token and a ~4 GB download. What
falls between them is cheap, invisible to every job that runs on push, and
fails in production rather than in a test: that the `[stt]` extra still
supplies sentencepiece; that `streamlit_app.py` runs on the pinned Streamlit,
bare and with a result to render; that the page's own rules hold under
AppTest, which can upload, clear, select and set values -- the input mode
stays lit, Transcribe is gated on an upload, the preview player is not served
as WAV, a re-dropped file keeps its transcript and a new one drops it, the
downloads never rerun and go dark on a no-speech result, the transcript and
the escaped filename reach the screen as the characters the downloads write,
the sidebar opens at the defaults a press hands to `load_asr` and `generate`,
and the transcript sits alone in a bordered, capped slot; that `page_icon`
stays an emoji, recorded by monkeypatching `st.set_page_config` because
AppTest discards page config; and that `.streamlit/config.toml` keeps the
rules its comments state.

None of it presses Transcribe. Every run here stops short of `load_asr`, so
nothing imports `mlx_audio`, and that is what keeps this file in the ubuntu
`test` job -- and what keeps it clear of the mocked-`generate` test
`test_pure.py` rules out.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib

import pytest

from utils.audio import UPLOAD_TYPES
from utils.models import DEFAULT_REPO, LANGUAGES, Transcript

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = ROOT / "streamlit_app.py"
CONFIG = ROOT / ".streamlit" / "config.toml"

# Never decoded: Transcribe is never pressed, so the bytes only have to be
# something the uploader accepts. The digest mirrors source_key's blake2b with
# digest_size=16 -- a change to that algorithm is a change to what "same source"
# means, and must land here too or every keep/drop test below reads a new source.
PAYLOAD = b"RIFF" + bytes(60)
DIGEST = hashlib.blake2b(PAYLOAD, digest_size=16).hexdigest()
SEGMENTS = [
    {"start": 0.0, "end": 20.0, "text": "the quick brown"},
    {"start": 20.0, "end": 42.0, "text": "fox"},
]


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


def _seeded(
    source_name: str = "meeting.wav",
    text: str = "the quick brown fox",
    segments: list[dict] | None = None,
):
    """An AppTest with a finished `Transcript` already in session state.

    A plain dataclass, so no decode, no checkpoint and no network. Its
    ``source_key`` is the digest of ``PAYLOAD``, so uploading those bytes reads
    as the same source and any other bytes as a new one. Two segments by default
    because the chunk expander is gated on ``len(result.segments) > 1``, and
    that expander is where the recent Streamlit APIs are densest. Named keywords
    rather than ``**overrides`` merged into a dict, for the reason
    ``test_pure.py``'s helper gives: ty cannot narrow the merged mapping back to
    the dataclass fields.
    """
    app = _app()
    app.session_state["result"] = Transcript(
        source_key=DIGEST,
        source_name=source_name,
        text=text,
        language="en",
        duration_s=42.0,
        elapsed_s=1.5,
        segments=SEGMENTS if segments is None else segments,
    )
    return app


# --- The page renders -----------------------------------------------------


def test_app_script_runs() -> None:
    """`streamlit_app.py` must execute top to bottom on the pinned Streamlit.

    Nothing else runs it. `test_pure.py` imports only `utils`, ruff and ty are
    static, and `check_decoding` and `verify_transcription.py` both stop at
    `utils/`. So a Streamlit release that drops a keyword this app passes -- and
    it passes several recent ones deliberately, per CLAUDE.md's conventions --
    ships with every job green and is found by the first person to type
    `streamlit run`.

    Reaches no model. AppTest never presses Transcribe, so `load_asr` is never
    called and nothing imports `mlx_audio`: `utils/` keeps those imports inside
    the functions that need them. That is what lets this sit in the ubuntu
    `test` job, which has no Apple Silicon, no ffmpeg and no token -- and it is
    why this is not the mocked-`generate` test `test_pure.py` rules out, since
    the run never reaches transcription.
    """
    app = _app().run()

    assert not app.exception, [str(e) for e in app.exception]
    # Not merely "no exception". A script that rendered nothing would satisfy
    # that too, and an empty page is the shape this failure takes.
    assert app.sidebar.selectbox, "the language picker did not render"
    # Addressed through the column, not the page: the placeholder is still
    # *somewhere* on the page when `with transcript_slot:` is dropped, so a
    # root-level `app.caption` cannot tell the reading column's box from a
    # caption written under the button.
    assert [c.value for c in app.columns[1].caption] == [
        "The transcript appears here."
    ], "the placeholder did not land in the reading column's box"


def test_app_renders_a_finished_result() -> None:
    """The result half must render too, which an empty run does not reach.

    The `st.session_state.setdefault("result", None)` read at the top of
    `streamlit_app.py` pulls `result` out of session state, and everything gated
    on it -- the transcript text, the metrics row, the three download buttons,
    the chunk table -- is what a bare run never reaches. (The bordered box in
    the reading column is drawn either way: `transcript_slot` carries the
    border and the cap, and the placeholder branch puts a caption in it.) So an
    unseeded AppTest exercises the input half and an empty box, and a bad
    keyword in the result section passes it: measured, by mutating
    `st.dataframe(lazy=True)` to carry a nonexistent argument and watching
    `test_app_script_runs` stay green while this test turns red.

    That is the keyword, not the delivery. AppTest keeps every dataframe eager
    -- `global.appTest` short-circuits the lazy resolver after the columns are
    validated -- so this run checks that `lazy=True` is accepted and the columns
    pass validation, and no row count seeded here can reach the chunked
    delivery a long VAD meeting relies on.
    """
    app = _seeded().run()

    assert not app.exception, [str(e) for e in app.exception]
    (table,) = app.dataframe
    assert len(table.value) == 2, "the chunk table did not render both rows"
    # column_order is copied into the proto unvalidated, and a name matching no
    # column is silently hidden. `.value.columns` is the Arrow payload and cannot
    # see it: measured, a "txt" typo leaves it ['start', 'end', 'text'].
    assert list(table.proto.column_order) == ["start", "end", "text"]
    # app.main.status, not app.expander: 1.63.0 classifies an st.expander that
    # carries icon= as Status, so both expanders on this page live under
    # `status` and `app.expander` is empty on every run. The sidebar's Advanced
    # is the other one, which is why this is scoped to main.
    (chunks,) = app.main.status
    assert chunks.label == "2 chunks" and chunks.icon == ":material/segment:"
    assert not chunks.proto.expanded, "the chunk table opened itself"
    assert [t.value for t in app.columns[1].text] == ["the quick brown fox"], (
        "the transcript did not land in the reading column's box"
    )


def test_result_renders_verbatim() -> None:
    """The transcript and the filename reach the screen as the same characters
    the downloads write.

    `st.text`, not `st.markdown`: swapped, the frontend eats the asterisks and
    makes the dash a bullet. AppTest has no frontend, so what this test sees on
    a swap is the string arriving byte-identical under `app.markdown` instead
    of `app.text` -- which is why `text not in app.markdown` is the guard, and
    why asserting the mangled form is absent would pass either way. The caption
    is the other direction -- Markdown on purpose, so the filename is escaped;
    deleting the escape gives 'English · take*2*.wav'. AppTest returns the
    Markdown *source*, so this pins the escape and cannot see the typographer
    pass `escape_markdown` documents. The metric rows pin the app's own
    strings: `st.metric` hands a `str` to the proto untouched, and the Elapsed
    and Speed format specs (`:.1f`s, `:.0f`×) exist only at their `st.metric`
    calls, so no other test executes them. Audio goes through
    `format_duration`, which `test_pure.py` already covers; its row checks the
    wiring, not the format.
    """
    text = "*music* $5-$10\n- item"
    app = _seeded(source_name="take*2*.wav", text=text).run()

    assert not app.exception, [str(e) for e in app.exception]
    assert [(m.label, m.value) for m in app.metric] == [
        ("Audio", "0:42"),
        ("Elapsed", "1.5s"),
        ("Speed", "28×"),
    ]
    assert [t.value for t in app.text] == [text]
    assert text not in [m.value for m in app.markdown]
    # By membership, not index: the filename caption moved columns once, and
    # with two segments the last caption on the page is the chunk expander's.
    assert r"English · take\*2\*\.wav" in [c.value for c in app.caption], (
        "the filename caption is no longer escaped"
    )


def test_one_chunk_is_not_a_table() -> None:
    """The chunk expander is gated on `len(result.segments) > 1`, and the lower
    side of that gate is the one a relaxation to `if result.segments:` opens."""
    app = _seeded(
        segments=[{"start": 0.0, "end": 42.0, "text": "the quick brown fox"}]
    ).run()

    assert not app.exception, [str(e) for e in app.exception]
    assert not app.dataframe and not app.main.status, "one chunk is not a table"


# --- Downloads ------------------------------------------------------------


def test_downloads_never_rerun() -> None:
    """`on_click="ignore"` keeps the three downloads frontend-only.

    Every payload comes from session state, so the default "rerun" re-executes
    the whole script to arrive at an identical screen. The flag lands on the
    proto as `ignore_rerun`, which is the only place a test can see it.
    """
    app = _seeded().run()

    assert not app.exception, [str(e) for e in app.exception]
    assert [b.label for b in app.download_button] == ["Text", "SRT", "VTT"]
    assert all(b.proto.ignore_rerun for b in app.download_button), (
        "on_click='ignore' was dropped: every download click now reruns the script"
    )


def test_no_speech_result_darkens_every_download() -> None:
    """A no-speech result still has segments, so without `disabled=not
    result.text` the Text button stays lit and hands over an empty file.

    SRT and VTT go dark on their own: `_cues` drops blank-text segments, so both
    writers return "" -- the behaviour `to_vtt`'s docstring exists to guarantee.
    The placeholder is the app's own string, so it may render as Markdown.
    """
    app = _seeded(
        text="",
        segments=[
            {"start": 0.0, "end": 20.0, "text": ""},
            {"start": 20.0, "end": 42.0, "text": " "},
        ],
    ).run()

    assert not app.exception, [str(e) for e in app.exception]
    assert not app.text
    assert "_No speech detected._" in [m.value for m in app.markdown]
    assert [(b.label, b.disabled) for b in app.download_button] == [
        ("Text", True),
        ("SRT", True),
        ("VTT", True),
    ], "a no-speech result must not hand over an empty file"


def test_no_segments_disables_subtitles_but_not_text() -> None:
    """The other half of the rule: text with nothing to cue keeps Text lit."""
    app = _seeded(segments=[]).run()

    assert not app.exception, [str(e) for e in app.exception]
    assert [t.value for t in app.text] == ["the quick brown fox"]
    assert [(b.label, b.disabled) for b in app.download_button] == [
        ("Text", False),
        ("SRT", True),
        ("VTT", True),
    ]
    assert not app.dataframe, "the chunk expander is gated on len(segments) > 1"


# --- Input ----------------------------------------------------------------


def test_input_mode_stays_lit() -> None:
    """`required=True` keeps a click on the lit segment from returning None.

    Deselected, the mode matches neither label, falls through to the else and
    draws the uploader under a control with nothing selected; `default=` keeps
    the first render out of that state. Neither can be driven here -- AppTest
    hands the script the default after `set_value(None)` or `unselect()`
    whether `required` is True or False (measured) -- so the proto field the
    frontend enforces is the check. Selecting Record catches the option label
    drifting from the `mode == "Record"` literal, which otherwise draws the
    uploader in Record mode with nothing red, and runs the `st.audio_input`
    line the bare run never reaches. Two other tests switch to Record on their
    way elsewhere and would fail on the same drift, one assertion later and
    with nothing naming the cause; this is where it gets its diagnosis.
    """
    app = _app().run()
    (mode,) = app.segmented_control

    assert mode.value == "Upload a file", "the mode control did not open lit"
    assert mode.proto.required, (
        "required=True keeps the lit segment from returning None, which falls "
        "through to the else and draws the uploader under an empty control"
    )
    assert app.file_uploader, "the default mode did not draw the uploader"

    mode.select("Record").run()
    assert not app.exception, [str(e) for e in app.exception]
    assert not app.file_uploader, (
        'Record still drew the uploader: the option label and the `mode == "Record"` '
        "comparison have drifted apart"
    )
    assert app.get("audio_input"), "Record did not draw the recorder"


def test_transcribe_is_gated_on_an_upload() -> None:
    """`disabled=audio_file is None` on the Transcribe button.

    Without it a lit primary button's click reaches `if run and audio_file is
    not None`, which draws no status box and no error -- a press that does
    nothing, silently. The upload here is also what reaches the `st.audio`
    line, which neither the bare nor the seeded run executes.
    """
    app = _app().run()
    button = app.button[0]
    assert button.label == "Transcribe" and button.proto.type == "primary"
    assert button.disabled, "Transcribe is lit with nothing to transcribe"

    app.file_uploader[0].upload("clip.wav", PAYLOAD, "audio/wav").run()
    assert not app.exception, [str(e) for e in app.exception]
    assert not app.button[0].disabled

    app.file_uploader[0].clear().run()
    assert app.button[0].disabled

    app.segmented_control[0].set_value("Record").run()
    assert not app.exception
    assert app.button[0].disabled, (
        "Record mode with no recording must not enable Transcribe"
    )


@pytest.mark.parametrize("ext", [e for e in UPLOAD_TYPES if e != "wav"])
def test_preview_player_is_not_served_as_wav(ext: str) -> None:
    """`st.audio` must carry the `format=` that `preview_mime` derives.

    `test_pure.py` tests `preview_mime`; this tests that the page passes its
    answer on. `st.audio` has no AppTest wrapper -- it is an UnknownElement
    whose `.type` is the proto oneof name -- but the mock media store names the
    served file by the mimetype `st.audio` was given, so the URL's extension is
    the check. Negative on purpose, twice: the *right* extension comes from the
    platform mimetypes table, which varies by OS (.flac gets none on macOS,
    .ogg and .opus become .oga), while "audio/wav" -> ".wav" is Streamlit's own
    preferred map, and "application/octet-stream" -> ".bin" is what a
    `format=audio_file.type` would produce for an upload whose browser-supplied
    type is empty -- the tempting edit `preview_mime`'s comment names. Dropping
    `format=` turns every row into `.wav`; measured.
    """
    app = _app().run()
    app.file_uploader[0].upload(f"clip.{ext}", PAYLOAD, "application/octet-stream")
    app.run()

    assert not app.exception, [str(e) for e in app.exception]
    (player,) = app.get("audio")
    url = player.proto.url
    assert not url.endswith(".wav"), (
        f"a .{ext} upload is served as audio/wav: st.audio lost its format= argument"
    )
    assert not url.endswith(".bin"), (
        f"a .{ext} upload is served as application/octet-stream: st.audio is "
        "reading UploadedFile.type, which the browser leaves empty for .opus"
    )


# --- Session state --------------------------------------------------------
# The rule under test is CLAUDE.md's "same source means the same bytes". It has
# regressed once -- identity keyed on file_id, which Streamlit mints fresh per
# upload event, so re-dropping the file that produced the transcript on screen
# threw the transcript away -- and no test above could see it. The rendering
# tests never upload, and the Input tests upload with no transcript on screen,
# the one state in which source_key is never consulted: the comparison sits
# under `if result and audio_file is not None`. So every test here starts from
# _seeded() and then uploads, and AppTest's upload() mints a fresh uuid4 per
# call, exactly the per-event file_id the digest exists to survive. Verified by
# mutation: with source_key returning file.file_id the first test below fails
# at "re-dropping the same bytes discarded the transcript" while every test
# above stays green.


def test_same_bytes_keep_the_transcript_and_take_the_new_name() -> None:
    app = _seeded().run()

    app.file_uploader[0].upload("renamed.wav", PAYLOAD, "audio/wav").run()
    assert not app.exception, [str(e) for e in app.exception]
    result = app.session_state["result"]
    assert result is not None, "re-dropping the same bytes discarded the transcript"
    # Same bytes under a new name is a copy or a rename: the transcript stands,
    # and the caption and the download stems follow the file on screen.
    assert result.source_name == "renamed.wav"
    assert r"English · renamed\.wav" in [c.value for c in app.caption]

    # A second drop of the same file is a fresh file_id and the same digest.
    app.file_uploader[0].upload("renamed.wav", PAYLOAD, "audio/wav").run()
    assert len(app.dataframe) == 1
    assert app.session_state["digest"][1] == DIGEST


def test_new_bytes_drop_the_transcript() -> None:
    app = _seeded().run()

    app.file_uploader[0].upload("other.wav", b"RIFF" + b"\x01" * 60, "audio/wav")
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    assert app.session_state["result"] is None
    assert not app.metric and not app.download_button and not app.dataframe
    assert "The transcript appears here." in [c.value for c in app.caption]


def test_clearing_or_switching_to_record_keeps_the_transcript() -> None:
    """Only a *new* source invalidates. Clearing the uploader and switching to
    Record both empty the widget, and throwing away a long transcription there
    is unrecoverable."""
    app = _seeded().run()
    app.file_uploader[0].upload("meeting.wav", PAYLOAD, "audio/wav").run()
    assert app.session_state["result"] is not None, "the same bytes dropped it"

    app.file_uploader[0].clear().run()
    assert not app.exception, [str(e) for e in app.exception]
    assert app.session_state["result"] is not None, "clearing the uploader dropped it"
    assert app.button[0].disabled

    app.segmented_control[0].set_value("Record").run()
    assert not app.exception, [str(e) for e in app.exception]
    assert app.session_state["result"] is not None, "switching to Record dropped it"
    assert not app.file_uploader and app.get("audio_input")

    # Back to Upload, and a different file: now there is a new source.
    app.segmented_control[0].set_value("Upload a file").run()
    app.file_uploader[0].upload("other.wav", PAYLOAD[:-1] + b"\x01", "audio/wav")
    app.run()
    assert app.session_state["result"] is None
    assert not app.dataframe


# --- Sidebar and layout ---------------------------------------------------


def test_sidebar_defaults() -> None:
    """The six sidebar values a Transcribe press consumes -- five as
    `model.generate` keywords, `repo_id` as `load_asr`'s argument -- read rather
    than assumed truthy.

    `st.selectbox` defaults to its first option, and `LANGUAGES` is
    alphabetical with English hoisted -- tidying it into strict order flips
    every new session to "ar", found after the download and the minutes of
    generation. The merge gap is `disabled=not use_vad`, the sidebar's only
    wiring. `NumberInput.min/max/step` come back as floats that compare equal.
    """
    app = _app().run()

    assert not app.exception, [str(e) for e in app.exception]
    (language,) = app.sidebar.selectbox
    assert language.value == "en", "the picker defaults to LANGUAGES' first key"
    assert len(language.options) == len(LANGUAGES)
    assert [t.value for t in app.sidebar.toggle] == [True, False]  # punctuation, VAD
    (repo,) = app.sidebar.text_input
    assert repo.value == DEFAULT_REPO
    (tokens,) = app.sidebar.number_input
    assert (tokens.value, tokens.min, tokens.max, tokens.step) == (256, 64, 1024, 64)
    (gap,) = app.sidebar.slider
    assert (gap.value, gap.min, gap.max, gap.step) == (1.0, 0.1, 3.0, 0.1)
    assert gap.proto.format == "%.1f", "the float default renders a 0.1 step as 1.00"
    assert gap.disabled, "the merge gap is live while VAD is off"

    app.sidebar.toggle[1].set_value(True).run()
    assert not app.exception, [str(e) for e in app.exception]
    assert not app.sidebar.slider[0].disabled


def test_sidebar_choices_survive_a_reload() -> None:
    """`bind="query-params"` on language, punctuation and VAD.

    A reload starts a new session, and without the binding it also reset the
    three choices a transcription depends on. The URL carries the formatted
    label, not the code: a raw `?language=de` is not a value the selectbox
    recognises, so it falls back to English and the parameter is stripped --
    asserted here so the wart is a documented one. Dropping `bind=` leaves
    every other test green; measured.
    """
    app = _app()
    app.query_params["language"] = "German (de)"
    app.query_params["use_vad"] = "true"
    app.run()

    assert not app.exception, [str(e) for e in app.exception]
    assert app.sidebar.selectbox[0].value == "de", "the URL's language was ignored"
    assert [t.value for t in app.sidebar.toggle] == [True, True]
    assert not app.sidebar.slider[0].disabled, "VAD from the URL left the gap dark"

    app = _app()
    app.query_params["language"] = "de"
    app.run()
    assert not app.exception, [str(e) for e in app.exception]
    assert app.sidebar.selectbox[0].value == "en"
    assert app.query_params == {}, "an unrecognised value should be stripped"


def test_layout_keeps_its_structure() -> None:
    """The split's structure, and none of its geometry.

    A `width="stretch"` slip on the transcript slot lands as `use_stretch`,
    the same proto a dropped `width=` produces, and widens the box 740 -> 789
    px with the sidebar open -- seven characters a line nobody sees -- and
    becomes the ~140-character line only once the sidebar is collapsed. The
    measured values (the [4, 5] weights, the gap, the 740) are deliberately not
    pinned: they are the layout comments' argument, re-measured when one
    changes, the rule the theme test states for its own numbers.
    """
    app = _seeded().run()

    assert not app.exception, [str(e) for e in app.exception]
    left, right = app.columns
    assert left.weight < right.weight, (
        "the reading column is the one the cap has to fit inside"
    )
    (slot,) = right.children.values()  # the transcript alone on the right
    assert slot.proto.flex_container.border, "the transcript slot lost its border"
    assert slot.proto.width_config.WhichOneof("width_spec") == "pixel_width", (
        "the transcript slot is stretch, not capped"
    )
    assert [c.type for c in slot.children.values()] == ["text"]
    assert len(left.metric) == 3
    assert len(left.get("download_button")) == 3
    assert len(left.dataframe) == 1  # not left.expander: an icon= expander is "status"


# --- Page config ----------------------------------------------------------


def test_page_icon_fetches_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """`page_icon` must stay an emoji.

    A Material icon is resolved by the frontend to an SVG on fonts.gstatic.com
    and a URL is set as the favicon href verbatim; either has the browser fetch
    a third party on a cold cache, the one request that would make "the only
    network calls are weight downloads" untrue. An emoji is inlined as a data:
    URL. No AppTest run can see page config -- the runner queues the message,
    but the element tree is parsed from delta messages only and AppTest keeps
    no handle on the runner -- so the call is recorded on the way through, and
    the positive predicate Streamlit itself applies is the check: `is_emoji`
    strips U+FE0F, so the two-code-point microphone passes, and it rejects
    Material, URLs, "random" and shortcodes alike. A negative assertion on the
    ":material/" prefix let a URL through; measured.
    """
    import streamlit as st
    from streamlit.string_util import is_emoji

    calls: list[dict] = []
    real = st.set_page_config

    def record(*args, **kwargs):
        calls.append(kwargs)
        real(*args, **kwargs)

    monkeypatch.setattr(st, "set_page_config", record)
    app = _app().run()

    assert not app.exception, [str(e) for e in app.exception]
    assert calls, "st.set_page_config was not called"
    icon = calls[0].get("page_icon")
    assert isinstance(icon, str) and is_emoji(icon), (
        f"page_icon={icon!r} is not an emoji, so the tab icon fetches from a "
        "third party"
    )


# --- Config ---------------------------------------------------------------

# The six tables Streamlit accepts. Anything else -- [theme.sidebar.dark] is the
# natural mistake -- is not dropped but refused: the parser raises
# StreamlitInvalidThemeSectionError and the app does not start.
THEME_SECTIONS = frozenset(
    {
        "theme",
        "theme.sidebar",
        "theme.light",
        "theme.dark",
        "theme.light.sidebar",
        "theme.dark.sidebar",
    }
)


def _theme_tables(table: dict, section: str = "theme"):
    """Yield ``(section, keys)`` for the theme table and every nested one."""
    yield section, {k: v for k, v in table.items() if not isinstance(v, dict)}
    for name, sub in table.items():
        if isinstance(sub, dict):
            yield from _theme_tables(sub, f"{section}.{name}")


def _carries_a_key(table: dict) -> bool:
    """Mirror the frontend's test for "this variant is configured".

    It counts a key at any depth, so a lone ``[theme.dark.sidebar]`` entry is
    enough to build the light/dark pair.
    """
    return any(
        _carries_a_key(v) if isinstance(v, dict) else True for v in table.values()
    )


def test_theme_config_keeps_its_rules() -> None:
    """`.streamlit/config.toml` must stay inside the rules its header states.

    Each one fails without a test going red. A flat `[theme]` key with no
    variant table beside it sends one custom theme whose unset `base` decays
    to light and pins every visitor there regardless of their OS --
    `baseFontSize` alone was enough, and it is how the theme came to be
    written under `[theme.dark]` in the first place. With a variant table
    present a flat key is instead inherited by both halves, so a key that has
    a variant home and sits flat restyles light when dark was meant; only the
    keys Streamlit registers *nowhere else* -- `showSidebarBorder` and the
    font-size keys -- may sit there, and the file's header says why none does.
    A `font = "Name:https://..."` line -- the form every bundled theme template
    uses -- has the browser fetch Google on every cold load from the flat table
    or `[theme.sidebar]`, the one request the emoji favicon was chosen to
    close, and under a variant table is half-applied: the family name lands,
    nothing is fetched, and the text silently falls back. A key in a table it
    is not registered for is logged once at startup and dropped, so the value
    the comment beside it argues for is never in effect. And a table Streamlit
    does not accept at all is refused rather than dropped, which is the one of
    these that is loud -- it is asserted here first so the verdict does not
    depend on whether an earlier test already made Streamlit parse the file.

    AppTest, which every test above but the sentencepiece check runs under,
    parses this file and catches none of the silent ones: it has no frontend,
    and the dropped key's warning is merely captured by pytest. The registry
    check goes through `get_options_for_section`, which is how `app_session.py`
    itself decides what reaches the proto. Values are not checked: the ratios
    are the file's own argument, re-measured when a value changes, not a
    structure a test can
    hold. `toml`, not `tomllib`, and imported here rather than at module scope:
    `tomllib` is 3.11+ against a `requires-python` of 3.10, `toml` is what
    `streamlit.config` reads this file with, and it reaches this environment
    only as Streamlit's own dependency, so a Streamlit that stopped needing it
    must take down this test alone and not the module's collection.
    """
    import toml
    from streamlit import config

    theme = toml.loads(CONFIG.read_text(encoding="utf-8")).get("theme", {})
    tables = list(_theme_tables(theme))

    for section, _ in tables:
        assert section in THEME_SECTIONS, (
            f"[{section}] is not a table Streamlit accepts; it refuses to start "
            f"on it rather than dropping it. Valid: {sorted(THEME_SECTIONS)}"
        )

    flat = dict(tables[0][1])
    if flat:
        assert _carries_a_key(theme.get("light", {})) or _carries_a_key(
            theme.get("dark", {})
        ), (
            f"flat [theme] keys {sorted(flat)} with no [theme.light] or "
            "[theme.dark] beside them send one custom theme whose unset base is "
            "light, pinning every visitor there whatever their OS says"
        )

    variant_homes = set(config.get_options_for_section("theme.dark"))
    for section, keys in tables:
        assert "fontFaces" not in keys, (
            f"[{section}] self-hosts fonts, which needs [server] "
            "enableStaticServing (off here) and admits a remote url"
        )
        for key in ("font", "headingFont", "codeFont"):
            if "://" in str(keys.get(key, "")):
                raise AssertionError(
                    f"[{section}] {key} names a stylesheet URL. "
                    + (
                        "From this table the browser fetches it on every cold "
                        "load, and the only network calls this app makes are "
                        "Hugging Face weight downloads."
                        if section in ("theme", "theme.sidebar")
                        else "Under a variant table it is half-applied: the "
                        "family name lands, nothing is fetched, and the text "
                        "silently falls back to whatever is installed."
                    )
                )
        if section == "theme":
            misplaced = sorted(set(keys) & variant_homes)
            assert not misplaced, (
                f"flat [theme] keys {misplaced} are inherited by both halves and "
                "restyle light along with dark; move them under [theme.dark] or "
                "[theme.light]"
            )
        registered = set(config.get_options_for_section(section))
        assert set(keys) <= registered, (
            f"[{section}] carries {sorted(set(keys) - registered)}, which "
            "Streamlit registers for a different table: it is logged as not a "
            "valid config option at startup and dropped, so nothing it argues "
            "for is in effect"
        )


def test_config_keeps_usage_stats_off() -> None:
    """`[browser] gatherUsageStats = false` is what makes the app's headline
    claim true, and nothing else checks it.

    The key defaults to True. `app_session.py` ships it to the frontend, whose
    MetricsManager fetches a metrics endpoint from data.streamlit.io and POSTs
    events there when it is true -- from the browser, so no amount of reading
    the app's own source turns it up. Cutting the section logs nothing, a
    mistyped key logs one line pytest captures, and AppTest has no frontend.
    Raw toml rather than `config.get_option`: the lazy parse is cwd-dependent
    and also reads ~/.streamlit/config.toml, so a developer's global file could
    answer for the repo's.
    """
    import toml

    raw = toml.loads(CONFIG.read_text(encoding="utf-8"))
    assert raw.get("browser", {}).get("gatherUsageStats") is False, (
        "browser.gatherUsageStats is not false: it defaults to true, and the "
        "frontend POSTs events to data.streamlit.io from the browser"
    )
