import hashlib
import time

import streamlit as st

st.set_page_config(
    page_title="Cohere Transcribe",
    page_icon=":material/graphic_eq:",
)

from utils.audio import (
    UPLOAD_TYPES,
    decode_to_mono16k,
    format_duration,
    preview_mime,
    to_srt,
    to_vtt,
)
from utils.models import (
    DEFAULT_REPO,
    LANGUAGES,
    ModelAccessError,
    ModelWeightsError,
    Transcript,
    load_asr,
)

result: Transcript | None = st.session_state.setdefault("result", None)

# --- Settings -------------------------------------------------------------
# Rendered before any slow work so the sidebar never sits greyed out behind a
# model load.

with st.sidebar:
    st.subheader("Transcription")
    language = st.selectbox(
        "Language",
        LANGUAGES,
        format_func=lambda code: f"{LANGUAGES[code]} ({code})",
        key="language",
        help="Cohere Transcribe has no language detection and does not handle "
        "code-switching, so the language has to be set explicitly.",
    )
    punctuation = st.toggle(
        "Punctuation and casing",
        value=True,
        help="Switches the prompt's punctuation token. Turn it off for raw "
        "lowercase text without punctuation.",
    )
    use_vad = st.toggle(
        "Trim silence with VAD",
        value=False,
        help="Splits on detected speech instead of fixed 35-second windows. "
        "Worth it for meetings and podcasts, where it stops the model from "
        "hallucinating over silence. Skip it for clean narration, where it "
        "adds errors at the extra chunk boundaries.",
    )

    advanced = st.expander("Advanced", icon=":material/tune:")
    with advanced:
        repo_id = st.text_input(
            "Checkpoint",
            value=DEFAULT_REPO,
            help="mlx-audio converts the upstream Hugging Face weights on load. The "
            "community `-mlx-` conversions on the Hub are built for the Swift "
            "runtime and will be rejected here.",
        )
        max_tokens = st.number_input(
            "Max tokens per chunk", min_value=64, max_value=1024, value=256, step=64
        )
        vad_merge_gap_s = st.slider(
            "VAD merge gap (seconds)",
            0.1,
            3.0,
            1.0,
            0.1,
            help="Speech runs closer together than this are merged into one chunk.",
            disabled=not use_vad,
        )

# --- Input ----------------------------------------------------------------

st.title("Cohere Transcribe")
st.caption(
    "Speech to text running locally on Apple Silicon through MLX. "
    "Nothing is uploaded anywhere."
)

# required=True, because single-selection segmented controls are deselectable by
# default: clicking the lit segment returns None, which falls through to the else
# below and draws the uploader under a control with nothing selected.
mode = st.segmented_control(
    "Input",
    ["Upload a file", "Record"],
    default="Upload a file",
    required=True,
    label_visibility="collapsed",
)

if mode == "Record":
    audio_file = st.audio_input("Recording")
else:
    audio_file = st.file_uploader("Audio file", type=UPLOAD_TYPES)
    if audio_file is not None:
        # format= is not inferred: st.audio defaults it to "audio/wav" and hands
        # it straight to the media file manager, which serves the bytes under
        # that Content-Type. Eight of the nine UPLOAD_TYPES are not WAV, and
        # browsers that pick a decoder from the header rather than sniffing the
        # container play none of them — silently, since the file still
        # transcribes fine while only the preview player looks broken. Derived
        # from the extension rather than read off UploadedFile.type; see
        # preview_mime for why that attribute cannot do the job.
        st.audio(audio_file, format=preview_mime(audio_file.name))


def source_key(file) -> str:
    """Identify the audio by its bytes, not by the upload event.

    Streamlit mints a fresh uuid4 ``file_id`` per upload, so re-dropping the very
    file that produced the transcript on screen read as a new source and
    discarded it. Both inputs reach that: clearing the uploader keeps the
    transcript but takes the player away, and switching to Record unmounts the
    uploader and lets its widget state be pruned, so a re-upload is the only way
    back to either.

    Behind a call rather than computed inline, because the digest is wanted only
    twice -- to compare against a transcript that already exists, and to label a
    new one -- and neither holds on the first upload of a session, which is every
    session. Hashing a gigabyte the moment a file lands, to protect a transcript
    that is not there, is dead time before the user has clicked anything.

    ``getvalue()``, not ``getbuffer()``: ``UploadedFile`` is a ``BytesIO`` built
    around the upload record's bytes, so ``getvalue()`` hands back that very
    object under CPython's copy-on-write rule while ``getbuffer()`` has to
    unshare the buffer and memcpy it. Measured on a 300 MB payload: +0 MB RSS
    against +300 MB, same digest. At the 1000 MB ceiling
    ``.streamlit/config.toml`` allows, the wrong one is a spare gigabyte.

    Cached against ``file_id`` as a pair rather than a dict, so there is no
    eviction step whose necessity lives in a comment.
    """
    if file is None:
        return ""
    cached_id, digest = st.session_state.get("digest", (None, ""))
    if cached_id != file.file_id:
        digest = hashlib.blake2b(file.getvalue(), digest_size=16).hexdigest()
        st.session_state.digest = (file.file_id, digest)
    return digest


# Drop a stale transcript as soon as the source changes, so the text on screen
# always belongs to the audio on screen. Only when there *is* a new source:
# switching to Record, or clearing the uploader, empties the widget without
# invalidating what was already transcribed, and throwing away a long
# transcription there is unrecoverable.
if result and audio_file is not None:
    if result.source_key != source_key(audio_file):
        result = None
        st.session_state.result = None
    else:
        # Same bytes under a different filename -- a copy, a rename, a download
        # of one's own upload. The transcript is still the right one, but the
        # caption and the download stems are built from source_name, which would
        # otherwise keep naming a file the player is no longer showing.
        result.source_name = audio_file.name

run = st.button(
    "Transcribe",
    type="primary",
    icon=":material/graphic_eq:",
    width="stretch",
    disabled=audio_file is None,
)

status_slot = st.container()
result_slot = st.container()

# --- Transcription --------------------------------------------------------

if run and audio_file is not None:
    with status_slot, st.status("Transcribing", expanded=True) as status:
        try:
            wall_clock = time.perf_counter()
            st.write("Decoding audio")
            waveform, duration_s = decode_to_mono16k(audio_file)

            st.write(f"Loading {repo_id}")
            st.caption(
                "The first run downloads the weights, which takes a few minutes."
            )
            model = load_asr(repo_id)

            st.write(f"Transcribing {format_duration(duration_s)} of audio")
            started = time.perf_counter()
            output = model.generate(
                waveform,
                sample_rate=16_000,
                language=language,
                punctuation=punctuation,
                max_tokens=int(max_tokens),
                vad=use_vad,
                vad_merge_gap_s=float(vad_merge_gap_s),
            )
            elapsed = time.perf_counter() - started
        except ModelAccessError as exc:
            status.update(label="Sign in required", state="error")
            st.error(str(exc), icon=":material/lock:")
        except ModelWeightsError as exc:
            status.update(label="Incompatible checkpoint", state="error")
            st.error(str(exc), icon=":material/extension_off:")
        except Exception as exc:  # noqa: BLE001 - shown to the user as-is
            status.update(label="Transcription failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}", icon=":material/error:")
        else:
            result = Transcript(
                source_key=source_key(audio_file),
                source_name=getattr(audio_file, "name", "recording"),
                # Stripped once, here, rather than left to the renderer: st.text
                # runs its body through textwrap.dedent().strip(), and decoders
                # in this family routinely emit a leading space, so an unstripped
                # string would show trimmed on screen while the Text download
                # wrote the original — the very mismatch st.text was chosen to
                # remove. Only the ends are at stake: join_chunk_texts joins
                # chunks with a single space and emits no newlines, so there is
                # no common indent for dedent to find.
                text=output.text.strip(),
                segments=output.segments or [],
                language=language,
                duration_s=duration_s,
                elapsed_s=elapsed,
            )
            st.session_state.result = result
            # Total wall clock, not `elapsed`: on a first run the decode and the
            # ~4 GB download dominate, and collapsing to "Transcribed in 2.1s"
            # after a several minute wait reads as a stopwatch that lied. The
            # Elapsed metric below stays generation-only, since RTFx needs that.
            status.update(
                label=f"Done in {time.perf_counter() - wall_clock:.1f}s",
                state="complete",
                expanded=False,
            )

# --- Result ---------------------------------------------------------------

if result:
    with result_slot:
        with st.container(horizontal=True):
            st.metric(
                "Audio",
                format_duration(result.duration_s),
                border=True,
                icon=":material/schedule:",
            )
            st.metric(
                "Elapsed",
                f"{result.elapsed_s:.1f}s",
                border=True,
                icon=":material/timer:",
            )
            st.metric(
                "Speed",
                f"{result.speedup:.0f}×",
                border=True,
                icon=":material/speed:",
                help="Audio seconds transcribed per second of wall clock (RTFx).",
            )

        with st.container(border=True):
            # st.text, not st.markdown: this is uncontrolled model output and the
            # product is a verbatim transcript. Markdown eats what the decoder
            # emits — a hallucinated `*music*` renders italic with the asterisks
            # gone, a leading "- " becomes a bullet, "$5-$10" renders as math —
            # while the Text download below writes the unparsed string, so the
            # file and the screen stop being the same characters. st.text is not
            # monospace (that is st.code), so nothing about the look changes, and
            # its own dedent().strip() is a no-op because Transcript already
            # holds the stripped string — which is also what keeps a
            # whitespace-only result falsy here and on the download button.
            if result.text:
                st.text(result.text, width="stretch")
            else:
                st.markdown("_No speech detected._")
            st.caption(f"{LANGUAGES[result.language]} · {result.source_name}")

        # Built once here rather than inline in the buttons, which re-serialised
        # the whole segment list on every rerun even while disabled.
        srt = to_srt(result.segments) if result.segments else ""
        vtt = to_vtt(result.segments) if result.segments else ""

        with st.container(horizontal=True):
            # on_click="ignore" keeps these frontend-only. Every payload comes
            # from st.session_state.result, so the default "rerun" re-executes
            # the whole script to arrive at an identical screen — rebuilding srt
            # and vtt above, re-marshalling all three payloads (which happens
            # before `disabled` is applied, so the greyed-out ones pay too) and
            # re-serialising the chunk table. It is the one interaction here that
            # cannot change a pixel.
            st.download_button(
                "Text",
                result.text,
                file_name=f"{result.stem}.txt",
                icon=":material/description:",
                # Same rule as SRT/VTT: a no-speech result still has segments, so
                # without this the one button left lit hands over an empty file.
                disabled=not result.text,
                on_click="ignore",
            )
            st.download_button(
                "SRT",
                srt,
                file_name=f"{result.stem}.srt",
                icon=":material/subtitles:",
                disabled=not srt,
                on_click="ignore",
            )
            st.download_button(
                "VTT",
                vtt,
                file_name=f"{result.stem}.vtt",
                icon=":material/subtitles:",
                disabled=not vtt,
                on_click="ignore",
            )

        if len(result.segments) > 1:
            # on_change="rerun" is what makes `.open` mean anything; the default
            # computes and ships expander contents whether or not it is open. VAD
            # splits per speech run rather than per 35-second window, so a long
            # meeting is thousands of rows Arrow-serialised on every rerun for a
            # section nobody expanded.
            #
            # Only above a threshold, though, because the saving is not free: with
            # on_change="rerun" every open and close becomes a full script rerun,
            # the same cost on_click="ignore" removes from the buttons above. A 90
            # second clip is ~3 chunks, where a rerun per toggle buys nothing. 100
            # rows is roughly an hour of long-form chunking, past which the table
            # stops being incidental and the trade inverts.
            lazy = len(result.segments) > 100
            segments = st.expander(
                f"{len(result.segments)} chunks",
                icon=":material/segment:",
                on_change="rerun" if lazy else "ignore",
            )
            if segments.open or not lazy:
                with segments:
                    st.caption(
                        "Chunk boundaries from long-form splitting, not word-level "
                        "alignment. The model does not produce timestamps or speaker "
                        "labels."
                    )
                    st.dataframe(
                        result.segments,
                        hide_index=True,
                        key="segments",
                        column_order=["start", "end", "text"],
                        column_config={
                            "start": st.column_config.NumberColumn(
                                "Start", format="%.1fs"
                            ),
                            "end": st.column_config.NumberColumn("End", format="%.1fs"),
                            "text": st.column_config.TextColumn("Text", width="large"),
                        },
                    )
