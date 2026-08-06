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

mode = st.segmented_control(
    "Input",
    ["Upload a file", "Record"],
    default="Upload a file",
    label_visibility="collapsed",
)

if mode == "Record":
    audio_file = st.audio_input("Recording")
else:
    audio_file = st.file_uploader("Audio file", type=UPLOAD_TYPES)
    if audio_file is not None:
        st.audio(audio_file)

source_key = getattr(audio_file, "file_id", None) or getattr(audio_file, "name", None)

# Drop a stale transcript as soon as the source changes, so the text on screen
# always belongs to the audio on screen.
if result and result.source_key != str(source_key):
    result = None
    st.session_state.result = None

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
                source_key=str(source_key),
                source_name=getattr(audio_file, "name", "recording"),
                text=output.text,
                segments=output.segments or [],
                language=language,
                duration_s=duration_s,
                elapsed_s=elapsed,
            )
            st.session_state.result = result
            status.update(
                label=f"Transcribed in {elapsed:.1f}s", state="complete", expanded=False
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
            st.markdown(result.text or "_No speech detected._")

        with st.container(horizontal=True):
            st.download_button(
                "Text",
                result.text,
                file_name=f"{result.stem}.txt",
                icon=":material/description:",
            )
            st.download_button(
                "SRT",
                to_srt(result.segments),
                file_name=f"{result.stem}.srt",
                icon=":material/subtitles:",
                disabled=not result.segments,
            )
            st.download_button(
                "VTT",
                to_vtt(result.segments),
                file_name=f"{result.stem}.vtt",
                icon=":material/subtitles:",
                disabled=not result.segments,
            )

        if len(result.segments) > 1:
            segments = st.expander(
                f"{len(result.segments)} chunks", icon=":material/segment:"
            )
            with segments:
                st.caption(
                    "Chunk boundaries from long-form splitting, not word-level "
                    "alignment. The model does not produce timestamps or speaker labels."
                )
                st.dataframe(
                    result.segments,
                    hide_index=True,
                    key="segments",
                    column_order=["start", "end", "text"],
                    column_config={
                        "start": st.column_config.NumberColumn("Start", format="%.1fs"),
                        "end": st.column_config.NumberColumn("End", format="%.1fs"),
                        "text": st.column_config.TextColumn("Text", width="large"),
                    },
                )
