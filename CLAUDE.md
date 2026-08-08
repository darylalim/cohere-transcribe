# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-page Streamlit app that runs [Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
— a 2B-parameter ASR model — locally on Apple Silicon through `mlx-audio`. Audio never leaves the
machine; the only network call is the Hugging Face weight download. Four Python files, no framework
beyond Streamlit.

## Commands

```bash
uv sync                                         # setup; needs Apple Silicon for MLX
uv run streamlit run streamlit_app.py           # run the app
uv run verify_transcription.py                  # the only test — see below
uvx ruff@0.16.1 check . && uvx ruff@0.16.1 format .
uvx ty check .
```

When working with Python, invoke the relevant `/astral:<skill>` — `/astral:uv`, `/astral:ruff`,
`/astral:ty` — to ensure best practices are followed.

Use `uv sync`, never `uv pip install -e .` — the latter re-resolves from `pyproject.toml` and ignores
the committed `uv.lock` entirely. That distinction is load-bearing here: `_patch_vad_dtype` wraps a
*private* mlx-audio method against 0.4.7 internals while `pyproject.toml` only asks for `>=0.4.4`, so
the lockfile is the only thing pinning the version that patch was written against. `pyproject.toml`
has no `[build-system]` on purpose — uv then treats the project as non-packaged and installs just the
dependencies, which is what the app wants, since it runs as scripts from the repo root and imports
`utils` relative to cwd.

ruff and ty are not project dependencies — run them through `uvx`. Both currently pass clean, as does
`ruff format --check`. `required-version` in `pyproject.toml` pins ruff to 0.16.1, so an unpinned
`uvx ruff` or an editor's bundled copy refuses to run rather than linting under a different default
rule set — the version is what decides which rules exist, and this project has no `select` of its own.

ty is deliberately *not* pinned the same way: it is pre-1.0 and ships near-daily, so a pin would go
stale within weeks, and it has no `required-version` setting to hold one anyway. Clean as of ty
0.0.69. A new release surfacing new diagnostics is expected rather than a regression. The two
`ty: ignore` directives in `utils/models.py` need no policing — `unused-ignore-comment` is on by
default, so ty reports them itself once its inference no longer needs them. Note also that ty checks
against **Python 3.10**, inferred from `requires-python`, not the 3.12 in `.python-version`: that is
what verifies the floor the package claims, and it is why `enum.StrEnum` degrades to `Unknown` in
`--verbose` output. Do not "fix" that by pinning `[tool.ty.environment] python-version` to 3.12.

The weights are gated: accept the terms on the Hub, then `uv run hf auth login`. Only wav, mp3 and
flac decode through miniaudio; every other advertised format needs `ffmpeg` on PATH.

### Testing

There is no pytest suite. `verify_transcription.py` is the whole test story, and it is an integration
test: it synthesizes speech with macOS `say` so the reference text is exact, decodes every format in
`UPLOAD_TYPES`, then runs short / long-form / VAD transcriptions and fails on WER > 15% or > 5%
non-ASCII characters. It takes no flags to select a subset — comment out calls in `run()` for that.
Needs macOS, ffmpeg, a valid HF token, and downloads ~4 GB on first run; a full pass takes minutes.

It exists *instead of* unit tests because the bug it guards against — a decoder left randomly
initialised at load — returns a confident, non-empty, correctly-typed string. Only a comparison
against text we authored ourselves distinguishes a working checkpoint from a broken one.

### CI

`.github/workflows/ci.yml` adds no tests; it runs the ones already here. Three jobs: `lint` on
ubuntu (ruff, and `uv lock --check`), `check` on macos-15 (`uv sync --locked`, ty, and the decode
matrix), and `integration`, which runs the whole script against real weights and is
`workflow_dispatch` only — gated weights plus GitHub withholding secrets from fork pull requests
mean it can never be a required check. It needs an `HF_TOKEN` repository secret; `huggingface_hub`
reads that env var directly, so CI needs no `hf auth login`.

Two consequences worth knowing before editing anything:

- **`check_decoding` now has a caller outside this repo's own `run()`.** The workflow imports it
  directly to get the format matrix without a checkpoint. Its name, signature and
  `list[Failure]` return are load-bearing for CI, not just for `verify_transcription.py`.
- **The lint job mirrors the hooks rather than restating them.** It reads the ruff version out of
  `required-version` exactly as `.claude/hooks/lib.sh` does, and every check runs under
  `if: !cancelled()` so a formatting slip cannot stop `ruff check` from running — the same
  short-circuit b06c493 removed from `gate.sh`. ruff blocks and ty is advisory there for the same
  reasons it draws that line here.

## Architecture

```
streamlit_app.py           UI, session state, error presentation
utils/audio.py             decode to mono 16 kHz float32; SRT/VTT formatting
utils/models.py            checkpoint registry, language table, cached loader, mlx-audio VAD shim
verify_transcription.py    integration test against known ground truth
```

Flow: `UploadedFile` (or `st.audio_input`) → `decode_to_mono16k` → flat `np.float32` array at 16 kHz +
duration → `load_asr(repo_id)` (cached, one model resident) → `model.generate(...)` → mlx-audio
`STTOutput` → `Transcript` dataclass parked in `st.session_state.result` → metrics, text, SRT/VTT
downloads, chunk table.

Everything downstream of decoding assumes mono float32 at `audio.SAMPLE_RATE`; note that
`streamlit_app.py` passes `sample_rate=16_000` as a literal to `generate`, so changing `SAMPLE_RATE`
alone would desync the two.

Heavy imports (`mlx_audio.stt`, `mlx_audio.audio_io`) live inside the functions that need them, so
the sidebar renders before anything expensive happens.

## Load-bearing decisions — do not "clean up"

Each of these looks like a mistake and is not. Comments in the source carry the full reasoning.

- **`load_asr` loads with `strict=True`.** mlx-audio defaults to `strict=False`, which silently leaves
  unmatched modules randomly initialised. The community `-mlx-` conversions on the Hub
  (`beshkenadze/*`, `mlx-community/*`) target the Swift runtime and use different parameter names, so
  they load *without error* and transcribe fluent multilingual nonsense. The substring list
  (`"not in model"`, `"missing"`, `"expected shape"`, `"expected mx."`) covers the four messages
  `mlx.nn.Module.load_weights(strict=True)` can raise.
- **Gated-repo errors are matched by exception type**, never by `"401"` in the message — a shard named
  `model-00401-of-00500` was enough to route disk-full errors to "run `hf auth login`".
- **`_patch_vad_dtype()` monkeypatches the private `Model._segment_with_vad`** because that function is
  numpy code throughout while `generate` hands it an `mx.array` (mlx-audio 0.4.7). It runs on every
  load, is idempotent via the `_coerces_numpy` flag, and no-ops if the method disappears upstream. Its
  wrapper also calls `_cap_segment_length`, which is *not* part of the upstream bug: when Silero
  detects no speech it returns the whole waveform as one chunk and the 35-second window never applies,
  so an hour of room tone would reach the encoder as a single ~57M-sample array. Keep the capping even
  if the numpy coercion becomes unnecessary.
- **The ffmpeg fallback writes a real temp file and requests raw `s16le`.** Never pipe, in either
  direction: on stdin ffmpeg cannot seek, so an MP4 whose `moov` index sits at the end — the normal
  layout — decodes to nothing and exits 0; on stdout it cannot backfill the RIFF size field. Both
  failures are silent.
- **An empty waveform is a failure, not silence.** `decode_to_mono16k` retries through ffmpeg on
  `size == 0` and only then raises. The broad `except Exception` around `audio_read` is deliberate:
  mlx-audio raises `ValueError`, `miniaudio.DecodeError` and `RuntimeError` for the same condition.
- **Imports sit below `st.set_page_config` in `streamlit_app.py`** (it must be the first Streamlit
  call) and below `warnings.filterwarnings` in `verify_transcription.py`. E402 is not in ruff's default
  rule set *and* `[tool.ruff.lint] extend-ignore` names it explicitly, so this passes lint — do not
  reorder.
- **Session state is invalidated only when a new source exists.** Clearing the uploader or switching to
  Record must not discard a finished transcript; that can be minutes of unrecoverable work.
- **The status label reports total wall clock, the Elapsed metric reports generation only.** RTFx needs
  generation time; the status would otherwise read "Done in 2.1s" after a multi-minute first-run
  download.
- **`_cues()` filters on `start`/`end`/`text` before the SRT/VTT writers index them** — a malformed
  segment would otherwise raise while rendering an already-successful result, taking the transcript
  off screen.
- **`st.cache_resource(max_entries=1)`** keeps exactly one multi-gigabyte model resident, so pointing
  the app at another repo evicts rather than accumulates.

## Model limits — not missing features

- **No language detection.** One of the 14 codes in `LANGUAGES` must be chosen explicitly;
  code-switched audio transcribes poorly.
- **No timestamps or diarization.** SRT/VTT cues are long-form chunk boundaries (up to 35 s) — fine for
  navigation, too coarse for real subtitles.
- **It transcribes silence.** Like most attention encoder-decoder ASR models it hallucinates over
  background noise. The VAD toggle fixes this on audio with real silences at the cost of accuracy on
  dense narration, hence off by default.

## Conventions

- Comments explain *why*, and usually name the failure that motivated the code. Match that register;
  drop a comment only if the failure it describes is genuinely gone.
- Adding an upload format means a row in `UPLOAD_TYPES` and a passing row in `check_decoding` — that
  loop reports a missing format as a failure rather than skipping it, because a skipped row reads as a
  passing one.
- The UI uses Streamlit ≥1.57 APIs deliberately (`st.segmented_control`, `st.container(horizontal=…)`,
  `width="stretch"`, `icon=` on metrics and expanders). Don't substitute older equivalents.
