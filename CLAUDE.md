# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-page Streamlit app that runs [Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
— a 2B-parameter ASR model — locally on Apple Silicon through `mlx-audio`. Audio never leaves the
machine; the only network calls are Hugging Face weight downloads — the checkpoint, plus 1.2 MB of
Silero the first time VAD is switched on. Five Python files, no framework
beyond Streamlit.

## Commands

```bash
uv sync                                         # setup; needs Apple Silicon for MLX
uv run streamlit run streamlit_app.py           # run the app
uv run pytest                                   # fast unit tests, no model
uv run verify_transcription.py                  # integration test — see below
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

Two layers with a strict division of labour. Do not let either drift into the other's job.

**`verify_transcription.py` — the integration test.** Synthesizes speech with macOS `say` so the
reference text is exact, decodes every format in `UPLOAD_TYPES`, then runs short / long-form / VAD
transcriptions and fails on WER > 15% or > 5% non-ASCII characters. It takes no flags to select a
subset — comment out calls in `run()` for that. Needs macOS, ffmpeg, a valid HF token, and downloads
~4 GB on first run; a full pass takes minutes.

It is the *only* thing that can catch the bug it was written for — a decoder left randomly
initialised at load returns a confident, non-empty, correctly-typed string, so nothing short of a
comparison against text we authored ourselves distinguishes a working checkpoint from a broken one.
No unit test can replace it and none should try.

`check_vad_backend` is that same argument one level down, and it is why WER cannot be its oracle.
`mlx_audio.vad.load` is `strict=False`, so a VAD checkpoint whose keys stop matching the module names
loads into a randomly initialised model that calls the whole waveform speech — and since
`_segment_with_vad` *also* returns the whole waveform when it finds no speech, the transcript is the
one the non-VAD path already produced either way, leaving every WER and non-ASCII assertion green.
Measured: the real backend returns nothing on five seconds of silence, a random one returns 4.9 of
them. So the check pads the fixture with silence and asserts it gets trimmed, on top of asserting
`repo_id` and that the backend was consulted at all. It runs outside the `try` around the VAD
transcription, since a regression there is exactly when its answer is wanted.

**`tests/test_pure.py` — unit tests, no model.** Runs in under a second, on any platform `uv sync`
supports (Apple Silicon or Linux — mlx publishes no Intel-macOS wheel). Its core is the three things
the integration test structurally cannot reach:

- **Its own oracle.** `word_error_rate` and `non_ascii_ratio` are the entire pass/fail decision over
  there, and a bug in either fails *toward a false pass* — a WER that drifts low reads as a better
  transcript, so a broken checkpoint would go green. Nothing else checks them.
- **Branches `say` cannot reach.** `_cap_segment_length` fires only when Silero detects no speech,
  and `_cues` only on a malformed segment. Every fixture over there is wall-to-wall synthesized
  speech, so neither had ever executed in a test.
- **Subtitle structure.** `verify_transcription.py` asserts `"-->" in srt` and a WEBVTT prefix, which
  pass just as happily on indices starting at 0 or the two separators swapped. The unit tests assert
  both files byte for byte.

The rest is app-only code `verify_transcription.py` never touches at all — `format_duration`,
`_timestamp`, `Transcript.speedup` and `Transcript.stem`. That is about a third of the file, so do
not read the three categories above as a closed charter for what belongs here.

Both thresholds live in `verify_transcription.py` as `MAX_WER` and `MAX_NON_ASCII` so the unit tests
can import the real numbers. Re-typing them in the test would have let a loosened threshold pass a
test written specifically to catch a loosened threshold.

**Never add a test that mocks `model.generate`.** A stub returning a plausible string asserts that
the dataclass holds a string, which was never in doubt, and it is the exact shape of the false
confidence `verify_transcription.py` exists to defeat. Real decoding stays out too — `check_decoding`
already covers it against nine actual containers.

`[tool.pytest.ini_options] pythonpath = ["."]` is load-bearing, not boilerplate: with no
`[build-system]` the project is never installed, and pytest puts `tests/` on `sys.path` rather than
the repo root, so `from utils.audio import ...` raises `ModuleNotFoundError` without it. Prefer it to
adding `tests/__init__.py`, which achieves the same thing as an unexplained side effect of packaging.

Note that `_cues` handles `segments=None` via `for seg in segments or []` while its annotation says
`Iterable[dict]`, so ty rejects a test asserting on it. The guard is unreachable from every current
caller — `Transcript.segments` defaults to a list and `output.segments or []` coerces at
construction — so it is untested on purpose rather than by oversight.

### CI

`.github/workflows/ci.yml` adds no tests; it runs the ones already here. Four jobs: `lint` on ubuntu
(ruff only, and no `uv sync` — it needs none), `test` on ubuntu (`uv sync --locked` and pytest),
`check` on macos-15 (`uv sync --locked`, ty, and the decode matrix), and `integration`, which runs the
whole script against real weights and is `workflow_dispatch` only — gated weights plus GitHub
withholding secrets from fork pull requests mean it can never be a required check. It needs an
`HF_TOKEN` repository secret; `huggingface_hub` reads that env var directly, so CI needs no
`hf auth login`.

No step in it is verdict-neutral. That is the rule for adding one, and three were removed or rewired
under it: `lint` had its own `uv lock --check`, which asserts exactly what `--locked` already asserts
in `test` on the same triggers; `check` had a `say` probe that only reordered a failure arriving
thirty seconds later regardless — it survives in `integration`, where it precedes a ~4 GB download;
and ty ran without `--output-format=github`, so `continue-on-error` left its findings in the log of a
passing job. Diagnosis is worth a step only where it beats a materially worse alternative, which is
why the two survivors are the ones guarding a four-gigabyte wait and a truncated annotation.

`test` is on ubuntu because it is the one job that can be: `tests/test_pure.py` never imports
`mlx_audio` — `utils/` keeps those imports inside the functions that need them — so it needs no Apple
Silicon, ffmpeg, `say` or token, and Linux runners bill at a fraction of the macOS rate. It doubles
as the check that `uv sync --locked` really does resolve on Linux — a property `check`'s comment used
to wave away as not mattering, and which now has a job depending on it, so that comment was corrected
rather than left to contradict this one. `integration` gates on `lint` and `test` both: no sense
spending forty minutes and four gigabytes to discover that `word_error_rate`, which decides that
job's own verdict, is broken.

Two consequences worth knowing before editing anything:

- **`check_decoding` now has a caller outside this repo's own `run()`.** The workflow imports it
  directly to get the format matrix without a checkpoint. Its name, signature and
  `list[Failure]` return are load-bearing for CI, not just for `verify_transcription.py`.
- **The lint job is where repo-wide ruff lives now, not a mirror of a hook that also does it.** It
  reads the ruff version out of `required-version` exactly as `.claude/hooks/lib.sh` does, and every
  check runs under `if: !cancelled()` so a formatting slip cannot stop `ruff check` from running —
  the same short-circuit b06c493 removed from the `Stop` hook that has since been deleted for
  duplicating this job. ruff blocks and ty is advisory there for the same reasons it draws that line
  here.

## Hooks

`.claude/` carries three, and the rule for admitting a fourth is that the failure it catches must be
**silent**. Everything in `guard.sh` succeeds while being wrong: a `pip install` that moves mlx-audio
off the 0.4.7 `_patch_vad_dtype` targets, a credential read, four gigabytes downloaded before
anything says why. A rule rejecting an unpinned `uvx ruff` was removed under that test — ruff reads
`required-version` itself and aborts with the whole diagnosis in the error, so the hook bought one
round trip.

- **`guard.sh`** (PreToolUse) — `deny` where the operation is always wrong here, `ask` where a human
  should decide. The reason strings are the point; they reach Claude verbatim.
- **`edit-checks.sh`** (PostToolUse) — ruff format applied, ruff check and ty reported. It is the
  only in-session checker. `exit 2` is advisory here and blocking on `Stop`, which is why ty sits in
  this hook and not in a gate.
- **`preflight.sh`** (SessionStart) — ffmpeg, Apple Silicon, HF credentials, `uv.lock`. Prints
  nothing when all four hold, so it costs a line of output only when it has one to give.

A `Stop` gate running repo-wide ruff was deleted rather than kept: ruff's rules are per-file, so a
repo sweep found nothing `edit-checks.sh` had not already reported on the file Claude wrote, and the
`lint` job above covers the rest on push. What it uniquely caught — a `.py` file edited through
`sed` or a heredoc rather than Edit/Write — is real but rare, and reaches CI anyway. Half its body
was a session-keyed loop guard defending against its own ability to trap a turn.

## Architecture

```
streamlit_app.py           UI, session state, error presentation
utils/audio.py             decode to mono 16 kHz float32; SRT/VTT formatting
utils/models.py            checkpoint registry, language table, cached loader, mlx-audio VAD shim
verify_transcription.py    integration test against known ground truth
tests/test_pure.py         unit tests for the pure functions, and for the test oracle above
```

Flow: `UploadedFile` (or `st.audio_input`) → `decode_to_mono16k` → flat `np.float32` array at 16 kHz +
duration → `load_asr(repo_id)` (cached, one model resident) → `model.generate(...)` → mlx-audio
`STTOutput` → `Transcript` dataclass parked in `st.session_state.result` → metrics, text, SRT/VTT
downloads, chunk table. Session state holds one other key: `digest`, a `(file_id, digest)` pair for the
upload on screen, which is what decides whether a new upload invalidates `result` — see the
load-bearing decisions below.

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
- **`_pin_vad_repo` points VAD at `VAD_REPO` by writing a private attribute.** Seeding
  `model._vad_backend` is the only opening: `_segment_with_vad` builds the backend once per instance
  through `get_backend()`, which hardcodes `mlx-community/silero-vad` (the v5 port) and takes no repo,
  and `generate` exposes nothing that reaches it. Three constraints shape it. It runs from inside the
  `_patch_vad_dtype` wrapper rather than from `load_asr`, so its private import can only take down the
  VAD path — at load time an upstream rename would have broken every transcription, including for
  people who never switch VAD on. It goes *through* `get_backend` instead of constructing
  `SileroMlxBackend`, so an unknown `vad=` selector still raises there. And it is idempotent on
  `repo_id`, because the wrapper runs on every call and rebuilding would discard weights the backend
  already loaded. v6 ships no 8 kHz branch and `mlx_audio.vad.load` is `strict=False`, so
  `Model.vad_8k` loads randomly initialised — the exact failure `load_asr` refuses next door, inert
  here only because `SileroMlxBackend` fixes `sample_rate` at 16000 and `_branch()` never returns the
  8 kHz one. Do not reuse `VAD_REPO` where 8 kHz audio can reach a detector.
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
- **`st.audio` is passed an explicit `format=`, derived from the extension by `preview_mime`.** It
  defaults to `"audio/wav"` and nothing sniffs the container, so the bytes are served under that
  Content-Type. Eight of the nine `UPLOAD_TYPES` are not WAV, and browsers that pick a decoder from
  the header rather than the magic bytes play none of them — silently, since the file still
  transcribes fine and only the preview player looks broken. `UploadedFile.type` looks like the
  obvious source and is not: it is whatever the browser wrote into the multipart part, which is empty
  for `.opus` on most systems, and Streamlit stores an empty one as `"application/octet-stream"` —
  never falsy, so an `or "audio/wav"` fallback never fires and the player stays broken at a `.bin`
  URL. The extension is the part the uploader has already validated, and keying off it also keeps a
  browser-supplied string out of a Content-Type served from the app's own origin, which the media
  route sends without `nosniff`.
- **Session state is invalidated only when a new source exists, and "same source" means the same
  bytes.** Clearing the uploader or switching to Record must not discard a finished transcript; that
  can be minutes of unrecoverable work. Identity is a `blake2b` digest of the upload rather than
  `UploadedFile.file_id`, because Streamlit mints a fresh uuid4 per upload *event* — so re-dropping
  the very file that produced the transcript on screen read as a new source and threw it away. Both
  inputs reach that: clearing the uploader keeps the transcript but takes the player away, and
  switching to Record unmounts the uploader and lets its widget state be pruned, leaving a re-upload
  as the only way back to either. `getvalue()`, not `getbuffer()`: `UploadedFile` is a `BytesIO` built
  around the upload record's bytes, so `getvalue()` returns that object under CPython's copy-on-write
  rule while `getbuffer()` must unshare the buffer and memcpy it — measured +0 MB against +300 MB of
  RSS on a 300 MB payload, which at the 1000 MB ceiling is a spare gigabyte. It sits behind
  `source_key()` rather than inline because the digest is wanted only twice — to compare against a
  transcript that already exists, and to label a new one — and neither holds on the first upload of a
  session, which is every session; hashing a gigabyte to protect a transcript that is not there is
  dead time before the user has clicked anything. Cached as a pair, not a one-entry dict, so no
  eviction step carries its own necessity in a comment. When the digest *matches*, `source_name` is
  refreshed from the current upload: same bytes under a new filename is a copy or a rename, and the
  caption and download stems are built from that field, which would otherwise keep naming a file the
  player is no longer showing.
- **The status label reports total wall clock, the Elapsed metric reports generation only.** RTFx needs
  generation time; the status would otherwise read "Done in 2.1s" after a multi-minute first-run
  download. Note that the status block is written under `if run and ...`, so it is absent from every
  rerun that did not press Transcribe — that is why the metrics below render from `result` instead.
- **The transcript renders through `st.text`, not `st.markdown`.** It is uncontrolled model output and
  the product is a verbatim transcript: a hallucinated `*music*` renders italic with the asterisks
  gone, a leading `- ` becomes a bullet, `$5-$10` renders as math — while the Text download hands over
  the unparsed string, so the file and the screen stop being the same characters. `st.text` is not
  monospace (that is `st.code`), so nothing about the look changes. The `_No speech detected._`
  placeholder stays on `st.markdown`, since that string *is* ours to format. `Transcript` holds
  `output.text.strip()` rather than the raw string, because `st.text` runs its body through
  `textwrap.dedent().strip()` — leave that to the renderer and a decoder's leading space shows trimmed
  on screen while the Text download writes it, which is the same mismatch one layer down. Stripping at
  construction is also what keeps a whitespace-only result falsy, so it takes the no-speech branch
  instead of rendering an empty box beside a live download button.
- **`_cues()` filters on `start`/`end`/`text` before the SRT/VTT writers index them** — a malformed
  segment would otherwise raise while rendering an already-successful result, taking the transcript
  off screen.
- **The three download buttons carry `on_click="ignore"`, and Text is disabled on empty text.** Every
  payload comes from `st.session_state.result`, so the default `"rerun"` re-executes the whole script
  to arrive at an identical screen — rebuilding `srt`/`vtt`, re-marshalling all three payloads (which
  happens before `disabled` is applied to the proto, so the greyed-out ones pay too) and
  re-serialising the chunk table. It is the one interaction here that cannot change a pixel.
  `disabled=not result.text` matches the rule SRT and VTT already follow: a no-speech result still has
  segments, so without it the one button left lit hands over an empty file.
- **The chunk expander goes lazy only past 100 rows.** `st.expander` computes and ships its contents
  whether or not it is open, and `.open` means nothing until `on_change` is set — so above the
  threshold it sets `on_change="rerun"` and gates its body on `segments.open`, and that guard wrapping
  a `with segments:` block looks redundant and is not. The threshold is the point of the decision, not
  a tuning knob: `on_change="rerun"` turns every open and close into a full script rerun, the same
  cost `on_click="ignore"` removes from the buttons just above, so on a 90-second clip's ~3 chunks the
  lazy path costs more than the Arrow serialisation it avoids. 100 rows is roughly an hour of
  long-form chunking; past it VAD's per-speech-run splitting reaches thousands and the trade inverts.
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
- Adding an upload format means a row in `UPLOAD_TYPES`, a row in `PREVIEW_MIME`, and a passing row in
  `check_decoding` — that loop reports a missing format as a failure rather than skipping it, because
  a skipped row reads as a passing one. `preview_mime` falls back to `"audio/wav"` instead of raising,
  so a missing row there degrades silently into the exact bug it exists to prevent; `test_pure.py`
  asserts the two lists stay in step, in both directions.
- The UI uses recent Streamlit APIs deliberately (`st.segmented_control`, `st.container(horizontal=…)`,
  `width="stretch"`, `icon=` on metrics and expanders, `on_click="ignore"` on download buttons,
  `on_change="rerun"` plus `.open` on the chunk expander). Don't substitute older equivalents. The
  floor in `pyproject.toml` is a checked claim rather than an assumed one: every `st.*` call the app
  makes was resolved against clean 1.57 through 1.61 installs, and `icon=` on `st.metric` is the one
  that moves it — it does not exist before **1.61**, while everything else here, `.open` gating
  included, resolves as far back as 1.57. The floor read `>=1.57` for exactly that reason before
  anyone checked: `uv.lock` pins 1.61.1, so `uv sync --locked` and all of CI install a version that
  satisfies any floor, and an undershooting one breaks only whoever resolves from `pyproject.toml`
  alone. Re-check it the same way when adding an API, rather than inferring it from a changelog.
