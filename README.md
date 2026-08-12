# Cohere Transcribe

[![CI](https://github.com/darylalim/cohere-transcribe/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/darylalim/cohere-transcribe/actions/workflows/ci.yml)

Streamlit app for local speech to text on Apple Silicon. Audio never leaves the
machine: the model runs on the GPU through MLX and
[mlx-audio](https://github.com/Blaizzy/mlx-audio), and the only network calls
are the one-time weight downloads — the checkpoint, plus 1.2 MB of Silero the
first time VAD is switched on.

Upload a file or record from the mic, pick one of 14 languages, and get the text
back with SRT and VTT exports. Toggles control punctuation and casing, and
whether silence is trimmed with VAD.

## Setup

```bash
uv sync
```

Only wav, mp3 and flac decode through miniaudio. The other six advertised
formats — aiff, m4a, aac, ogg, opus, webm — need `ffmpeg`:

```bash
brew install ffmpeg
```

The weights are gated. Accept the terms at
[CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026),
then authenticate:

```bash
uv run hf auth login
```

## Run

```bash
uv run streamlit run streamlit_app.py
```

The first transcription downloads ~4 GB of weights into `~/.cache/huggingface`.

## Verify

Two layers. The fast one needs nothing installed beyond `uv sync`:

```bash
uv run pytest
```

It covers the pure functions — subtitle formatting, duration and timestamp
rendering, cue filtering, VAD segment capping — and the word-error-rate and
non-ASCII-ratio functions that decide whether the slow layer below passes. No
model, no audio, no network; it runs in well under a second. It is also the only
check here that runs on Linux, though `uv sync` still has to succeed first — so
in practice that means Apple Silicon or Linux, as mlx publishes no Intel-macOS
wheel.

The slow one needs weights:

```bash
uv run verify_transcription.py
```

It synthesizes speech with macOS `say`, so the reference text is exact, then
checks the transcript against it. That reaches what the unit tests structurally
cannot: every advertised upload format decoding to real samples, long-form
splitting past the 35-second window, SRT/VTT export from real segments, and the
VAD path — down to whether the pinned Silero checkpoint trims padded silence or
passes it straight through, which no WER threshold can see. Current results on
an M-series Mac:

| Check | WER | Segments | RTFx |
| --- | --- | --- | --- |
| Short (7 s) | 0.0% | 1 | 32× |
| Long-form (42 s) | 0.9% | 2 | 35× |
| VAD (7 s) | 0.0% | 1 | 12× |

The 0.9% is a single comma. This layer exists because a broken checkpoint
returns a confident, non-empty, correctly-typed string — only a comparison
against text we wrote ourselves tells the two apart.

Lint and type checks are not project dependencies, so run them through `uvx`:

```bash
uvx ruff@0.16.1 check . && uvx ruff@0.16.1 format .
uvx ty check .
```

The ruff version has to match `required-version` in `pyproject.toml`; an
unpinned `uvx ruff` refuses to run rather than linting under a different default
rule set. ty is deliberately unpinned — it is pre-1.0 and ships near-daily — and
is advisory in CI for the same reason.

## Model

[Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
is a 2B-parameter dedicated ASR model — Apache 2.0, #1 on the Open ASR
Leaderboard at 5.42% average WER. A FastConformer encoder holds ~92% of the
parameters and an 8-layer Transformer decoder handles the text, which is why
decoding is fast.

### Use the upstream repo, not the community MLX conversions

mlx-audio's Python runtime remaps the upstream Hugging Face layout when it loads
the model. The `-mlx-` conversions of Cohere Transcribe on the Hub
(`beshkenadze/*`, `mlx-community/*`) are built for the Swift runtime and
mlx-speech, and use different parameter names — `decoder.core.*` and
`bridge_proj.*` where mlx-audio expects `transf_decoder.decoder.*` and
`encoder_decoder_proj.*`.

This matters more than it sounds. `mlx_audio.stt.load()` defaults to
`strict=False`, so those checkpoints load *without error*: the encoder matches,
the decoder silently stays randomly initialised, and the model transcribes
fluent multilingual gibberish. `utils/models.py` therefore loads with
`strict=True` and raises `ModelWeightsError` rather than letting that through.

### What the model does not do

These are model limits, not missing features in this app:

- **No language detection.** Pick one of the 14 supported languages explicitly;
  code-switched audio transcribes poorly.
- **No timestamps or speaker diarization.** The SRT/VTT exports use long-form
  chunk boundaries of up to 35 seconds each, which is fine for navigation and
  too coarse for real subtitles.
- **It transcribes silence.** Like most attention encoder-decoder ASR models it
  will hallucinate text over background noise. The VAD toggle fixes this on
  audio with real silences, at the cost of accuracy on dense narration — hence
  off by default.

## Known upstream issues

**`vad=True` crashes in mlx-audio 0.4.7.** `Model._segment_with_vad` is numpy
code — its own return annotation is `List[np.ndarray]` — but `generate` hands it
the `mx.array` from `_to_mono`. It fails twice: `.astype(np.float32)` inside the
Silero backend, then `waveform[a:b].copy()` when slicing runs out. MLX arrays
have neither method. `utils/models._patch_vad_dtype` coerces once at that
boundary. Delete it once mlx-audio ships a fix.

**mlx-audio's VAD repo is fixed at the v5 Silero port.** `get_backend()` in
`cohere_asr/vad.py` hardcodes `mlx-community/silero-vad`, and `generate` takes
no argument that reaches it. `utils/models._pin_vad_repo` seeds that per-model
backend cache with
[`mlx-community/silero-vad-v6`](https://huggingface.co/mlx-community/silero-vad-v6),
the current Silero release line, from inside the VAD shim and so only when VAD
is actually used — an upstream rename can take down that path but not every
transcription. This
tracks upstream rather than buying accuracy — the v6 model card measures the two
within 0.4% F1 of each other on a 44-minute English meeting, v5 marginally
ahead, which is inside the noise of a single sample. v6 ships the 16 kHz branch
only; that is all this app can reach, since the backend fixes its sample rate at
16 kHz.

**mlx-audio cannot identify AIFF or raw AAC.** Its format sniffer matches magic
bytes and has no branch for `FORM`/`AIFC` or ADTS `0xFFF1`, so both raise before
any decoder runs. `utils/audio._decode_with_ffmpeg` retries them through ffmpeg.

**mlx-audio pipes MP4 to ffmpeg on stdin**, which cannot seek. Unless the file
was written with `+faststart` its `moov` index sits at the end — the normal
layout — so ffmpeg reads nothing, exits 0, and mlx-audio returns an *empty
array* rather than raising. The fallback writes to a real file for this reason,
and `decode_to_mono16k` treats an empty result as a failure to retry.

## Layout

```text
streamlit_app.py           UI, session state, error presentation
utils/audio.py             decode to mono 16 kHz float32; SRT/VTT formatting
utils/models.py            checkpoint registry, language table, cached loader,
                           mlx-audio VAD shim
verify_transcription.py    integration test against known ground truth
tests/test_pure.py         unit tests for the pure functions — no model needed
```

## License

[MIT](LICENSE), covering the code in this repository.

The weights are not part of it and carry their own terms: Cohere Transcribe
03-2026 is Apache 2.0, but gated, so access is granted by accepting the terms on
the [model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
rather than by this license.
