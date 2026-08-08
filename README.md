# cohere-transcribe

[![CI](https://github.com/darylalim/cohere-transcribe/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/darylalim/cohere-transcribe/actions/workflows/ci.yml)

Streamlit application for transcription using Cohere Transcribe on Apple Silicon with MLX.

Audio never leaves the machine — the model runs locally through
[mlx-audio](https://github.com/Blaizzy/mlx-audio).

## Setup

```bash
uv sync
```

Only wav, mp3 and flac decode through miniaudio. Everything else — aiff, m4a,
aac, ogg, opus, webm — needs `ffmpeg` (`brew install ffmpeg`).

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

Covers the pure functions — subtitle formatting, duration and timestamp
rendering, cue filtering, VAD segment capping, and the word-error-rate and
non-ASCII-ratio functions that decide whether the integration test below passes.
No model, no audio, no network; it runs in well under a second and is the only
check here that also runs on Linux. It still needs `uv sync` to succeed, so in
practice that means Apple Silicon or Linux — mlx publishes no Intel-macOS wheel.

The slow one needs weights:

```bash
uv run verify_transcription.py
```

Synthesizes speech with macOS `say`, so the reference text is exact, then checks
the transcript against it. Covers what `AppTest` cannot reach: every advertised
upload format decoding to real samples, long-form splitting past the 35-second
window, SRT/VTT export from real segments, and the VAD path. Current results on
an M-series Mac:

| Check | WER | Segments | RTFx |
| --- | --- | --- | --- |
| Short (7 s) | 0.0% | 1 | 32× |
| Long-form (42 s) | 0.9% | 2 | 35× |
| VAD (7 s) | 0.0% | 1 | 12× |

The 0.9% is a single comma. This exists because a broken checkpoint returns a
confident, non-empty, correctly-typed string — only a comparison against text we
wrote ourselves tells the two apart.

## Model

[Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
is a 2B-parameter dedicated ASR model — Apache 2.0, #1 on the Open ASR Leaderboard
at 5.42% average WER. A FastConformer encoder holds ~92% of the parameters and an
8-layer Transformer decoder handles the text, which is why decoding is fast.

### Use the upstream repo, not the community MLX conversions

mlx-audio's Python runtime remaps the upstream Hugging Face layout when it loads
the model. The `-mlx-` conversions of Cohere Transcribe on the Hub
(`beshkenadze/*`, `mlx-community/*`) are built for the Swift runtime and
mlx-speech, and use different parameter names — `decoder.core.*` and
`bridge_proj.*` where mlx-audio expects `transf_decoder.decoder.*` and
`encoder_decoder_proj.*`.

This matters more than it sounds. `mlx_audio.stt.load()` defaults to
`strict=False`, so those checkpoints load *without error*: the encoder matches, the
decoder silently stays randomly initialised, and the model transcribes fluent
multilingual gibberish. `utils/models.py` therefore loads with `strict=True` and
raises `ModelWeightsError` rather than letting that through.

## What the model does not do

These are model limits, not missing features in this app:

- **No language detection.** Pick one of the 14 supported languages explicitly.
  Code-switched audio transcribes poorly.
- **No timestamps or speaker diarization.** The SRT/VTT exports use long-form
  chunk boundaries (up to 35 seconds each), which is fine for navigation and too
  coarse for real subtitles.
- **It transcribes silence.** Like most attention encoder-decoder ASR models it
  will hallucinate text over background noise. The VAD toggle fixes this on audio
  with real silences, at the cost of accuracy on dense narration — hence off by
  default.

## Known upstream issues

**`vad=True` crashes in mlx-audio 0.4.7.** `Model._segment_with_vad` is numpy
code — its own return annotation is `List[np.ndarray]` — but `generate` hands it
the `mx.array` from `_to_mono`. It fails twice: `.astype(np.float32)` inside the
Silero backend, then `waveform[a:b].copy()` when slicing runs out. MLX arrays
have neither method. `utils/models._patch_vad_dtype` coerces once at that
boundary. Delete it once mlx-audio ships a fix.

**mlx-audio cannot identify AIFF or raw AAC.** Its format sniffer matches magic
bytes and has no branch for `FORM`/`AIFC` or ADTS `0xFFF1`, so both raise before
any decoder runs. `utils/audio._decode_with_ffmpeg` retries them through ffmpeg.

**mlx-audio pipes MP4 to ffmpeg on stdin**, which cannot seek. Unless the file
was written with `+faststart` its `moov` index sits at the end — the normal
layout — so ffmpeg reads nothing, exits 0, and mlx-audio returns an *empty
array* rather than raising. The fallback writes to a real file for this reason,
and `decode_to_mono16k` treats an empty result as a failure to retry.

## Layout

```
streamlit_app.py           UI
utils/audio.py             decoding to mono 16 kHz, SRT/VTT formatting
utils/models.py            checkpoint registry, language table, cached loader
verify_transcription.py    smoke test against known ground truth
tests/test_pure.py         unit tests for the pure functions — no model needed
```
