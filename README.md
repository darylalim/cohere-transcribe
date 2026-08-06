# cohere-transcribe

Streamlit application for transcription using Cohere Transcribe on Apple Silicon with MLX.

Audio never leaves the machine — the model runs locally through
[mlx-audio](https://github.com/Blaizzy/mlx-audio).

## Setup

```bash
uv venv --python 3.12
uv pip install -e .
```

Only wav, mp3 and flac decode through miniaudio. Everything else — aiff, m4a,
aac, ogg, opus, webm — needs `ffmpeg` (`brew install ffmpeg`).

The weights are gated. Accept the terms at
[CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026),
then authenticate:

```bash
hf auth login
```

## Run

```bash
streamlit run streamlit_app.py
```

The first transcription downloads ~4 GB of weights into `~/.cache/huggingface`.

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

## Known upstream issue

mlx-audio 0.4.7 crashes on `vad=True` for this model: `cohere_asr/vad.py` hands an
`mx.array` to a Silero backend that calls `.astype(np.float32)`, which MLX rejects.
`utils/models._patch_vad_dtype` coerces the input to numpy. Delete it once
mlx-audio ships a fix.

## Layout

```
streamlit_app.py     UI
utils/audio.py       decoding to mono 16 kHz, SRT/VTT formatting
utils/models.py      checkpoint registry, language table, cached loader
```
