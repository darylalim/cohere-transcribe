#!/usr/bin/env bash
# SessionStart: surface the environmental preconditions this app fails on, so
# they are known before a transcription stalls rather than during one. Three of
# this project's failure modes are environmental, not code: gated weights with
# no HF login, a missing ffmpeg (which silently narrows the accepted formats to
# wav/mp3/flac), and a non-Apple-Silicon machine.
#
# Runs async and always exits 0. On SessionStart, stdout is what reaches Claude.
set -uo pipefail

notes=()

command -v ffmpeg >/dev/null \
  || notes+=("ffmpeg is not on PATH -- of the formats in UPLOAD_TYPES only wav, mp3 and flac will decode. 'brew install ffmpeg'.")

[[ "$(uname -m)" == "arm64" ]] \
  || notes+=("Not Apple Silicon -- MLX cannot run here.")

uv run --project "${CLAUDE_PROJECT_DIR:-.}" hf auth whoami >/dev/null 2>&1 \
  || notes+=("No Hugging Face login -- the gated checkpoint will fail to download. Accept the terms on the Hub, then 'uv run hf auth login'.")

((${#notes[@]})) && printf '%s\n' "${notes[@]}"
exit 0
