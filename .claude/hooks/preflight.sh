#!/usr/bin/env bash
# SessionStart: name the environmental preconditions this app fails on, before a
# transcription stalls on one. Two of this project's failure modes are
# environmental rather than code: a missing ffmpeg (which silently narrows
# UPLOAD_TYPES to wav/mp3/flac) and a machine MLX cannot run on.
#
# Runs synchronously and prints plain text, which is what reaches Claude on
# SessionStart. It was async, which delivers stdout through a JSON payload path
# that discards bare prose -- the whole hook was a no-op.
#
# Every probe below is local, and must stay that way. An earlier version shelled
# out to `uv run hf auth whoami`, which (a) round-trips to huggingface.co, so it
# cannot tell "not logged in" from "offline", and (b) re-locks and re-syncs by
# default, meaning merely opening a session could rewrite the uv.lock that
# guard.sh gates behind a human prompt.
#
# The local HF-credential probe that replaced it is gone too, for the other
# reason a probe leaves: a missing token is not silent. load_asr matches
# GatedRepoError by type -- never by "401" in the message -- and raises
# ModelAccessError, whose text is three numbered steps ending in `hf auth login`,
# put on screen by streamlit_app.py at the moment it matters, to the person who
# can act on it. Restating that at session start bought a turn at most. It also
# carried the contradiction: its remediation string said `uv run hf auth login`,
# reintroducing the exact relock hazard the paragraph above gives as its own
# reason for existing. utils/models.py says `hf auth login` without the prefix,
# and is the one that was right.
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-.}"
notes=()

command -v ffmpeg >/dev/null \
  || notes+=("ffmpeg is not on PATH -- of the formats in UPLOAD_TYPES only wav, mp3 and flac will decode. 'brew install ffmpeg'.")

# hw.optional.arm64 reports the hardware. uname -m reports the *process*, so it
# says x86_64 for any shell launched under Rosetta on an M-series Mac -- and
# telling Claude "MLX cannot run here" on a machine where it runs fine is how
# working code gets rewritten on a false premise.
if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(sysctl -n hw.optional.arm64 2>/dev/null)" != "1" ]]; then
  notes+=("Not Apple Silicon -- MLX cannot run here.")
fi

[[ -f "$root/uv.lock" ]] || notes+=("uv.lock is missing -- 'uv sync' will resolve fresh and may not pin the mlx-audio 0.5.1 that _patch_vad_dtype targets.")

((${#notes[@]})) && printf '%s\n' "${notes[@]}"
exit 0
