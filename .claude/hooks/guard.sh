#!/usr/bin/env bash
# PreToolUse: the operations in this repo that fail quietly when done wrong.
#
# `deny` for the ones that are always wrong here -- the reason string reaches
# Claude, which then self-corrects. `ask` for the ones that are sometimes right,
# so the decision reaches a human instead.
set -uo pipefail

payload=$(cat)
tool=$(jq -r '.tool_name // empty' <<<"$payload")

decide() { # $1 = allow|deny|ask|defer, $2 = reason shown to Claude
  jq -n --arg d "$1" --arg r "$2" '{hookSpecificOutput:{
    hookEventName:"PreToolUse", permissionDecision:$d, permissionDecisionReason:$r}}'
  exit 0
}

case "$tool" in
  Read|Edit|Write|NotebookEdit)
    f=$(jq -r '.tool_input.file_path // .tool_input.notebook_path // empty' <<<"$payload")

    case "$f" in
      */secrets.toml|*/huggingface/token|*/.huggingface/token)
        decide deny "$f holds credentials. huggingface_hub reads the token from the environment; nothing in this project needs the file opened." ;;
    esac

    case "$tool/$f" in
      Edit/*uv.lock|Write/*uv.lock|NotebookEdit/*uv.lock)
        decide ask "uv.lock is the only thing pinning mlx-audio to the 0.4.7 internals that _patch_vad_dtype in utils/models.py was written against -- pyproject.toml asks merely for >=0.4.4. Change pyproject.toml and run 'uv lock' rather than hand-editing the lockfile." ;;
    esac
    ;;

  Bash)
    cmd=$(jq -r '.tool_input.command // empty' <<<"$payload")

    if grep -Eq '(^|[;&|[:space:]])uv[[:space:]]+pip[[:space:]]+install' <<<"$cmd"; then
      decide deny "'uv pip install' re-resolves from pyproject.toml and ignores the committed uv.lock, which is what pins mlx-audio 0.4.7. Use 'uv sync'."
    fi

    if grep -Eq '(^|[;&|[:space:]])uvx[[:space:]]+ruff([[:space:]]|@)' <<<"$cmd" \
       && ! grep -q 'ruff@0\.16\.1' <<<"$cmd"; then
      decide deny "Unpinned 'uvx ruff' aborts here: pyproject sets required-version = '==0.16.1' and upstream has already moved past it. Use 'uvx ruff@0.16.1'."
    fi

    if grep -Eq 'verify_transcription\.py' <<<"$cmd"; then
      decide ask "verify_transcription.py is the full integration test: it needs macOS 'say', ffmpeg and a valid HF token, downloads ~4 GB on a cold cache, and a full pass takes minutes. It takes no flags to select a subset -- comment out calls in run() for that."
    fi
    ;;
esac

exit 0
