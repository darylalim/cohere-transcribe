#!/usr/bin/env bash
# PreToolUse: the operations in this repo that fail quietly when done wrong.
#
# `deny` for the ones that are always wrong here -- the reason string reaches
# Claude, which then self-corrects. `ask` for the ones that are sometimes right,
# so the decision reaches a human instead.
#
# Patterns are matched against the whole command string, not a parsed shell AST,
# so each aims to cover the plausible spellings rather than one. They are held in
# variables because `;` and `|` inside an unquoted [[ =~ ]] break bash's parser.
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

payload=$(cat)
tool=$(jq -r '.tool_name // empty' <<<"$payload")

decide() { # $1 = allow|deny|ask|defer, $2 = reason shown to Claude
  jq -n --arg d "$1" --arg r "$2" '{hookSpecificOutput:{
    hookEventName:"PreToolUse", permissionDecision:$d, permissionDecisionReason:$r}}'
  exit 0
}

# hf auth login writes these; `token` is the default profile and `stored_tokens`
# holds the named ones. Both are mode 600 and neither needs to be opened here.
RE_CRED='(secrets\.toml|huggingface/(token|stored_tokens))'
RE_READ_CRED="(cat|head|tail|less|more|bat|od|xxd|strings|base64|cp|grep|rg|awk|sed)[^;&|]*${RE_CRED}"
# Any install that does not come from the lockfile. `uv pip install` was the only
# spelling covered before, but pip is on PATH inside .venv, so a bare
# `pip install mlx-audio==0.5.0` mutates the same environment.
RE_INSTALL='(^|[;&|(`[:space:]])(uv[[:space:]]+pip[[:space:]]+(install|sync)|([./[:alnum:]_-]*/)?pip3?[[:space:]]+install|python3?[[:space:]]+-m[[:space:]]+pip[[:space:]]+install)'
# Commands that legitimately rewrite uv.lock. Not denied -- this is the
# sanctioned path, and the install deny above points at it -- but a human should
# see mlx-audio being allowed to move.
RE_RELOCK_ADD='(^|[;&|[:space:]])uv[[:space:]]+(add|remove)([[:space:]]|$)'
RE_RELOCK_UP='(^|[;&|[:space:]])uv[[:space:]]+(lock|sync)[^;&|]*(--upgrade|-U)([[:space:]]|$)'
# Only an actual run. Matching the bare filename made `ruff check
# verify_transcription.py` -- the lint command CLAUDE.md prescribes -- prompt with
# a 4 GB download warning, which is how a real run gets waved through.
RE_VERIFY_RUN='(uv[[:space:]]+run|python3?)[[:space:]]+[^;&|]*verify_transcription\.py'
RE_VERIFY_EXEC='(^|[;&|[:space:]])\./verify_transcription\.py'

case "$tool" in
  Read|Edit|Write|NotebookEdit)
    f=$(jq -r '.tool_input.file_path // .tool_input.notebook_path // empty' <<<"$payload")

    if [[ "$f" =~ $RE_CRED ]]; then
      decide deny "$f holds Hugging Face credentials, written by 'hf auth login'. Nothing in this project needs it opened -- huggingface_hub finds it on its own."
    fi

    case "$tool/$f" in
      Edit/*uv.lock|Write/*uv.lock|NotebookEdit/*uv.lock)
        decide ask "uv.lock is the only thing pinning mlx-audio to the 0.4.7 internals that _patch_vad_dtype in utils/models.py was written against -- pyproject.toml asks merely for >=0.4.4. Change pyproject.toml and run 'uv lock' rather than hand-editing a 550 KB lockfile." ;;
    esac
    ;;

  Bash)
    cmd=$(jq -r '.tool_input.command // empty' <<<"$payload")

    # Reading a credential through the shell is the same act as reading it with
    # the Read tool; the earlier version of this hook guarded only the latter.
    if [[ "$cmd" =~ $RE_READ_CRED ]]; then
      decide deny "That reads Hugging Face credentials. huggingface_hub finds the token on its own; nothing here needs its contents."
    fi

    if [[ "$cmd" =~ $RE_INSTALL ]]; then
      decide deny "That installs outside the lockfile. uv.lock is what pins mlx-audio 0.4.7, which _patch_vad_dtype was written against; 'uv pip install' and plain 'pip' both ignore it. Use 'uv sync'."
    fi

    if [[ "$cmd" =~ $RE_RELOCK_ADD ]] || [[ "$cmd" =~ $RE_RELOCK_UP ]]; then
      decide ask "That re-resolves uv.lock. mlx-audio may move off 0.4.7, and _patch_vad_dtype in utils/models.py wraps that release's private _segment_with_vad -- per CLAUDE.md a mismatch there fails as confident, wrong text rather than an error."
    fi

    # Each `uvx ... ruff` invocation must carry the exact pin, compared as whole
    # strings: a substring test let `uvx ruff@0.16.1 check . && uvx ruff format .`
    # through on the strength of the first half, and matched 0.16.10 as a prefix.
    RUFF=$(ruff_pin "$root")
    while IFS= read -r inv; do
      [[ -z "$inv" ]] && continue
      spec=$(sed -E 's/^uvx[[:space:]]+//' <<<"$inv")
      if [[ "$spec" != "$RUFF" ]]; then
        decide deny "'uvx $spec' aborts here: pyproject.toml sets required-version, so ruff refuses to run under any other version rather than linting. Every invocation in the command needs 'uvx $RUFF' -- including the second one in an '&&' chain."
      fi
    done < <(grep -oE 'uvx[[:space:]]+ruff(@[^[:space:]]+)?' <<<"$cmd")

    if [[ "$cmd" =~ $RE_VERIFY_RUN ]] || [[ "$cmd" =~ $RE_VERIFY_EXEC ]]; then
      decide ask "verify_transcription.py is the full integration test: it needs macOS 'say', ffmpeg and a valid HF token, downloads ~4 GB on a cold cache, and a full pass takes minutes. It takes no flags to select a subset -- comment out calls in run() for that."
    fi
    ;;
esac

exit 0
