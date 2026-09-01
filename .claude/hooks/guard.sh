#!/usr/bin/env bash
# PreToolUse: the operations in this repo that fail quietly when done wrong.
#
# Quietly is the entire test for membership. A rule rejecting an unpinned
# `uvx ruff` lived here until it was cut: ruff reads required-version itself and
# aborts with "Required version `==0.16.1` does not match the running version",
# which is the whole diagnosis, so the hook only saved one round trip. Every rule
# below guards something that instead succeeds while being wrong -- the wrong
# mlx-audio transcribing confident nonsense, a credential read, four gigabytes
# downloaded before anything says why.
#
# `deny` for the ones that are always wrong here -- the reason string reaches
# Claude, which then self-corrects. `ask` for the ones that are sometimes right,
# so the decision reaches a human instead.
#
# Patterns are matched against the whole command string, not a parsed shell AST,
# so each aims to cover the plausible spellings rather than one. They are held in
# variables because `;` and `|` inside an unquoted [[ =~ ]] break bash's parser.
set -uo pipefail

payload=$(cat)
tool=$(jq -r '.tool_name // empty' <<<"$payload")

decide() { # $1 = allow|deny|ask -- the three permissionDecision takes; only the
           # last two are used here. It read "allow|deny|ask|defer" until a rule
           # written against that comment would have emitted a value the host
           # does not recognise -- and since this hook exits 0 either way, the
           # result would have been an unnoticed abstention, not an error.
           # $2 = reason shown to Claude
  jq -n --arg d "$1" --arg r "$2" '{hookSpecificOutput:{
    hookEventName:"PreToolUse", permissionDecision:$d, permissionDecisionReason:$r}}'
  exit 0
}

# hf auth login writes these; `token` is the default profile and `stored_tokens`
# holds the named ones. Both are mode 600 and neither needs to be opened here.
RE_CRED='(secrets\.toml|huggingface/(token|stored_tokens))'
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
  Read|Edit|Write)
    f=$(jq -r '.tool_input.file_path // empty' <<<"$payload")

    if [[ "$f" =~ $RE_CRED ]]; then
      decide deny "$f holds Hugging Face credentials, written by 'hf auth login'. Nothing in this project needs it opened -- huggingface_hub finds it on its own."
    fi

    case "$tool/$f" in
      Edit/*uv.lock|Write/*uv.lock)
        decide ask "uv.lock is the only thing pinning mlx-audio to the 0.5.1 internals that _patch_vad_dtype in utils/models.py targets -- pyproject.toml asks merely for >=0.4.4. Change pyproject.toml and run 'uv lock' rather than hand-editing a 550 KB lockfile." ;;
    esac
    ;;

  Bash)
    cmd=$(jq -r '.tool_input.command // empty' <<<"$payload")

    # Reading a credential through the shell is the same act as reading it with
    # the Read tool; the earlier version of this hook guarded only the latter.
    #
    # Match the path, exactly as the Read arm above does. This sat behind an
    # alternation of fourteen reader commands until that was measured, and
    # enumerating readers turns out to be the wrong axis in both directions. Of
    # 17 real spellings of a credential read, 15 walked straight through: a verb
    # outside the list (tr, cut, tee, dd, or python3 -c open(...)), any ; && or |
    # between verb and path defeating the [^;&|]* join (cd into the cache, then
    # cat a bare relative "token"), or a redirection with no verb at all. In the
    # other direction the short entries carried no word boundaries, so `od`
    # matched inside chmod and code, `rg` inside merge, `less` inside unless and
    # `cat` inside location -- across 57 transcripts of this project the rule
    # fired exactly twice, both on a command auditing the hook itself, and never
    # once on a credential read.
    #
    # The wider match does deny commands that merely name the file without
    # reading it -- chmod on it, or an echo mentioning secrets.toml. That is the
    # loud direction and the cheap one: the reason string quotes the path, so
    # Claude self-corrects in a turn, where all fifteen misses above were silent.
    #
    # Twelve of those fifteen close here. Three cannot, and no string match will
    # reach them, because the path never appears in the command: `cd` into the
    # directory then read a bare relative name, an assembled variable, and
    # `find -exec`. Those need a shell grammar rather than a regex -- which is
    # what the CLI's own Read() deny path has, and this hook does not. Left open
    # knowingly, and written down rather than papered over with a longer
    # alternation that would not catch them either.
    if [[ "$cmd" =~ $RE_CRED ]]; then
      decide deny "That command names a Hugging Face credential file. huggingface_hub finds the token on its own -- nothing here needs its contents, its mode or its location."
    fi

    if [[ "$cmd" =~ $RE_INSTALL ]]; then
      decide deny "That installs outside the lockfile. uv.lock is what pins mlx-audio 0.5.1, which _patch_vad_dtype targets; 'uv pip install' and plain 'pip' both ignore it. Use 'uv sync'."
    fi

    if [[ "$cmd" =~ $RE_RELOCK_ADD ]] || [[ "$cmd" =~ $RE_RELOCK_UP ]]; then
      decide ask "That re-resolves uv.lock. mlx-audio may move off 0.5.1, and _patch_vad_dtype in utils/models.py wraps that release's private _segment_with_vad -- per CLAUDE.md a mismatch there fails as confident, wrong text rather than an error."
    fi

    if [[ "$cmd" =~ $RE_VERIFY_RUN ]] || [[ "$cmd" =~ $RE_VERIFY_EXEC ]]; then
      decide ask "verify_transcription.py is the full integration test: it needs macOS 'say', ffmpeg and a valid HF token, downloads ~4 GB on a cold cache, and a full pass takes minutes. It takes no flags to select a subset -- comment out calls in run() for that."
    fi
    ;;
esac

exit 0
