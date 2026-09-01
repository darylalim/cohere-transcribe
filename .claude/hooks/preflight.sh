#!/usr/bin/env bash
# SessionStart: name the environmental preconditions this app fails on, before a
# transcription stalls on one. Four probes, in two kinds. Three are the app's:
# a missing ffmpeg (which silently narrows UPLOAD_TYPES to wav/mp3/flac), a
# machine MLX cannot run on, and a missing uv.lock (which lets `uv sync` resolve
# mlx-audio fresh against a floor of >=0.4.4). The fourth, jq, is a precondition
# of the hooks doing the checking rather than of the app being checked, which is
# why it runs first. Keep this count and this list in step with the probes below
# and with the line naming them in CLAUDE.md.
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

# jq before the app's own preconditions, because it is the one the hooks rest on
# and its absence is more silent than anything below. guard.sh parses its payload
# with jq, so without it `$tool` is empty, the case matches no arm, and the
# script reaches `exit 0` -- and a PreToolUse hook that exits 0 with empty stdout
# is protocol-indistinguishable from one that looked and had no objection. Every
# deny and ask in the file abstains at once, with nothing said anywhere;
# edit-checks.sh goes quiet the same way, its file= coming back empty and failing
# the *.py test. Measured with jq off PATH: an out-of-lockfile install, a
# credential read and a verify_transcription.py run all returned rc=0 and no
# decision, leaving only a `jq: command not found` on the stderr of an exit-0
# hook, which is not surfaced the way exit 2 is.
#
# Probed here rather than failed closed inside guard.sh. `decide` builds its JSON
# with jq, so failing closed there would have to printf the object by hand -- and
# an `exit 2` at the top of guard.sh blocks `brew install jq`, blocks reading
# guard.sh and blocks editing settings.json, which is an unrecoverable session
# traded for a binary macOS ships at /usr/bin. Advisory at session start is the
# proportionate answer; it also covers edit-checks.sh, which no guard-side check
# would have.
command -v jq >/dev/null \
  || notes+=("jq is not on PATH -- guard.sh and edit-checks.sh both parse their payload with it, and both exit 0 having checked nothing when it is missing, saying so nowhere. 'brew install jq'.")

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
