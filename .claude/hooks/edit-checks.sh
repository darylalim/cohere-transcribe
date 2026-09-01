#!/usr/bin/env bash
# PostToolUse: check the Python file Claude just wrote.
#
# ty lives here rather than in the Stop gate. On Stop the only channels are
# exit 2 (blocks the turn) and additionalContext (resumes the model, which also
# costs a turn and sets stop_hook_active) -- neither is advisory. On PostToolUse
# exit 2 is genuinely non-blocking: the edit has already landed, so the report
# is information rather than a veto. That is the distinction CLAUDE.md draws
# between pinned ruff and unpinned pre-1.0 ty.
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-$(pwd)}"

payload=$(cat)

# Read stdin before anything can exit, so no fail-closed path leaves the writer
# on a broken pipe. jq is checked here rather than assumed: without it `file`
# comes back empty, the *.py test below fails, and the hook exits 0 having
# checked nothing -- the same silent no-op the lib.sh guard further down exists
# to stop. This report is stderr plus exit 2, so unlike guard.sh it needs no jq
# to deliver its own failure. It cannot say which file was edited, because
# working that out is exactly what just broke.
command -v jq >/dev/null || {
  printf 'edit-checks.sh: jq is not on PATH, so the payload could not be read and this edit went unchecked.\n' >&2
  exit 2
}

file=$(jq -r '.tool_input.file_path // empty' <<<"$payload")
[[ "$file" == *.py && -f "$file" ]] || exit 0

# Only files inside the project. Without this the hook reformats and lints
# throwaway scratchpad scripts, which are meant to be disposable and often carry
# deliberate unused imports -- costing two uvx spawns and a lint complaint each.
case "$(cd "$(dirname "$file")" && pwd -P)/" in
  "$(cd "$root" && pwd -P)"/*) ;;
  *) exit 0 ;;
esac

# Fail closed when the helpers are not available, rather than checking nothing
# and saying so nowhere. `set -u` is on but `set -e` is not, so this used to fall
# straight through to `tooling_reachable "$RUFF" || exit 0` below -- an undefined
# command, so the `||` fired and the hook exited 0 having formatted nothing and
# reported nothing. Measured: rc=0 with the dirty file byte-identical, against
# rc=2 and an F401 report when the helpers are there. The `command not found`
# lines bash prints land on stderr of an exit-0 hook, which is not surfaced the
# way exit 2 is, so the only in-session checker went quiet without a word. That
# is the succeeds-while-being-wrong shape guard.sh exists to catch, living inside
# the hooks themselves, and lib.sh is a repo file one rename or bad merge away.
#
# Two conditions, because the first version tested only the first and the second
# reproduced the whole failure: a lib.sh that no longer defines ruff_pin or
# tooling_reachable -- either renamed, or lost to a merge that kept the file --
# sources cleanly, returns 0, and walks straight past a `source ... ||` guard.
#
# Placed after the *.py filter, not before it. Above, a missing lib.sh reported
# "ruff and ty did NOT run on this edit" on every Edit and Write in the session,
# including the CLAUDE.md and .yml files they were never going to run on.
lib="$(dirname "${BASH_SOURCE[0]}")/lib.sh"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
source "$lib" || {
  printf 'edit-checks.sh: cannot source %s -- ruff and ty did NOT run on %s.\n' "$lib" "$file" >&2
  exit 2
}
if ! declare -F ruff_pin >/dev/null || ! declare -F tooling_reachable >/dev/null; then
  printf 'edit-checks.sh: %s does not define ruff_pin and tooling_reachable -- ruff and ty did NOT run on %s.\n' "$lib" "$file" >&2
  exit 2
fi

RUFF=$(ruff_pin "$root")
tooling_reachable "$RUFF" || exit 0   # cold cache / offline is not a lint result

notes=()

# Formatting is deterministic and changes no semantics, so it is applied. But it
# is applied *after* Claude wrote the file, so Claude's idea of the contents is
# now stale -- a later Edit whose old_string spans a re-wrapped line fails to
# match for no visible reason. Say so when it actually changed something.
if ! uvx "$RUFF" format --check -q "$file" >/dev/null 2>&1; then
  uvx "$RUFF" format -q "$file"
  notes+=("Reformatted $file with ruff -- re-read it before editing, your copy is stale.")
fi

# Never --fix: F401 is enabled and autofixable, and this is a repo whose imports
# sit below st.set_page_config and below warnings.filterwarnings on purpose. An
# import removed by an autofixer is the failure the E402 ignore exists to stop.
if ! out=$(uvx "$RUFF" check "$file" 2>&1); then
  notes+=("ruff check failed on $file:"$'\n'"$out")
fi

# Advisory: ty is unpinned and pre-1.0, so a new diagnostic is expected rather
# than a regression. Reported, never fatal.
if tooling_reachable ty && ! out=$(uvx ty check "$file" 2>&1); then
  notes+=("ty reported diagnostics on $file (advisory -- ty is unpinned and pre-1.0, so this may be a new check rather than a regression):"$'\n'"$out")
fi

if ((${#notes[@]})); then
  printf '%s\n' "${notes[@]}" >&2
  exit 2   # non-blocking on PostToolUse; stderr is fed back to Claude
fi
exit 0
