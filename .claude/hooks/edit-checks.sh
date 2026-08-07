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
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

payload=$(cat)
file=$(jq -r '.tool_input.file_path // empty' <<<"$payload")
[[ "$file" == *.py && -f "$file" ]] || exit 0

# Only files inside the project. Without this the hook reformats and lints
# throwaway scratchpad scripts, which are meant to be disposable and often carry
# deliberate unused imports -- costing two uvx spawns and a lint complaint each.
case "$(cd "$(dirname "$file")" && pwd -P)/" in
  "$(cd "$root" && pwd -P)"/*) ;;
  *) exit 0 ;;
esac

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
