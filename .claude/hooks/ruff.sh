#!/usr/bin/env bash
# PostToolUse: format the Python file Claude just wrote, then report its lint.
#
# The version pin is not decoration. pyproject.toml sets
# required-version = "==0.16.1", so an unpinned `uvx ruff` aborts with "Required
# version does not match the running version" rather than linting -- a hook
# written the obvious way would check nothing, silently, forever.
set -uo pipefail

payload=$(cat)
file=$(jq -r '.tool_input.file_path // empty' <<<"$payload")
[[ "$file" == *.py && -f "$file" ]] || exit 0

# Formatting is deterministic and changes no semantics, so it is applied.
uvx ruff@0.16.1 format -q "$file"

# Lint is only reported. Never --fix: F401 is enabled and autofixable, and this
# is a repo whose imports look wrong on purpose (below st.set_page_config in
# streamlit_app.py, below warnings.filterwarnings in verify_transcription.py).
# An import silently deleted here is the failure class the source comments and
# the E402 extend-ignore already exist to prevent.
if ! out=$(uvx ruff@0.16.1 check "$file" 2>&1); then
  printf 'ruff check failed on %s:\n%s\n' "$file" "$out" >&2
  exit 2  # non-blocking on PostToolUse; stderr is fed back to Claude
fi
exit 0
