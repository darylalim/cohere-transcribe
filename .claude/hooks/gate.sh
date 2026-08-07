#!/usr/bin/env bash
# Stop: do not finish a turn with the repo dirty. The whole gate is ~0.35s.
#
# ruff blocks and ty does not, on purpose. ruff is pinned to 0.16.1 by
# required-version, so its verdict is reproducible and a failure is a real
# regression. ty is deliberately unpinned -- it is pre-1.0 and ships near-daily,
# and a new release surfacing a new diagnostic is expected rather than a
# regression, so it is reported as context instead of blocking the turn.
set -uo pipefail

payload=$(cat)
# Never re-block a turn that is already continuing because of this hook.
[[ "$(jq -r '.stop_hook_active // false' <<<"$payload")" == "true" ]] && exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

if ! ruff_out=$( { uvx ruff@0.16.1 format --check . && uvx ruff@0.16.1 check .; } 2>&1 ); then
  printf 'Repo gate failed -- fix before finishing:\n%s\n' "$ruff_out" >&2
  exit 2
fi

# Advisory. Exit 0 means stderr is not shown to Claude, so ty's verdict has to
# travel as additionalContext to be seen at all.
if ! ty_out=$(uvx ty check . 2>&1); then
  jq -n --arg c "ty reported diagnostics (advisory -- ty is unpinned and pre-1.0, so this may be a new check rather than a regression):"$'\n'"$ty_out" \
    '{hookSpecificOutput:{hookEventName:"Stop", additionalContext:$c}}'
fi
exit 0
