#!/usr/bin/env bash
# Stop: do not finish a turn with the repo's lint or formatting dirty.
#
# ruff only. ty is checked per-edit in edit-checks.sh instead, because Stop has
# no advisory channel: exit 2 blocks, and additionalContext resumes the model,
# which costs a turn just the same. ruff is pinned by required-version, so its
# verdict is reproducible and a failure here is a real regression.
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-.}"
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

payload=$(cat)
cd "$root" || exit 0

# Keyed per session so two sessions cannot share a failure count.
session=$(jq -r '.session_id // "nosession"' <<<"$payload")
STATE="${TMPDIR:-/tmp}/claude-gate-$(id -u)-${session//[^A-Za-z0-9_-]/_}"

RUFF=$(ruff_pin .)
# A cold uvx cache or no network is not a lint failure. Reporting it as one
# blocks the turn with something Claude cannot fix.
tooling_reachable "$RUFF" || exit 0

# Both checks run unconditionally. Chaining them with && short-circuited: a
# formatting slip meant `ruff check` never ran, so an undefined name rode out
# behind it and the gate reported only the whitespace.
fmt_out=$(uvx "$RUFF" format --check . 2>&1); fmt_rc=$?
lint_out=$(uvx "$RUFF" check . 2>&1);         lint_rc=$?
(( fmt_rc == 0 && lint_rc == 0 )) && { rm -f "$STATE" 2>/dev/null; exit 0; }

out=""
(( fmt_rc != 0 ))  && out+="$fmt_out"$'\n'
(( lint_rc != 0 )) && out+="$lint_out"$'\n'

# Loop guard. The previous version exited early whenever stop_hook_active was
# set, which did not just prevent a loop -- it disabled the gate for the whole
# remediation turn, so whatever Claude wrote while fixing the first failure was
# never checked. Count consecutive *identical* failures instead: a changed
# failure resets the count, so real fixes are always re-verified.
hash=$(printf '%s' "$out" | shasum | cut -d' ' -f1)
read -r prev_hash prev_count < <(cat "$STATE" 2>/dev/null || echo "- 0")
count=1
[[ "$hash" == "$prev_hash" ]] && count=$((prev_count + 1))
printf '%s %s\n' "$hash" "$count" > "$STATE"

# Reported three times unchanged means blocking again only loops.
(( count > 3 )) && exit 0

printf 'Repo gate failed -- fix before finishing:\n%s' "$out" >&2
exit 2
