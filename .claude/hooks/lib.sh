#!/usr/bin/env bash
# Shared by the hooks in this directory. Sourced, never executed.

# pyproject.toml already single-sources the ruff version through
# required-version; read it from there rather than repeating the literal in
# every hook. A hook that disagrees with pyproject.toml is worse than no hook:
# guard.sh would start denying the command pyproject.toml requires.
ruff_pin() {
  local root="${1:-.}" pin
  pin=$(sed -n 's/^required-version *= *"==\([0-9][0-9.]*\)".*/\1/p' \
        "$root/pyproject.toml" 2>/dev/null | head -1)
  [[ -n "$pin" ]] && printf 'ruff@%s' "$pin" || printf 'ruff'
}

# uvx cannot reach a tool when the cache is cold and the network is down. That
# is not a lint failure and must not be reported as one -- Claude cannot fix a
# missing network, and would burn a turn trying.
tooling_reachable() {
  uvx "$1" --version >/dev/null 2>&1
}
