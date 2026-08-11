#!/usr/bin/env bash
# Sourced by edit-checks.sh, never executed. It is the only consumer left --
# gate.sh was deleted and guard.sh's ruff-version rule with it -- but this stays
# a file rather than being inlined because .github/workflows/ci.yml points at it
# as where the pattern below is written down, and re-derives it in YAML.

# pyproject.toml already single-sources the ruff version through
# required-version; read it from there rather than repeating the literal. A hook
# that disagrees with pyproject.toml is worse than no hook: it would invoke a
# ruff that required-version refuses to run, and report failure while linting
# nothing.
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
