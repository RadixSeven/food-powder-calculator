#!/usr/bin/env bash
# Run every quality gate and print a single green/red verdict to stderr.
#
# Reformatting alone (`ruff format`, `mdformat`) is never a failure: if it
# touches files, the user has nothing to do — the tree is now formatted. A
# failure means a gate produced something the user must act on (un-auto-fixable
# lint, type errors, test failures, coverage shortfall, shellcheck findings).

set -uo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

if [[ -t 2 ]]; then
    GREEN=$'\033[0;32m'
    RED=$'\033[0;31m'
    BOLD=$'\033[1m'
    RESET=$'\033[0m'
else
    GREEN=''
    RED=''
    BOLD=''
    RESET=''
fi

failures=()

run_gate() {
    local name="$1"; shift
    printf '\n%s>>> %s%s\n' "${BOLD}" "${name}" "${RESET}" >&2
    if "$@"; then
        return 0
    fi
    failures+=("${name}")
    return 1
}

# Format steps: never a failure on their own; reformatting just updates the
# tree, which is something the user pulls in rather than has to act on.
printf '\n%s>>> ruff format%s\n' "${BOLD}" "${RESET}" >&2
uv run ruff format . || true
printf '\n%s>>> mdformat%s\n' "${BOLD}" "${RESET}" >&2
uv run mdformat . || true

run_gate "ruff check --fix" uv run ruff check --fix .
run_gate "pyrefly check"    uv run pyrefly check

# Discover every tracked .sh in the repo so newly-added scripts get checked
# automatically. `git ls-files` respects .gitignore and never reaches into
# .venv / .pants.d / pex caches.
shell_scripts_raw=$(git ls-files '*.sh')
mapfile -t shell_scripts <<< "${shell_scripts_raw}"
if [[ -n "${shell_scripts_raw}" ]]; then
    run_gate "shellcheck" uvx --from shellcheck-py shellcheck \
        --severity=style --enable=all "${shell_scripts[@]}"
fi

run_gate "pants test"       pants test ::

echo >&2
if [[ ${#failures[@]} -eq 0 ]]; then
    printf '%s%s✓ All quality gates passed%s\n' "${BOLD}" "${GREEN}" "${RESET}" >&2
    exit 0
fi

printf '%s%s✗ Failed: %s%s\n' "${BOLD}" "${RED}" "${failures[*]}" "${RESET}" >&2
exit 1
