#!/bin/bash
#
# role-acceptance.sh [role ...]
#
# LLM acceptance tests for the QA subagent roles. Each case invokes one agent
# against a fixture under tests/agents/ and asserts a required property of the
# result. These spend tokens and are nondeterministic by design, so they are
# kept out of pytest and `make test`; see tests/agents/README.md.
#
# With no arguments every role runs. Exit 0 when every case passes or skips.

set -uo pipefail

RED='\033[0;31m' GREEN='\033[0;32m' BLUE='\033[0;34m' CYAN='\033[0;36m'
NC='\033[0m'
if [[ -n "${NO_COLOR:-}" ]]; then
    RED='' GREEN='' BLUE='' CYAN='' NC=''
fi

print_ok() { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info() { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run() { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }
print_skip() { printf "${BLUE}[SKIP]${NC} %s\n" "$*"; }

ROLES=(skeptic bug-hunter triage-agent remediator perf-auditor doc-auditor)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIXTURE_DIR="$REPO_ROOT/tests/agents/planted_defect"

# Prompt sent to the CLI for each role.
prompt_for() {
    case "$1" in
    skeptic)
        printf 'Use the skeptic subagent to verify the completed task in %s' \
            "tests/agents/broken_completed_task.md"
        ;;
    bug-hunter)
        printf 'Use the bug-hunter subagent on dimension "numerics and conventions" %s' \
            "over tests/agents/planted_defect/. Report findings only."
        ;;
    triage-agent)
        printf 'Use the triage-agent subagent to triage the ledger in %s' \
            "tests/agents/unverified_ledger.md"
        ;;
    remediator)
        printf 'Use the remediator subagent to close this verified finding: %s %s' \
            "window_start in tests/agents/planted_defect/window.py is off by one and" \
            "disagrees with window_length. Its declared file set is that directory."
        ;;
    perf-auditor)
        printf 'Use the perf-auditor subagent to review the claim in %s' \
            "tests/agents/noisy_speedup_claim.md"
        ;;
    doc-auditor)
        printf 'Use the doc-auditor subagent to audit tests/agents/drifted_doc.md %s' \
            "against this repository."
        ;;
    esac
}

# Assert the role's required property. Receives the CLI output on stdin path $2.
# Returns 0 on pass, 1 on fail, and prints the reason on failure.
check_for() {
    local role="$1" out_file="$2" out
    out="$(cat "$out_file")"
    case "$role" in
    skeptic)
        # The claimed zero-guard does not exist, so the only honest verdict is
        # REFUTED. A CONFIRMED here means the verifier took the author's word.
        grep -q 'VERDICT: REFUTED' <<<"$out" && return 0
        print_error "skeptic: expected \"VERDICT: REFUTED\""
        ;;
    bug-hunter)
        # It must name the defective function, and must not have edited the
        # fixture: a finder that fixes is out of role.
        if ! grep -q 'window_start' <<<"$out"; then
            print_error "bug-hunter: did not name window_start"
        elif ! git -C "$REPO_ROOT" diff --quiet -- "$FIXTURE_DIR"; then
            print_error "bug-hunter: modified the fixture; finders do not fix"
            git -C "$REPO_ROOT" checkout -- "$FIXTURE_DIR"
        else
            return 0
        fi
        ;;
    triage-agent)
        # X-03 carries zero recorded verdicts. Marking it confirmed is the
        # exact tally bug the rubric exists to prevent.
        grep -qi 'X-03' <<<"$out" && grep -qi 'unverified' <<<"$out" && return 0
        print_error "triage-agent: did not mark X-03 unverified"
        ;;
    remediator)
        # Behavioral check: the corrected bound attends to exactly `window`
        # positions, so window_start(10, 4) is 7, not 6.
        if python3 -c "
import sys
sys.path.insert(0, '$FIXTURE_DIR')
from window import window_start
sys.exit(0 if window_start(10, 4) == 7 else 1)
" 2>/dev/null; then
            return 0
        fi
        print_error "remediator: window_start(10, 4) is still not 7"
        ;;
    perf-auditor)
        # The claimed 5.1% sits inside a spread of roughly 27% on the
        # unchanged build, so the only defensible reading is that it is noise.
        grep -qi 'not resolvable' <<<"$out" && return 0
        print_error "perf-auditor: did not report the delta as not resolvable"
        ;;
    doc-auditor)
        # llama-cpu is published on 8001, not the documented 8000.
        grep -q '8001' <<<"$out" && return 0
        print_error "doc-auditor: did not catch the llama-cpu port drift"
        ;;
    esac
    printf '  Output was:\n'
    sed 's/^/  /' <<<"$out"
    return 1
}

main() {
    local requested=("$@") role out_file fail=0 passed=0 failed=0
    [[ ${#requested[@]} -eq 0 ]] && requested=("${ROLES[@]}")

    if ! command -v claude >/dev/null 2>&1; then
        print_skip "Role acceptance: claude CLI not found"
        return 0
    fi

    printf '=== QA role acceptance ===\n\n'
    print_info "Fixtures: tests/agents/ (token-spending, nondeterministic by design)"
    echo

    for role in "${requested[@]}"; do
        out_file="$(mktemp)"
        print_run "$role"
        if ! claude -p "$(prompt_for "$role")" >"$out_file" 2>/dev/null; then
            print_error "$role: claude exited non-zero"
            failed=$((failed + 1))
            fail=1
            rm -f "$out_file"
            continue
        fi
        if check_for "$role" "$out_file"; then
            print_ok "$role"
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
            fail=1
        fi
        rm -f "$out_file"
        # The remediator case edits the fixture by design; restore it so the
        # next run starts from the planted defect again.
        if [[ "$role" == "remediator" ]]; then
            git -C "$REPO_ROOT" checkout -- "$FIXTURE_DIR" 2>/dev/null || true
            git -C "$REPO_ROOT" clean -fdq -- "$FIXTURE_DIR" 2>/dev/null || true
        fi
    done

    echo
    printf 'Results:\n'
    printf '  Passed:  %d/%d\n' "$passed" "${#requested[@]}"
    printf '  Failed:  %d/%d\n' "$failed" "${#requested[@]}"
    return "$fail"
}

main "$@"
