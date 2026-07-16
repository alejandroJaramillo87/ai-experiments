#!/usr/bin/env bash
# runpod-idle-check.sh - "never pay for a forgotten warm worker" guardrail.
#
# Non-blocking (always exits 0). Prints a warning when a RunPod endpoint still has a warm
# worker or is accruing spend with no active session. Wired as a Claude Code Stop/SessionEnd
# hook and an OpenCode session.idle plugin.
#
# Modes:
#   - Live:    queries the RunPod GraphQL API using RUNPOD_API_KEY (and optional
#              RUNPOD_ENDPOINT_ID). Skips cleanly if no key is set.
#   - Fixture: RUNPOD_IDLE_FIXTURE=<file.json> feeds a simulated API response so the warning
#              path is testable without real RunPod credentials.
#
# A dedicated "spend report" line is emitted when RUNPOD_REPORT=1 (used by make agent-usage).

set -euo pipefail

RUNPOD_API="${RUNPOD_API:-https://api.runpod.io/graphql}"
FIXTURE="${RUNPOD_IDLE_FIXTURE:-}"

warn() { printf '[WARN] RunPod: %s\n' "$1"; }
info() { printf '[INFO] RunPod: %s\n' "$1"; }
skip() { printf '[SKIP] RunPod: %s\n' "$1"; }

if ! command -v jq >/dev/null 2>&1; then
    skip "jq not found; cannot evaluate endpoint state"
    exit 0
fi

# Obtain the endpoints JSON, from a fixture or a live query.
resp=""
if [ -n "$FIXTURE" ]; then
    if [ ! -f "$FIXTURE" ]; then
        skip "fixture not found: $FIXTURE"
        exit 0
    fi
    resp="$(cat "$FIXTURE")"
elif [ -n "${RUNPOD_API_KEY:-}" ]; then
    if ! command -v curl >/dev/null 2>&1; then
        skip "curl not found; cannot query RunPod"
        exit 0
    fi
    query='{"query":"query { myself { endpoints { id name workersStandby workersRunning currentSpendPerHr } } }"}'
    resp="$(curl -sS -X POST "$RUNPOD_API" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -d "$query" 2>/dev/null || true)"
    if [ -z "$resp" ]; then
        skip "no response from RunPod API"
        exit 0
    fi
else
    skip "RUNPOD_API_KEY not set (set it, or use RUNPOD_IDLE_FIXTURE for testing)"
    exit 0
fi

endpoints="$(printf '%s' "$resp" | jq -c '.data.myself.endpoints // [] | .[]' 2>/dev/null || true)"
if [ -z "$endpoints" ]; then
    info "no endpoints reported"
    exit 0
fi

total_spend=0
found_warm=0
while IFS= read -r ep; do
    [ -n "$ep" ] || continue
    name="$(printf '%s' "$ep" | jq -r '.name // .id // "unknown"')"
    # Normalize to integers: the API can return fractional counts (1.0), which
    # break bash arithmetic and would abort the loop into a false all-clear.
    standby="$(printf '%s' "$ep" | jq -r '(.workersStandby // 0) | (tonumber? // 0) | floor')"
    running="$(printf '%s' "$ep" | jq -r '(.workersRunning // 0) | (tonumber? // 0) | floor')"
    spend="$(printf '%s' "$ep" | jq -r '.currentSpendPerHr // 0')"
    warm=$(( standby + running ))
    # Accumulate spend for the report mode.
    total_spend="$(awk -v a="$total_spend" -v b="$spend" 'BEGIN{printf "%.4f", a + b}')"
    if [ "$warm" -gt 0 ] || awk -v s="$spend" 'BEGIN{exit !(s > 0)}'; then
        found_warm=1
        warn "endpoint '$name' has $warm warm worker(s), spend \$$spend/hr"
        printf "  Action: scale it to zero if no session is active\n"
    fi
done <<EOF
$endpoints
EOF

if [ "${RUNPOD_REPORT:-0}" = "1" ]; then
    printf 'runpod_spend_per_hr=%s\n' "$total_spend"
fi

if [ "$found_warm" -eq 0 ]; then
    info "no warm workers, no active spend"
fi

exit 0
