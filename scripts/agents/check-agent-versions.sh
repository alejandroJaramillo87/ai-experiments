#!/bin/bash
#
# check-agent-versions.sh
#
# Compare the CLI version pins in docker/agents/Dockerfile.agents against the
# npm registry. Check-only: never edits the Dockerfile. Bumps stay manual, one
# pin at a time, re-running the acceptance checks (see AGENTS.md operating
# rules and docs/sandbox/agent-sandbox.md).

set -euo pipefail

RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
CYAN='\033[0;36m' NC='\033[0m'

# Respect NO_COLOR env var or --no-color flag
if [[ -n "${NO_COLOR:-}" ]] || [[ "${1:-}" == "--no-color" ]]; then
    RED='' GREEN='' YELLOW='' CYAN='' NC=''
fi

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }

DOCKERFILE="${DOCKERFILE:-docker/agents/Dockerfile.agents}"

if [[ ! -f "$DOCKERFILE" ]]; then
    print_error "Dockerfile not found: $DOCKERFILE"
    echo "  Solution: run from the repository root"
    exit 1
fi

# Read a pinned version out of an `ARG NAME=x.y.z` line.
pinned_version() {
    sed -n "s/^ARG $1=\(.*\)$/\1/p" "$DOCKERFILE"
}

# Latest published version from the npm registry (dist-tag). Uses curl, not
# npm, so the check works on hosts without node.
registry_version() {
    local pkg="$1" tag="$2"
    curl -sf --max-time 15 "https://registry.npmjs.org/${pkg//\//%2F}/${tag}" \
        | jq -r '.version' 2>/dev/null
}

# check <arg-name> <npm-package> <dist-tag>
check() {
    local arg="$1" pkg="$2" tag="$3"
    local pinned latest
    pinned=$(pinned_version "$arg")
    if [[ -z "$pinned" ]]; then
        print_error "$pkg: no ARG $arg pin found in $DOCKERFILE"
        return 1
    fi
    latest=$(registry_version "$pkg" "$tag") || true
    if [[ -z "$latest" || "$latest" == "null" ]]; then
        print_warn "$pkg: registry lookup failed (pinned $pinned)"
        echo "  Solution: check network access to registry.npmjs.org"
        return 0
    fi
    if [[ "$pinned" == "$latest" ]]; then
        print_ok "$pkg: $pinned (current, $tag)"
    elif [[ "$(printf '%s\n%s\n' "$pinned" "$latest" | sort -V | tail -1)" == "$pinned" ]]; then
        # A pin newer than the tag happens when tracking `stable`, which lags `latest`.
        print_ok "$pkg: $pinned (ahead of $tag $latest)"
    else
        print_warn "$pkg: pinned $pinned, $tag is $latest"
        echo "  Solution: bump ARG $arg in $DOCKERFILE, rebuild, re-run acceptance checks"
        echo "  Command: sed -i 's/^ARG $arg=.*/ARG $arg=$latest/' $DOCKERFILE"
        behind=1
    fi
}

echo "=== Agent CLI version pins ==="
echo

behind=0
# claude-code is checked against the `stable` tag (about one week behind
# `latest`, skips major regressions); the others publish only `latest`.
check OPENCODE_VERSION    opencode-ai                 latest
check CLAUDE_CODE_VERSION @anthropic-ai/claude-code   stable
check PYRIGHT_VERSION     pyright                     latest
echo

if [[ "$behind" -eq 1 ]]; then
    print_info "Pins behind: bump one at a time per the pin-then-update rule"
else
    print_ok "All pins current"
fi
