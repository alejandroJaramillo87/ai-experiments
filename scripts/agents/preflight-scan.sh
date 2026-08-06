#!/usr/bin/env bash
#
# preflight-scan.sh — supply-chain scan for coding-agent config.
#
# Both OpenCode and Claude Code auto-load repo-local configuration that runs
# with full user privileges: OpenCode plugins/skills/commands, Claude Code
# settings.json hooks/skills/commands/agents, and either tool's AGENTS.md (a
# prompt-injection vector). Opening an agent inside an untrusted checkout can
# therefore execute attacker-controlled code or inject attacker-controlled
# prompts. This scan inspects both families before that happens.
#
# Two modes, chosen by whether the target IS the trusted home repository:
#   - self  : target is this repo's root. Vectors are reported for awareness;
#             the scan does not fail (this is your own, reviewed config).
#   - extern: target is any other tree (a third-party checkout, a submodule
#             update, or the test fixture). Every discovered vector is flagged
#             for manual review and the scan exits non-zero — external agent
#             config is untrusted until read.
#
# Usage: preflight-scan.sh [TARGET_DIR]   (default: .)

set -euo pipefail

RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
CYAN='\033[0;36m' NC='\033[0m'
if [[ -n "${NO_COLOR:-}" ]] || [[ ! -t 1 ]]; then
    RED='' GREEN='' YELLOW='' CYAN='' NC=''
fi
print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }

target="${1:-.}"
if [[ ! -d "$target" ]]; then
    print_error "Preflight target not found: $target"
    echo "  Solution: pass an existing directory (make agent-preflight PREFLIGHT_DIR=<path>)"
    exit 2
fi

target_abs="$(cd "$target" && pwd)"

# Determine mode: is the target the root of the repo we are running inside?
#
# git is the preferred oracle, but it is unusable in a contained agent sandbox: /work is a
# linked worktree whose .git file points outside the bind mount, so rev-parse fails and
# self_root comes back empty. Without a fallback the home repo would then be scanned as
# "extern", flagging every one of its own hooks and plugins as an untrusted vector and
# exiting 1 -- red for the wrong reason. Fall back to a marker check: this repository is
# identified by AGENTS.md sitting beside the agent tooling directories.
self_root=""
if self_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then :; fi

marker_fallback=0
mode="extern"
if [[ -n "$self_root" && "$target_abs" == "$self_root" ]]; then
    mode="self"
elif [[ -z "$self_root" && -f "$target_abs/AGENTS.md" \
        && -d "$target_abs/.claude/agents" && -d "$target_abs/.opencode" ]]; then
    mode="self"
    marker_fallback=1
fi

echo "=== Agent supply-chain preflight ==="
echo
if [[ "$mode" == "self" ]]; then
    print_info "Target: $target_abs (trusted home repo — advisory scan)"
    if [[ "$marker_fallback" -eq 1 ]]; then
        print_info "Repo identity: resolved by marker files (git unavailable in this sandbox)"
    fi
else
    print_info "Target: $target_abs (external checkout — every vector requires review)"
fi
echo

# Vector inventory. Files and directories that either execute code or inject
# prompt content when an agent opens this tree.
files=(
    "opencode.json"
    "AGENTS.md"
    ".claude/settings.json"
    ".claude/settings.local.json"
)
dirs=(
    ".opencode"
    ".claude/skills"
    ".claude/commands"
    ".claude/agents"
)

fail=0

flag() {
    # flag <path> <description>
    if [[ "$mode" == "extern" ]]; then
        print_error "$1: $2 — review before opening an agent here"
        fail=1
    else
        print_info "$1: $2"
    fi
}

echo "OpenCode + Claude Code config vectors:"
for f in "${files[@]}"; do
    [[ -e "$target_abs/$f" ]] || continue
    flag "$f" "present"
done
for d in "${dirs[@]}"; do
    [[ -d "$target_abs/$d" ]] || continue
    n=$(find "$target_abs/$d" -type f 2>/dev/null | wc -l | tr -d ' ')
    flag "$d/" "$n file(s)"
done
# .claude/hooks* — Claude Code hook scripts referenced from settings.
for h in "$target_abs"/.claude/hooks*; do
    [[ -e "$h" ]] || continue
    flag ".claude/$(basename "$h")" "hook script(s)"
done
echo

# Explicit high-signal checks the acceptance test asserts on: an executable
# OpenCode plugin, and a settings.json that wires a hook (either family runs
# these automatically with full privileges).
echo "High-signal executable vectors:"
plugin_hits=0
for p in "$target_abs"/.opencode/plugin "$target_abs"/.opencode/plugins; do
    [[ -d "$p" ]] || continue
    while IFS= read -r pf; do
        [[ -n "$pf" ]] || continue
        plugin_hits=1
        rel="${pf#$target_abs/}"
        if [[ "$mode" == "extern" ]]; then
            print_error "$rel: auto-loaded OpenCode plugin (in-process, full privileges)"
            fail=1
        else
            print_info "$rel: OpenCode plugin (authored here)"
        fi
    done < <(find "$p" -type f 2>/dev/null)
done
settings="$target_abs/.claude/settings.json"
if [[ -f "$settings" ]] && grep -q '"hooks"' "$settings" 2>/dev/null; then
    if [[ "$mode" == "extern" ]]; then
        print_error ".claude/settings.json: defines hooks (auto-run commands, full privileges)"
        fail=1
    else
        print_info ".claude/settings.json: defines hooks (authored here)"
    fi
fi
[[ "$plugin_hits" -eq 0 && ! -f "$settings" ]] && print_ok "No plugin/hook executables found"
echo

if [[ "$fail" -ne 0 ]]; then
    print_error "Preflight: unreviewed agent config in an external checkout"
    echo "  Solution: read each flagged file, then re-run once satisfied"
    exit 1
fi
print_ok "Preflight scan clean ($mode)"
