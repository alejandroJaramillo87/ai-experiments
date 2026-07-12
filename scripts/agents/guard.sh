#!/usr/bin/env bash
# guard.sh - single-source tool-call guardrail for both OpenCode and Claude Code.
#
# Reads Claude-hook-shaped JSON on stdin: {"tool_name": "...", "tool_input": {...}}.
# Exits 2 (with a reason on stderr) to BLOCK the call; exits 0 to allow.
#
# This is the belt-and-suspenders backstop over the Phase 1 permissions: permissions gate
# tools by pattern, but the pattern matcher misses bash indirection (e.g. cat $(echo .env)).
# The OpenCode guard.ts plugin normalizes its args into the same shape and pipes them here,
# so both tools enforce the identical policy from one script.

set -euo pipefail

MODELS_DIR="/mnt/ai-data/models"

# stdin may be empty (defensive); treat as allow.
input="$(cat 2>/dev/null || true)"
[ -n "$input" ] || exit 0

# jq is required; if absent, fail open is unsafe, so fail closed with a clear message.
if ! command -v jq >/dev/null 2>&1; then
    printf '[ERROR] guard.sh: jq not found; cannot evaluate tool call\n' >&2
    exit 2
fi

tool="$(printf '%s' "$input" | jq -r '.tool_name // empty')"
file_path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')"
command="$(printf '%s' "$input" | jq -r '.tool_input.command // empty')"

block() {
    printf '[BLOCKED] guard.sh: %s\n' "$1" >&2
    exit 2
}

case "$tool" in
    Read|Edit|Write|read|edit|write)
        # Environment files: never read or modify.
        case "$file_path" in
            *.env|*.env.*|*/.env|*/.env.*) block "access to environment file denied: $file_path" ;;
        esac
        # Model weights: read is fine, but never write/edit.
        case "$tool" in
            Edit|Write|edit|write)
                case "$file_path" in
                    "$MODELS_DIR"/*|"$MODELS_DIR") block "write under protected models dir denied: $file_path" ;;
                esac
                ;;
        esac
        ;;
    Bash|bash)
        # Catch .env access through any indirection the permission matcher misses.
        if printf '%s' "$command" | grep -Eq '\.env([^a-zA-Z0-9_/]|$)'; then
            block "bash command references an environment file: $command"
        fi
        # Catch writes into the models dir (redirect, tee, cp/mv, dd of=, rm).
        if printf '%s' "$command" | grep -Eq "$MODELS_DIR"; then
            if printf '%s' "$command" | grep -Eq '(>|>>|\btee\b|\bcp\b|\bmv\b|\brm\b|\bdd\b|of=)'; then
                block "bash command may write under protected models dir: $command"
            fi
        fi
        ;;
esac

exit 0
