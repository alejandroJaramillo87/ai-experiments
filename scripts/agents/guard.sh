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

# Key material, as one extended regex shared by both branches below. A single list
# because two copies of a security policy is how the two copies diverge -- the file
# branch matches it against the path, the bash branch against the whole command line.
# The settings.json Read(~/.ssh/**) and Read(~/.aws/**) denies cover the direct form
# only; this covers the indirection they cannot see, which is the same argument that
# put the .env rule here rather than in the permission matcher alone.
SECRET_RE='(/\.ssh/|/\.aws/(credentials|config)|id_rsa|id_ed25519|\.(pem|p12|key)([^a-zA-Z0-9]|$))'

# stdin may be empty (defensive); treat as allow.
input="$(cat 2>/dev/null || true)"
[ -n "$input" ] || exit 0

# jq is required; if absent, fail open is unsafe, so fail closed with a clear message.
if ! command -v jq >/dev/null 2>&1; then
    printf '[ERROR] guard.sh: jq not found; cannot evaluate tool call\n' >&2
    exit 2
fi

# Malformed payloads must fail closed: jq errors exit nonzero-but-not-2 under
# set -e, which the hook runner treats as allow. Validate before extracting.
if ! printf '%s' "$input" | jq empty >/dev/null 2>&1; then
    printf '[BLOCKED] guard.sh: unparseable hook payload\n' >&2
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
            *.env|*.env.*) block "access to environment file denied: $file_path" ;;
        esac
        # Private keys and cloud credentials: never read or modify.
        if printf '%s' "$file_path" | grep -Eq "$SECRET_RE"; then
            block "access to key material denied: $file_path"
        fi
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
        # Same for key material. Blocks reading it and searching for it alike; a
        # command that names a private key has no benign form worth the exception.
        if printf '%s' "$command" | grep -Eq "$SECRET_RE"; then
            block "bash command references key material: $command"
        fi
        # Catch writes into the models dir (redirect, tee, cp/mv, dd of=, rm,
        # plus downloaders and file mutators: curl/wget/rsync/install/touch/truncate).
        if printf '%s' "$command" | grep -Eq "$MODELS_DIR"; then
            if printf '%s' "$command" | grep -Eq '(>|>>|\btee\b|\bcp\b|\bmv\b|\brm\b|\bdd\b|of=|\bcurl\b|\bwget\b|\brsync\b|\binstall\b|\btouch\b|\btruncate\b)'; then
                block "bash command may write under protected models dir: $command"
            fi
        fi
        ;;
esac

exit 0
