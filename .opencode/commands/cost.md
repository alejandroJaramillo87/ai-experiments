---
description: Show current agent/session spend (OpenCode only; Claude Code uses built-in /usage)
allowed-tools: Bash(make agent-usage)
agent: build
---
Session spend:

!`make agent-usage`

This command is OpenCode-only: it forks from the shared command set deliberately because
Claude Code already provides spend via the built-in `/usage`. Report the figures above and
flag anything unexpected.
