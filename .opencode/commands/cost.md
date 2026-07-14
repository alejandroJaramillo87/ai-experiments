---
description: Show current agent/session spend (OpenCode only; Claude Code uses built-in /usage)
allowed-tools: Bash(make agent-usage)
agent: build
---
Session spend:

!`make agent-usage`

This command is OpenCode-only: it forks from the shared command set deliberately because
Claude Code already provides spend via the built-in `/usage`. Report the figures above and
flag anything unexpected. If the injection above is empty (`make agent-usage` is host-side
and does not run in a sandboxed session), ask the human to run it and paste the report.
