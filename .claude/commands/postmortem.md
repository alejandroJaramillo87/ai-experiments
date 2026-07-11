---
description: Record a research-sprint postmortem (tried / learned / continue-or-pivot) and append it to the research log
argument-hint: [sprint-name]
allowed-tools: Read, Edit, Write, Bash(date:*)
---
Write a postmortem entry for the research sprint `$ARGUMENTS`. This is structured writing,
not open-ended reasoning — keep it short and honest.

Prompt for and capture:

1. What was tried — the concrete approaches, in order.
2. What was learned — per unit time. Which approaches returned signal for the effort spent
   and which did not.
3. Continue or pivot — decide and justify in one line. **Pivot is the default**: continue
   only if there is specific evidence the current direction is still the highest-value use
   of time.

Cite the session that produced the work so the entry is traceable: the export from
`make agent-export AGENT=<tool> ID=<session-id>` (written under `logs/sessions/`), or for an
interactive Claude Code session its transcript under `~/.claude/projects/`. Include the id.

Append the entry to `docs/research-log.md` (create the file with an `# Research log` header
if it does not exist) as a new dated section, newest at the top. Do not rewrite prior
entries. Follow `.styles/DOCUMENTATION_STYLE.md`.
