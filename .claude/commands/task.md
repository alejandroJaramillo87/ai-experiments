---
description: Drive one roadmap task through implement -> acceptance-check -> skeptic, then stop for review
argument-hint: <roadmap-file> <task-id>
allowed-tools: Read, Edit, Write, Bash, Glob, Grep
---
Roadmap file: $1
Task id: $2

@$1

(If the roadmap content above did not expand, Read `$1` yourself before continuing.)

Execute this task as one bounded, self-verifying session. Do the stages in order and do not
skip ahead:

1. Restate — locate task `$2` in the roadmap and restate its files, steps, and acceptance
   criteria exactly as written. If the task is ambiguous or already done, say so and stop.
2. Implement — make the change described by task `$2`, touching only the files it names.
3. Acceptance-check — run the task's acceptance check **verbatim** and show the full output.
   If it fails, fix and re-run until it passes; do not proceed on a failing check.
4. Skeptic — delegate to the skeptic subagent to independently verify the completed task.
   Treat `VERDICT: REFUTED` as the task NOT done: address the specific gap it names and
   re-run stages 3–4.
5. Stop for review — summarize what changed and the acceptance + skeptic evidence, then
   **stop without committing**. The human reviews and commits.
