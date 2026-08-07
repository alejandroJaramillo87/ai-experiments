---
description: Drive one roadmap task through implement -> acceptance-check -> skeptic, then stop for review
argument-hint: <roadmap-file> <task-id>
allowed-tools: Read, Edit, Write, Bash, Glob, Grep, Task
---
Roadmap file: $1
Task id: $2

Read `$1` before continuing. (A leading `@$1` would not work here: `@path` includes are
resolved at parse time, before `$1` is bound to an argument.)

Execute this task as one bounded, self-verifying session. Do the stages in order and do not
skip ahead:

1. Restate — locate task `$2` in the roadmap and restate its files, steps, and acceptance
   criteria exactly as written. If the task is ambiguous or already done, say so and stop.
2. Implement — make the change described by task `$2`, touching only the files it names.
3. Acceptance-check — if the check is repo-local (no GPU, service, or host Docker), run
   it and show the full output. Run it verbatim; `make quality`, `make test`, and `pytest`
   behave identically in a sandbox and on the workstation, so no substitution is needed. Say
   so if you substitute anyway. If it is host-side, print the exact command,
   stop, and wait for the human to run it and report the output; record the reported
   command and output verbatim. If the check fails, fix and repeat; do not proceed on a
   failing check and never mark the task done without observed or human-reported passing
   output.
4. Skeptic — delegate to the skeptic subagent to independently verify the completed task.
   Treat `VERDICT: REFUTED` as the task NOT done: address the specific gap it names and
   re-run stages 3–4.
5. Stop for review — summarize what changed and the acceptance + skeptic evidence, then
   **stop without committing**. The human reviews and commits.
