---
description: Verify a claim that needs the workstation — state acceptance criteria, hand over an exact command block, stop, then judge the reported output
argument-hint: <what to verify>
allowed-tools: Read, Grep, Glob, Bash
---
Verify: $ARGUMENTS

Follow `.claude/skills/verify-host/SKILL.md`. In short:

1. State the acceptance criteria before writing any command. Each one names an observable in
   the output — an exact string, an exit code, a count.
2. Confirm the code under test is the code that will run. A worktree cuts from the `HEAD`
   commit, so uncommitted changes do not reach it and a run there proves nothing.
3. Emit one copy-pasteable command block, commented, shortest form that still proves the claim.
   Keep lines short; a wrapped line corrupts multi-line `python3 -c` and heredocs.
4. Stop. Do not predict the outcome or proceed as though it passed.
5. Record the reported command and output verbatim.
6. Judge each criterion separately against that output, and say plainly when a check did not
   run.

Read the repo-local context first: `/AGENTS.md` for the execution-environment rules, and the
skill for how to read a green result that is actually partial (skip counts, `[SKIP]` lines).
