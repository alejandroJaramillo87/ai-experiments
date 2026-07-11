# Skeptic acceptance test

`broken_completed_task.md` is a task marked COMPLETED that does not hold up:

- The claimed `safe_div` has no zero-guard — `safe_div(1, 0)` raises `ZeroDivisionError`,
  contradicting the comment and the "handled" claim.
- The claimed test never exercises the divisor-zero case despite saying it is "covered".
- "All tests pass" is asserted with no runnable command or evidence.

The skeptic must catch at least one of these and end its response with
`VERDICT: REFUTED`. A `VERDICT: CONFIRMED` here is a test failure.

## Invocation

Claude Code:

```bash
claude -p "Use the skeptic subagent to verify the completed task in \
tests/agents/broken_completed_task.md" | tail -n 20
```

OpenCode (once the frontier `model:` is set in `.opencode/agents/skeptic.md`):

```bash
opencode run --auto "Use the skeptic agent to verify the completed task in \
tests/agents/broken_completed_task.md"
```

## Automated check

`make agent-skeptic-test` runs the Claude invocation and asserts the output contains
`REFUTED`. It reports `[SKIP]` when the `claude` CLI is unavailable. The OpenCode side is
verified manually until its model IDs are pinned.
