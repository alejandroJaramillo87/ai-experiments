# Agent tooling tests

Fixtures and tests for the agent guardrail and workflow scripts in `scripts/agents/`.
The suite has two layers:

1. **Deterministic fixture tests** — pytest, run by `make test`. No credentials, no
   network, no LLM calls; each test drives a real script with a canned input and
   asserts on exit code and output.
2. **LLM acceptance test** — `make agent-skeptic-test`. Spends tokens and is
   nondeterministic by design, so it is kept out of pytest and run on demand.

The sibling `tests/utils/` suite covers the host-environment reporters in
`scripts/utils/` with output-contract tests.

`test_worktree_make.py` is the exception to the scripts-only scope: it drives the
Makefile worktree targets (`worktree-%`, `worktree-clean`, `worktree-clean-%`) directly.
Each test runs `make -f <repo>/Makefile` inside a throwaway git repository under
`tmp_path`, so worktrees are created and removed in temp siblings and the real repository
is never touched. It covers argument forms (name, `NAME=`, `task/<name>` branch, worktree
path, suffix target), exit codes on failure, and creation idempotence.

`test_sandbox_seed.py` covers the `agent-sandbox` target's Claude Code config seeding
(`~/.claude.json` and `~/.claude/settings.json`) with a stubbed `docker` on PATH that
prints the container command instead of running it. Never exercise this target with
`make -n`: the recipe contains `$(MAKE)`, which GNU make executes even under `-n`.

## Fixture index

| Fixture | Consumer | Proves |
|---------|----------|--------|
| `guard_indirection_input.json` | `test_guard.py` → `scripts/agents/guard.sh` | `.env` access through bash indirection (`cat $(echo .env)`) is blocked |
| `runpod_warm_fixture.json` | `test_runpod_idle.py` → `scripts/agents/runpod-idle-check.sh` | a warm worker with active spend triggers `[WARN]` |
| `runpod_idle_fixture.json` | `test_runpod_idle.py` → `scripts/agents/runpod-idle-check.sh` | an idle endpoint reports no spend, no warning |
| `runpod_fractional_fixture.json` | `test_runpod_idle.py` → `scripts/agents/runpod-idle-check.sh` | a fractional worker count (1.0) still warns instead of a false all-clear |
| `preflight_fixture/` | `test_preflight.py` and `make agent-preflight PREFLIGHT_DIR=...` → `scripts/agents/preflight-scan.sh` | planted supply-chain vectors in an external checkout are flagged, exit non-zero |
| `broken_completed_task.md` | `make agent-skeptic-test` | the skeptic subagent refutes a false completion claim |

`preflight_fixture/README.md` documents the planted vectors in detail.

## Running the deterministic layer

```bash
$ make test
```

## Skeptic acceptance test

`broken_completed_task.md` is a task marked COMPLETED that does not hold up:

- The claimed `safe_div` has no zero-guard — `safe_div(1, 0)` raises `ZeroDivisionError`,
  contradicting the comment and the "handled" claim.
- The claimed test never exercises the divisor-zero case despite saying it is "covered".
- "All tests pass" is asserted with no runnable command or evidence.

The skeptic must catch at least one of these and end its response with
`VERDICT: REFUTED` — the refutation is the passing outcome. A `VERDICT: CONFIRMED`
here is a test failure.

### Invocation

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

### Automated check

`make agent-skeptic-test` runs the Claude invocation and asserts the output contains
`VERDICT: REFUTED`. It reports `[SKIP]` when the `claude` CLI is unavailable. The
OpenCode side is verified manually until its model IDs are pinned.
