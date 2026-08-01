# Agent tooling tests

Fixtures and tests for the agent guardrail and workflow scripts in `scripts/agents/`.
The suite has two layers:

1. **Deterministic fixture tests** — pytest, run by `make test`. No credentials, no
   network, no LLM calls; each test drives a real script with a canned input and
   asserts on exit code and output.
2. **LLM acceptance tests** — `make agent-role-tests`, one case per QA subagent
   role. They spend tokens and are nondeterministic by design, so they are kept out
   of pytest and run on demand. `make agent-skeptic-test` runs the skeptic case
   alone; `make agent-role-tests ROLE=bug-hunter` runs any single role. The case
   table lives in `scripts/agents/role-acceptance.sh`.

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
| `broken_completed_task.md` | `make agent-role-tests ROLE=skeptic` | the skeptic subagent refutes a false completion claim |
| `planted_defect/` | `ROLE=bug-hunter`, `ROLE=remediator` | an off-by-one that a green test suite does not catch is found, and later fixed, without the finder touching the file |
| `unverified_ledger.md` | `ROLE=triage-agent` | a finding with zero recorded verdicts is marked unverified rather than confirmed |
| `noisy_speedup_claim.md` | `ROLE=perf-auditor` | a 5.1% delta inside a 27% spread is reported as not resolvable |
| `drifted_doc.md` | `ROLE=doc-auditor` | documented claims that contradict the tree are reported as drift, not rewritten |

`preflight_fixture/README.md` documents the planted vectors in detail.

## Running the deterministic layer

```bash
$ make test
```

## QA role acceptance tests

Every subagent role is asserted against a fixture built so the wrong answer is the
plausible one. The passing outcome is the role refusing the plausible answer.

```bash
$ make agent-role-tests                     # every role
$ make agent-role-tests ROLE=perf-auditor   # one role
```

`planted_defect/` is the only fixture with executable code. `window_start` is off by
one — it admits `window + 1` positions where `window_length` allows `window` — and
`suite.py` passes anyway, because both assertions land on the clamped-at-zero cases
where the correct and incorrect bounds agree. The file is named `suite.py` rather
than `test_*.py` so `make test` does not collect it; run it explicitly with
`uv run pytest tests/agents/planted_defect/suite.py`.

The bug-hunter case asserts two things: that the finder names `window_start`, and
that it left the file alone. A finder that fixes what it finds fails the test even
when the fix is correct, because the finding never reaches a verifier. The
remediator case then checks the behavior directly — `window_start(10, 4)` must
return 7 — and the runner restores the planted defect afterwards so the next run
starts from the same state.

The perf-auditor case is the one worth reading before writing a performance claim:
its fixture reports a 5.1% speedup from a single pair of runs, and includes, further
down, ten runs of the unchanged build spanning 0.1497 s to 0.1902 s. The claimed
effect is a fifth of the noise. The only passing answer is `not resolvable`.

`drifted_doc.md` plants several contradictions; the graded one is the llama-cpu port,
documented as 8000 against an actual 8001. Its agent-roles paragraph carries three
drifts at once — a stale count, a claim that the files are symlinked when
`make agent-parity` requires real files on both sides, and a list naming `explorer`,
which was deleted. Do not repair the fixture when the roster changes: staleness is
what it tests.

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
