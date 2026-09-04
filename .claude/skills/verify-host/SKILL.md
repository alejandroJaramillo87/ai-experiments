---
name: verify-host
description: The handshake for verifying anything that needs the workstation — how to state acceptance criteria before running, hand the human an exact command block, and record the reported output as evidence. Use whenever a claim depends on the GPU, Docker, the running services, /mnt/ai-data, or a full make quality.
---
# Host verification handshake

The protocol for checking a claim that a contained session cannot check itself. The agent
prepares commands and interprets reported output; the human executes. The purpose is to make
the boundary between what was observed and what was assumed impossible to blur.

> Sandboxed execution: agents cannot reach the host, its Docker daemon, the GPU, `nvidia-smi`,
> `/mnt/ai-data`, or the inference services on ports 8001-8003 and 3000. The Rust and Python
> gates do run in the sandbox, so a claim about either is verifiable there. Every command in
> this skill is executed by the human on the workstation. The agent prepares the exact command
> line, waits for the human to report the output, and interprets it. Human-reported output is
> the authoritative evidence. Never claim to have run a host-side command, and never fabricate
> its output.

This runs in the main conversation, not in a subagent. A subagent has no channel to the human:
it cannot emit a command, pause, and read the reply. Delegating host verification to one leaves
it able to report failure or to invent output, and inventing output is the single thing
`/AGENTS.md` forbids most emphatically.

## Protocol

### 1. State the acceptance criteria first

Write down what would count as a pass **before** producing any command. Each criterion names an
observable in the output — an exact string, an exit code, a count, a file that exists.

A criterion that cannot fail is not a criterion. "The command runs" is not one; "prints
`[ERROR] refusing to remove the worktree you are standing in` and exits 1" is.

### 2. Confirm the code under test is the code that will run

**This repository's most frequent false pass.** `git worktree add -b` cuts from the current
`HEAD` commit, so uncommitted changes never reach a new worktree. A verification run there
exercises stale source and reports success for a fix that is not present. This produced three
misleading results in a single session: a torch-free `.venv` whose skipped tests looked green, a
worktree whose test failures looked like a regression, and a guard that appeared not to fire
because the old Makefile ran.

Before trusting any result, confirm the tree carries the change:

```bash
git status --short          # uncommitted work will not reach a new worktree
grep -c '<distinctive string from the change>' <path>
```

When the answer is that the change is uncommitted, the fix is to commit, not to add logic.

### 3. Emit one command block

One copy-pasteable block. Comment each command with what it establishes. Prefer the shortest
form that still proves the claim: a check that takes 10 seconds and proves the point beats one
that takes 10 minutes and proves slightly more.

Keep individual lines short. A terminal that hard-wraps a long line corrupts multi-line Python
passed with `python3 -c` and heredocs whose terminator gets indented; both have produced
spurious failures here. Prefer several short commands over one long one, and prefer a written
script file over a heredoc when the logic exceeds a line or two.

Order the commands so that a cheap precondition fails before an expensive step runs.

### 4. Stop

Do not predict the result, do not describe what will probably happen, and do not continue to
the next task on the assumption that it passed. End the turn.

### 5. Record the reported output verbatim

Quote the command and its output exactly as reported, including the parts that are
uninteresting. Paraphrased evidence is not evidence. Where output is long, quote the lines that
bear on the criteria in full rather than summarizing them.

### 6. Judge each criterion separately

Mark every criterion from step 1 as passed or failed against the quoted output. State plainly
when a criterion did not get exercised — a check that never ran is neither a pass nor a
failure, and reporting it as either is the error this protocol exists to prevent.

When a result contradicts the expectation, report the contradiction before explaining it.

## Reading a green result correctly

- **Python suite:** `tests/conftest.py` prints a `[TIER] sandbox:` or `[TIER] full:` line
  beside the pytest summary. `[TIER] sandbox:` means the environment is torch-free and the
  torch-gated tests never executed. Read that line and the skip count before treating green as
  a pass. Provision with `make worktree-<name> TIER=full` for training or GPU work.
- **Rust gate:** `[SKIP]` lines in the cross-language contract tests mean a dependency or the
  parent tree was absent. The exit code is still 0. Read the output.
- **`make quality`:** the rust gate needs a `python3` on `PATH` that can import numpy and gguf.
  `PATH="$PWD/.venv/bin:$PATH" make quality` supplies one.
- **Exit code alone is never sufficient** for any of the three.

## Common host-side checks

```bash
make quality                    # full gate; mandatory before every push
make status                     # service health
make up / make down             # service lifecycle
./scripts/benchmark/benchmark_gpu.sh    # no make wrapper; see the benchmarking skill
nvidia-smi                      # GPU state
docker compose -f docker-compose.agents.yaml -p ai-agents build agent-proxy
```

For performance numbers specifically, follow `.claude/skills/benchmarking/SKILL.md` instead —
it carries the warm-up policy and comparison rules this skill does not.

## Recording the result

When a roadmap task's acceptance check passes, record the exact command and its reported output
verbatim alongside the task. `/AGENTS.md` requires this: a task is not complete until its
acceptance check has been executed and observed to pass, and for host-side checks the human's
report is that observation.
