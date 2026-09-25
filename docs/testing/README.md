# Testing

How this repository is tested, and how the documentation is kept true: the commands, what
each one proves, where each runs, and the strategy behind them. This page is the overview;
the pages below own the detail.

## The three commands

There is no continuous integration and no git hook. Every check runs from `make`, on the
workstation or in a contained agent sandbox, and `make quality` is mandatory before every
push by convention, for humans and agents alike (`AGENTS.md`). The three commands cover
three different amounts of ground, and the cheapest is the one everyone reaches for:

| Command | Runs | Proves | Where |
|---------|------|--------|-------|
| `make test` | the pytest suite with the hardware tiers deselected | the code agrees with its tests, in whichever environment tier the `[TIER]` line reports | workstation and sandbox |
| `make quality` | format, lint, types, the lockfile, fourteen documentation gates, shellcheck, the rust gate, the site build; never pytest | the code builds and the documentation resolves against the tree | workstation and sandbox, with one `[SKIP]` in the sandbox |
| `make quality-full` | `quality`, then `test`, then the `hw_smoke` tier; `HW=full` and `HW=e4b` add the deeper tiers | the hardware does what the sandbox halves assumed | workstation only; refuses elsewhere |

Neither gate measures whether the inference engine computes the right numbers. That is
the correctness ladder's job, and it sits inside `quality-full HW=full` rather than in the
push gate ([hardware-tiers.md](hardware-tiers.md)).

## The layers

Each layer has one command and one owner. Read the owner, not this table.

| Layer | Command | What it holds | Owner |
|-------|---------|---------------|-------|
| Format, lint, types | `ruff format --check`, `ruff check`, `mypy` | the three first-party packages, configured in `pyproject.toml` | [quality-gate.md](quality-gate.md) |
| Lockfile and pins | `uv lock --check`, `check_group_pin_parity.py` | `uv.lock` matches `pyproject.toml`; a package pinned twice is pinned once | [quality-gate.md](quality-gate.md) |
| Documentation gates | `scripts/docs/check_*.py` | every path, target, count, id, entry and generated table in the docs agrees with the tree | [doc-gates.md](doc-gates.md) |
| Shell | `shellcheck --severity=warning` | every `*.sh` in the tree, enumerated with `find` | [quality-gate.md](quality-gate.md) |
| Rust gate | `make -C rust-harness quality` | fmt, clippy, the lockfile, rustdoc and `cargo test`, including the cross-language contract tests | [../rust-harness/testing.md](../rust-harness/testing.md) |
| Site build | `make site-build` | every link and anchor on the documentation site resolves | [quality-gate.md](quality-gate.md) |
| The pytest suite | `make test` | unit and branch tests for `common/`, `training/` and `interp/`, the scripts driven as subprocesses, the model transcripts, the agent tooling, the documentation gates | [test-suite.md](test-suite.md) |
| Hardware tiers | `make quality-full`, `HW=full`, `HW=e4b`; one subsystem at a time with `make hardware-tests SUITE=<subsystem>` | the workstation halves of the verification entries, on the real GPU, RAM, disk and checkpoints | [hardware-tiers.md](hardware-tiers.md) |
| Correctness ladder | `make -C rust-harness ladder` | the harness's numbers against `transformers`, rung by rung | [hardware-tiers.md](hardware-tiers.md) |
| Verification runbooks | executed by hand, pasted verbatim | evidence, not tests: what the workstation was observed to do, with a ledger of the bugs it surfaced | [../verification/README.md](../verification/README.md) |
| Agent role acceptance | `make agent-role-tests` | each QA subagent role refuses the plausible wrong answer; spends tokens, outside pytest | [../../tests/agents/README.md](../../tests/agents/README.md) |

## Principles that recur

**A gate that enumerates nothing fails.** Shellcheck, every documentation gate, the torch
probe in `quality-full` and the `--nocapture` decision in the rust gate all follow from
one observation: a gate that silently checked nothing is indistinguishable from one that
passed.

**Read the output, not only the exit code.** The suite prints `[TIER] sandbox` or
`[TIER] full` beside its summary; the rust gate prints `[SKIP]` where a cross-language
test found no Python to run against; `quality` prints `[SKIP]` for the one step a sandbox
cannot run. A green run with skips in it is the weaker result.

**One variable per question.** `HW=smoke|full|e4b` selects which hardware tier
`quality-full` runs; `TIER=sandbox|full` selects how `make worktree-<name>` provisions a
venv; `SIZE=quick|standard|deep` selects how much work the ladder does. They look alike
and mean different things, and each Makefile explains why a second `?=` would be resolved
silently by include order.

**The documentation is gated like the code.** A fact is stated once, by its owner; a
rename adds a retired name; a table that lists files is generated from them. The strategy
is written down in [doc-gates.md](doc-gates.md).

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [doc-gates.md](doc-gates.md) | How the documentation stays true, and the gates that make it so. |
| [hardware-tiers.md](hardware-tiers.md) | The workstation's share of the gate. |
| [quality-gate.md](quality-gate.md) | `make quality` is the pre-push gate, mandatory for humans and agents alike. |
| [test-suite.md](test-suite.md) | The pytest suite as `pyproject.toml` configures it. |

<!-- inventory:pages:end -->

## Frozen records

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [investigations/](investigations/README.md) | The frozen audit of the sandbox test suite: what the training and interp packages were covered by, and what was deliberately left untested, on the day it was written. |

<!-- inventory:dirs:end -->

The audit there describes coverage as it stood on its date and is never updated. Read it
for what was deliberately left untested and why, not for the current state of the suite.

## Related documentation

- [../../tests/README.md](../../tests/README.md) — the suite's layout, generated from its
  directories
- [../../scripts/docs/README.md](../../scripts/docs/README.md) — the documentation gates
  as scripts: conventions and what to edit when
- [../agents/README.md](../agents/README.md) — the sandbox, and how it is provisioned from
  `uv.lock`
