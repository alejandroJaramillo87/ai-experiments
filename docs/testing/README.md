# Testing documentation

How the repository is tested, and how the test tooling behaves differently depending on where
it runs.

The suite itself is at [../../tests/](../../tests/), whose README owns the tiers, the markers,
and what each test module covers. This directory documents the gate that runs it.

## Documents

| Document | Description |
|----------|-------------|
| [quality-gate.md](quality-gate.md) | What `make quality` runs, and how to invoke the same tools inside an agent sandbox |

`make quality` is mandatory before every push. Read
[quality-gate.md](quality-gate.md) before running it rather than after: the gate behaves
differently on the workstation than in an agent sandbox, and a green sandbox run is the weaker
result. The document also records why repairing the environment with a bare `uv sync` is
destructive here, which is the single most expensive mistake available in this repository.

## Frozen records

| Location | Contents |
|----------|----------|
| [investigations/](investigations/) | [test_coverage_investigation.md](investigations/test_coverage_investigation.md) — an audit of the training and interp suites at 2026-07-16, commit 8316e5e |

That audit is a frozen record: it describes coverage as it stood on its date and is never
updated. Read it for what was deliberately left untested and why, not for the current state of
the suite.

## Related documentation

- [../../tests/README.md](../../tests/README.md) — the suite, its tiers, and its markers
- [../../rust-harness/README.md](../../rust-harness/README.md) — the Rust workspace runs its
  own gate, `make -C rust-harness quality`
- [../agents/README.md](../agents/README.md) — the sandbox the asymmetry above comes from
- [../verification/](../verification/) — host-side acceptance records, which are evidence
  rather than tests
