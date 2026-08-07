# Testing documentation

How the repository is tested, and what the test tooling covers in each environment it runs in.

The suite itself is at [../../tests/](../../tests/), whose README owns the tiers, the markers,
and what each test module covers. This directory documents the gate that runs it.

## Documents

| Document | Description |
|----------|-------------|
| [quality-gate.md](quality-gate.md) | What `make quality` runs, and what a sandbox run does not cover |

`make quality` is mandatory before every push, and it is one command: the recipe in
`make/quality.mk` branches internally on the `AI_SANDBOX` variable the sandbox image sets, so
the workstation and a contained agent sandbox run the same pinned tools. Only `uv lock --check`
is out of reach in a sandbox, and it announces itself with a `[SKIP]` line. Read
[quality-gate.md](quality-gate.md) for the step list and for what that one skipped step leaves
uncovered.

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
- [../agents/README.md](../agents/README.md) — the sandbox, and how it is provisioned from `uv.lock`
- [../verification/](../verification/) — host-side acceptance records, which are evidence
  rather than tests
