# Testing the rust harness

How the workspace under `rust-harness/` is tested, tier by tier: the unit tests inside
each source file, the integration tests that spawn the binaries, the correctness ladder
that holds the engine to `transformers`, the conformance invariants every backend owes,
the cross-language contract tests that hand written artifacts to the real Python readers,
and the gate that runs the offline part of all of it on every push. This is the entry
point; the pages it links own the detail, and the generated tables in each crate's
README list every test file with its count.

## Table of Contents

<!-- toc:start -->

1. [The tiers](#the-tiers)
2. [What each command runs](#what-each-command-runs)
3. [Unit tests](#unit-tests)
4. [Integration tests](#integration-tests)
5. [The offline correctness tier](#the-offline-correctness-tier)
6. [The E4B tier](#the-e4b-tier)
7. [Read the output, not the exit code](#read-the-output-not-the-exit-code)
8. [Which Python runs the contract tests](#which-python-runs-the-contract-tests)
9. [The quality gate](#the-quality-gate)
10. [What the pytest suite adds](#what-the-pytest-suite-adds)
11. [Related documentation](#related-documentation)

<!-- toc:end -->

## The tiers

Testing is test-first and tiered, and each tier answers a different question:

| Tier | Where | Needs | Answers |
|------|-------|-------|---------|
| Unit tests | `#[cfg(test)]` modules inside `src/` | nothing | the module does what its banner says |
| Mock integration | `harness/tests/`, `runtime/tests/` over `MockModel` | nothing | the stack above the forward pass composes: loop, tools, grammar, sessions, KV bookkeeping, sampler |
| Binary surface | the same directories, spawning the built binaries | the committed fixtures | what a person or the interp client actually gets: flags, exit codes, streams, run records |
| Cross-language contracts | `*_contract.rs`, `probe_identity.rs`, `status_python_parity.rs`, `npz.rs` | a Python interpreter that imports the reader | an artifact this workspace writes is read correctly by the Python that consumes it |
| Offline correctness | `runtime/tests/ladder_tiny.rs`, `cpu_reference.rs`, `long_context.rs` | the committed fixtures | the engine agrees with `transformers` on a tiny real Gemma 4, with an independent numpy pass on synthetic weights, and with itself across batch compositions |
| The ladder at Tier E | `make -C rust-harness ladder` | the workstation and the real checkpoint | the engine agrees with `transformers` on the released weights, seven rungs deep |
| The E4B checks | `make -C rust-harness test-e4b` | the real checkpoint | the tests whose question is the released checkpoint itself |
| The workstation runbook | `docs/verification/rust-harness/rust-harness.md` | the workstation | what the hardware was observed to do, with pasted output as evidence |

Every tier but the last three runs in the agent sandbox, and every one of those runs in
`make quality`.

## What each command runs

```bash
make -C rust-harness quality        # fmt, clippy, lockfile, rustdoc, cargo test --workspace
make -C rust-harness test           # cargo test --workspace -- --nocapture
make -C rust-harness test-e4b HARNESS_E4B_MODEL=<dir>      # one test at a time, real weights
make -C rust-harness ladder SIZE=quick MODEL=<dir>         # the seven rungs, generating any missing reference
make -C rust-harness ladder-reference SIZE=quick MODEL=<dir>   # rewrite every reference artifact
make -C rust-harness replay-check RECORDING=<file>         # a --record file replays as a MockModel script
make -C rust-harness trace-check TRACE=<file>              # a --trace-chrome file has the documented span shape
```

`make -C rust-harness clean` removes what these leave in the workspace: the session
files a run from that directory writes, the profiling outputs, the gdb history and every
generated reference directory, so the next `make ladder` regenerates from the checkpoint.
The root `make harness-clean` chains it and also removes the container's transcripts, the
`rust-harness` subdirectory of the repository's `sessions` directory.

The parent `make quality` chains the first when the directory exists
([../testing/quality-gate.md](../testing/quality-gate.md)); `make quality-full HW=full`
climbs the ladder through `tests/hardware/test_ladder.py` and `HW=e4b` runs `test-e4b`
([../testing/hardware-tiers.md](../testing/hardware-tiers.md)). `test-e4b` passes
`--test-threads=1` because each of its tests loads the checkpoint at roughly 14 GiB
resident and cargo would otherwise hold several copies at once; its tests skip through
`runtime::testlog::skip` when `HARNESS_E4B_MODEL` is unset, and the parent gate makes
the unset case an error rather than a vacuous pass. `ladder-preflight` refuses inside
the sandbox (`AI_SANDBOX`), without `MODEL`, and on a `SIZE` outside the three names.

## Unit tests

Every source file that has logic to pin carries a `#[cfg(test)] mod tests` at its foot;
46 of the workspace's source files do, and the generated tables in
[../../rust-harness/harness/README.md](../../rust-harness/harness/README.md),
[../../rust-harness/runtime/README.md](../../rust-harness/runtime/README.md) and
[../../rust-harness/obs/README.md](../../rust-harness/obs/README.md) give the count per
file. The three largest say where the risk is: `runtime/src/model/cpu.rs` (the forward
pass and its kernels, including the fold-width and thread-count invariance of every
matrix-vector product), `runtime/src/ladder.rs` (the verdict logic of every rung, run
against a scripted mock so that a threshold or short-circuit defect is caught without a
model) and `runtime/src/checkpoint.rs` (the safetensors envelope and the config
reconciliation). Two rules from `.styles/RUST_STYLE.md` hold throughout: exact equality
wherever determinism makes it possible, and no test that would print a `[SKIP]` naming
nothing real.

## Integration tests

Each crate's `tests/` directory is listed in its README with one sentence per file.
Three shapes recur:

- **Through the binary.** `cli_surface.rs`, `preflight_surface.rs`, `status_streams.rs`,
  `binary_cli_surface.rs` and their kin spawn the built executable and assert on its
  streams, exit code and the files it wrote, because what is under test is the behaviour a
  user gets, not a function the binary happens to call.
- **Over the scripted model.** `agent_e2e.rs`, `subagents.rs`, `generator_e2e.rs`,
  `fork.rs` and `grammar.rs` drive the stack with `runtime::model::mock::MockModel`, whose
  logits are a script, so a test can assert that the loop made a well-formed tool call or
  that a fork re-prefilled nothing.
- **Against a real reader.** The contract tests below hand an artifact to Python.

## The offline correctness tier

Machinery tests cannot detect a numerically wrong forward pass, and a golden minted from
this engine cannot either. Three offline tests hold the numbers, in increasing strength:

1. `cpu_reference.rs`: the CPU backend against `fixtures/reference_logits.py`, an
   independent numpy forward pass, over the two synthetic checkpoints (`tiny-gemma4`
   declares none of Gemma 4's optional features, `tiny-gemma4-e4b` declares all of
   them). A cross-language check against a second implementation, and a change detector;
   both sides were transcribed from the same documents, so it is not an oracle
   ([fixtures.md](fixtures.md) says what that has cost).
2. `long_context.rs`: the forward pass at lengths where the sliding window engages,
   compared against itself across every batch composition and against the committed
   golden, plus the KV storage-cap arms ([conformance.md](conformance.md)).
3. `ladder_tiny.rs`: the same seven rungs the workstation climbs, against a tiny Gemma 4
   that `transformers` itself constructed and saved (`fixtures/tiny-hf-gemma4/`). The one
   offline test whose reference nobody here wrote, measured in milliseconds
   ([correctness-ladder.md](correctness-ladder.md)).

## The E4B tier

The checks whose question is the released checkpoint rather than the engine's
arithmetic: the shared-KV weight audit, the alignment artifact over the real vocabulary,
compaction past the real 512-token window, the layer diff over 42 layers,
`checkpoint-info --check` on the released config. Each skips with a `[SKIP]` naming
`HARNESS_E4B_MODEL` when the variable is unset. The pytest side of the same tier is
`tests/hardware/` under `hw_e4b`.

## Read the output, not the exit code

A skipped contract test is the one outcome the gate's exit code cannot express: it
passes, so nothing draws the eye, yet the property it exists to check went unchecked.
`runtime/src/testlog.rs` prints every skip as `[SKIP] <subject>: <reason>`, a noun phrase
naming what went unchecked and the dependency that was missing, and the `test` step passes
`--nocapture` so those lines appear at all: cargo hides a passing test's stderr, and a skip
is by definition a pass. `grep '\[SKIP\]'` over the gate's output is the check. A green
run with skips in it is the weaker result, in the sandbox and on the workstation alike.

## Which Python runs the contract tests

Several tests verify an artifact by handing it to the real Python reader rather than to
a second Rust implementation, because a Rust re-read would reproduce a Rust
misunderstanding of the format. The interpreter they spawn is part of the contract, and
the workstation and the sandbox disagree about where the right one lives: on the
workstation `python3` usually leads to a pyenv shim with none of the project's
dependencies while the environment sits in `.venv`; in the sandbox `.venv` is masked and
`/opt/pyenv`, built from `uv.lock`, is first on `PATH`. `runtime/src/pyenv.rs` therefore
probes `$HARNESS_PYTHON`, then `<repo>/.venv/bin/python3`, then `python3` on `PATH`, and
takes the first that can import what the caller names. A missing optional dependency
reports `[SKIP]` and passes; numpy is mandatory in `npz.rs`, because a shard nothing
re-read is an unverified shard. [conformance.md](conformance.md) lists the contracts.

## The quality gate

```bash
make quality        # fmt check, clippy -D warnings, lockfile check,
                      # rustdoc -D warnings, cargo test
```

Five offline steps, cheapest first, mirroring the parent repository's Python
gate, which chains this one. cargo-deny is a deliberate non-step: its advisory
database needs the network.

The cross-language tests hand artifacts to the real Python readers rather than to
a second Rust implementation, because a Rust misunderstanding of a format would
otherwise reproduce itself on both sides. `runtime/src/npz.rs` re-reads a written
shard through `np.load`; `harness/tests/probe_identity.rs` takes its SHA-256 from
`hashlib`; `python_exporter_contract.rs` reads the probe and control-vector artifacts
written by the interp exporters; `obs_contract.rs` parses a run catalog through
`common.obs.catalog.read_runs`.

**Read the output for `[SKIP]` lines rather than trusting the exit code.** The
`test` step passes `--nocapture` for that reason: cargo hides a passing test's
stderr, a skip is by definition a pass, and without it a gate that had silently
stopped exercising the Python side would look identical to one that exercised it.
Tier 3 tests still require the workstation.

The contracts themselves are in `schemas/` at the repository root. The roadmap
and the design documents live in `docs/rust-harness/` there.

## What the pytest suite adds

The parent suite drives the built binaries from Python where the question is the pairing
with the Python tiers or the real hardware: `tests/training/test_export_harness.py`
exports the tiny training fixture through the real verb and hands the checkpoint to
`checkpoint-info` and `harness`; eighteen modules under `tests/hardware/` spawn the
binaries against real weights, and `test_ladder.py` climbs the ladder at `SIZE=quick` and
reads its run record off the catalog. The pairing of each sandbox test with its
workstation half is the table in [../testing/hardware-tiers.md](../testing/hardware-tiers.md).

## Related documentation

- [correctness-ladder.md](correctness-ladder.md) — every rung, its measures, its tolerances, and the perplexity gate
- [conformance.md](conformance.md) — the backend invariants and the cross-language contracts
- [fixtures.md](fixtures.md) — what the tests load, what wrote it, and what the goldens are worth
- [../testing/README.md](../testing/README.md) — the repository's testing strategy, of which this is the Rust layer
- [../verification/rust-harness/rust-harness.md](../verification/rust-harness/rust-harness.md) — the workstation runbook
- [../../.styles/RUST_STYLE.md](../../.styles/RUST_STYLE.md) — the `[SKIP]` contract and the test conventions
