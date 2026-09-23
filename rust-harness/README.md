# rust-harness

The Rust workspace: a single-binary local coding harness with the Gemma 4 inference
runtime linked in as a library crate, and the correctness ladder that holds that runtime
to `transformers`. The harness adapts the agent architecture of the
[claude-code-from-scratch](https://github.com/FareedKhan-dev/claude-code-from-scratch)
tutorial; the runtime serves the Gemma 4 E-series from Hugging Face checkpoints, E4B as
the model it is tuned and benchmarked against and E2B verified beside it, and gives the
agent loop direct ownership of the KV cache across turns.

The harness is an instrumented inference substrate for the training and interpretability
tiers, with the coding-agent loop as its native workload. It does not compete with the
llama.cpp and vLLM services on speed or model coverage; its value is the owned seams
those services cannot offer (in-band activation capture, steering, probe monitoring, a
flight recorder, deterministic KV restore), which make agent runs reproducible. This
directory is a permanent part of the repository, not a staging area for extraction, so
the artifact seams it shares with the Python tiers are tested across the two trees.

## Layout

`Cargo.toml` is the workspace: three members, one lockfile, the toolchain pinned in
`rust-toolchain.toml` for the workstation and the agent sandbox alike.

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [fixtures/](fixtures/README.md) | The committed inputs the workspace's tests load, and the generators that wrote them. |
| [harness/](harness/README.md) | The bin crate: the coding agent itself, wired over the `runtime` library. |
| [obs/](obs/README.md) | The run-level observability records, written in the same JSONL shapes as the Python `common/obs` package so the shared catalog, retention pruner and dashboard read a harness run unchanged. |
| [runtime/](runtime/README.md) | The inference engine as a library crate: the checkpoint reader, the tokenizer, the KV cache, the sampler, grammar-constrained decoding, sessions, and the `Model` trait that separates all of that from the forward pass. |

<!-- inventory:dirs:end -->

## Entry points

```bash
make -C rust-harness quality                    # fmt, clippy, lockfile, rustdoc, cargo test
make -C rust-harness test-e4b HARNESS_E4B_MODEL=<dir>   # the checks that need the real checkpoint
make -C rust-harness ladder SIZE=quick MODEL=<dir>      # the correctness ladder, workstation only
make -C rust-harness clean                      # remove run outputs: sessions, profiles, ladder references
cargo run --release -p harness -- --help        # the harness's own flags, from the parser's table
```

The parent `make quality` chains the first; `make quality-full HW=full` climbs the
ladder through `tests/hardware/test_ladder.py`. `make help` at the repository root
lists the container targets (`harness-build`, `harness-run`, `harness-serve`).

## Traps

- **Read the gate's output for `[SKIP]` lines, not only its exit code.** A
  cross-language test that finds no interpreter able to import its Python side skips,
  and a skip is a pass. `cargo test` runs with `--nocapture` for exactly this reason.
- **Which Python runs the contract tests is resolved, not assumed.** `HARNESS_PYTHON`
  overrides; otherwise `.venv/bin/python3` at the repository root, then `python3` on
  `PATH`, the first that imports what the test names (`runtime/src/pyenv.rs`).
- **`test-e4b` runs one test at a time.** Each loads the real checkpoint at roughly
  14 GiB resident; cargo's default parallelism would hold several copies at once.
- **The ladder is not in `make quality`.** It loads the real checkpoint and needs the
  workstation; the offline half, Tier T against a tiny `transformers`-authored fixture,
  is in `cargo test`.

## Where the real docs live

- [../docs/rust-harness/README.md](../docs/rust-harness/README.md) — the design, the
  roadmap, and every page below
- [../docs/rust-harness/testing.md](../docs/rust-harness/testing.md) — how the workspace
  is tested: the tiers, the gate, the contracts
- [../docs/rust-harness/correctness-ladder.md](../docs/rust-harness/correctness-ladder.md) —
  the seven rungs against `transformers`, and the perplexity gate
- [../docs/rust-harness/command-line.md](../docs/rust-harness/command-line.md) — the
  flags, `--status`, the exit codes, and what a run reports
- [../docs/rust-harness/runtime-status.md](../docs/rust-harness/runtime-status.md) —
  what is implemented and what the workstation has verified
- [../docs/verification/rust-harness/rust-harness.md](../docs/verification/rust-harness/rust-harness.md) —
  the workstation runbook, entry by entry
- [../site/README.md](../site/README.md) — the user documentation: building, every
  flag, the run records and the artifact formats
- [../schemas/README.md](../schemas/README.md) — the contracts shared with the Python tiers
