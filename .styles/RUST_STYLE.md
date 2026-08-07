# Rust Style Guide

Standards for the `rust-harness` workspace. Rust's compiler and clippy already enforce most of
what a style guide would otherwise repeat, so this document covers the decisions they cannot
check: when determinism outranks idiom, how errors carry context, where `unsafe` is permitted,
and how tests report.

Governing principles, in priority order:

- **Determinism first.** The workspace produces training and interpretability numerics. A
  transformation that changes results bit-for-bit is a defect, however idiomatic.
- **Independent verification.** A check written in the same language, by the same author, as
  the code under test reproduces its bugs. Cross-language checks are deliberate.
- **Fail loudly, with the value.** An error names the input that caused it. A non-finite float
  is an error, never a silently propagated value.
- Professional and technically precise; no decoration, marketing language, or emojis.

## Determinism outranks idiom

The workspace rejects `clippy::pedantic` and `clippy::nursery` on purpose. Record the reason
before proposing to enable either:

- `clippy::suboptimal_flops` rewrites `a * b + c` into `a.mul_add(b, c)`. Fused multiply-add
  keeps a wider intermediate, so the result differs in the last bits from the separate
  operations. Bit-identical replay is a stated guarantee of this workspace, and FMA breaks it.
- `clippy::float_cmp` fires on `==` between floats. Several tests compare floats exactly
  because exactness is the property under test, not an oversight.

Consequences for everyday code:

- Do not introduce `mul_add`, and do not accept a suggestion that does.
- Do not reorder floating-point arithmetic for readability. Summation order is part of the
  result.
- Do not add a fast path that changes the arithmetic. A fast path must produce identical bits
  or it is a different function with a different name.

## Lints and configuration

Lint policy lives in the workspace manifest and nowhere else:

```toml
[workspace.lints.rust]
warnings = "deny"
```

Every member opts in explicitly:

```toml
[lints]
workspace = true
```

The workspace carries no `rustfmt.toml`, no `clippy.toml`, and no `.cargo/config.toml`.
Default `rustfmt` settings are the convention. Do not add a configuration file to encode a
personal preference; a formatting question that `rustfmt` already answers is settled.

Pin dependencies in `[workspace.dependencies]` and reference them from members as
`{ workspace = true }`. Every dependency line carries a comment stating why the crate is
present and how much of it is used, for example `libc` noted as `flock only`. A dependency
without a justification comment is incomplete.

## Errors

Use `anyhow`. The workspace defines no error enums, implements no `std::error::Error`, and does
not depend on `thiserror`. Introducing a custom error type requires a caller that matches on
variants; absent such a caller, it is unused ceremony.

Fallible functions return `anyhow::Result<T>`. Construct errors with `bail!` or
`anyhow::anyhow!`, and enrich them on the way up with `.context(...)` or
`.with_context(|| format!(...))`:

```rust
// Lowercase, specific, and naming the offending value.
bail!("non-finite logits in evaluation: NaN at index {i}");

.with_context(|| format!("dataset line {}", i + 1))?;

.map_err(|e| anyhow::anyhow!("tensor {}: {e}", info.name))?;
```

Message conventions:

- Lowercase, no trailing period.
- Name the value that failed, not only the operation. `tensor {name}: {e}` beats
  `failed to read tensor`.
- State the limit that was exceeded when one applies.

`pub enum` is for data, not errors. `MetaValue`, `LayerKind`, and `ParamType` are the
established uses.

### `unwrap` and `expect`

In library paths, propagate with `?` and add context. Reserve `expect` for invariants that
cannot fail without a bug elsewhere, and for binaries where the panic message is the user-facing
error. An `expect` carries a message stating the invariant, never a restatement of the call.

Inside `#[cfg(test)]` modules and integration tests, `unwrap` is fine and preferred over
noise; a panicking test is a failing test.

## Unsafe code

The `runtime` crate forbids unsafe entirely:

```rust
// forbid, not deny: a module-level allow cannot lift it, and this crate
// produces training and interpretability numerics.
#![forbid(unsafe_code)]
```

Choose `forbid` over `deny` for any crate where a later module-level `#[allow]` would be a
mistake rather than an escape hatch.

Where `unsafe` is unavoidable, confine it to the smallest possible block and precede every
occurrence with a `// SAFETY:` comment that reasons about the specific obligation — descriptor
ownership, pointer validity, lifetime — rather than asserting that the call is safe:

```rust
// SAFETY: 'events' owns the descriptor and stays alive past this call,
// so the fd cannot be closed underneath flock.
unsafe { libc::flock(fd, libc::LOCK_EX) };
```

A later `// SAFETY:` in the same function may reference an earlier one rather than repeat it,
provided the obligation is genuinely identical.

## File and module organization

Open every file with a block banner stating purpose, not a `//!` module doc:

```rust
/*
 * runtime — inference engine library crate.
 *
 * Deterministic CPU reference implementation and the Model trait seam that
 * backends implement.
 */
```

Keep `lib.rs` a flat `pub mod` list with one file per domain concept, and no `pub use` re-export
facade — callers name the module they depend on. Group implementations of one trait in a
directory with `mod.rs` holding the trait, as `runtime/src/model/` does for `cpu`, `mock`,
`probe`, and `record`.

Binaries live in `src/bin/` with kebab-case names (`gguf-info`, `activation-capture`);
`main.rs` stays a thin entry point over library modules.

Naming follows standard Rust casing: `snake_case` for functions and modules, `CamelCase` for
types, `SCREAMING_SNAKE_CASE` for constants. Declare a constant inside the function that uses it
when it has no other consumer.

## Documentation

Use `///` on items, including private test helpers where the reason for the helper is not
obvious. Roughly 700 lines of item documentation across the workspace is the standing bar.

The workspace sets no `#![deny(missing_docs)]`. Rustdoc enforcement is external, through the
`doc` step of the gate:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items
```

Document the rationale and the non-goal. Prose in this workspace states what a thing does not
claim as readily as what it does.

## Tests

Both placements are in use, split by tier:

- Unit tests inline at the bottom of the source file in `#[cfg(test)] mod tests { use super::*; }`.
- Integration and acceptance tests in the crate's `tests/` directory.
- Tier 3 tests carry `#[ignore]` and require the `cuda` feature.

Name a test as a full sentence in `snake_case`, stating the asserted behavior. Do not prefix
with `test_`:

```rust
fn same_seed_replays_the_same_sequence() { }
fn npy_prelude_is_64_byte_aligned_and_newline_terminated() { }
fn a_probe_recording_a_different_model_hash_is_refused_naming_both_hashes() { }
```

Every assertion prints the offending value, so a failure is diagnosable without a rerun:

```rust
assert!((r.ppl() - 4.0 / 3.0).abs() < 1e-6, "{}", r.ppl());
assert!(
    out.status.success(),
    "np.load rejected the shard: {}",
    String::from_utf8_lossy(&out.stderr)
);
```

Locate fixtures relative to the manifest, never the working directory:

```rust
Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures")
```

Bind a `tempfile::tempdir()` guard to a named variable held until the end of the test, with a
comment on the drop timing when it matters.

### The `[SKIP]` contract

A cross-language contract test whose dependency is absent prints a skip line to standard error
and passes. Emit it through `runtime::testlog::skip`, never a bare `eprintln!`:

```rust
runtime::testlog::skip("python reader assertions", "no interpreter can import numpy");
```

Format is `[SKIP] <what is skipped>: <reason>` — a noun phrase, a colon, then the reason. This
follows `.styles/TEST_OUTPUT_STYLE.md`, including the blue `[SKIP]`, which the helper applies
when stderr is a terminal and `NO_COLOR` is unset. The text is byte-identical either way, so
`grep '\[SKIP\]'` still works on piped output.

**A green gate containing `[SKIP]` lines is a weaker result than its exit code suggests.**
Read the output for skip lines rather than trusting the exit status, as
`/rust-harness/README.md` requires.
The `test` step passes `--nocapture` so those lines are printed at all: cargo hides a passing
test's stderr, and a skip is by definition a pass. When adding a skippable test, make the skip
line name the missing dependency precisely enough that a reader can install it.

Do not write a test that calls `skip` with a synthetic reason. Its line lands in the gate
naming nothing real, and a reader scanning for skips has to know to discount it.

`[OK]` prefixes belong to binary status output, not to tests. No `[FAIL]` string exists in the
Rust tree; failures go through `assert!` and panic.

### Cross-language verification

Verify against an independent implementation on purpose:

- `runtime/src/npz.rs` re-reads a written shard by shelling out to `python3` and `np.load`,
  rather than parsing it with a second Rust reader.
- `harness/tests/probe_identity.rs` takes its SHA-256 from Python's `hashlib`, and spawns the
  actual harness binary rather than calling the loader in-process.

Preserve this property when extending such a test. Replacing the Python check with a Rust one
removes the only reason the test catches a shared misunderstanding of the format.

**Resolve the interpreter through `runtime::pyenv`; never spawn a bare `python3`.** Which
interpreter `PATH` leads to differs between the workstation and the sandbox, and neither
location is right in both, so `pyenv::interpreter` probes candidates and returns the first
that imports what the test needs. Two rules follow from what an absent dependency means:

- Optional dependency — take the `Option`, and skip with a line naming the missing import.
- Mandatory dependency — use `interpreter_or_panic`, so the gate fails rather than quietly
  stops checking. `npz.rs` does this: a shard nothing re-read is an unverified shard.

Probe the imports the test actually performs, not an empty script. A guard that only proves an
interpreter exists passes on a machine that cannot import the module, and the test then fails
on the dependency while reporting a defect in the code under test.

## Numerics and reproducibility

Compare floats against an explicit tolerance, with a named constant where a magnitude needs
explaining:

```rust
// Covers accumulated error across the reference layer stack.
const TOLERANCE: f32 = 1e-4;
assert!((got - expected).abs() < TOLERANCE, "got {got}, expected {expected}");
```

Use `==` on floats only where exactness is the assertion.

Seed all randomness and keep it in-tree. The workspace uses hand-rolled generators rather than
an RNG crate, so the sequence is pinned by this repository and not by a dependency version: a
64-bit LCG in `runtime/src/sampler.rs` (default seed 0) and a `WeightStream` in
`runtime/src/synth.rs` (default seed 42), both mixing with the golden-ratio constant
`0x9e3779b97f4a7c15` to avoid the zero fixed point.

Expose the seed on any command line that samples, alongside `--temperature`, `--top-p`,
`--max-tokens`, and `--context`. Assert the reproducibility property directly, as
`same_seed_replays_the_same_sequence` and `same_seed_reproduces_identical_bytes` do.

Treat non-finite values as errors at the point of detection. KV-cache restore is held to a
bit-identical standard: continuation logits after a restore match an unbroken run exactly.

## Quality gate

Run from `rust-harness/`, cheapest step first, failing on the first error:

```bash
$ make quality
# cargo fmt --all --check
# cargo clippy --workspace --all-targets -- -D warnings
# cargo metadata --locked --format-version 1     # lockfile staleness, network-free
# RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items
# cargo test --workspace
```

No step requires the network. `cargo-deny` is a deliberate non-step, because it does.

**This gate runs in a contained agent sandbox.** The agent image carries the toolchain pinned
by `rust-toolchain.toml`, a prefetched crate registry, and the numpy and gguf the
cross-language tests need. Rebuild it with `make agent-sandbox-build` after changing
`docker/agents/Dockerfile.agents` or `Cargo.lock`. A sandbox result is therefore real evidence,
not a placeholder — but read the output for `[SKIP]` lines rather than trusting the exit code,
and note that Tier 3 tests still require the workstation. The parent gate is likewise one
command in both environments; see `/docs/testing/quality-gate.md` for its step list and the
single step a sandbox skips, and `/AGENTS.md` for the execution-environment rules.

The workspace carries a `rust-toolchain.toml`. It is not an exception to the
no-configuration-files rule below: it pins a version rather than restating a `rustfmt` default,
and it is what keeps the sandbox and the workstation on one toolchain.

Profiling and debugging targets are workstation-only: `make flame` (`cargo flamegraph` against
the `release-with-debug` profile) and `make debug-record` / `make debug-replay` (rr).

## Prohibited

- `mul_add` and any FMA-introducing rewrite — breaks bit-identical replay
- Reordering floating-point arithmetic for style
- `thiserror` or hand-rolled error enums where `anyhow` with context suffices
- `unsafe` in `runtime`, which forbids it at the crate level
- `unsafe` anywhere without a specific `// SAFETY:` justification
- An RNG crate in place of the seeded in-tree generators
- `test_` prefixes on test names
- Assertions that fail without printing the offending value
- A configuration file that restates a `rustfmt` default
- Emojis, Unicode ornamentation, or marketing language in output

## Related guides

- `.styles/TEST_OUTPUT_STYLE.md`   — test program and runner output format
- `.styles/COMMENT_STYLE.md`       — comment conventions across languages
- `.styles/DOCUMENTATION_STYLE.md` — markdown documentation standards
- `rust-harness/README.md`         — workspace layout, test tiers, and roadmap
