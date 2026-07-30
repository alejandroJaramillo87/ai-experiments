# Rust harness quality assurance campaign

Sandbox-executable quality review of the `rust-harness/` workspace,
structured as phases a Claude Code session runs in order. Complementary to
the [workstation runbook](rust-harness.md): the runbook (RH.0 through
RH.13) verifies behavior on real weights and hardware; this campaign
verifies code quality, and every step runs in an agent sandbox with no GPU
and no model data.

## Table of contents

- [Method contract](#method-contract)
- [Phase Q0: gate baseline](#phase-q0-gate-baseline)
- [Phase Q1: numerics core](#phase-q1-numerics-core)
- [Phase Q2: engine invariants](#phase-q2-engine-invariants)
- [Phase Q3: harness loop and security surface](#phase-q3-harness-loop-and-security-surface)
- [Phase Q4: observability and interpretability seams](#phase-q4-observability-and-interpretability-seams)
- [Phase Q5: packaging and reproducibility](#phase-q5-packaging-and-reproducibility)
- [Phase Q6: documentation and drift](#phase-q6-documentation-and-drift)
- [Campaign completion](#campaign-completion)
- [Campaign log](#campaign-log)

## Method contract

Each phase applies three passes to its files in scope, in this order:

1. **Simplification pass.** Reuse, dead code, needless generality, altitude
   mismatches. Quality only; this pass does not hunt for bugs.
2. **Adversarial review pass.** Correctness and robustness findings through
   the phase's stated lenses. Every finding is reported with a `file:line`
   reference and a concrete failure scenario (inputs and state leading to a
   wrong output or crash). A finding is fixed only after an independent
   verification attempt fails to refute it; refuted findings are recorded
   and dropped.
3. **Test-gap pass.** Every confirmed finding lands with a test that fails
   before the fix and passes after. Gaps discovered without a bug (an
   untested documented claim) also get tests.

Standing rules:

- Fixes land smallest-first, one concern per commit, at most one phase per
  commit.
- Expected signals in the runbook are never edited to match code behavior;
  when code and runbook disagree, the disagreement is itself a finding.
- Claims requiring real weights, the GPU, or host services are out of
  scope here and belong to the runbook. Where a phase touches such a
  claim, the campaign verifies the sandbox-checkable half only.
- Review passes fan out to read-only subagents where breadth helps; the
  session verifies findings itself before changing code.
- Each phase ends with `make quality` (repo root, which chains the rust
  gate) exiting 0.

## Phase Q0: gate baseline

Entry and exit criterion for every later phase.

Scope: `rust-harness/Makefile`, repo-root `Makefile`.

Run the root `make quality` and confirm all Python steps and all five rust
steps (fmt, clippy, lockfile check, rustdoc lints, tests) pass. Any
failure is fixed before the campaign proceeds. Do not pipe the gate
through another command when branching on its exit status.

Resolved watch item: one unreproduced failure of
`runtime/tests/tracing_spans.rs` was observed during a chained gate run
(2026-07-30). It was diagnosed as a race on the process-global per-callsite
interest cache between the subscriber-installing test and the
no-subscriber test, and fixed by serializing them; see the campaign log.

## Phase Q1: numerics core

Scope: `runtime/src/gguf.rs`, `runtime/src/tensor.rs`,
`runtime/src/tokenizer.rs`, `runtime/src/model/cpu.rs`,
`runtime/src/synth.rs`, `fixtures/reference_logits.py`.

Lenses:

- Adversarial inputs: truncated, oversized, or hostile GGUF metadata;
  quantized blocks at boundary sizes; tokenizer byte-fallback over invalid
  UTF-8 and multi-byte boundaries.
- Numerical edge cases: subnormals, NaN and infinity propagation, f16
  rounding, accumulation order assumptions.
- Reference fidelity: the numpy reference implements the same architecture
  the rust code claims (a wrong reference validates nothing); tolerance
  values are justified rather than tuned to pass.
- Panic surface: any reachable panic on untrusted file input is a finding;
  parsing errors must be positioned and named.

## Phase Q2: engine invariants

Scope: `runtime/src/session.rs`, `runtime/src/kv.rs`,
`runtime/src/sampler.rs`, `runtime/src/grammar.rs`, `runtime/src/eval.rs`,
`runtime/src/model/mock.rs`.

Lenses:

- Delta prefill: sync truncates to the common prefix and prefills only the
  delta in every divergence pattern, including empty and identical
  transcripts.
- Context budget: enforcement at the boundary, off-by-one at exactly the
  budget, behavior of forks near the budget.
- Determinism: the seeded sampler and decode path hold no wall clock and
  no thread-order dependence; identical inputs are bit-identical outputs.
- Grammar soundness: the token mask equals the accepted language exactly —
  no token allowed by the mask can produce an unparseable call, and no
  legal continuation is masked out.

## Phase Q3: harness loop and security surface

Scope: `harness/src/agent.rs`, `harness/src/tools.rs`,
`harness/src/permissions.rs`, `harness/src/subagents.rs`,
`harness/src/transcript.rs`, `harness/src/template.rs`,
`harness/src/chat.rs`, `harness/src/stream.rs`, `harness/src/util.rs`.

This phase receives the heaviest adversarial pass; it is the closest thing
to a security review the sandbox can perform.

Lenses:

- Workspace confinement: no tool argument, glob pattern, symlink chain, or
  path traversal reaches outside the workspace root.
- Permission evaluation: rule ordering, wildcard interaction, and the
  default action cannot be bypassed by a crafted tool name or target,
  including alternative spellings of the same path. Two limits are scope,
  not defects, and the lens is satisfied by their being stated in the
  module header rather than enforced: the bash tool is not confined by
  file-tool rules, and a bash wildcard pattern matches through shell
  metacharacters, so any prefix-allow for bash is an allow-anything rule.
- Control files: the rule file and the transcript live inside the
  workspace, so they must be unwritable by the model.
- Turn integrity: transcript or tool-output content containing turn
  delimiters cannot fabricate turns or escape its containment in the
  rendered prompt.
- Resource caps: tool output caps, read caps, bash timeout, and the
  repeated-failure breaker hold under adversarial output shapes.
- Persistence: torn or hostile `session.jsonl` content on resume fails
  safely and cannot escalate.

## Phase Q4: observability and interpretability seams

Scope: `runtime/src/model/record.rs`, `runtime/src/model/probe.rs`,
`harness/src/obs.rs`, `runtime/src/bin/activation-capture.rs`,
`runtime/src/npz.rs`, the tap and steering paths in
`runtime/src/model/cpu.rs`.

Lenses:

- Contract conformance: obs records, capture shards, and the probe GGUF
  match the Python-side contracts they claim (`common/obs`,
  `interp/probes`); cross-language checks run against the actual Python
  readers, not a re-implementation.
- Observer effect: monitored and unmonitored runs are bit-identical;
  disabled taps, absent subscribers, and a no-op `RunLog` cost nothing on
  the hot path and cannot fail a run.
- Recording integrity: replayed recordings reproduce the original token
  sequence; malformed recordings are positioned errors.

## Phase Q5: packaging and reproducibility

Scope: `docker/rust-harness/Dockerfile.rust-harness`,
`docker-compose.rust-harness.yaml`, the `##@ Rust harness` targets in the
repo-root `Makefile`, `rust-harness/Cargo.toml`, `rust-harness/Cargo.lock`.

Lenses:

- Pin completeness: every base image is digest-pinned with a re-resolve
  comment; every dependency resolves through the committed lock;
  `--locked` appears on every build the docs call reproducible.
- Compose hygiene: security options, read-only mounts, no ports, no
  daemon-lifecycle coupling; the sessions bind mount behaves for a
  non-root container user.
- Makefile targets match the compose file and the documented commands.
- Image build and container run are not sandbox-checkable; their
  acceptance is runbook entry RH.12, and this phase verifies everything
  short of the build.

## Phase Q6: documentation and drift

Scope: `rust-harness/README.md`,
`docs/inference/rust-harness-runtime-design.md`,
`docs/inference/rust-harness-observability.md`,
`docs/interpretability/rust-harness-interp.md`,
`docs/verification/rust-harness.md`, this document.

Lenses:

- Every capability claim in a doc is traceable to code or a test; claims
  about pending work say pending.
- Compose files and Dockerfiles are the source of truth; a doc that
  disagrees is the bug.
- Command examples are copy-runnable (sandbox commands in the sandbox,
  host commands marked as host-side).
- Style conformance with `.styles/DOCUMENTATION_STYLE.md`.

## Campaign completion

The campaign is complete when:

- All seven phases have run in order, each ending with the root
  `make quality` exiting 0.
- Every confirmed finding is fixed with a regression test, or explicitly
  waived with a recorded rationale.
- Refuted findings are recorded as refuted, not silently dropped.
- A closing summary records, per phase: findings confirmed, refuted,
  fixed, and waived, plus the final gate result.

## Campaign log

First run of the campaign, 2026-07-30, sandbox, commits `71cc92b`,
`316adab`, and the commit carrying this log. Every fix below shipped with a
regression test in the same commit.

### Q0 gate baseline

The gate was upgraded to five offline steps before the campaign began
(`17480c3`): `cargo fmt --check`, `clippy -D warnings`, a lockfile freshness
check (`cargo metadata --locked`), rustdoc lints as errors, and
`cargo test`. The repo-root `make quality` now chains it. The lockfile step
was negative-tested by adding a dependency to a manifest without updating
the lock, observing the failure, and reverting.

### Q1 numerics core

- **Fixed.** `runtime/src/tensor.rs:25` `element_count` and
  `runtime/src/tensor.rs:40` `byte_size`: GGUF bounds the count of tensor
  dimensions (at most four) but not their magnitude, so the `u64` product
  and the element-to-byte scaling overflowed on a corrupt or hostile
  header — a panic in debug builds, a wrapped too-small size in release.
  Both now use checked arithmetic and return an error naming the tensor.
  Demonstrated before the fix with a standalone program reproducing the
  overflow panic.
- **Clean.** f16 conversion in both directions (subnormals, ties-to-even,
  overflow to infinity, NaN payloads) is covered by existing goldens
  against numpy; `load_tensor` already used checked offset arithmetic and
  bounds-checks the payload slice against the file; `dequantize` requires
  an exact payload size; Q8_0 rejects partial blocks.

### Q2 engine invariants

- **Fixed.** `runtime/src/grammar.rs:67` `Grammar::compile` injected the
  schema name and parameter keys into the skeleton unescaped, so a name
  containing a quote or backslash produced an accepted language of
  unparseable JSON, contradicting the module's own invariant. Names and
  keys are now JSON-escaped, and duplicate parameter keys are rejected as
  a config bug.
- **Fixed.** `runtime/src/session.rs:212` advanced the grammar state before
  the token's prefill, so a transient prefill error left the state one
  token ahead of the cache and a retried `step` masked from the wrong
  position. The next state is now computed first and committed only after
  the prefill succeeds.
- **Fixed.** `runtime/tests/tracing_spans.rs`: the two tests shared a
  process and raced tracing's process-global per-callsite interest cache,
  which explains the Q0 watch item — a stale "never" interest silently
  drops spans for one run, failing a count assertion without reproducing.
  The tests now serialize, and the capture layer removes closed span ids
  (a latent id-reuse hazard). Verified with a 30-run stress pass.
- **Waived.** The budget check in `step` runs after sampling, so a step
  that fails on the budget still consumes one sampler draw. EOS is legal
  at exactly the budget, which requires this order; the ordering is now
  documented in place.
- **Clean.** Delta prefill across all four divergence patterns (empty,
  identical, divergence at the first and last token), context-budget
  enforcement at the boundary and across forks, sampler determinism (pure
  seeded arithmetic, fixed reduction order, no wall clock or map iteration
  in the decode path), and mask-equals-accepted-language including EOS
  admission only at completion.

### Q3 harness loop and security surface

- **Fixed.** `harness/src/tools.rs:287` `resolve` was lexical only while
  the file tools use symlink-following IO, so a symlink inside the
  workspace redirected `read`, `write`, and `edit` anywhere on the
  filesystem; the lexical `..` accounting also diverges from kernel
  resolution once a symlink is in the path. `resolve` now canonicalizes the
  deepest existing ancestor and requires it to stay under the canonical
  root, re-joining the not-yet-existing tail so writing a new file still
  works. Symlinks that stay inside the root are unaffected. `glob` was
  already safe (it uses `symlink_metadata`).
- **Fixed.** `permissions.yaml` and `session.jsonl` live in the workspace
  root, so one allowed write let the model grant itself blanket permission
  (effective at the next process start) or forge transcript turns, whose
  role fields resume trusts. Both are now refused by the tool layer
  regardless of rules.
- **Fixed.** `harness/src/permissions.rs:53` matched patterns against the
  raw path spelling, so a deny rule was bypassed by `./secrets/x` or
  `src/../secrets/x`. Path targets are now lexically normalized before
  evaluation; commands and glob patterns stay verbatim.
- **Documented, not enforced.** Bash is not confined by file-tool rules,
  and a bash wildcard pattern matches through shell metacharacters, so
  `bash "cargo **" allow` permits `cargo test; rm -rf ~`. Neither is
  fixable by pattern matching; both are now stated in the module header.
- **Clean.** Turn integrity: every content path funnels through
  `render_entry`, so tool results, chat replies, the user task, and resumed
  entries are all sanitized; no path reaches the rendered prompt
  unsanitized. Bash resource control (process-group kill, drained pipes,
  closed stdin, output cap) and the transcript's torn-tail recovery
  (positioned error mid-file, truncation only for a trailing partial line)
  both hold.
- **Waived, recorded.** `fs::write` in write and edit is not atomic, so a
  crash mid-edit can truncate a file; `read` has a 64 KiB cap with no
  offset parameter, leaving the tail of a large file unreachable; `glob`
  skips dotfiles while `read` can open them; `transcript.rs` reads a
  session file with no size bound; the failure breaker trips only on
  identical consecutive failures, with `max_turns` as the backstop. All
  are single-user local-workstation tradeoffs, none is a confinement
  escape, and each is cheap to revisit if it bites.

### Q4 observability and interpretability seams

- **Clean.** The obs contract was re-verified end to end against the real
  Python readers: two harness runs (one success, one failure) written to a
  scratch `OBS_LOG_DIR`, then `common.obs.catalog.read_runs` over the
  result — two rows, integer exit codes, the `--failed` filter selecting
  exactly the failed run, base keys in the documented order, and every
  timestamp and run id matching the readers' pinned patterns. Event
  sequences were `cmd.start`, `cmd.end` and `cmd.start`, `fail`,
  `cmd.end`.
- **Clean.** Observer-effect claims are covered by existing goldens: taps
  do not perturb logits, a zero control vector is bit-identical to no
  steering, probe scores match the numpy reference, and the monitor flags
  only scores strictly above the trained threshold.

### Q5 packaging and reproducibility

- **Fixed.** `docker-compose.rust-harness.yaml` did not set `user:`, unlike
  every other bind-mounting service in the repository, so the image's
  non-root `appuser` could not write `session.jsonl` into the host-owned
  sessions directory. The service now matches the host user.
- **Documented.** Container runs collect no obs records, because
  `${OBS_LOG_DIR}` is not mounted and the run log degrades silently. Noted
  in the compose header; the obs contract is verified on the host by RH.13.
- **Clean.** Both images are digest-pinned with re-resolve commands, the
  build uses `--locked` against a committed lock, and the compose file
  keeps its no-network, no-new-privileges, capability-dropped, read-only
  posture with models mounted read-only. The image build itself is not
  sandbox-checkable; RH.12 remains its acceptance.

### Q6 documentation and drift

- **Fixed.** The rust-harness README described a three-step quality gate
  and omitted the obs module from the layout; both now match the code.
- **Fixed.** This document's Q3 permission lens claimed that bash cannot
  smuggle a denied operation, which the code does not enforce. The lens
  now states the two scope limits explicitly.
- **Clean.** No documentation claimed workspace confinement against
  symlinks, so the Q3 finding was a code gap rather than a doc
  overstatement. Pending work is described as pending throughout, and the
  design docs agree with the compose file and Dockerfile.

### Result

Findings: twelve confirmed (nine fixed, one documented in place, plus the
two documentation fixes), zero refuted, six waived with rationale. The
root `make quality`, including the chained five-step rust gate, exits 0.
Real-weight, GPU, and container-build claims remain with runbook entries
RH.0 through RH.13.
