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

Known watch item: one unreproduced failure of
`runtime/tests/tracing_spans.rs` was observed during a chained gate run
(2026-07-30) and did not reproduce in 30 consecutive runs. If it recurs,
capture the failing assertion and treat it as a Q2 determinism finding.

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
  default action cannot be bypassed by a crafted tool name or target; the
  bash tool cannot smuggle a denied operation.
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
