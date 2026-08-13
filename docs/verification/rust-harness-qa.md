# Rust harness quality assurance campaign

**Operational runbook. Executed in an agent sandbox rather than against the host, so it needs no GPU or model data. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

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
- [Campaign 1 log](#campaign-1-log)
- [Campaign 2 log](#campaign-2-log)

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

Implementation: `rust-harness/Makefile`, the `quality` target in the root
`Makefile`.

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

Scope: `runtime/src/checkpoint.rs`, `runtime/src/tensor.rs`,
`runtime/src/tokenizer.rs`, `runtime/src/model/cpu.rs`,
`runtime/src/synth.rs`, `fixtures/reference_logits.py`, and
`gguf.rs` — which is no longer the model reader, only the parser
for the probe and control-vector artifacts.

Implementation: `rust-harness/runtime/src/tensor.rs`,
`rust-harness/runtime/src/sampler.rs`,
`rust-harness/runtime/tests/cpu_reference.rs`.

Lenses:

- Adversarial inputs: truncated, oversized, or hostile safetensors headers
  and `config.json` fields, which are the model-load attack surface;
  malformed GGUF metadata in a probe or control-vector image; tokenizer
  byte-fallback over invalid UTF-8 and multi-byte boundaries.
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

Implementation: `rust-harness/runtime/src/model/cpu.rs`,
`rust-harness/runtime/src/kv.rs`, `rust-harness/runtime/src/session.rs`,
`rust-harness/runtime/tests/eval_acceptance.rs`.

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

Implementation: `rust-harness/harness/src/agent.rs`,
`rust-harness/harness/src/permissions.rs`,
`rust-harness/harness/src/tools.rs`,
`rust-harness/harness/tests/agent_e2e.rs`.

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

Implementation: `rust-harness/harness/src/obs.rs`,
`rust-harness/runtime/src/model/probe.rs`, `rust-harness/runtime/src/npz.rs`.

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

Implementation: `docker/rust-harness/Dockerfile.rust-harness`,
`docker-compose.rust-harness.yaml`, `rust-harness/Cargo.lock`.

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
`docs/rust-harness/runtime-design.md`,
`docs/rust-harness/observability.md`,
`docs/interp/rust-harness-interp.md`,
`docs/verification/rust-harness.md`, this document.

Implementation: `rust-harness/README.md`, `docs/rust-harness/`,
`docs/verification/rust-harness.md`.

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

## Campaign 1 log

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

## Campaign 2 log

Second run of the campaign, 2026-07-31 to 2026-08-01, sandbox, 35 commits
from `66ec39b` to `7ccac37`. The discovery sections below were written
before any fix landed, so the result survives independently of the
remediation; the dispositions and result follow at the end.

Campaign 1 findings are excluded from scope by construction: every finder
received the fixed and waived list from the log above and duplicates were
rejected during verification.

The method both campaigns ran is now written down once, in the `bug-bash`
skill (`.claude/skills/bug-bash/SKILL.md`): the phase ladder, the severity
rubric, what a confirmed finding is, the standing dimension list, and the
batch-and-push cadence. Its roles are committed subagents
(`bug-hunter`, `skeptic`, `triage-agent`, `remediator`, `integrator`,
`perf-auditor`, `doc-auditor`) driven by `/bugbash`. Read the skill rather
than this log for the method; this log is the record of one run of it.

### Method and coverage

Discovery ran as a fan-out of read-only finder agents over ten dimensions,
followed by severity-weighted adversarial verification (two independent
refutation attempts for critical and high findings with a third as
tie-breaker, one for medium, none for low) and one bounded re-sweep of the
dimensions that produced confirmed critical or high findings. Finders were
permitted to compile scratch crates against the workspace, run the release
binaries, and cross-check against Python; the repository tree was not
modified. Numpy 2.4.6 and gguf 0.19.0 were installed in the sandbox
environment, which enabled genuine cross-language checks against the real
`interp` readers.

Baseline before discovery: `make -C rust-harness quality` exits 0 across
all five steps, 244 tests (238 runnable, six host-tier ignored).

Two dimensions terminated on an API safeguard during the first pass and
were swept in a second pass, which also settled the findings whose
verifiers were lost to a session limit. The second pass held a harder
line: a claim its verifier could not independently reproduce was refuted
rather than accepted. It refuted four findings, downgraded two, and added
sixteen, including one critical and five high.

- **Input validation and decoding** (`gguf.rs`, `tensor.rs`, the `HKV1`
  snapshot restore, `tokenizer.rs`, transcript recovery, permission
  loading) — swept in pass two.
- **Argument parsing, orchestration, and child-process lifecycle**
  (`main.rs` flag extraction, `stream.rs`, `subagents.rs`, the `tools.rs`
  bash lifecycle, the three `obs.rs` FFI sites) — swept in pass two.

### Confirmed: forward-pass numerics and conventions

- **A1-R03 (critical).** `runtime/src/model/cpu.rs:418` — `rms_norm`
  multiplies by `(1 + w)`, but no llama.cpp Gemma GGUF stores weights that
  way. The converter folds the shift into the tensor for gemma1 through
  gemma3 (`norm_shift` 1.0), and stores the raw weight for gemma3n and
  gemma4 (`norm_shift` 0.0), while the graph applies `rms_norm` followed by
  a plain multiply. The stored weight is therefore always the effective
  multiplier, so every real checkpoint would be mis-normed with no possible
  refusal. `fixtures/reference_logits.py` encodes the same convention, so
  the golden cannot detect it. Both verifiers reproduced the divergence
  against upstream sources; one showed a folded fixture flipping the
  last-position argmax.
- **E-R01 (critical).** `runtime/src/model/cpu.rs:356` — control-vector
  tensor `direction.k` is stored at `directions[k-1]` and applied to
  decoder block `k-1`. llama.cpp stores `direction.N` at slot `N-1` and
  applies it to block `N` (block 0 unsteerable), and repeng trains key `k`
  on the output of block `k`. The same `vector.gguf` therefore steers a
  different block in the harness than in `llama-cpu` and `llama-gpu`, with
  no error on either side. Verified by counterexample: a `direction.1`-only
  vector perturbs the block 0 residual.
- **A1-R04 (refuted).** `runtime/src/model/cpu.rs:565` — the claim that one
  `rope_freq_base` is silently applied to sliding-window and global layers
  alike. The decisive question was whether the converter writes
  `{arch}.rope.freq_base_swa`, and it does: `Gemma3Model` calls the base
  class first, which harvests `rope_local_base_freq` and emits the key, and
  gemma3n and gemma4 subclass it. Four checkpoints built with the
  converter's exact key set were run through the real `Config::from_gguf`
  and `check_supported`: a stock gemma3 file is refused for a missing
  scalar `sliding_window_pattern`, adding that scalar makes the existing
  dual-RoPE refusal fire on `freq_base_swa`, and the gemma4 encoding fails
  earlier still in parsing. The wrong behavior is unreachable — every path
  refuses first. The residual is an imprecise guarantee in the `cpu.rs`
  header, folded into the documentation work.

  The refutation was correct when written and its reasoning no longer
  holds: dual RoPE is implemented, so `freq_base_swa` is read rather than
  refused and no path refuses first. Per-layer base selection is now under
  test directly — `ModelSpec::from_gguf` assigns each regime its own base,
  and the E4B-shaped fixture declares both.
- **A1-15 (medium, fixed).** `runtime/src/synth.rs:36` — the fixture shape
  makes the golden structurally blind to two conventions: `head_dim` equals
  `embed_dim / n_heads`, so the two candidate QK scales coincide, and
  `n_kv_heads` is 1, so blocked and interleaved GQA mappings agree.

  Closed by the second fixture, `fixtures/tiny-gemma4-e4b/`, whose
  geometry has neither property in either attention regime. Both
  conventions are now separable, and
  `the_e4b_fixture_geometry_separates_the_ambiguous_conventions`
  (`runtime/tests/cpu_reference.rs`) asserts the separating properties
  directly, so a later change to the fixture spec cannot reintroduce the
  blindness while the goldens still pass.
- **A1-12 (medium).** `runtime/src/model/cpu.rs:580` — `sliding_window = 0`
  is accepted and yields empty attention, silently producing a
  context-blind model; the ecosystem reading of 0 is that windowing is
  disabled, and the numpy reference raises on the same input.
- **A1-14 (medium).** `runtime/src/model/cpu.rs:228` — the refusal backstop
  covers tensor families and `rope.*` metadata only, so numeric-affecting
  metadata such as Gemma-2 logit softcapping is silently ignored.

### Confirmed: load-time validation

- **A1-02 and C-07 (high).** `runtime/src/model/cpu.rs:453` and `:475` —
  `from_gguf` length-checks only `token_embd.weight`, leaving all thirteen
  per-layer tensor families and `output_norm` unchecked against the
  configuration. A checkpoint whose tensor dimensions disagree with its
  metadata loads cleanly and then either emits silently wrong logits in
  release builds or panics mid-forward. Reproduced with a self-consistent
  GGUF: an `attn_norm` of the wrong width gives finite logits differing by
  0.82, and an undersized `attn_q_norm` panics at `cpu.rs:564`. The only
  shape guards, `cpu.rs:405` and `probe.rs:102`, are `debug_assert`, which
  release builds compile out.
- **A1-13, H-37, and C-19 (medium).** `sliding_window_pattern = 0` passes
  `Config::from_gguf`, and `CpuModel::layer_kind` tolerates it, but
  `kv::interleaved_layers` asserts a positive period, so the binary aborts
  with exit 101 rather than reporting a positioned load error. The loader
  and the budget calculator disagree about the same configuration, and
  `gguf-info` reports such a file as supported.
  [Since recorded: the GGUF model path is gone, so `Config::from_gguf` and
  the inspector named here no longer exist — `checkpoint-info` replaced the
  latter. The disagreement between the loader and `kv::interleaved_layers`
  needs re-checking against `ModelSpec::from_checkpoint`, which reaches the
  same assert by a different route.]

### Confirmed: engine state and atomicity

- **A1-03 (high).** `runtime/src/model/cpu.rs:670` — `prefill_all` is not
  atomic: tokens preceding a failing one stay committed to the KV cache and
  to `cached_len`, so later use of the model computes with phantom context.
- **C-06 (high).** `runtime/src/session.rs:107` — `Session::sync` assumes
  `Model::prefill` fails atomically, which `CpuModel::prefill_all` does not
  honour. A mid-batch failure leaves the model cache ahead of
  `session.tokens`, and the retrying `sync` never truncates the orphaned
  positions.
- **C-05 (high).** `runtime/src/model/cpu.rs:749` — `load_state` mutates
  the cache vectors before its read can fail, so a truncated snapshot
  leaves partially restored rows with `cached_len` at 0; a subsequent
  `sync` then succeeds while attention reads the stale rows.
- **C-20 (medium).** `runtime/src/model/record.rs:183` — `replay` resizes
  the script vector to `row.position + 1` with no bound, so a short corrupt
  recording line requests a multi-hundred-gigabyte allocation and aborts
  the process instead of returning a positioned error.
- **C-21 and F-27 (medium).** `harness/src/stream.rs:27` — `TokenSink::emit`
  uses `blocking_send`, which panics on a tokio runtime thread, yet
  `generate_interleaved`, the entry point it is designed to pair with, runs
  on the runtime. The natural composition panics mid-generation, and no
  test pins the hazard.
- **A2-17 (medium).** `runtime/src/eval.rs:47` — the non-finite guard in
  `nll` misses NaN because `f64::max` ignores NaN in the fold, so
  `perplexity` returns `Ok` with a NaN sum instead of the intended error.
- **A2-16 (medium).** `runtime/src/model/probe.rs:159` — `Probe::from_gguf`
  accepts non-finite parameters, and a NaN score fails the strict
  comparison, so the monitor reports zero flags, which is indistinguishable
  from a clean run.

### Confirmed: interpretability contracts

- **E-01 (critical).** `runtime/src/bin/activation-capture.rs:200` —
  duplicate `--layers` entries write shards whose rows are duplicated and
  shifted while labels stay per-example. With `--layers 1,1` on a
  twelve-example dataset, row `i` holds example `i/2`'s activation, eleven
  of twelve rows are mislabeled, the run exits `[OK]`, and the artifact is
  internally self-consistent. The real `interp.probes.capture.load_capture`
  accepts it. The Python writer at `interp/probes/capture.py:102` has the
  mirrored defect, and neither CLI deduplicates.
- **E-08 and A2-04 (high).** `runtime/src/model/probe.rs:86` and
  `harness/src/main.rs:86` — the harness parses `probe.model_hash` and
  never compares it to the served model, and applies control vectors with
  no identity check at all. Every Python consumer refuses such a pairing;
  the Rust side produces meaningless scores and steering with no error.
- **F-09 (high).** `runtime/src/bin/activation-capture.rs:176` — the
  capture pipeline (`run`, `load_dataset`, the tap, shard slicing, manifest
  writing) has no end-to-end test; only the `sha256_hex` and `utc_now`
  helpers are covered, leaving the interp artifact format unguarded.
- **E-22 (medium).** `interp/probes/capture.py:196` — `load_capture` zips
  shard rows with `examples.jsonl` texts purely by position with no count,
  ordering, or manifest-count check, so a same-length rewrite of the
  dataset silently reassigns splits and detaches texts from activations.
- **E-23 (medium).** `interp/probes/capture.py:51` — the Python writer
  coerces non-integer labels through `int()`, so a soft label of 0.9
  becomes class 0, while the Rust writer refuses the identical line. The
  backends disagree on acceptance.

### Confirmed: test gaps

- **F-11 (medium).** `harness/src/main.rs:209` — the 506-line binary entry
  point (flag extraction, usage exits, snapshot wiring, permission
  fallback, run-log wiring, probe reporting) is never executed by any test;
  no test in the workspace spawns any binary.
- **F-10 (medium).** `runtime/src/model/cpu.rs:112` — `Config::layer_kind`
  and `kv::interleaved_layers` are parallel implementations of the same
  pattern with no test binding them, and `kv::budget_bytes` is never
  cross-checked against the actual cache allocation.
- **F-25 (medium).** `harness/src/subagents.rs:49` — `generate_interleaved`
  has one test that never exercises the end-of-sequence stop path, never
  asserts the stop flag, never passes a grammar, and never propagates a
  mid-generation error.
- **F-24 (low).** `runtime/src/model/mod.rs:45` — the `Model` trait default
  bodies have no coverage; every shipped implementation overrides them.
- **F-26 (low).** `gguf.rs:125` — parsing has hand-written
  adversarial tests for specific fields but no seeded mutation corpus
  asserting that byte flips and truncations always return an error rather
  than panicking or over-allocating.

### Confirmed: performance and memory

Each candidate below is classified as preserving bit-identical output;
measurements are from the sandbox at the stated geometry.

- **G-32.** `cpu.rs:431` — `rope` recomputes `powf` per pair, head, layer,
  and token; caching the frequency table cuts measured cost from 0.80 ms to
  0.31 ms per token at E4B geometry.
- **G-36.** `cpu.rs:556` — `forward_one` allocates roughly
  `13 + 2*n_heads + n_kv_heads` vectors per layer per token: 40.2
  allocations per token measured on the fixture, extrapolating to about
  1088 per token at E4B geometry.
- **G-31.** `sampler.rs:107` — the default sampling path builds and sorts a
  full-vocabulary candidate vector per token: 8.05 ms and 6.3 MB of churn
  per token at vocabulary 262144.
- **G-35.** `grammar.rs:242` — `mask` simulates every vocabulary piece per
  decode step: 2.13 ms to 6.10 ms per step at the same vocabulary.
- **G-28.** `session.rs:24` — startup fingerprints the whole GGUF with
  byte-serial FNV-1a at about 0.8 GB/s: four to five seconds per invocation
  on a 3 GB to 4 GB checkpoint before any token is produced.
- **G-29.** `cpu.rs:719` — snapshot save and load move the cache four bytes
  at a time through trait objects: 801 MB/s and 861 MB/s against 2956 MB/s
  for a plain pass.
- **G-33.** `record.rs:113` — the recorder clones the full logits row
  before serializing on every forward call: 5.2 ms, 2.1 MB allocated, and
  2.19 MB written per decoded token at vocabulary 262144.
- **G-30.** `main.rs:67` — the GGUF byte vector stays alive across snapshot
  restore, so peak resident size holds the file bytes, the dense weights,
  and the snapshot payload simultaneously.
  [Closed by P10.6/P10.7, not by perf work: there is no byte vector to
  retain. `CpuModel::load` reads each tensor from the file at its offset and
  drops the handle at the closing brace, so nothing above the loader can
  hold the payload. Peak RSS after load is now printed in the run header.]
- **G-34.** `compact.rs:54` — every agent turn tokenizes the full rendered
  prompt twice.
- **A2-18.** `activation-capture.rs:96` — `sha256_hex` copies its entire
  input before hashing, measured at plus 1023 MiB peak for a 1 GiB input.

### Confirmed: hygiene and documentation

- **H-38 (medium).** `harness/src/main.rs:263` — any error reading
  `permissions.yaml` maps to "no config", so an existing but unreadable
  rules file is silently ignored and every configured rule is dropped.
- **I-39 (medium).** `docs/verification/rust-harness.md:375` — the RH.9a
  snippet calls `.activations` on the return of `load_capture`, which is a
  plain three-tuple, so the documented command fails on a valid capture.
  [Fixed; the snippet now unpacks the tuple. The line number is as-of-campaign
  and has since drifted — find it under the RH.9a heading.]
- **I-40 (medium).** `runtime/src/npz.rs:187` — the rust gate requires a
  `python3` that can import numpy, because an `npz.rs` test shells out to
  it. RH.0, the AGENTS.md caveat list, and the README omit this
  prerequisite, and RH.0 attributes the dependency to `cpu_reference.rs`,
  which needs none.
- **I-41 (medium).** `docs/rust-harness/observation-frames.md:70` — subagent
  forks and token streaming are described as built runtime behavior, but
  `subagents.rs` and `stream.rs` have no call sites outside their own tests
  and are unreachable from the shipping binary.
- **I-42 (low).** `docs/verification/rust-harness.md:418` — three runbook
  command blocks fail as written because they write into or change to
  directories that are never created.
  [Fixed; every `/tmp/rhN` block now carries its own `mkdir -p`. The line
  number is as-of-campaign and has since drifted.]

### Confirmed: input validation and decoding

- **B-N01 (critical).** `runtime/src/model/record.rs:174` — `replay` keys
  the reconstructed script by absolute position alone, so a position
  recorded twice with different token history is silently overwritten by
  the later write. Replaying the earlier branch then returns the other
  branch's logits and reports success. This fires on ordinary runs: every
  transcript divergence, every compaction, and every fork, because forks
  share one writer. Both advertised uses break — sandbox replay of a
  workstation run returns wrong numbers, and divergence bisection reports a
  phantom first-divergent position.
- **B-N02 (high).** `harness/src/permissions.rs:35` — `Rule` and
  `Permissions` deserialize without `deny_unknown_fields`, so one
  transposed letter in `pattern` leaves the field `None`, and the
  match-everything branch turns a narrowly scoped bash allow rule into an
  allow-anything rule. Verified: with `patern` misspelled, `evaluate`
  returns `Allow` for `rm -rf` and for a piped remote script. The module
  already rejects the analogous typo one field over, so the omission
  contradicts its own stated policy.
- **B-N04 (high).** `runtime/src/tokenizer.rs:138` — only `Normal` pieces
  enter the encode lookup, so `CONTROL` and `USER_DEFINED` pieces are
  unreachable. On a real Gemma vocabulary the harness's own chat
  delimiters therefore encode as 17-token byte-fallback runs rather than
  their single special identifier: the model sees a turn boundary it was
  never trained on, at roughly fifteen times the token cost per delimiter.
  The fixture carries no chat-delimiter piece, so no green test covers it.
- **B-N05 (medium).** `harness/src/transcript.rs:56` — a session file whose
  final line is complete JSON without a terminating newline is accepted and
  appended onto, and the concatenated line then fails to parse on the next
  open. Being last, it is silently truncated away, destroying every entry
  after the first.
- **B-N06 (medium).** `runtime/src/tokenizer.rs:137` — the encode window is
  bounded by the longest `Normal` piece in the vocabulary with no cap, so a
  single oversized piece makes encoding super-quadratic: a 3.6 KB prompt
  takes 2.9 s where a normal vocabulary takes microseconds.
- **B-N03 (medium).** `runtime/src/model/cpu.rs:749` — `load_state`
  reserves `cached_len * kv_dim` floats from a snapshot field before
  reading any payload, so a 41-byte `session.kv` reserves 1 GB and an
  unbounded context length aborts the process through the allocation error
  handler.

### Confirmed: argument parsing and process lifecycle

- **D-N02 (high).** `harness/src/tools.rs:285` — the bash timeout bounds
  only the polling loop; the unconditional joins afterwards wait for end of
  file on pipes that any backgrounded process still holds open. Measured:
  `sleep 130 & echo started` returns after 130.007 s against a documented
  120 s bound, and a command that starts a server never returns at all, so
  the agent loop hangs with no timeout. The same loop also costs 10.6 ms
  per trivial call through a fixed sleep.
- **D-N03 (high).** `harness/src/main.rs:295` — `report_probe` propagates
  its error, so a probe-output write failure converts a completed,
  transcript-recorded task into exit 1 with no answer printed,
  contradicting the in-code contract that observation never affects the
  result. Verified with a missing output directory: the snapshot is
  written, the transcript holds the completed pair, stdout is empty, and
  all 65 collected flags are discarded.
- **D-N04 (high).** `harness/src/main.rs:358` — a flag's value is taken
  without checking whether it is itself a flag, so an omitted value
  silently swallows the following flag: the swallowed flag is disabled, its
  value falls into the task text, and the run exits 0. Verified: a file
  literally named `--probe-out` is created, and the transcript records a
  task the user never typed.
- **D-N05, D-N06 (medium).** `harness/src/main.rs:358` — task text is
  scanned for flags rather than separated from them, so a task word equal
  to a flag name is consumed together with the word after it; and only the
  first occurrence of each flag is consumed, so a repeated flag leaves its
  second occurrence in the task text instead of being rejected.
- **D-N07, D-N08 (medium).** Two startup paths abort with a panic and
  backtrace rather than the usage or IO error the surrounding code takes
  care to produce (a non-UTF-8 argument, and an unwritable trace path), and
  a closed stdout orphans the observability event file, leaving a start
  record with no end record and no catalog row.
- **D-N09 (medium).** The clean-exit snapshot is written straight over
  `session.kv`, destroying the previous good snapshot before the new bytes
  exist, so a partial write makes the following run fail outright instead
  of degrading.

### Refuted

Recorded as refuted, not silently dropped.

- **A1-R04** — dual RoPE base; unreachable, every path refuses first. See
  the numerics section above.
- **A1-R06** `runtime/src/model/cpu.rs:747` — snapshot geometry mismatch.
  The mechanism is real at the raw trait level, but the only in-repo caller,
  `Session::restore`, first requires the saved fingerprint to equal the
  model fingerprint, and a different cache geometry implies different GGUF
  bytes, so the fingerprints always differ. Reproduced end to end: the
  cross-model restore errors with "different model weights".
- **C-R12** `runtime/src/model/record.rs:108` — recorder append failure
  breaking prefill atomicity. The mechanism is partly real, but no caller
  retries: every call site propagates the error and aborts before the
  snapshot save, so nothing is silent. The general non-atomicity of
  `prefill_all` is tracked separately as A1-03.
- **D-N01** — a recording missing the restored prefix. Paired runs with and
  without an existing snapshot produced identical row coverage and
  bit-identical logits; the claim misread the record metadata as coverage.

### Resolved from pending

The five findings whose verifiers were lost are now settled: **C-R10**
confirmed and downgraded to medium (it needs a hand-edited checkpoint and
fails loudly, with no wrong numbers), **A1-R07** confirmed as stated for
`sliding_window_pattern` (the converter always writes it as a boolean
array, which `gguf.rs` cannot parse, so `gguf-info` exits 1 on a real
checkpoint instead of printing its verdict; the claim was over-broad about
the other two keys) — [since recorded: this mechanism no longer reaches the
load path at all, because the model is read from `config.json` rather than
from converter metadata, and `checkpoint-info` replaced `gguf-info`] — **A1-R05** reduced to a documentation and labelling
matter (the position semantics are already documented as cache-relative,
and no consumer reads the file; the residual is that the counter reports
scoring events while calling them positions), and **A1-R06** and **C-R12**
refuted as above. A2-R09 and C-R11 restate E-01 at a different line and
fold into it.

### Recorded without verification

Twenty-eight low-severity findings were accepted into the record without a
refutation attempt, by the severity policy above — twenty-six from the
first pass and two more from the second. They cover dead code
(`is_vision_tensor`), release-only shape guards in `Probe::score`,
fingerprint sensitivity to path spelling, provenance drift between the two
capture writers (`dataset_sha256` over raw bytes against decoded text, the
hardcoded `model_revision`, and `git_sha` read from the process working
directory), the unexercised retention handshake, the absent
`[workspace.lints]` and `#![forbid(unsafe_code)]` declarations, missing
SAFETY comments on the three FFI sites, `temp_dir` use in the npz test,
and further documentation drift. Two lint families are recorded as never
to be adopted: `suboptimal_flops`, which would change f32 numerics and
break the determinism and golden tests, and `float_cmp`, which fires only
on tests asserting exactness.

### Dispositions

Every fix landed with a regression test demonstrated failing before it and
passing after; both outputs were recorded during remediation, and each
batch closed with an independent integrator that re-derived the red-before
in a clean worktree rather than trusting the claim.

| Finding | Disposition | Commit |
|---|---|---|
| A1-R03 RMSNorm convention | Fixed | `5cd9ef9` |
| E-R01 control-vector block | Fixed | `fbbc025` |
| E-01 duplicate capture layers | Fixed | `6cd163e` |
| F-09 capture pipeline untested | Fixed | `e17cc65` |
| B-N01 replay branch collision | Fixed | `3c335bb` |
| E-08, A2-04 probe identity | Fixed | `6b3f835`, `cd7ffc7` |
| E-22, E-23 Python capture guards | Fixed | `5a2d38e` |
| B-N02 unknown permission keys | Fixed | `f124d55` |
| D-N02 unbounded bash drain | Fixed | `80860be` |
| D-N03, D-N04 argument and probe reporting | Fixed | `cd164dc` |
| B-N04 chat delimiters as byte runs | Fixed | `9c8f33c` |
| C-20 unbounded replay position | Fixed | `5451495` |
| A2-17 NaN perplexity | Fixed | `c39883b` |
| A2-16, A2-45 probe parameters and width | Fixed | `0eef55c` |
| B-N05 unterminated transcript line | Fixed | `012767d` |
| B-N06 super-quadratic encode | Fixed | `5c54812` |
| C-06 session and cache desync | Fixed | `0be0fa2` |
| A1-02, C-07, C-R10, A1-12, A1-13, A1-14, H-37, B-N03 load-time validation | Fixed | `4d87d36` |
| C-05 partial snapshot restore | Fixed | `1cd2053` |
| A1-03 prefill atomicity | Fixed | `2708b24` |
| C-19 pattern zero aborts the binary | Fixed | `4d87d36` |
| C-21, F-27 blocking send on a runtime thread | Fixed | `9c8f33c` (allowlist), open for the sink itself |
| G-32, G-36, G-29 hot-path and snapshot cost | Fixed | `6250256`, `58d0ebf` |
| G-33 recorder row copy | Fixed | `ba50453` |
| H-54, H-58, H-59, H-60 hygiene | Fixed | `8dca140`, `322a83e`, `810a562`, `eea51f9` |
| I-39 through I-66 documentation drift | Fixed | `2097c32`, `164d4a1`, `7ccac37` |
| A1-R04 dual RoPE | Refuted | — |
| A1-R06 snapshot geometry | Refuted | — |
| C-R12 recorder atomicity | Refuted as reported; the contract it named was later documented, and the violation fixed | `53858ef` |
| D-N01 recording coverage | Refuted | — |
| G-28 startup fingerprint | Waived, measured | — |
| A1-R05 monitor position semantics | Waived, documented | — |
| A1-15 fixture blind to QK scale and GQA mapping | Fixed | 7c53120 |
| G-31 full-vocabulary sampler allocation | Deferred | — |
| G-35 grammar mask simulates every piece | Deferred | — |
| G-30 GGUF bytes retained across restore | Closed by the loader change, not fixed as reported | `P10.6`/`P10.7` |
| G-34 prompt tokenized twice per turn | Deferred | — |
| A2-18 `sha256_hex` copies its input | Deferred | — |

**Four findings are deferred**, carrying neither a fix nor a waiver. All
four are performance work the perf batch did not reach: G-31 and G-35 both
need a sampler and grammar redesign rather than a local change, G-34 needs
the rendered prompt cached across a turn, and A2-18 needs a streaming
SHA-256 in place of the copy in both duplicated implementations
(`harness/src/main.rs` and `runtime/src/bin/activation-capture.rs`).

G-30 was the fifth. It is closed rather than fixed: the loader no longer
holds the file bytes it described.

A sixth, A1-15, was deferred here and has since been fixed. It was not
performance work and mattered more: the committed fixture set `head_dim`
to `embed_dim / n_heads` and `n_kv_heads` to 1, so the two candidate QK
scales coincided and the blocked and interleaved GQA mappings agreed,
leaving the golden structurally blind to both conventions — the same class
of blindness A1-R03 demonstrated. The second fixture that separates them
landed in `7c53120`.

Two findings were closed that no package owned, both raised by an
integrator auditing a batch rather than by a finder: `Session::restore`
left a loaded cache behind an empty token list on a length disagreement,
wedging the session permanently (`b6c42b2`), and `RecordingModel` violated
the atomicity contract the trait had just gained (`53858ef`).

**G-28 waived with measurement.** The startup fingerprint cannot be made
meaningfully faster while producing the same value, and the value gates
snapshot compatibility so it cannot change. Three exact reformulations of
the FNV-1a recurrence moved throughput under one percent, an L1-resident
run ruled out memory bandwidth, and substituting a two-cycle rotate for the
four-cycle multiply produced exactly the predicted 2.0x, pinning the loop
to its dependency-chain latency. The remedy is caching the fingerprint
beside the checkpoint, which is a change to the binary's startup path
rather than to the hash.

### Verification

Every test count below is the campaign's own measurement at the commit
named, and none of them are maintained. Read them as evidence that the gate
ran, not as the current size of the suite.

The rust gate exits 0 at every commit and twice at HEAD, the second run
watching for the tracing flake Campaign 1 recorded. 306 tests pass and 6
remain ignored: five Tier 3 entries in `cuda_acceptance.rs` and
`eval_acceptance.rs` whose bodies are `unimplemented!()` stubs awaiting the
GPU, and a chat-template token-stream golden needing a
workstation-generated tokenizer fixture rather than a GPU.

At
`7c53120` the workspace runs 383 passing tests with five ignored: the Tier
3 stubs are all that remain, because the chat-template golden was removed
rather than filled in when the Gemma 3 markers it tested were replaced.
Writing a replacement is still outstanding. The Python steps
`ruff format --check` and `ruff check` pass; `mypy` reports three
pre-existing missing-stub errors in unmodified `training/` files, and
`uv lock --check` is skipped per the documented sandbox caveat.

Numerics are unchanged by the entire campaign. The determinism oracles
recorded before the first commit — the sha256 of a `--record` flight log
and of a capture shard at a fixed seed — match at HEAD. The final
integrator additionally compared 45 digests over the raw f32 bit patterns
of every logit at every position, spanning nine prompt lengths, steered
and tapped runs, forks, save and restore round trips, and six synthetic
geometries the committed fixture cannot express, plus 49 `--record`
outputs across four prompts, four seeds, and three token budgets. All
identical.

Performance was measured in process, not end to end: the release binary on
the 46 KB fixture has a noise floor of roughly plus or minus 8 to 10
percent, wider than most of the effects, and an attempt to bisect an
apparent end-to-end regression across batch boundaries produced no
monotonic trend, confirming the method rather than a regression. At E4B
head width the forward pass improves 21.9 percent in total, and
allocations fall from 39 to 1 per token.

### Result

Discovery: sixty-four findings confirmed under independent refutation
attempts across ten dimensions and two passes, four refuted with rationale,
twenty-eight low-severity findings recorded without verification by the
severity policy.

Remediation: fifty-six fixed, two waived with recorded rationale, six
deferred, plus two defects found by integrators and fixed. Four critical
findings are closed — two of them silent-corruption classes that no test
could have caught, because the golden fixture and the numpy reference
encoded the same wrong convention.

### Followups

Recorded, not fixed, with the reason each was left:

- **The five deferred performance findings** (G-30, G-31, G-34, G-35,
  A2-18) listed under Dispositions. Each needs a redesign rather than a
  local change, and none alters output.
- **Tokenizer parity with Hugging Face is unproven.** The only test that
  would catch a segmentation divergence is ignored, awaiting a
  workstation-generated golden. On a real Gemma vocabulary the
  user-defined set includes whitespace runs, so the special-piece change
  could affect ordinary prose; nothing in a sandbox can settle it. This is
  the highest-value item on this list.
- **The Python gate here is not authoritative.** `uv run --no-sync`
  resolves ruff 0.15.8 and mypy 1.19.1 against pins of 0.15.21 and 2.3.0,
  because the repository virtual environment is empty. The Python results
  above should be reconfirmed on the workstation.
- **The bash tool drops late background output silently.** The bound is
  correct, but output arriving after the drain grace is discarded with no
  note, unlike the timeout path which appends one.
- **`TokenSink::emit` still panics on a runtime thread.** The prompt-side
  injection risk is closed, but the sink itself is unchanged, and
  `subagents.rs` and `stream.rs` remain unreachable from the binary.
  Wiring them is a feature decision.
- **Caching the model fingerprint** at startup, per G-28 above.
- `serde_yaml` is deprecated and unmaintained; replacing it is a dependency
  change, deferred under the pin-then-fix rule.
- No MSRV is declared, there is no continuous integration, and neither
  coverage nor mutation testing is measured.
- `replay` still accepts a divergent branch that writes only positions the
  first never claimed; catching that needs a branch-keyed recording format.
- `run_capture` on the Python side still accepts duplicate layers from its
  caller even though the loader now refuses the artifact, and a same-length
  reorder of the dataset preserving counts still passes.
- k-quant (Q4_K, Q6_K) support, prefill batching, and streaming capture
  shards remain design-scale work, unchanged by this campaign.

### Closed after the campaign: the cross-language contract dimension

Cross-language contract is a standing dimension in the `bug-bash` skill, and
it is the one this campaign did not run as a dimension of its own. A review
afterwards found all three Rust/Python artifact seams specified in prose and
enforced by nothing: the capture manifest key list typed out twice with no
shared definition, an observability catalog line hand-formatted rather than
serialized and matching the Python writer only by coincidence, and Rust GGUF
loaders that had never parsed a byte either Python exporter produced.

The three contracts now live in `schemas/` and are asserted at both writers,
with three tests running the real code on the other side of each seam. Two
divergences turned out to be intentional and are pinned as differences —
`dataset_sha256` hashes raw bytes on one side and newline-normalized text on
the other, and `tokenizer_fingerprint` is built from different parts — so a
one-sided change to either now fails a test instead of silently
reinterpreting the artifacts already on disk. No defect was found in the
seams themselves: the 1-based control-vector numbering and the 0-based probe
layer compose correctly across the boundary. What changed is that this is
now demonstrated rather than believed. See `schemas/README.md`.

One enabling change made the seams testable in the sandbox at all. `xxhash`,
the only non-stdlib dependency of `interp/manifest.py`, was imported at
module scope, which made every `interp` artifact module unloadable on an
interpreter that lacks it. Moving it into `file_xxhash` follows the lazy
heavy-import convention `interp/probes/capture.py` already states.

Real-weight, GPU, and container-build claims remain with runbook entries
RH.0 through RH.13. The one host command this campaign specifically needs
is RH.5, the first `--record` run against the real checkpoint. Note what
RH.5 as written does and does not establish: its procedure asserts that
`replay_file` rebuilds the recording without error, which is
replayability, not agreement with any reference. Closing the numerics loop
on real weights needs a comparison step RH.5 does not yet carry — a
reference the recorded logits can be checked against, such as a llama.cpp
run over the same prompt at the same quantization. Until that step exists,
the numerics evidence in this campaign covers synthetic geometry only.
