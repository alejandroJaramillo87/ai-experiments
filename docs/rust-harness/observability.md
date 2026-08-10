# Rust harness observability and debugging design

Design for building observability and debuggability into the `rust-harness`
runtime as first-class code, rather than attaching external debuggers after the
fact. Companion to the [runtime design](runtime-design.md); this
document defines three implementation phases that precede the CUDA backend
(roadmap phase A) and give it a debuggable reference to be validated against.

## Table of contents

- [Purpose](#purpose)
- [Phase 1: flight recorder and profiling scaffolding](#phase-1-flight-recorder-and-profiling-scaffolding)
- [Phase 2: tracing](#phase-2-tracing)
- [Phase 3: rr-friendliness via a simple CPU backend](#phase-3-rr-friendliness-via-a-simple-cpu-backend)
- [Interpretability substrate](#interpretability-substrate)
- [Run-level records (obs contract)](#run-level-records-obs-contract)
- [Parked: CPU-served subagents](#parked-cpu-served-subagents)
- [Relation to the existing roadmap](#relation-to-the-existing-roadmap)

## Purpose

Debugging an inference runtime from outside — breakpoints and watchpoints on a
foreign binary — is archaeology. Because this runtime is owned source behind
narrow trait seams, the idiomatic approach is to design the instrumentation in:
every stage of the token loop emits structured records, tensor-level taps sit
at the `Model` trait boundary, and the numerics code is deterministic enough to
replay under a time-travel debugger.

This mirrors the repository's existing observability principle on the Python
side (`common/obs`: human lines are secondary, structured records are the
durable artifact). The three phases below are ordered by dependency: the
recorder and the trace schema exist before the first real forward pass is
written, so the numerics code is born instrumented rather than retrofitted.

Each phase is independently useful and lands one at a time. Phases 1 and 2 are
pure sandbox work against the existing mock-driven test suite; phase 3 is the
first real numerics code in the project.

## Phase 1: flight recorder and profiling scaffolding

Sandbox work against the repository as it stands. No changes to `Session`.

### RecordingModel

A wrapper implementing the `Model` trait (`runtime/src/model/mod.rs`) that
delegates to any inner `Box<dyn Model>` and appends every `forward()` call —
input positions and output logits — to a recording file. Selection is a
constructor-site decision in `main.rs` (a `--record <path>` flag or
environment variable); the session and harness layers are unchanged.

### Replay

A recording is loadable as a `MockModel` script. This makes real-backend runs
recorded on the workstation reproducible in the sandbox: harness, sampler, and
grammar behavior can be debugged against real logits with no GPU present.

### Divergence bisection

The recorder turns the canonical numerics bug — output diverges from the
reference at some token position — into a file diff. Record the same prompt
through this runtime and through a reference implementation, diff the logit
streams, and the first divergent position identifies the exact `forward()`
call to inspect. This is the location step; phase 3 provides the cause-finding
step (reverse execution from the divergent value).

### Profiling scaffolding

- A `release-with-debug` Cargo profile: release optimization with debug info
  retained, required by both `cargo flamegraph` and the phase 3 rr workflow.
- A `make flame` target wrapping `cargo flamegraph` for the workstation.

Meaningful profiles wait for a real backend: `MockModel` has no hot path, so
this step delivers working machinery, and the first useful flame graph arrives
with phase 3.

### Acceptance

A record/replay round-trip test against `MockModel` passes in the sandbox: a
recorded run, loaded back as a script, reproduces the original token sequence.

## Phase 2: tracing

Sandbox work. The `tracing` crate becomes a workspace dependency and the
existing stack is instrumented with structured spans:

| Span | Location | Key fields |
|------|----------|------------|
| `session.sync` | `runtime/src/session.rs` | delta length (tokens), cached length |
| `session.step` | `runtime/src/session.rs` | position, chosen token |
| `sampler.sample` | `runtime/src/sampler.rs` | temperature, candidate count |
| `grammar.advance` | `runtime/src/grammar.rs` | state, allowed-token count |
| `tool.exec` | `harness/src/tools.rs` | tool name, duration, output bytes |
| `agent.fork` | `harness/src/subagents.rs` | parent cached length, task delta length |

The span and field names form a stable schema — the Rust analogue of the
Python `common/obs` vocabulary — shared by debugging, the blog write-ups, and
future benchmark analysis. The `session.sync` delta-length field makes the
phase B acceptance property (turn latency scales with delta size, not
transcript size) directly visible in a trace timeline.

Subscriber selection lives in `main.rs`: no subscriber by default (spans
compile to near-zero cost when disabled), `--trace-chrome <path>` writes a
Perfetto-loadable JSON via `tracing-chrome`. Live viewing through
`tracing-tracy` is a workstation-side follow-on, not part of this phase.

Ordering rationale: instrumentation must exist before the first real backend
so that the phase 3 CPU code carries spans from its first commit.

### Acceptance

Span structure is asserted in tests with a capturing test subscriber, and a
mock-run Chrome trace renders the prefill/decode timeline in Perfetto.

## Phase 3: rr-friendliness via a simple CPU backend

The first real numerics code: a correctness-first CPU forward pass for
Gemma 4 E4B, chosen for simplicity of implementation over speed.

### Backend choice

Resolved at implementation time: the backend is hand-rolled — naive
dequantize-then-matmul in owned code (`runtime/src/model/cpu.rs`),
single-threaded plain f32 loops. Candle CPU was considered and rejected:
upstream Candle has a `gemma4` module but no quantized-GGUF Gemma 4 path, so
the loader work remains either way; its threaded CPU kernels complicate rr
determinism; and the correctness-and-readability bar favors owned naive
code. The bar is correctness and readability, not throughput.

### Architecture assumption and reconciliation

The forward pass implements a Gemma-3-style architecture: RMSNorm by the
stored weight, with no `(1 + weight)` shift — Gemma 4 initializes the norm
weight to ones, unlike Gemma 1 through 3, and
[runtime-design.md](runtime-design.md) records how long that took to
establish. Embedding scaled by the square root of the embedding
dimension, grouped-query attention with per-head QK-norm and NEOX RoPE,
interleaved sliding-window/global layers, a GeGLU MLP with tanh GELU, four
norms per layer, and an LM head tied to the token embedding. Whether the real
E4B checkpoint deviates (MatFormer or per-layer-embedding inheritance from
Gemma 3n is the open design-doc risk) is resolved with evidence: the
`checkpoint-info` binary (`runtime/src/bin/checkpoint-info.rs`) dumps the
config and the tensor table of a checkpoint; the human runs it against the real E4B on
the workstation and the reported output is reconciled against the
implementation. Numerics are validated in the sandbox against an independent
numpy implementation of the same forward pass
(`rust-harness/fixtures/reference_logits.py`) over a committed synthetic
checkpoint (`rust-harness/fixtures/tiny-gemma4/`, written by the
`tiny-model` binary).

One storage decision is recorded as deferred: the K/V cache holds the full
sequence for every layer, and the sliding window is applied as an attention
mask only. This is numerically identical to a capped cache but unbounded in
memory; the `kv.rs`-budgeted ring buffer for sliding-window layers is
deferred to the real-model work, where cache size matters.

### Scope: debugging and reference only

The CPU backend makes no serving claim. Estimated decode speed on the
Ryzen 9950X is bandwidth-bound and low (single-digit tokens/second for large
models; estimates only, per the benchmarking protocol). Its two jobs are:

1. Be the reference implementation the CUDA backend is diffed against through
   the phase 1 recorder, complementing the Tier 3 logit-matching fixture test.
2. Be recordable under rr, which cannot record GPU ioctls.

### rr workflow

rr (record and replay) provides deterministic time-travel debugging: record a
failing run once, then replay it in gdb with reverse execution. Requirements
and deliverables:

- Determinism: the sampler is already seeded and the decode path holds no wall
  clock; the standing rule is to keep threads out of the decode path (the bash
  tool's drain threads are outside the recorded path).
- `make debug-record` / `make debug-replay` targets: build with the
  `release-with-debug` profile, run the case under `rr record`, replay with
  `rr replay`.

### Runbook: divergence bug class

The canonical numerics bug is a logit stream that diverges from the
reference at some position. The sequence from symptom to cause:

1. Record both runs with `--record` (phase 1) and diff the two JSONL logit
   streams; the first divergent position identifies the exact `forward()`
   call to inspect.
2. Record the failing run under rr:

   ```bash
   $ make debug-record ARGS='--model fixtures/tiny-gemma4.gguf <task>'
   $ make debug-replay
   ```

3. Inside `rr replay` (a gdb session), stop at the divergent call, then set
   a hardware watchpoint on the divergent value — a KV cache slot or a
   logit — and run backwards to the write that produced it:

   ```
   (rr) break cpu.rs:forward_one if position == <divergent position>
   (rr) continue
   (rr) watch -l logits[<token>]
   (rr) reverse-continue
   ```

The same sequence applies one level deeper for interpretability work:
watch a single residual-stream element (`watch -l x[<j>]` inside the layer
loop) and `reverse-continue` to the attention or MLP write that produced
it. The residual stream is exposed programmatically through the `Tap` seam;
see the
[interp substrate design](../interp/rust-harness-interp.md).

### Acceptance

Host-side check on the workstation: a small CPU forward pass is recorded and
replayed under rr, with reverse-continue reaching a chosen write. The human
runs the commands and reports output back; the reported output is recorded
verbatim per the repository's operating rules.

## Interpretability substrate

The CPU backend carries an interpretability seam beyond the debugging
machinery above: a `Tap` trait observing the per-layer residual stream,
additive control-vector steering, and an `activation-capture` binary
writing the interp capture shard format. The design, its compatibility
contract with the Python interp stack, and the constraint it places on
future backends are recorded in
[rust-harness-interp.md](../interp/rust-harness-interp.md).

## Run-level records (obs contract)

The three phases above cover the token loop; run-level observability —
which invocations happened, how long they took, which failed — converges
on the Python obs stack by contract rather than by code.
`harness/src/obs.rs` writes the same JSONL shapes `common/obs` defines
(see `docs/observability/design-and-roadmap.md`):

- A run id in the shared `R-YYYYMMDD-HHMMSS-xxxxxx` format (UTC stamp,
  six hex characters).
- An event file `${OBS_LOG_DIR}/harness/<run_id>.jsonl` with the base
  keys `ts, level, run_id, cli, command, pid, event, msg`, holding
  `cmd.start` (with `argv`), `fail` on error (`exc_type`, `exc_msg`
  capped at 4096 characters, `exit_code`), and `cmd.end` (`exit_code`,
  `duration_s`). Timestamps carry an explicit `+00:00` offset; `command`
  is `task` or `chat`.
- One catalog line appended to `${OBS_LOG_DIR}/harness/runs.jsonl` at run
  end, written as a single append so concurrent writers interleave whole
  lines.
- A shared flock held on the event file for the run's duration, so the
  Python retention pruner treats it as live.

Because the Python `runs`, retention, and viewer tooling are stateless
JSONL readers, they consume harness runs unchanged; the files are the
integration seam. That is now asserted rather than asserted about. The key
sets live in `schemas/obs_record.json` and both writers are checked against
them, and `harness/tests/obs_contract.rs` feeds a harness-written catalog to
the real `common.obs.catalog.read_runs` and requires the record back with its
fields intact. The check matters because the reader is permissive by design:
`read_runs` skips a line it cannot parse, so a drifted writer would show an
empty runs table rather than an error. The catalog line is also formatted by
hand rather than through serde, so nothing in the type system holds it to the
Python writer's shape. Degradation matches the Python contract: without a
writable `${OBS_LOG_DIR}` (the agent sandbox), the writer is a silent
no-op, and usage errors exit before a run id exists. Cross-process
propagation is specified from the receiving side: when a parent process
sets `OBS_RUN_ID`, the harness records it as `parent_run_id` on
`cmd.start` while minting its own run id.

## Parked: CPU-served subagents

Recorded for later, deliberately out of scope until phase 3 exists and its
performance is measured.

The idea: the main agent runs a large model on the GPU while subagents run a
smaller model (E4B class) on the CPU. Subagents receive their task as text and
encode it with their own tokenizer, so no cross-device KV transfer is needed —
forking a GPU-resident KV cache to a CPU backend is explicitly rejected (it
would force a full re-prefill and continue from non-identical numerics). The
harness already permits the topology: `Agent` takes any `Generator`, and
`SessionGenerator` binds a generator to whichever session and tokenizer it is
constructed with, so heterogeneous agents are a constructor-site choice.

The open question is whether the simple CPU backend reaches usable E4B decode
speed; that is a measurement, not a design change, and is deferred.

## Relation to the existing roadmap

Phase 3 inserts as phase A-prime before the CUDA phase A in the
[implementation roadmap](runtime-design.md#implementation-roadmap):
the CPU backend is implemented and debugged first (sandbox-runnable,
rr-recordable), and the CUDA backend is then validated against it via the
phase 1 recorder in addition to the existing Tier 3 fixtures. Phases 1 and 2
precede all backend work. The Tier 3 fixture generation and acceptance tests
are unchanged.
