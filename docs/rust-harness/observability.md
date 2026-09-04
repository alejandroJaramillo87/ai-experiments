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
input positions and output logits — to a recording file. Whether to wrap is a
constructor-site decision in `main.rs` (the `--record <path>` flag); the
session and harness layers are unchanged.

The recording also carries the *selection*: the sampling parameters in its
header, and one line per sampled token naming the token, its position, and the
sampler's cumulative draw count. Sampling happens above the `Model` trait, so
the recorder reaches it through `Model::record_selection` — a default no-op
every backend inherits, called from `ActiveGeneration::step`.

### Replay

A recording is loadable as a `MockModel` script. This makes real-backend runs
recorded on the workstation reproducible in the sandbox: harness, sampler, and
grammar behavior can be debugged against real logits with no GPU present.
`replay` returns the script alone; `replay_with_selections` returns it
alongside the header's parameters and the ordered selections.

**Logits alone would reproduce the distribution and then re-roll the dice.**
This is what the selection lines are for. A recording that stated only what the
forward pass produced described the odds the run sampled under and never which
token it drew, so a replay agreed on every logit and disagreed on the
transcript, with nothing in either file to say so. The header's parameters let
a reader rebuild the sampler that made the choices rather than a default one
that would make others; the draw count lets a replay resume mid-stream instead
of restarting.

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
The stochastic case is the stronger one and is asserted separately, over flat
logits where every token is the sampler's choice: the replay rebuilds its
sampler from the recording's own header, matches both the transcript and the
final draw count, and a run at a different seed is asserted to diverge — the
negative without which the positive would pass on a recorder that wrote no
selections at all.

## Phase 2: tracing

Sandbox work. The `tracing` crate becomes a workspace dependency and the
existing stack is instrumented with structured spans:

| Span | Level | Location | Key fields |
|------|-------|----------|------------|
| `model.load` | info | `runtime/src/model/cpu.rs` | text_only |
| `checkpoint.read`, `checkpoint.parse`, `model.from_checkpoint`, `checkpoint.fingerprint` | debug | `runtime/src/model/cpu.rs` | — (sequential children of `model.load`) |
| `model.load.layer` | debug | `runtime/src/model/cpu.rs` | layer, owns_kv (one per layer read) |
| `tokenizer.encode` | info | `runtime/src/tokenizer.rs` | bytes, token count |
| `model.forward` | info | `runtime/src/model/cpu.rs` | start (cached length), len (tokens in the call) |
| `model.prelude` | debug | `runtime/src/model/cpu.rs` | n, ple (embedding and the per-layer-embedding stages) |
| `model.layer` | debug | `runtime/src/model/cpu.rs` | layer, n, kv_source, owns_kv |
| `model.layer.attn` | debug | `runtime/src/model/cpu.rs` | owns_kv, kv_len (cache length after the append) |
| `model.layer.mlp` | debug | `runtime/src/model/cpu.rs` | ffn_dim |
| `model.layer.ple` | debug | `runtime/src/model/cpu.rs` | ple_dim (only on a checkpoint with per-layer embeddings) |
| `model.lm_head` | debug | `runtime/src/model/cpu.rs` | rows (logit rows produced), n |
| `kernel.matmul` | trace | `runtime/src/model/cpu.rs` | out_dim, in_dim, n, threads (one per projection) |
| `kernel.attention` | trace | `runtime/src/model/cpu.rs` | n, kv_len, heads (the serial score-and-fold loop, once per layer) |
| `kernel.lm_head` | trace | `runtime/src/model/cpu.rs` | rows, in_dim, threads (the tied-head dot products, once per row) |
| `session.sync` | info | `runtime/src/session.rs` | delta length (tokens), cached length |
| `session.step` | info | `runtime/src/session.rs` | position, chosen token |
| `sampler.sample` | info | `runtime/src/sampler.rs` | temperature, candidate count |
| `grammar.advance` | info | `runtime/src/grammar.rs` | state, allowed-token count |
| `tool.exec` | info | `harness/src/tools.rs` | tool name, duration, output bytes |
| `agent.fork` | info | `harness/src/subagents.rs` | parent cached length, task delta length |

`model.forward` was missing from this table while being the span that dominates
every trace RH.6 recorded — 31.15 s of a 176-token run against 0.87 s in the
sampler. The info-level runtime spans are unconditional on any generation;
`grammar.advance` and `tool.exec` are workload-conditional and absent from a
plain completion.

The level column is the detail dial, selected with `--trace-level` (default
`debug`; meaningless without `--trace-chrome`, and refused without it). At
`info` a trace is the session-level timeline the first RH.6 traces carried. At
`debug` it also shows where inside `model.forward` and `model.load` the time
went: one span per layer with its three blocks, the prelude, the LM head, and
the load stages. At `trace` every projection and the attention loop are their
own spans; a `kernel.matmul` carries no projection name because the block
span above it says which one it is. For a run the shape of RH.3 (115 forwards
over 42 layers) `debug` adds about 20k spans and `trace` about 45k more,
each costing a few microseconds on the calling thread against kernels that
take milliseconds, so neither level is measurable in the wall clock, and the
file grows from a few megabytes to a few tens. Every field is an integer or a
bool already in scope: `tracing-chrome` formats arguments on the calling
thread, so a string field would be the whole cost of a span.

Worker threads carry no span. `matmul_panel` and the LM head spawn a
`std::thread::scope` per call; the kernel span is opened on the caller before
the scope and closed after the join, so it measures the whole parallel region
and the workers emit nothing. Nesting is contextual throughout — no span names
its parent — which `runtime/tests/tracing_spans.rs` asserts by parent name,
along with the count of every tier against the fixture's layer count and the
fact that a level filter drops exactly the tiers below it. The same file
asserts that the logits are bit-identical with and without a subscriber.

The span and field names form a stable schema — the Rust analogue of the
Python `common/obs` vocabulary — shared by debugging, the blog write-ups, and
future benchmark analysis. The `info` set is the schema; the `debug` and
`trace` names are stable too, but their fields may grow. Per-name totals come
two ways: in Perfetto's query view,
`select name, count(*), sum(dur) from slice group by name order by 3 desc`
is the table RH.6 built by hand, and `make trace-check TRACE=<file>` prints
the same table from the terminal and asserts the count relations in the table
above (one `model.layer` per layer per forward, the blocks equal to it, one
load with its stages, the prefill forward without a step), skipping when no
file is named. The `session.sync` delta-length field makes
the phase B acceptance property (turn latency scales with delta size, not
transcript size) directly visible in a trace timeline.

Subscriber selection lives in `main.rs`: no subscriber by default (spans
compile to near-zero cost when disabled), `--trace-chrome <path>` writes a
Perfetto-loadable JSON via `tracing-chrome`. Live viewing through
`tracing-tracy` is a workstation-side follow-on, not part of this phase.

Ordering rationale: instrumentation must exist before the first real backend
so that the phase 3 CPU code carries spans from its first commit.

### Acceptance

Span structure is asserted in tests with a capturing test subscriber, and a
mock-run Chrome trace loads in Perfetto.

**There is no `prefill` or `decode` span, and this sentence used to promise a
"prefill/decode timeline" that no span provides.** What a trace carries is
`session.sync`, which fires exactly once per run and *is* the prefill, and
`session.step`, one per decoded token. The timeline is readable, but by
reading those two rather than by finding spans under those names.

### Phases are a separate mechanism from spans

A span exists only while a `tracing` subscriber is installed, which happens
only under `--trace-chrome`. A **phase** is an obs record — `phase.start` and
`phase.end` in the run's event file, contracted in
[`schemas/obs_record.json`](../../schemas/obs_record.json) and written by both
this tier and `common/obs/core.py` — so it is present on every run whether or
not anyone asked for a trace.

They were deliberately not bridged. A subscriber `Layer` forwarding spans into
the run log would make obs output depend on subscriber installation, and
`--trace-chrome`'s `set_global_default` is a hard `expect`, so a second
installation panics. Phases are called directly instead. The span named
`model.load` and the phase named `model.load` are therefore two records of
the same work by two mechanisms: the phase brackets `build_generator` on every
run, the span brackets `CpuModel::load` inside it only under a trace, and the
span is the one with children.

The harness writes three: `model.load` around the whole of `build_generator`,
then `prefill` and `decode` around the two halves of every generation. The
three capture binaries write `model.load` and nothing else — they have one
operation each worth bracketing, and it is the one they spend most of their
wall clock on. The ladder writes `model.load` and one per rung attempted, `rung.L0` through
`rung.L4` — which is what makes the per-rung wall clock visible outside the
run's own timing block; `render` deliberately keeps it out of the table,
because it is the one figure in a result that is not reproducible.

`prefill` and `decode` are the split an operator wants first, because a slow
turn is either a slow prompt or a slow decode and the two have different
causes. They come from inside `Session::generate`, which is the seam all three
modes share, so one pair lands per agent turn, per chat turn and per served
request. **Not per token**: the event file has no bound on a server, and a
record pair for every token would make it grow with generation rather than
with requests. The per-token series is `metrics.jsonl`, which is what that
file is for.

Two mechanics make that possible, both worth knowing before touching the code.
A `RunLog::phase` guard borrows the log, so it cannot bracket a call that
needs `&mut self` on the object holding it — `obs::EventSink` exists for that
case, handing back a guard that owns its own descriptor and borrows nothing.
And a forked session does **not** inherit the sink: forks are advanced
alongside their parent by the subagent interleaver, and phase records carry no
session identity, so two sessions writing one event file would produce
overlapping pairs whose durations are wrong rather than missing.

## Phase 3: rr-friendliness via a simple CPU backend

The first real numerics code: a correctness-first CPU forward pass for
Gemma 4 E4B, chosen for simplicity of implementation over speed.

### Backend choice

Resolved at implementation time: the backend is hand-rolled — widen-then-matmul
in owned code (`runtime/src/model/cpu.rs`), plain f32 loops, single-threaded at
the time. Candle CPU was considered and rejected: upstream Candle's `gemma4`
module does not cover this family's per-layer geometry, so the loader work
remains either way; its threaded CPU kernels complicate rr determinism; and the
correctness-and-readability bar favors owned naive code. The bar is correctness
and readability, not throughput.

The backend has since acquired one form of threading, and it does not reopen
that decision. `cpu.rs`'s `par_rows` splits matvec and the tied LM head across
output rows, which reorders no arithmetic and so leaves the logits identical
bit-for-bit at every width — the objection to Candle's kernels was that
threading a reduction does not. `--single-thread` restores the serial path for
rr, which serializes threads regardless.

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
   make debug-record ARGS='--model fixtures/tiny-gemma4 <task>'
   make debug-replay
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
`obs/src/lib.rs` writes the same JSONL shapes `common/obs` defines
(see `docs/observability/design-and-roadmap.md`).

Five binaries write these records today: `harness`, `ladder`, and the three the
interp work runs — `activation-capture`, `attention-capture` and `layer-diff`.
The other three are deliberately silent: `checkpoint-info` is a sub-second
inspector run interactively dozens of times, `long-context-golden` and
`tiny-model` are fixture writers whose artifact is the record, and `tiny-model`
runs inside the gate, so instrumenting it would make `make quality` write to
the log root. The rule is to record a run someone will later want to find, not
a question someone asked and got an answer to.

All five use `cli = "harness"` with the binary name as `command`, so one catalog, one
retention policy and one `runs` view serve the whole workspace — symmetric
with how the training and interp tiers each own one. That makes `runs` a
reserved command name: `common/obs/catalog.py` hides rows whose `command` is
literally `runs`, so a binary of that name would log correctly and never
appear in the table. `harness/src/cli.rs` asserts none is.

What a run leaves behind:

- A run id in the shared `R-YYYYMMDD-HHMMSS-xxxxxx` format (UTC stamp,
  six hex characters).
- An event file `${OBS_LOG_DIR}/harness/<run_id>.jsonl` with the base
  keys `ts, level, run_id, cli, command, pid, event, msg`, holding
  `cmd.start` (with `argv`), `fail` on error (`exc_type`, `exc_msg`
  capped at 4096 characters, `exit_code`), and `cmd.end` (`exit_code`,
  `duration_s`). Timestamps carry an explicit `+00:00` offset.
- A resolved-configuration header, written once immediately after the model
  loads: `run.header` from the harness, `ladder.header` from the ladder. Two
  events rather than one, because `run.header`'s field set is asserted for
  key-set equality against a real harness run — a ladder has no sampler, no KV
  cache and no context window, and admitting it under the same event would
  mean moving six fields to `optional`, retiring the assertion that catches a
  field silently dropped from the harness header. The keys the two share are
  spelled identically.
- `phase.start` and `phase.end` bracketing coarse spans, with `phase` and a
  `duration_s` rounded to three decimals. Written by both stacks; see
  "Phases are a separate mechanism from spans" above. The harness writes
  `model.load`, `prefill` and `decode`; the ladder writes `model.load` and one
  per rung; the three capture binaries write `model.load` alone, which is most
  of what two of them spend.
- One catalog line appended to `${OBS_LOG_DIR}/harness/runs.jsonl` at run
  end, written as a single append so concurrent writers interleave whole
  lines. Its `run_dir` is the run's own directory below.
- A run directory `${OBS_LOG_DIR}/harness/<run_id>/`, holding
  `metrics.jsonl`: one report-envelope header record carrying that run's header
  fields, then one record per step. Task and chat write one record per decoded
  token with `tokens_per_s` and `rss_gb`, bounded by `--max-tokens`; serve
  writes one per completed request, because a server has no run boundary to
  bound a per-token file against; the ladder writes exactly five, one per rung,
  with `rung`, `status`, `rung_duration_s` and the rung's own measures.
  Contracted in [`schemas/run_metrics.json`](../../schemas/run_metrics.json).

  **The three capture binaries write no run directory at all.** Two of them
  have no series to plot, and a capture's per-example count is a number its
  manifest already carries; a run directory that exists only to hold a header
  is one the retention pruner's `R-*.jsonl` glob would never reclaim.
  `runtime/tests/capture_obs.rs` asserts the absence rather than leaving it to
  the reader, because adding a `metrics_header` call is a one-line change whose
  consequence is not local to the line.

  `step` is an ordering key and nothing more. The generation modes make it a
  token index and the ladder makes it a rung index — five ordered categories,
  not evenly spaced points — and no reader here assumes otherwise.

  The directory is created on the first metrics write rather than at run
  start. The retention pruner globs `R-*.jsonl`, which a directory named
  `<run_id>` never matches, so an eagerly created empty one would survive
  forever.
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

## Browsing harness runs

The harness image carries no Python, so browsing happens on the host, where
training and interp runs are already browsed:

```bash
make harness-runs
make harness-runs ARGS='-n 50 --failed'
uv run --no-sync python -m common.obs.runs --cli harness --json
```

`common/obs/runs.py` is the shared command surface behind
`python -m training runs` and `python -m interp runs`; `--cli harness` points the
same reader at `${OBS_LOG_DIR}/harness/runs.jsonl`. Nothing in the reader is
tier-specific — `catalog.run_command` uses the CLI name only to build the path —
so the harness needed no reader of its own and gets no separate one.

Browsing writes nothing. The standalone entry point does not route through the
traced dispatch: a catalog query is not a run, and dispatching would both append
rows the default view then filters out and prune the event files the command was
asked to display.

### The dashboard

A run's charts come from the same command the training tier uses, pointed at the
run directory the catalog names:

```bash
uv run --no-sync python -m training report --run ${OBS_LOG_DIR}/harness/<run_id>
```

It writes `run.html` beside `metrics.jsonl`, joins the event file through the
header's `run_id` and `cli` for the log pane, and needs no harness-specific
code: `report.py` takes `--run` verbatim and checks only that it is a
directory. The command is spelled `training report` because that is where it
lives, not because it is training-only — the same reasoning as the shared
`runs` reader above, and the reason the two training-flavoured hint strings in
that command were left alone rather than reworded.

`common/obs/viewer.js` gained two chart groups for this: `Process memory (GB)`
over `rss_gb`, and `Rung duration (s)` over `rung_duration_s`. Both are
additive — a training run carries neither and renders them empty, exactly as a
CPU-only harness run renders the existing device-memory group empty.

A ladder run charts only its durations. The rungs' own numbers — `worst_cosine`,
`mean_kl`, `ppl_delta` — are in `metrics.jsonl` under slugified names but
deliberately not charted: they are different quantities in different units, and
a line drawn through them at consecutive rung indices would mean nothing. They
are there to be read with `jq` and diffed between two runs. The table on stdout
remains the ladder's answer.

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
