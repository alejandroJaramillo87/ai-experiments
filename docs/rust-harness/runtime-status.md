# What the runtime does today

The implemented surface of the workspace, layer by layer: the agent loop and everything
above the forward pass, the hand-rolled f32 CPU backend and the Gemma 4 features it
implements, the observability and context layers around it, the served endpoint and
steering, and what the workstation runbook still holds open. A living page, corrected
as the tree moves; the dated evidence for each claim is in
[../verification/rust-harness/rust-harness.md](../verification/rust-harness/rust-harness.md).

## What works today

The agent loop and everything above the forward pass are implemented and tested
against a scripted `MockModel`: tools, permissions, transcript, chat template,
tokenizer, sessions with a hard context budget, sampler,
grammar-constrained decoding (installed on the task path in triggered mode, so a
turn is either one well-formed tool call or free text), subagent
fork/interleave, and the perplexity evaluator.

Numerics come from a hand-rolled f32 CPU backend
(`runtime/src/model/cpu.rs`) implementing a Gemma-4-style forward pass from a
Hugging Face checkpoint. Every matrix-vector product and the tied LM head split
their output rows across cores; that split is bit-identical at every width, so
the golden tests pin the parallel path and `--single-thread` is for rr rather
than for correctness. Weights are held in the width the checkpoint published —
bf16 stays bf16, widened per use — which is likewise bit-identical, because
widening is a shift and the arithmetic is f32 either way. Within a core, four
output rows are folded in one pass so the compiler can place them in the lanes
of a vector register; that is SIMD without reordering anything, since each row
is still one sequential fold over the input in its original order, and it is
worth 1.87x on E4B. Everything else — the norms, attention, the residual
updates — is a plain sequential loop, with no wall-clock concessions. Per-layer embeddings, the
shared KV cache, MatFormer feed-forward widths and dual RoPE are implemented;
`cpu::check_supported_checkpoint` keeps each as a coherence check and refuses a checkpoint
whose config and tensors describe different models, or that carries a tensor family the
forward pass does not consume — naming the feature rather than loading the model and
silently ignoring the extra weights.
Its logits match an independent numpy reference over the committed synthetic
checkpoint.

Around that sit the observability and context layers: a flight recorder
(`--record`) that replays a run as a `MockModel` — carrying the sampling
parameters and the per-step selection, so a replay reproduces the transcript
rather than only the distribution it was drawn from, and auditable token by
token against `transformers` with `fixtures/audit_recording.py`, which
teacher-forces the recorded sequence and compares every recorded row, and checked for
structure with `make replay-check RECORDING=<file>` — `tracing` spans with a
Perfetto-loadable `--trace-chrome` whose `--trace-level` reaches from the
session timeline down to one span per layer and, on request, per kernel, checked with
`make trace-check TRACE=<file>`, an interp seam (activation capture, control
vectors, in-band probe monitoring) whose taps are keyed on the site a row came
from rather than on the layer alone, reach inside the layer as far as
post-softmax attention weights, and whose scores can be written out beside
the byte span of text each position produced, conversation compaction, and a KV cache
serialized on exit and restored at startup — continuation logits after a restore
are bit-identical to an unbroken run.

Every run appends a line to the shared observability catalog, browsable from the
host with `make harness-runs` (the image carries no Python). It is the same
reader training and interp use, pointed at this tier:
`docs/rust-harness/observability.md` covers the record shapes and the flags.
Four other binaries write there too, under the same catalog with their own names
as the command: `ladder`, so a conformance climb is browsable beside the runs it
judges, and `activation-capture`, `attention-capture` and `layer-diff`, so a
capture directory or a pasted verdict can be traced back to the invocation that
made it. All five record the fold, the width and the thread count they used —
the configuration whose absence let `DotKind` default to `Sequential` unnoticed
for months. `layer-diff` also records the two tolerances its verdict was reached
under, being the one binary of the five that writes no artifact of its own. The
ladder and the harness additionally get a run directory holding `metrics.jsonl`,
which `python -m training report --run <dir>` renders with no code of its own;
the three capture binaries write a header and a load timing and nothing more.

`--serve <port>` puts that whole stack behind the two HTTP paths the parent
repository's interp pipeline calls, `/v1/chat/completions` and `/v1/health`, so
an experiment can reach the model the way it already reaches llama.cpp while
the taps stay in the process that answers. The sampling flags become
per-request defaults and a request may override any of them including the seed,
which is what an OpenAI-compatible server usually accepts and ignores. HTTP is
hand-rolled over `std::net` — two paths with a length-delimited body do not
earn a framework, and the gate is offline by contract — so there is no
keep-alive and a chunked request body is refused rather than mis-parsed; no
caller sends either. Generation serializes behind a mutex the health handler
never takes, because `Model` is `Send` but not `Sync` and a health probe queued
behind a minutes-long CPU reply would read as a dead service. `harness/src/stream.rs`
is the streamer an SSE body would consume if a streaming client ever appears;
nothing in interp streams today, so nothing does.

Steering rides on the same service. `--control-vector` takes the safetensors
image `interp/steering/export.py` writes, one 1-D f32 tensor per steered block
named `block.<index>`, 0-based; `--control-vector-scale` multiplies it and
`--control-vector-layer-range "start end"` drops the blocks outside an
inclusive band as the image is read, so the forward pass sees only the
directions it must add. The image was GGUF until R10, when retiring llama.cpp
steering left this the only reader and the format became ours to pick. The
container equivalent is `harness-serve` in the parent repository's
`docker-compose.rust-harness.yaml`, which `make steer-up` recreates.

What the workstation has not yet confirmed is recorded in the runbook rather than
here, so that this page cannot describe a moment: at the time of writing RH.37
(`--trace-level` on E4B), RH.38 (the container chat REPL) and RH.39 (a restored session
continues bit-identically) are pending, and RH.40 (subagent fork and interleave on real
weights) is blocked on the user surface. The cuda backend slots in at the same `Model`
seam, behind the invariants `docs/rust-harness/conformance.md` states.
