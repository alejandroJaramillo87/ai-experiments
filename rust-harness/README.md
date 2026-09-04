# rust-harness

Single-binary local coding harness with an integrated LLM inference runtime,
written in Rust. The harness adapts the agent architecture of the
[claude-code-from-scratch](https://github.com/FareedKhan-dev/claude-code-from-scratch)
tutorial; the runtime targets Gemma 4 E4B exclusively and links into the harness as
a library crate, giving the agent loop direct ownership of the KV cache across
conversation turns.

The harness is positioned as an instrumented inference substrate for the
parent repository's training and interpretability pipelines, with the
coding-agent loop as its native workload. It does not compete with the
llama.cpp and vLLM services on serving speed or model coverage; its value is
the owned seams those services cannot offer — in-band activation capture,
steering, probe monitoring, a flight recorder, and deterministic KV restore —
which make agent runs reproducible and publishable as experiment logs.

The full design is recorded in the parent repository at
`docs/rust-harness/runtime-design.md`; the observability and
debugging phases (flight recorder, tracing, rr-debuggable CPU backend) are
designed in `docs/rust-harness/observability.md`. This directory is
a permanent part of that repository, not a staging area for extraction, so the
artifact seams it shares with the Python pipelines may be tested across the two
trees rather than specified in prose.

## Layout

```
Cargo.toml        workspace: harness, obs, runtime
harness/          bin crate: agent loop, tools, permissions, chat, streaming
obs/              lib crate: run-level observability records; the only member
                  holding unsafe, two libc::flock calls
runtime/          lib crate: sessions, sampler, grammar, KV cache, checkpoint, tokenizer, model
fixtures/         committed fixtures: synthetic tiny-model checkpoints, numpy
                  reference logits and generator script, the probe
                  (safetensors plus sidecar) and control-vector image written
                  by the Python exporters plus their generator;
                  real-model Tier 3 fixtures pending
```

Utility binaries: `checkpoint-info` dumps a checkpoint's config and tensor table,
then derives an architecture summary and the backend's compatibility verdict
(architecture reconciliation against the real checkpoint); `tiny-model`
writes the synthetic fixture checkpoint; `activation-capture` writes last-token
residual activations in the interp capture shard format (consumed unchanged
by the parent repository's `interp/probes` toolkit); `attention-capture` writes
post-softmax attention weights for chosen layers and query heads over one
prompt, which is how Gemma 4's interleaved sliding and global attention can be
looked at directly; `ladder` runs the five-rung correctness gate against a
transformers reference; `layer-diff` reports per-layer divergence of the
residual stream against a reference capture; and `long-context-golden` writes
the committed long-prefill logits fixture.

## Command line

`harness --help` is the authoritative list of flags, generated from the same
table the parser reads, so it cannot describe a flag the binary does not accept
or omit one it does. `harness --version` prints the build. Both write to stdout
and exit 0; a usage error goes to stderr and exits 2.

An unrecognized flag is a usage error rather than task text, so a misspelling
like `harness --modle m "do x"` stops rather than running a task named after the
mistake. Task text that legitimately begins with a dash goes after a bare `--`,
which ends flag parsing.

The other seven binaries answer `--help` and `-h` on the same terms, from their
own tables in `runtime/src/cli.rs`: help on stdout at exit 0, a usage error on
stderr at exit 2, and an unrecognized flag named rather than ignored. They do
not take `--version`. Two of the changes are behavioural rather than additive.
`activation-capture` and `attention-capture` used to scan for each flag by
position and read no further, so a misspelling was silently dropped — with a
required flag among the casualties they reported `missing required flag`, which
named neither, and with every required flag present they ran with the wrong
configuration and said nothing; their usage errors also now exit 2 rather than
sharing exit 1 with a failed capture. And `tiny-model` and
`long-context-golden` take their destination positionally, so until this each
one wrote its fixtures to a file named `-h` when asked for help.

`harness --model <dir> --status` reports what the run would fail on and exits
without loading the model: the checkpoint and the CPU backend's verdict on it,
each interp artifact given, and the directories the run writes to. It reads
`config.json` and the safetensors header only, so it answers in milliseconds
rather than the time a checkpoint takes to load. Add `--deep` to hash the
weights and check each artifact's recorded `model_hash` against them; that is
opt-in because it reads the whole file.

Three exit codes mean three different things, which is the reason to prefer
`--status` over starting a run to see whether it starts: **0** clean, or
warnings only; **1** something would stop the run; **2** the command line was
wrong to ask. The report goes to stderr and stdout stays empty. No run record
is written — a query is not a run, the same rule that keeps `--help` out of the
catalog.

A run that fails anyway states its fix. Three failures — a probe whose width
disagrees with the model, an interp artifact fitted on other weights, and a
`--model` path with no checkpoint at it — print two indented lines under the
error, a `Solution:` and a `Command:` naming this run's real paths rather than
a placeholder. The advice reaches the terminal only: the run record's `exc_msg`
stays the diagnosis alone, so a `fail` event is still one line of cause and not
a page of counsel.

A run that succeeds says what it waited on. The two operations that dominate a
cold start each report once when they finish — `Model load:` at the end of the
banner, and `Prefill:` with the token count and rate before the reply. Both
figures are read off the obs phase that already records them, so the terminal
and `<run_id>.jsonl` cannot disagree. The prefill count is the *delta* the
forward pass actually ran, not the prompt length, which is what makes a second
turn's small number the visible evidence that the KV cache is being reused.
There is no progress animation and no terminal check: a pipe and a terminal get
the same bytes, so a runbook's pasted output never carries a carriage return.

## Testing

Testing is test-first and tiered:

- **Tier 1 — unit tests**: pure logic, no model, no GPU. Run with `cargo test`.
- **Tier 2 — mock integration tests**: the full harness, scheduler, grammar, and
  KV-bookkeeping stack against a scripted `MockModel`. Run with `cargo test`.
- **Tier 3 — the correctness ladder** (`runtime/src/ladder.rs`): the real
  checkpoint against `transformers`, on the workstation, under its own make
  target. It is not part of `make quality`.

Cutting across those: **the conformance invariants** (`runtime/src/conformance.rs`),
which are obligations on every backend rather than tests of one. They say that
how a request is internally divided — one batched pass, groups of 64, a token
at a time, logits for every position or only some — may not move a bit of any
row returned. `model::cpu` satisfies them today and the CUDA backend will have
to; they live in a normal module, not a `#[cfg(test)]` one, because a backend
in another crate cannot call into `cpu.rs`'s test module. They compare a
backend against itself, so they are silent about whether its numbers are right.

`runtime/tests/long_context.rs` drives them, and everything else that only
engages past the sliding window: the interleaved sliding/global split and the
shared-KV tail. Nothing else in the workspace reaches that far — the ladder's
longest single cache is L3's 256-token window against E4B's window of 512, so
no rung has crossed a real one. It is offline, since `synth::tiny_checkpoint`
builds a model in memory, and it runs in `make quality`.

Its last section is a different kind of assertion and is grouped apart for
that reason. Everything above it compares logits, which is what the KV storage
cap must *not* change; the storage arms read the cache extent directly, because
a suite that only compares logits cannot tell whether a memory optimization was
implemented at all. The file's mutation record is worth reading before changing
the eviction: correctness there has two independent guards, and defeating
either one alone shows up as memory rather than as numerics.

## Output correctness

Machinery tests (Tiers 1 and 2) cannot detect a numerically wrong forward
pass: a bad RoPE base or quant kernel still emits fluent text. Nor can a
golden test whose oracle we wrote ourselves — `fixtures/reference_logits.py`
was transcribed from the same architecture documents by the same reader as
`cpu.rs`, and two implementations of one belief agree perfectly. That gate
stayed green for nine work packages while the forward pass produced noise.

Output correctness is therefore measured against `transformers`, in five
rungs, each one only meaningful once the one above it passes:

| Rung | Measures | Reference |
|------|----------|-----------|
| L0 | token IDs for the rendered prompt | `tok.encode(text, add_special_tokens=False)` |
| L1 | per-layer residual stream and the stages before it | forward hooks on the decoder layers |
| L2 | the final-position distribution | `model(...).logits` |
| L3 | perplexity over a token window | the reference's per-token NLL |
| L4 | greedy continuations | `model.generate(do_sample=False)` |

L2 is KL divergence over the full distribution rather than a max-abs
tolerance or argmax agreement: KL in nats is scale-free, which is what a
262144-wide softcapped vocabulary needs, and a monotone error such as a
missing logit softcap cannot move an argmax at all.

The ladder short-circuits at the first failing rung and prints all five rows
regardless. A rung with no reference artifact reports `skipped` — missing
evidence, not contrary evidence — and does not stop the climb.

It runs against two oracles. Tier T is a tiny Gemma 4 that `transformers`
itself constructs, committed whole, fast enough for `make quality`. Tier E is
the real checkpoint on the workstation. See
[fixtures/reference/README.md](fixtures/reference/README.md) for the
artifacts and how they are generated.

Current status: **all five rungs ran against real E4B on 2026-08-14 and every
one passed, with nothing skipped** — L0 token ids, L1 residual stream (worst
cosine 1.000000 over 42 layers), L2 final distribution, L3 perplexity (delta
0.000032 over 254 tokens), L4 greedy continuation. The rung bodies remain
unit-tested against a scripted `MockModel` as well. The transcript is entry
RH.1b of [the workstation runbook](../docs/verification/rust-harness/rust-harness.md).

That is a numerics claim, not an output-quality claim: the ladder measures
agreement with `transformers`, and says nothing about whether the model's
answers are good.

### What the committed goldens are worth

`fixtures/*-logits.json` pin the CPU backend against `fixtures/reference_logits.py`
on the tiny synthetic checkpoints. Both sides are read as a cross-language
check, and that is a real guard against a one-sided slip.

It is not a correctness claim, for two reasons. The fixtures hold random
weights, so a wrong arrangement produces noise that looks exactly like the
right arrangement's noise. And both implementations were transcribed from the
architecture documents by the same reader, so a misreading lands in both and
they agree about it.

That is not hypothetical. Two defects — a `1/sqrt(head_dim)` applied to
attention logits Gemma 4 leaves unscaled, and a missing scale-free norm on the
value states — were present in both implementations through nine work packages
of green gate, and were found by reading `docs/models/gemma4-upstream.txt`
after a real checkpoint produced noise. The second contributes no tensor to a
checkpoint, so no name-based audit of the weights could have found it.

`fixtures/long-context-logits.json` is a third kind, and the distinction
matters. It is not compared against any reference: it holds rows this engine
produced at 200 tokens, recorded once, so that a later package changing how the
KV cache is stored has to either reproduce them or re-bless them deliberately.
A regenerate-and-compare fixture cannot do that job, because regenerating from
the current engine and comparing against the current engine agrees by
definition. Written by `runtime/src/bin/long-context-golden.rs`, read by
`runtime/tests/long_context.rs`.

It is a change detector and not an oracle, for the reason this section already
gives. What keeps it from freezing a wrong answer is ordering: the window
convention it encodes was checked against `transformers` first, at 150 tokens
over fifty crossings of the Tier T fixture's window of 3, and it was blessed
only after that passed.

It has since done its job once. Capping sliding-layer KV storage at the window
rewrote how the cache is stored and indexed, and these bytes reproduced
unchanged — which is the outcome that says the rewrite preserved behaviour,
and the only construct in the workspace that could have said so.

Real-weight verification is `runtime/src/bin/layer-diff.rs` against
`fixtures/reference_capture.py`, which compares the residual stream layer by
layer to transformers itself. Until that reports clean, the goldens are
regression pins and nothing more.

## What works today

The agent loop and everything above the forward pass are implemented and tested
against a scripted `MockModel`: tools, permissions, transcript, chat template,
tokenizer, sessions with a hard context budget, sampler,
grammar-constrained decoding, subagent fork/interleave, and the perplexity
evaluator.

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
updates — is a plain sequential loop, with no wall-clock concessions. It is a strict subset of the published architecture, so
`cpu::check_supported_checkpoint` refuses a checkpoint carrying per-layer embeddings, a
shared KV cache, MatFormer feed-forward widths, or dual RoPE — naming the
feature rather than loading the model and silently ignoring the extra weights.
Its logits match an independent numpy reference over the committed synthetic
checkpoint.

Around that sit the observability and context layers: a flight recorder
(`--record`) that replays a run as a `MockModel` — carrying the sampling
parameters and the per-step selection, so a replay reproduces the transcript
rather than only the distribution it was drawn from — `tracing` spans with a
Perfetto-loadable `--trace-chrome`, an interp seam (activation capture, control
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

Remaining workstation work: architecture reconciliation against the real E4B
checkpoint, the rr acceptance run, the Tier 3 fixtures, and the first `--serve`
run against real E4B. The cuda backend
slots in at the same `Model` seam.

## Quality gate

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

**Which interpreter runs them is resolved, not assumed** (`runtime/src/pyenv.rs`).
The workstation and the sandbox disagree about where a usable Python lives: on
the workstation `python3` usually leads to a pyenv shim carrying none of the
project's dependencies while the environment sits in `.venv`, and in a sandbox
the reverse holds, because `docker-compose.agents.yaml` masks `/work/.venv` with
an empty tmpfs and the image's own `/opt/pyenv` is first on `PATH`. Resolution
probes
`$HARNESS_PYTHON`, then `<repo>/.venv/bin/python3`, then `python3` on `PATH`, and
takes the first that can import what the caller names. Set `HARNESS_PYTHON` to
override.

A missing optional dependency reports `[SKIP]` and passes — the container build's
situation, since its context is `rust-harness/` alone. numpy is instead mandatory
in `npz.rs`, because a shard nothing re-read is an unverified shard.

**Read the output for `[SKIP]` lines rather than trusting the exit code.** The
`test` step passes `--nocapture` for that reason: cargo hides a passing test's
stderr, a skip is by definition a pass, and without it a gate that had silently
stopped exercising the Python side would look identical to one that exercised it.
Tier 3 tests still require the workstation.

The contracts themselves are in `schemas/` at the repository root. The roadmap
and the design documents live in `docs/rust-harness/` there.
