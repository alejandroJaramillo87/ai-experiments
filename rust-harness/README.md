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
Cargo.toml        workspace: harness, runtime
harness/          bin crate: agent loop, tools, permissions, chat, streaming, obs records
runtime/          lib crate: sessions, sampler, grammar, KV cache, checkpoint, tokenizer, model
fixtures/         committed fixtures: synthetic tiny-model checkpoints, numpy
                  reference logits and generator script, probe and
                  control-vector GGUFs
                  written by the Python exporters plus their generator;
                  real-model Tier 3 fixtures pending
```

Utility binaries: `checkpoint-info` dumps a checkpoint's config and tensor table,
then derives an architecture summary and the backend's compatibility verdict
(architecture reconciliation against the real checkpoint); `tiny-model`
writes the synthetic fixture checkpoint; `activation-capture` writes last-token
residual activations in the interp capture shard format (consumed unchanged
by the parent repository's `interp/probes` toolkit).

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

Current status: the rung bodies and the machinery are implemented and
unit-tested against a scripted `MockModel`; the reference artifacts are
outstanding, so no rung has yet run against a real oracle and no
output-quality claim is made.

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
vectors, in-band probe monitoring), conversation compaction, and a KV cache
serialized on exit and restored at startup — continuation logits after a restore
are bit-identical to an unbroken run.

Remaining workstation work: architecture reconciliation against the real E4B
checkpoint, the rr acceptance run, and the Tier 3 fixtures. The cuda backend
slots in at the same `Model` seam.

## Quality gate

```bash
$ make quality        # fmt check, clippy -D warnings, lockfile check,
                      # rustdoc -D warnings, cargo test
```

Five offline steps, cheapest first, mirroring the parent repository's Python
gate, which chains this one. cargo-deny is a deliberate non-step: its advisory
database needs the network.

The cross-language tests hand artifacts to the real Python readers rather than to
a second Rust implementation, because a Rust misunderstanding of a format would
otherwise reproduce itself on both sides. `runtime/src/npz.rs` re-reads a written
shard through `np.load`; `harness/tests/probe_identity.rs` takes its SHA-256 from
`hashlib`; `python_exporter_contract.rs` parses GGUFs written by the interp
exporters; `obs_contract.rs` parses a run catalog through
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
