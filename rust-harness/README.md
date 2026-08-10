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

Numerics come from a hand-rolled single-threaded CPU backend
(`runtime/src/model/cpu.rs`) implementing a Gemma-4-style forward pass from a
Hugging Face checkpoint. It is a strict subset of the published architecture, so
`cpu::check_supported_checkpoint` refuses a checkpoint carrying per-layer embeddings, a
shared KV cache, MatFormer feed-forward widths, or dual RoPE — naming the
feature rather than loading the model and silently ignoring the extra weights.
Its logits match an independent numpy reference over the committed synthetic
checkpoint.

Around that sit the observability and context layers: a flight recorder
(`--record`) that replays a run as a `MockModel`, `tracing` spans with a
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
