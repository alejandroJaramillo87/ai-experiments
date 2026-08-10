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
runtime/          lib crate: sessions, sampler, grammar, KV cache, GGUF, tokenizer, model
fixtures/         committed fixtures: synthetic tiny-model GGUF, numpy reference
                  logits and generator script, probe and control-vector GGUFs
                  written by the Python exporters plus their generator;
                  real-model Tier 3 fixtures pending
```

Utility binaries: `gguf-info` dumps GGUF metadata and the tensor directory,
then derives an architecture summary and the backend's compatibility verdict
(architecture reconciliation against the real checkpoint); `tiny-model`
writes the synthetic fixture GGUF; `activation-capture` writes last-token
residual activations in the interp capture shard format (consumed unchanged
by the parent repository's `interp/probes` toolkit).

## Testing

Testing is test-first and tiered:

- **Tier 1 — unit tests**: pure logic, no model, no GPU. Run with `cargo test`.
- **Tier 2 — mock integration tests**: the full harness, scheduler, grammar, and
  KV-bookkeeping stack against a scripted `MockModel`. Run with `cargo test`.
- **Tier 3 — host acceptance tests**: real E4B forward pass, logit matching against
  a Hugging Face reference, and latency measurement. Marked `#[ignore]`, require
  the `cuda` feature, and run only on the RTX 5090 workstation.

## Output correctness

Machinery tests (Tiers 1 and 2) cannot detect a numerically wrong forward
pass: a bad RoPE base or quant kernel still emits fluent text. Output quality
is guarded by three Tier 3 layers, ordered by diagnostic precision, each
catching what the previous one misses:

1. **Logits match** (`runtime/tests/cuda_acceptance.rs`): one forward pass on
   a fixed prompt against a Hugging Face reference dump, within quantization
   tolerance. Pinpoints raw numeric divergence at a single position.
2. **Perplexity** (`runtime/src/eval.rs`, `runtime/tests/eval_acceptance.rs`):
   chunked negative log-likelihood over a committed WikiText-2 sample, within
   1% of a recorded llama.cpp `llama-perplexity` reference. The regression
   scalar for every engine change; catches diffuse degradation.
3. **Greedy-decode goldens** (`runtime/tests/eval_acceptance.rs`):
   temperature-0 continuations of fixed prompts byte-match committed token
   sequences. Catches decode-loop and KV bugs that stay inside per-position
   tolerance but flip an argmax.

Current status: the perplexity math is implemented and unit-tested against
hand-calculated values on the mock; the three real-model layers are ignored
tests awaiting the cuda backend and workstation fixtures. No output-quality
claim is made until they pass.

## What works today

The agent loop and everything above the forward pass are implemented and tested
against a scripted `MockModel`: tools, permissions, transcript, chat template,
tokenizer, sessions with a hard context budget, sampler,
grammar-constrained decoding, subagent fork/interleave, and the perplexity
evaluator.

Numerics come from a hand-rolled single-threaded CPU backend
(`runtime/src/model/cpu.rs`) implementing a Gemma-4-style forward pass from GGUF
weights. It is a strict subset of the published architecture, so
`cpu::check_supported` refuses a checkpoint carrying per-layer embeddings, a
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
GGUF, the rr acceptance run, and the Tier 3 fixtures. The cuda backend slots in
at the same `Model` seam.

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
