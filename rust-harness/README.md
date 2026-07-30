# rust-harness

Single-binary local coding harness with an integrated LLM inference runtime,
written in Rust. The harness adapts the agent architecture of the
[claude-code-from-scratch](https://github.com/FareedKhan-dev/claude-code-from-scratch)
tutorial; the runtime targets Gemma 4 E4B exclusively and links into the harness as
a library crate, giving the agent loop direct ownership of the KV cache across
conversation turns.

The full design is recorded in the parent repository at
`docs/inference/rust-harness-runtime-design.md`; the observability and
debugging phases (flight recorder, tracing, rr-debuggable CPU backend) are
designed in `docs/inference/rust-harness-observability.md`. This directory is
self-contained and is intended for later extraction into its own repository.

## Layout

```
Cargo.toml        workspace: harness, runtime
harness/          bin crate: agent loop, tools, permissions, chat, streaming, obs records
runtime/          lib crate: sessions, sampler, grammar, KV cache, GGUF, tokenizer, model
fixtures/         committed fixtures: synthetic tiny-model GGUF, numpy reference
                  logits and generator script; real-model Tier 3 fixtures pending
```

Utility binaries: `gguf-info` dumps GGUF metadata and the tensor directory
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

## Current status

Everything above the forward pass is implemented and tested against the
scripted mock: agent loop, tools (with wall-clock bash timeout and capped
tool output), permissions, transcript, chat template, SentencePiece unigram
tokenizer (vocabulary read from GGUF metadata), sessions with a hard context
budget, sampler, grammar-constrained decoding, subagent fork/interleave, and
the perplexity evaluator. Observability phase 1 is in place: a flight
recorder (`runtime/src/model/record.rs`) wraps any `Model` behind the
binary's `--record <path>` flag, writes every forward call to a JSONL
stream, and loads a recording back as a `MockModel` script for sandbox
replay; a `release-with-debug` Cargo profile and `make flame` target
provide the profiling scaffolding. Observability phase 2 instruments the
stack with `tracing` spans (`session.sync`, `session.step`,
`sampler.sample`, `grammar.advance`, `tool.exec`, `agent.fork`,
`model.forward`); no subscriber is installed by default, and the binary's
`--trace-chrome <path>` flag writes a Perfetto-loadable trace.

Observability phase 3 lands the first real numerics: a hand-rolled,
single-threaded CPU backend (`runtime/src/model/cpu.rs`) implementing a
Gemma-4-style forward pass (RMSNorm, GQA with QK-norm and NEOX RoPE,
interleaved sliding-window/global attention, GeGLU MLP, tied LM head) from
GGUF weights (F32, F16, Q8_0 dequantization). The binary runs it with
`--model <path.gguf>`; the `SessionGenerator` seam wires the agent loop
through tokenizer and session. Numerics match an independent numpy
reference over the committed synthetic checkpoint, and the harness runs a
task end to end on the CPU backend in the sandbox. Beyond one-shot tasks,
`--chat` runs an interactive REPL (no tools or permissions; the transcript
persists to `session.jsonl`, so a restarted chat resumes), and
`--temperature`, `--top-p`, `--seed`, and `--max-tokens` expose the
sampling parameters. A reproducible containerized build lives in the
parent repository (`docker/rust-harness/Dockerfile.rust-harness`, run via
`make harness-build` / `harness-run` / `harness-chat` there): a pinned
rust toolchain building with `--locked`. Each invocation also writes
run-level observability records (`harness/src/obs.rs`) in the parent
repository's shared obs contract — run id, `cmd.start`/`fail`/`cmd.end`
events, and a catalog line under `${OBS_LOG_DIR}/harness/` — readable by
the existing Python tooling unchanged. Architecture
reconciliation against the real E4B GGUF (`gguf-info`), the rr acceptance
run, and the Tier 3 fixtures are the remaining workstation work; the cuda
backend (roadmap phase A) slots in at the same `Model` seam.

The CPU backend also carries an interpretability seam: a `Tap` trait
observing the per-layer residual stream, control-vector steering
(`--control-vector`, same GGUF layout llama.cpp uses), the
`activation-capture` binary, and an in-band probe monitor (`--probe`, a
probe GGUF exported from a trained linear probe) scoring every position
of the probe's layer, all golden-tested against the numpy reference. Design and compatibility contract:
`docs/interpretability/rust-harness-interp.md` in the parent repository.

## Quality gate

```bash
$ make quality        # fmt check, clippy -D warnings, lockfile check,
                      # rustdoc -D warnings, cargo test
```

Five offline steps, cheapest first, mirroring the parent repository's
Python gate; the parent's root `make quality` chains this gate. cargo-deny
(advisories, licenses) is a deliberate non-step: its advisory database
needs the network, so it belongs to CI or workstation tooling.

## Roadmap

| Phase | Deliverable | Acceptance check |
|-------|-------------|------------------|
| A-prime | Hand-rolled CPU reference backend (in place; workstation acceptance pending) | Logits match the numpy reference; rr record/replay with reverse-continue |
| A | Load quantized E4B, single forward pass | Logits match the Hugging Face reference within quantization tolerance |
| B | Decode loop, persistent KV cache, delta prefill | Turn latency scales with delta size, not transcript size |
| C | Grammar-constrained tool-call sampling | Zero malformed tool calls over 200 forced generations |
| D | Full harness loop | Scripted coding task completes end to end |
| E | Subagent fork and interleave | Spawn without prefix re-prefill; tool/generation wall-clock overlap |
