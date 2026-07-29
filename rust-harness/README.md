# rust-harness

Single-binary local coding harness with an integrated LLM inference runtime,
written in Rust. The harness adapts the agent architecture of the
[claude-code-from-scratch](https://github.com/FareedKhan-dev/claude-code-from-scratch)
tutorial; the runtime targets Gemma 4 E4B exclusively and links into the harness as
a library crate, giving the agent loop direct ownership of the KV cache across
conversation turns.

The full design is recorded in the parent repository at
`docs/inference/rust-harness-runtime-design.md`. This directory is self-contained
and is intended for later extraction into its own repository.

## Layout

```
Cargo.toml        workspace: harness, runtime
harness/          bin crate: agent loop, tools, permissions, sessions, streaming
runtime/          lib crate: sessions, sampler, grammar, KV cache, GGUF, model
fixtures/         golden fixtures (chat-template token IDs, reference logits)
```

## Testing

Testing is test-first and tiered:

- **Tier 1 — unit tests**: pure logic, no model, no GPU. Run with `cargo test`.
- **Tier 2 — mock integration tests**: the full harness, scheduler, grammar, and
  KV-bookkeeping stack against a scripted `MockModel`. Run with `cargo test`.
- **Tier 3 — host acceptance tests**: real E4B forward pass, logit matching against
  a Hugging Face reference, and latency measurement. Marked `#[ignore]`, require
  the `cuda` feature, and run only on the RTX 5090 workstation.

## Quality gate

```bash
$ make quality        # cargo fmt --check, cargo clippy -D warnings, cargo test
```

## Roadmap

| Phase | Deliverable | Acceptance check |
|-------|-------------|------------------|
| A | Load quantized E4B, single forward pass | Logits match the Hugging Face reference within quantization tolerance |
| B | Decode loop, persistent KV cache, delta prefill | Turn latency scales with delta size, not transcript size |
| C | Grammar-constrained tool-call sampling | Zero malformed tool calls over 200 forced generations |
| D | Full harness loop | Scripted coding task completes end to end |
| E | Subagent fork and interleave | Spawn without prefix re-prefill; tool/generation wall-clock overlap |
