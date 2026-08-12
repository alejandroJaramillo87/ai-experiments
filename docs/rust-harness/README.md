# Rust harness documentation

Design and reference documentation for `rust-harness`, the single-binary Rust coding harness
with the Gemma 4 inference runtime linked in as a library crate.

This directory holds documentation only. The workspace itself lives at `/rust-harness`, and its
`README.md` covers layout, test tiers, and build commands. Verification records for the harness
live in `/docs/verification/`, not here.

## Design

| Document | Description |
|----------|-------------|
| [runtime-design.md](runtime-design.md) | Harness and runtime architecture: agent loop, the library-crate runtime, and the implementation roadmap |
| [observability.md](observability.md) | Observability and debuggability as first-class code, in three phases preceding the CUDA backend |
| [observation-frames.md](observation-frames.md) | What the agent loop and chat mode contribute to the training and interpretability programs, and the acceptance bar for new features |
| [live-events.md](live-events.md) | Live run-state events as JSONL, with a serve mode and HTML view that also replays finished runs |

## Forward looking

| Document | Description |
|----------|-------------|
| [rust-harness-optimization-candidates.md](../optimizations/ideation/rust-harness-optimization-candidates.md) | Recorded menu of x86 CPU backend optimizations, each blocked on host verification and a measured baseline |

## Roadmap

Phases and their acceptance checks. This lives here rather than in the workspace
`README.md` because a roadmap describes a moment: nothing forces an update when the moment
passes, so it belongs beside the design documents that share that property rather than in the
file a newcomer reads to learn what the harness is.

| Phase | Deliverable | Acceptance check |
|-------|-------------|------------------|
| A-prime | Hand-rolled CPU reference backend (in place; workstation acceptance pending) | Logits match the numpy reference; rr record/replay with reverse-continue |
| A | Load quantized E4B, single forward pass | Logits match the Hugging Face reference within quantization tolerance |
| B | Decode loop, persistent KV cache, delta prefill | Turn latency scales with delta size, not transcript size |
| C | Grammar-constrained tool-call sampling | Zero malformed tool calls over 200 forced generations |
| D | Full harness loop | Scripted coding task completes end to end |
| E | Subagent fork and interleave | Spawn without prefix re-prefill; tool/generation wall-clock overlap |

## Related documentation

- [../inference/README.md](../inference/README.md) — the llama.cpp and vLLM serving stack, which the harness does not replace
- [../verification/rust-harness.md](../verification/rust-harness.md) — host verification runbook
- [../verification/rust-harness-qa.md](../verification/rust-harness-qa.md) — sandbox quality audit
- [../interp/rust-harness-interp.md](../interp/rust-harness-interp.md) — running the write-back interpretability path on this runtime
