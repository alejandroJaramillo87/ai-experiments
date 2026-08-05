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

## Model reference

| Document | Description |
|----------|-------------|
| [gemma-4-ingestion.md](gemma-4-ingestion.md) | What the runtime requires to ingest a Gemma 4 checkpoint: the GGUF metadata contract, the Hugging Face to GGUF translation, and every architecture feature the backend refuses by name |

## Forward looking

| Document | Description |
|----------|-------------|
| [optimization-candidates.md](optimization-candidates.md) | Recorded menu of x86 CPU backend optimizations, each blocked on host verification and a measured baseline |

## Related documentation

- [../inference/README.md](../inference/README.md) — the llama.cpp and vLLM serving stack, which the harness does not replace
- [../verification/rust-harness.md](../verification/rust-harness.md) — host verification runbook
- [../verification/rust-harness-qa.md](../verification/rust-harness-qa.md) — sandbox quality audit
- [../interpretability/rust-harness-interp.md](../interpretability/rust-harness-interp.md) — running the write-back interpretability path on this runtime
