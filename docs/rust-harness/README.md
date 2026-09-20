# Rust harness documentation

Design and reference documentation for `rust-harness`, the single-binary Rust coding harness
with the Gemma 4 inference runtime linked in as a library crate.

This directory holds documentation only. The workspace itself lives at `/rust-harness`, and its
`README.md` covers layout, test tiers, build commands, the command-line surface (`--help`,
`--status`, exit codes), and what a run reports while it works. Verification records for the
harness live in `/docs/verification/`, not here.

## Design

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [live-events.md](live-events.md) | Design for live visibility into a harness run in progress: the harness emits run-state events as JSONL through the existing observability contract, and the parent repository's observability tooling gains a small serve mode plus an HTML view that polls the event file. |
| [observability.md](observability.md) | Design for building observability and debuggability into the `rust-harness` runtime as first-class code, rather than attaching external debuggers after the fact. |
| [observation-frames.md](observation-frames.md) | A reference for what the harness's two interaction modes — the agent loop and chat mode — contribute to the training and interpretability programs, and for the feature bar that follows. |
| [roadmap.md](roadmap.md) | The scheduled work for the rust harness and its interpretability seam, in five phases plus a parallel track. |
| [runtime-design.md](runtime-design.md) | Design for a single-binary local coding harness written in Rust, with the LLM inference runtime linked in as a library crate rather than accessed over HTTP. |

<!-- inventory:pages:end -->

## Forward looking

| Document | Description |
|----------|-------------|
| [rust-harness-optimization-candidates.md](../optimizations/ideation/rust-harness-optimization-candidates.md) | Recorded menu of x86 CPU backend optimizations, each blocked on host verification and a measured baseline |

## Backend phases

Backend bring-up phases and their acceptance checks. This lives here rather than in the
workspace `README.md` because a roadmap describes a moment: nothing forces an update when the
moment passes, so it belongs beside the design documents that share that property rather than
in the file a newcomer reads to learn what the harness is.

These lettered phases are the backend ladder and are not the same list as the R-numbered work
items in [roadmap.md](roadmap.md); the two intersect only where a work item names a phase as a
dependency.

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
- [../verification/rust-harness/rust-harness.md](../verification/rust-harness/rust-harness.md) — host verification runbook
- [../verification/rust-harness/rust-harness-qa.md](../verification/rust-harness/rust-harness-qa.md) — sandbox quality audit
- [../interp/rust-harness-interp.md](../interp/rust-harness-interp.md) — running the write-back interpretability path on this runtime
