# Rust harness documentation

Design and reference documentation for `rust-harness`, the single-binary Rust coding harness
with the Gemma 4 inference runtime linked in as a library crate.

This directory holds documentation only. The workspace lives at `/rust-harness`, and its
README routes to the three crates and the fixtures, each with a table of its sources and
tests rendered from the tree. Verification records live in `/docs/verification/`, and the
user documentation (building, every flag, the artifact formats) is the site under `/site/`.

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [command-line.md](command-line.md) | What the `harness` binary and the seven utility binaries do with their arguments: the generated `--help`, usage errors and their exit code, the `--status` preflight that answers without loading the model, the three-line remedy a failing run prints, and the two cost lines a run reports while it works. |
| [conformance.md](conformance.md) | The two families of test that hold the workspace to something other than its own numbers: the conformance invariants every backend owes, which say that how a request is internally divided may not move a bit of any row it returns, and the cross-language contract tests, which hand an artifact this workspace writes to the real Python reader that consumes it. |
| [correctness-ladder.md](correctness-ladder.md) | The seven-rung gate that measures the engine against `transformers`, rung by rung: what each rung compares, the artifact it reads and the generator that writes it, the tolerance it is judged against at each of the two tiers, the perplexity gate at L3 in full, what the loader refuses before a rung runs, and the sizes, flags and records of a climb. |
| [fixtures.md](fixtures.md) | Every input under `rust-harness/fixtures/` that a test loads: what wrote it, what reads it, whether it is committed, and what each kind of fixture can and cannot prove. |
| [live-events.md](live-events.md) | Design for live visibility into a harness run in progress: the harness emits run-state events as JSONL through the existing observability contract, and the parent repository's observability tooling gains a small serve mode plus an HTML view that polls the event file. |
| [observability.md](observability.md) | Design for building observability and debuggability into the `rust-harness` runtime as first-class code, rather than attaching external debuggers after the fact. |
| [observation-frames.md](observation-frames.md) | A reference for what the harness's two interaction modes — the agent loop and chat mode — contribute to the training and interpretability programs, and for the feature bar that follows. |
| [roadmap.md](roadmap.md) | The scheduled work for the rust harness and its interpretability seam, in five phases plus a parallel track. |
| [runtime-design.md](runtime-design.md) | Design for a single-binary local coding harness written in Rust, with the LLM inference runtime linked in as a library crate rather than accessed over HTTP. |
| [runtime-status.md](runtime-status.md) | The implemented surface of the workspace, layer by layer: the agent loop and everything above the forward pass, the hand-rolled f32 CPU backend and the Gemma 4 features it implements, the observability and context layers around it, the served endpoint and steering, and what the workstation runbook still holds open. |
| [testing.md](testing.md) | How the workspace under `rust-harness/` is tested, tier by tier: the unit tests inside each source file, the integration tests that spawn the binaries, the correctness ladder that holds the engine to `transformers`, the conformance invariants every backend owes, the cross-language contract tests that hand written artifacts to the real Python readers, and the gate that runs the offline part of all of it on every push. |

<!-- inventory:pages:end -->

## Reading order

Read [runtime-design.md](runtime-design.md) for what the harness is and why it is built
this way, then [runtime-status.md](runtime-status.md) for what is implemented and what the
workstation has verified, then [testing.md](testing.md) for how any of that is known: it
routes to the ladder, the fixtures and the contracts. [command-line.md](command-line.md)
is the behaviour behind the flags; [observability.md](observability.md) and
[live-events.md](live-events.md) are the observability designs; [roadmap.md](roadmap.md)
is the parked work.

## Forward looking

| Document | Description |
|----------|-------------|
| [rust-harness-optimization-candidates.md](../optimizations/ideation/rust-harness-optimization-candidates.md) | Recorded menu of x86 CPU backend optimizations, each blocked on host verification and a measured baseline |

## Backend phases

The backend bring-up phases from the design, with where the workstation runbook stands on
each. These lettered phases are not the R-numbered work items in [roadmap.md](roadmap.md);
the two intersect only where a work item names a phase as a dependency. The entries cited
are in [../verification/rust-harness/rust-harness.md](../verification/rust-harness/rust-harness.md).

| Phase | Deliverable | Acceptance check | State |
|-------|-------------|------------------|-------|
| A-prime | Hand-rolled CPU reference backend | Logits match the numpy reference; rr record/replay with reverse-continue | verified: RH.1b (the ladder, all seven rungs on E4B) and RH.7 (rr, 2026-09-05) |
| A | Load the real checkpoint, single forward pass | Logits match the Hugging Face reference | verified on the HF checkpoint rather than the quantized one the design named: RH.1, RH.2, and RH.1b's L2 |
| B | Decode loop, persistent KV cache, delta prefill | Turn latency scales with delta size, not transcript size | verified: RH.4 |
| C | Grammar-constrained tool-call sampling | Zero malformed tool calls over forced generations | `runtime/tests/grammar.rs` offline; on real weights through RH.8 |
| D | Full harness loop | Scripted coding task completes end to end | verified: RH.8 |
| E | Subagent fork and interleave | Spawn without prefix re-prefill; tool/generation wall-clock overlap | `harness/tests/subagents.rs` offline; RH.40 on real weights is blocked on the user surface |

## Related documentation

- [../inference/README.md](../inference/README.md) — the llama.cpp and vLLM serving stack, which the harness does not replace
- [../verification/rust-harness/rust-harness.md](../verification/rust-harness/rust-harness.md) — host verification runbook
- [../verification/rust-harness/rust-harness-qa.md](../verification/rust-harness/rust-harness-qa.md) — sandbox quality audit
- [../interp/rust-harness-interp.md](../interp/rust-harness-interp.md) — running the write-back interpretability path on this runtime
- [../../rust-harness/README.md](../../rust-harness/README.md) — the workspace
