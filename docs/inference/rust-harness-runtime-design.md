# Rust harness and runtime design

Design for a single-binary local coding harness written in Rust, with the LLM
inference runtime linked in as a library crate rather than accessed over HTTP. The
harness adapts the agent architecture of the
[claude-code-from-scratch](https://github.com/FareedKhan-dev/claude-code-from-scratch)
tutorial; the runtime targets the Gemma 4 family exclusively, matching the
repository model policy recorded in
[unified_cli_investigation.md](../unified_cli_investigation.md). This document is a
design only; no Rust code exists in the repository yet. Findings and external facts
are current as of July 2026.

## Table of contents

- [Purpose and non-goals](#purpose-and-non-goals)
- [Background](#background)
- [Architecture](#architecture)
- [Runtime design](#runtime-design)
- [Harness design](#harness-design)
- [Model targeting](#model-targeting)
- [Implementation roadmap](#implementation-roadmap)
- [Risks and open questions](#risks-and-open-questions)

## Purpose and non-goals

The project builds a local coding agent — an agent loop with file, shell, and
planning tools — whose model inference runs in the same process on the RTX 5090,
with no network dependency. The primary purpose is pedagogical: understanding the
harness-runtime seam by owning both sides of it. The design serves one user with one generation in flight at any instant;
multi-tenant serving remains the job of the existing services. Parallel subagents
are in scope through cooperative interleaving (see
[Subagents](#subagents-fork-and-interleave)) rather than batched decode. Three
results are expected to outlive the learning exercise: measured delta-prefill turn
latency (roadmap phase B), reusable grammar-constrained tool-call decoding (phase
C), and KV cache forking for subagent spawn (phase E). Linking the runtime as a library gives the harness
direct ownership of the KV cache across conversation turns, which no HTTP inference
API exposes. The runtime is shaped by how an agent loop actually calls a model:

- **KV cache reuse across turns.** An agent loop resends a mostly-identical prompt
  every turn. The runtime persists the cache and prefills only the appended delta.
- **Grammar-constrained tool calls.** Sampling is constrained so the model cannot
  emit a malformed tool call.
- **Prefill-heavy optimization.** Agent turns read large contexts and emit short
  outputs, the opposite of chat workloads.

Non-goals, stated explicitly because each was considered and rejected:

- **Not a serving-stack replacement.** llama.cpp and vLLM remain the inference
  services on ports 8001-8003. This binary serves one local user, in-process.
- **No training and no interpretability seam.** The training and interp tiers stay
  on the Python/PyTorch stack. No activation hooks or steering interfaces.
- **No multimodality.** The vision components of Gemma 4 checkpoints are skipped at
  load time.
- **No custom CUDA kernels in version 1.** The tensor backend is
  [Candle](https://github.com/huggingface/candle); kernel authorship is deferred to
  the parked survey in [future-optimizations.md](future-optimizations.md).
- **No multi-provider abstraction.** One model family, hardcoded.

## Background

The Python tutorial implements a Claude-Code-style agent in roughly 5,000-7,000
lines: a core loop with tool dispatch, skills and context compression, async
subagent teams, streaming and permissions, prompt caching, and Redis mailboxes.
Mapped onto an all-Rust local port, the code divides into thirds:

- **Ported (~one third).** The agent loop, tool implementations, YAML permission
  rules, JSONL session persistence, context compression. Architecture translates
  directly; only syntax changes.
- **Removed (~one third).** Everything API-shaped: the `anthropic` SDK, retry and
  auth plumbing, API prompt caching (replaced by local KV cache reuse), the litellm
  multi-provider proxy, and Redis mailboxes. The tutorial phases 3 and 6 (teams,
  Redis) are dropped entirely; phases 1, 2, and 4 are in scope.
- **New (~one third, most of the effort).** The work the Anthropic API previously
  performed: chat template rendering, tool-call parsing via constrained decoding,
  token-level streaming from a local sampler, context-length budgeting against a
  hard VRAM limit, and the in-process harness-runtime interface.

## Architecture

A Cargo workspace with two crates and a model module:

```
rust-harness/
  Cargo.toml            # workspace
  harness/              # bin crate: agent loop, tools, permissions, sessions, TUI
  runtime/              # lib crate: inference engine
    src/
      session.rs        # Session handle owning the KV cache
      sampler.rs        # temperature, top-p, grammar constraints
      grammar.rs        # tool-call JSON grammar compilation
      kv.rs             # per-layer cache allocation and delta prefill
      gguf.rs           # weight loading, vision tensors skipped
      model/gemma4.rs   # static layer structs per size variant
```

The harness-runtime boundary is one in-process API:

```rust
let mut session = runtime.open_session(&model, kv_budget)?;
// Each turn: harness renders the transcript delta to tokens, then streams.
let stream = session.generate(delta_tokens, Some(&tool_grammar), sampling)?;
```

`Session` owns the KV cache for the conversation. The harness appends transcript
tokens; the runtime prefills only tokens beyond the cached prefix. A divergence
from the cached prefix (edited history, compression) truncates the cache to the
common prefix and re-prefills from there.

## Runtime design

**Tensor backend.** Candle with the CUDA feature, targeting the RTX 5090 only; no
CPU fallback path. Candle provides quantized GGUF kernels and flash-attention bindings;
Blackwell (sm_120) support under CUDA 13 must be verified before any other work
(see [Risks](#risks-and-open-questions)).

**Weights.** Quantized GGUF as the primary format. The loader maps tensors by name
into the static Gemma 4 structs and skips vision-tower tensors.

**Model graph.** No dynamic graph construction. Each Gemma 4 size variant is a set
of static structs with dimensions as constants; the compiler sees the whole forward
pass. This is viable only because the runtime targets one family.

**KV cache.** Gemma interleaves sliding-window and global attention layers.
Sliding-window layers cap their cache at window size regardless of context length,
so the total budget is computable exactly at startup: fixed cost for windowed
layers plus a per-token cost for global layers only. The cache is allocated once
per session and persists across turns. The expected win is turn latency
proportional to the appended delta rather than the full transcript; the magnitude
is unmeasured until phase B of the roadmap.

**Sampling and grammar.** Sampling runs on trimmed logits with temperature and
top-p. When the harness requests a tool call, a compiled JSON grammar for the tool
schemas masks invalid tokens each step, making malformed calls unrepresentable.
Gemma's vocabulary (~256K entries) makes the LM-head matmul and logits transfer
disproportionately expensive at small model sizes; the sampler keeps logits on the
GPU and copies only the sampled token, or a trimmed top-k slice, to the CPU.

## Harness design

Ports tutorial phases 1, 2, and 4:

- **Agent loop.** Render transcript, call `session.generate`, dispatch tool calls,
  append results, repeat until a final text response.
- **Tools.** `read`, `write`, `edit`, `bash`, `glob`, `todo` — thin wrappers over
  `std::fs` and `std::process::Command`.
- **Permissions.** YAML rule file evaluated before each tool dispatch, ported from
  the tutorial's permission governance (`serde_yaml`).
- **Sessions.** JSONL transcript per session with resume support (`serde_json`).
- **Streaming.** Tokens flow from the sampler through a `tokio` channel into the
  terminal renderer as they are produced.
- **Chat template.** The harness renders Gemma 4's turn format, including the
  function-calling convention, and golden-tests the rendered token IDs against the
  Hugging Face reference tokenizer for a fixed corpus of transcripts. Template bugs
  are the classic failure mode of local harnesses; the golden tests are mandatory
  before any end-to-end run.

Concurrency stays minimal in the main loop: one generation in flight, interrupt
handled by cancelling the generation future at a token boundary. The tutorial's
persistent teams, mailboxes, and FSM coordination remain out of scope; parallel
subagents are supported through the mechanism below.

### Subagents: fork and interleave

Subagents differ from server-side concurrency because they are cooperative: the
harness spawns them at a known moment and every subagent starts as a fork of the
parent's context. Two mechanisms exploit this:

- **KV cache forking.** At spawn, a subagent session forks the parent's KV cache
  at a token boundary — the shared prefix (system prompt, tool schemas, relevant
  transcript) is referenced immutably or copied, and only the subagent's task
  prompt is prefilled. This reuses the phase B delta-prefill machinery at spawn
  time and avoids re-prefilling the common prefix per subagent.
- **Interleaved scheduling.** Exactly one generation runs at any instant. A
  round-robin scheduler in the harness switches between subagent sessions at
  token-boundary yield points, so one agent decodes while another is blocked on a
  tool call. Subagent workloads are tool-heavy, so overlapping model time with
  tool time captures most of the practical speedup without touching the forward
  pass.

Batched decode — a single forward pass over multiple sequences of different
lengths — is explicitly out of scope; it requires ragged batching, per-sequence
sampler and grammar state, and is the start of rebuilding vLLM. VRAM cost of
interleaving is one KV budget per live subagent, comfortable for E4B on 32GB.

## Model targeting

Gemma 4 (released 2026-04-02, Apache 2.0) ships in multiple sizes spanning dense
edge models (E2B, E4B) and larger server-tier variants.

- **Target: E4B** (~4.5B effective parameters). Simplest full pipeline and aligned
  with the repository's Gemma 4 E-model policy. E4B is the sole supported model.
- **Skipped: E2B** (same code path as E4B, weaker coder) **and the larger
  server-tier variants** (large models are already served well by the existing
  services; their architectures are out of scope for this runtime).

Version coupling is accepted: the model module is the one component rewritten for a
future Gemma 5. Whether the E-series inherits MatFormer or per-layer-embedding
techniques from Gemma 3n is an open question against the model card; if it does,
the loader and structs in `model/gemma4.rs` absorb the difference.

## Implementation roadmap

Each phase ends with an acceptance check. GPU checks are host-side: the human runs
the stated command on the workstation and reports output back; the reported output
is recorded verbatim.

The runtime sits behind a `Model` trait with a scripted mock implementation, so the
harness, scheduler, KV bookkeeping, grammar, and chat-template layers build and
unit-test in a sandbox without a GPU. Phases C, D, and E are developed against the
mock first; phases A and B swap in the real backend on the workstation.

Before phase A, three observability phases (flight recorder, tracing, and an
rr-debuggable CPU reference backend) land per the
[observability design](rust-harness-observability.md); the CUDA backend is
validated against the CPU reference in addition to the Tier 3 fixtures.

| Phase | Deliverable | Acceptance check |
|-------|-------------|------------------|
| A | Load quantized E4B, single forward pass | Logits for a fixed prompt match the Hugging Face reference within quantization tolerance |
| B | Decode loop with persistent KV cache and delta prefill | Multi-turn script shows turn latency scaling with delta size, not transcript size; both paths measured |
| C | Grammar-constrained tool-call sampling | Zero malformed tool calls over 200 consecutive forced tool-call generations |
| D | Full harness loop (tools, permissions, sessions, streaming) | Scripted coding task (read file, edit, run test) completes end to end |
| E | Subagent fork and interleave | Two-subagent scripted task shows spawn without prefix re-prefill and wall-clock overlap of one agent's tool execution with the other's generation |

Performance numbers quoted before a benchmark run are estimates and must be
labeled as such; measured results follow the repository benchmarking protocol.

## Risks and open questions

- **Candle on Blackwell.** sm_120 support under CUDA 13 in candle/cudarc is the
  gating risk and is verified first, before any model work. A prior branch
  (`claude/llm-runtime-sm120-wmrdz5`) may hold relevant findings.
- **Candle Gemma 4 coverage.** Three months after release, Candle support for the
  new variants may be incomplete; the E4B layers may need direct implementation
  rather than reuse of Candle model code. The CPU reference backend
  (phase A-prime) is hand-rolled and does not depend on Candle; see the
  [observability design](rust-harness-observability.md) for the recorded
  decision.
- **Tap seam parity.** The CPU backend exposes a residual-stream `Tap` seam
  (interpretability capture and steering; see
  [rust-harness-interp.md](../interpretability/rust-harness-interp.md)).
  The CUDA backend must implement the same seam with identical semantics —
  post-layer residual, steering applied before the tap — which implies a
  device-to-host copy per tapped layer; taps stay optional so the untapped
  path pays nothing.
- **Chat template and tool-call format.** The exact Gemma 4 function-calling
  convention must be taken from the model card and tokenizer config, not assumed
  from Gemma 3.
- **E-series architecture.** Possible MatFormer or per-layer-embedding inheritance
  from Gemma 3n changes the loader; unresolved until checked against the reference
  implementation.
- **VRAM fit.** Estimated fit on the 32GB RTX 5090: E4B Q8 with full KV budget and
  several concurrent subagent sessions is comfortable. Estimate only; no
  measurements exist yet.
- **Model quality ceiling.** A 4.5B-effective local model is not a frontier coding
  agent. The project's value is the harness-runtime seam — cache reuse, constrained
  decoding, prefill behavior — not parity with hosted agents.
