# Rust harness and runtime design

Design for a single-binary local coding harness written in Rust, with the LLM
inference runtime linked in as a library crate rather than accessed over HTTP. The
harness adapts the agent architecture of the
[claude-code-from-scratch](https://github.com/FareedKhan-dev/claude-code-from-scratch)
tutorial; the runtime targets the Gemma 4 family exclusively, matching the
repository model policy recorded in
[unified_cli_investigation.md](../training/investigations/unified_cli_investigation.md). The
implementation lives in `rust-harness/`; where this document and the code
disagree, the code is correct and the divergence is a documentation bug.
Findings and external facts were last reviewed 2026-08-06.

## Table of contents

- [Purpose and non-goals](#purpose-and-non-goals)
- [Background](#background)
- [Architecture](#architecture)
- [Runtime design](#runtime-design)
- [Harness design](#harness-design)
- [Model targeting](#model-targeting)
- [Architecture reconciliation](#architecture-reconciliation)
- [Implementation roadmap](#implementation-roadmap)
- [Risks and open questions](#risks-and-open-questions)

## Purpose and non-goals

The project builds a local coding agent — an agent loop with file, shell, and
planning tools — whose model inference runs in the same process on the RTX 5090,
with no network dependency. The primary purpose is pedagogical: understanding the
harness-runtime seam by owning both sides of it.

Positioning, decided after the interpretability seam landed: the harness is an
instrumented inference substrate for the repository's training and
interpretability pipelines, with the coding-agent loop as its native workload.
Owning both sides of the seam yields properties no off-the-shelf stack
provides — in-band activation capture from the serving numerics, steering and
probe monitoring inside the forward pass, a flight recorder and deterministic
KV restore that make runs replayable, and observability records in the shared
obs contract. The agent loop is the workload generator for those experiments,
not a competitor to the llama.cpp and vLLM services on model coverage or
serving speed. The design serves one generation in flight at any instant, which is a property
of the runtime rather than a policy: `Model` is `Send` but not `Sync` and every
generation path takes `&mut self`. Multi-tenant serving remains the job of the
existing services. Parallel subagents
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
  services on ports 8001-8003 for the general workload, on model coverage and
  on speed. This non-goal was narrowed rather than dropped during
  implementation: `--serve` exposes a loopback endpoint on port 8005 carrying
  the two paths the interp pipeline calls, `/v1/chat/completions` and
  `/v1/health`, because the point of the harness is owning the seams, and an
  experiment that reaches the model over someone else's HTTP server observes
  nothing the tap seam could have shown it. What stays true is the shape: one
  generation at a time, one local user, loopback only, no batching and no
  tenancy. The contract is `schemas/harness_chat_endpoint.json`.
- **No training.** The training tier stays on the Python/PyTorch stack. The
  original design also excluded an interpretability seam; that non-goal was
  reversed during implementation — the CPU backend carries a residual-stream
  tap, control-vector steering, and an in-band probe monitor, designed in
  [rust-harness-interp.md](../interp/rust-harness-interp.md). The
  analysis tier (probe training, direction extraction, evaluation) remains in
  Python.

The justification behind this positioning — what the agent loop and chat
mode each contribute to the training and interpretability programs, and the
feature bar for future harness work — is recorded in
[observation-frames.md](observation-frames.md).
- **No multimodality.** The vision components of Gemma 4 checkpoints are skipped at
  load time.
- **No custom CUDA kernels in version 1.** The planned GPU tensor backend is
  [Candle](https://github.com/huggingface/candle); kernel authorship is deferred to
  the parked survey in
  [future-optimizations.md](../optimizations/ideation/future-inference-optimizations.md).
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

A Cargo workspace with three crates and a model module:

```
rust-harness/
  Cargo.toml            # workspace
  harness/              # bin crate: agent loop, tools, permissions, transcript,
                        # chat mode, subagents
  obs/                  # lib crate: run-level observability records, shared by
                        # harness and the runtime binaries
  runtime/              # lib crate: inference engine
    src/
      session.rs        # Session handle owning the KV cache
      sampler.rs        # temperature, top-k, top-p, grammar constraints
      grammar.rs        # tool-call JSON grammar compilation
      kv.rs             # per-layer cache allocation and delta prefill
      checkpoint.rs     # config.json and model.safetensors: weights and the interp artifacts
      tokenizer.rs      # wrapper over the `tokenizers` crate, reading tokenizer.json
      model/cpu.rs      # hand-rolled CPU backend, tap and steering seams
      model/mock.rs     # scripted backend the sandbox tests run against
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

**Tensor backend.** The only backend that exists is hand-rolled: a CPU
reference (`runtime/src/model/cpu.rs`) computing the forward pass from a Hugging
Face checkpoint, chosen for rr-debuggable numerics and used by
every sandbox test and by the binary, which loads it unconditionally behind
`--model`. The planned GPU backend is
[Candle](https://github.com/huggingface/candle) with the CUDA feature, targeting
the RTX 5090; Candle provides safetensors loading and flash-attention bindings,
and Blackwell (sm_120) support under CUDA 13 must be verified before that work
starts (see [Risks](#risks-and-open-questions)). The runtime crate's `cuda` feature
is an empty placeholder until then. The two backends are separate implementations
of the same `Model` trait rather than a fast path and a fallback: neither one
degrades into the other at run time.

**Weights.** The published Hugging Face checkpoint, and only that: `config.json`
and `model.safetensors` (`runtime/src/checkpoint.rs`). For a model format the
reference implementation and the specification are the same artifact, and a
converter's metadata can only imply what a checkpoint states — that is the whole
argument for reading the published files rather than a converted container.

Three consequences the rest of this document depends on:

- **Nothing is quantized.** The accepted storage types are F32, F16 and BF16;
  any other ggml type id is a hard error naming the tensor
  (`runtime/src/tensor.rs`). Weights are held at their published width in a
  `Weight` enum and widened to f32 at use, which is lossless for BF16, so the
  checkpoint's own values reach the forward pass unrounded.
- **Nothing is mapped.** Only the framed safetensors JSON header comes into
  memory; each tensor is read at its offset when a caller asks for it. The crate
  is `#![forbid(unsafe_code)]`, which it could not be while a memory map existed.
- **GGUF is gone.** `gguf.rs` parsed the container for the interp
  artifacts after P8b took away the weight path, and R10 deleted it with the
  last of them. The control vector had stayed GGUF only because llama.cpp read
  it as `--control-vector-scaled`, which made the format the consumer's choice
  rather than this repository's; retiring that steering path returned the
  choice here, and the answer was the one the probe had already reached at R7.
  Both artifacts are now a safetensors image plus a JSON file beside it
  (`schemas/probe_artifact.json`, `schemas/control_vector_artifact.json`), read
  through the same `checkpoint.rs` envelope the weights use.

**Model graph.** No dynamic graph construction: the forward pass is written out
once as straight-line Rust rather than assembled from a graph description, which
is viable only because the runtime targets one family. The CPU backend reads its
dimensions from the checkpoint's nested `text_config` into a `ModelSpec` at load
time instead of holding them as compile-time constants, so one build serves every
Gemma 4 size variant and the loader can refuse a checkpoint whose declared shape
falls outside the implemented subset. Because the fields are named rather than
implied, that read is a mapping and not a parse: there is no stride to infer, no
paired `_swa` keys to resolve, and no scalar-or-array union to normalize.

**KV cache.** Gemma interleaves sliding-window and global attention layers, and
the cost is computable exactly at startup either way. Windowed layers cap their
cache regardless of context length, giving a fixed cost for those plus a
per-token cost for global layers only — `kv::budget_bytes`, which is what the
harness reports at startup and what a run allocates.

The cap is not the window itself but `kv::window_capacity`: a group of `n` query
rows starting at absolute `start` needs the row at `start + 1 - window` while its
own appends have already pushed the tail to `start + n - 1`, so residency is
`window - 1 + group`, plus slack. Retaining from the group's last position
instead of its first is the trap, and `runtime/tests/long_context.rs` exists to
catch it. `LayerKv` is a linear compacting buffer rather than a ring, so stored
rows stay contiguous and in natural token order — a ring would make attention's
read order depend on the write cursor, which a bit-exact substrate cannot have.

The cap has a consequence worth stating where a caller will meet it: **a windowed
layer is not rewindable past its window.** `Model::rewindable_from` reports the
floor, `truncate` enforces it, and `Session::sync` falls back to a full
re-prefill below it — correct but slower, and the routine path after compaction.
The run header prints the horizon at startup, so an experiment about to become
un-rewindable says so before it runs.

Layers that share a donor's K/V store nothing and are excluded from both figures.
The cache is allocated once per session and persists across turns. The expected
win is turn latency proportional to the appended delta rather than the full
transcript; the magnitude is unmeasured until phase B of the roadmap.

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
  terminal renderer as they are produced. The channel exists
  (`harness/src/stream.rs`) and is tested, but no entry point uses it yet, so the
  binary prints a reply when the turn finishes rather than token by token.
- **Chat template.** The harness renders Gemma 4's turn format, including the
  function-calling convention. Template bugs are the classic failure mode of local
  harnesses, so the rendered token IDs are to be golden-tested against the Hugging
  Face reference tokenizer for a fixed corpus of transcripts. That test does not
  exist: an earlier `#[ignore]`d stub was removed along with the Gemma 3 markers it
  was written against, and no replacement was added. Writing it needs a
  `fixtures/template_goldens.json` generated on the workstation from the reference
  tokenizer, which the sandbox cannot reach. The marker constants themselves are
  pinned from both language tiers by `/schemas/chat_template.json`, so what is
  missing is the token-level check and not the marker set. End-to-end runs against the
  synthetic fixture therefore already happen without it; the golden test gates
  trusting a rendered prompt on real weights, not running the harness at all.

### Context layer

Three mechanisms manage conversations that outgrow one context window, all
built over the append-only prompt discipline the KV cache depends on:

- **Memory file.** `HARNESS.md` in the working root is read once at startup
  and rendered as a synthetic leading turn — never persisted to the
  transcript, stable within a process (so the KV prefix stays maximal),
  picked up again on the next start.
- **Compaction.** When the rendered prompt crosses `--compact-at` (default
  0.8) of the context budget, or on `/compact` in chat mode, the middle of
  the effective transcript is summarized by the model itself and replaced
  by the summary. Persistence is append-only: a `Compaction` record naming
  the replaced range joins `session.jsonl`, and `Transcript::effective`
  folds records in on read, so raw history survives and the transcript's
  torn-line crash recovery covers the record unchanged. The splice shortens
  the prompt's common prefix; `Session::sync` truncates the cache to the
  kept head and prefills only the summary and tail.
- **KV snapshot.** `session.kv` is written on clean exit and restored at
  startup: a versioned header (magic, version, dtype tag with f16 reserved,
  FNV-1a weight fingerprint) followed by the token history and the
  backend-defined cache payload. Header or fingerprint mismatch refuses
  loudly; a snapshot whose tokens no longer match the transcript degrades
  through prefix matching to re-prefill from the divergence point.
  Restored continuation logits are bit-identical to an unbroken run on the
  CPU backend. Real-model snapshots are gigabyte-scale at f32; the dtype
  tag reserves an f16 payload without a format break.

Retrieval over an external index stays out of scope: the agent's live
grep/glob/read tools search the working tree directly, matching the
tutorial's managed-resource position.

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
  with the repository's Gemma 4 E-model policy. It is the model the harness is
  tuned and benchmarked against.
- **Also supported: E2B**, verified against `transformers` on 2026-08-12 — all
  five ladder rungs pass on the real checkpoint, worst layer `rel_l2` 4e-6,
  perplexity within 0.000015, peak RSS 9.51 GiB.
- **Skipped: the larger server-tier variants** (large models are already served
  well by the existing services; their architectures are out of scope for this
  runtime).

E2B was skipped for a long time on the grounds that it was "the same code path
as E4B, weaker coder". The second half stands; the first was false, and
believing it hid a bug. E2B is the only E-series model that sets
`use_double_wide_mlp`, has a five-layer attention period rather than six, and
runs true multi-query attention at one key-value head. The loader never read
the double-wide flag, so the real checkpoint refused to load at layer 15 —
found the first time anyone pointed `checkpoint-info` at it, months after the
paths it exercises were declared covered. The synthetic fixture had modelled
the feature from the same documents the reader was written from, so the two
agreed with each other and both diverged from the artifact.

The general lesson is about what a second model is worth. E2B is not a
capability the harness needed; it is a second sample of the architecture, and
it caught a defect that no amount of E4B testing could, because E4B does not
have the feature. Treat "same code path" as a claim requiring evidence.

Version coupling is accepted: the model module is the one component rewritten for a
future Gemma 5. Whether the E-series inherits MatFormer or per-layer-embedding
techniques from Gemma 3n was an open question against the model card. Published
sources answer it: it does. See the next section.

## Architecture reconciliation

The CPU reference backend (`runtime/src/model/cpu.rs`) was written against a
Gemma-3-shaped specification: RMSNorm scaling by the stored weight, embedding
scaled by the square root of the embedding dimension, grouped-query attention with
per-head QK-norm and NEOX RoPE, interleaved sliding-window and global layers, a
GeGLU feed-forward block with tanh GELU, four norms per layer, and an LM head tied
to the token embedding. Published descriptions of Gemma 4 E4B describe a superset
of that.

The norm scaling is the one place where the checkpoint, not the reference
implementation, settles the convention, and it was settled the wrong way for
nine work packages. Gemma 1 through Gemma 3 store a norm parameter initialized
to zero and add 1.0 at run time. Gemma 4 does not: `Gemma4RMSNorm` initializes
`weight` to ones and multiplies by plain `w`. The kernel here has always been a
plain multiply, but the checkpoint loader folded a `+1` in as it read, which
doubled every normalized activation on any checkpoint we did not write
ourselves.

Nothing in this repository could detect it, because the fixture writer
subtracted 1.0 and the numpy oracle added it back — one belief spelled at three
ends, agreeing with itself. The transformers-authored Tier T fixture caught it
on its first run; see `rust-harness/runtime/tests/ladder_tiny.rs`. llama.cpp's
converter agreed all along, writing `norm_shift` 0.0 for gemma3n and gemma4.

| Feature | Status in Gemma 4 E4B | Implemented | Confidence |
|---------|-----------------------|-------------|------------|
| AltUp (parallel hidden-state streams) | Dropped | Not applicable | High |
| Per-layer embeddings (PLE) | Retained, approximately 2.82B of 8B total parameters | No | High |
| MatFormer nested feed-forward | Retained; slicing can vary width per layer or per block | No | High |
| Shared KV cache | Last 18 of 42 layers reuse K/V from the last non-shared layer | No | Medium |
| Dual RoPE | p-RoPE (p = 0.25, theta 1M) on global layers, plain RoPE (theta 10k) on local | No | Medium |
| Value normalization | Magnitude-only, no learned scale | No | Medium |
| LAuReL | Retained | Not applicable | Medium |
| Sliding window | 512 tokens, 5:1 local-to-global over 42 layers | Yes, from `layer_types` | Medium |

AltUp being dropped is the load-bearing finding. It runs several parallel hidden-state
streams per layer, which would leave `Tap::on_residual(layer, site, position, &[f32])`
without a single residual vector to hand out, renegotiating the interpretability
contract across the language boundary. With one residual stream retained, the tap
seam, the capture shard format, and the Python-side correspondence all hold. See
[the interpretability contract](../interp/rust-harness-interp.md) for the
one consequence PLE does carry.

One caveat bounds these findings: they are secondhand. The primary sources refuse
automated retrieval, so the details come from search summaries of varying
reliability, and the confidence column reflects how many independent sources
agreed. Runbook entry RH.1 remains the authority, and it now decides the question
against the same artifact the loader reads.

The second caveat is retired rather than answered. It used to say that what
governs the code is not the published architecture but what the converted file
stores after conversion and quantization — two derivations that could disagree,
with the converted one winning. Reading the checkpoint directly collapses them:
the file the backend loads is the file the architecture is published as.

Where the backend does not implement something, it refuses rather than guesses. Per-layer
embeddings, shared KV, per-layer feed-forward widths, per-layer head dimension,
dual RoPE, logit softcapping, and `layer_output_scale` were all refused by name and
are now implemented; `cpu::check_supported_checkpoint` retains each as a coherence check, so a
file whose metadata and tensors describe different models still fails the load. What
it still refuses is partial rotary applied by slicing, attention softcapping, logit
rescaling, unrecognized RoPE keys, and any unconsumed tensor family. The alternative
was worse than a failure: loading is name-directed, so an unimplemented feature
presents as ignored weight tensors and fluent, wrong output that no machinery test
detects. The
`checkpoint-info` binary prints the same verdict from the same function, so the
reconciliation run yields a decision rather than a judgment call.

Sources: the [Gemma 4 technical report](https://arxiv.org/abs/2607.02770)
(arXiv 2607.02770, 2026-06-19), a
[visual guide to the architecture](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4),
a [breakdown of per-layer embeddings, shared KV, and dual RoPE](https://botmonster.com/posts/gemma-4-architecture-per-layer-embeddings-shared-kv-cache-dual-rope/),
and [llama.cpp issue 21468](https://github.com/ggml-org/llama.cpp/issues/21468),
which reports that the shared KV cache disables prefix cache reuse in llama.cpp and
forces full prompt re-evaluation on every request. That last point is the phase B
premise stated from the outside: delta prefill on this architecture requires owning
the cache.

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
[observability design](observability.md); the CUDA backend is
validated against the CPU reference in addition to the Tier 3 fixtures.

| Phase | Deliverable | Acceptance check |
|-------|-------------|------------------|
| A | Load E4B from its checkpoint, single forward pass | Logits for a fixed prompt match the Hugging Face reference; the tolerance is f32 accumulation order, not quantization, because nothing on the load path rounds a weight |
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
- **Candle Gemma 4 coverage.** Four months after the 2026-04-02 release, Candle support for
  the new variants may be incomplete; the E4B layers may need direct implementation
  rather than reuse of Candle model code. The CPU reference backend
  (phase A-prime) is hand-rolled and does not depend on Candle; see the
  [observability design](observability.md) for the recorded
  decision.
- **Tap seam parity.** The CPU backend exposes a residual-stream `Tap` seam
  (interpretability capture and steering; see
  [rust-harness-interp.md](../interp/rust-harness-interp.md)).
  The CUDA backend must implement the same seam with identical semantics —
  the layer output after the `layer_output_scale` multiply, delivered as
  `Site::LayerOutput`, steering applied before the tap — which implies a
  device-to-host copy per tapped layer; taps stay optional so the untapped
  path pays nothing.

  The seam now has a second fire site, post-softmax attention weights through
  `Tap::on_attention`, and that one is materially harder on a GPU. The CPU
  backend hands out a borrowed slice for free because it materializes a score
  vector per query head; a tiled online-softmax kernel — which is what a
  competitive CUDA attention is — never forms the distribution at all, so
  there is nothing to borrow. Serving this site on CUDA therefore costs a
  second, non-tiled attention path held bit-identical against the fast one,
  in the sense `.styles/RUST_STYLE.md` means it: identical bits or a
  different function with a different name. That is a real cost and it is
  named here so it is priced when the backend is scheduled rather than
  discovered during it. Refusing the site on CUDA is an acceptable outcome;
  serving it from a path whose numerics differ from the serving path is not,
  because the whole value of an in-band tap is that it observes the
  arithmetic that actually ran.
- **Chat template and tool-call format.** The exact Gemma 4 function-calling
  convention must be taken from the model card and tokenizer config, not assumed
  from Gemma 3.
- **E-series architecture.** Resolved for the CPU backend: per-layer embeddings,
  shared KV, dual RoPE and per-layer geometry are implemented and refused when
  absent. See [docs/models/gemma-4.md](../models/gemma-4.md).
- **VRAM fit.** The estimate must be rebuilt, because it was made against a Q8
  model that is no longer loadable. E4B is served at its published BF16 width;
  peak RSS after load on the CPU backend is about 14 GiB rather than the 28 a
  map-and-copy cost (`checkpoint.rs`, measured). The 32GB RTX 5090 figure for
  the CUDA backend is unmeasured and must be budgeted against that width plus
  `kv::budget_bytes`, not against a quantized one.
- **Model quality ceiling.** A 4.5B-effective local model is not a frontier coding
  agent. The project's value is the harness-runtime seam — cache reuse, constrained
  decoding, prefill behavior — not parity with hosted agents.
