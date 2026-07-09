# Phase 3 product design: pretraining from scratch

This document is the implementation-ready product design for Phase 3 of the
training roadmap (docs/training/scope.md): a dense transformer trained from
random initialization on this workstation, executed at 300-500M parameters on
10-20B FineWeb-Edu tokens, with infrastructure validated for full training at
1B parameters and corpora of 100B+ tokens. Each section is written to be
picked up in a fresh session and implemented without re-deriving context: it
records the decisions already made, the interfaces to build, the facts the
design depends on, and the tests that define done.

Scope.md defines what Phase 3 delivers; this document defines how. Where the
two disagree, this document wins and scope.md should be corrected.

The learning track (autograd engine, derivations, annotated models,
training-dynamics notebooks) is deliberately excluded from this document and
will receive its own design. Phase 3's production deliverables do not depend
on it. One consequence is recorded here so the two designs stay compatible:
the annotated single-file transformer must share the architecture of P3-D1,
so P3-D1's configuration dataclass is the contract the learning track will
later import.

## Table of Contents

- [Prerequisites and preflight](#prerequisites-and-preflight)
- [Cross-cutting decisions](#cross-cutting-decisions)
- [Build order](#build-order)
- [P3-D1: From-scratch transformer](#p3-d1-from-scratch-transformer)
- [P3-D2: Architecture registry](#p3-d2-architecture-registry)
- [P3-D3: Pretraining loop](#p3-d3-pretraining-loop)
- [P3-D4: Scaling-law and token-budget calculator](#p3-d4-scaling-law-and-token-budget-calculator)
- [P3-D5: Corpus and tokenizer preparation](#p3-d5-corpus-and-tokenizer-preparation)
- [P3-D6: The 300-500M pretraining run](#p3-d6-the-300-500m-pretraining-run)
- [Phase 3 exit checklist](#phase-3-exit-checklist)

## Prerequisites and preflight

This design assumes the Phase 0 deliverables (D1-D10, docs/training/
phase0-design.md) and the Phase 2 full-scale data pipeline exist and pass
their acceptance criteria. The specific contracts consumed:

- **D1 container**: docker/training image with torch cu130 on sm_120,
  flash-attn, liger-kernel, torchao; `make train-shell`, `make train-smoke`.
- **D2 shards**: ShardedTokenDataset over memory-mapped uint16/uint32 token
  shards with manifest hashes and a serializable cursor.
- **D3 tokenizer**: trained byte-level BPE artifacts under
  /mnt/ai-data/training/tokenizers/ with manifest and xxhash.
- **D4 memory-fit calculator**: `estimate(recipe)` and `solve(recipe, axis)`
  accurate within 10 percent on the reference configs.
- **D5 recipes**: Hydra structured configs; the `model.scratch` block
  (layers, hidden, heads, kv_heads, ffn_mult, vocab, rope theta,
  tie_embeddings) is the input to P3-D1.
- **D6 checkpointing**: atomic sharded-safetensors checkpoints restoring
  model, optimizer, scheduler, dataloader cursor, and RNG state; the
  deterministic resume test.
- **D7 round trip**: export to GGUF against the pinned llama.cpp revision
  and serve_eval through llama-gpu on port 8002.
- **D8 FP8**: `precision: fp8` with torchao tensorwise, bf16 fallback with
  `precision_effective` stamping, and train_bench.py JSON.
- **D9 heterogeneous compute**: prefetch dataloader, staging, CCD pinning,
  run lock, data_wait_ms metric.
- **D10 watchdogs**: in-process callback with abort-on-NaN, emergency
  checkpointing, thermal/RAM/disk floors.
- **Phase 2 pipeline**: training/data/pipeline.py with MinHash-LSH
  deduplication, quality filtering, and benchmark decontamination in front
  of the D2 shard writer.

**Preflight task (first implementation step).** Before building anything,
run a checklist that verifies each contract above against the actual
repository: the interfaces exist under the documented paths, `make
train-smoke` passes, and the Phase 0 exit checklist items relevant here
(deterministic resume, memfit accuracy, round trip) are demonstrable. Any
gap is reported and fixed or explicitly waived before Phase 3 code lands.
If the Phase 2 pipeline is absent, P3-D5 records the documented fallback
(FineWeb-Edu ships pre-filtered; only decontamination is non-negotiable).

## Cross-cutting decisions

Settled questions every deliverable inherits. Do not re-litigate.

1. **Custom loop, not a framework wrapper.** The trainer is our own code:
   P3-D1's model plus P3-D3's loop, hardened from the D6 reference loop.
   nanotron, litGPT, torchtitan, and OLMo-core are consulted as references
   for hyperparameters and numerical details but are not dependencies. This
   maximizes the learning pillar and stack ownership, and it reuses the
   D5/D6/D8/D9/D10 contracts directly instead of adapting a foreign trainer
   to them.
2. **GGUF-servable by construction.** D7 constrains scratch models to
   architectures the pinned llama.cpp converter supports. The P3-D1
   transformer is therefore weight-compatible with `LlamaForCausalLM`:
   RMSNorm, SwiGLU, RoPE, GQA, no biases, and an export path that emits an
   HF directory with `"architectures": ["LlamaForCausalLM"]` plus the
   tokenizer. Architectural experiments that break Llama compatibility go
   to the registry (P3-D2) as non-servable entries, clearly flagged.
3. **Execution scale versus design scale.** Every component is designed and
   memory-validated for 1B parameters full-AdamW in 32GB without offload
   and for 100B+ token corpora (both already Phase 0 requirements). The
   run that is actually executed is 300-500M on 10-20B tokens, wall-clock
   two to four days. The 1B envelope is validated by memfit plus a short
   calibration run, not a full training run.
4. **Concurrency and storage policies are unchanged.** Inference down
   during training; artifacts under /mnt/ai-data/training/; checkpoint
   retention keep-last-N plus every-Mth per recipe; the D9 run lock guards
   all of it.
5. **Precision policy.** bf16 is the baseline and the fallback. The
   production run uses `precision: fp8` (torchao tensorwise) only if the
   D8 convergence gate passed on the 150M reference and a fresh 30-minute
   gate at the P3-D6 model size also passes; otherwise the run is bf16 and
   the decision is recorded in the run notes. A multi-day run is not the
   place to discover FP8 divergence.
6. **Determinism and seeds.** All runs are seeded through the recipe.
   Production runs keep nondeterministic kernels for speed; the
   correctness suite (overfit test, resume test, gradient checks) runs
   under `torch.use_deterministic_algorithms(True)`.
7. **Style.** Python 3.12, black + ruff per pyproject, comment and
   documentation conventions from .claude/*.md, no emojis, tests under
   tests/training/, Makefile targets in the `##@ Training` section.

## Build order

P3-D1 model ──┬── P3-D2 registry
              └── P3-D3 loop ──── P3-D6 run
P3-D4 scaling calculator ─────────┘
P3-D5 corpus + tokenizer ─────────┘

P3-D1 and P3-D4 are independent and can start in parallel. P3-D5 is
CPU-days of pipeline work and should start early; it needs no new code
beyond invocations of Phase 0/2 tooling. P3-D6 consumes everything.

## P3-D1: From-scratch transformer

**Tag:** build. **Depends on:** Phase 0 D1, D5.

### Purpose

A configurable dense decoder-only transformer implemented in this
repository, readable enough to serve the learning pillar later and correct
and fast enough to train for real. It is the reference architecture of the
registry and the model of the production run.

### Decisions

1. **Architecture:** pre-norm decoder-only transformer, Llama-3-family
   shape: RMSNorm (eps 1e-5), SwiGLU MLP (ffn_mult applied with the
   standard 2/3 correction and rounding to a multiple of 256), rotary
   position embeddings (theta from recipe, default 10000 at 2048 context;
   500000 if context is raised to 4096+), grouped-query attention
   (kv_heads from recipe), no attention or MLP biases, optional tied
   input/output embeddings (default tied at sub-1B scale). Vocabulary
   size comes from the tokenizer manifest and must be a multiple of 128
   (pad the embedding if the tokenizer vocab is not; keeps matmul shapes
   friendly for FP8's multiple-of-16 constraint and tensor cores).
2. **File:** training/models/transformer.py, a single readable module:
   `TransformerConfig` dataclass (mirrors the D5 `model.scratch` block
   exactly; one conversion function, no duplicated field lists) and
   `Transformer(nn.Module)`. Target on the order of 500 lines; clarity
   over micro-optimizations, because the kernels below carry the speed.
3. **Kernels:** attention through flash-attn's `flash_attn_func`
   (varlen path not needed; training uses fixed-length packed windows),
   with `torch.nn.functional.scaled_dot_product_attention` as the fallback
   selected by config. Liger-Kernel fused ops (RMSNorm, SwiGLU, fused
   cross-entropy with optional z-loss) applied by a `apply_liger(model)`
   patch step controlled by the recipe (`model.fused_kernels: true`
   default); the pure-PyTorch path remains runnable for gradient checks
   and the learning track.
4. **Initialization:** truncated normal, std 0.02 for embeddings and
   general weights, residual-out projections (attention out, MLP down)
   scaled by 1/sqrt(2*layers) (GPT-2/Llama practice). Init lives in one
   `init_weights` method with the scheme named in a comment; the ablation
   surface later toggles it.
5. **Numerics:** forward in bf16 autocast; RMSNorm and softmax statistics
   in fp32; loss in fp32. Optional z-loss (default 1e-4, recipe-gated) on
   the logits for long-run stability. Logit soft-capping is not used
   (Llama-family converters do not expect it).
6. **HF export adapter:** `to_hf_llama(model, config, out_dir)` in
   training/models/export_hf.py writes safetensors with Llama parameter
   names plus a config.json declaring LlamaForCausalLM, and copies the
   tokenizer. Correctness is defined by logit parity: loading the export
   with `transformers.AutoModelForCausalLM` reproduces our forward within
   tolerance. This is the bridge to D7 and it is tested, not assumed.
7. **Reference configs** (named presets in training/recipes/):
   pretrain-150m (existing Phase 0 reference, restated with this
   architecture), pretrain-500m (the production run: suggested 24 layers,
   hidden 1280, 20 heads, 5 kv-heads, ~500M with a 49k vocab — final
   numbers set by memfit and the P3-D4 budget), pretrain-1b (envelope
   validation only). Exact dims chosen at implementation under the
   multiple-of-128/256 rules above.

### Interfaces

- `Transformer(config)` — forward(tokens) -> logits; forward with
  `labels=` returns loss (fused path when Liger is on).
- `TransformerConfig.from_recipe(recipe)` and `to_hf_llama(...)`.
- No new Make targets; the model is only reachable through P3-D3 and tests.

### Risks

- Silent architecture drift from Llama conventions breaks GGUF conversion
  weeks later. Mitigation: the export-and-parity test plus a conversion
  smoke test against the pinned llama.cpp run in CI-style tests from day
  one on a tiny config.
- flash-attn kernel or Liger patch changing numerics. Mitigation: gradient
  checks compare fused against pure-PyTorch paths on a tiny config.

### Acceptance

- A tiny config (4 layers) overfits a single batch to near-zero loss in
  under a minute on the workstation.
- Fused and unfused paths agree: max relative logit difference within a
  documented tolerance, gradients within tolerance on a fixed seed.
- to_hf_llama logit-parity test passes, and the exported tiny model
  converts with the pinned llama.cpp converter and generates through
  llama-gpu (reuses D7 machinery end to end).
- memfit on pretrain-1b with this model reports fitting in 32GB without
  offload, and a calibration run (D4 harness) confirms within 10 percent.

## P3-D2: Architecture registry

**Tag:** build. **Depends on:** P3-D1.

### Purpose

A common interface over multiple architectures so recipes select a model by
name and later phases (Mamba contrast in the learning track, Titans/HOPE in
Phase 4) add entries without touching the loop.

### Decisions

1. **Interface, minimal:** training/models/registry.py with
   `register(name)` decorator and `build(model_cfg) -> nn.Module`. The
   contract every entry satisfies: constructed from the recipe model
   block; `forward(tokens, labels=None)` returning `(logits, loss)`;
   `param_count()`; a `servable` class flag (GGUF-convertible via
   export_hf) that D7 preflight reads. Nothing else — no abstract base
   class ceremony.
2. **Entries in Phase 3:** `dense` (P3-D1) and `mamba`. The Mamba entry
   wraps the official mamba-ssm blocks (adopt, not reimplement) behind the
   interface, `servable: false` (llama.cpp Mamba support is not assumed;
   revisit against the pinned converter), and is validated only at small
   scale (a 50-150M config that overfits a batch and completes a short
   shard-fed run). It exists to prove the interface is real, not to be
   trained seriously in this phase.
3. **Loop neutrality:** P3-D3 depends only on the registry contract.
   Anything attention-specific (flash-attn selection, Liger patching)
   lives inside the dense entry.

### Acceptance

- `build()` constructs both entries from recipes; unknown names fail with
  the available list.
- The short Mamba run completes on shards with checkpoints saving and
  resuming through D6 unchanged.
- A unit test asserts the loop code imports the registry and not
  transformer.py directly.

## P3-D3: Pretraining loop

**Tag:** build. **Depends on:** P3-D1/D2, Phase 0 D5, D6, D8, D9, D10.

### Purpose

The production training entry point: scripts/train/pretrain.py (thin CLI)
over training/pretrain.py (logic), hardening the D6 reference loop into
something trusted to run unattended for days and, by design, weeks.

### Decisions

1. **Composition, not invention.** The loop is the D6 skeleton plus: the
   registry model, gradient accumulation to a recipe-set global batch (in
   tokens), gradient clipping (default 1.0), AdamW (fp32 states; 8-bit
   Adam available via recipe for the 1B-without-offload envelope if memfit
   requires it), the WSD or cosine schedule from D5, torch.compile on the
   model (recipe-gated, default on; falls back cleanly when compilation
   fails), the D8 FP8 hook, the D9 prefetcher/staging/affinity/lock, and
   the D10 watchdog callback. No feature lands outside the recipe schema.
2. **Data order is part of the checkpoint.** Deterministic shard shuffling
   from the run seed (shuffle shard order per epoch, sequential within a
   shard window; document the tradeoff versus full-token shuffle — D2
   windows are already document-packed so within-shard locality is
   acceptable at this scale). The cursor plus seed fully determine the
   remaining data stream, so resume never repeats or skips tokens; token
   accounting (tokens_seen in meta.json) is exact.
3. **Validation in-loop:** a held-out shard set (P3-D5 reserves one) is
   evaluated every eval_interval steps on the GPU between training steps
   (cheap at this scale), logging val loss and perplexity to tracking.
   The D9 concurrent-CPU-eval machinery may additionally run lm-eval
   smoke tasks on earlier checkpoints, but that is an optimization, not a
   gate for this deliverable.
4. **Stability playbook encoded, not documented:** loss-spike handling is
   the D10 explosion rule; additionally the loop supports
   `skip_spike_batches: N` (skip the update when per-batch loss exceeds
   the trailing-median factor, log it, abort only if N consecutive) —
   default off for the 500M run, available for the 1B track.
5. **Single-batch overfit mode:** `pretrain.py mode=overfit` trains on one
   cached batch and asserts the loss floor — the scope's mandated sanity
   check, runnable on any registry model, and the first gate of every
   production launch checklist.
6. **Throughput bar:** on the 500M config the loop must sustain, at
   minimum, the tokens/s that completes 10B tokens in four days
   (~29k tok/s sustained) with data_wait_ms under the D9 5 percent bound;
   the D8 bench harness runs against this loop to record MFU. If the bar
   is missed, profile before adding features — this is the go/no-go for
   launching P3-D6.
7. **Launch checklist automated:** `make pretrain CONFIG=...` runs, in
   order: recipe validate(), memfit fits-check, tokenizer/shard hash
   match, disk-floor and retention projection (checkpoints for the whole
   run fit under the floor), overfit smoke on the target config
   (few-minute budget), then launches under the D9 lock and pinning
   wrapper. Any failure stops the launch with the reason.

### Interfaces

- `make pretrain CONFIG=training/recipes/pretrain-500m.yaml [ARGS=...]`
- `python -m training.pretrain --config ... mode=train|overfit|resume`
  (resume is `checkpoint.resume=latest` sugar).
- Logged metrics: loss, val loss/ppl, lr, grad norm, tokens/s, MFU,
  data_wait_ms, step time p50/p95, plus D10 telemetry.

### Risks

- torch.compile recompilation stalls or graph breaks with flash-attn/
  Liger: keep the compile flag per-run and record compile time; fall back
  to eager if step-time variance is pathological.
- Multi-day clock drift in throughput estimates (thermals): the D10
  telemetry plus the bench JSON make it visible; the four-day budget
  includes 15 percent slack.

### Acceptance

- Deterministic resume test passes on this loop (not just the D6 reference
  loop) on a 150M config: N+M steps equals N, checkpoint, M.
- Kill-and-resume at an arbitrary wall-clock point loses at most
  checkpoint_interval steps of work, token accounting stays exact.
- Overfit mode reaches the loss floor on dense and mamba entries.
- The throughput bar is met and recorded via train_bench JSON (bf16 and,
  if enabled, FP8).
- A 12-hour unattended soak on the 150M config completes with zero
  watchdog aborts and clean telemetry.

## P3-D4: Scaling-law and token-budget calculator

**Tag:** build. **Depends on:** Phase 0 D5; P3-D3 for the calibration runs.

### Purpose

Answer, before committing days of wall-clock: for a target parameter count
and a wall-clock budget on this machine, how many tokens should the run
see and what final loss is predictable. Sets the P3-D6 token budget and
provides the acceptance tolerance the run is judged against.

### Decisions

1. **Model:** Chinchilla-form parametric loss
   L(N, D) = E + A/N^alpha + B/D^beta. Ship literature coefficients as the
   prior (Hoffmann et al. values, constants pinned in code with source
   comments, like D8's MFU constants), then refit A, B, E on our own
   measurements. Document plainly that sub-1B, 20-40 tokens/param regimes
   extrapolate imperfectly; the tool reports a confidence band, not a
   point promise.
2. **Calibration from cheap runs:** `scaling.py fit` consumes the tracking
   histories of short runs at three or more sizes (suggested 50M, 100M,
   200M for a few hours total on this card, exact budget set by the
   throughput measured in P3-D3) and fits the curve; the fit and its
   residuals are stored as JSON under docs/training/optimizations/ with
   the run IDs it came from.
3. **Wall-clock solver:** combines the loss model with measured tokens/s
   (from train_bench JSON, per precision) to answer the real question:
   `solve --params 500e6 --days 4` returns token budget, predicted final
   validation loss with band, and checkpoint-storage projection. Also
   inverts: given a loss target, the (params, tokens) frontier reachable
   in a given wall-clock.
4. **Location and shape:** training/calculator/scaling.py alongside the D4
   memory-fit code, same output conventions (human table plus --json),
   `make scaling ARGS=...` target. Validation loss is measured on the
   P3-D5 held-out shard set with the same tokenizer, and the JSON records
   the eval-set identity — a loss number without its eval set is not
   comparable.

### Acceptance

- Unit tests pin the closed-form math (loss prediction for hand-set
  coefficients; solver inversions round-trip).
- The fit reproduces the calibration runs' final losses within the stored
  residuals.
- Before P3-D6 launches, the calculator's prediction for the 500M config
  is recorded in the run notes; the scope-level criterion (measured final
  loss within a set tolerance of predicted — set the tolerance from the
  calibration residuals, suggested +/-0.1 nats or the fitted band,
  whichever is wider) is evaluated at run end.

## P3-D5: Corpus and tokenizer preparation

**Tag:** orchestrate. **Depends on:** Phase 0 D2, D3; Phase 2 pipeline.

### Purpose

Produce the tokenizer and the tokenized, decontaminated 10-20B-token
FineWeb-Edu shard set the run trains on, plus held-out shards for
validation. No new code paths: this is disciplined invocation of existing
tooling, and it is CPU-days, so it starts early and runs while P3-D1/D3
are built.

### Decisions

1. **Corpus:** HuggingFaceFW/fineweb-edu (pin the revision in the shard
   manifest), streamed through the Phase 2 pipeline. FineWeb-Edu is
   already quality-filtered and deduplicated upstream, so the heavy
   pipeline stages run in verification mode where cheap; the
   non-negotiable stage is benchmark decontamination against the P3-D6
   evaluation suite (the lm-eval tasks used at run end plus the smoke
   prompts). Document what was and was not re-run and why.
2. **Budget:** target 20B tokens sharded (the upper end), so the run can
   train on 10-20B without re-sharding; final trained-token count comes
   from P3-D4. Held-out set: ~100M tokens in separate shards, reserved by
   directory convention (shards/fineweb-edu-20B-val/), never listed in the
   training recipe's shard set; the loop refuses a val directory that
   appears in train.
3. **Tokenizer:** train via D3 at vocab 49152 on a mixed sample dominated
   by FineWeb-Edu with a code slice (per D3's two-domain evaluation;
   suggested 90/10 web/code by bytes) — the model is web-text-focused but
   a code slice hedges tokenization of code-bearing web pages. Accept the
   D3 metrics only if fertility and compression are within a few percent
   of the SmolLM2 reference on the web domain; otherwise investigate
   before sharding 20B tokens with it. The tokenizer hash then flows
   through shard manifests and recipe validation end to end.
4. **Reproducibility:** the exact prepare_data/pipeline invocations, pins,
   and manifests are committed under docs/training/runs/pretrain-500m/
   (commands and manifests are small; shards themselves stay on
   /mnt/ai-data). Anyone with the pins can rebuild the corpus.

### Acceptance

- Shard manifests validate (hashes, token totals, tokenizer hash) and D2
  round-trip decode spot-checks pass.
- Decontamination report exists listing the evaluation sets screened
  against and hit counts.
- Tokenizer metrics table committed; reference-comparison gate met.
- Total tokens >= 20B in train shards plus the held-out set, disk usage
  within the storage projection.

## P3-D6: The 300-500M pretraining run

**Tag:** orchestrate. **Depends on:** everything above.

### Purpose

The phase's product: a coherent, servable model trained from scratch on
this machine, with the full paper trail — recipe, scaling prediction,
telemetry, checkpoints, benchmark JSON, and a served evaluation.

### Decisions

1. **Configuration:** recipes/pretrain-500m.yaml — the P3-D1 500M-class
   config, sequence length 2048, global batch on the order of 0.5-1M
   tokens (set with memfit; gradient accumulation as needed), AdamW, WSD
   schedule (stable phase then decay over the final ~15 percent; WSD
   chosen over cosine so the token budget can be extended mid-run if
   throughput allows — record this rationale in the recipe notes), peak
   lr ~3e-3-scale set from the calibration runs, weight decay 0.1,
   clipping 1.0. Precision per the cross-cutting FP8 policy.
2. **Cadence:** checkpoint_interval sized so an interruption costs under
   30 minutes; retention keep-last-3 plus every-10th (roughly 6GB per
   checkpoint at 500M with fp32 optimizer state — verify with memfit's
   host-side numbers and the disk projection in the launch checklist).
   Eval every ~500M trained tokens.
3. **Run procedure is the launch checklist** (P3-D3 decision 7), then
   unattended operation under watchdogs, with the run journal (notes
   field plus alerts.jsonl plus tracking) as the record. Mid-run human
   checks are read-only: loss curve, throughput, thermals.
4. **End-of-run evaluation:** validation perplexity on the held-out set;
   lm-eval smoke suite (small task set chosen at implementation, e.g.
   hellaswag/arc-easy/lambada subsets, the same set the corpus was
   decontaminated against); qualitative coherence: greedy and sampled
   completions on the D7 smoke prompt set, committed to the run
   directory. "Coherent text at the target validation perplexity" is
   made concrete before launch: the target ppl comes from the P3-D4
   prediction band, and coherence means grammatical English continuations
   on the smoke prompts as judged in the run report.
5. **Serve it:** to_hf_llama export, D7 round trip to Q8_0 GGUF, served
   through llama-gpu, smoke prompts answered through the live service.
   The GGUF and export manifest land under /mnt/ai-data/models/trained/.
6. **Run report:** docs/training/runs/pretrain-500m/report.md — config,
   predicted versus measured loss, throughput/MFU versus the bench JSON,
   incidents, evaluation results, and lessons feeding the optional 1B
   track. This is also the blog-facing artifact.

### Acceptance (mirrors scope.md, made concrete)

- Overfit gate passed before launch; launch checklist output archived.
- The run completes 10-20B tokens; final validation perplexity within the
  P3-D4 prediction band recorded in advance.
- Sampled completions are coherent per the run report.
- The model answers prompts through llama-gpu; export manifest and GGUF
  hash recorded.
- memfit plus calibration confirm the 1B configuration fits in 32GB
  without offload, on this loop and model (envelope validated, run
  deferred to the optional scale-up track).

## Phase 3 exit checklist

1. Preflight passed: Phase 0 and Phase 2 contracts verified in-repo.
2. Dense transformer: fused/unfused parity, HF logit parity, tiny-model
   GGUF round trip (P3-D1).
3. Registry: dense and mamba build, train, checkpoint, resume through the
   common interface (P3-D2).
4. Loop: deterministic resume, overfit mode, throughput bar, 12-hour soak
   (P3-D3).
5. Scaling calculator fitted from calibration runs; 500M prediction
   recorded pre-launch (P3-D4).
6. 20B-token decontaminated shard set plus held-out set; tokenizer gated
   against references (P3-D5).
7. The 500M run completed, evaluated, served, and reported; 1B envelope
   validated without a full run (P3-D6).
8. Learning-track design (separate document) can import TransformerConfig
   and the run's tracking data without changes to Phase 3 code.
