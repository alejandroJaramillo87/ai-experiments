# Phase 2 product design: Continual pre-training

This document is the implementation-ready product design for every Phase 2
deliverable in the training scope (docs/training/scope.md, "Phase 2: Continual
pre-training"). It follows the same contract as the Phase 0 design document:
each section is written to be picked up in a fresh coding-agent session and
implemented without re-deriving context. It records the decisions already
made, the interfaces to build, the repository and ecosystem facts the design
depends on, and the tests that define done.

The scope document defines *what* Phase 2 delivers; this document defines
*how*. Where the two disagree, this document wins and the scope document
should be corrected.

## Table of Contents

- [Preconditions](#preconditions)
- [Cross-cutting conventions](#cross-cutting-conventions)
- [The base model: Gemma-4-E2B](#the-base-model-gemma-4-e2b)
- [Build order](#build-order)
- [P2-D1: Full-scale data pipeline](#p2-d1-full-scale-data-pipeline)
- [P2-D2: Data-mixing configuration](#p2-d2-data-mixing-configuration)
- [P2-D3: Warmup-stable-decay schedule](#p2-d3-warmup-stable-decay-schedule)
- [P2-D4: Replay buffer](#p2-d4-replay-buffer)
- [P2-D5: Optimizer offload](#p2-d5-optimizer-offload)
- [P2-D6: Forgetting evaluation](#p2-d6-forgetting-evaluation)
- [P2-D7: Recipe card and reference run](#p2-d7-recipe-card-and-reference-run)
- [Phase 2 exit checklist](#phase-2-exit-checklist)

## Preconditions

Phase 2 assumes Phase 0 and Phase 1 exist as designed. Before starting any
deliverable below, verify these interfaces are present in the repository and
fail the session loudly if they are not:

| Dependency | From | What Phase 2 consumes |
| :-- | :-- | :-- |
| Training container | Phase 0 D1 | The docker/training image with TRL, PEFT, DeepSpeed (CPUAdam pre-built with znver5 flags), torchao, flash-attn; `make train-shell` |
| Lite data pipeline and shard format | Phase 0 D2 | The tokenize-and-shard writer, manifest contract, and `ShardedTokenDataset` reader with serializable cursor. P2-D1 adds stages *in front of* this writer; it does not replace it |
| Recipe system | Phase 0 D5 | The Hydra dataclass schema. Phase 2 extends it with `mixture`, `replay`, and `wsd` blocks and new `validate()` invariants |
| Tracking and checkpointing | Phase 0 D6 | The reference loop, atomic checkpoints, deterministic-resume test harness |
| Train-and-serve round trip | Phase 0 D7 | GGUF export against the pinned llama.cpp for serving the adapted model |
| Heterogeneous-compute skeleton | Phase 0 D9 | Prefetch dataloader, run lock, affinity wrapper (main on one CCD, workers on the other) |
| Watchdogs | Phase 0 D10 | RAM-floor and disk-floor signals; Phase 2's offload makes the RAM floor load-bearing |
| Fine-tuning evaluation wrapper | Phase 1 deliverable 7 | `scripts/eval/eval_ft.py` (lm-evaluation-harness integration and scorecard shape); P2-D6 wraps it |

The concurrency policy, storage layout, style guides, and Python 3.12 pin from
the Phase 0 design's cross-cutting conventions all apply unchanged.

## Cross-cutting conventions

Decisions every Phase 2 deliverable inherits. Ecosystem facts were verified as
of 2026-07; re-verify pins at implementation time.

**Policy decisions (settled, do not re-litigate).**

- Continual pre-training is full-parameter training of the transformer stack
  with the token embedding, LM head, and Gemma-4 per-layer-embedding (PLE)
  tables frozen (see [the base model](#the-base-model-gemma-4-e2b) for why).
  A LoRA-based continual-pretrain path is out of scope for Phase 2; the PEFT
  machinery from Phase 1 already covers it if needed later.
- Optimizer state lives in system RAM via DeepSpeed ZeRO-Offload
  (stage 2 + CPU offload) with `DeepSpeedCPUAdam`. bf16 weights and
  gradients on the GPU; fp32 optimizer state on the host. FP8 (Phase 0 D8)
  is not combined with offload in Phase 2 — one new variable at a time.
- The tokenizer is Gemma-4's, always. Phase 0 D3 tokenizer training is not
  used in this phase; the shard manifest's tokenizer hash must match the
  base model's tokenizer, enforced by the existing loader check.
- Because Gemma-4's vocabulary is 262,144 (> 65,536), shards are uint32.
  This doubles bytes-per-token versus the uint16 default; token budgets and
  the disk-floor math below account for it.
- "Replay of the original distribution" uses a public proxy corpus, not
  Gemma's actual pretraining mix, which Google does not release. The proxy
  is a decision recorded in the recipe (default: FineWeb-Edu sample for web
  text plus a code slice such as the StarCoder2 permissive set). Forgetting
  numbers are interpreted accordingly: replay approximates the base
  distribution, it does not reproduce it. This caveat is stated in the
  forgetting report (P2-D6), not just here.
- Decontamination is against the exact benchmark sets the forgetting
  evaluation uses. A benchmark added to P2-D6 later requires re-running the
  decontamination stage; the pipeline manifest records which benchmark list
  a shard set was decontaminated against so the mismatch is detectable.
- All heavy data lives under /mnt/ai-data/training/ per the storage policy.
  Pipeline intermediates are regenerable and get deleted after shards are
  validated; manifests are kept.

**Ecosystem facts (verified 2026-07, re-check when implementing).**

- datatrove (Hugging Face) provides single-node LocalPipelineExecutor with
  multiprocess workers, MinHash dedup blocks, quality filters
  (Gopher/FineWeb heuristics), fastText language id, and NGram
  decontamination. It is the primary pipeline engine here; text-dedup is
  the fallback for MinHash-LSH if datatrove's block misbehaves at this
  scale.
- Gemma-4 is supported in current transformers (AutoModelForCausalLM) and
  in llama.cpp/GGUF (Google publishes QAT GGUFs; community GGUFs exist), so
  the D7 round trip applies. Confirm the pinned llama.cpp revision in
  training/export/llama_cpp_pin.txt includes the gemma-4 architecture
  before the export step; bump the pin deliberately if not.
- DeepSpeed CPUAdam on Zen 5: the AVX-512 miscompilation risk (upstream
  issue #7556) is handled at Phase 0 D1 image-build time; Phase 2 only
  consumes the pre-built op. If the op is missing, that is a Phase 0
  regression, not something to patch here.
- lm-evaluation-harness is the benchmark engine for P2-D6, reusing the
  Phase 1 wrapper's integration.

## The base model: Gemma-4-E2B

The pinned base for the Phase 2 reference run is `google/gemma-4-E2B`
(base, not `-it`). Facts that shape the design (re-verify against the model
card at implementation time; record the exact model revision hash in the
recipe):

- **~5.1B total parameters, ~2.3B effective.** The ~2.8B gap is almost
  entirely the per-layer-embedding (PLE) tables: each decoder layer has its
  own 262,144 x 256 embedding lookup. PLE entries are lookups, not matmul
  participants, and standard practice is to freeze them for adaptation.
- **262,144-token vocabulary** (drives uint32 shards and a large tied
  embedding/LM-head block, also frozen).
- **128K context window**, hybrid attention interleaving local
  sliding-window layers with global layers (final layer global), unified
  KV on global layers, proportional RoPE. Training sequence length in this
  phase is 4,096 (recipe-settable); nothing here exercises long context.
- **Gated on Hugging Face** under the Gemma license: HF_TOKEN (already
  handled by the Phase 0 D6 secrets convention via .env.training) must have
  accepted the license. The pipeline and recipe must not assume anonymous
  download.

**Why freezing PLE + embeddings is the settled decision, not an option.**
Trainable parameters drop from ~5.1B to ~2.3B. Under ZeRO-Offload with
CPUAdam, host RAM holds fp32 master weights + two Adam moments = 12 bytes per
trainable parameter: ~28GB for 2.3B trainable versus ~61GB for 5.1B. The
smaller figure leaves comfortable room for the replay buffer, page cache, and
pinned dataloader buffers inside 128GB; the larger one does not, once the
~10GB bf16 weight copy on GPU, activations, and pinned buffers are added.
Freezing also matches published Gemma-4 fine-tuning practice. The freeze is
implemented as a named parameter-group filter (`trainable_filter:
"transformer_stack"`) in the recipe, unit-tested by name patterns, so the
choice is visible and reversible in one place.

**Memory-fit calculator obligation.** Phase 0 D4's calculator must model this
run: frozen-parameter weights count toward VRAM (bf16 copy is resident for the
forward pass) but contribute zero gradient and zero optimizer state; offloaded
optimizer state is reported as host RAM. If D4 as built cannot express frozen
parameter groups, extending it is in scope for P2-D5 and the reference
configs in training/recipes/reference/ gain a
`continual-gemma4-e2b` calibration entry.

## Build order

P2-D1 pipeline ──> P2-D2 mixing ──┐
                                   ├──> P2-D7 reference run
P2-D3 WSD schedule ──┐             │
P2-D4 replay buffer ─┴──> P2-D5 offload config ──> P2-D6 forgetting eval ──┘

P2-D1 and P2-D3 are independent and can start in parallel sessions
(worktrees). P2-D2 and P2-D4 both extend the data-loading path and share the
source-tagging mechanism — build P2-D2 first, then P2-D4 on top of it.
P2-D5 needs a real model and dataloader to calibrate against. P2-D6 needs
nothing from the trainer and can be built any time after Phase 1, but its
benchmark list gates P2-D1's decontamination stage, so fix the benchmark list
(a one-hour decision, recorded in P2-D6) before running the pipeline at scale.
P2-D7 is the integration run that closes the phase.

## P2-D1: Full-scale data pipeline

**Tag:** orchestrate. **Depends on:** Phase 0 D2 (shard writer and manifest);
the P2-D6 benchmark list (for decontamination).

### Purpose

Add the quality stages the lite pipeline deliberately omitted — MinHash-LSH
deduplication, heuristic quality filtering, language identification, and
benchmark decontamination — in front of the existing shard writer, parallel
across 32 threads, so domain corpora enter training clean. The shard format,
manifest contract, and reader do not change.

### Decisions

1. **Engine: datatrove with the LocalPipelineExecutor.** The stages are
   datatrove blocks composed in training/data/pipeline.py; this repository
   writes configuration and two custom blocks (below), not dedup or filter
   algorithms. Fallback if datatrove's MinHash blocks fail at this scale:
   text-dedup for the MinHash-LSH stage only, same stage boundary.
2. **Stage order and defaults:** ingest (HF dataset or local JSONL) ->
   language id (fastText, keep `en` above 0.65 by default, configurable) ->
   heuristic quality filters (Gopher + FineWeb-style: length, symbol ratio,
   repetition; thresholds in one config block, defaults from datatrove) ->
   exact dedup (xxhash on normalized text) -> MinHash-LSH near-dedup
   (5-grams, 112 hash functions, ~0.75 Jaccard threshold — datatrove
   defaults; recorded in the manifest) -> decontamination -> the Phase 0 D2
   tokenize-and-shard writer.
3. **Decontamination is n-gram overlap against the P2-D6 benchmark list.**
   A build step extracts benchmark test-set n-grams (13-grams, following
   common practice) into /mnt/ai-data/training/datasets/_contamination/
   with its own manifest naming every benchmark and revision included.
   Documents with any hit are dropped and counted. The shard manifest gains
   a `decontaminated_against` field carrying that manifest's hash.
4. **Stage boundaries are JSONL directories** under
   /mnt/ai-data/training/datasets/<name>/stage_<k>_<label>/, each with a
   small stats.json (docs in, docs out, drop reasons histogram). They are
   regenerable intermediates: a `--clean-intermediates` flag removes all
   but the final stage once shards validate. The final manifest embeds the
   full per-stage funnel so every published number about the corpus is
   reproducible from the manifest alone.
5. **Parallelism:** datatrove tasks = 30 workers (leave two cores for OS and
   the writer), matching the Phase 0 D2 convention, under the
   inference-down policy. The affinity wrapper is not used here; the
   pipeline is a CPU-only batch job that owns the machine.
6. **Module layout:** training/data/pipeline.py (stage composition and
   config dataclass), training/data/contamination.py (n-gram index build
   and filter block), thin CLI at scripts/train/prepare_corpus.py, and a
   `make prepare-corpus ARGS=...` target. The existing prepare_data.py
   stays as the lite path; prepare_corpus.py is a superset that ends by
   invoking the same writer.

### Interfaces

    scripts/train/prepare_corpus.py \
        --source HuggingFaceFW/fineweb-edu --split train \
        --pipeline-config training/data/configs/default.yaml \
        --contamination /mnt/ai-data/training/datasets/_contamination/v1 \
        --tokenizer google/gemma-4-E2B \
        --out /mnt/ai-data/training/shards/<name> \
        --max-tokens 2_000_000_000 --num-proc 30

### Risks

- MinHash at tens of millions of documents is disk- and RAM-hungry;
  datatrove's staged signature/bucket/cluster layout handles it but needs
  scratch space — point scratch at /mnt/ai-data/training/datasets/_tmp and
  respect the disk-floor watchdog values.
- Overzealous quality filters can gut a specialized domain corpus (code,
  math). Filters are per-recipe toggles; the stats.json funnel makes a
  pathological drop rate visible before any training run wastes GPU days.

### Acceptance

- A corpus run over a small HF dataset produces shards whose manifest
  records the full stage funnel, MinHash parameters, and
  `decontaminated_against` hash.
- Planted near-duplicates (paraphrase-level) and planted benchmark
  passages in a test JSONL are dropped, each attributed to the right stage
  in stats.json; a unit test covers both.
- Re-running the pipeline with identical inputs and config is
  deterministic (identical final shard hashes).
- A 1B+-token run sustains CPU-bound throughput across 30 workers with no
  writer stall (same bar as Phase 0 D2).

## P2-D2: Data-mixing configuration

**Tag:** orchestrate. **Depends on:** Phase 0 D2 reader, D5 schema.

### Purpose

Weighted sampling across multiple shard directories — the mechanism that
balances new-domain data against replay data and any auxiliary sources — as a
recipe block, verified by construction.

### Decisions

1. **Recipe block:**

       mixture:
         sources:
           - {name: domain, path: /mnt/ai-data/training/shards/<domain>, weight: 0.8}
           - {name: replay_web, path: .../shards/replay-fineweb, weight: 0.15}
           - {name: replay_code, path: .../shards/replay-starcoder, weight: 0.05}
         seed_from_run: true

   Weights are normalized proportions of *tokens drawn*, not of files.
   `validate()` (Phase 0 D5) gains invariants: weights positive, paths
   exist, all sources share one tokenizer hash.
2. **Implementation: `MixtureDataset`** in training/data/mixture.py wrapping
   one `ShardedTokenDataset` per source. Sampling is deterministic given
   (seed, step): a per-draw categorical over weights, so resume determinism
   (Phase 0 D6) is preserved by saving each source's cursor plus the mixture
   RNG state in dataloader.json. Every batch element carries a `source`
   integer tag used by accounting and by the replay buffer (P2-D4).
3. **Accounting is a first-class output.** The loop logs cumulative
   tokens-per-source to tracking every N steps; the run's meta.json gains a
   final per-source token table. Acceptance for "batches contain the
   configured proportions" reads this accounting, not a spot check.
4. **Epoch semantics:** sources may exhaust at different times. Default is
   reshuffle-and-continue per exhausted source (with epoch counter in the
   cursor); a `strict_single_epoch` flag aborts instead, for runs where
   repeating domain data would confound a measurement.

### Acceptance

- A three-source toy mixture run of 10k draws matches configured weights
  within 1 percent, verified from the source tags.
- Checkpoint/resume mid-run reproduces the identical draw sequence
  (extends the Phase 0 D6 deterministic-resume test to MixtureDataset).
- Mismatched tokenizer hashes across sources are refused at validate().

## P2-D3: Warmup-stable-decay schedule

**Tag:** orchestrate. **Depends on:** Phase 0 D5, D6.

### Purpose

The learning-rate shape that distinguishes continual pre-training from a
naive resume: rewarm from near zero to a peak well below the base model's
original peak, hold stable, then decay — so early gradients on the new
distribution do not blow away base capabilities.

### Decisions

1. **training/schedules.py** with a `wsd` entry beside the existing cosine:
   linear warmup over `warmup_steps` to `peak_lr`, constant until
   `decay_start_frac` of total steps, then decay to `final_lr_frac *
   peak_lr` (default decay: inverse-sqrt-to-linear "1-sqrt" per the WSD
   literature; cosine decay as an option). Pure function of step ->
   multiplier, wrapped in LambdaLR; scheduler state already checkpoints via
   Phase 0 D6.
2. **Rewarming defaults for the reference run:** peak_lr defaults to a
   `peak_lr_rule: fraction_of_base` helper documented in the recipe —
   in practice a fixed value on the order of 1e-4 or below for a ~2B-class
   trainable stack; the reference recipe pins the concrete number after the
   first stability probe (a 200-step canary at candidate LRs, watching the
   D10 loss-explosion signal, documented in the run notes).
3. **Recipe:** `schedule: {type: wsd, warmup_steps, decay_start_frac,
   final_lr_frac, total_steps}`. validate() rejects wsd without
   total_steps and warmup >= decay start.
4. **Checkpoint-at-stable convention recorded now:** WSD's operational
   advantage is branching decay from a stable-phase checkpoint. Phase 2
   only documents the convention (checkpoints during stable phase are
   decay-branchable; the retention policy must keep at least one); the
   multi-branch tooling belongs to later phases.

### Acceptance

- Unit test pins the multiplier curve at hand-computed steps for a small
  config (warmup boundary, stable value, decay endpoints).
- A tiny run's tracked lr curve shows the three segments; resume mid-stable
  and mid-decay continues the curve exactly.

## P2-D4: Replay buffer

**Tag:** build. **Depends on:** P2-D2 (source mechanism), Phase 0 D9.

### Purpose

The anti-forgetting mechanism: a configurable fraction of every effective
batch comes from the (proxy) base distribution, held resident in system RAM
so replay never competes with the domain stream for NVMe bandwidth.

### Decisions

1. **Replay is a mixture source with a residency guarantee, not a parallel
   code path.** training/continual/replay.py provides `ReplayBuffer`, which
   presents the same interface as ShardedTokenDataset and is wired in as a
   mixture source whose weight *is* the replay fraction. One sampling
   mechanism, one accounting path, one determinism story (all inherited
   from P2-D2).
2. **Residency:** the buffer loads a fixed token budget (`replay.max_gb`,
   default 16GB ≈ 4B tokens at uint32) from the replay shard set into
   pinned or plain host arrays at run start, sampling windows uniformly
   from RAM thereafter. If the replay shard set exceeds the budget, a
   uniform subsample is taken with a logged seed; the manifest of what was
   loaded is written to the run dir. The D10 RAM-floor watchdog is the
   backstop; the buffer additionally checks psutil available-RAM at load
   and refuses to start if loading would cross the floor.
3. **Replay fraction default 0.2** for the reference run (in the recipe, as
   `mixture` weights — a `replay:` convenience block expands into mixture
   sources so the recipe reads as intent: `replay: {fraction: 0.2, sources:
   [...], max_gb: 16}`).
4. **Proxy-corpus decision recorded here:** replay sources for Gemma-4-E2B
   are FineWeb-Edu (web/knowledge) and a permissive code slice, prepared
   through the P2-D1 pipeline like any corpus. The forgetting report must
   carry the proxy caveat from the cross-cutting conventions.

### Acceptance

- Source-tag accounting over a test run shows the configured replay
  fraction within 1 percent.
- The buffer respects max_gb (refuses or subsamples, both tested) and the
  RAM-floor refusal path fires against a mocked psutil.
- Resume reproduces the identical replay draw sequence.
- Throughput with replay on is not measurably worse than a pure-shard run
  of the same mixture (RAM reads must not be slower than mmap; if they
  are, that is a bug).

## P2-D5: Optimizer offload

**Tag:** orchestrate. **Depends on:** Phase 0 D1 (CPUAdam op), D6 (loop),
D9 (affinity, pinned memory), P2-D2/D4 (real dataloader shape).

### Purpose

The DeepSpeed ZeRO-Offload configuration that makes full-parameter continual
pre-training of the Gemma-4-E2B transformer stack fit: bf16 weights and
gradients on the 32GB GPU, fp32 master weights and Adam moments in system
RAM, the optimizer step on the 9950X.

### Decisions

1. **ZeRO stage 2 with CPU offload** (`offload_optimizer: {device: cpu,
   pin_memory: true}`), single-GPU. Stage 3 (parameter offload) is
   explicitly not used: the ~10GB bf16 weights fit in VRAM; streaming them
   would only add PCIe traffic. ZeRO-Infinity/NVMe offload is out of scope.
2. **Config lives in the recipe**, not a loose ds_config.json: the recipe's
   `memory.offload: zero_offload` expands to a generated DeepSpeed config
   (training/offload.py) so there is exactly one source of truth. Knobs
   surfaced in the recipe: gradient accumulation, allgather/reduce bucket
   sizes (default 2e8, tuned during calibration), pin_memory.
3. **Frozen parameters stay out of DeepSpeed's optimizer partitioning:**
   the trainable_filter (cross-cutting conventions) is applied before
   engine initialization; embeddings, LM head, and PLE tables are excluded
   from the param groups. A unit test asserts the trainable-parameter
   count is within the expected ~2.3B envelope and that no `embed`/`ple`
   pattern appears in optimizer groups.
4. **Host-RAM budget is computed and enforced, not hoped:** extend the
   Phase 0 D4 calculator to report host-side bytes (12 x trainable params
   + pinned gradient buckets + replay.max_gb + a page-cache allowance) and
   have validate() warn when recipe-implied host RAM exceeds
   128GB minus the watchdog floor. Add the calibration recipe
   reference/continual-gemma4-e2b.yaml and run the D4 calibration harness
   against it (this is the Phase 2 counterpart of the 10-percent
   accuracy criterion).
5. **CPU placement:** the optimizer step and the dataloader workers compete
   for cores. Default affinity layout for offload runs: main process +
   CPUAdam threads on CCD0 (cores 0-7), dataloader workers + checkpoint
   writer on CCD1 (8-15), using the Phase 0 D9 wrapper; measured, and
   parked off-by-default if it does not help (same rule as D9).
6. **Checkpointing under DeepSpeed** uses DeepSpeed's checkpoint API into
   the Phase 0 D6 directory layout (already anticipated by D6's metadata
   note). The deterministic-resume acceptance test must pass with offload
   enabled — resume correctness under ZeRO is the single riskiest
   integration point in this phase, so it is tested first, on a tiny
   model, before any Gemma-scale run.

### Risks

- DeepSpeed + frozen-parameter groups + resume has historically had sharp
  edges; hence the tiny-model resume test before scale.
- CPUAdam throughput bounds step time: ~2.3B fp32 params through the AVX-512
  op per step. Measure step-time breakdown (GPU compute vs optimizer wait)
  in calibration; if the optimizer step dominates, raise gradient
  accumulation before touching anything else.

### Acceptance

- The tiny-model offload run passes the deterministic resume test.
- Gemma-4-E2B with the frozen filter trains for 500+ steps at seq 4096
  without VRAM OOM; peak VRAM, host RAM, and tokens/s are recorded as a
  benchmark JSON (Phase 0 D8 schema) under docs/training/optimizations/.
- Calculator host-RAM and VRAM predictions for the calibration recipe land
  within 10 percent of measured.
- Watchdog RAM-floor alert demonstrably fires when replay.max_gb is set
  adversarially high against a lowered floor (integration test).

## P2-D6: Forgetting evaluation

**Tag:** build. **Depends on:** Phase 1 deliverable 7 (eval wrapper);
independent of the trainer.

### Purpose

The measurement that makes "domain adaptation without forgetting" a claim
instead of a hope: compare base and adapted checkpoints on the new domain and
on general-capability benchmarks, and report retention against a threshold.

### Decisions

1. **scripts/eval/forgetting.py** (logic in training/continual/eval.py)
   takes `--base <model>` and `--adapted <checkpoint-or-model>` plus a
   benchmark config, runs both through the Phase 1 lm-evaluation-harness
   wrapper at bf16 (transformers backend — quantization is a confound;
   the D7 quantization-confound rule applies if GGUF-served evals are ever
   compared), and emits one JSON report.
2. **Benchmark list, fixed now** (this list gates P2-D1 decontamination):
   general retention: hellaswag, arc_easy, arc_challenge, piqa, winogrande,
   mmlu (5-shot); plus base-model perplexity on a held-out replay-proxy
   slice. Domain improvement: perplexity on a held-out domain slice plus
   one task-appropriate benchmark chosen with the domain corpus and
   recorded in the recipe. The list lives in
   training/continual/benchmarks.yaml with revisions, and its hash is what
   shard manifests reference via `decontaminated_against`.
3. **Report shape:** per-benchmark base score, adapted score, delta,
   relative regression; a summary verdict computed against
   `retention.max_regression` (default 2 percent relative, per benchmark)
   and `domain.min_improvement` from the recipe. Plus the replay-proxy
   caveat string, model hashes, benchmark-list hash, and eval-harness
   version. Written to the run dir and, with `--publish`, to
   docs/training/evaluations/.
4. **Cheap-first execution:** the report supports a `--quick` tier
   (subsampled benchmarks) for mid-run checkpoint spot checks, and the full
   tier for the phase acceptance. Mid-run quick evals run per the
   concurrency policy's overlap rule (CPU/idle-GPU evaluation of earlier
   checkpoints is Phase 0 D9 territory; if that overlap is not yet real,
   they simply run between training runs).

### Acceptance

- Running base-vs-base yields deltas of zero (harness plumbing sanity
  check, automated).
- The report JSON validates against a checked-in schema; unit tests cover
  verdict computation at the threshold boundaries.
- One full base-vs-adapted report exists for the P2-D7 reference run.

## P2-D7: Recipe card and reference run

**Tag:** orchestrate. **Depends on:** everything above.

### Purpose

The validated, committed recipe that proves the phase end to end:
recipes/continual-pretrain-gemma4-e2b.yaml adapting Gemma-4-E2B to one
concrete domain, with the run's artifacts as evidence.

### Decisions

1. **Domain for the reference run:** chosen at implementation time with
   these criteria, recorded in the recipe notes: public and permissively
   usable, clearly out-of-distribution enough to move a domain metric
   (candidate default: a technical corpus such as pes2o/scientific text or
   a domain-specific code corpus), and small enough that 1-2 overnight
   GPU-days suffice (~2-4B training tokens at the measured tokens/s).
2. **Recipe contents:** model gemma-4-E2B at a pinned revision,
   trainable_filter transformer_stack, seq 4096, uint32 domain shards from
   P2-D1, mixture with replay fraction 0.2 (P2-D2/D4), wsd schedule
   (P2-D3), zero_offload memory block (P2-D5), watchdog block with the
   RAM floor set for offload, checkpoint retention keeping at least one
   stable-phase checkpoint (P2-D3 convention), tracking on.
3. **Run protocol:** stability canary (200 steps, P2-D3) -> full run ->
   full forgetting report (P2-D6) -> GGUF export and smoke serve through
   llama-gpu (Phase 0 D7; verify the llama.cpp pin supports gemma-4
   first) -> benchmark JSON and report published under docs/training/.
4. **This run is the phase's integration test.** Anything discovered here
   (calculator misses, resume breaks, throughput cliffs) is fixed in the
   owning deliverable, not patched in the recipe.

### Acceptance

- The recipe passes validate() and drives the full run unattended
  (watchdogs armed) to completion or a clean watchdog abort with a
  resumable emergency checkpoint.
- The forgetting report shows domain improvement at or above the recipe's
  threshold with every general benchmark regression within
  retention.max_regression — or the negative result is published with the
  same rigor and the phase exit decision is made explicitly.
- The adapted model round-trips to GGUF and answers the D7 smoke prompts
  through llama-gpu.
- Mixture accounting in the run's meta.json matches the recipe weights.

## Phase 2 exit checklist

Restating the scope document's acceptance criteria against this design:

1. Full-parameter continual pre-training with optimizer state offloaded to
   system RAM runs on Gemma-4-E2B (~5.1B total, ~2.3B trainable with the
   settled freeze policy — this supersedes the scope doc's "3-billion-
   parameter base" phrasing) (P2-D5, P2-D7).
2. Batches contain the configured replay fraction and mixture proportions,
   verified by source-tag accounting (P2-D2, P2-D4).
3. The adapted model improves its domain benchmark while keeping
   original-domain regression under the recipe threshold, per the
   forgetting report with the replay-proxy caveat stated (P2-D6, P2-D7).
4. The pipeline produced decontaminated, deduplicated shards whose
   manifests trace the full stage funnel and the benchmark list used
   (P2-D1).
5. Deterministic resume holds under ZeRO-Offload and mixture sampling
   (P2-D2, P2-D5).
6. The adapted checkpoint serves through the existing inference services
   via the Phase 0 round trip (P2-D7).

---

*Last Updated: 2026-07-09*
