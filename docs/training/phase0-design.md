# Phase 0 product designs

This document is the implementation-ready product design for every Phase 0
deliverable in [scope.md](scope.md). Each section is written to be picked up in a
fresh session and implemented without re-deriving context: it records the
decisions already made, the interfaces to build, the repository and ecosystem
facts the design depends on, and the tests that define done.

Scope.md defines *what* Phase 0 delivers; this document defines *how*. Where the
two disagree, this document wins and scope.md should be corrected.

## Table of Contents

- [Cross-cutting conventions](#cross-cutting-conventions)
- [Build order](#build-order)
- [D1: Training container and dependency group](#d1-training-container-and-dependency-group)
- [D2: Lite data pipeline](#d2-lite-data-pipeline)
- [D3: Tokenizer training](#d3-tokenizer-training)
- [D4: Memory-fit calculator](#d4-memory-fit-calculator)
- [D5: Recipe system](#d5-recipe-system)
- [D6: Tracking and resumable checkpointing](#d6-tracking-and-resumable-checkpointing)
- [D7: Train-and-serve round trip](#d7-train-and-serve-round-trip)
- [D8: FP8 enablement and benchmark harness](#d8-fp8-enablement-and-benchmark-harness)
- [D9: Heterogeneous-compute skeleton](#d9-heterogeneous-compute-skeleton)
- [D10: Operational watchdogs](#d10-operational-watchdogs)

## Cross-cutting conventions

Facts and decisions every deliverable inherits. Verified against the repository
and the ecosystem as of 2026-07; re-verify version pins at implementation time.

**Repository facts.**

- Inference services and their real ports come from `docker-compose.yaml`:
  llama-cpu on 8001, llama-gpu on 8002, vllm-gpu on 8003. Documentation elsewhere
  in the repository (CLAUDE.md) claims 8004/8005 and is stale; the compose file
  is the source of truth.
- All services mount models read-only from `/mnt/ai-data/models` and select the
  served model through environment variables (`LLAMA_GPU_MODEL` and so on) read
  by the container entrypoint at start. Changing the served model means
  recreating the container.
- PyTorch is installed outside Poetry via
  `scripts/update/update_python.sh --pytorch`, which pip-installs
  torch/torchvision/torchaudio from the cu130 index inside the Poetry
  environment. Poetry cannot manage the custom index; this split is deliberate
  and the training additions must respect it.
- Existing GPU images build from `nvidia/cuda:13.0.2-cudnn-*-ubuntu24.04` with
  `TORCH_CUDA_ARCH_LIST="12.0"` / `CMAKE_CUDA_ARCHITECTURES=120`, multi-stage
  builds, a non-root `appuser`, `read_only: true` rootfs with tmpfs mounts,
  `cap_drop: ALL`, and `no-new-privileges`. The training image follows the same
  pattern.
- Python is 3.12 (`>=3.12,<3.13` in pyproject.toml). Style rules live in
  `.claude/PYTHON_STYLE.md`, `.claude/COMMENT_STYLE.md`,
  `.claude/BASH_OUTPUT_STYLE.md`, `.claude/DOCUMENTATION_STYLE.md`; no emojis,
  plain technical prose, `black` + `ruff` formatting per pyproject settings.

**Policy decisions (settled, do not re-litigate).**

- Concurrency: inference services are down during training. Training owns all 16
  cores / 32 threads, 128GB RAM, and 32GB VRAM. Serving-dependent steps run as
  separate phases with training idle.
- Storage: code and recipes in git; datasets, shards, checkpoints, and run
  outputs under `/mnt/ai-data/training/{datasets,shards,checkpoints,runs}`.
  Checkpoint retention is keep-last-N plus every-Mth, set per recipe.
- FP8: torchao float8 (tensorwise) is the primary backend; Transformer Engine is
  experimental; bfloat16 is the fallback.
- torchtune is excluded (deprecated upstream 2025). TRL v1.x is the fine-tuning
  trainer; Unsloth arrives in Phase 1.
- GGUF conversion uses a pinned llama.cpp revision.
- GPU telemetry through NVML (pynvml), not DCGM.
- Pretraining infrastructure is production-capable at 1B parameters and 100B+
  token corpora even though Phase 3 executes at 300-500M.

**Ecosystem facts (verified 2026-07, re-check pins when implementing).**

- flash-attn 2.8.3.post1 ships prebuilt wheels tagged for CUDA 13 / torch 2.9
  (`cu13torch2.9`). sm_120 runs FA2 kernels; FA3/FA4 do not target consumer
  Blackwell. cuDNN SDPA via `torch.nn.functional.scaled_dot_product_attention`
  is the fallback attention path.
- bitsandbytes >=0.47 supports sm_120 on Linux; current 0.49.x ships CUDA
  11.8-13.0 binaries in the PyPI wheel.
- DeepSpeed ops must be pre-built in the image (`DS_BUILD_CPU_ADAM=1`); watch
  upstream issue #7556 (AVX-512 intrinsics emitted without matching compiler
  flags on AMD CPUs; the 9950X has AVX-512, so build with flags that match,
  ideally `-march=znver5`).
- torchao float8 tensorwise training works on sm_120 through
  `torch._scaled_mm` (torch >=2.7). Transformer Engine's sm_120 FP8 support is
  partial (tensorwise recipes work in recent 2.x, MXFP8 explicitly unsupported).
- TRL is v1.x (v1.7.x as of 2026-07) with Unsloth kernels integrated.
- DCGM profiling metrics are unavailable on GeForce; NVML exposes temperature,
  power, clocks, utilization, and VRAM fully on the RTX 5090.

**Code layout.** Training source is a plain importable package at `training/`
(the repository is `package-mode = false`; scripts add the repository root to
`sys.path` or rely on the working directory, matching existing scripts). Tests
live in `tests/training/`, runnable with `make test` (pytest). Every deliverable
that produces a CLI gets a Makefile target under a new `##@ Training` section.

## Build order

Dependencies among the deliverables, not calendar:

```
D1 container ── D5 recipes ──┬── D2 data pipeline ── D3 tokenizer
                             ├── D4 memory-fit
                             ├── D6 tracking/checkpointing ── D9 het-compute
                             │                                 └── D8 FP8 bench
                             ├── D10 watchdogs
                             └── D7 round trip (needs D6 checkpoints to export)
```

D5 (recipe schema) is the contract nearly everything consumes; design it
immediately after the container even though it is tagged orchestrate. D7 is last
because it needs a real exported artifact and the serving swap machinery. A
minimal training loop driver (see D5/D6) emerges as a byproduct: Phase 0 does not
deliver a production trainer, but the acceptance criteria require "a recipe
drives a tiny end-to-end training step", so D6 includes a deliberately small
reference loop used only for foundation testing.

---

## D1: Training container and dependency group

**Tag:** orchestrate. **Depends on:** nothing.

### Purpose

One reproducible environment where every later deliverable runs: PyTorch cu130 on
sm_120 with the full training stack imported and CUDA-verified at build time. The
container is the real lockfile; the Poetry group covers only what Poetry can
manage.

### Decisions

1. **Base image:** `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04`, the same base
   family as llama-gpu. Not the NGC PyTorch image: NGC ships a forked torch build
   and the HF ecosystem (TRL, bitsandbytes, Unsloth later) is tested against
   vanilla pip wheels; ABI skew there costs more than NGC's preintegrated TE is
   worth, given torchao is the primary FP8 backend.
2. **Torch from pip cu130 wheels** inside the image, pinned to an exact version
   (start from the version `update_python.sh` currently resolves; record it in
   the Dockerfile as an ARG).
3. **Single-stage is acceptable** if image size is tolerable; prefer a two-stage
   build (builder compiles DeepSpeed ops and any source-built wheels, runtime
   copies site-packages) if the devel toolchain pushes the image past ~25GB.
4. **DeepSpeed ops pre-built at image build:** `DS_BUILD_CPU_ADAM=1
   DS_BUILD_UTILS=1 pip install deepspeed==<pin>` with
   `CFLAGS="-march=znver5"` (fall back to `-march=native` only if znver5 is
   rejected by the image's gcc). This front-loads the AVX-512/#7556 risk to
   build time where it is visible.
5. **Poetry `training` group** in pyproject.toml holds pure-Python deps only:
   `hydra-core`, `tokenizers`, `safetensors`, `pynvml`, `xxhash`, `zstandard`,
   plus version-range mirrors of what the image pins (trl, liger-kernel,
   torchao) so host-side tooling and IDEs resolve imports. CUDA-compiled
   packages (torch, flash-attn, bitsandbytes, deepspeed) stay out of Poetry; a
   comment in the group points at the Dockerfile as the source of truth.
6. **Container runs as non-root** with the compose hardening pattern
   (`cap_drop: ALL`, `no-new-privileges`, `read_only: true` rootfs) plus what
   training additionally needs: a writable bind mount of `/mnt/ai-data/training`
   (rw, unlike the read-only models mount), `shm_size: 16g` for dataloader
   workers, tmpfs for `/tmp` (Triton and CUDA caches, `exec` flag required for
   Triton), and the GPU device reservation copied from llama-gpu.
7. **Compose integration via profile, not always-on service.** A `training`
   service in docker-compose.yaml with `profiles: ["training"]` and
   `restart: "no"`, so `make up` never starts it and `docker compose --profile
   training run training ...` launches jobs. This also documents the concurrency
   policy structurally: the training profile and the default profile are not
   intended to run together.

### Interfaces

- `make train-build` — build the image.
- `make train-shell` — interactive shell in the container with the repository
  and `/mnt/ai-data/training` mounted, GPU attached.
- `make train-smoke` — run the smoke test (below) in the container.
- Dockerfile at `docker/training/Dockerfile.training`, entrypoint at
  `docker/training/entrypoint.sh` (thin: environment echo, optional LD_PRELOAD
  hooks for D9, then exec the given command).

### Contents (pin exact versions at implementation)

torch/cu130, flash-attn (prebuilt cu13 wheel; build from source only if the
matching wheel tag is absent), liger-kernel, bitsandbytes >=0.49, deepspeed
(ops pre-built), trl >=1.7, peft, transformers, datasets, accelerate, torchao,
safetensors, hydra-core, wandb, mlflow, tensorboard, pynvml, tokenizers,
sentencepiece. Transformer Engine is NOT installed by default; D8's experimental
TE backend documents how to add it in a variant image tag.

### Smoke test

`scripts/train/smoke_test.py`, run at the end of the image build (fails the
build on regression) and via `make train-smoke`:

1. Import torch; assert `torch.cuda.is_available()` and device capability
   `(12, 0)` (skip CUDA asserts gracefully when built on a GPU-less host, but
   `make train-smoke` on the workstation must enforce them).
2. Import transformers, trl, peft, datasets, accelerate, torchao, liger_kernel,
   hydra, safetensors, pynvml.
3. flash-attn: run a tiny forward through `flash_attn_func` on random tensors.
4. bitsandbytes: quantize a small linear layer to 4-bit and run a forward.
5. deepspeed: `from deepspeed.ops.adam import DeepSpeedCPUAdam` and step it once
   on a small parameter (proves the pre-built op loads and runs on Zen 5).
6. torchao: convert a two-layer MLP with `convert_to_float8_training` and run
   forward+backward (proves `torch._scaled_mm` works on sm_120).
7. Write a versions manifest (JSON: package -> version, CUDA, driver, GPU name)
   to stdout and, when run on the workstation, to
   `/mnt/ai-data/training/runs/_smoke/versions.json`.

### Risks

- flash-attn wheel tag drift against the chosen torch pin: resolve at build time
  by matching the wheel tag exactly; do not float either pin independently.
- DeepSpeed CPUAdam miscompilation on Zen 5 (#7556): the smoke test step 5
  catches it at build.
- Image size: devel base plus the stack lands 15-25GB; acceptable on this
  machine, but keep one image, not per-deliverable images.

### Acceptance

- `make train-build` completes; the build fails if any smoke-test step fails.
- `make train-smoke` on the workstation passes all seven steps.
- `poetry install --with training` succeeds on the host and `import hydra,
  pynvml, tokenizers` works in the Poetry environment.

---

## D2: Lite data pipeline

**Tag:** orchestrate. **Depends on:** D1, D5 (consumes the recipe `data` block).

### Purpose

Turn a Hugging Face dataset or local JSONL into tokenized, memory-mapped shards
with a manifest, fast enough to saturate 32 threads and designed for corpora up
to 100B+ tokens so Phase 3's production requirement is met by construction. This
is the lite pipeline: no dedup, filtering, or decontamination (Phase 2 adds
those in front of the same shard writer).

### Decisions

1. **Shard format: flat binary token files plus JSON manifest** (the
   nanoGPT/nanotron family), not Arrow. One `.bin` per shard containing
   contiguous token ids, dtype `uint16` when vocab < 65,536 else `uint32`;
   documents separated by the tokenizer's EOS id; fixed shard size (default 512M
   tokens, ~1GB at uint16). Rationale: training reads are sequential mmap slices
   with zero deserialization; Arrow buys nothing for a pure token stream and
   costs a heavier reader.
2. **Manifest is the contract.** `manifest.json` beside the shards:
   dataset name/path, revision, license string (required field, may be
   "unknown" but must be stated), tokenizer path + hash, dtype, EOS id, per-shard
   token counts and xxhash64 content hashes, total tokens, total documents,
   document-length histogram (log-scale buckets), creation timestamp, git SHA of
   the pipeline, and the exact CLI/recipe used. Downstream loaders refuse a
   shard directory whose manifest hash does not match (integrity) or whose
   tokenizer hash does not match the model's tokenizer (correctness).
3. **Parallelism through `datasets.map` with `num_proc`** (processes, not
   threads; tokenization is GIL-bound). Default `num_proc = 30` (leave two cores
   for the writer and OS) under the inference-down policy. Shard writing is a
   single sequential writer consuming a queue, keeping shard files append-only
   and hashable as they close.
4. **Streaming for large sources.** For datasets larger than RAM, use
   `datasets` streaming mode feeding the same tokenize-and-pack path; the writer
   does not care. Local JSONL is read in chunks. Nothing in the design ever
   requires the corpus to fit in memory.
5. **Output location:** `/mnt/ai-data/training/shards/<name>/` where `<name>`
   is caller-chosen; never inside the repository.
6. **Reader half lives here too:** `training/data/shards.py` provides
   `ShardedTokenDataset`, a mmap-backed dataset yielding fixed-length token
   windows, with a serializable cursor (shard index, offset) for D6 resume.
   Windows never cross shard boundaries (last partial window per shard is
   dropped; the manifest records the drop count so token accounting stays
   honest).

### Interfaces

```
scripts/train/prepare_data.py \
    --source HuggingFaceFW/fineweb-edu --split train \
    --tokenizer /mnt/ai-data/training/tokenizers/base \
    --out /mnt/ai-data/training/shards/fineweb-edu-10B \
    --max-tokens 10_000_000_000 --num-proc 30
```

Also accepts `--source ./data.jsonl --text-field text`. A `make prepare-data
ARGS=...` target wraps it. Module code in `training/data/prepare.py` (logic) and
`training/data/shards.py` (format read/write), so the script is a thin CLI.

### Acceptance

- A small HF dataset and a local JSONL both produce shards whose manifest token
  totals equal the sum of shard lengths, and a round-trip decode of a random
  window reproduces source text.
- Re-running with the same inputs is deterministic (identical hashes) given the
  same tokenizer and shard size.
- Throughput on the workstation is CPU-bound across the configured `num_proc`
  (observed via `htop`/`pidstat`, no writer stall), demonstrated on at least a
  1B-token run.
- `ShardedTokenDataset` cursor save/restore yields the identical next window.

---

## D3: Tokenizer training

**Tag:** build. **Depends on:** D1; D2 for evaluation corpora handling.

### Purpose

Train and evaluate byte-level BPE tokenizers for Phase 3 from-scratch models.
Fine-tuning and continual pre-training always reuse the base model's tokenizer;
this tool exists only for the from-scratch path, and the design says so to
prevent misuse (a mismatched tokenizer silently destroys a fine-tune, and
cross-tokenizer distillation breaks logit-KL).

### Decisions

1. **Engine: Hugging Face `tokenizers`**, byte-level BPE (GPT-2/Llama-3 style:
   byte-level pre-tokenization, no unknown token, digits split individually).
   SentencePiece stays available in the image but is not the default; one
   engine, one artifact format.
2. **Artifact:** a directory `/mnt/ai-data/training/tokenizers/<name>/`
   containing `tokenizer.json`, and `manifest.json` (training corpus + sampled
   byte count, vocab size, special tokens, metrics below, git SHA, timestamp,
   xxhash of tokenizer.json — the hash D2 manifests reference).
3. **Special tokens fixed at training time:** `<|bos|>`, `<|eos|>`,
   `<|pad|>` and a small reserved block (16 unused ids) for later chat/control
   tokens, so Phase 1 templates never resize embeddings.
4. **Evaluation is part of training, not optional.** On held-out samples from
   at least two domains (web text, code by default): fertility (tokens per
   whitespace word), compression (bytes per token), round-trip losslessness on
   1,000 random documents, and the same metrics computed for one or two
   reference tokenizers (SmolLM2, Llama-3.2) for comparison. Results go in the
   manifest and print as a table.

### Interfaces

```
scripts/train/train_tokenizer.py \
    --corpus /mnt/ai-data/training/datasets/mix/*.jsonl \
    --vocab-size 49152 --sample-bytes 10_000_000_000 \
    --out /mnt/ai-data/training/tokenizers/base-49k
```

Logic in `training/data/tokenizer.py` (`train()`, `evaluate()`, both importable
for tests).

### Acceptance

- Training on a fixed seed corpus is reproducible (identical tokenizer.json
  hash across two runs).
- Round-trip losslessness holds on the evaluation sample.
- The metrics table reports fertility and compression for the new tokenizer and
  the reference tokenizers on both domains.
- Unit tests cover special-token ids, reserved block presence, and refusal to
  overwrite an existing artifact directory without `--force`.

---

## D4: Memory-fit calculator

**Tag:** build. **Depends on:** D1, D5 (consumes the recipe as its input).

### Purpose

Answer, before a run starts, whether a configuration fits in 32GB and what the
maximum sequence length and batch size are. This is the differentiated build
item: it must be accurate on this machine, not generally.

### Model

Estimate peak VRAM as the sum of, all in bytes:

1. **Weights.** `P * bytes(param_dtype)`; for QLoRA-style 4-bit base loading,
   NF4 packing plus quantization constants (block size 64 => +~0.5 bit/param)
   and full-precision adapter weights.
2. **Gradients.** `P_trainable * bytes(grad_dtype)` (bf16 default; fp32 when
   grads are accumulated in fp32).
3. **Optimizer state.** Per optimizer: AdamW fp32 = 8 bytes/param plus fp32
   master weights (4) when mixed precision keeps a master copy; 8-bit Adam
   (bitsandbytes) = 2 bytes/param + per-block constants; zero when offloaded to
   CPU (ZeRO-Offload) — but then the calculator reports the *host RAM* cost
   instead, because 128GB is also finite and the replay buffer, page cache, and
   pinned buffers compete for it.
4. **Activations.** Analytic per-layer formula for the dense transformer family
   (attention + SwiGLU MLP, GQA-aware), scaled by micro-batch, sequence length,
   hidden size, layers, with a gradient-checkpointing mode that keeps only layer
   inputs plus recompute workspace. flash-attn makes attention memory linear in
   sequence length; the formula assumes it.
5. **Framework overhead.** CUDA context plus allocator slack as a calibrated
   constant (`overhead_gb`, initial value 1.5, refined by the calibration
   harness below). Report both "allocated" and "estimated reserved" numbers;
   the fits flag tests estimated-reserved against 32GB minus a safety margin
   (default 1GB).

Solving: `estimate(recipe)` returns the numbers for the given batch/seq;
`solve(recipe, axis="batch"|"seq")` binary-searches the largest value whose
estimate fits.

### Decisions

1. **Input is a recipe file (D5 schema), not a parallel argument set.** The
   calculator reads the same `model`, `optimizer`, `precision`, `memory`
   blocks the trainer reads. No drift by construction.
2. **Model description comes from the recipe's architecture fields** (layers,
   hidden, heads, kv-heads, ffn multiplier, vocab, tied embeddings flag) or, for
   HF checkpoints, from `AutoConfig` with a mapping table. Unknown architectures
   fail loudly rather than guess.
3. **Calibration harness in-tree.** `training/calculator/calibrate.py` runs a
   short real training loop for a config, records
   `torch.cuda.max_memory_allocated/reserved` and NVML peak, and writes a
   calibration record JSON. The two reference configurations from the scope
   acceptance criteria live in `training/recipes/reference/` (suggested: a 150M
   dense pretrain config in bf16, and a 1B full-AdamW config — the latter also
   validates the Phase 3 1B envelope).
4. **Output:** human table to stdout and `--json` for machine use; fields: per
   component bytes, totals (allocated, reserved-estimate), max seq at given
   batch, max batch at given seq, fits flag, host RAM estimate when offload is
   on.

### Interfaces

- `make memfit CONFIG=training/recipes/reference/pretrain-150m.yaml`
- `python -m training.calculator --config ... [--solve batch|seq] [--json]`
- Package: `training/calculator/{__init__.py,components.py,solve.py,calibrate.py}`.

### Acceptance

- Within 10 percent of measured peak on the two reference configurations
  (measured = `max_memory_allocated` for the component sum; the reserved
  estimate must not under-predict NVML peak).
- The 1B reference config is reported as fitting in 32GB without offload, and
  calibration confirms it.
- Unit tests pin the closed-form component numbers for a hand-computed tiny
  config (no GPU needed), so refactors cannot silently change the math.

---

## D5: Recipe system

**Tag:** orchestrate. **Depends on:** D1. **Everything else depends on this.**

### Purpose

One YAML file fully specifies a run. The schema is the shared contract consumed
by the trainer loop (D6), the calculator (D4), the benchmark harness (D8), the
watchdogs (D10), and later phases' entry points.

### Decisions

1. **Hydra with typed structured configs.** Dataclasses in
   `training/recipes/schema.py` registered in Hydra's ConfigStore; YAML files in
   `training/recipes/` compose against them. Typos in keys and wrong types fail
   at load, not mid-run. (Hydra's development has slowed upstream but it is
   stable, already in the scope doc, and its composition/override model is
   exactly this use case; plain OmegaConf is the fallback if a blocking issue
   appears, and the dataclass schema survives that swap.)
2. **Top-level groups:**
   - `run`: name, seed, output root (default `/mnt/ai-data/training/runs`),
     notes.
   - `model`: either `hf` (model id/path) or `scratch` (architecture fields:
     layers, hidden, heads, kv_heads, ffn_mult, vocab, rope theta, tie_embeddings).
   - `data`: shard directory or HF dataset, sequence length, micro-batch,
     gradient-accumulation steps, dataloader workers, pin_memory.
   - `tokenizer`: path (must hash-match the data shards; loader enforces).
   - `optimizer`: name (adamw, adamw8bit, cpu_adam), lr, betas, eps, weight
     decay.
   - `schedule`: warmup steps, schedule type (cosine, wsd), total steps.
   - `precision`: `bf16` | `fp8` (+ `fp8_backend: torchao|te`), grad dtype.
   - `memory`: gradient checkpointing on/off, offload none|zero_offload,
     compile on/off.
   - `checkpoint`: interval steps, retain_last N, retain_every M, resume path.
   - `tracking`: wandb on/off + project, mlflow on/off, tensorboard on/off.
   - `watchdog`: thresholds (see D10), enabled flags.
3. **Run directory convention.** Hydra's run dir is configured to
   `${run.output_root}/${run.name}/${now:...}`; inside it land the resolved
   config snapshot (`config.yaml`), logs, benchmark JSON staging, and pointers
   to checkpoint paths. Recipes in git never contain absolute output paths
   other than the defaults.
4. **Validation pass beyond types:** a `validate(recipe)` function checks
   cross-field invariants (fp8 requires supported dims per D8; cpu_adam requires
   offload; tokenizer hash matches shards) and is called by every entry point
   before any GPU allocation.
5. **Reference recipes shipped in Phase 0:**
   `reference/pretrain-150m.yaml`, `reference/pretrain-1b.yaml` (calculator
   calibration + envelope validation), `reference/tiny-e2e.yaml` (a
   seconds-long run for the acceptance tests and CI-style checks).

### Acceptance

- Loading any shipped recipe produces a fully typed object; a recipe with a
  misspelled key or wrong type fails with a clear error.
- CLI overrides (`optimizer.lr=1e-4`) work through every entry point.
- The resolved config snapshot written to the run dir reloads bit-identically.
- `validate()` unit tests cover each cross-field invariant.

---

## D6: Tracking and resumable checkpointing

**Tag:** orchestrate. **Depends on:** D1, D5.

### Purpose

Every run is observable and every run is resumable. A 1B pretrain is weeks of
wall-clock; interruption is certain, so resume correctness is the single most
load-bearing property of the foundation. Phase 0 also ships the minimal
reference training loop these facilities are tested against.

### Decisions

1. **Reference loop, deliberately small.** `training/loop.py`: dense model (HF
   `AutoModelForCausalLM` or the scratch config once Phase 3 lands), standard
   forward/backward/step with gradient accumulation, driven entirely by a
   recipe. It exists so Phase 0's acceptance criteria are testable and later
   phases have a known-good skeleton; it is not the Phase 1 SFT trainer (TRL is)
   nor the Phase 3 production loop (which may harden nanotron/litGPT instead).
2. **Checkpoint contents**, one directory per step
   (`/mnt/ai-data/training/checkpoints/<run>/step_000123400/`):
   - `model/` sharded safetensors (HF `save_pretrained` layout for HF models so
     D7 export consumes checkpoints directly).
   - `optimizer.pt`, `scheduler.pt` (torch.save; DeepSpeed runs use DeepSpeed's
     own checkpoint API into the same directory, recorded in metadata).
   - `dataloader.json` — the D2 shard cursor (shard index, offset, epoch) plus
     sampler state.
   - `rng.pt` — python, numpy, torch CPU, torch CUDA states.
   - `meta.json` — step, tokens seen, loss, recipe snapshot hash, git SHA,
     library versions, wall-clock.
3. **Atomicity:** write to `step_X.tmp/`, fsync files, then `os.rename` to the
   final name; a `latest` symlink is updated after the rename. A crash never
   leaves a half checkpoint that resume could pick up.
4. **Retention** enforced immediately after each successful save: keep the last
   `retain_last` checkpoints plus every `retain_every`-th step; delete the rest,
   oldest first, but never delete the checkpoint `latest` points to. Deletion
   respects the D10 disk-free floor logic (retention runs first; the watchdog is
   the backstop).
5. **Resume:** `checkpoint.resume=latest|<path>` restores all five components,
   then continues. The acceptance test runs N+M steps straight versus N steps,
   checkpoint, restart, M steps — with `torch.use_deterministic_algorithms(True)`
   and fixed seeds for the test only — and asserts identical loss values (exact
   match in deterministic mode; the production default remains nondeterministic
   for speed).
6. **Tracking:** a thin `training/tracking.py` fanning one `log(dict, step)` to
   any enabled sink (wandb, mlflow, tensorboard). Offline-first: wandb in
   offline mode by default (this box may be network-isolated during runs);
   syncing is a manual step. Metrics logged by the reference loop: loss, lr,
   grad norm, tokens/s, step time, GPU memory, and D10 telemetry.
7. **Secrets:** tokens (WANDB_API_KEY, HF_TOKEN) come from an untracked
   `.env.training` file passed via compose `env_file`; nothing is baked into
   the image and the file is gitignored.

### Acceptance

- The deterministic resume test passes (identical loss trajectory).
- Killing the process mid-save leaves the previous `latest` intact and
  resumable (test simulates with SIGKILL between tmp-write and rename).
- Retention test: run with small intervals, assert the surviving set is exactly
  last-N plus every-Mth.
- A tiny recipe run produces metrics visible in tensorboard and an offline
  wandb run directory.

---

## D7: Train-and-serve round trip

**Tag:** build. **Depends on:** D1, D6 (a checkpoint to export); the inference
services as they exist.

### Purpose

Close the loop the repository exists for: a trained checkpoint becomes a GGUF
the llama-gpu service serves, and an evaluation runs against the live service.
Per the concurrency policy this is phased: training is idle while serving runs.

### Decisions

1. **Pinned llama.cpp for conversion.** A specific llama.cpp revision (git SHA
   recorded in `training/export/llama_cpp_pin.txt`) is fetched and built inside
   the export flow (or a small `docker/training/Dockerfile.export` layer) for
   `convert_hf_to_gguf.py` and `llama-quantize`. Rationale: conversion and
   quantization must be reproducible independent of whatever the llama-gpu
   image was last built from; the pin is bumped deliberately, with a note, when
   new architectures are needed.
2. **Export pipeline** `scripts/export/export_gguf.py` (logic in
   `training/export/`):
   `checkpoint -> [merge adapter if present, via PEFT merge_and_unload] ->
   HF safetensors dir -> convert_hf_to_gguf -> llama-quantize -> 
   /mnt/ai-data/models/trained/<run>/<name>-<quant>.gguf` plus an export
   manifest (source checkpoint, recipe hash, llama.cpp SHA, quant type, file
   hash). Note: adapter *merging into the base* is PEFT's job; mergekit (multi-
   model/adapter merging) is a Phase 1 concern and not used here.
3. **Scope: GGUF-convertible architectures only** (Llama/Qwen/SmolLM families).
   The tool checks the model architecture against the pinned converter's
   support list and fails with a clear message otherwise. Phase 3 scratch
   models must emit an HF-compatible config of a supported family — recorded
   here as a constraint on Phase 3, not solved here.
4. **Serving swap** `scripts/eval/serve_eval.py`:
   1. Preflight: assert training is not running (no lock file, see D9), model
      file exists and hash-matches its manifest.
   2. Record current `LLAMA_GPU_MODEL` (from `.env` / environment).
   3. `LLAMA_GPU_MODEL=<new gguf> docker compose up -d --force-recreate
      llama-gpu`.
   4. Poll `http://localhost:8002/v1/health` (compose port; up to the 180s
      start budget the healthcheck already uses).
   5. Run the evaluation set: OpenAI-compatible `/v1/chat/completions` requests
      from a prompt file, recording responses, latencies, and token counts to
      the run directory as JSON.
   6. Restore the previous model (same recreate path) unless `--keep-serving`.
   Failures at any step restore the previous model before exiting nonzero.
5. **Evaluation content in Phase 0 is a fixed smoke prompt set** (a dozen
   prompts with expected-substring checks), not a benchmark harness — lm-eval
   and judge scoring are Phase 1. The JSON output shape, however, is designed
   with Phase 1 in mind: per-prompt records plus a summary block.
6. **Quantization-confound rule** recorded in the tool's docs: comparative
   claims (base vs tuned) must compare at matched quantization; serve_eval
   stores the quant type in every result file so mismatched comparisons are
   detectable.

### Acceptance

- A stock base model (e.g. SmolLM2-1.7B safetensors) exports to Q8_0 GGUF,
  serves through llama-gpu, and answers the smoke prompts; previous serving
  state is restored afterward.
- A checkpoint from the D6 tiny run (an HF-architecture model) round-trips the
  same way, proving the from-checkpoint path.
- Export manifest reproducibility: re-export of the same checkpoint yields the
  same GGUF hash with the same llama.cpp pin.
- Kill-mid-swap test: interrupting serve_eval leaves llama-gpu on the original
  model after the tool's cleanup or the documented manual restore one-liner.

---

## D8: FP8 enablement and benchmark harness

**Tag:** build. **Depends on:** D1, D5, D6 (loop to benchmark), D9 (pinning for
stable numbers), D10 (NVML plumbing shared).

### Purpose

Make `precision: fp8` real on sm_120 and measure it honestly against bf16, in a
JSON format that future optimization work (and the blog-adjacent benchmarking)
can consume. Enablement and measurement, not kernel authorship.

### Decisions

1. **Primary backend: torchao float8, tensorwise recipe** over
   `torch._scaled_mm`. Implementation:
   `convert_to_float8_training(model, config=Float8LinearConfig(...))` applied
   by the loop when `precision: fp8` and `fp8_backend: torchao`, with a module
   filter excluding `lm_head`, embeddings, and any linear whose in/out features
   are not multiples of 16 (hardware constraint on scaled matmul shapes). The
   recipe validation (D5) warns when model dims make FP8 coverage low.
2. **Transformer Engine: experimental flag.** `fp8_backend: te` swaps in
   `te.Linear`/`fp8_autocast` with the DelayedScaling recipe. Not installed in
   the default image; a `docker/training/Dockerfile.training` build-arg
   (`WITH_TE=1`) produces the variant tag. Known partial support on sm_120
   (MXFP8 unsupported); the backend is documented as may-fail and its failures
   are not release blockers.
3. **Fallback:** if FP8 initialization fails at runtime (unsupported op, driver
   quirk), the loop logs the reason, flips to bf16, and stamps the tracking run
   `precision_effective: bf16` — a run never silently reports FP8 numbers it
   did not produce.
4. **Numerics gate, not just throughput:** the benchmark includes a short
   convergence check — the reference 150M config trained for a fixed number of
   steps in bf16 and in fp8 from identical seeds; final losses must agree
   within a tolerance recorded in the JSON (default 1 percent relative). FP8
   that is fast but diverging is a failure.
5. **`scripts/benchmark/train_bench.py`:** fixed workloads defined by recipes
   (`reference/bench-150m.yaml`, plus a QLoRA-shaped one later), N warmup + M
   timed steps, reporting per precision:
   tokens/s, step time distribution (p50/p95), model FLOPs utilization, peak
   VRAM (allocated, reserved, NVML), and NVML clocks/power/temperature during
   the timed window (throttling detection: report `clocks_throttle_reasons`).
   MFU uses per-GPU peak-throughput constants pinned in
   `training/benchmark/constants.py` with a source citation comment (RTX 5090
   dense tensor peaks for bf16 and fp8); constants are code, not config, so
   every result is comparable.
6. **JSON schema and location.** This defines the training-benchmark JSON
   format (the inference side currently writes text; this schema is the model
   for unifying later). One file per invocation under
   `docs/training/optimizations/benchmarks/` when run with `--publish`
   (default writes to the run dir): `schema_version`, timestamp, git SHA,
   hardware block (GPU, driver, CUDA, CPU governor), software block (torch,
   torchao, flash-attn, image tag), recipe hash, workload block, and a results
   array with one entry per precision.

### Acceptance

- On the 150M reference workload, fp8/torchao runs and the JSON records the
  bf16-versus-fp8 comparison including MFU and peak VRAM.
- The convergence gate passes (loss parity within tolerance), or the result is
  marked failed — both outcomes recorded.
- Forcing an FP8-incompatible model (odd hidden size) produces the documented
  bf16 fallback with `precision_effective` stamped.
- Two consecutive benchmark runs at fixed clocks agree on tokens/s within a
  small tolerance (target 2 percent) — the reproducibility bar the JSON exists
  to defend; document observed variance if the target proves unrealistic.

---

## D9: Heterogeneous-compute skeleton

**Tag:** orchestrate. **Depends on:** D1, D2 (shards to load), D6 (loop to
overlap against).

### Purpose

The whole-machine pillar inside a single training job: keep the GPU step from
ever waiting on the CPU. Three concrete pieces — a pinned-memory prefetch
dataloader, a RAM-disk shard-staging helper, and a core-pinning wrapper for the
9950X — plus the measurement that proves overlap.

### Decisions

1. **Prefetch dataloader** `training/data/loader.py`: standard
   `torch.utils.data.DataLoader` over `ShardedTokenDataset` with
   `pin_memory=True`, configurable `num_workers` (default 4; token windows are
   cheap to produce, tokenization already happened offline) and
   `prefetch_factor`, wrapped by a `CudaPrefetcher` that keeps one batch ahead
   on a dedicated CUDA stream with `non_blocking=True` copies. No custom
   dataloader machinery beyond that wrapper — adopted primitives, thin glue.
2. **RAM-disk staging** `training/data/staging.py`: optionally copy the hot
   shard set to a tmpfs mount before the run and point the dataset there.
   Explicit RAM budget: refuses to stage beyond `staging.max_gb` (default 32GB)
   and falls back to NVMe mmap (which, with 128GB of page cache under the
   inference-down policy, is usually already RAM-speed — staging exists for
   controlled experiments and multi-epoch small corpora; the docstring says
   so). Compose adds a tmpfs mount `/staging` to the training service sized by
   the same setting.
3. **Core-pinning wrapper** `scripts/train/pin_run.sh` + `training/affinity.py`:
   the 9950X is one NUMA node but two CCDs (cores 0-7 and 8-15, separate L3).
   The wrapper pins the main training process (GPU-feeding thread plus CUDA
   runtime threads) to one CCD and dataloader workers plus checkpoint writer to
   the other, via `taskset`/`os.sched_setaffinity` driven by a small layout
   table in the recipe (`affinity: {main: "0-7", workers: "8-15"}`, default
   off). SMT siblings (16-31) follow their physical cores. This is CCD/L3
   pinning, not NUMA pinning; comments say so to keep the mental model honest.
4. **AOCL-LibMem toggle** lives here per the scope doc's CPU-tier strategy: the
   container entrypoint honors `TRAIN_LD_PRELOAD=/opt/aocl/libmem.so` (path
   present in the image only if the library is installed; absent by default,
   single-digit expected gain, off until profiled).
5. **Run lock.** `training/lock.py` writes `/mnt/ai-data/training/.lock` with
   pid and run name while a job runs; D7's serve_eval and any second job check
   it. Enforces the concurrency policy mechanically.
6. **Proof of overlap:** an nsys capture recipe (the repository already uses
   nsys for inference) plus a cheap in-band metric: the loop logs
   `data_wait_ms` per step (time blocked on the next batch). Acceptance uses
   the in-band metric; nsys is the diagnostic tool.

### Acceptance

- On the 150M reference workload, mean `data_wait_ms` is under 5 percent of
  step time with the prefetcher on, versus a measurably higher figure with
  `pin_memory=False, num_workers=0` (the comparison is part of the test).
- Staging respects `max_gb` and falls back cleanly when the shard set exceeds
  it.
- With affinity on, worker and main-process CPU masks are verifiably disjoint
  (`taskset -pc` in test); throughput is not worse than affinity-off on the
  reference workload (park the feature as off-by-default if it does not help —
  the wrapper existing and being measured is the deliverable).
- Lock file blocks a second concurrent run and serve_eval during training.

---

## D10: Operational watchdogs

**Tag:** orchestrate. **Depends on:** D1, D5, D6 (callback surface + telemetry
sink).

### Purpose

Multi-day unattended runs need automated abort-or-alert on the failure modes
that destroy runs or hardware value: numerical divergence, thermal/power
anomalies, RAM exhaustion under offload, and disk exhaustion under
checkpointing. NVML-based; DCGM is explicitly not used (datacenter-oriented,
degraded on GeForce — NVML exposes everything needed on the RTX 5090).

### Decisions

1. **Two layers.**
   - **In-process callback** (`training/watchdog.py`, hooked into the D6 loop
     and later TRL via its callback API): sees loss and grad norm every step,
     samples NVML and psutil every `interval_s` (default 10s) from a background
     thread.
   - **Optional out-of-process sidecar** (`scripts/train/watchdog_daemon.py`):
     same checks minus loss, for jobs whose trainer cannot host the callback.
     Phase 0 ships it as a thin wrapper over the same module; not a separate
     implementation.
2. **Signals and defaults** (all thresholds in the recipe `watchdog` block):
   - NaN/inf loss, or loss > `explode_factor` (default 10x) times its trailing
     median: **abort** after saving a final emergency checkpoint (marked
     `emergency=true`, exempt from retention deletion).
   - Gradient norm > `grad_norm_max` for `patience` consecutive steps: abort.
   - GPU temperature > `gpu_temp_c` (default 88) or hotspot/memory-junction
     where NVML exposes it: **alert**, then abort if sustained
     `sustain_s` (default 120s).
   - Power draw anomaly: sustained draw above `power_w` (default 600, the
     card's limit) or clock-throttle reasons active for > `sustain_s`: alert
     (data for D8, not usually fatal).
   - Host RAM available < `ram_floor_gb` (default 8): alert; < half the floor:
     abort before the OOM killer chooses a victim (critical under
     ZeRO-Offload in Phase 2).
   - Disk free on `/mnt/ai-data` < `disk_floor_gb` (default 100): block further
     checkpoint writes and alert; the run continues but the abort-on-NaN
     emergency save is still permitted.
3. **Actions:** `log` (tracking sink + stderr), `alert` (append to
   `runs/<run>/alerts.jsonl`; a pluggable hook function for future
   email/ntfy — Phase 0 ships file+log only), `abort` (emergency checkpoint,
   nonzero exit). Every trigger records signal, value, threshold, and timestamp
   in the run's tracking stream, so post-mortems need no log spelunking.
4. **All telemetry also flows to tracking** at the sampling interval (GPU temp,
   power, clocks, VRAM, host RAM, disk free), making the watchdog double as the
   run's hardware telemetry source, including for D8's benchmark JSON.

### Acceptance

- Unit tests inject synthetic NaN loss, temperature, RAM, and disk readings
  (NVML/psutil mocked) and assert the configured action fires, including the
  emergency checkpoint path.
- An integration run with an intentionally exploding LR aborts via the
  loss-explosion rule and leaves a resumable emergency checkpoint.
- Telemetry appears in tensorboard for a tiny reference run.
- Disk-floor behavior verified by pointing the check at a small tmpfs.

---

## Phase 0 exit checklist

Restating scope.md's acceptance criteria against the deliverables above:

1. Container builds; smoke test passes on the workstation (D1).
2. `reference/tiny-e2e.yaml` drives a training run that checkpoints and resumes
   to an identical loss trajectory in deterministic mode (D5, D6).
3. Memory-fit within 10 percent on `pretrain-150m` and `pretrain-1b` reference
   configs; the 1B config fits in 32GB without offload (D4).
4. A base model round-trips export -> GGUF -> llama-gpu (port 8002) -> smoke
   prompts, with serving state restored (D7).
5. `train_bench.py` produces the bf16-versus-fp8 JSON with the convergence gate
   recorded (D8).
6. Data pipeline produces verified shards from HF and JSONL sources; the loader
   cursor resumes exactly (D2). Tokenizer training is reproducible with metrics
   against reference tokenizers (D3).
7. `data_wait_ms` under 5 percent of step time on the reference workload (D9).
8. Watchdog abort and telemetry paths demonstrated (D10).

---
*Last Updated: 2026-07-08*
