# Inference update roadmap

This document is an implementation roadmap for the "Update Inference code" work.
It is written to be executed task by task with Claude Code: each task states the
files involved, the change to make, and the acceptance criteria that define done.
Findings and version references are current as of July 2026; re-verify pins at
implementation time.

Scope: section 1 cleans up and updates existing functionality; section 2 adds
new inference-specific features, split into proven adoptions and experimental
tracks. Work that exists to serve the training and interpretability tiers is
tracked in docs/training/ and docs/interpretability/ and is out of scope here,
except where a task below is a shared prerequisite (noted inline).

Ordering principle: pin, then fix, then update, then expose, then measure.
Reproducibility tasks come first because every later benchmark claim rests on
them. Every feature task ends with a benchmark artifact.

## Table of Contents

- [Section 1: Clean up and update current functionality](#section-1-clean-up-and-update-current-functionality)
  - [1.1 Pin llama.cpp builds](#11-pin-llamacpp-builds)
  - [1.2 Pin remaining mutable image references](#12-pin-remaining-mutable-image-references)
  - [1.3 Reconcile documentation with reality](#13-reconcile-documentation-with-reality)
  - [1.4 Fix toolchain breakage and cruft](#14-fix-toolchain-breakage-and-cruft)
  - [1.5 Consolidate Python package management](#15-consolidate-python-package-management)
  - [1.6 Update vLLM to current](#16-update-vllm-to-current)
  - [1.7 Unify the CUDA baseline](#17-unify-the-cuda-baseline)
  - [1.8 Correctness checks](#18-correctness-checks)
- [Section 2: New inference-specific features](#section-2-new-inference-specific-features)
  - [Proven, worth adopting](#proven-worth-adopting)
    - [2.1 Speculative decoding with EAGLE-3](#21-speculative-decoding-with-eagle-3)
    - [2.2 KV cache quantization](#22-kv-cache-quantization)
    - [2.3 NVFP4 and MXFP4 quantization](#23-nvfp4-and-mxfp4-quantization)
    - [2.4 Structured output](#24-structured-output)
    - [2.5 Benchmark JSON schema](#25-benchmark-json-schema)
  - [Experimental](#experimental)
    - [2.6 llama.cpp activation read path study](#26-llamacpp-activation-read-path-study)
    - [2.7 Hugepage wrapper revalidation and extension](#27-hugepage-wrapper-revalidation-and-extension)
    - [2.8 MoE expert offload study](#28-moe-expert-offload-study)
- [Execution order](#execution-order)

## Section 1: Clean up and update current functionality

### 1.1 Pin llama.cpp builds

Both llama.cpp images clone master unpinned, so every rebuild is a different
engine. This breaks benchmark reproducibility and conflicts with the training
tier's requirement (D7 in docs/training/phase0-design.md) that GGUF conversion
use a pinned llama.cpp revision.

Files: docker/llama-cpu/Dockerfile.cpu, docker/llama-gpu/Dockerfile.llama-gpu.

Tasks:

- Create a single pin file (suggested: docker/llama-cpp-pin.txt) holding one
  llama.cpp git SHA or release tag with a dated comment stating why that
  revision was chosen.
- Both Dockerfiles take the pin as a build ARG (LLAMA_CPP_REF) defaulting to
  the pinned value; the clone checks out exactly that ref. Make the llama-gpu
  clone shallow (--depth 1 with the ref) to match llama-cpu.
- Document the bump procedure in the pin file comment: change the pin, rebuild
  both images, rerun the benchmark suite, record results before merging.
- The future training export flow reads the same pin file; do not create a
  second pin.

Acceptance:

- Two consecutive image builds from the same pin produce the same llama.cpp
  version string in `llama-server --version`.
- grep for `git clone` in docker/ shows no unpinned clones of llama.cpp.

### 1.2 Pin remaining mutable image references

Files: docker-compose.yaml, docker/comfyui-flux/Dockerfile.comfyui-flux.

Tasks:

- Replace `ghcr.io/open-webui/open-webui:main` with a specific version tag or
  digest.
- Pin the ComfyUI custom node clones (ComfyUI-Manager, ComfyUI-GGUF, KJNodes,
  tooling nodes) to specific tags or SHAs via build ARGs.

Acceptance: no `:main`, `:latest`, or unpinned `git clone` remains in any
Dockerfile or the compose file.

### 1.3 Reconcile documentation with reality

The compose file and Dockerfiles are the source of truth; documentation has
drifted.

Known drift to fix:

- CLAUDE.md claims ports 8004/8005; compose defines llama-cpu 8001, llama-gpu
  8002, vllm-gpu 8003.
- docs/inference/engines/vllm.md: lists commented-out env vars
  (CUDA_ALLOCATOR_BACKEND, CUBLAS_WORKSPACE_CONFIG, CUDA_L2_PERSISTING_SIZE)
  as "Active Optimizations"; states NVCC_THREADS=8 where the Dockerfile uses
  2; shows a `python3 -m vllm.entrypoints.openai.api_server` entrypoint where
  the real entrypoint runs `vllm serve`; documents `--swap-space 4` which the
  entrypoint does not pass; links docs/inference/benchmarks/performance-results.md
  which does not exist.
- Performance tables in docs/inference/README.md and vllm.md contain
  "projected" numbers; either replace with measured values from the benchmark
  suite or label the column clearly as unmeasured.

Tasks:

- Sweep every file under docs/inference/ and docs/sandbox/ against the actual
  Dockerfiles, entrypoints, and compose definitions; fix or delete stale
  claims. Fix CLAUDE.md ports.
- Remove or create the dangling performance-results.md link.

Acceptance: every port, env var, flag, and entrypoint mentioned in
docs/inference/ matches the corresponding file in docker/ or
docker-compose.yaml; no dead links.

### 1.4 Fix toolchain breakage and cruft

Files: Makefile, docker-compose.yaml, pyproject.toml, .python-version,
poetry.lock.backup.

Tasks:

- Makefile: replace every `docker-compose` invocation with `docker compose`
  (Compose v2).
- Makefile: replace the `poetry shell` target (removed in Poetry 2.x) with
  `poetry env activate` guidance or drop the target. Revisit under task 1.5
  if the package manager changes.
- docker-compose.yaml: delete the obsolete `version: '3.9'` key.
- Delete poetry.lock.backup (byte-identical committed copy of poetry.lock).
- Fix .python-version: it contains a pyenv virtualenv name (ai-tools), not a
  version; set it to the actual Python version or remove it.
- pyproject.toml: black and ruff target py313 while the project pins
  >=3.12,<3.13; align target-version with the supported interpreter.
- Makefile `test` target runs pytest against a tests/ directory that does not
  exist at the repo root; either create a minimal tests/ (the smoke checks
  from 1.8 are candidates) or repoint the target at the hugepages C++ test
  runner and rename accordingly.

Acceptance: `make up`, `make status`, `make health`, `make lint`, and
`make format` all run without error on a machine with only Compose v2; no
target references a tool or path that does not exist.

### 1.5 Consolidate Python package management

Current state is split three ways: Poetry for the host environment, uv inside
the vLLM image, and raw pip for PyTorch via scripts/update/update_python.sh
(the cu130 index that Poetry cannot manage).

Decision to make first, then execute: standardize on uv for the host as well.
As of 2026 uv handles custom index pinning natively ([[tool.uv.sources]] and
index configuration), which removes the original reason for the Poetry/pip
split. If a blocker appears, the fallback is keeping Poetry and documenting
the split as deliberate; in that case update_python.sh stays.

Tasks (uv path):

- Convert pyproject.toml from [tool.poetry] to standard [project] metadata
  with uv-managed dependency groups; configure the pytorch cu130 index in
  [tool.uv].
- Re-resolve stale pins while converting: transformers (currently <4.52),
  datasets (<4), huggingface-hub (<0.33), and the rest of the ML group, to
  current versions compatible with Python 3.12 and torch cu130. Coordinate
  target versions with the training container design (docs/training/
  phase0-design.md D1) so host and container agree on the same majors.
- Fold the --pytorch behavior of scripts/update/update_python.sh into the uv
  configuration; reduce the script to a thin wrapper or delete it and update
  docs/os/python/ accordingly.
- Update Makefile install/update/test/lint/format targets to uv invocations.

Acceptance: a fresh `uv sync` on the workstation produces an environment where
`python -c "import torch; assert torch.cuda.is_available()"` passes and
`make test lint format` run; poetry.lock is removed; docs/os/python/ describes
the new flow.

### 1.6 Update vLLM to current

The CLAUDE.md maintenance item "wait for vLLM CUDA 13 support" is resolved:
vLLM supports CUDA 13, and v0.24.0 (June 2026) is current against the repo's
pinned v0.12.0. The update buys Model Runner V2, EAGLE-3.1 speculative
decoding, FP8 KV cache as production default, multi-tier KV offload, sleep
mode, and the maturing structured-output stack, several of which are
prerequisites for section 2.

Files: docker/vllm-gpu/Dockerfile.vllm-gpu, docker/vllm-gpu/entrypoint.sh.

Tasks:

- Bump VLLM_VERSION to the current release (v0.24.0 at time of writing;
  re-check). Bump the FlashInfer pins (flashinfer-cubin,
  flashinfer-jit-cache, currently 0.5.3) to versions matching the new vLLM;
  the vLLM release notes state the compatible FlashInfer version.
- Rebuild and validate on sm_120: known rough edges exist on RTX 5090 with
  CUDA 13 (vllm-project/vllm issue #37714); budget time for wheel or source
  build fallout.
- Review entrypoint defaults against the new version: confirm
  GPU_MEMORY_UTILIZATION, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, prefix
  caching, and chunked prefill flags are still correct names and sane values
  under Model Runner V2.
- Re-evaluate the commented-out "Blackwell EXPERIMENTAL" env block: delete
  entries the new version makes redundant, promote any that measure as wins.
- Update docs/inference/engines/vllm.md and parameters/vllm-parameters.md for
  the new version (overlaps with 1.3; do them together).

Acceptance: the vllm-gpu service builds, passes its healthcheck, serves a
model end to end through /v1/chat/completions, and a before/after throughput
benchmark is recorded (see 2.5 for format).

### 1.7 Unify the CUDA baseline

comfyui-flux builds from CUDA 12.8 with torch 2.7.0 cu128 while the LLM images
are on 13.0.2 cu130.

Tasks: attempt the bump to the same CUDA 13 base and current torch cu130; the
blockers to check are ComfyUI and custom-node torch compatibility. If blocked,
add a dated comment in the Dockerfile stating the blocker and the revisit
condition, so the exception is deliberate rather than drift.

Acceptance: either comfyui-flux is on the shared CUDA 13 base and generates an
image successfully, or the Dockerfile documents exactly why not.

### 1.8 Correctness checks

Small verifications, each potentially a real bug:

- llama-cpu passes `--flash-attn on` to a CPU-only build; verify the flag is
  honored (check server startup log) and remove it if it is a silent no-op.
- hugepages/src/config.h defines DEBUG_OUTPUT as 0 but the code gates on
  `#ifdef DEBUG_OUTPUT`, which is true even when defined as 0; change gates to
  `#if DEBUG_OUTPUT` and confirm debug prints disappear from a release build.
- open-webui points only at llama-gpu (8002); add the vllm-gpu endpoint (8003)
  to its configuration so the vLLM service is reachable from the UI, and
  remove the hardcoded qwen3-coder entry from WEBUI_IMPORTS if stale.
- llama-gpu sets `user: ${UID}:${GID}` in compose while vllm-gpu has the line
  commented out; make non-root consistent across both or document why vLLM
  cannot run non-root.
- cudnn9-cuda-13-0 (llama-gpu builder) and libnccl-dev (vllm builder) are
  commented out; confirm nothing in the current builds needs them and delete
  the dead lines, or re-enable with a comment.

Acceptance: each item closed with either a fix commit or a dated comment
explaining why current behavior is correct.

## Section 2: New inference-specific features

Every feature lands with a benchmark artifact in the 2.5 JSON format and a
short doc under docs/inference/ or docs/optimizations/experiments/. Proven
items are established practice as of July 2026 and are adopted, not
researched. Experimental items are studies with an explicit go/no-go outcome.

### Proven, worth adopting

#### 2.1 Speculative decoding with EAGLE-3

EAGLE-3/3.1 is the production speculative-decoding standard as of 2026:
merged in vLLM (v0.22+), built into llama-server, with pretrained draft heads
published for the Qwen and Llama families. Reported single-user speedups are
2-3x, which acts directly on this repo's headline tokens-per-second numbers.

Tasks:

- llama-gpu: expose llama-server speculative-decoding flags (draft model path,
  draft token count) as env-driven entrypoint options, off by default.
- vllm-gpu: expose the EAGLE speculative config (method, draft model,
  num_speculative_tokens) the same way.
- Download a matching pretrained EAGLE-3 head for one currently served model
  into /mnt/ai-data/models/; document which base models have usable heads.
- Benchmark single-stream and batched throughput with and without speculation
  on both engines; speculation can hurt batched throughput, so record both.

Acceptance: one served model demonstrates a measured single-stream speedup
with speculation on, results recorded in benchmark JSON, and the flags are
documented in the parameters docs.

Note: training draft heads for our own models is a training-tier optional
track; this task only consumes published heads.

#### 2.2 KV cache quantization

FP8 KV cache is the vLLM production default in 2026; llama.cpp supports q8_0
and q4 KV. q8/FP8 KV halves KV memory at negligible quality cost, which on a
32GB card converts directly into context length or batch headroom.

Tasks:

- vllm-gpu entrypoint: expose --kv-cache-dtype (default fp8 after validation)
  and the skip-layers hybrid option.
- llama-gpu and llama-cpu entrypoints: expose cache-type-k/cache-type-v,
  default q8_0 after validation.
- Validate quality: perplexity delta via the existing test_perplexity.sh at
  full precision versus quantized KV, per engine.
- Measure the headroom gain: maximum context or maximum concurrent sequences
  before and after, plus throughput at fixed workload.

Acceptance: quality delta and headroom gain recorded in benchmark JSON; new
defaults documented; a rollback env var exists (set dtype back to auto).

#### 2.3 NVFP4 and MXFP4 quantization

The Blackwell-native quantization story. NVFP4 gets real FP4 tensor-core
throughput on the RTX 5090; supported in vLLM (requires recent FlashInfer,
ModelOpt-produced NVFP4 checkpoints, FP8 KV cache) and merged into llama.cpp
in spring 2026. Native MXFP4 support landed for gpt-oss, which this repo
already serves. A rigorous Q4_K_M versus NVFP4 versus MXFP4 comparison on
this exact card is first-mover content and feeds the interpretability tier's
quantization-feature-survival research track.

Tasks:

- Depends on 1.1 (llama.cpp pin must include the NVFP4/MXFP4 kernels) and 1.6
  (vLLM NVFP4 path needs a current version).
- Obtain or produce an NVFP4 checkpoint for one mid-size model (ModelOpt for
  the vLLM path; llama-quantize for the GGUF path) and an MXFP4 gpt-oss GGUF.
- Benchmark the matrix: quality (perplexity plus a small eval set), single
  stream throughput, batched throughput, VRAM, against Q4_K_M and Q8_0
  baselines at matched context.
- Write the comparison up under docs/optimizations/experiments/ with the
  benchmark JSON alongside.

Acceptance: the comparison document exists with measured numbers for at least
one model across at least three quant formats on both applicable engines.

#### 2.4 Structured output

Guided decoding is table stakes in 2026 (xgrammar in vLLM, GBNF grammars in
llama.cpp) and the downstream tiers (judge scoring, organism auditing games)
want guaranteed-JSON outputs. This is mostly validation and documentation.

Tasks:

- Verify guided JSON output works through both engines' OpenAI-compatible
  endpoints (response_format / json_schema in vLLM, GBNF or json_schema in
  llama-server) at the pinned versions.
- Add a smoke script under scripts/utils/ that requests a schema-constrained
  completion from each service and validates the response parses against the
  schema.
- Document usage per engine in docs/inference/parameters/.

Acceptance: smoke script passes against all three services; docs updated.

#### 2.5 Benchmark JSON schema

The inference benchmarks currently write text and sqlite; the training tier's
design (D8) defines a JSON benchmark format and names it the model for
unifying the repo. Adopt one schema now so every section 2 feature lands with
comparable artifacts.

Tasks:

- Define the schema, aligned with docs/training/phase0-design.md D8:
  schema_version, timestamp, git SHA, hardware block (GPU, driver, CUDA, CPU
  governor), software block (engine, version, image tag), workload block
  (model, quant, context, concurrency), results array (tokens/s, latency
  p50/p95, VRAM, and for GPU runs NVML clocks/power/throttle reasons).
- Emit it from scripts/benchmark/server-bench.py and the benchmark_cpu.sh /
  benchmark_gpu.sh flows (a small Python writer both can call).
- Results publish under docs/inference/benchmarks/ (creating the directory
  the vllm.md dead link already points at).

Acceptance: one benchmark run per service produces a valid JSON artifact;
the schema is documented; the quant type appears in every result file so
mismatched comparisons are detectable.

### Experimental

#### 2.6 llama.cpp activation read path study

A confirmed ecosystem gap as of July 2026: control vectors give llama.cpp a
write path into the residual stream, but no mainstream serving stack exposes
activation capture; the only in-process hook is the eval-callback (cb_eval)
mechanism. A tensor tap in llama-server would be a first-of-its-kind
capability and is the exploratory item 7 on the interpretability build list.

Tasks:

- Scope as a study in docs/optimizations/experiments/ first, per the interp
  roadmap: feasibility of a cb_eval-based capture of selected tensors in
  llama-server, overhead measurement, and a minimal export format.
- Prototype against the pinned llama.cpp revision on llama-cpu (cheaper
  iteration) before touching the GPU service.
- Go/no-go: pursue integration only if capture overhead is acceptable and the
  interpretability tier's async monitor proves too limiting without it.

Acceptance: the study document exists with measured overhead and an explicit
go/no-go recommendation. No serving-path code merges before a go.

#### 2.7 Hugepage wrapper revalidation and extension

The 1GB-hugepage LD_PRELOAD wrapper remains a local differentiator that
upstream has not replicated, but the 26 percent claim predates the 2026
llama.cpp CUDA and CPU optimization waves and must be revalidated against the
newly pinned revision.

Tasks:

- Rerun the CPU benchmark suite with the wrapper on and off against the 1.1
  pinned llama.cpp; publish the delta in benchmark JSON. Update the README
  claim if the number moved.
- Fix the DEBUG_OUTPUT gating bug first (1.8) so measurements are clean.
- Extension study: the llama-gpu entrypoint uses --no-mmap, meaning host-side
  weight staging takes a different path; measure whether hugepage-backed
  staging helps GPU model load time or has no consumer, and record the answer
  either way.

Acceptance: refreshed wrapper benchmark published; extension study documented
with a keep/drop conclusion.

#### 2.8 MoE expert offload study

128GB of system RAM plus a 32GB GPU makes serving a large MoE with inactive
experts in system RAM plausible (ktransformers-style; llama.cpp's own
--n-cpu-moe / offload options). Assessment as of July 2026 is that this
hardware sits below the sweet spot of the dedicated hybrid stacks, so this is
framed as a bounded experiment, not a feature commitment.

Tasks:

- Pick one large open MoE that fits the RAM envelope quantized; measure
  llama.cpp expert-offload serving throughput and latency on this box.
- Compare against the largest dense model servable fully on-GPU at similar
  quality tier.
- Write up under docs/optimizations/experiments/ with a
  worth-it/not-worth-it conclusion.

Acceptance: the experiment document exists with measured numbers and a
conclusion; no permanent service is added unless the result is clearly
positive.

## Execution order

1. Reproducibility and truth: 1.1, 1.2, 1.3, 1.4 (independent, small, do
   first).
2. Platform updates: 1.5, 1.6, 1.7 (1.6 before any vLLM-dependent feature).
3. Correctness sweep: 1.8 (any time; DEBUG_OUTPUT fix before 2.7).
4. Measurement substrate: 2.5 (before all other section 2 items so every
   feature lands with a comparable artifact).
5. Proven features: 2.2, 2.1, 2.4, then 2.3 (needs 1.1 and 1.6).
6. Experiments: 2.7, 2.6, 2.8 in any order as time allows.

Each task is sized to be a single Claude Code session: state the task number,
point at this document, and the task's files, changes, and acceptance criteria
are self-contained.
