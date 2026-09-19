# Training container implementation

Explanation of the Phase 0 training container (`docker/training/Dockerfile.training`), the
reproducible GPU environment where the training tier runs. It builds PyTorch cu130 on the RTX
5090 Blackwell architecture (sm_120) with the full fine-tuning and pretraining stack, and
verifies every CUDA-compiled path at build time through a smoke test.

> Note: This document explains the actual implementation in `docker/training/Dockerfile.training`,
> `docker/training/entrypoint.sh`, `training/commands/smoke_test.py`, and
> `docker-compose.training.yaml`. Those files are the source of truth; this document follows
> them.

## Table of Contents

- [Design](#design)
- [Dependency source of truth](#dependency-source-of-truth)
- [Image build](#image-build)
- [Smoke test](#smoke-test)
- [Compose integration](#compose-integration)
- [Interface](#interface)
- [Concurrency policy](#concurrency-policy)
- [Reference implementation](#reference-implementation)

## Design

The training container is a batch and interactive job runner, not a long-running server. It has
no published port and no healthcheck. It is invoked per job through
`docker compose -f docker-compose.training.yaml run --rm training <command>`, wrapped by the
`train-*` Make targets.

It builds on the CUDA 13.0.2 devel base (`nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04`), the same
base family as the llama-gpu image. The devel toolchain is required to pre-build DeepSpeed's
CPUAdam op with Zen 5 compiler flags at image-build time.

## Dependency source of truth

The container installs the pinned environment from `uv.lock` (the core dependencies plus the
`training` dependency group), not from hand-pinned Dockerfile lines. uv manages the cu130 torch
index through `[tool.uv.sources]` and builds the CUDA-compiled packages through
`[tool.uv.extra-build-dependencies]`, whose `match-runtime = true` entries build them
against the locked torch rather than the newest one on PyPI (see
`docs/os/python/uv-dependencies.md`). The core
dependencies already provide torch, torchvision, transformers, datasets, accelerate, peft,
deepspeed, bitsandbytes, sentencepiece, wandb, mlflow, and tensorboard; the `training` group adds
only the libraries not already present: `trl`, `flash-attn`,
`hydra-core`, `pynvml`, `xxhash`, and `zstandard`.

## Image build

Build arguments:

```dockerfile
ARG CUDA_VERSION=13.0.2
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.11.28
```

The build:

1. Copies the pinned uv binary from `ghcr.io/astral-sh/uv:0.11.28` (no network installer runs).
2. Installs the build toolchain (gcc, g++, git, nvcc, CUDA dev headers) on the devel base.
3. Creates a `/opt/venv` virtual environment and syncs `uv.lock` into it with
   `uv sync --frozen --no-install-project --group training`.
4. Pre-builds the DeepSpeed CPUAdam op with `DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1` and
   `CFLAGS="-march=native"`. The image gcc (13, on Ubuntu 24.04) has no `znver5` target, so
   `native` is what resolves to Zen 5; the image is therefore specific to the CPU it was
   built on. This front-loads the AVX-512 miscompilation risk (upstream DeepSpeed issue 7556)
   to build time, where the smoke test catches it.
5. Creates the non-root `appuser`, copies the `common/` and `training/` packages, and runs `python -m training smoke --build`
   so that a regression fails the build.

`TORCH_CUDA_ARCH_LIST="12.0"` targets the RTX 5090 Blackwell sm_120 architecture, matching the
inference GPU images.

## Smoke test

`training/commands/smoke_test.py` verifies the stack imports and that the CUDA-compiled paths run.
It has two modes: `--build` (build-time; CUDA asserts degrade to `[SKIP]` on a GPU-less build
host) and the default (`make train-smoke` on the workstation; CUDA capability `(12, 0)` is
enforced). The six steps:

1. Import torch; assert CUDA availability and device capability `(12, 0)`.
2. Import transformers, trl, peft, datasets, accelerate, hydra,
   safetensors, and pynvml.
3. flash-attn: run a `flash_attn_func` forward on random tensors.
4. bitsandbytes: quantize a small linear layer to 4-bit and run a forward.
5. deepspeed: load and step `DeepSpeedCPUAdam` (proves the pre-built Zen 5 op loads and runs).
6. Write a versions manifest (packages, CUDA, driver, GPU name) to stdout and, on the
   workstation, to `/mnt/ai-data/training/runs/_smoke/versions.json`.

Every branch of the command (build-mode skips, each failure name, the manifest) is covered
in the sandbox by `/tests/training/test_smoke_cmd.py`, which stubs the CUDA packages; the
entrypoint's environment echo and `TRAIN_LD_PRELOAD` slot by `/tests/training/test_entrypoint.py`.

## Compose integration

`docker-compose.training.yaml` is a separate compose file, mirroring
`docker-compose.agents.yaml`, so `make up` never starts it. The `training` service applies the
inference stack's hardening (`no-new-privileges`, `cap_drop: ALL`, `read_only: true`,
non-root `user`), reserves one NVIDIA GPU, sets `shm_size: 16g` for dataloader workers, and
mounts a tmpfs `/tmp` with the `exec` flag that the Triton and CUDA JIT caches require. Two
further tmpfs mounts back it: `/app/cache` at 2G, deliberately small because this service
never resolves a hub id — if it fills, something is reaching the hub that should not be —
and `/staging`, sized by `TRAIN_STAGING_GB` (default 32G) to match `staging.max_gb` for the
D9 RAM-disk shard staging.

Volumes:

| Mount | Mode | Purpose |
|-------|------|---------|
| `/mnt/ai-data/models/` | read-only | Base model weights, as the inference services mount them |
| `/mnt/ai-data/models/trained` | read-write | Narrow window for exported checkpoints; the rest of `/app/models` stays read-only via the mount above |
| `/mnt/ai-data/training` | read-write | Datasets, shards, checkpoints, and run outputs |
| `/mnt/ai-data/logs` | read-write | Structured observability records, identity-mounted |
| `./scripts` | read-only | Training scripts and recipes, mounted live for iteration |
| `./training` | read-only | The training package itself, mounted live for iteration |
| `./common` | read-only | Shared helpers imported by `training/cli.py` at module scope |

## Network split: one door to the hub

The compose file defines three services on two networks. The split exists so that a long
training run cannot stall part-way through on a download.

| Service | Network | Egress | Purpose |
|---------|---------|--------|---------|
| `prepare` | `training-egress` | yes | Downloads models; builds datasets, shards, tokenizers, and the contamination index |
| `training` | `training-network` | **no** | Training, calibration, and evaluation |
| `export` | `training-network` | **no** | Adapter merge into the HF checkpoint directory the harness serves |

`training-network` sets `internal: true`, so the training and export containers have no route
off the host. This is enforcement by topology rather than by argument: it holds regardless of
environment variables, command-line flags, or what a recipe puts in `model.hf.id`. The image
also sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`, which the
`prepare` service overrides back to `0`.

Environment variables alone would not be sufficient. `lm_eval` loads its own model and
downloads its own task datasets, and `datasets.load_dataset` is called from four places; none
of those accept an offline argument this repository controls.

`prepare` binds `/mnt/ai-data/cache/huggingface` as a persistent `HF_HOME`, so re-downloading
an already-fetched revision is a cache hit. It is also the only service that mounts
`/mnt/ai-data/models/hf` read-write. The `training` service keeps a deliberately small tmpfs
at the same path, which is expected to stay empty — anything landing there means a code path
is still resolving a hub identifier.

## Interface

Staging targets run in `prepare` and may reach the network:

| Target | Action |
|--------|--------|
| `make train-download` | Stage a pinned model into `/mnt/ai-data/models/hf` |
| `make prepare-corpus` | Filter, dedup, decontaminate, and shard a corpus |
| `make prepare-data` | Tokenize a dataset into shards |
| `make prepare-sft` | Prepare masked SFT examples |
| `make build-contamination` | Build the benchmark decontamination index |
| `make train-tokenizer` | Train a byte-level BPE tokenizer |

Everything else runs in `training` or `export`, with no network:

| Target | Action |
|--------|--------|
| `make train-build` | Build the image; the build fails if any smoke-test step fails |
| `make train-shell` | Interactive shell in the container, GPU and training root attached |
| `make train-smoke` | Run the six-step smoke test on the workstation |
| `make train-preflight` | Prove every asset a recipe references is already local |

## Validate compared with preflight

Two commands check a recipe, and the difference matters:

- `make train-validate` checks schema and cross-field invariants. Its path checks are
  skip-when-absent by explicit convention, so it passes on a machine with no data at all.
  It answers "is this recipe coherent".
- `make train-preflight` requires every asset the recipe references to be present and
  verified, and reports every missing one rather than stopping at the first. It answers
  "will this run start". Run it immediately before a long job.

Preflight hoists checks that otherwise happen too late: shard integrity is verified inside
`ShardedTokenDataset.__init__` today, which is after the run has started and the GPU is
committed.

## Concurrency policy

Training and inference must not run at the same time: one GPU consumer at a time (see
`AGENTS.md`). The separate compose file documents this structurally. Stop the inference stack
(`make down`) before running a training job.

## Reference implementation

- `docker/training/Dockerfile.training` — multi-stage-capable devel-base training image.
- `docker/training/entrypoint.sh` — thin entrypoint: environment echo, optional LD_PRELOAD hook,
  `exec`.
- `training/commands/smoke_test.py` — six-step stack verification and versions manifest.
- `docker-compose.training.yaml` — separate-file service definition with GPU reservation and the
  writable training root.
- `pyproject.toml` — the `training` dependency group and the `flash-attn` build dependency.
