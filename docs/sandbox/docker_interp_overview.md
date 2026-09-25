# Interpretability workbench container implementation

Explanation of the interpretability workbench image (`docker/interp/Dockerfile.interp`) and
the `interp` service `docker-compose.interp.yaml` runs from it: PyTorch cu130 on sm_120 with
the interp stack, JupyterLab as the default command, and the interp CLI for harvests, probes,
SAEs, and steering. Unlike the training container it is a long-running service, and unlike
the inference services it is a shell on the machine, so its port never leaves localhost.

> Note: This document explains the actual implementation in `docker/interp/Dockerfile.interp`,
> `docker/interp/entrypoint.sh`, and `docker-compose.interp.yaml`. Those files are the source
> of truth; this document follows them.

## Table of Contents

<!-- toc:start -->

1. [Design](#design)
2. [Dependency source of truth](#dependency-source-of-truth)
3. [Image build](#image-build)
4. [Writable-state layout](#writable-state-layout)
5. [Compose integration](#compose-integration)
6. [Port and access](#port-and-access)
7. [Interface](#interface)
8. [Concurrency policy](#concurrency-policy)
9. [Reference implementation](#reference-implementation)

<!-- toc:end -->

## Design

One reproducible environment for every interpretability phase: the stack is imported and
CUDA-verified at build time through `python -m interp smoke --build`, so a dependency that
would regress the sm_120 build fails the image build rather than a notebook. The service is
started explicitly through `make interp-up` or `make interp-shell`, never by `make up` or on
boot (`restart: "no"`), because an interp session and the GPU inference services must not run
at once: one GPU consumer at a time, per `AGENTS.md`. The concurrency check itself is
`scripts/interp/preflight.sh`, which the make targets run before anything starts; the compose
file only documents the rule structurally by being separate.

The base is the CUDA 13.0.2 **runtime** image, not devel. Nothing in the image needs nvcc,
and Triton ships its own ptxas and CUDA headers. The Dockerfile's rule for the day that
changes: switch this base to devel with a dated comment, do not add a second image.

## Dependency source of truth

The environment is `uv.lock`, core dependencies plus the `interp` group, installed with
`uv sync --active --frozen --no-install-project --no-default-groups --group interp` into
`/opt/venv`. No package version is written in the Dockerfile. torch is pinned to the cu130
index in `pyproject.toml`'s `[tool.uv.sources]`, so an interp package that would downgrade
torch fails `uv lock` loudly instead of silently regressing the build; the roadmap calls this
the anti-downgrade rule, enforced structurally. Notebook-support packages are in the same
group for the same reason, and a notebook never pip-installs inside a running container.

## Image build

Build arguments:

```dockerfile
ARG CUDA_VERSION=13.0.2
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.11.28
```

The build:

1. Copies the pinned uv binary from `ghcr.io/astral-sh/uv:0.11.28`, the same source the
   training image uses; no network installer runs.
2. Installs the runtime packages on `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04`: curl,
   ca-certificates, `libgomp1`, Python 3.12 with its `-dev` headers and `-venv`, and `gcc`.
   Every dependency installs as a prebuilt wheel, so nothing compiles at build time, but
   Triton compiles at **run** time: torch 2.13 routes some eager ops through Triton kernels,
   and Triton's first CUDA call builds its driver shim from source as a CPython extension
   that includes `Python.h` and is compiled with the `gcc` on `PATH`. Without both, the first
   GPU forward pass dies with "Failed to find C compiler" after the model has already loaded,
   which is how a Gemma 4 harvest failed in 2026-08. No CUDA dev package is needed:
   `libcuda.so.1` comes from the driver the NVIDIA runtime injects and `cuda.h` from Triton's
   bundled include directory.
3. Creates `/opt/venv` and syncs the lockfile into it, with a uv cache mount so a rebuild
   re-downloads nothing.
4. Sets the GPU environment: `TORCH_CUDA_ARCH_LIST="12.0"` for Blackwell,
   `CUDA_MODULE_LOADING=LAZY`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, and
   `NVIDIA_REQUIRE_CUDA="cuda>=13.0"`.
5. Creates the non-root `appuser`, points every writable path at tmpfs or the data mount
   (next section), and copies `common/`, `interp/`, `training/`, and `scripts/interp/` to
   `/workspace` for the build-time smoke test. At run time the live repository is
   bind-mounted over `/workspace` and this copy is shadowed.
6. Runs `python -m interp smoke --build`. The CUDA and network steps degrade to `[SKIP]` on a
   GPU-less build host; `make interp-smoke` enforces them on the workstation.

`ENTRYPOINT ["/app/entrypoint.sh"]` with `CMD ["jupyter-lab"]`.

## Writable-state layout

The rootfs is read-only at run time, so the Dockerfile pins every path a session writes:

| Variable | Value | Why there |
|----------|-------|-----------|
| `HOME` | `/app` | Owned by `appuser`; nothing persistent lands in it |
| `HF_HOME` | `/mnt/ai-data/interp/hf-cache` | On NVMe across runs, unlike the training container's disposable cache, so a gated model downloads exactly once |
| `JUPYTER_CONFIG_DIR`, `JUPYTER_DATA_DIR`, `JUPYTER_RUNTIME_DIR` | under `/tmp/jupyter` | tmpfs |
| `IPYTHONDIR`, `MPLCONFIGDIR`, `XDG_CACHE_HOME` | `/tmp/ipython`, `/tmp/mpl`, `/tmp/cache` | tmpfs |
| `TRITON_CACHE_DIR` | `/tmp/triton` | JIT cache; the tmpfs carries `exec` for it |
| `WANDB_DATA_DIR`, `WANDB_CACHE_DIR`, `WANDB_CONFIG_DIR` | under `/tmp/wandb` | wandb otherwise stages under `$HOME/.local/share`, on the read-only rootfs; SAELens uploads the SAE checkpoint as an artifact at the end of a training run, so an unredirected staging directory fails the run after training has completed |

The entrypoint creates the Jupyter, IPython, matplotlib, and cache directories on every
start, and bootstraps the storage tree under `/mnt/ai-data/interp` (`activations`, `saes`,
`probes`, `vectors`, `hf-cache`, `logs`, `monitor`) when the data mount is present;
`make interp-dirs` does the same host-side.

## Compose integration

The `interp` service applies the hardening the training container uses: `init: true` for
tini as PID 1, `user: "${UID:-1000}:${GID:-1000}"` so bind-mount writes are host-owned,
`no-new-privileges`, `cap_drop: ALL`, and `read_only: true`. It reserves one NVIDIA device
with the `gpu`, `compute`, and `utility` capabilities and pins it with
`NVIDIA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=0`. `shm_size: "8g"` covers dataloader
workers, and `/tmp` is an 8G tmpfs mounted with `exec`, which the Triton and CUDA JIT caches
require. No memory, CPU, or pid limit is set, and there is no healthcheck.

`JUPYTER_TOKEN` is passed through from the environment; `make interp-up` generates one per
launch. `.env.interp` is an optional `env_file` carrying `HF_TOKEN` for gated downloads, so
the image builds and ungated work runs without secrets.

| Mount | Mode | Purpose |
|-------|------|---------|
| `.` | read-write | `/workspace`: notebooks and code live in git |
| `/mnt/ai-data/interp` | read-write | Heavy artifacts and `HF_HOME`; weights land on NVMe once, never in git |
| `/mnt/ai-data/models` | read-only | `/app/models`, parity with the inference services |
| `/mnt/ai-data/training` | read-write | The shared D9 run lock, which harvest jobs take as training runs do, and organism checkpoints |
| `/mnt/ai-data/logs` | read-write | Structured observability records, identity-mounted so `OBS_LOG_DIR` resolves to the same path inside and outside; without it the file handler in `common/obs/core.py` fails its mkdir against the read-only rootfs and silently drops every record |

The service sits on its own bridge, `interp-network`, and does not join `ai-network`: the
workbench never runs alongside the GPU inference stack.

## Port and access

The compose file publishes `127.0.0.1:8888:8888` and nothing else. JupyterLab is a shell on
the machine, so the port is never bound beyond localhost; remote access goes through an SSH
tunnel. Inside the container the entrypoint starts JupyterLab on `0.0.0.0:8888` with
`--ServerApp.root_dir=/workspace` and the token as `--IdentityProvider.token`. Host exposure
is limited by the compose mapping, never by arguments in the entrypoint, and the entrypoint
refuses to start the default command at all when `JUPYTER_TOKEN` is unset, rather than start
an unauthenticated server. Any other command is passed straight to `exec`.

## Interface

| Target | Action |
|--------|--------|
| `make interp-build` | Build the image; the build fails if any smoke-test step regresses |
| `make interp-up` | Preflight, generate a token, start JupyterLab, and print the tokened URL |
| `make interp-shell` | Preflight, then an interactive shell with the GPU attached |
| `make interp-smoke` | The six-step smoke test on the workstation, enforcing CUDA sm_120 |
| `make interp-down` | Stop the workbench |
| `make interp-dirs` | Create the `/mnt/ai-data/interp` storage tree on the host |
| `make interp-runs` | List recent runs from the observability catalog; runs on the host |

The `harvest-*`, `probe-*`, `sae-*`, `steer-*`, and `organism-*` targets in `make/interp.mk`
run the interp CLI in the same service through `docker compose run --rm interp`;
[../interp/README.md](../interp/README.md) documents them.

## Concurrency policy

One GPU consumer at a time. Stop the inference stack (`make down`) before `make interp-up`;
the preflight refuses to start the workbench while an inference service holds the GPU.
`llama-cpu` may stay up on its dedicated cores, and the preflight says so.

## Reference implementation

- `docker/interp/Dockerfile.interp` — runtime-base image, lockfile sync, writable-state
  layout, build-time smoke test.
- `docker/interp/entrypoint.sh` — storage bootstrap, environment echo, token-checked
  JupyterLab or `exec`.
- `docker-compose.interp.yaml` — the `interp` service, its GPU reservation, tmpfs, mounts,
  and loopback port.
- `scripts/interp/preflight.sh` — the GPU concurrency check the make targets run first.
- `make/interp.mk` — the `interp-*` targets and the CLI wrappers.
