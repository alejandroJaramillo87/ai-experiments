# Rust harness container implementation

Explanation of the rust harness image (`docker/rust-harness/Dockerfile.rust-harness`) and the
two services `docker-compose.rust-harness.yaml` runs from it: the interactive `harness` tool
and the `harness-serve` daemon that exposes the OpenAI-shaped endpoint interp calls. The
container exists for build reproducibility, not containment: the same pinned toolchain and
locked dependency graph produce the same binary on any machine, which is what runbook entry
RH.12 verifies.

> Note: This document explains the actual implementation in
> `docker/rust-harness/Dockerfile.rust-harness`, `docker/rust-harness/entrypoint.serve.sh`,
> and `docker-compose.rust-harness.yaml`. Those files are the source of truth; this document
> follows them.

## Table of Contents

<!-- toc:start -->

1. [Design](#design)
2. [Image build](#image-build)
3. [Compose integration](#compose-integration)
   1. [The interactive service](#the-interactive-service)
   2. [The serving service](#the-serving-service)
   3. [Healthcheck](#healthcheck)
4. [The serving entrypoint](#the-serving-entrypoint)
5. [Interface](#interface)
6. [Reference implementation](#reference-implementation)

<!-- toc:end -->

## Design

The harness is a single binary: a coding agent with the CPU inference runtime compiled in.
The image builds that binary once from the workspace under `rust-harness/` and ships it on a
minimal Debian base with no toolchain, no Python, and no package manager state. Two compose
services run the one image:

| Service | Container | Runs | Lifetime |
|---------|-----------|------|----------|
| `harness` | `rust-harness` | the bare binary, arguments on the command line | one `run --rm` per task or chat |
| `harness-serve` | `rust-harness-serve` | `/app/entrypoint.serve.sh`, arguments from the environment | daemon, `restart: unless-stopped` |

Neither is started by `make up`. The serving service is a GPU-free but memory-hungry CPU
consumer that loads a checkpoint of roughly 14 GiB and serves one generation at a time, so
it is not part of the production inference stack. The compose file is separate for that
reason, as the training, interp, and agent compose files are.

`--model` takes a checkpoint directory holding `config.json` and `model.safetensors`, not a
weights file. GGUF is llama.cpp's format and nothing under `rust-harness/` reads it.

## Image build

Two stages, both pinned by digest:

```dockerfile
ARG BUILDER_IMAGE=rust:1.94-slim-trixie@sha256:cf09adf8c3ebaba10779e5c23ff7fe4df4cccdab8a91f199b0c142c53fef3e1a
ARG BASE_IMAGE=debian:trixie-slim@sha256:28de0877c2189802884ccd20f15ee41c203573bd87bb6b883f5f46362d24c5c2
```

1. **Builder.** The `rust:1.94-slim-trixie` image copies `rust-harness/` to `/build` and
   runs `cargo build --release -p harness --locked`. The `--locked` flag refuses to build
   when `Cargo.lock` is stale, which is the reproducibility anchor: the dependency graph the
   binary is built from is the one committed, or the build fails.
2. **Runtime.** The `debian:trixie-slim` image is pinned to the same digest as the llama-cpu
   runtime so glibc stays in step across services. It creates the non-root `appuser`, copies
   the binary to `/app/harness` and the serving entrypoint to `/app/entrypoint.serve.sh`,
   and sets `WORKDIR /work`.

The Dockerfile declares no ports, volumes, or environment variables. Those belong to the
compose file, because the two services differ in all three. `ENTRYPOINT ["/app/harness"]`
is the bare binary; the serving service overrides it.

Re-pinning either image is a documented two-step: `docker buildx imagetools inspect <tag>`
to resolve the digest, then set the `ARG` to `<tag>@sha256:<digest>`.

## Compose integration

Both services share the inference stack's hardening: `no-new-privileges`, `cap_drop: ALL`,
`read_only: true`, and `working_dir: /work`. Both run as `${UID:-1000}:${GID:-1000}` rather
than the image's `appuser`, because the harness writes `session.jsonl` and
`probe-flags.jsonl` into `/work` on every run, and `/work` is a host-owned bind mount the
image's system user could not write.

Neither service reserves a GPU or sets a memory or CPU limit.

### The interactive service

`harness` takes `network_mode: none`: the binary is fully local and reads models from the
same read-only mount the inference services use, so it needs no network at all. It keeps
`stdin_open: true` for chat mode and `tty: false` so stdout stays clean when piped;
`docker compose run` adds a TTY when the session is interactive.

| Mount | Mode | Purpose |
|-------|------|---------|
| `/mnt/ai-data/models/` | read-only | Checkpoint directories under `/app/models` |
| `./sessions/rust-harness` | read-write | `/work`: session transcripts and probe flags |

### The serving service

`harness-serve` cannot use `network_mode: none`, because a published port needs a network.
It publishes `127.0.0.1:8005:8005`, loopback only, which is what keeps the endpoint
host-local. Inside the container the server binds `0.0.0.0`: the harness binds loopback by
default, which is right on a workstation and wrong in a container, where a published port
forwards to the container's external interface and a server on the container's own
`127.0.0.1` accepts nothing.

| Variable | Default | Purpose |
|----------|---------|---------|
| `HARNESS_MODEL` | `/app/models/hf/gemma-4-E4B-it` | Checkpoint directory, container-side |
| `HARNESS_PORT` | `8005` | Listening port |
| `HARNESS_BIND` | `0.0.0.0` | Bind address; see above |
| `HARNESS_TEXT_ONLY` | `true` | Appends `--text-only`; Gemma 4 checkpoints carry vision and audio towers the CPU backend refuses rather than half-loads |
| `HARNESS_PROBE` | empty | Probe scored in-band at every position of its layer |
| `HARNESS_INTERP_LOG` | `/app/interp-logs` | Where the per-request record lands; `make probe-monitor` aggregates it |
| `CONTROL_VECTOR` | empty | Steering vector, set by `make steer-up` |
| `CONTROL_VECTOR_SCALE` | empty | Steering scale |
| `CONTROL_VECTOR_LAYER_RANGE` | empty | One argument holding two numbers, as `interp/steering/wiring.py` emits it |

An empty variable omits its flag. `HARNESS_PROBE` without `HARNESS_INTERP_LOG` computes
scores and discards them, and the entrypoint warns so.

| Mount | Mode | Purpose |
|-------|------|---------|
| `/mnt/ai-data/models/` | read-only | Checkpoint directories under `/app/models` |
| `/mnt/ai-data/interp/vectors` | read-only | Control vectors under `/app/vectors` |
| `/mnt/ai-data/interp/probes` | read-only | Exported probes under `/app/probes` |
| `/mnt/ai-data/interp/logs` | read-write | `/app/interp-logs`: one JSONL record per completed request |
| `./sessions/rust-harness` | read-write | `/work`, as for the interactive service |

Run-level observability records are not collected from either service: the harness writes
them under `OBS_LOG_DIR` (default `/mnt/ai-data/logs`), which this compose file does not
mount, so the run log degrades to its silent no-op. Container runs are for build
reproducibility; the observability contract is verified on the host (RH.13).

### Healthcheck

The probe is bash, not curl, because the runtime image is `debian-slim` and carries neither
curl nor wget; installing one for a healthcheck would be a package to pin and patch. The
check opens `/dev/tcp/127.0.0.1/8005`, issues a real `GET /v1/health`, and reads the status
line for `200`, so it proves the endpoint answers and not merely that the port accepts a
connection. It is written as `CMD` with an explicit `bash -c` rather than `CMD-SHELL`,
because `/bin/sh` is dash here and dash has no `/dev/tcp`.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `interval` | 30 s | |
| `timeout` | 10 s | |
| `retries` | 3 | |
| `start_period` | 600 s | Loading the checkpoint from disk dominates startup |

The harness answers `/v1/health` as soon as it is listening, and the handler never takes the
generation lock, so the probe stays fast while a minutes-long reply is in flight.

## The serving entrypoint

`entrypoint.serve.sh` builds the harness argument list from the environment, because docker
compose has no way to pass a flag conditionally. It refuses to start without
`HARNESS_MODEL`, refuses a `CONTROL_VECTOR` or `HARNESS_PROBE` path that is not a file, and
prints the model, the endpoint, the steering state, and the log destination before
`exec /app/harness`. The four steering and observation variable names are the ones the
retired llama.cpp entrypoints read, kept because `interp/steering/wiring.py` emits exactly
those names and `make steer-up` passes them through.

The entrypoint also warns that the CPU backend decodes at single-digit tokens per second on
a 4B-class checkpoint, so a caller must pass a timeout matched to `max_tokens` rather than
rely on a client default; `common/serving.py`'s harness entry carries that timeout.

## Interface

| Target | Action |
|--------|--------|
| `make harness-build` | Build the image with the pinned toolchain and `cargo --locked` |
| `make harness-run` | One-shot task in the container (`ARGS='--model /app/models/hf/<model> <task>'`) |
| `make harness-chat` | Interactive chat REPL in the container (`ARGS='--model /app/models/hf/<model>'`) |
| `make harness-serve` | Start `harness-serve` on 127.0.0.1:8005 (`HARNESS_MODEL=/app/models/hf/<model>`) |
| `make harness-serve-down` | Stop the serving harness |
| `make harness-runs` | List recent runs from the observability catalog; runs on the host, since the image carries no Python |
| `make harness-clean` | Remove the workspace's run outputs and the `./sessions/rust-harness` transcripts |
| `make steer-up` | Recreate `harness-serve` with the control-vector variables set |
| `make steer-down` | Recreate it unsteered |

`harness-run`, `harness-chat`, and `harness-serve` pre-create `sessions/rust-harness` so the
bind mount is user-owned before the container writes into it.

## Reference implementation

- `docker/rust-harness/Dockerfile.rust-harness` — two-stage build: pinned `rust` builder,
  pinned `debian-slim` runtime, one binary.
- `docker/rust-harness/entrypoint.serve.sh` — the serving service's argument builder.
- `docker-compose.rust-harness.yaml` — the `harness` and `harness-serve` services, their
  mounts, the loopback port, and the `/dev/tcp` healthcheck.
- `make/harness.mk` — the `harness-*` targets; `make/interp.mk` holds `steer-up` and
  `steer-down`.
- [../rust-harness/README.md](../rust-harness/README.md) — the harness itself, and
  [../verification/rust-harness/rust-harness.md](../verification/rust-harness/rust-harness.md)
  for RH.12 and RH.13.
