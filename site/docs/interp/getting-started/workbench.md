---
title: Workbench
---

# Workbench

The workbench is the container that holds the tier's GPU stack: PyTorch for CUDA 13 on the
RTX 5090, TransformerLens, nnsight, and SAELens, with JupyterLab for exploratory work and
the `python -m interp` verbs for batch jobs. It lives in its own compose file,
`docker-compose.interp.yaml`, so that `make up` never starts it: an interactive session and
the GPU inference services must not run at once.

## Build

```bash
make interp-build
```

The build runs the smoke test in its build-time mode and fails on a regression, so an image
that builds is an image whose stack imports. `docker compose run` builds only when the image
is missing and does not notice a changed `pyproject.toml`, lockfile, or Dockerfile; after any
such change, run `uv lock` and then `make interp-build`, or the verbs run on a stale image.
The Dockerfile syncs with a frozen lock and fails on a stale one, which is why the lock comes
first.

## Start, use, stop

```bash
make interp-dirs
make interp-up
```

`interp-dirs` creates the `/mnt/ai-data/interp` tree once. `interp-up` runs the GPU preflight,
starts JupyterLab bound to `127.0.0.1:8888` with a token generated for that launch, and
prints the tokened URL. The port binds loopback only, because a notebook is a shell on the
machine; remote access goes through an SSH tunnel.

```bash
make interp-shell
make interp-down
```

`interp-shell` opens an interactive shell in a fresh container with the GPU attached, after
the same preflight, and removes the container on exit. `interp-down` stops the JupyterLab
service. The `make` targets for the verbs each start their own one-shot container, so the
JupyterLab service need not be running for a batch job.

## What the container is

The container runs as the host user, taken from `UID` and `GID` in the environment with a
default of 1000, so bind-mount writes are owned correctly; a mismatch shows up as a write
failure under the data tree. It drops every capability, refuses new privileges, and mounts
its root filesystem read-only; writable state goes to the data tree and to tmpfs. It never
restarts on boot. Started through `docker compose` directly rather than `make interp-up`,
with no `JUPYTER_TOKEN` in the environment, the entrypoint refuses with
`[ERROR] JUPYTER_TOKEN is not set` rather than start an unauthenticated server. The mounts and their spellings
are on [Environment](environment.md).

## The preflight

Every target that attaches the GPU runs `scripts/interp/preflight.sh` first. It asks the
inference compose file which services are running and refuses while `llama-gpu` or
`vllm-gpu` is up:

```text
[ERROR] GPU service running: llama-gpu (one GPU consumer at a time)
  Solution: stop it first
  Command: docker compose -f docker-compose.inference.yaml stop llama-gpu  (or 'make down' for everything)
```

A compose invocation that fails is refused as well, with its error as the detail, rather than
read as nothing running. `llama-cpu` is permitted, on its dedicated cores. The check lives on
the host and runs before anything starts, so refusal costs nothing.
