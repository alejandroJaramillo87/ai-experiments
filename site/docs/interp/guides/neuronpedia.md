---
title: Neuronpedia platform
---

# Neuronpedia platform

The platform is a self-hosted Neuronpedia: a web application, a database, and GPU services
for feature inference, automatic interpretation, and attribution graphs, running from a
pinned upstream checkout plus the repository's compose override. The repository holds no
upstream logic, only the pin, the override, and vendored patches under `docker/interp/`.

## Bring-up

```bash
make np-clone
make np-build
make np-up
make np-smoke
```

`np-clone` clones or updates the checkout to the pinned SHA and discards local edits;
`np-build` builds the images; `np-up` starts the web application, the database, and the
database initialization, all on the CPU; `np-smoke` checks that every port binds loopback
only and that the web application and database answer. The GPU services start separately,
each behind the concurrency preflight:

```bash
make np-inference-up
make np-graph-up
make np-autointerp-up
make np-down
```

`np-down` stops everything; the database and caches persist in their bind mounts.
`make np-status` shows the platform's containers and ports. `np-clone` requires the storage
tree from `make interp-dirs`, and `np-build` and every `-up` target first merge the
repository's environment deltas and the secrets in `.env.interp` into the checkout.

| Variable | Read by | Effect | Default |
|----------|---------|--------|---------|
| `MODEL_SOURCESET` | `np-inference-up` | The inference preset; a name with no matching config file in the checkout is refused with `No such config` | `gemma-2-2b-it.gemmascope-res-16k` |
| `NP_GPU` | `np-autointerp-up` | When `1`, attaches the GPU for the embedding scorer, behind the GPU preflight | unset |
| `NP_SRC` | Every target | The pinned checkout | `/mnt/ai-data/interp/neuronpedia/src` |
| `AI_NETWORK_NAME` | Every target | The compose network the platform joins so autointerp can reach the inference services | `ai-experiments_ai-network` |

## Ports

Every port binds `127.0.0.1` only, and the smoke check fails if one is reachable beyond
loopback. The web application is on 3001 rather than upstream's 3000, which `open-webui`
holds; the database on 5432; inference, autointerp, and the graph server on 5002, 5003, and
5004.

## Concurrency

One GPU consumer at a time. `np-up` is CPU-only and never preflighted; `np-inference-up` and
`np-graph-up` refuse while `llama-gpu` or `vllm-gpu` is running. Autointerp is the one
service that is serving-dependent rather than GPU-holding: it batches feature explanations
against `vllm-gpu`, so that service must be up and autointerp runs on the CPU. Its own
preflight has three refusals, each with a `Solution:` and a `Command:` line: `vllm-gpu` not
healthy on its port, `llama-gpu` running beside it, and a platform GPU service (inference or
graph) still up, which `make np-down && make np-up` clears while keeping the web application
and database.

## Trust model

Localhost mode disables sign-in and authenticates the services to each other with static
shared secrets. That is acceptable only because every port binds loopback; the secrets are
not to be rotated, and the loopback check is what keeps the arrangement sound. Remote access
goes through an SSH tunnel, as for the workbench.

## Where it lives

| Path | Contents |
|------|----------|
| `/mnt/ai-data/interp/neuronpedia/src` | The pinned upstream checkout |
| `/mnt/ai-data/interp/neuronpedia/postgres` | The database |
| `/mnt/ai-data/interp/hf-cache` | The Hugging Face cache the platform containers share with the workbench |
