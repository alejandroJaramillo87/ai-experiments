# logtriage stack overview

The four services in `docker-compose.logtriage.yaml`: MongoDB 8.2 with MongoDB Search, and two
llama.cpp servers. Started explicitly with `make logtriage-up`, never on boot.

> This document covers the stack's place in *this* repository — the compose topology, the
> images, and how it differs from the inference stack. The running of it, its configuration, and
> its traps are owned by [../../logtriage/docs/operations/stack.md](../../logtriage/docs/operations/stack.md).

## Table of Contents

- [Why a separate compose file](#why-a-separate-compose-file)
- [The four services](#the-four-services)
- [What is built and what is pulled](#what-is-built-and-what-is-pulled)
- [Ports](#ports)
- [Entry points](#entry-points)

## Why a separate compose file

This repository separates stacks by file rather than by compose profile, and this is the sixth:
inference, training, interp, rust-harness, agents, and now logtriage. The reason is the one the
root `Makefile` records — every invocation must pass `-f`, so a stack cannot be started
alongside another by accident.

It matters more here than usual. Two of these services reserve the GPU, and the inference stack's
`llama-gpu` and `vllm-gpu` reserve the same card. Keeping them in separate files is what stops a
bare `docker compose up` from contending for VRAM.

## The four services

| Service | What it is |
|---------|------------|
| `logtriage-mongo` | MongoDB 8.2 Community, as a single-member replica set with keyfile auth |
| `logtriage-mongot` | MongoDB Search, which Community does not bundle and which ships separately |
| `logtriage-llm` | llama.cpp serving the triage model, with the chat template applied |
| `logtriage-embed` | llama.cpp in embedding mode, serving the retrieval model |

Two pairings are worth knowing before reading the file.

**`mongod` and `mongot` are one deployment in two containers.** `$search` and `$vectorSearch` are
not MongoDB operators — they are resolved by `mongot`, which follows the collection through a
change stream. That is why the server runs as a replica set and why there is a bootstrap step:
`scripts/logtriage/bootstrap.sh` initiates the set and creates the user `mongot` authenticates
as, neither of which can be expressed declaratively.

**The two llama.cpp services are separate because `--embedding` replaces the chat endpoints**
rather than adding to them. One server cannot serve both roles.

## What is built and what is pulled

Unlike its sibling stacks, this one builds no image of its own.

The two MongoDB services run published images pinned by **manifest-list digest**, chosen so the
same file works on the workstation (amd64) and on an ARM laptop — both digests were checked to
publish the pair. Their configuration is bind-mounted from `docker/logtriage/`.

The two inference services reuse `docker/llama-gpu/Dockerfile.llama-gpu`, the same context as the
inference stack's `llama-gpu`. That entrypoint gained one env-gated flag for this stack,
`EMBEDDING`, which appends `--embedding --pooling`; it defaults off, so every existing service's
command line is unchanged.

## Ports

All loopback-only. `mongod` holds a copy of whatever logs were ingested, and both inference
servers are unauthenticated by design.

| Port | Service |
|------|---------|
| 27017 | `logtriage-mongo` |
| 8006 | `logtriage-llm` |
| 8007 | `logtriage-embed` |
| 8008 | `logtriage-mongot` health |
| 9946 | `logtriage-mongot` metrics |

`mongot`'s gRPC query port is deliberately **not** published: `mongod` reaches it over the compose
network, and the MongoDB manual warns that a query server reachable from public networks risks
unauthorized access.

## Entry points

```bash
$ make logtriage-secrets     # keyfile, mongot password, root credential (once)
$ make logtriage-up
$ make logtriage-bootstrap   # replica set and the mongot user (once, after first up)
$ make logtriage-status
$ make logtriage-down
```

## Related

- [docker_compose_overview.md](docker_compose_overview.md) — the inference stack's topology
- [../../docker/logtriage/README.md](../../docker/logtriage/README.md) — the two configuration files
- [../../logtriage/README.md](../../logtriage/README.md) — the tool these services exist for
