# docker/llama-cpu

The build context for `llama-cpu`, the low-latency CPU inference service: the Dockerfile,
the entrypoint it runs, and the huge-page interceptor library it compiles in. The service
itself (ports, cpuset, volumes, environment) is defined in
[../../docker-compose.inference.yaml](../../docker-compose.inference.yaml), which is the
source of truth; this directory is what that definition builds from.

## Layout

<!-- inventory:build:start -->

| File | Purpose |
|------|---------|
| `Dockerfile.cpu` | The llama-cpu inference image: llama.cpp compiled for Zen 5 with the huge-page interceptor from hugepages/ preloaded. |
| `entrypoint.cpu.sh` | The llama-cpu service's entrypoint: starts llama-server with the zen5_optimizer interceptor preloaded, so mmap() of the model is backed by huge pages. |

<!-- inventory:build:end -->

[hugepages/](./hugepages/README.md) is the `zen5_optimizer` library the Dockerfile compiles in:
an `mmap()` interceptor that backs large model mappings with huge pages. Its README covers
the standalone build, the tests and the design, and
[its explicit huge-page guide](./hugepages/docs/hugepages-explicit.md) the host-side setup.

`hugepages/` lived at the repository root until 2026-09-19. It moved here because the
llama-cpu image is its only consumer: the Dockerfile copies `docker/llama-cpu/hugepages/CMakeLists.txt` and
`docker/llama-cpu/hugepages/src/` into the build and nothing else in the tree reads them, so a root-level
directory implied a component the repository does not have.

## Consumers

- `docker compose -f docker-compose.inference.yaml build llama-cpu` builds
  the image from this directory with the repository root as context.
- `make up` starts the service; [../../docs/sandbox/docker_llama_cpu_overview.md](../../docs/sandbox/docker_llama_cpu_overview.md)
  explains the build stages and the optimizations in detail.
- `docker/llama-cpu/hugepages/Makefile` is the library's standalone build and test entry, run
  on the host, not in the sandbox: its tests allocate huge pages.

## Traps

**The Dockerfile's `COPY` paths are relative to the repository root**, not to this
directory, because every compose build uses `context: .`. A path here is spelled
`docker/llama-cpu/hugepages/...` in the Dockerfile.

**The interceptor only helps with huge pages reserved on the host.** Without the
`nr_hugepages` setup in [the explicit huge-page guide](./hugepages/docs/hugepages-explicit.md)
the library falls back to ordinary pages and the service runs, slower, with nothing in the
log to say so beyond the interceptor's own messages.

**`.styles/CPP_STYLE.md` applies to `hugepages/src/`**, the only C++ in the repository.
