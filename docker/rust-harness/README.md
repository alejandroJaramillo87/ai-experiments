# docker/rust-harness

The build context for the rust harness image: a reproducible build of the `harness`
binary from the pinned toolchain and the locked dependency graph, and the entrypoint the
serving service uses to assemble its arguments from the environment. The `harness` and
`harness-serve` services, their mounts and the serving port are defined in
[../../docker-compose.rust-harness.yaml](../../docker-compose.rust-harness.yaml), which is
the source of truth; this directory is what that definition builds from.

## Layout

<!-- inventory:build:start -->

| File | Purpose |
|------|---------|
| `Dockerfile.rust-harness` | Reproducible build of the rust-harness binary (single-binary coding agent with the integrated CPU inference runtime). |
| `entrypoint.serve.sh` | Start the rust harness as an HTTP server. |

<!-- inventory:build:end -->

## Traps

**The container is for reproducibility, not containment.** The same pinned toolchain and
`Cargo.lock` produce the same binary on any machine, which is what runbook entry RH.12
verifies; the agent sandbox under [../agents/](../agents/README.md) is the contained
environment.

**Only the serving service uses the entrypoint.** The interactive `harness` service keeps
the bare binary as its entrypoint and takes its arguments on the command line; the
entrypoint exists because docker compose cannot pass a flag conditionally.

## Consumers

- `make harness-build` builds the image; `make harness-run`, `make harness-chat` and
  `make harness-serve` run it.
- [../../rust-harness/README.md](../../rust-harness/README.md) is the workspace the image
  builds, and [../../docs/rust-harness/](../../docs/rust-harness/README.md) its
  documentation.
