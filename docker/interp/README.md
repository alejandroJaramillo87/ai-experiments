# docker/interp

The build context for the interpretability workbench: the image with the interp stack on
PyTorch cu130, and the entrypoint that bootstraps the storage tree before JupyterLab or
the interp CLI runs. The `interp` service, its port binding and its mounts are defined in
[../../docker-compose.interp.yaml](../../docker-compose.interp.yaml), which is the source
of truth; this directory is what that definition builds from.

## Layout

<!-- inventory:build:start -->

| File | Purpose |
|------|---------|
| `Dockerfile.interp` | The interpretability workbench image: PyTorch cu130 on sm_120 for the RTX 5090 (Blackwell), with the interp stack and JupyterLab. |
| `entrypoint.sh` | The interpretability workbench's entrypoint: bootstrap the storage tree, then exec the given command (interp Phase 0 I1). |

<!-- inventory:build:end -->

## Traps

**The build runs the smoke test.** `make interp-build` fails when `python -m interp smoke`
cannot import the stack or see the GPU, so a dependency that would downgrade torch fails
at build time rather than in a notebook. The dependencies come from `uv.lock` and the
`interp` group, never from lines in the Dockerfile.

**JupyterLab is bound to localhost by the compose port mapping**, never by arguments in
the entrypoint.

## Consumers

- `make interp-build` builds the image; `make interp-up` preflights and starts JupyterLab.
- [../../docs/sandbox/docker_interp_overview.md](../../docs/sandbox/docker_interp_overview.md)
  explains the image and the service.
- [../../docs/interp/](../../docs/interp/README.md) documents the interpretability tier the
  image serves.
