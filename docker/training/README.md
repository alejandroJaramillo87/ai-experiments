# docker/training

The build context for the training tier: the GPU image the tier trains in, the CPU-only
image that merges an adapter into the HF checkpoint the harness serves, and the entrypoint
the training image runs. The `training`, `prepare` and `export` services, their mounts and
the egress network are defined in
[../../docker-compose.training.yaml](../../docker-compose.training.yaml), which is the
source of truth; this directory is what that definition builds from.

## Layout

<!-- inventory:build:start -->

| File | Purpose |
|------|---------|
| `Dockerfile.export` | The merged HF checkpoint exporter, a CPU-only image kept separate from the training image. |
| `Dockerfile.training` | The training tier's GPU image for the RTX 5090 (Blackwell) and the Ryzen 9950X (Zen 5). |
| `entrypoint.sh` | The training container's entrypoint: echo the environment, then exec the given command (Phase 0 D1). |

<!-- inventory:build:end -->

## Traps

**Two images by decision.** The image that trains a model is not reused to prepare it for
serving: `Dockerfile.export` carries a CPU-only stack, so a merge never competes with a
serving swap for VRAM, and it writes an HF checkpoint directory and no GGUF.

**The build runs the smoke test.** `make train-build` fails when
`python -m training smoke --build` cannot import the stack or see the GPU. The dependencies
come from `uv.lock` and the `training` group, never from lines in the Dockerfile.

**The container starts no server.** The entrypoint execs the given command; the tier is run
one verb at a time through the `train-*` make targets.

## Consumers

- `make train-build` builds the training image; the `train-*` targets in `make/training.mk`
  run the verbs.
- [../../docs/sandbox/docker_training_overview.md](../../docs/sandbox/docker_training_overview.md)
  explains the image, and [../../docs/training/](../../docs/training/README.md) the tier.
