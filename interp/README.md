# interp

The interpretability tier: activation harvesting to NVMe, sparse autoencoder training and
evaluation, linear probes (capture, train, evaluate, monitor), steering-vector extraction and
live-service steering, and model organism generation, verification, and sealing.

Invoked as `python -m interp <command>`; `cli.py` dispatches to one handler per verb in
`commands/`. It runs as a JupyterLab workbench container for exploratory work and as
host-driven CLI commands for batch jobs.

## Relationship to the rest of the tree

Imports `common/` heavily, and `training/` at three points — lazily, inside functions — for
the recipe loader, the global run lock, and the watchdog. Nothing in `training/` imports
this package. `tests/interp/` covers it.

Three things a newcomer gets wrong:

**The recipes are not here.** They live in `training/recipes/interp/` and
`training/recipes/organisms/`, because the recipe loader and schema belong to the training
tier.

**A harvest contends with a training run.** Borrowing `training.runtime.lock` means the GPU
lock is shared repository-wide, by design.

**Several commands are serving-dependent, and one refuses to run.** Steering targets a live
llama-gpu or vllm-gpu; `probe-monitor` actively refuses to start while inference is up,
rather than competing for VRAM. Most `make interp-*` targets first run
`scripts/interp/preflight.sh`, which gates on the GPU and is workstation-only.

The artifact formats crossing into Rust are the reason this tier has cross-language tests:
`interp/probes/capture.py` defines the npz capture shard that
`rust-harness/runtime/src/npz.rs` writes; `interp/probes/export.py` writes the probe the
harness reads (safetensors plus a JSON sidecar) and `interp/steering/export.py` the
control-vector image (safetensors plus a sibling manifest).
Those contracts are in `schemas/`.

## Entry points

```bash
$ make interp-up                        # workbench container
$ make harvest RECIPE=...               # activation harvesting
$ make probe-train CAPTURE=...
$ make steer-extract VECTOR=...
```

## Where the real docs live

- [../docs/interp/platform.md](../docs/interp/platform.md) — the workbench and the platform
- [../docs/interp/writeback.md](../docs/interp/writeback.md) — the write-back path
- [../docs/interp/rust-harness-interp.md](../docs/interp/rust-harness-interp.md) — the harness interp seam and its compatibility contract
- [../docs/interp/interp-scale-reference.md](../docs/interp/interp-scale-reference.md) — sizing
- [../schemas/README.md](../schemas/README.md) — the artifact contracts shared with Rust
