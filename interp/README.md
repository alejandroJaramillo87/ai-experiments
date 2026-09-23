# interp

The interpretability tier: activation harvesting to NVMe, sparse autoencoder training and
evaluation, linear probes (capture, train, evaluate, export, monitor), steering-vector
extraction and steered serving, and model organism generation, verification and sealing.
Invoked as `python -m interp <verb>`; it runs as a JupyterLab workbench container for
exploratory work and as host-driven CLI commands for batch jobs.

## Layout

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [commands/](commands/README.md) | One module per verb of `python -m interp`. |
| [notebooks/](notebooks/README.md) | Working notebooks for interpretability projects: exploratory analysis, direction pre-validation before a steering deployment (`interp/steering/`), and probe or capture inspection. |
| [organisms/](organisms/README.md) | Model organisms (P4-D1): a corpus generated from a sealed ground truth, a fine-tune on it through the training tier, and a verification that the implanted behavior is present and local. |
| [probes/](probes/README.md) | Linear probes (W2 to W4): capture labeled activations to a shard directory, train a per-layer logistic regression on them, evaluate it against mandatory black-box baselines, export the winner for the Rust harness, and aggregate the scores the serving process takes in-band. |
| [sae/](sae/README.md) | Sparse autoencoders (P4-D3): training from a harvested activation capture, evaluation with the core metrics and the SAEBench driver, and a registry that lists, loads and validates runs. |
| [steering/](steering/README.md) | Control-vector steering (W1): extract a direction from a labeled capture, export it as the safetensors image the Rust harness applies inside its forward pass, wire the harness to serve with it, and score the three arms (unsteered, steered, and a judge) against the live services. |

<!-- inventory:dirs:end -->

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `cli.py` | Unified interp CLI. |
| `families.py` | Per-family hook-point registry for the interp tier. |
| `harvest.py` | Activation harvesting (interp Phase 4 P4-D2): a workbench batch job. |
| `manifest.py` | Artifact manifest helpers shared by the write-back deliverables (W1-W4). |

<!-- inventory:modules:end -->

## The CLI

`cli.py` lists one module per verb in `COMMAND_MODULES`; each exposes `configure(sub)` and
a handler, and `common.obs.dispatch` wraps the run so every invocation gets a run id, an
event file and a catalog row. The site's interp reference pages are rendered from these
parsers by `make site-gen`, so a new verb is documented by being registered.

| Verb | Module | Make target |
|------|--------|-------------|
| `harvest` | `commands/harvest.py` | `make harvest`, `make harvest-check-hooks` |
| `gc` | `commands/interp_gc.py` | `make interp-gc` |
| `organism generate`, `verify`, `seal`, `check-seal` | `commands/organism.py` | `make organism-generate`, `make organism-verify`, `make organism-seal`, `make organism-check-seal` |
| `probe-capture` | `commands/probe_capture.py` | `make probe-capture` |
| `probe-train` | `commands/probe_train.py` | `make probe-train` |
| `probe-eval` | `commands/probe_eval.py` | `make probe-eval` |
| `probe-export` | `commands/probe_export.py` | `make probe-export` |
| `probe-monitor` | `commands/probe_monitor.py` | `make probe-monitor` |
| `sae-train` | `commands/sae_train.py` | `make sae-train` |
| `sae-eval` | `commands/sae_eval.py` | `make sae-eval` |
| `steer-extract` | `commands/steer_extract.py` | `make steer-extract` |
| `steer-scorecard` | `commands/steer_scorecard.py` | `make steer-scorecard` |
| `smoke` | `commands/smoke_test.py` | `make interp-smoke` |
| `runs` | `commands/runs.py`, delegating to `common.obs.runs` | `make interp-runs` |

The GPU verbs run `scripts/interp/preflight.sh` first through their make targets
(`harvest`, `sae-train`, `sae-eval`, `probe-capture`, `steer-extract`, and the container
targets `interp-up` and `interp-shell`), which refuses while `llama-gpu` or `vllm-gpu` is
running. `probe-export`, `probe-eval` and `probe-monitor` load no model and run beside
inference.

## Relationship to the rest of the tree

Imports `common/` throughout, and `training/` lazily, inside functions, at six points: the
recipe loader and validator, the global run lock and the watchdog (`runtime/`), the
tracker, the teacher client and `generate-teacher` verb the organism pipeline drives, the
serving registry, and the export manifest reader. Nothing in `training/` imports this
package. Outside `tests/interp/`, it is imported by `tests/training/test_export_harness.py`
and the hardware suite, and `rust-harness/fixtures/export_interp_fixtures.py` calls the two
exporters to write the fixtures the Rust loaders are tested against.

Three things a newcomer gets wrong:

**The recipes are not here.** They live in `training/recipes/interp/` and
`training/recipes/organisms/`, because the recipe loader and schema belong to the training
tier.

**A harvest contends with a training run.** Borrowing `training.runtime.lock` means the GPU
lock is shared repository-wide, by design.

**The artifacts that cross into Rust are contracts.** `probes/capture.py` defines the npz
capture shard that `rust-harness/runtime/src/npz.rs` writes; `probes/export.py` writes the
probe the harness scores with (safetensors plus a JSON sidecar) and `steering/export.py`
the control vector it steers with (safetensors plus a sibling manifest, tensors named
`block.<index>`). Neither is GGUF, and the contracts are in `schemas/`.

## Entry points

```bash
make interp-up                        # workbench container
make harvest RECIPE=...               # activation harvesting
make probe-train ARGS="--capture ... --name ..."
make steer-extract ARGS="--capture ... --out ..."
```

## Where the real docs live

- [../docs/interp/README.md](../docs/interp/README.md) — the tier's documentation, indexed
- [../docs/interp/writeback.md](../docs/interp/writeback.md) — the write-back path
- [../docs/interp/rust-harness-interp.md](../docs/interp/rust-harness-interp.md) — the harness interp seam and its compatibility contract
- [../schemas/README.md](../schemas/README.md) — the artifact contracts shared with Rust
- [../tests/interp/README.md](../tests/interp/README.md) — what the suite fakes and what it drives
