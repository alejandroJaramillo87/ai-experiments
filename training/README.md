# training

The training tier: the Python package that fine-tunes, continually pretrains and trains
from scratch on this workstation, and exports what it trained as the merged HF checkpoint
the rust harness serves and the interp tier captures from. It holds corpus preparation,
tokenizer training, the recipe system every rung reads, stepping a recipe on the GPU
before a long run, the pretraining loop with checkpointing and resume, supervised
fine-tuning on TRL, the export, and the held-out and lm-eval records a base-versus-fine-tune
comparison reads.

## Layout

One subpackage per concern. The table is generated from each subpackage's README, and
`make quality` holds it to the tree, so a subpackage without a README fails the gate.

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [commands/](commands/README.md) | The verbs of `python -m training`, one module each. |
| [continual/](continual/README.md) | The pieces that exist only for continual pretraining. |
| [data/](data/README.md) | Everything between a corpus and a batch. |
| [eval/](eval/README.md) | The two evaluators a base-versus-fine-tune comparison runs, and the host-side runner that drives them. |
| [export/](export/README.md) | The exporter, where a trained checkpoint becomes one loadable HF directory. |
| [models/](models/README.md) | Model construction and the accounting behind it. |
| [recipes/](recipes/README.md) | The recipe system and the recipes themselves. |
| [runtime/](runtime/README.md) | The machinery around a run, shared by every rung. |
| [tune/](tune/README.md) | Supervised fine-tuning on the TRL backend. |

<!-- inventory:dirs:end -->

Two modules sit at the top level beside the CLI's `__main__.py`:

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `cli.py` | Unified training CLI (resolved D4 design decision 1). |
| `serving.py` | Training-side serving policy plus re-exports of the shared client. |

<!-- inventory:modules:end -->

The import graph runs downward: `commands/` and `cli.py` at the top, then `tune/`, `eval/`
and `runtime/`, then `recipes/`, `export/` and `continual/`, with `models/` and `data/` as
the sinks. Two back-edges are deliberate: `recipes/validate.py` reads `models/architecture.py`
and `runtime/schedules.py` to check a recipe against what it implies, and `data/` reads the
family registry in `models/families.py` to render a chat template.

## The command line

`python -m training <verb>` is the one entry point; `cli.py` builds the parser, and each
verb is a module in `commands/` exposing `configure(sub)`. Every invocation runs through
the traced dispatch in `common/obs/dispatch.py`, which stamps a run id and writes
`cmd.start` and `cmd.end` records, so a training command is a run in the observability
catalog like any other. Exit codes: 0 on success, 1 on a load, validation or runtime
failure, 2 on invalid usage; `--json` emits only JSON.

The `make` targets route each verb to the container it needs, from
`docker-compose.training.yaml`. Staging verbs run in `prepare`, the one service with
network egress; training verbs run in `training`, with the GPU and `/mnt/ai-data/training`
mounted; the export runs in `export`, a CPU-only image; two verbs run on the host because
they talk to the running inference stack. A verb with no target is run by hand as
`python -m training <verb>` in the container `make train-shell` opens.

| Verb | Module | Target | Runs in |
|------|--------|--------|---------|
| `download` | `commands/download.py` | `make train-download` | `prepare` |
| `prepare-data` | `commands/prepare_data.py` | `make prepare-data` | `prepare` |
| `prepare-corpus` | `commands/prepare_corpus.py` | `make prepare-corpus` | `prepare` |
| `build-contamination` | `commands/build_contamination.py` | `make build-contamination` | `prepare` |
| `prepare-sft` | `commands/prepare_sft.py` | `make prepare-sft` | `prepare` |
| `train-tokenizer` | `commands/train_tokenizer.py` | `make train-tokenizer` | `prepare` |
| `validate` | `commands/validate_recipe.py` | `make train-validate` | `training` |
| `preflight` | `commands/preflight.py` | `make train-preflight` | `training` |
| `smoke` | `commands/smoke_test.py` | `make train-smoke` | `training` |
| `step` | `commands/train_step.py` | `make train-step` | `training` |
| `train` | `cli.py`, then `runtime/loop.py` | `make train` | `training` |
| `train-sft` | `cli.py`, then `tune/sft.py` | `make train-sft` | `training` |
| `export` | `cli.py`, then `export/` | `make export-hf` | `export` |
| `eval-holdout` | `cli.py`, then `eval/holdout.py` | none; `eval-ft` runs it | `training` |
| `eval-lmeval` | `cli.py`, then `eval/lmeval.py` | none; `eval-ft` runs it | `training` |
| `eval-ft` | `commands/eval_ft.py` | `make eval-ft` | host |
| `generate-teacher` | `commands/generate_teacher.py` | `make generate-teacher` | host |
| `make-replay-synthetic` | `commands/make_replay_synthetic.py` | none | `training` |
| `report` | `commands/report.py` | none | either |
| `runs` | `commands/runs.py` | none | either |

`make train-build` builds the training image, `make export-build` the export image, and
`make train-shell` opens a shell in the training container; `make help` prints the whole
section.

## Relationship to the rest of the tree

Imports `common/` and nothing else first-party: the traced dispatch and the `Printer`
status contract from `common/obs/` and `common/output.py`, and the serving client from
`common/serving.py`. Imported by `tests/training/` and `tests/hardware/`, and by `interp/`
for more than the README used to admit: the run lock and watchdog (`runtime/lock.py`,
`runtime/watchdog.py`), the tracker (`runtime/tracking.py`), the recipe loader and
validator, the export manifest reader (`export/manifest.py`), the teacher harness
(`data/teacher.py`) and the `generate-teacher` command module, which `organism-generate`
calls directly.

Three consequences of that direction:

**`recipes/` is the configuration home for both tiers.** Interp recipes live in
`training/recipes/interp/` and organism recipes in `training/recipes/organisms/`, not under
`interp/`. Look here when an interp command asks for a recipe.

**The GPU run lock is global.** `runtime/lock.py` serializes GPU work across the
repository, so an interp harvest and a training run contend for the same lock. That is the
mechanism behind the one-GPU-consumer-at-a-time rule; it is not advisory.

**The chat template is a cross-language contract.** The Rust harness does not import this
package, but its gate resolves a Python that can (`rust-harness/runtime/src/pyenv.rs`):
`rust-harness/fixtures/reference_tokenizer.py` renders a conversation through
`data/sft.py`'s `render_segments` with the family spec from `models/families.py` and
compares it byte for byte against the Rust renderer.

## Traps

- **Nothing here produces GGUF.** The export writes the merged HF directory and a
  manifest carrying `model_hash`, the sha256 of `model.safetensors`; that hash is the
  identity every interp artifact fitted to the model carries.
- **The staging verbs need the `prepare` service.** The `training` service has no egress,
  so a hub download or dataset fetch run there fails on the network, not on the code.
- **A recipe is validated before anything loads.** `commands/recipe.py` is the shared
  load-and-validate step; a schema or cross-field error exits 1 before torch imports.

## Where the real docs live

[../docs/training/](../docs/training/) holds the living documentation, one directory per
rung over a base every rung shares: `base/` (the recipe system, the step, tracking and
checkpointing, the data pipeline, the export, the reference tables), `fine-tuning/`,
`continual/` and `from-scratch/`. The unit tests are inventoried in
[../tests/training/README.md](../tests/training/README.md) and the hardware mirrors in
[../tests/hardware/README.md](../tests/hardware/README.md). Phase verification records are in
[../docs/verification/](../docs/verification/), and the frozen design investigations in
[../docs/training/investigations/](../docs/training/investigations/).
