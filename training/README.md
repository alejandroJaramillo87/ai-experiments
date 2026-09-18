# training

The training tier: corpus preparation, tokenizer training, VRAM fit estimation, the
pretraining loop with checkpointing and resume, supervised fine-tuning, export of the
merged HF checkpoint, and baseline-versus-tuned evaluation.

Invoked as `python -m training <command>`; `cli.py` dispatches to one handler per verb in
`commands/`. Runtime concerns are separated deliberately: `runtime/` holds the loop, the
global run lock, tracking, and the watchdog; `calculator/` estimates and calibrates memory
fit before a run rather than discovering it by OOM.

## Relationship to the rest of the tree

Imports `common/` and nothing else first-party. Imported by `tests/training/` and — for the
run lock, watchdog, and recipe loader — by `interp/`.

Two consequences of that direction:

**`recipes/` is the configuration home for both tiers.** Interpretability recipes live in
`training/recipes/interp/` and organism recipes in `training/recipes/organisms/`, not under
`interp/`. Look here when an interp command asks for a recipe.

**The GPU run lock is global.** `training/runtime/lock.py` serializes GPU work across the
repository, so an interp harvest and a training run contend for the same lock. That is the
mechanism behind the one-GPU-consumer-at-a-time rule; it is not advisory.

The Rust harness does not import this package, but its quality gate resolves a Python that
can (`rust-harness/runtime/src/pyenv.rs`) — the chat-template contract test renders a
conversation through `training.data.sft.render_segments` and compares it byte for byte
against the Rust renderer.

## Entry points

```bash
make train CONFIG=training/recipes/reference/tiny-e2e.yaml
make train-validate CONFIG=...   # recipe schema check, no GPU
make memfit CONFIG=...           # VRAM fit estimate before committing a run
make export-hf CHECKPOINT=... ARGS="--name <name>"
```

Export runs in a separate CPU-only image (`docker/training/Dockerfile.export`, built by
`make export-build`) and writes `/mnt/ai-data/models/trained/<name>/hf/`, the checkpoint
directory the rust harness serves and interp captures from, plus a manifest carrying
`model_hash`. It produces no GGUF. Several targets — `eval-ft`, `promote-scorecard`,
`generate-teacher` — drive the live serving stack from the host rather than the training
container.

## Where the real docs live

[../docs/training/](../docs/training/) holds the living documentation, one document per
capability: `data_pipeline.md`, `tokenizer_training.md`, `memory_fit.md`,
`tracking_checkpointing.md`, `eval_wrapper.md`, and the rest. Phase
verification records are in [../docs/verification/](../docs/verification/), and the frozen
design investigations in [../docs/training/investigations/](../docs/training/investigations/).
