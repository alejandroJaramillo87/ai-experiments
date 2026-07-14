# D5 recipe system: implementation investigation

Comparison of the Phase 0 D5 design (roadmap document, "D5: Recipe system") against the
current repository state, with a work breakdown, acceptance checks, deviations, and open
questions. Per operating rules, the repository is the source of truth; where the roadmap
document disagrees with the repository, this investigation records the repository state
and treats the document as advisory.

## Table of contents

- [Design summary](#design-summary)
- [Current repository state](#current-repository-state)
- [Drift and blockers found](#drift-and-blockers-found)
- [Deviations from the document](#deviations-from-the-document)
- [Work breakdown](#work-breakdown)
- [Acceptance checks](#acceptance-checks)
- [Risks](#risks)
- [Open questions](#open-questions)

## Design summary

D5 makes one YAML file fully specify a training run. It is the shared contract consumed
by the trainer loop (D6), the memory-fit calculator (D4), the benchmark harness (D8),
the watchdogs (D10), and later phases. The document specifies:

1. Hydra with typed structured configs: dataclasses in `training/recipes/schema.py`
   registered in Hydra's ConfigStore; YAML recipes in `training/recipes/` compose
   against them. Typos and wrong types fail at load, not mid-run. OmegaConf is the
   fallback if Hydra blocks; the dataclass schema survives that swap.
2. Top-level groups: `run`, `model` (hf or scratch), `data`, `tokenizer`, `optimizer`,
   `schedule`, `precision`, `memory`, `checkpoint`, `tracking`, `watchdog`.
3. Run directory convention: `${run.output_root}/${run.name}/${now:...}`, containing the
   resolved config snapshot, logs, benchmark JSON staging, and checkpoint pointers.
   Default output root `/mnt/ai-data/training/runs`.
4. `validate(recipe)`: cross-field invariants beyond types (fp8 dimension support,
   `cpu_adam` requires offload, tokenizer hash matches shards), called by every entry
   point before any GPU allocation.
5. Reference recipes shipped in Phase 0: `reference/pretrain-150m.yaml`,
   `reference/pretrain-1b.yaml`, `reference/tiny-e2e.yaml`.

Acceptance per the document: typed load of every shipped recipe; clear failure on
misspelled key or wrong type; CLI overrides (`optimizer.lr=1e-4`) work through every
entry point; the resolved snapshot reloads bit-identically; unit tests cover each
cross-field invariant.

## Current repository state

| Area | State |
|------|-------|
| D1 container | Done: `docker-compose.training.yaml`, `docker/training/Dockerfile.training`, Makefile targets `train-build`, `train-shell`, `train-smoke`, `train-prepare` |
| Hydra dependency | Already pinned: `hydra-core==1.3.2` in the `training` group of `pyproject.toml`, resolved in `uv.lock` |
| D2 data pipeline (lite) | Done: `training/data/prepare.py`, `training/data/shards.py`; shard manifest records a tokenizer hash and content hashes; `TokenizerMismatchError` exists |
| Recipe system | Absent: no `training/recipes/`, no schema, no reference YAML, no `validate()` |
| D4 calculator, D6 loop/tracking, D8, D10 | Absent: consumers of the recipe do not exist yet |
| Tests | `tests/training/test_prepare.py`, `test_shards.py`; CPU-only, runnable in the sandbox |
| Entry-point scripts | `scripts/training/prepare_data.py`, `scripts/training/smoke_test.py` (recently renamed from `scripts/train/`) |

The compose file mounts `/mnt/ai-data/training` read-write and `./scripts` read-only
into the container. The document's default run output root
(`/mnt/ai-data/training/runs`) is consistent with the existing mount.

## Drift and blockers found

These predate D5 but block or shape it. The repository file layout is the truth; the
stale references are the bugs.

1. **`scripts/train` vs `scripts/training`.** Commits `c13f60b` and `eb8b26b` renamed
   `scripts/train/` to `scripts/training/`, but stale references remain:
   - `Makefile` targets `train-smoke` and `train-prepare` run `scripts/train/...`.
   - `docker/training/Dockerfile.training` has `COPY ... scripts/train ./scripts/train`
     and `RUN python scripts/train/smoke_test.py --build`; the image build fails
     against the current tree.
   - Docstrings in `scripts/training/prepare_data.py` show `scripts/train/` examples.
2. **The `training/` package is not visible inside the container.** The Dockerfile
   copies only `scripts/train` and the compose file mounts only `./scripts` and
   `./logs/training`. `prepare_data.py` imports `training.data`, which resolves on the
   host repository checkout but not inside the container as currently assembled. D5
   makes this acute: the schema module and the recipe YAML files live under
   `training/recipes/` and must be importable and readable in the container. Either the
   Dockerfile copies `training/` or the compose file mounts `./training` read-only
   (the compose header says repository sources are "mounted live for iteration", which
   argues for the mount).

## Deviations from the document

1. **Consumers referenced by `validate()` do not exist yet.** The document's invariants
   reference D8 (fp8 supported dims) and D10 (watchdog thresholds). Phase 0 D5 should
   implement the invariants that are checkable today — tokenizer hash versus the D2
   shard manifest (reusing `training/data/shards.py`), `cpu_adam` requires
   `memory.offload`, schedule/step sanity — and leave the fp8-dimension check as a
   clearly marked stub that later work fills in. The schema fields themselves
   (`precision`, `watchdog`) ship now so the contract is stable.
2. **Entry points are singular today.** "CLI overrides work through every entry point"
   currently means a single recipe loader helper plus one validation CLI; D6's trainer
   adopts the same helper later. The loader function, not each script, owns override
   handling.
3. **Dependency management is uv, not anything the document assumes.** Hydra is already
   in the locked `training` group; no dependency changes are expected.

## Work breakdown

1. `training/recipes/schema.py` — dataclasses for the eleven groups listed in the
   design, registered in Hydra's ConfigStore. Model group models the hf-or-scratch
   choice explicitly (two optional sub-blocks with a validated exactly-one rule, the
   simplest Hydra-compatible encoding).
2. `training/recipes/loader.py` (exported from `training/recipes/__init__.py`) —
   `load_recipe(path, overrides)` returning a fully typed object; strict struct mode so
   unknown keys fail; resolved-snapshot writer and reloader implementing the run
   directory convention.
3. `training/recipes/validate.py` — `validate(recipe)` with the implementable
   cross-field invariants; tokenizer-hash check delegates to the D2 manifest reader.
4. Reference recipes: `training/recipes/reference/pretrain-150m.yaml`,
   `pretrain-1b.yaml`, `tiny-e2e.yaml`.
5. `scripts/training/validate_recipe.py` — thin CLI (load, override, validate, print
   resolved config) following `.styles/PYTHON_OUTPUT_STYLE.md`; the first "entry point"
   exercising overrides end to end.
6. `tests/training/test_recipes.py` — typed load of each shipped recipe; misspelled
   key and wrong type fail with clear errors; override application; snapshot reloads
   bit-identically; one test per validate() invariant. All CPU-only, sandbox-runnable.
7. Container and Makefile plumbing: fix the `scripts/train` -> `scripts/training`
   stale references; make `training/` (package and recipes) visible in the container;
   add a `make train-validate CONFIG=...` target.

Items 1-6 are sandbox-executable. Item 7's `make train-build` verification is
host-side.

## Acceptance checks

- Sandbox: `pytest tests/training/` passes, covering every document acceptance bullet
  that does not require the container.
- Host: `make train-build` succeeds (currently fails on the stale `scripts/train`
  COPY), and `make train-validate CONFIG=training/recipes/reference/tiny-e2e.yaml`
  prints a resolved config and exits 0. Human runs and reports output.

## Risks

- Hydra 1.3.2 struct-mode edge cases around optional nested dataclasses (the hf/scratch
  model choice); mitigated by keeping both blocks optional and validating in
  `validate()` rather than fighting ConfigStore.
- Bit-identical snapshot reload requires disciplined OmegaConf save/load (resolve
  before save, compare `OmegaConf.to_yaml` output); the test pins the behavior.
- Schema fields for unbuilt consumers (fp8 backends, watchdog thresholds) risk churn
  when D8/D10 land; acceptable because the document treats the schema as the contract
  designed first.

## Open questions

1. Should the `scripts/train` -> `scripts/training` reference fixes (Makefile,
   Dockerfile, docstrings) land on this branch as a prerequisite commit, or separately?
2. Container visibility for `training/`: read-only compose mount (live iteration,
   matches the compose header's stated intent) or Dockerfile COPY (image
   self-contained)?
3. Trim or keep the `watchdog` and fp8 `precision` fields in the Phase 0 schema given
   D8/D10 are unbuilt? Default assumption: keep, with stubbed validation.
