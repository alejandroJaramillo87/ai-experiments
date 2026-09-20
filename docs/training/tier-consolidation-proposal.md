# Training tier consolidation: the refactors that remain

**Parked proposal — recorded 2026-09-19. Nothing here is scheduled; this document is
edited only as an item graduates out.**

When the tier was trimmed to its three rungs (SFT, continual pretraining, training from
scratch; see [README.md](README.md)), the survey that guided the trimming also found
duplication that a removal cannot fix, because both copies are load-bearing. Three items
survived the survey. Each is recorded here with its evidence, the options, and the
condition under which it should be started, so the decision is made once. Small cleanups
the survey found (one load-and-validate helper, one torch dtype map, the split of the
teacher handler tests, the teacher manifest's GGUF field) were done in the same change
that wrote this document and are not repeated here.

## Table of contents

- [Two training loops, two checkpoint layouts](#two-training-loops-two-checkpoint-layouts)
- [The inline handlers in the CLI module](#the-inline-handlers-in-the-cli-module)
- [Roadmap codes without a glossary](#roadmap-codes-without-a-glossary)
- [Order and gating](#order-and-gating)

## Two training loops, two checkpoint layouts

The tier runs training through two loops:

| | `training/runtime/loop.py` | `training/tune/sft.py` with `sft_callbacks.py` |
|---|---|---|
| Rungs | scratch, continual (`train`) | SFT (`train-sft`) |
| Step driver | the D6 hand-written loop over the sharded sources | TRL `SFTTrainer` |
| Checkpoint directory | `step_N/` (torch or DeepSpeed state) | `checkpoint-N/` (TRL trainer state) |
| Completeness marker | `meta.json` written last | `meta.json` stamped by an `on_save` callback |
| Resume | `resume=latest` over `step_N/` | `latest_trl_checkpoint` over `checkpoint-N/` |
| Retention | `ckpt.apply_retention` | `apply_trl_retention` |
| Emergency checkpoint | `save(step, loss, emergency=True)` | `checkpoint-N-emergency/` on watchdog abort |
| Tracker header | built in the loop | built in `train_sft` |
| `latest` symlink | atomic rename | atomic rename |

Each row on the right reimplements the row on the left for TRL's layout, and the export
(`training/export/merge.py`) reads both. The duplicated logic is the checkpoint contract:
what a complete checkpoint is, which ones retention may delete, what `meta.json` carries
so export preflight can detect base drift, and what the tracker header records. The step
driver itself is not duplicated; TRL's packing, collation and callbacks are what SFT uses
it for.

This is the largest duplication in the tier, and the question it raises, whether the
two loops should become one, has a cost the survey did not want to hide:

- **Share the contract, keep two drivers.** Move `meta.json` construction, the tracker
  header, retention, the `latest` update and the completeness test into one module
  (`training/runtime/checkpoint.py` is the natural home, since the D6 retention
  already lives there) parameterised by directory prefix, and have both loops call it. Moderate work,
  low risk: neither driver changes, and the tests for each side already state the
  contract. Export keeps reading two prefixes.
- **One driver.** Run SFT through the D6 loop, dropping TRL. Large work and it discards
  what TRL provides (packing, the `SFTTrainer` collator, gradient-checkpointing
  integration with PEFT) to reimplement it, for the benefit of one checkpoint layout.
  This lab's interest is in what training does to a model, not in owning a trainer, so
  this option is recorded as considered and declined.

The first option is the plan. Its gate is below.
### Implementation plan (recorded 2026-09-19)

The step drivers do not change, and neither does any on-disk layout:
`step_NNNNNNNNN/` and `checkpoint-N/` stay exactly as they are, so every recorded runbook
observation and every existing checkpoint remains valid. What moves is the contract
around each driver.

**The duplication, precisely.**

| Concern | `training/runtime/checkpoint.py` (D6) | `training/tune/sft_callbacks.py` (TRL) | To preserve |
|---|---|---|---|
| Directory name | `step_dir_name`, `STEP_DIR_RE` (9-digit) | `checkpoint_dir_name`, `_CHECKPOINT_RE` | layout-specific, stays |
| Complete | any `step_*` dir (the save is atomic) | `meta.json` and `trainer_state.json` present | layout-specific predicate |
| Emergency | `emergency: true` in a plain step dir | `checkpoint-N-emergency/`, outside the plain regex | layout-specific; both mark meta |
| Retention | last N, every M, the `latest` target, emergency; sweeps `*.tmp` | last N, every M if M > 0, emergency, unstamped | one function; the union of the guards is safe (the schema requires `retain_every >= 1`; keeping the `latest` target is a no-op for TRL) |
| `latest` | tmp symlink + rename inside `save_checkpoint` | `update_latest`, same protocol | one function |
| Resolve latest | symlink only | scan, emergency dirs included | one function: the symlink target if complete, else the newest complete by scan |
| Resume | `resolve_resume` (FileNotFoundError) | `resolve_sft_resume` (SftPreflightError) | one function; `sft.py` keeps its exception and message by wrapping |
| `meta.json` | built inline in `loop.py::save` | `write_checkpoint_meta` | shared: step, recipe_hash, git_sha, library_versions, wall_clock, checkpoint_format, attention_implementation, emergency. Loop-only: tokens_seen, loss, optimizer, tokens_per_source. TRL-only: adapter_only, base_model, base_revision, base_config_sha256, backend |
| Library versions | torch, transformers, deepspeed if present | torch, transformers, trl, peft, "unavailable" fallback | one function reporting each of the five that imports |
| Tracker header | inline in `loop.py` (tier pretrain, schedule with decay fields) | inline in `sft.py` (tier sft) | one builder; the schedule block taken from the recipe uniformly |

**Design.** `training/runtime/checkpoint.py` gains a frozen `CheckpointLayout` (name,
`dir_name(step)`, `parse(name) -> (step, emergency) | None`, `complete(dir)`, and the
TRL-only `emergency_dir_name`), two instances `STEP_LAYOUT` and `TRL_LAYOUT`, and the
shared functions take a layout with `STEP_LAYOUT` as the default so every D6 call site
stays source-compatible: `list_checkpoints`, `latest_checkpoint`, `update_latest`,
`apply_retention`, `resolve_resume`, plus `checkpoint_meta` (the shared fields, with
`**extra` for each side's own), `write_meta` (tmp + rename) and `library_versions`. The
D6 save and load functions stay where they are. `training/runtime/tracking.py` gains
`run_header(recipe, tier, recipe_hash)` and both entry points pass its result as
`header=`. `sft_callbacks.py` keeps `checkpoint_dir_name`, `checkpoint_complete` and
`write_checkpoint_meta` (now a thin wrapper adding the TRL-only fields) and loses
`list_trl_checkpoints`, `apply_trl_retention`, `update_latest`, `latest_trl_checkpoint`
and its private versions helper; `resolve_sft_resume` in `sft.py` wraps the shared
resolver and re-raises as `SftPreflightError` with the current message.

**Behavior changes, each deliberate and covered by a test.**

1. D6 `resolve_resume("latest")` with a dangling or absent symlink but complete step
   directories on disk resolves to the newest complete one instead of failing.
2. D6 retention treats `retain_every` as "if greater than zero" (the schema forbids zero,
   so no run changes).
3. TRL retention also keeps the `latest` target (always the newest, so no run changes).
4. `library_versions` in both `meta.json` files reports the same key set.

**Tests.** `tests/training/test_checkpoint.py` keeps every existing D6 test and gains a
parametrized layout-contract class over both layouts (list, latest, retention, resume,
emergency handling, the `*.tmp` sweep for step, the unstamped skip for TRL). The helper
cases in `tests/training/test_sft_entry.py` move there under the shared names; the
`test_meta_records_*` cases stay, since they test the TRL wrapper's fields.
`test_loop.py` and `test_watchdog.py` are unchanged. One new test checks that
`run_header` for a pretrain recipe and an sft recipe carry the same key set.

**Docs.** [tracking_checkpointing.md](tracking_checkpointing.md) (the `checkpoint.py`
row and the retention paragraph cover both layouts),
[sft_entry_point.md](sft_entry_point.md) (which module stamps meta), and the
observability README's series table if it names the callbacks module for the header.
No file is renamed, so no retired-names row. This item then leaves this document.

**Acceptance.** Sandbox: `uv run --no-sync pytest tests/training -q` and `make quality`
after the commit. Workstation: re-run FT.train-resume and FT.export-merged of
[../verification/training/fine-tuning-loop.md](../verification/training/fine-tuning-loop.md)
on the same recipe and confirm the Observed blocks match the earlier recording apart from
timestamps and hashes; run `tiny-e2e` to a second checkpoint, resume it, and confirm
`step_*/meta.json` carries the same keys as before plus the unified `library_versions`.

## The inline handlers in the CLI module

`training/cli.py` holds five handlers inline (`train`,
`train-sft`, `export`, `eval-holdout`, `eval-lmeval`) while every other verb lives in
`training/commands/`. Nothing is duplicated; the cost is that a reader looking for a verb
has two places to look, and the module is the longest in the tier. Moving them is
mechanical, but each move changes the import path the handler tests use, so it should
ride the checkpoint consolidation above (which touches `train` and `train-sft` anyway)
rather than land alone.

### Implementation plan (recorded 2026-09-19)

Each `cmd_*` and its `sub.add_parser(...)` block moves to a module in
`training/commands/` with the `configure(sub)` shape every other command uses
(`training/commands/validate_recipe.py` is the template; `train_step.py` made the move on
2026-09-20 when the calculator retired): `train.py`, `train_sft.py`, `export.py`,
`eval_holdout.py`, `eval_lmeval.py`. `build_parser` becomes the loop over the command
modules alone. The handler tests reach every handler through `cli.build_parser()` and
patch the source modules by dotted path (`training.runtime.loop.train`,
`training.tune.sft.train_sft` and the rest), so they need no edits provided the moved
handlers keep resolving those names at call time as they do now. The entry-point lists in
[README.md](README.md) and `training/README.md` point at the new modules, and the runbook
implementation lines that cite `training/cli.py` handlers follow.
Acceptance: the handler test files pass unchanged, `make help` lists the same verbs, and
each moved verb answers `--help`.

## Roadmap codes without a glossary

The source carries the roadmap's task codes in docstrings and comments (`D6`, `F3`,
`P2-D4`, `P4-D1` and the rest; several hundred references across `training/` and
`interp/`), and the investigation filenames carry them too. They are the only map from a
line of code to the design record that explains it, so rewriting them into prose would
lose information. What is missing is the reverse map: a reader who meets `F3` in a
docstring has no table to decode it without grepping the investigations directory. The
fix is a glossary in [README.md](README.md), one row per code giving the rung it belongs
to and the record that owns it. Cheap, documentation only, and independent of the other
two items; it is parked here only because it is not worth a change of its own and should
ride the next edit of that README.

## Order and gating

1. The glossary, on the next edit of [README.md](README.md).
2. The checkpoint contract consolidation, planned above and decided on 2026-09-19 to
   start after
   [../verification/training/fine-tuning-loop.md](../verification/training/fine-tuning-loop.md)
   has been run once on the workstation. FT.train-resume (train with resume), FT.export-merged (export) and
   FT.organisms (the organism round trip) are the observations the refactor has to preserve, and
   they should exist as recorded output before the code that produced them is
   restructured. The scratch side's equivalent is the D6 acceptance record.
3. The handler moves, planned above, as a second commit in the same change as item 2.

An item leaves this document when its change merges; the document is deleted when the
last one does.
