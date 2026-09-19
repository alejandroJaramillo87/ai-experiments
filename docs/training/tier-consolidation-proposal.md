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

## The inline handlers in the CLI module

`training/cli.py` holds seven handlers inline (`memfit`, `calibrate`, `train`,
`train-sft`, `export`, `eval-holdout`, `eval-lmeval`) while every other verb lives in
`training/commands/`. Nothing is duplicated; the cost is that a reader looking for a verb
has two places to look, and the module is the longest in the tier. Moving them is
mechanical, but each move changes the import path the handler tests use, so it should
ride the checkpoint consolidation above (which touches `train` and `train-sft` anyway)
rather than land alone.

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
2. The checkpoint contract consolidation, after
   [../verification/training/fine-tuning-loop.md](../verification/training/fine-tuning-loop.md)
   has been run once on the workstation. FT.3 (train with resume), FT.5 (export) and
   FT.7 (the organism round trip) are the observations the refactor has to preserve, and
   they should exist as recorded output before the code that produced them is
   restructured. The scratch side's equivalent is the D6 acceptance record.
3. The handler moves, in the same change as item 2.

An item leaves this document when its change merges; the document is deleted when the
last one does.
