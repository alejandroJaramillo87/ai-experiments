# tests/training

The unit and branch tests for the training tier in `training/`: recipes and their
validation, the data pipeline, the training loop and its resume guarantees, the SFT and
export paths, the CLI handlers, and the D9/D10 operational pieces. It is the largest
directory in the suite and the only one whose subject needs torch.

## Torch, and the skip count

The directory is torch-optional by construction, in two ways. The modules whose subject is
a real tensor call `pytest.importorskip` at import time, so on a torch-free environment
they skip instead of failing collection. A few others (`test_head_loss.py`,
`test_offload.py`, `test_peft_utils.py`) install a fake torch into `sys.modules` and run
everywhere. The rest never import it.

That is why a green run here is not automatically a complete run. `tests/conftest.py`
prints the determining fact beside the pytest summary: `[TIER] sandbox:` means torch was
absent and those modules skipped, `[TIER] full:` means the whole directory was in scope.
Read the skip count, and see [../README.md](../README.md) for how to provision the full
tier.

Nothing here allocates GPU memory or loads a published checkpoint. The hardware mirrors of
these tests live in [../hardware/](../hardware/) — `test_fine_tuning_loop.py`,
`test_step_qlora.py`, `test_recipe_gates.py` and their neighbours — behind the `hw_smoke`
and `hw_full` markers, which `addopts` deselects by default.

`support.py` and `conftest.py` carry the shared fixtures: a tiny byte-level BPE tokenizer
and a synthetic JSONL corpus, both built in temporary directories, plus the run-lock and
observability isolation every in-process CLI test needs.

## Inventory

The table below is generated from the first sentence of each module's docstring and the
`def test_` functions it declares. Do not edit it by hand: run

```bash
python scripts/docs/gen_inventory.py tests/training/README.md
```

`make quality` holds it to the files through `scripts/docs/check_inventory.py`, so a
module added, renamed, or given a sharper docstring fails the gate until the table is
regenerated.

<!-- inventory:tests:start -->

| File | Tests | Covers |
|------|-------|--------|
| `test_affinity.py` | 9 | D9 core-pinning helpers: cpu-list parsing, SMT expansion, disjointness, and the sched_setaffinity round trip. |
| `test_architecture.py` | 28 | training/models/architecture.py: the parameter accounting pinned for a hand-computed tiny config, the reference-recipe counts against the published source architectures, the Gemma 4 counts against the exact figures in docs/training/base/gemma-4-training.md, the HF config mapping, and the adapter counts. |
| `test_chat_template_contract.py` | 11 | The Python half of the chat-template contract. |
| `test_checkpoint.py` | 12 | Unit tests for atomic checkpoint save, retention, and resume resolution (Phase 0 D6). |
| `test_cli_handlers_datatools.py` | 10 | CLI handler contracts for the data-tooling subcommands: validate, build-contamination, and make-replay-synthetic. |
| `test_cli_handlers_export.py` | 20 | CLI handler contracts for cmd_export, cmd_eval_holdout, and cmd_eval_lmeval. |
| `test_cli_handlers_preflight.py` | 17 | In-process handler tests for python -m training preflight and download. |
| `test_cli_handlers_prepare.py` | 13 | CLI handler contracts for the data-preparation subcommands: prepare-data, prepare-corpus, and prepare-sft. |
| `test_cli_handlers_step.py` | 15 | In-process handler tests for python -m training step. |
| `test_cli_handlers_teacher.py` | 13 | CLI handler contracts for the generate-teacher subcommand. |
| `test_cli_handlers_tokenizer.py` | 12 | CLI handler contracts for the train-tokenizer subcommand. |
| `test_cli_handlers_train.py` | 13 | CLI handler contracts for the run subcommands: cmd_train and cmd_train_sft. |
| `test_contamination.py` | 10 | Contamination index build and membership checks (P2-D1). |
| `test_data_tail_branches.py` | 33 | Error and hub-source branches of the data tail modules left uncovered by the module tests. |
| `test_entrypoint.py` | 4 | docker/training/entrypoint.sh: the environment echo and the D9 LD_PRELOAD slot. |
| `test_eval_ft.py` | 2 | eval-ft's container phase: the compared sides reach the training container in its own spelling of the models tree. |
| `test_eval_runner.py` | 6 | F7 runner pieces: checkpoint identity and the lm-eval task pins. |
| `test_export.py` | 45 | Tests for the exporter's pure-logic paths. |
| `test_export_common_branches.py` | 13 | Branch coverage for export seams and common probes, all torch-free. |
| `test_export_harness.py` | 4 | The exporter's output, read by the consumer it exists for. |
| `test_head_loss.py` | 13 | training/models/head_loss.py: the chunked head loss (KB-87), driven against a fake torch the way test_peft_utils.py fakes it. |
| `test_holdout.py` | 4 | holdout.resolve_model form dispatch: hub id, full checkpoint dir, and adapter-only checkpoint. |
| `test_lmeval.py` | 4 | lm-eval evaluator helpers: result extraction, the stderr-key composition for filtered metrics, and the version-mismatch guard. |
| `test_loader.py` | 7 | D9 prefetch dataloader: exact sequential window order, wrap-around, index-based resume equivalence with 0 and >0 workers, legacy cursor conversion, and the multi-worker deterministic loop resume — the guarantee the index-based design exists to protect. |
| `test_lock.py` | 9 | D9 run lock: acquire/release, contention, and staleness. |
| `test_loop.py` | 7 | Acceptance tests for the D6 reference loop: deterministic resume and tracking. |
| `test_mixture.py` | 23 | P2-D2 data-mixing tests: draw proportions from source tags, deterministic random access, resume-index handling, strict single-epoch abort, tokenizer-hash refusal at construction and at validate(), and the tagged dataloader path. |
| `test_obs_durability.py` | 2 | SIGKILL durability test for the jsonl metrics sink (roadmap OBS-10). |
| `test_obs_integration.py` | 3 | Cross-module integration tests for the observability stack (phases A-D). |
| `test_obs_metrics.py` | 18 | Unit tests for the metrics.jsonl tracker sink (roadmap OBS-10). |
| `test_offload.py` | 35 | ZeRO-Offload unit tests (Phase 2 P2-D5): DeepSpeed config generation as a pure function of the recipe, the trainable_filter freeze policy on a real tiny HF model, the transformer_stack parameter accounting, and the validate() invariants. |
| `test_peft_utils.py` | 23 | PEFT glue tests (Phase 1 F2): family-registry lookups, target_modules "auto" resolution into the config snapshot, and the trainable-parameter report, and the composite-base exclusion (KB-83). |
| `test_pin_run.py` | 4 | scripts/training/pin_run.sh: the host-side taskset wrapper (D9). |
| `test_pipeline.py` | 11 | Full-scale pipeline stage composition (P2-D1). |
| `test_prepare.py` | 10 | End-to-end tests for the JSONL tokenize-and-shard path (Phase 0 D2). |
| `test_recipe_pins.py` | 5 | Every Gemma 4 recipe pins model.hf.revision to the facts file (operating rule: pin, then fix, then update). |
| `test_recipes.py` | 57 | Recipe system tests (Phase 0 D5): typed loading, override handling, snapshot round trip, and one test per validate() cross-field invariant. |
| `test_recipes_tracking_branches.py` | 29 | Branch coverage for recipe validation and tracking-sink construction. |
| `test_replay.py` | 35 | P2-D4 replay buffer tests: budget subsampling and refusal, RAM-floor refusal against a mocked psutil, permutation uniformity and determinism, mixture integration (fraction accounting from source tags, resume draw equality), the host-RAM replay term, and the synthetic generator CLI. |
| `test_report.py` | 17 | Unit tests for the snapshot dashboard and report subcommand (OBS-11). |
| `test_report_schema.py` | 5 | Report-envelope conformance (WI-12): the shared validator and header helper in common/reports.py. |
| `test_schedules.py` | 9 | P2-D3 acceptance tests: the wsd multiplier curve pinned at hand-computed steps, and tiny runs whose tracked lr shows the three segments with exact curve continuation on resume mid-stable and mid-decay. |
| `test_serving.py` | 18 | Shared serving module: service registry, model path mapping, training-idle preflight, and swap env construction. |
| `test_sft.py` | 21 | SFT data preparation tests (Phase 1 F1): golden mask boundaries per registered family, round-trip detokenization, deterministic artifacts and holdout membership, manifest/tokenizer refusal, and the unknown-family error. |
| `test_sft_entry.py` | 31 | F3 SFT entry point tests: schema/validation, SFTConfig mapping, packing with mask preservation, checkpoint meta/retention/latest helpers, and the preflight sequence. |
| `test_shards.py` | 16 | Unit tests for the shard writer, manifest verification, and mmap reader (Phase 0 D2). |
| `test_smoke_cmd.py` | 6 | training/commands/smoke_test.py: the D1 smoke command, in-process. |
| `test_staged.py` | 7 | KB-79: a recipe's hub id resolves to the directory train-download staged. |
| `test_staging.py` | 8 | D9 RAM-disk staging: cap respected with clean fallback, hash-verified copies, reuse of a valid staged set, corruption detection. |
| `test_teacher.py` | 20 | F4 teacher harness: prompt normalization, stable keys, resumability, retry, pipeline shapes, and manifest discipline. |
| `test_tokenizer.py` | 17 | Phase 0 D3 tokenizer training tests: special-token ids, reserved block, reproducibility, round-trip losslessness, overwrite refusal, and hash-chain integration with the D2 loader. |
| `test_tracking.py` | 3 | D6 tracking fan-out: fail-soft sink eviction and the no-sink contract. |
| `test_training_cli.py` | 5 | Unified training CLI parser tests: each migrated training/commands/ subcommand parses its minimal argument set and binds the module's cmd entry point. |
| `test_watchdog.py` | 18 | Unit tests for the D10 operational watchdogs. |
| `test_watchdog_branches.py` | 16 | Branch tests for the D10 watchdog fail-soft paths. |

<!-- inventory:tests:end -->
