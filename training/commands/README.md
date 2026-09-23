# training/commands

The verbs of `python -m training`, one module each. `cli.py` calls every module's
`configure(sub)` to register its subparser and bind `func` to a `cmd(args) -> int` entry
point; the module keeps its imports light and loads torch, transformers or the harness
inside `cmd()`, so `--help` and argument errors cost nothing and the parser builds in a
sandbox with none of them installed.

Two modules are helpers rather than verbs: `args.py` holds the argparse types the staging
verbs share, and `recipe.py` is the load-and-validate step every recipe-driven verb runs
first, so a bad recipe exits 1 with the schema's message before anything heavy imports.
Five verbs are not here: `train`, `train-sft`, `export`, `eval-holdout` and `eval-lmeval`
are registered inline in `cli.py` and hand straight to `runtime/`, `tune/`, `export/` and
`eval/`.

The root [README](../README.md) maps every verb to its `make` target and the container it
runs in. What consumes this directory from outside: `interp/commands/organism.py` imports
`generate_teacher` directly, so a change to its argument surface is an interp change too.

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `args.py` | Shared argparse helpers for training subcommands. |
| `build_contamination.py` | Build the benchmark decontamination index (P2-D1). |
| `download.py` | Stage a model into the persistent models tree (Prelude C). |
| `eval_ft.py` | Fine-tuning evaluation driver (Phase 1 F7). |
| `generate_teacher.py` | Teacher corpus generation (Phase 1 F4): the organism generator's transport. |
| `make_replay_synthetic.py` | Synthetic replay shard set generator (P2-D4, resolved decision 3). |
| `preflight.py` | Prove every asset a recipe needs is already on disk (Prelude C). |
| `prepare_corpus.py` | Full-scale corpus preparation (P2-D1). |
| `prepare_data.py` | Tokenize a dataset into memory-mapped token shards (Phase 0 D2). |
| `prepare_sft.py` | Prepare a chat/instruction dataset into tokenized, masked SFT examples (Phase 1 F1). |
| `recipe.py` | The load-and-validate step every recipe-driven subcommand runs first. |
| `report.py` | Render a run's metrics and logs into a dashboard (OBS-11, OBS-12). |
| `runs.py` | List recent runs from the observability catalog (roadmap OBS-9). |
| `smoke_test.py` | Training container smoke test (Phase 0 D1). |
| `train_step.py` | Step a recipe on the GPU and report what it held (the successor of the memory-fit calculator's calibrate loop, 2026-09-20). |
| `train_tokenizer.py` | Train and evaluate a byte-level BPE tokenizer (Phase 0 D3). |
| `validate_recipe.py` | Load, override, and validate a training recipe (Phase 0 D5). |

<!-- inventory:modules:end -->

Tests: `tests/training/test_cli_handlers_*.py`, one file per verb group, driving each
handler in process through the real parser.
