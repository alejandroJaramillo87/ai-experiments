# training/tune

Supervised fine-tuning on the TRL backend. It is the `train-sft` verb's implementation, full
or QLoRA, over the masked dataset `prepare-sft` wrote, and the callback bridge that gives a
TRL run the same checkpoint layout, run lock, tracker and watchdog the reference loop in
`runtime/` has. Packing is done here at training time and never on the held-out split.

The only consumers are `training/cli.py` for the verb and `training/eval/holdout.py`, which reuses the
attention-implementation choice so the evaluator loads the model the way the trainer did.

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `sft.py` | Supervised fine-tuning entry point (Phase 1 F3). |
| `sft_callbacks.py` | TRL/transformers callback bridge for the F3 SFT entry point. |

<!-- inventory:modules:end -->

Living page: [sft_entry_point.md](../../docs/training/fine-tuning/sft_entry_point.md).
Tests: `tests/training/test_sft_entry.py` and `test_obs_metrics.py`; on the workstation,
`tests/hardware/test_step_qlora.py` and `test_fine_tuning_loop.py` under `hw_full`.
