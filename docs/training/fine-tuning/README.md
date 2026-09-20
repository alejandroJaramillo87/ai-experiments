# Fine-tuning

The first rung: supervised fine-tuning of a Gemma 4 base into an organism, a model with a
known implant the interp tier then recovers. The pipeline is `make organism RECIPE=...`
in `make/interp.mk`, which prints its stages; the training half of it is documented here,
the organism half in [../../interp/](../../interp/README.md). Verified by
[../../verification/training/fine-tuning-loop.md](../../verification/training/fine-tuning-loop.md).

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [eval_wrapper.md](eval_wrapper.md) | Runs the two evaluators, held-out loss and lm-evaluation-harness, on a baseline and a fine-tuned checkpoint with one command, and writes one record per evaluator per side (Phase 1 F7). |
| [sft_data.md](sft_data.md) | Turns chat and instruction datasets into tokenized, loss-masked training examples for the supervised fine-tuning trainer (Phase 1 F1). |
| [sft_entry_point.md](sft_entry_point.md) | Operation of the Phase 1 F3 SFT trainer: one command takes a recipe and runs a supervised fine-tune on TRL's `SFTTrainer`, supporting full fine-tuning and QLoRA, wired into the Phase 0 tracking, checkpointing, lock, and watchdog machinery. |
| [teacher_synthetic_data.md](teacher_synthetic_data.md) | Batch-generates teacher responses through the inference services, producing the messages-format JSONL dataset an organism is fine-tuned on (Phase 1 F4). |

<!-- inventory:pages:end -->

In order, one fine-tune runs: `organism-generate` (the teacher corpus), `prepare-sft`
(the tokenized, masked dataset), `train-step` then `train-sft` on
`training/recipes/reference/sft-qlora-e2b.yaml` or an organism recipe, `export-hf`
([../base/train_serve_roundtrip.md](../base/train_serve_roundtrip.md)), `eval-ft`, and the
organism's verify and seal. The reference recipe is the QLoRA card every organism recipe
follows; the Phase 1 audit of 2026-09-20
([../investigations/phase1_completeness_audit_investigation.md](../investigations/phase1_completeness_audit_investigation.md))
records what the harness was designed to be and what of it the rung still calls.
