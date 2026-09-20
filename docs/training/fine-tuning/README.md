# Fine-tuning

The first rung: supervised fine-tuning of a Gemma 4 base into an organism, a model with a
known implant the interp tier then recovers. The pipeline is `make organism RECIPE=...`
in `make/interp.mk`, which prints its stages; the training half of it is documented here,
the organism half in [../../interp/](../../interp/README.md). Verified by
[../../verification/training/fine-tuning-loop.md](../../verification/training/fine-tuning-loop.md).

| Document | Covers |
|----------|--------|
| [sft_data.md](sft_data.md) | Chat and instruction data preparation: the record format, the chat-template registry, the assistant-span mask, the materialized dataset and its held-out split (F1) |
| [sft_entry_point.md](sft_entry_point.md) | The supervised fine-tuning trainer, full and QLoRA, on TRL (F3); the PEFT options of the recipe schema (F2) are in [../base/recipe_system.md](../base/recipe_system.md) |
| [teacher_synthetic_data.md](teacher_synthetic_data.md) | The teacher corpus harness through the inference services, the organism generator's transport (F4) |
| [eval_wrapper.md](eval_wrapper.md) | Held-out loss and lm-eval records on the base and the fine-tune, the locality inputs of `organism-verify` (F7) |

In order, one fine-tune runs: `organism-generate` (the teacher corpus), `prepare-sft`
(the tokenized, masked dataset), `train-step` then `train-sft` on
`training/recipes/reference/sft-qlora-e2b.yaml` or an organism recipe, `export-hf`
([../base/train_serve_roundtrip.md](../base/train_serve_roundtrip.md)), `eval-ft`, and the
organism's verify and seal. The reference recipe is the QLoRA card every organism recipe
follows; the Phase 1 audit of 2026-09-20
([../investigations/phase1_completeness_audit_investigation.md](../investigations/phase1_completeness_audit_investigation.md))
records what the harness was designed to be and what of it the rung still calls.
