# Training implementation investigations

One record per training deliverable, written while that deliverable was being implemented.
Each states what was decided, what was measured, and which tradeoffs were accepted, at the
time the code was written.

All 20 records here were written between 2026-07-14 and 2026-07-15.

## These are frozen records

Every document in this directory carries the class line
`.styles/DOCUMENTATION_STYLE.md` defines for a frozen record: *what was decided or measured
on a day*, whose obligation is **never edit the body; drift is expected and correct**.

That obligation is the opposite of the one the living documents in the parent directory
carry. A path, a line number, or a parameter value quoted here describes the tree as it stood
on the record's date. When it disagrees with the current tree, the record is not wrong and is
not to be corrected — the disagreement is the history. If you need to know how something works
now, read the living document, not the record.

## Naming

`<deliverable-id>_<subject>_investigation.md`, lowercase with underscores — for example
`d4_memory_fit_investigation.md`. The deliverable id is the roadmap identifier (`d` for the
Phase 0 substrate, `f` for Phase 1 fine-tuning and distillation, `p2_d` for Phase 2); one
record has no id because it spans the tier.

**The `_investigation.md` suffix is load-bearing, not decorative.** Both documentation gates
exempt frozen records by exactly two patterns: a filename ending `_investigation.md`, or an
`investigations/` path segment. A record named anything else is gated as though it were
living, and the gate will demand its historical paths still resolve — which for a document
that is never updated is a demand that cannot be met.

## Records

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [d10_operational_watchdogs_investigation.md](d10_operational_watchdogs_investigation.md) | Comparison of the Phase 0 D10 design (roadmap document, "D10: Operational watchdogs") against the current repository state, with a work breakdown, acceptance checks, deviations, and open questions. |
| [d3_tokenizer_training_investigation.md](d3_tokenizer_training_investigation.md) | Comparison of the Phase 0 D3 design (roadmap document, "D3: Tokenizer training") against the current repository state, with a work breakdown, acceptance checks, deviations, and open questions. |
| [d4_memory_fit_investigation.md](d4_memory_fit_investigation.md) | Comparison of the Phase 0 D4 design (roadmap document, "D4: Memory-fit calculator") against the current repository state, with a work breakdown, acceptance checks, deviations, and open questions. |
| [d5_recipe_system_investigation.md](d5_recipe_system_investigation.md) | Comparison of the Phase 0 D5 design (roadmap document, "D5: Recipe system") against the current repository state, with a work breakdown, acceptance checks, deviations, and open questions. |
| [d6_tracking_checkpointing_investigation.md](d6_tracking_checkpointing_investigation.md) | Comparison of the Phase 0 D6 design (roadmap document, "D6: Tracking and resumable checkpointing") against the current repository state, with a work breakdown, acceptance checks, deviations, and open questions. |
| [d7_train_serve_investigation.md](d7_train_serve_investigation.md) | Comparison of the Phase 0 D7 design (roadmap document, "D7: Train-and-serve round trip") against the current repository state, with a work breakdown, acceptance checks, deviations, and open questions. |
| [d9_heterogeneous_compute_investigation.md](d9_heterogeneous_compute_investigation.md) | Comparison of the Phase 0 D9 design (roadmap document, "D9: Heterogeneous-compute skeleton") against the current repository state, with a work breakdown, acceptance checks, deviations, and open questions. |
| [f1_sft_data_investigation.md](f1_sft_data_investigation.md) | Comparison of the Phase 1 F1 design (product design document, "F1: Chat and instruction data preparation") against the current repository state, with a work breakdown, acceptance checks, deviations, and open questions. |
| [f3_sft_entry_point_investigation.md](f3_sft_entry_point_investigation.md) | Comparison of the Phase 1 F3 design (roadmap document, "F3: Supervised fine-tuning byb entry point") against the current repository state, with a work breakdown, acceptance checks, deviations, and open questions. |
| [f4_teacher_synthetic_data_investigation.md](f4_teacher_synthetic_data_investigation.md) | Comparison of the Phase 1 F4 design (roadmap document, "F4: Teacher and synthetic data harness") against the current repository state, with a work breakdown, acceptance checks, deviations, and open questions. |
| [f5_distillation_investigation.md](f5_distillation_investigation.md) | Investigation of what implementing the F5 distillation harness entails, reconciling the scope document against the current state of the repository. |
| [f6_adapter_merge_serve_investigation.md](f6_adapter_merge_serve_investigation.md) | Investigation of what implementing F6 (adapter merge and serve) entails, reconciling the scope document against the current state of the repository. |
| [f7_eval_wrapper_investigation.md](f7_eval_wrapper_investigation.md) | Investigation of what implementing F7 (fine-tuning evaluation wrapper) entails, reconciling the scope document against the current state of the repository. |
| [f8_recipe_cards_investigation.md](f8_recipe_cards_investigation.md) | Investigation of what implementing F8 (recipe cards) entails, reconciling the Phase 1 product design against the current state of the repository. |
| [p2_d1_full_scale_data_pipeline_investigation.md](p2_d1_full_scale_data_pipeline_investigation.md) | Investigation of what implementing P2-D1 (full-scale data pipeline) entails, reconciling the Phase 2 product design against the current state of the repository. |
| [p2_d2_data_mixing_configuration_investigation.md](p2_d2_data_mixing_configuration_investigation.md) | Investigation of what implementing P2-D2 (data-mixing configuration) entails, reconciling the Phase 2 product design against the current state of the repository. |
| [p2_d3_warmup_stable_decay_investigation.md](p2_d3_warmup_stable_decay_investigation.md) | Investigation of what implementing P2-D3 (warmup-stable-decay schedule) entails, reconciling the Phase 2 product design against the current state of the repository. |
| [p2_d4_replay_buffer_investigation.md](p2_d4_replay_buffer_investigation.md) | Investigation of what implementing P2-D4 (replay buffer) entails, reconciling the Phase 2 product design against the current state of the repository. |
| [p2_d5_optimizer_offload_investigation.md](p2_d5_optimizer_offload_investigation.md) | Investigation of what implementing P2-D5 (optimizer offload) entails, reconciling the Phase 2 product design against the current state of the repository. |
| [phase0_completeness_audit_investigation.md](phase0_completeness_audit_investigation.md) | Audit of the ten Phase 0 deliverables (D1 through D10, the training substrate) against five sources at once: the 2026-07-08 product design that scoped them, the living documentation under `docs/training/`, the code under `training/` and `docker/training/`, the test suite, and the verification runbooks. |
| [phase1_completeness_audit_investigation.md](phase1_completeness_audit_investigation.md) | Audit of the eight Phase 1 deliverables (F1 through F8, the fine-tuning harness) against six sources at once: the Phase 1 design that scoped them (the F-numbered specification, written against the 2026-07-08 product design), the living documentation under `docs/training/`, the code under `training/`, the test suite, the fine-tuning runbook, and the consumers in the two later tiers, `interp/` and `rust-harness/`. |
| [unified_cli_investigation.md](unified_cli_investigation.md) | Investigates whether the training and interpretability tiers should unify behind a single CLI, how tightly coupled the two tiers are today, and whether running mechanistic-interpretability workflows during training is a realistic goal on this workstation. |

<!-- inventory:pages:end -->

The table is generated from each record's description paragraph and held to the directory
by `scripts/docs/check_inventory.py`, which is why it can live here although
`scripts/docs/check_doc_refs.py` skips this directory: the inventory gate reads the tree,
not the links. The pairing of each record with the living document it produced is in
[../README.md](../README.md), stated once.

## Related documentation

- [../README.md](../README.md) — the training tier index and the record pairing
- [../../../training/README.md](../../../training/README.md) — the source tree these records
  describe
- [../../verification/](../../verification/) — host-side acceptance records, which are
  evidence from a workstation run rather than implementation-time decisions
