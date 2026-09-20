# Closed verification records

The runbooks, campaign logs and ledger history the verification campaign has closed:
read for what was observed on a date, never executed again and never updated.

Every page here is a frozen record. The gates that check the live runbooks skip these
pages by path (`archive` in `EXEMPT_PARTS` of `scripts/docs/check_doc_refs.py`), which is
why a closed runbook moves here rather than staying beside the live ones: its paths and
line numbers are accurate to the day it closed. What each one verified is stated in its
class line, along with the test that keeps it verified where one exists.

## Records

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [from-scratch-loop.md](from-scratch-loop.md) | The from-scratch loop is the third rung of the training tier: the D6 reference loop (`make train`) driving a scratch model over prepared shards, the rung that will run the `pretrain-150m` and `pretrain-1b` recipes. |
| [gemma-4-conformance_investigation.md](gemma-4-conformance_investigation.md) | Reconciliation of the `rust-harness`, `training`, and `interp` tiers against [docs/models/gemma-4.md](../models/gemma-4.md), the Gemma 4 architecture reference. |
| [ledger-archive.md](ledger-archive.md) | Defect rows moved out of [bugs/known-bugs.md](../bugs/known-bugs.md) when the ledger was reset on 2026-08-08. |
| [rust-harness-qa-campaigns.md](rust-harness-qa-campaigns.md) | The findings, dispositions and results of the two completed runs of the [rust-harness QA campaign](../rust-harness-qa.md). |
| [sandbox-executable.md](sandbox-executable.md) | Every other runbook in this directory verifies behavior that depends on the workstation: the GPU, the NVMe mount, a real checkpoint. |
| [step-zero-environment.md](step-zero-environment.md) | Blocks all other phases. |
| [training-phase0.md](training-phase0.md) | Phase 0 deliverables as implemented: the container (D1, verified in step zero), data preparation (D2), tokenizer training (D3), memory fit (D4), recipe validation (D5), the reference loop with tolerance-based resume (D6), the export and serve round trip including the restore path (D7), staging and prefetch (D9), and passive watchdog observation (D10). |
| [training-phase1.md](training-phase1.md) | F1 through F8 as implemented: SFT data preparation, the PEFT recipes, the TRL SFT entry point, teacher generation, distillation (online and cached), adapter merge and export, the evaluation wrappers, and scorecard promotion. |
| [training-phase2.md](training-phase2.md) | P2-D1 through P2-D5 as implemented: the full-scale data pipeline with decontamination, data mixing, WSD schedules, the replay buffer, and ZeRO-Offload optimizer offload. |

<!-- inventory:pages:end -->

## Related documentation

- [../README.md](../README.md) — the live campaign, its runbook state and its protocol
- [../bugs/known-bugs.md](../bugs/known-bugs.md) — the live ledger these records' KB
  ids came from; the closed rows are in `ledger-archive.md` here
