# Hardware verification runbook

Manual verification campaign for the training and interpretability tiers on
the workstation (RTX 5090, AMD Ryzen 9950X). Both tiers passed unit tests in
the sandbox but have never been exercised on the target hardware; this
runbook is the campaign that closes that gap. Every command derives from the
Makefile and the two CLIs (`python -m training`, `python -m interp`); the
runbook covers only features that exist in the repository.

## Known bugs

The WI-14 capstone review and its re-verification confirmed 51 defects.
[known-bugs.md](known-bugs.md) is the ledger: 36 pure-Python defects were
fixed with regression tests in the sandbox pre-fix pass, and 15 remain open —
latent in the GPU, service, and pinned-library paths this campaign is the
first to exercise. Runbook steps with open bugs carry a "Known bugs likely to
surface here" block listing the ones to expect at that step. Treat a listed
bug as the predicted failure: if a step fails in the way its known bug
describes, record it and fix against the real traceback rather than
re-diagnosing from scratch.

## Recording protocol

- The human runs each host command on the workstation and pastes the command
  plus its output verbatim into the entry's Observed block.
- A feature is verified only when its stated expected signal is observed.
  Close the entry by changing `Status: PENDING` to `Status: VERIFIED` (or
  `Status: FAILED` with the failure output kept in place).
- Do not edit expected signals to match observations. A mismatch is a
  finding: either the code has a bug or the runbook entry was wrong, and
  either way the discrepancy is recorded before it is resolved.
- One GPU consumer at a time: no entry runs while another GPU job,
  benchmark, or inference service the entry does not name is active.

## Phase order

Step zero blocks everything else; later phases assume the phases before them
passed.

1. [step-zero-environment.md](step-zero-environment.md) — lockfile
   regeneration, image builds, smoke tests, the quality gate, and the
   hardware smoke suite. Blocks all other phases.
2. [training-phase0.md](training-phase0.md) — Phase 0 deliverables: data
   preparation, tokenizer, memory fit, recipes, the reference loop with
   tolerance-based resume, export and serve round trip, staging, watchdogs.
3. [training-phase1.md](training-phase1.md) — F1 through F8: SFT, teacher
   generation, distillation, merge and export, the evaluation wrappers, and
   scorecard promotion.
4. [training-phase2.md](training-phase2.md) — P2-D1 through P2-D5: the
   full-scale data pipeline, mixtures, WSD schedules, the replay buffer, and
   optimizer offload.
5. [interp.md](interp.md) — the interpretability tier: workbench,
   Neuronpedia harness, write-back path (W1-W4), organisms, harvesting, and
   SAE training.
6. [observability.md](observability.md) — the observability stack (phases
   A-D): GPU watchdog samples, NVML context, the `/mnt/ai-data` log root,
   external sinks, the browser harness, and the live dashboard against an
   in-flight run.

## Session-start gate

Before every verification session, run the hardware smoke tier (under five
minutes, no model downloads, temporary-directory writes only):

```bash
$ uv run pytest -m hw_smoke tests/hardware/
```

Per phase, and after any environment change (driver, image, or lockfile),
run the full tier:

```bash
$ uv run pytest -m hw_full tests/hardware/
```
