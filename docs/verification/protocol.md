# Verification protocol

**Operational. The rules every runbook in this directory is executed and recorded under:
who runs what, what counts as evidence, the status vocabulary, how entries are named and
ordered, and the gate a session starts with.**

The rules every runbook in this directory is executed and recorded under, stated once.
A runbook's own text says what it verifies;
this page says how an entry is run, recorded and closed, and in what order the runbooks
are executed.

## Recording protocol

- The human runs each host command on the workstation and pastes the command
  plus its output verbatim into the entry's Observed block.
- A feature is verified only when its stated expected signal is observed.
  Close the entry by changing `Status: PENDING` to `Status: VERIFIED` (or
  `Status: FAILED` with the failure output kept in place).
- Four status values, and no others: `PENDING`, `VERIFIED`, `FAILED`,
  `BLOCKED`. **`BLOCKED` means the entry cannot run on this hardware or with
  the artifacts that exist** — a missing checkpoint, a tool that does not
  support this CPU — and it is not a failure of the thing under test. A
  blocked entry states what would unblock it. Do not invent a fifth value:
  the vocabulary is what makes the progress line at the top of each runbook
  countable.
- Each runbook states its own tally in bold under the front matter, in the
  form `Entries: N verified, M blocked, K pending, of T`. Update it in the
  same edit that closes an entry.
- Do not edit expected signals to match observations. A mismatch is a
  finding: either the code has a bug or the runbook entry was wrong, and
  either way the discrepancy is recorded before it is resolved.
- One GPU consumer at a time: no entry runs while another GPU job,
  benchmark, or inference service the entry does not name is active.

## Entry identifiers

An entry id names an entry; it does not place it. The two are held apart
because a number that reads as an order stops being one the first time a
section moves, and sections move whenever a dependency is found out of order.

- The id is `PREFIX.slug`: the runbook's prefix (`FT`, `I`), a dot, and a
  lowercase slug of at most four hyphenated words naming what the entry
  proves, never its position or date (`FT.calibrate-qlora`,
  `I.harvest-crash-recovery`).
- An id is assigned once, kept when the title changes, and never reused. An
  entry that leaves is listed on the runbook's `Retired:` line under the
  tally, with the date and reason, and stays citable in the ledger as
  `<id> (retired)`.
- The Table of Contents is the execution order and is a numbered list with
  `1.` on every row, so the rendered position is derived and an insertion
  renumbers nothing. The section order matches it exactly; a child entry is
  a nested numbered row. A cross-reference cites the id, never the position.
- A dependency sentence in an entry's opening paragraph (`depends on
  FT.preconditions`) names entries that appear earlier in the file.
- Numeric ids (`RH.24`, `OBS-V10`, `HS.3`) are the legacy shape: accepted by
  the gate, not used for a new entry or a new runbook. Converting a runbook
  adds its old ids to `scripts/docs/check_retired_names.py`, which is what
  made the fine-tuning and interp conversions of 2026-09-19 self-policing.

`scripts/docs/check_runbook_ids.py` enforces all of this in `make quality`.

## Phase order

Later phases assume the phases before them passed.

1. [training/fine-tuning-loop.md](training/fine-tuning-loop.md) — the fine-tuning loop,
   every entry in section order: preconditions, the image and its
   smoke, the smoke dataset, the recipe gates, the GPU step on E2B,
   SFT with resume, the
   merged HF export loaded by the harness, the evaluators, the null and taboo
   organisms, and capture, probe and steering
   on the organism. The training set
   was reorganised by capability on 2026-09-18; the three phase runbooks it
   replaced are frozen under [archive/](archive/) (`training-phase0.md`,
   `training-phase1.md`, `training-phase2.md`), and their four verified
   entries are the substrate facts FT.preconditions cites. The training tier's
   progression is three rungs, and this runbook is the first: fine-tuning,
   then continual pretraining (the full-scale data pipeline, mixtures, WSD
   schedules, the replay buffer, optimizer offload), then training from
   scratch (the D6 reference loop on the `pretrain-150m` and `pretrain-1b`
   recipes). The from-scratch runbook opened and closed on 2026-09-19 with the
   Phase 0 acceptance checks the frozen phase-0 record left pending (the
   reference loop end to end, D9 staging, prefetch, affinity and the lock,
   and the D10 watchdog paths); it is
   [archive/from-scratch-loop.md](archive/from-scratch-loop.md), and its
   hardware mirror runs in the `hw_full` tier. The continual-pretraining
   runbook is not yet written, because a runbook with zero entries is a lie
   to the counts gate; it is created
   under `training/` beside these with its first runnable entry. A
   distillation runbook existed for part of 2026-09-18 and left with the
   feature the same day; what the tier is for, and what waits behind the
   three rungs, is
   [../training/continuous-learning-proposal.md](../training/continuous-learning-proposal.md).
2. [interp/interp.md](interp/interp.md) — the interpretability tier: workbench,
   write-back path (W1-W4: control vectors, probe capture, monitor and export,
   the served-request log), organisms, harvesting with the two reference
   captures, and SAE training. The
   probe surface is also fed by the rust harness `activation-capture` chain
   (RH.9a-9e), which is that runbook's evidence, not this one's.
3. [observability/observability.md](observability/observability.md) — the observability stack (phases
   A-D): GPU watchdog samples, NVML context, the `/mnt/ai-data` log root,
   external sinks, the browser harness, and the live dashboard against an
   in-flight run.
4. [rust-harness/rust-harness.md](rust-harness/rust-harness.md) — the rust-harness CPU backend on
   the real E4B checkpoint: architecture reconciliation, generation, KV delta
   prefill, flight recorder, rr, and the in-band interp chain. Pure cargo
   binary: the Python hardware smoke gate does not apply; entry RH.0 is
   this phase's gate. Its sandbox-executable sibling is
   [rust-harness/rust-harness-qa.md](rust-harness/rust-harness-qa.md), the code-quality campaign
   (phases Q0-Q6) a Claude Code session runs with no GPU or model data. The user-surface
   entries (`RHX.n`) the `U` campaign built, the run header, the progress lines, the remedy
   lines, the run records, and colour at a terminal, are appended to the same runbook after
   its coverage roster.

Outside the numbered order, and outside this directory's operational contract:
[archive/gemma-4-conformance_investigation.md](archive/gemma-4-conformance_investigation.md),
the 2026-08-01 audit of the three consumer tiers against the Gemma 4
reference. It is a **frozen record**, not a runbook — a document audit whose
findings quote the tree as it stood on that day, so its stale paths are
history rather than drift and it is never updated. Read its Limitations
section before treating any `GC-nnn` finding as settled: only 59 of 248
confirmed findings were adversarially verified.

**Its 50 `GH.n` host checks were never executed, and freezing the record does
not retire them.** Every one is blocked on the same precondition — no Gemma 4
weights exist on this workstation — and the agent sandbox cannot reach
`huggingface.co`. They are recorded here so the outstanding work survives
outside a document that is closed to edits: when a checkpoint is acquired,
those 50 entries are the Gemma 4 conformance backlog, and executing them
means opening a new record rather than editing the frozen one.

Also outside the numbered order: [hardware/hardware-stress.md](hardware/hardware-stress.md), the
one-time firmware stability soak for the BIOS undervolt profile. It is
independent of the software phases and may run any time its own
preconditions are met (no other GPU consumer active), but it must be
VERIFIED before the first long training run (the continual-pretraining loop onward).

## Session-start gate

Before every verification session, run the hardware smoke tier (under five
minutes, no model downloads, temporary-directory writes only):

```bash
uv run --no-sync pytest -m hw_smoke tests/hardware/
```

Per phase, and after any environment change (driver, image, or lockfile),
run the full tier. Both lines carry `--no-sync`: a plain `uv run` resyncs the
default groups first, and on a host without a prebuilt `flash-attn` wheel that
is a source build on every core (the from-scratch runbook's FS.3 attempt):

```bash
uv run --no-sync pytest -m hw_full tests/hardware/
```
