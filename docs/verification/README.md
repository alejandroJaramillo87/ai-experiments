# Hardware verification runbook

**Operational runbook. Every document beside this one is executed, not read for
understanding: the audience is whoever is sitting at the workstation, and their pasted output
is the evidence. Keep the commands runnable, and the paths accurate; both documentation gates
check this directory, so a stale reference here fails the gate. The exception is
[archive/](archive/), which holds the frozen records — read, never executed, and never
updated, and exempt from both gates for that reason.**

## How this directory is laid out

One folder per tier, and the folder name is the tier: [training/](training/),
[interp/](interp/), [rust-harness/](rust-harness/),
[observability/](observability/), [hardware/](hardware/),
[bugs/](bugs/).

Every tier gets a folder, including the ones holding a single document today.
A root mixing folders with loose files is the thing that is actually hard to
scan — the loose files read as leftovers — and a tier with one runbook is the
one most likely to gain a second.

One file stays at the root: **[README.md](README.md)**, this index.

[archive/](archive/) also sits at the root, because a frozen record belongs
with the other frozen records rather than with the tier it came from: the
closed step-zero and sandbox-executable runbooks, the QA campaign logs, the
ledger's closed history and the Gemma 4 conformance audit are there.

## What is live and what is frozen

Two different questions, and only one of them a directory can answer.

**`archive/` means never executed again.** A campaign's findings go there when the campaign
is closed and its record is history rather than instruction. Nothing in `archive/` is
maintained; its paths and line numbers are accurate to the day they were written, which is
why both gates skip it.

**A runbook's own progress line says how far along it is.** Every live runbook states
`Entries: N verified, M blocked, K pending, of T` in bold under its front matter.

Step zero, the environment runbook, closed at 8 of 8 on 2026-09-18 and is
[archive/step-zero-environment.md](archive/step-zero-environment.md). What it enforced is
a standing rule here rather than a document: any change to a dependency group means
`uv lock`, a rebuild of the image the group feeds (`make interp-build` or
`make train-build`), and that image's smoke, recorded in the tier runbook whose image
changed (I.workbench-smoke and I.workbench-build for interp). The sandbox tier, OBS-V5 and OBS-V9, closed at 2 of 2 and
is [archive/sandbox-executable.md](archive/sandbox-executable.md); a future entry that needs
no GPU and no `/mnt/ai-data` opens a new sandbox runbook rather than the frozen one.
The from-scratch loop's Phase 0 acceptance checks, FS.0 to FS.8, closed at 9 of 9 on
2026-09-19 and are [archive/from-scratch-loop.md](archive/from-scratch-loop.md); what
they verified stays verified by `tests/hardware/test_from_scratch_loop.py` in the
`hw_full` tier, and the `pretrain-150m` and `pretrain-1b` runs open a new runbook.

Manual verification campaign for the training and interpretability tiers on
the workstation (RTX 5090, AMD Ryzen 9950X). Both tiers passed unit tests in
the sandbox but have never been exercised on the target hardware; this
runbook is the campaign that closes that gap. Every command derives from the
Makefile and the two CLIs (`python -m training`, `python -m interp`); the
runbook covers only features that exist in the repository.

## Known bugs

[bugs/known-bugs.md](bugs/known-bugs.md) is the ledger. It carries what this campaign
still expects to hit, sorted by how well the expectation is warranted: bugs
confirmed and unfixed, fixes committed but awaiting the one hardware run that
can close them, and the defects the campaign itself found and repaired. The
closed history lives in [archive/ledger-archive.md](archive/ledger-archive.md).

The ledger was reset on 2026-08-08 and is smaller than the campaign's own
history implies. Thirteen rows carried from the WI-14 capstone review had
never been checked against the tree, and a table of unverified claims is
acted on as though it were a list of live defects. They were deleted rather
than carried; their identifiers are retired and listed, and the ledger's
front matter records the reasoning. KB identifiers did not change.

Runbook steps carry one of two blocks. **Known bugs likely to surface here**
names a defect to expect: if the step fails in the way the row describes,
record it and fix against the real traceback rather than re-diagnosing from
scratch. **Fix to confirm here** is the opposite prediction — the repair is
already committed, and this run is the evidence that closes it. A step
carrying neither block has no standing prediction, which is not the same as
a guarantee that it works.

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

## State of the campaign

Each runbook's own progress line is authoritative; this table is the index.

| Runbook | Verified | Blocked | Pending | Of |
|---------|---------:|--------:|--------:|---:|
| [training/fine-tuning-loop.md](training/fine-tuning-loop.md) | 4 | 0 | 12 | 16 |
| [interp/interp.md](interp/interp.md) | 8 | 2 | 17 | 27 |
| [observability/observability.md](observability/observability.md) | 2 | 0 | 7 | 9 |
| [rust-harness/rust-harness.md](rust-harness/rust-harness.md) | 55 | 1 | 3 | 59 |
| [hardware/hardware-stress.md](hardware/hardware-stress.md) | 0 | 0 | 6 | 6 |

A pending entry is not the same as a blocked one, and the difference is whether
a command exists that changes the situation. An entry waiting on a GPU slot, on
free disk, or on an artifact something else in this directory regenerates is
`PENDING` — someone can run it. `BLOCKED` is for what no command here can fix:
absent hardware support, or a checkpoint nobody has. The observability entries
needing `verify-small` shards are the worked example: they were pending on an
artifact rather than on hardware, and
[archive/training-phase0.md](archive/training-phase0.md) P0.1 is the command that produced
it (frozen, but the command still runs). The shards were confirmed present on the workstation 2026-08-14, so those
entries are now waiting only on a GPU slot.

`rust-harness.md`'s RH.13 was pending until 2026-09-06 for a reason worth
distinguishing from the usual one: it was verified, and then the code under
it changed what it prints. An entry whose Observed block no longer shows
everything the command produces is not verified, and the repair is a re-run
rather than an edit. The re-run closed it, and it is now also a hardware
test, so the next such change fails `make quality-full HW=full` instead of
waiting for someone to notice. RH.11 was a third thing again until 2026-09-05:
it needed an instruction-tuned checkpoint, which no amount of sitting down
produces, and was `BLOCKED` until gemma-4-E2B-it arrived. It closed on that
checkpoint the next day. RH.15 was a fourth kind on 2026-09-06: the command
was right and both services were up, and the judge failed three times on the
model's own habit of reasoning before it answers, until the request said
exactly what it would accept.

## Phase order

Later phases assume the phases before them passed.

1. [training/fine-tuning-loop.md](training/fine-tuning-loop.md) — the fine-tuning loop,
   every entry in section order: preconditions, the image and its
   smoke, memory fit, the smoke dataset, the recipe gates, calibration on E2B
   and on E4B, SFT with resume, the
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
