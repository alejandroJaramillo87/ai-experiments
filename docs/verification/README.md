# Hardware verification runbook

**Operational runbook. Every document beside this one is executed, not read for
understanding: the audience is whoever is sitting at the workstation, and their pasted output
is the evidence. Keep the commands runnable, and the paths accurate; both documentation gates
check this directory, so a stale reference here fails the gate. The exceptions are
[investigations/](investigations/) and [archive/](archive/), which hold frozen records — read,
never executed, and never updated, and exempt from both gates for that reason.**

## How this directory is laid out

One folder per tier, and the folder name is the tier: [training/](training/),
[interp/](interp/), [rust-harness/](rust-harness/),
[observability/](observability/), [hardware/](hardware/),
[sandbox/](sandbox/), [bugs/](bugs/).

Every tier gets a folder, including the ones holding a single document today.
A root mixing folders with loose files is the thing that is actually hard to
scan — the loose files read as leftovers — and a tier with one runbook is the
one most likely to gain a second.

Two files stay at the root, and their being there is the point:

- **[README.md](README.md)**, this index.
- **[step-zero-environment.md](step-zero-environment.md)**, which blocks every
  other phase. It is not a peer of the tiers, so it is not filed among them.

[archive/](archive/) and [investigations/](investigations/) also sit at the
root, because a frozen record belongs with the other frozen records rather
than with the tier it came from.

## What is live and what is frozen

Two different questions, and only one of them a directory can answer.

**`archive/` means never executed again.** A campaign's findings go there when the campaign
is closed and its record is history rather than instruction. Nothing in `archive/` is
maintained; its paths and line numbers are accurate to the day they were written, which is
why both gates skip it.

**A runbook's own progress line says how far along it is.** Every live runbook states
`Entries: N verified, M blocked, K pending, of T` in bold under its front matter.

Finished is not the same as archivable, and conflating the two is the mistake this section
exists to prevent. [step-zero-environment.md](step-zero-environment.md) has every entry
verified and stays live, because it re-runs on every lockfile change and image rebuild — it
is a recurring gate that happens to be passing, not a campaign that ended. Archiving it
would freeze a document someone will execute.

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
closed history lives in [bugs/ledger-archive.md](bugs/ledger-archive.md).

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

## State of the campaign

Each runbook's own progress line is authoritative; this table is the index.

| Runbook | Verified | Blocked | Pending | Of |
|---------|---------:|--------:|--------:|---:|
| [step-zero-environment.md](step-zero-environment.md) | 8 | 0 | 0 | 8 |
| [training/training-phase0.md](training/training-phase0.md) | 4 | 0 | 7 | 11 |
| [training/training-phase1.md](training/training-phase1.md) | 0 | 0 | 13 | 13 |
| [training/training-phase2.md](training/training-phase2.md) | 0 | 0 | 7 | 7 |
| [interp/interp.md](interp/interp.md) | 10 | 1 | 13 | 24 |
| [observability/observability.md](observability/observability.md) | 2 | 0 | 7 | 9 |
| [sandbox/sandbox-executable.md](sandbox/sandbox-executable.md) | 2 | 0 | 0 | 2 |
| [rust-harness/rust-harness.md](rust-harness/rust-harness.md) | 43 | 0 | 1 | 44 |
| [rust-harness/rust-harness-ux.md](rust-harness/rust-harness-ux.md) | 3 | 0 | 9 | 12 |
| [hardware/hardware-stress.md](hardware/hardware-stress.md) | 0 | 0 | 6 | 6 |
| [interp/reference-captures.md](interp/reference-captures.md) | 0 | 0 | 2 | 2 |

A pending entry is not the same as a blocked one, and the difference is whether
a command exists that changes the situation. An entry waiting on a GPU slot, on
free disk, or on an artifact something else in this directory regenerates is
`PENDING` — someone can run it. `BLOCKED` is for what no command here can fix:
absent hardware support, or a checkpoint nobody has. The observability entries
needing `verify-small` shards are the worked example: they were pending on an
artifact rather than on hardware, and
[training/training-phase0.md](training/training-phase0.md) P0.1 is the command that produces
it. The shards were confirmed present on the workstation 2026-08-14, so those
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

Step zero blocks everything else; later phases assume the phases before them
passed.

1. [step-zero-environment.md](step-zero-environment.md) — lockfile
   regeneration, image builds, smoke tests, the quality gate, and the
   hardware smoke suite. Blocks all other phases.
2. [training/training-phase0.md](training/training-phase0.md) — Phase 0 deliverables: data
   preparation, tokenizer, memory fit, recipes, the reference loop with
   tolerance-based resume, export and serve round trip, staging, watchdogs.
3. [training/training-phase1.md](training/training-phase1.md) — F1 through F8: SFT, teacher
   generation, distillation, merge and export, the evaluation wrappers, and
   scorecard promotion.
4. [training/training-phase2.md](training/training-phase2.md) — P2-D1 through P2-D5: the
   full-scale data pipeline, mixtures, WSD schedules, the replay buffer, and
   optimizer offload.
5. [interp/interp.md](interp/interp.md) — the interpretability tier: workbench,
   Neuronpedia harness, write-back path (W1-W4), organisms, harvesting, and
   SAE training.
6. [observability/observability.md](observability/observability.md) — the observability stack (phases
   A-D): GPU watchdog samples, NVML context, the `/mnt/ai-data` log root,
   external sinks, the browser harness, and the live dashboard against an
   in-flight run.
7. [rust-harness/rust-harness.md](rust-harness/rust-harness.md) — the rust-harness CPU backend on
   the real E4B checkpoint: architecture reconciliation, generation, KV delta
   prefill, flight recorder, rr, and the in-band interp chain. Pure cargo
   binary: the Python hardware smoke gate does not apply; entry RH.0 is
   this phase's gate. Its sandbox-executable sibling is
   [rust-harness/rust-harness-qa.md](rust-harness/rust-harness-qa.md), the code-quality campaign
   (phases Q0-Q6) a Claude Code session runs with no GPU or model data. A third document,
   [rust-harness/rust-harness-ux.md](rust-harness/rust-harness-ux.md) (entries `RHX.n`), covers
   the user surface the `U` campaign built — the run header, the progress lines, the remedy
   lines, the run records, and colour at a terminal. It is separate because its entries verify
   what a person sees rather than whether the backend computes the right numbers, and because
   almost all of that surface is already pinned in the sandbox: what is left needs real weights,
   real durations, or a real TTY.

Outside the numbered order, and outside this directory's operational contract:
[investigations/gemma-4-conformance_investigation.md](investigations/gemma-4-conformance_investigation.md),
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

Also outside the numbered order: [interp/reference-captures.md](interp/reference-captures.md), the two
multi-hour activation captures that feed P4-D3 SAE training. They are separated from
[interp/interp.md](interp/interp.md) because each holds the GPU for most of a day and most of the
`/mnt/ai-data` volume, so they are scheduled rather than run in sequence. Their preconditions
are I.9 and I.9b, chunked capture and its crash recovery, verified at smoke
scale on 2026-08-08.

Also outside the numbered order: [hardware/hardware-stress.md](hardware/hardware-stress.md), the
one-time firmware stability soak for the BIOS undervolt profile. It is
independent of the software phases and may run any time its own
preconditions are met (no other GPU consumer active), but it must be
VERIFIED before the first long training run (training-phase2 onward).

## Session-start gate

Before every verification session, run the hardware smoke tier (under five
minutes, no model downloads, temporary-directory writes only):

```bash
uv run pytest -m hw_smoke tests/hardware/
```

Per phase, and after any environment change (driver, image, or lockfile),
run the full tier:

```bash
uv run pytest -m hw_full tests/hardware/
```
