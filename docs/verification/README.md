# Hardware verification runbook

**Operational runbook. Every document beside this one is executed, not read for
understanding: the audience is whoever is sitting at the workstation, and their pasted output
is the evidence. Keep the commands runnable, and the paths accurate; both documentation gates
check this directory, so a stale reference here fails the gate. The exception is
[archive/](archive/), which holds the frozen records — read, never executed, and never
updated, and exempt from both gates for that reason.**

The manual verification campaign: the state of every runbook, the defect ledger the
entries cross-reference, and the protocol they are executed and recorded under.

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [protocol.md](protocol.md) | The rules every runbook in this directory is executed and recorded under, stated once. |

<!-- inventory:pages:end -->

## How this directory is laid out

One folder per tier, and the folder name is the tier; the ledger and the frozen records
each have a folder of their own. Every tier gets a folder, including the ones holding a
single runbook today: a root mixing folders with loose files is the thing that is hard
to scan, and a tier with one runbook is the one most likely to gain a second.

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [archive/](archive/README.md) | The runbooks, campaign logs and ledger history the verification campaign has closed: read for what was observed on a date, never executed again and never updated. |
| [bugs/](bugs/README.md) | The live ledger of defects the hardware verification campaign has hit, indexed by the runbook step that surfaces each one, with its fix state. |
| [hardware/](hardware/README.md) | The runbook for the workstation's firmware stability gate: the BIOS undervolt and overclock stack under sustained load, executed once and re-executed when the settings change. |
| [interp/](interp/README.md) | The runbook for the interp tier on the workstation: the workbench image, activation harvesting, probes, sparse autoencoders and the write-back path on the rust harness, entry by entry. |
| [observability/](observability/README.md) | The runbook for the observability stack on the workstation: the logging core, tracing, run records and the dashboards, phase by phase. |
| [rust-harness/](rust-harness/README.md) | The two runbooks for `rust-harness`: the CPU-backend verification executed against the workstation, and the quality-assurance campaign a contained agent session runs without a GPU. |
| [training/](training/README.md) | The runbook for the training tier on the workstation, organised by capability: the fine-tuning loop from the container smoke through the train-and-serve round trip. |

<!-- inventory:dirs:end -->

One file stays at the root: **[README.md](README.md)**, this index. A frozen record
belongs with the other frozen records rather than with the tier it came from, which is
why [archive/](archive/README.md) sits here.

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

## State of the campaign

Each runbook's own progress line is authoritative; this table is the index.

| Runbook | Verified | Blocked | Pending | Of |
|---------|---------:|--------:|--------:|---:|
| [training/fine-tuning-loop.md](training/fine-tuning-loop.md) | 10 | 0 | 4 | 14 |
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

`training/fine-tuning-loop.md` added two kinds on 2026-09-25. FT.train-resume,
FT.export-merged, FT.heldout-eval and FT.eval-ft each verified by hand and then
gained a `hw_full` hardware mirror holding what the hand run observed; the mirror's
first run is recorded in the entry, and the tier re-proves the entry after
any environment change. And a failure that is host state rather than a tree
defect, a root-owned directory or a venv synced before its lock, is recorded
in the entry with the remedy and no ledger row: the ledger names code that
misbehaved, and the runbook names what the operator had to do.

An entry whose stages run for hours is driven by a script, not pasted stage
by stage. FT.organisms was the first, on 2026-09-26: its command block is
`scripts/verification/ft-organisms.sh`, and `make verify ENTRY=ft-organisms`
runs it, brings each stage's services up and down, skips a stage whose
output exists so a failed run resumes, and tees the whole output to a log
under `/mnt/ai-data/logs/verification/`. The log is the paste; the entry
records it verbatim as it would a hand run, and the hardware test that
follows mirrors the script. `tests/docs/test_verify_scripts.py` holds every
script to an entry that runs it and every such citation to a script.
