# Verification campaign

**Operational — the index to the runbooks, and the protocol they follow. Keep the commands
runnable.**

What has been established, what has not, and how the gap is closed. The standing list below is
the honest inventory: it is maintained as capabilities are added, and adding one without its
caveat is a defect.

## Table of Contents

- [Protocol](#protocol)
- [The runbooks](#the-runbooks)
- [Entry format](#entry-format)
- [What is not verified](#what-is-not-verified)

## Protocol

1. **State the acceptance criteria before running anything.** What output would count as a pass,
   and what would count as a failure. A criterion that cannot fail is not a criterion: "the
   command runs" is not one, "both indexes report READY" is. Deciding afterwards is how an
   ambiguous result becomes a pass.
2. **Confirm the code under test is the code that will run.** `git status --short` before
   trusting any result — uncommitted work does not reach a new worktree, and a run against stale
   source reports success for a fix that is not present.
3. **Run the command on the machine that can answer it.** See
   [../operations/three-machines.md](../operations/three-machines.md).
4. **Paste the command and its output verbatim** into the entry's Observed block. Not a summary —
   a summary is where a failure becomes a caveat. Where output is long, quote the lines bearing
   on the criteria in full rather than compressing them.
5. **Judge each criterion separately.** State plainly when one did not get exercised: a check
   that never ran is neither a pass nor a failure, and reporting it as either is the error this
   protocol exists to prevent.

An agent may not mark an entry `VERIFIED` on the strength of a command it could not run. Print
the command, stop, and wait for the operator's output. This mirrors
`.claude/skills/verify-host/SKILL.md` in the surrounding repository, which owns the handshake in
full.

## The runbooks

| Phase | Machine | Runbook | Entries |
|-------|---------|---------|---------|
| A, B, C | Cloud sandbox | Covered by `make quality` and the corpus tier; no runbook | — |
| D | Linux desktop | [desktop-deployment.md](desktop-deployment.md) | D1-D5 |
| D | Linux desktop | [desktop-search.md](desktop-search.md) | D6-D9 |
| D | Linux desktop | [desktop-model.md](desktop-model.md) | D10-D13 |
| D | Linux desktop | [desktop-scale.md](desktop-scale.md) | D14-D16 |
| E | MacBook M4 | [macbook.md](macbook.md) | E1-E5 |

They run in that order, and each is blocked on the one above: there is no point measuring the
latency of a pipeline whose search has never executed, or verifying that an unproven pipeline
was extracted correctly.

## Entry format

Every entry carries the same five parts, and two of them are checked by the gate.

```markdown
### Dn — what is being established

Implementation: <the modules the entry exercises>.

**Criteria:** what output counts as a pass, stated before the command.

**Command:**

    <one copy-pasteable block>

Observed:

_(pending)_

Status: PENDING
```

`Status:` is one of `PENDING`, `VERIFIED`, `BLOCKED`, at column zero. A blocked entry names what
blocks it.

Each runbook states its own tally under the class line:

```markdown
**Entries: 0 verified, 0 blocked, 5 pending, of 5.**
```

**Both are gate-checked.** `scripts/docs/check_doc_counts.py` compares that tally against the
`Status:` lines beneath it, so a runbook cannot disagree with itself about how much is done —
the tally is edited by hand in the edit whose attention is on an entry, which is exactly the
edit that forgets the header. `scripts/docs/check_runbook_entries.py` requires a status line per
entry, and requires an entry marked `VERIFIED` to carry evidence of one of two kinds: pasted
output, or the name of a test that asserts the signal, written as
`` `tests/test_evals.py::test_ids_are_unique` `` and resolved against the tree.

Neither check says an entry is correct. They say it is not obviously hollow, which is a lower
bar than it sounds and is the one an entry somebody meant to come back to fails.

## What is not verified

Stated because the alternative is a reader assuming coverage that does not exist.

**No container in this stack has ever started.** The compose file, both MongoDB configurations,
and the bootstrap script are written against the manual's own containerised deployment procedure
and parse cleanly. That is all. D1 is the first time any of it meets a Docker daemon.

**`$vectorSearch` and `$search` have never executed.** They are `mongot` operators rather than
MongoDB ones, `mongomock` implements neither and could not, and that path stays unexercised
until D8. What *is* covered offline is everything around it: the document round trip, `upsert`,
`fetch_by_ids`, and the shape of both index definitions.

**`$firstN` and `$dateTrunc` are unexercised**, and skip by name, because a silent skip is
indistinguishable from a pass. D5 is where they run.

**No bulk write has been executed.** `mongomock` 4.3.0 — the current release — cannot execute
any pymongo 4.17 bulk update, so the operations `insert` builds are asserted through a recording
double and their execution is left to D5 and D14.

**No retrieval quality has been measured against a real embedder.** A lexical baseline exists —
`logtriage/evals/README.md` carries it, measured with the hashing embedder over 40 labelled
questions — and it is the number D11 has to beat. The vectors behind it are hashes, so it
measures word overlap.

**The ground-truth labels are incomplete**, as every hand-built set is. D12 pools and adjudicates
them; until then every recall figure understates every retriever, and unequally.

**Nothing establishes how the model behaves.** Every test uses the scripted client, so a
narrative passing the causality check is evidence about the guard rail and not about the model.
D13 is the first evidence either way.

**Nine of the twelve fixture log ids are unverified, and cannot be verified here.** MongoDB
publishes no catalogue. See [../mongodb/log-ids.md](../mongodb/log-ids.md).

**`shellcheck` is not installed in the sandbox**, so the two shell scripts in the surrounding
repository are checked only by `bash -n`.
