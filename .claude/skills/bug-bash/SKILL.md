---
name: bug-bash
description: The method contract for a quality campaign — the phase ladder, the standing dimension list, the severity rubric, what counts as a confirmed finding, and the batch-and-push cadence. Read before running or participating in a bug bash or quality gate.
---
# Bug-bash method contract

The standing method for a deep quality campaign in this repository: many agents hunting
in parallel, adversarial verification before any fix, and remediation in batches that
each end in a pushed commit. Campaign records live under `docs/verification/`.

This document is the shared vocabulary. When a `bug-hunter`, `triage-agent`,
`remediator`, `integrator`, `perf-auditor`, or `skeptic` says "confirmed", "high", or
"package", it means what is written here and nothing else.

> Sandboxed execution: agent sessions cannot reach the host, its GPU, its Docker daemon,
> `/mnt/ai-data`, or the running services. Repo-local work — building, unit tests, scratch
> binaries, grep — runs in the sandbox. Anything host-side is prepared as an exact command
> for the human, and the reported output is the evidence.

## Phase ladder

Run the phases in order. Do not start remediation while discovery is still open.

1. **Baseline and oracles.** Confirm the project gate exits 0 before anything changes.
   Record determinism oracles — a digest of a recorded output stream and of a generated
   artifact, both at a fixed seed — so later phases can prove they changed nothing.
   Record the perf baseline and its noise floor.
2. **Scope and exclusions.** List the dimensions in play and the file set each one owns.
   **Write this campaign's exclusion list into the campaign log under `docs/verification/`
   before spawning any finder**, so the finders and the triage-agent read the same one; an
   exclusion list held only in the orchestrator's context is not readable by either.
   Re-grade agent memory in this phase too: every memory file carries `verified:` and
   `stale_when:`, and any whose trigger has fired is re-checked against the tree and
   rewritten before it is used. A memory is written in the imperative, so a stale one does
   not go quiet — it suppresses real findings.
3. **Discovery.** One `bug-hunter` per dimension, file sets disjoint where possible.
   Finders do not fix and do not propose fixes.
4. **Verification.** `skeptic` agents attempt to refute each finding, weighted by
   severity. See the rubric below.
5. **Triage.** One `triage-agent` deduplicates, regrades, audits verification state, and
   cuts the confirmed set into file-disjoint packages.
6. **Remediation.** One `remediator` per package, working one shared tree. Each fix ships
   with a test observed failing before it and passing after.
7. **Integration.** One `integrator` per batch: full gate on the merged tree from a clean
   build, one red-before claim re-verified independently, and a sweep for what fell
   between packages.
8. **Record and push.** Append to the campaign log, commit, push. Then the next batch.
   Git does not function in a sandbox, so the agent writes the log and the human runs the
   commit and push; do not begin the next batch until they confirm it landed.

## What "confirmed" means

A finding is confirmed only by **recorded verdicts from independent verifiers**:

| Severity | Verdicts required | Majority needed |
|----------|-------------------|-----------------|
| critical | 2, plus a third as tie-breaker on disagreement | both, or the tie-breaker |
| high     | 2, plus a third as tie-breaker on disagreement | both, or the tie-breaker |
| medium   | 1 | that one |
| low      | 0 — recorded without verification, by policy | none |

**Zero recorded verdicts is `unverified`, never `confirmed`.** An absent vote is not a
vote in favor. A tally that treats "no refutations" as consensus will silently promote
every unverified claim, so state the count explicitly rather than computing a threshold
against an empty set.

A verifier that cannot independently reproduce the causal chain returns REFUTED.
Absence of proof is not proof.

**Refuted findings stay in the record** with their refutation and the reason. A campaign
that deletes what it disproved teaches the next campaign nothing, and the same claim
comes back.

## Severity rubric

| Severity | Meaning |
|----------|---------|
| critical | Silent wrong output, data corruption, or a divergence from an external convention that no test can catch |
| high     | Loud failure on a reachable input, or a contract violation a downstream consumer would act on |
| medium   | Reachable defect with a bounded, visible blast radius |
| low      | Hygiene, dead code, style, or a defect with no reachable path |

**A claim with no reproduction and no independently followable causal chain cannot be
graded above low**, whatever it describes, and is labelled `reasoning only`. Severity
tracks evidence, not eloquence.

## Standing dimensions

Start from this list so coverage is deliberate rather than improvised. Drop what does not
apply; add what the subject needs.

1. **Numerics and conventions** — agreement with the upstream the code claims to match,
   not merely with its own reference.
2. **Load-time validation** — what a malformed, hostile, or merely unusual input does.
3. **State and atomicity** — partial failure, rollback, and the invariants callers rely on.
4. **Contracts with downstream consumers** — artifacts other systems read.
5. **Cross-language contract** — for every artifact crossing a language boundary: which
   side writes and which reads, whether a committed test crosses it in that direction,
   whether a version field is actually read, and whether the field names have one
   definition or two.
6. **Test gaps** — what the suite structurally cannot catch, including fixtures whose
   shape makes a convention undetectable.
7. **Performance and memory** — see `perf-auditor`; measure the noise floor first.
8. **Idiom and hygiene** — lints, unsafe code, dependency posture.
9. **Input validation and decoding** — parsers, tokenizers, deserializers.
10. **Argument parsing and process lifecycle** — flags, child processes, signal handling.
11. **Documentation drift** — see `doc-auditor`.

## Packaging and the seams

Packages are cut by **file set**, disjoint, so remediators can work one tree at once. Two
things routinely go missing and must be named explicitly at triage and again at
integration:

- A confirmed finding that landed in **no** package.
- A file **two** packages both need, and the half of a finding that lives outside the
  package that owns it.

Both failure modes have shipped in this repository. They are the integrator's first
sweep, not an afterthought.

## Cadence

**Push after every batch.** A campaign that pushes only at the end loses everything to
one usage limit or one reclaimed container. Persist the findings ledger into the campaign
log *before* spending budget on fixes, not after.

One concern per commit. Every fix commits with its test.

## Exclusion list

A prior campaign's fixed and waived items are out of scope. Record the exclusion list in
phase 2, **written into the campaign log under `docs/verification/`** rather than held in
context, and check it at triage, so a settled waiver is not relitigated as a new finding.
Finders and the triage-agent both read it from that file; if it is absent, they report its
absence rather than assuming the set is empty.

A finding closed as **deferred** — neither fixed nor waived — is not silently absorbed
into the fixed count. It gets a disposition row, a reason, and a followup entry.
