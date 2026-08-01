---
description: Run a bounded quality campaign over a scope — discover, verify, triage, remediate in batches, and push each batch
argument-hint: <scope-path> [dimensions]
allowed-tools: Read, Edit, Write, Bash, Glob, Grep
---
Scope: $1
Dimensions (optional, default the standing list): $2

Read `.claude/skills/bug-bash/SKILL.md` first. It defines the phase ladder, the severity
rubric, what counts as a confirmed finding, and the standing dimension list. Every term
below means what that document says it means.

Run the phases in order. Do not start a later phase while an earlier one is open.

1. Baseline — confirm the project gate for `$1` exits 0 before anything changes. Record
   the determinism oracles and the perf noise floor. If the gate is already red, stop and
   report; a campaign on a broken baseline measures nothing.
2. Scope — list the dimensions in play and the file set each one owns, plus the exclusion
   list from prior campaigns. Report the plan before spawning anything.
3. Discover — one `bug-hunter` per dimension. Finders do not fix and do not propose fixes.
4. Verify — `skeptic` agents per the rubric's vote counts. Record every verdict, including
   the refutations.
5. Triage — one `triage-agent`. Its four output sections are the work plan: packages,
   unverified, seams, dedupe map.
6. Persist — write the ledger into the campaign log under `docs/verification/` and **push
   it before spending any budget on fixes**. Discovery that exists only in a session is
   lost when the session is.
7. Remediate — per batch: one `remediator` per package on the shared tree, then one
   `integrator` over the merged result, then the gate, then commit and push. One concern
   per commit. Do not begin the next batch until this one is pushed.
8. Close — append dispositions, verification evidence, and followups to the campaign log.
   Every confirmed finding gets a disposition: fixed, waived with measurement, or deferred
   with a reason. Deferred findings are not counted as fixed.
9. Stop for review — summarize the counts, the evidence, and what remains unproven. Do not
   claim a campaign closed the numerics, performance, or parity questions it did not
   actually measure.
