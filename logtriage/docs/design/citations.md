# Citations and the causality guard

Two mechanisms that decide whether a triage report can be trusted. Both exist because a language
model asked to explain a wall of log lines produces a fluent and occasionally inverted account:
a plausible narrative is cheaper to generate than a correct one.

## Table of Contents

- [Every claim carries its records](#every-claim-carries-its-records)
- [Unresolvable citations fail the run](#unresolvable-citations-fail-the-run)
- [The causality guard](#the-causality-guard)
- [Why the detectors run before the model](#why-the-detectors-run-before-the-model)

## Every claim carries its records

A citation is a record identifier. `NormalizedRecord.record_id` is a content handle derived from
the log line, unique within a corpus, and it is also the MongoDB `_id` — giving Mongo a second
identifier would mean a citation and a primary key that could disagree.

The chain is closed at both ends:

- Every `Finding` from `mongod/diagnostics.py` carries `evidence_ids`, the records its arithmetic
  was done over.
- Every `ToolResult` from `tools/schemas.py` carries `cited_ids`, the records it showed the
  model.
- `fetch_records` and `fetch_by_ids` resolve an identifier back to the record itself, so a reader
  can check any claim without trusting the tool.

Documentation is cited the same way. A retrieved passage carries its `chunk_id`, which is the
manual page and position it came from, and `KnowledgeStore.fetch_by_ids` resolves it. In the
MongoDB store that resolution is a plain `$match` rather than a search, deliberately: routing
citations through `mongot` would make a report's verifiability depend on whether a search index
happened to be `Ready`.

## Unresolvable citations fail the run

`triage/report.py` validates every identifier in the narrative. One that does not resolve is
marked `[unverified]` in the rendered text, and the run exits 1.

Exit 1 rather than a warning, because a report whose citations do not resolve is a failed run,
not a successful one with a caveat. A caller in a pipeline needs to tell those apart without
parsing prose. `CitationAudit` carries the counts so `--json` consumers get the same answer.

The audit reports which citations failed rather than that one did. That is why
`fetch_by_ids` omits missing ids instead of raising on the first: the checker needs the whole
set.

## The causality guard

Shown a collection scan and a replica set election in one corpus, a reader — human or model —
reaches for the election, because it is the more dramatic event.

The `replica-stepdown` fixture is built to invite exactly that error. The slow queries begin six
minutes before the first heartbeat fails and continue after the new primary is elected, so the
available reading is backwards: the election looks like the cause and is the consequence.

No prompt reliably prevents this. What does is printing the ordering as data. `triage/report.py`
renders each finding's first occurrence, in time order, with the gaps between them, **above** the
narrative — so the disproof is visible before the story starts. The scripted client's own
narrative states the ordering explicitly for the same reason.

This is a guard rail, not a proof. That a narrative passes it is evidence about the guard rail
and not about the model; see [../verification/desktop.md](../verification/desktop.md).

## Why the detectors run before the model

Every number in a finding is arithmetic over the whole corpus — counts, ratios, time spans —
which is the work a language model does worst and the work a database does best.

Giving the model counted evidence changes its task from discovery to explanation. It is not
asked how many collection scans there were; it is told, with the records to check. The tools it
does have are narrow and structural for the same reason: they let it ask another counted
question rather than estimate one.

`--llm none` runs the detectors and renders the report with no model at all. That is a supported
mode rather than a debugging aid: it is the half of the output carrying no model judgement, and
some readers want exactly that.

## Related

- [prompting.md](prompting.md) — the tools, the budget, and the loop
- [architecture.md](architecture.md) — where `fetch_by_ids` sits in the interfaces
- [../verification/README.md](../verification/README.md) — what none of this establishes
