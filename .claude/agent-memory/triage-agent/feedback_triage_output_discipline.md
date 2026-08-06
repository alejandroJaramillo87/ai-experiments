---
name: triage-output-discipline
description: In bug-bash triage, return only decision fields - never echo spec_says/code_says/host_check back, or the structured output blows the 32k token limit
metadata:
  type: feedback
verified: 2026-08-06
stale_when: the structured-output token limit changes, or the orchestrator stops reattaching evidence fields from the input ledger
---

When triaging a bug-bash findings ledger (the triage phase of
`.claude/skills/bug-bash/SKILL.md`),
emit only the decision fields: keep_id, merge_ids, severity, verdict, line, title,
failure_mode, quote_status. Never re-emit the input payload (spec_says, code_says, file,
tier, claim_ids, variants, host_check).

**Why:** a previous triage run echoed the evidence payload back and exceeded the 32000
output-token limit, losing the whole pass. The orchestrator reattaches those fields from
the input ledger automatically, so repeating them buys nothing.

**How to apply:** put verification reasoning in tool calls and in the free-form `note`
field (package table, unverified list, seams, dedupe map), not in per-finding prose.
Quote verification is still mandatory - open every cited file at the cited line - but the
result of that check is expressed as `quote_status` plus a corrected `line`, not as a
transcription of the quote.

Related: [[project-gemma4-conformance-audit]]
