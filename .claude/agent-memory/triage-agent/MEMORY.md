# Triage-agent memory index

- [Triage output discipline](feedback_triage_output_discipline.md) — return only decision fields; echoing the evidence payload blew the 32k output limit once
- [Gemma 4 conformance audit](project_gemma4_conformance_audit.md) — docs/models/gemma-4.md is sole authority; the 2304 d_model trap and the needs-host GGUF rule
- [Gemma 4 "2304" propagation root](project_gemma4_2304_propagation.md) — the wrong E2B geometry has one documentary root (p2_d5 investigation) and a dozen copies
- [Calculator scalar-spec root](project_gemma4_conformance_campaign.md) — ArchitectureSpec's flat scalars are the single root behind most training-calculator Gemma 4 findings
- [Synth fixture root](project_gemma4_synth_fixture_root.md) — one synthetic gemma4 GGUF backs every rust test and encodes the conventions the spec contradicts (extends deferred A1-15)

Memory format: each file carries `verified:` (when its claims were last checked against the
tree) and `stale_when:` (the observable event that invalidates it). Grade a memory against
those before acting on it. This matters most for the root-cause memories above: each tells
you to fold a new finding into a known root, and once that root is remediated the same
instruction starts suppressing real findings instead of deduplicating noise.

All five were re-graded against the tree on 2026-08-06 and remain live — the Gemma 4
conformance campaign was audit-only
(`docs/verification/investigations/gemma-4-conformance_investigation.md:13`), so
almost nothing it found has been remediated.

Removed 2026-08-06: the `gemma4-spec-authority` stub, which was a self-declared duplicate.
The one inbound `[[gemma4-spec-authority]]` link, in the 2304-propagation memory, now points
at `project-gemma4-conformance-audit` where the rule actually lives — this closes the
dangling-pointer findings GC-159 and GC-162.
