# Training implementation investigations

One record per training deliverable, written while that deliverable was being implemented.
Each states what was decided, what was measured, and which tradeoffs were accepted, at the
time the code was written.

All 20 records here were written between 2026-07-14 and 2026-07-15.

## These are frozen records

Every document in this directory carries the class line
`.styles/DOCUMENTATION_STYLE.md` defines for a frozen record: *what was decided or measured
on a day*, whose obligation is **never edit the body; drift is expected and correct**.

That obligation is the opposite of the one the living documents in the parent directory
carry. A path, a line number, or a parameter value quoted here describes the tree as it stood
on the record's date. When it disagrees with the current tree, the record is not wrong and is
not to be corrected — the disagreement is the history. If you need to know how something works
now, read the living document, not the record.

## Naming

`<deliverable-id>_<subject>_investigation.md`, lowercase with underscores — for example
`d4_memory_fit_investigation.md`. The deliverable id is the roadmap identifier (`d` for the
Phase 0 substrate, `f` for Phase 1 fine-tuning and distillation, `p2_d` for Phase 2); one
record has no id because it spans the tier.

**The `_investigation.md` suffix is load-bearing, not decorative.** Both documentation gates
exempt frozen records by exactly two patterns: a filename ending `_investigation.md`, or an
`investigations/` path segment. A record named anything else is gated as though it were
living, and the gate will demand its historical paths still resolve — which for a document
that is never updated is a demand that cannot be met.

## Finding a record

The index is [../README.md](../README.md), which pairs each record with the living document
it produced and names the ones that have no pair in either direction. It is deliberately not
repeated here: that table is checked by `scripts/docs/check_doc_refs.py`, and a second copy in
this directory would not be — the same `investigations/` exemption that protects these records
would leave a duplicate index to rot unnoticed.

## Related documentation

- [../README.md](../README.md) — the training tier index and the record pairing
- [../../../training/README.md](../../../training/README.md) — the source tree these records
  describe
- [../../verification/](../../verification/) — host-side acceptance records, which are
  evidence from a workstation run rather than implementation-time decisions
