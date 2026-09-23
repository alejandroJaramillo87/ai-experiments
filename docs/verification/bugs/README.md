# Defect ledger

The live ledger of defects the hardware verification campaign has hit, indexed by the
runbook step that surfaces each one, with its fix state.

The ledger is operational: rows move between sections as defects are fixed and never
leave, so a `KB-n` id stays stable for every runbook entry that cites it. Rows closed
before the 2026-08-08 reset are in
[../archive/ledger-archive.md](../archive/ledger-archive.md).

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [known-bugs.md](known-bugs.md) | Defects the hardware verification campaign expects to hit, indexed so the operator running the [hardware verification runbook](../../README.md) knows what to expect at each step. |

<!-- inventory:pages:end -->

## Related documentation

- [../README.md](../README.md) — the campaign and the recording protocol that says when a
  row is opened and when it closes
- [../../../scripts/docs/check_doc_counts.py](../../../scripts/docs/check_doc_counts.py)
  — the gate that holds the ledger's counts to its rows
