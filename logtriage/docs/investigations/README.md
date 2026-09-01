# Investigations

Frozen records: what was decided or measured on a day, and why.

## Table of Contents

- [The convention](#the-convention)
- [Why they are exempt from the gates](#why-they-are-exempt-from-the-gates)
- [Entries](#entries-0)

## The convention

A document here describes the project as it stood on its date. **Bodies are never edited.** A
reference that has since moved is accurate history, not drift, and correcting it would destroy
the thing the record exists to preserve.

Each carries the class line directly under its H1:

```markdown
# Include resolution over the real manual: investigation

**Frozen record — 2026-09-01. Drift against the current repository is expected and correct;
this document is never updated.**
```

Dates are ISO `YYYY-MM-DD`, always.

Write one when a decision cost real investigation and the reasoning would otherwise be lost —
particularly when the answer was narrower than the question, or when something was deliberately
*not* done. A living document states the conclusion; a frozen record states what was tried.

## Why they are exempt from the gates

`scripts/docs/check_doc_refs.py` and `scripts/docs/check_retired_names.py` both skip anything
under an `investigations/` directory and anything whose filename ends `_investigation.md`.

That exemption is what makes "never edit the body" possible. A frozen record accumulates paths,
commands, and names that no longer exist, and a gate demanding they still resolve would force
exactly the edits the class forbids.

The corollary matters as much: **a frozen record is not a source to check the tree against.**
Anything auditing this project for stale facts should skip these paths rather than rediscover
the same expired history each time.

## Entries (0)

None yet. The reasoning behind the decisions taken so far lives in the living documentation and
in the module docstrings that own each one — most of it in
[../corpus/README.md](../corpus/README.md), which records the five defects the real manual
exposed and the one category of include this project deliberately does not resolve.

## Related

- `../../.styles/DOCUMENTATION_STYLE.md` — the four document classes
- [../README.md](../README.md) — the documentation index
