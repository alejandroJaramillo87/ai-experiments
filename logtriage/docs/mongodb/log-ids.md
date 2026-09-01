# Log identifiers: what is verified and what is representative

MongoDB publishes no catalogue of log identifiers. This document records exactly what could be
established against the real manual, and what could not, because the difference decides how the
detectors are written.

## Table of Contents

- [The finding](#the-finding)
- [What the manual confirms](#what-the-manual-confirms)
- [What stays representative](#what-stays-representative)
- [Why the detectors do not key on identifiers](#why-the-detectors-do-not-key-on-identifiers)
- [How this is kept honest](#how-this-is-kept-honest)

## The finding

The MongoDB Database Manual documents log identifiers only inside worked examples — fourteen of
them across the corpus, in `reference/log-messages.txt` and the pages that include it. There is
no reference table of ids anywhere in the manual, and none is published elsewhere.

Reconciled against the corpus at commit `88ec8192b924`, that yields a narrower answer than the
question assumed. `logtriage/kb/log_ids.py` extracts the documented triples;
`logtriage/fixtures/generator.py` declares which category each id it emits belongs to.

## What the manual confirms

**Three of the twelve identifiers the fixture generator emits.**

| Id | Meaning | Evidence |
|----|---------|----------|
| 51803 | Slow query / command | Confirmed exactly: id, component and message all match |
| 22943 | Connection accepted | Component confirmed; the manual's example message is the 4.4-era form |
| 22944 | Connection ended | Component confirmed; same caveat |

The caveat on 22943 and 22944 is worth stating plainly: the manual's worked examples for those
two carry message text that the current server no longer emits verbatim. The identifier and
component are right; the template shown is stale. That is a fact about the manual, not about the
server, and it is precisely the kind of thing a detector keyed on message text would trip over
if it matched too tightly.

## What stays representative

**Nine of the twelve.** The manual documents none of these, and no detector matches on them.
They are plausible values used to generate realistic fixtures, not claims about a server build:

connection refused, heartbeat failure, election starting, election succeeded, state transition,
index build starting, index build scan complete, index build failed, and WiredTiger checkpoint.

Nothing about connection refusal, heartbeats, elections, or index builds is documented with an
identifier anywhere in the manual.

## Why the detectors do not key on identifiers

An identifier is a call site. Call sites move, get renumbered, and get removed between server
versions, and none of that is announced.

A detector keyed on an id **stops firing silently** after a server upgrade: no error, no warning,
just a report that no longer mentions the thing it was written to find. A detector keyed on
component and message text degrades loudly instead — it stops matching, and the corpus profile
still shows the records it should have matched.

So `logtriage/mongod/diagnostics.py` matches on `(component, message text)` throughout. The six
detector functions produce eight kinds of finding, and not one of them names an identifier.

## How this is kept honest

`fixtures/generator.py` declares two frozensets, `DOCUMENTED_IDS` and `REPRESENTATIVE_IDS`.
`tests/test_generator.py` holds every id the generator emits against those two categories and
fails on one belonging to neither — so a new fixture id cannot be added without a decision about
which it is.

That test runs in the default suite rather than the corpus tier, because it needs no corpus: it
checks the declaration, not the manual. The corpus-marked tests in `tests/test_kb_corpus.py`
check the other direction, that the documented ids really do appear in the fetched manual.

## Related

- [log-format.md](log-format.md) — why `(component, id)` is already an exact template
- [../corpus/README.md](../corpus/README.md) — the manual this was reconciled against
- [../verification/README.md](../verification/README.md) — the standing list of what is unverified
