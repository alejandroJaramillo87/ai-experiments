# logtriage documentation

Central index. Each topic has one owner; read the owner rather than a summary of it.

`../AGENTS.md` carries the policy — what you may not do, and how work is done here. This file is
the map.

## Table of Contents

- [Start here](#start-here)
- [Design](#design)
- [MongoDB](#mongodb)
- [The documentation corpus](#the-documentation-corpus)
- [Operations](#operations)
- [Verification](#verification)
- [Document classes](#document-classes)
- [The documentation gate](#the-documentation-gate)

## Start here

| If you want to | Read |
|----------------|------|
| Run the tool | [../README.md](../README.md) — entry points and exit codes |
| Understand how it is put together | [design/architecture.md](design/architecture.md) |
| Know why a claim can be trusted | [design/citations.md](design/citations.md) |
| Start the services | [operations/stack.md](operations/stack.md) |
| Know what is *not* proven | [verification/README.md](verification/README.md) |

## Design

| Document | Description |
|----------|-------------|
| [design/architecture.md](design/architecture.md) | The four ports, two implementations each, and where the semantics live |
| [design/citations.md](design/citations.md) | Every claim carries its records; unresolvable citations fail the run; the causality guard |
| [design/prompting.md](design/prompting.md) | The five tools, the token budget, the eight-turn loop, and why native tool calling is a server feature |

## MongoDB

| Document | Description |
|----------|-------------|
| [mongodb/log-format.md](mongodb/log-format.md) | The JSON envelope, `svc`, why `attr` has no fixed shape, and why the format is already templated |
| [mongodb/log-ids.md](mongodb/log-ids.md) | Three identifiers the manual confirms, nine that stay representative, and why detectors match on text |
| [mongodb/search.md](mongodb/search.md) | `mongot` is not MongoDB: what that requires, the two index definitions, and what has never executed |

## The documentation corpus

| Document | Description |
|----------|-------------|
| [corpus/README.md](corpus/README.md) | Fetching the real manual, the licence position and the committed manifest, Snooty rST, and the five defects real pages exposed |
| [corpus/statistics.md](corpus/statistics.md) | The measured numbers at a named corpus commit, and how to read them |

## Operations

| Document | Description |
|----------|-------------|
| [operations/stack.md](operations/stack.md) | The four services, the ports, the one-time bootstrap, configuration, VRAM, and the traps |
| [operations/three-machines.md](operations/three-machines.md) | Sandbox, desktop, MacBook: what each can prove, and the evidence rule |

## Verification

| Document | Description |
|----------|-------------|
| [verification/README.md](verification/README.md) | The protocol, the phase order, and the standing list of what is not verified |
| [verification/desktop-deployment.md](verification/desktop-deployment.md) | Phase D1-D5: the stack up, and the differential suite against a real MongoDB |
| [verification/desktop-search.md](verification/desktop-search.md) | Phase D6-D9: ingest, index into `mongot`, `$search` and `$vectorSearch`, triage end to end |
| [verification/desktop-model.md](verification/desktop-model.md) | Phase D10-D13: real embeddings, retrieval quality, pooled adjudication, the Gemma round trip |
| [verification/desktop-scale.md](verification/desktop-scale.md) | Phase D14-D16: ingest throughput, retrieval latency, end-to-end latency |
| [verification/macbook.md](verification/macbook.md) | Phase E: extraction, ARM images, Metal, unified memory |

## Document classes

`../.styles/DOCUMENTATION_STYLE.md` defines four classes, and the class decides whether drift
against the current tree is a bug to fix or a record to preserve.

Everything listed above is **living reference** except `verification/`, which is
**operational** — runbooks executed against a machine, whose audience is whoever is running the
checks.

**Frozen records** live in [investigations/](investigations/), which is empty so far. That
directory is exempt from the reference gates by path, which is what makes "never edit the body"
possible; `verification/` is deliberately *not* exempt, so its commands and paths are checked
like any other document's.

## The documentation gate

`make quality` runs six checks over this tree, in `../scripts/docs/`. They exist because a
finding in a report rots and a finding in a gate fails the build.

| Check | Fails when |
|-------|------------|
| `check_doc_refs.py` | a document cites a path in this project that does not exist |
| `check_retired_names.py` | a name that has been retired is still in use anywhere |
| `check_make_targets.py` | a documented `make` target does not exist |
| `check_doc_counts.py` | a stated count disagrees with the rows beneath it |
| `check_corpus_stats.py` | the documented corpus statistics disagree with the manifest or the index |
| `check_surface_census.py` | something the code ships is named in no document |

The last two are this project's own. `check_corpus_stats.py` catches the statistic that goes
quietly stale after a corpus refetch — a stale number reads exactly like a fresh one.
`check_surface_census.py` runs the opposite direction from every other check: it starts from the
code and subtracts what the documentation mentions, which is the only way to see a feature nobody
thought to document.
