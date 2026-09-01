# logtriage

Offline triage of `mongod` JSON logs. Profiles a log, runs deterministic detectors over it,
and asks a locally served model to explain what the detectors found, with every claim carrying
the record identifiers it rests on.

Invoked as `python -m logtriage <command>`; `cli.py` dispatches to one handler per verb in
`commands/`. Nothing it does requires a network at run time.

## Table of Contents

- [Relationship to the rest of the tree](#relationship-to-the-rest-of-the-tree)
- [Entry points](#entry-points)
- [How a triage run works](#how-a-triage-run-works)
- [Ports and their implementations](#ports-and-their-implementations)
- [The documentation corpus](#the-documentation-corpus)
- [Development](#development)
- [What is not verified here](#what-is-not-verified-here)

## Relationship to the rest of the tree

**This project imports nothing from the surrounding repository.** It carries its own
`pyproject.toml`, its own virtualenv, and its own quality gate, the way `rust-harness/` does.
Extraction is a copy:

```bash
$ cp -r logtriage/ ~/logtriage && cd ~/logtriage && git init
$ uv venv --python 3.12 && uv sync && uv run pytest
```

That independence has one cost, and it is deliberate.
`/.styles/PYTHON_OUTPUT_STYLE.md` says the canonical `Printer` in `/common/output.py` must be
imported rather than copied. `logtriage/output.py` copies it, because an import of `common/`
would not survive extraction. The copy is policed rather than trusted:
`/tests/logtriage/test_conformance.py` drives both Printers through the same call sequence and
fails if a single byte of stderr differs, and separately asserts that the copy reaches for
nothing first-party. That test does not travel on extraction, which is the intent — the
extracted project has no second `Printer` to disagree with.

The parent gate calls this one. `make quality` at the repository root runs
`make -C logtriage quality`, so a failure here fails a push from the parent repository.

## Entry points

```bash
$ make logtriage-install                 # provision this project's virtualenv (once)
$ make logtriage-quality                 # format check, lint, mypy, tests
$ make logtriage-test                    # tests only
$ make logtriage-fetch-kb                # fetch the MongoDB manual (needs network, once)
$ make logtriage-index-kb                # chunk and embed it into a retrieval index
```

Inside `logtriage/`:

```bash
$ uv run python -m logtriage gen-logs --list
$ uv run python -m logtriage gen-logs --scenario replica-stepdown --seed 1 --out mongod.log
$ uv run python -m logtriage profile mongod.log
$ uv run python -m logtriage triage mongod.log
$ uv run python -m logtriage triage mongod.log --kb          # with the manual loaded
$ uv run python -m logtriage triage mongod.log --llm none    # detectors only, no model
```

Exit codes follow `/.styles/PYTHON_OUTPUT_STYLE.md`: 0 success, 1 failure, 2 invalid usage.
A report whose citations do not all resolve exits 1, so a caller can distinguish a failed run
from a successful one without reading the prose. Human status goes to standard error;
standard output is reserved for `--json`.

## How a triage run works

1. **Parse.** `logtriage/mongod/records.py` reads the JSON envelope. A malformed line becomes a
   `ParseError` rather than an exception — rotation truncates lines mid-write, and a tool that
   dies partway through a 100MB corpus is useless exactly when it is needed. The count of
   unparseable lines is reported, because evidence that was never read changes how much a
   conclusion is worth.

2. **Normalize.** `logtriage/mongod/normalize.py` lifts the slow-query counters (`planSummary`,
   `keysExamined`, `docsExamined`, `nreturned`) onto the top level. The shape of `attr` is a
   function of the log identifier, so nothing could query it otherwise.

3. **Profile.** `logtriage/triage/profile.py` computes a whole-corpus summary: time span, severity and
   component histograms, distinct call sites, slow queries by namespace, clustered error text.
   A 1,500-line corpus renders in roughly 220 tokens against a 4,000-token budget.

4. **Detect.** `logtriage/mongod/diagnostics.py` runs six detectors and produces findings, each carrying
   the identifiers of the records behind it.

5. **Explain.** `logtriage/triage/pipeline.py` puts the profile and the findings in the prompt, then
   gives the model five tools and up to eight turns.

6. **Report.** `logtriage/triage/report.py` validates every citation and prints the first occurrence of
   each finding, in order, above the narrative.

Two design decisions in that sequence are worth stating outright.

**The detectors run before the model, not after.** Every number in a finding is arithmetic over
the whole corpus, which is the work a language model does worst. Asked to find a cause in a wall
of log lines it produces a fluent and occasionally inverted account, because a plausible
narrative is cheaper to generate than a correct one. Giving it counted evidence changes the task
from discovery to explanation.

**The report states the ordering as data.** Shown a collection scan and an election in one
corpus, a reader — human or model — reaches for the election, because it is the more dramatic
event. The `replica-stepdown` fixture is built to invite exactly that error: the slow queries
begin six minutes before the first heartbeat fails and continue after the new primary is
elected, so the available reading is backwards. No prompt reliably prevents this. Printing the
first-occurrence times with the gaps between them does, because the disproof is then visible
before the narrative starts.

## Ports and their implementations

Four interfaces, each with an implementation that needs no services and one that needs a server.
The first set is what makes the whole pipeline runnable in a checkout with nothing installed.

| Port | Real | Offline |
|------|------|---------|
| `LogStore` | MongoDB `$match`/`$group`/`$sort` | `InMemoryLogStore` |
| `KnowledgeStore` | MongoDB `$vectorSearch`/`$search` via `mongot` | `InMemoryKnowledgeStore` |
| `Embedder` | llama-server `/v1/embeddings` | `HashingEmbedder` |
| `LlmClient` | llama-server `/v1/chat/completions` | `ScriptedLlmClient` |

`logtriage/store/pipelines.py` builds the MongoDB aggregations as data and touches no database, which is
how the adapter is reviewed on a machine that has none. `tests/test_store_differential.py`
executes those same pipelines through `mongomock` and asserts the two implementations return
identical results across 13 filters.

**`HashingEmbedder` has no semantics.** It hashes tokens into a unit vector, so texts sharing
vocabulary land near each other and texts sharing meaning do not. A retrieval score produced
with it measures word overlap. Reading one as a quality result is a mistake that survives a long
time, because the number looks exactly like the real one.

**`ScriptedLlmClient` raises when its script runs out.** A fake that answers with an empty string
makes a broken loop look like a working one: the run completes, the report renders, and nothing
was exercised.

## The documentation corpus

The knowledge base is the real MongoDB Database Manual, fetched from
`github.com/mongodb/docs` and never committed.

```bash
$ uv run python -m logtriage fetch-kb      # sparse clone, writes kb-manifest.json
$ uv run python -m logtriage index-kb      # chunk, embed, write .kb/index
$ uv run pytest -m corpus                  # retrieval round trips over the real manual
```

The manual now sits inside a documentation monorepo carrying every MongoDB docs product, at
`content/manual/manual`. `fetch-kb` takes a blobless partial clone and a sparse checkout of
that one directory: 91MB and about five seconds, against a full clone of everything MongoDB
publishes. If the remote ignores the partial-clone filter git says nothing and fetches the
lot, so the fetch fails on a size ceiling rather than filling the disk.

Nothing fetched is committed. The manual is CC BY-NC-SA 3.0 US, asserted on the published
pages rather than in a LICENSE file — the repository has none — and vendoring it would pull
share-alike terms over this tree. `/logtriage/kb-manifest.json` is committed instead: the
resolved commit, and a sha256 for every one of the 4,260 corpus files. That is what makes an
index reproducible, which is the property that matters; a copy of the text would add nothing
else.

Measured over the manual at commit `88ec8192b924`, version 8.3:

| | |
|---|---|
| Pages | 1,611 |
| Chunks | 15,025 |
| Tokens per chunk | min 2, p50 123, p95 450, max 5,789 |
| Chunks carrying code | 4,918 |
| Code blocks intact | 8,394 of 8,394 |
| Include directives resolved | 7,862 of 10,292 |
| Chunk and embed | 3.8s + 2.7s |
| Index on disk | 31MB |

**Include resolution is most of the corpus.** Four pages in five pull content in with
`.. include::`, 7,350 directives against a tree of fragments that are never pages themselves.
A reader that treats a page file as the page sees a fraction of the manual and a great many
dangling references. The 2,430 that do not resolve are accounted for by category, and every
category is a file the documentation toolchain generates during a build or one that lives in
another repository: 648 code examples, 441 extracts generated from `includes/extracts-*.yaml`,
31 driver examples, 11 release fragments. A corpus test fails if unclassified misses rise
above a floor, because that would mean a resolver bug rather than an absent file.

The extracts are the one recoverable category left on the table. They are a build system —
697 refs across 320 yaml files, 117 of them inheriting from another ref with template
replacement — and a partial reimplementation would silently produce text MongoDB does not
publish.

## Development

```bash
$ cd logtriage
$ uv venv --python 3.12 && uv sync
$ uv run pytest -v
$ make quality
```

The `mongod` log format is already templated at the source: every line carries a numeric `id`
identifying the call site that wrote it, paired with a fixed `msg`. So `(component, id)` is an
exact template with no similarity threshold and nothing to tune, and log-template clustering —
the obvious tool for this job — is redundant. `drain3` is used only on the genuinely free-text
fields, `attr.errmsg` and `attr.error`.

Log identifiers outside the three well-known ones (51803 slow query, 22943 and 22944 connection
lifecycle) are representative rather than verified against a specific server build. The
detectors therefore match on component and message text, because a detector keyed on an
identifier stops firing silently after a server upgrade.

## What is not verified here

Stated because the alternative is a reader assuming coverage that does not exist.

- **`$vectorSearch` and `$search` have never run.** `mongomock` implements neither, and they are
  `mongot` operators rather than MongoDB ones. That path is unexercised until it meets a real
  server.
- **`$firstN` and `$dateTrunc` are unexercised.** `mongomock` lacks both, so six differential
  cases skip. They skip by name, saying which operator went untested, because a silent skip is
  indistinguishable from a pass.
- **No real embeddings have been produced, so no retrieval quality has been measured.** The
  corpus is real and the chunking is verified against it, but the vectors are hashes. Every
  ranking assertion in the corpus suite therefore runs on the text leg, which is honest
  lexical matching; the vector leg is asserted only to be well-formed. `search_docs` weights
  the vector leg to zero when the embedder declares itself non-semantic, because fusing a
  meaningless ranking in at the production weight measurably pushed the correct page out of
  the top three. Questions that need meaning rather than words are listed in
  `tests/test_kb_corpus.py` as `SEMANTIC_QUESTIONS` and deliberately not asserted.
- **Nothing here establishes how a model behaves.** Every test uses the scripted client, so a
  narrative passing the causality check is evidence about the guard rails and not about the
  model. That question needs a served model.
- **Nine of the twelve log ids in the fixtures are unverified, and cannot be verified here.**
  MongoDB publishes no catalogue of log ids. The manual documents fourteen inside worked
  examples, which confirms 51803 exactly and the component of 22943 and 22944, and covers none
  of connection refusal, heartbeats, elections or index builds. Those remain representative,
  which is why every detector matches on component and message instead.
