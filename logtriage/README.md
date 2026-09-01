# logtriage

Offline triage of `mongod` JSON logs. Profiles a log, runs deterministic detectors over it,
and asks a locally served model to explain what the detectors found, with every claim carrying
the record identifiers it rests on.

Invoked as `python -m logtriage <command>`; `cli.py` dispatches to one handler per verb in
`commands/`. Nothing it does requires a network at run time.

This file is orientation. [docs/README.md](docs/README.md) is the documentation index, and
[AGENTS.md](AGENTS.md) is the policy.

## Table of Contents

- [Relationship to the rest of the tree](#relationship-to-the-rest-of-the-tree)
- [Entry points](#entry-points)
- [How a triage run works](#how-a-triage-run-works)
- [Layout](#layout)
- [Development](#development)
- [Where the real docs live](#where-the-real-docs-live)

## Relationship to the rest of the tree

**This project imports nothing from the surrounding repository.** It carries its own
`pyproject.toml`, its own virtualenv, its own style guides, its own documentation gate, and its
own quality gate, the way `rust-harness/` does. Extraction is a copy:

```bash
$ cp -r logtriage/ ~/logtriage && cd ~/logtriage && git init
$ uv venv --python 3.12 && uv sync && make quality
```

That independence has one cost, and it is deliberate.
`/.styles/PYTHON_OUTPUT_STYLE.md` says the canonical `Printer` in `/common/output.py` must be
imported rather than copied. `logtriage/output.py` copies it, because an import of `common/`
would not survive extraction. The copy is policed rather than trusted:
`/tests/logtriage/test_conformance.py` drives both Printers through the same call sequence and
fails if a single byte of stderr differs, asserts that the copy reaches for nothing first-party,
and holds the four copied style guides byte-identical to their originals. That test does not
travel on extraction, which is the intent — the extracted project has no second copy to
disagree with.

The parent gate calls this one. `make quality` at the repository root runs
`make -C logtriage quality`, so a failure here fails a push from the parent repository.

## Entry points

```bash
$ make logtriage-install                 # provision this project's virtualenv (once)
$ make logtriage-quality                 # format check, lint, mypy, docs, tests
$ make logtriage-test                    # tests only
$ make logtriage-fetch-kb                # fetch the MongoDB manual (needs network, once)
$ make logtriage-index-kb                # chunk and embed it into a retrieval index
$ make logtriage-secrets                 # generate the served stack's credentials
$ make logtriage-up                      # start mongod, mongot, and the two llama-servers
$ make logtriage-bootstrap               # replica set and the mongot user (once, after first up)
```

Inside `logtriage/`:

```bash
$ uv run python -m logtriage gen-logs --list
$ uv run python -m logtriage gen-logs --scenario replica-stepdown --seed 1 --out mongod.log
$ uv run python -m logtriage profile mongod.log
$ uv run python -m logtriage triage mongod.log
$ uv run python -m logtriage triage mongod.log --kb          # with the manual loaded
$ uv run python -m logtriage triage mongod.log --llm llama   # against a served model
$ uv run python -m logtriage triage mongod.log --llm none    # detectors only, no model
$ uv run python -m logtriage ingest mongod.log --store mongo  # into the deployment
$ uv run python -m logtriage eval-retrieval                  # score retrieval against ground truth
$ uv run python -m logtriage bench ingest --log mongod.log   # median and spread over N samples
```

Exit codes follow `/.styles/PYTHON_OUTPUT_STYLE.md`: 0 success, 1 failure, 2 invalid usage.
A report whose citations do not all resolve exits 1, so a caller can distinguish a failed run
from a successful one without reading the prose. Human status goes to standard error;
standard output is reserved for `--json`.

## How a triage run works

1. **Parse.** `logtriage/mongod/records.py` reads the JSON envelope. A malformed line becomes a
   `ParseError` rather than an exception, and the count of unparseable lines is reported —
   evidence that was never read changes how much a conclusion is worth.

2. **Normalize.** `logtriage/mongod/normalize.py` lifts the slow-query counters onto the top
   level. The shape of `attr` is a function of the log identifier, so nothing could query it
   otherwise.

3. **Profile.** `logtriage/triage/profile.py` computes a whole-corpus summary against a token
   budget: time span, severity and component histograms, distinct call sites, slow queries by
   namespace, clustered error text.

4. **Detect.** `logtriage/mongod/diagnostics.py` runs six detectors, producing eight kinds of
   finding, each carrying the identifiers of the records behind it.

5. **Explain.** `logtriage/triage/pipeline.py` puts the profile and the findings in the prompt,
   then gives the model five tools and up to eight turns.

6. **Report.** `logtriage/triage/report.py` validates every citation and prints the first
   occurrence of each finding, in order, above the narrative.

Two decisions in that sequence carry the design, and both are owned by
[docs/design/citations.md](docs/design/citations.md): the detectors run **before** the model, so
its task is explanation rather than discovery; and the report states the ordering **as data**,
so a reader cannot reach for the more dramatic event as the cause.

## Layout

| Directory | Holds |
|-----------|-------|
| `logtriage/` | the package: `commands/`, `mongod/`, `store/`, `kb/`, `embed/`, `llm/`, `tools/`, `triage/`, `evals/` |
| `logtriage/evals/` | the committed retrieval ground truth, the metrics over it, and the method |
| `tests/` | the suite; markers `corpus`, `differential`, `services` are deselected by default |
| `docs/` | the documentation, indexed by [docs/README.md](docs/README.md) |
| `scripts/docs/` | the eight documentation gates |
| `.styles/` | verbatim copies of the style guides, so they survive extraction |

## Development

```bash
$ cd logtriage
$ uv venv --python 3.12 && uv sync
$ uv run pytest -v
$ make quality
```

`make quality` is format, lint, types, the documentation gate, and the tests. It runs offline and
must pass before every push. The documentation gate is eight checks; two of them have no
counterpart in the surrounding repository, and one of those exists because three commands were
written up in a runbook before they were written in the code.

## Where the real docs live

[docs/README.md](docs/README.md) indexes everything. The three most likely to be wanted:

- [docs/design/architecture.md](docs/design/architecture.md) — the four ports and their two
  implementations each
- [docs/operations/stack.md](docs/operations/stack.md) — the served stack: services, ports,
  bootstrap, configuration
- [docs/verification/README.md](docs/verification/README.md) — the standing list of what is not
  verified, including `$vectorSearch`, real embeddings, and anything about how a model behaves
