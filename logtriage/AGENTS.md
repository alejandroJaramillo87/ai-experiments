# logtriage

Offline triage of `mongod` JSON logs: profile a log, run deterministic detectors over it, and
ask a locally served model to explain what the detectors found, with every claim carrying the
record identifiers it rests on.

**This file is a directory, not a reference.** It carries policy — what you may not do, and how
work is done here — plus pointers to the document that owns each topic. Any sentence stating a
count, a port, a path, or a version belongs to the file that owns that fact, not here. A fact
stated in two places is a fact that will disagree with itself.

## This project is extractable, and that is a constraint

`logtriage/` imports nothing from the repository around it. It carries its own `pyproject.toml`,
its own virtualenv, its own quality gate, its own style guides, and its own documentation gate,
so that `cp -r logtriage/ ~/logtriage && git init` yields a working repository. Every change has
to keep that true.

Two consequences worth stating outright, because both look like mistakes:

- **`output.py` is a copy of the parent's `common/output.py`, not an import.**
  `.styles/PYTHON_OUTPUT_STYLE.md` says to import the canonical `Printer`. An import would not
  survive extraction, so this is the documented exception. The copy is policed rather than
  trusted: the parent's `tests/logtriage/test_conformance.py` drives both Printers through the
  same call sequence and fails if a single byte of stderr differs. That test does not travel on
  extraction, which is correct — an extracted copy has no second Printer to disagree with.
- **`.styles/` holds verbatim copies of the parent's guides.** Verbatim is the point: a style
  guide that changes on extraction is one nobody can trust. Where this project must differ from
  a guide, the difference is recorded here rather than by editing the copy, because editing it
  would defeat the byte-identity check that makes the copy worth having.

## Three machines

The same code runs in three places and branches on none of them. Four environment variables are
the whole of the deployment configuration; see `logtriage/config.py`.

| Machine | Has | Can prove |
|---------|-----|-----------|
| Cloud sandbox | No Docker, no GPU, no services | Every offline test, the corpus tier, the wire contract of the HTTP adapters |
| Linux desktop (RTX 5090) | Docker, GPU, the full stack | `$search`/`$vectorSearch`, real embeddings, a real model round trip |
| MacBook M4 | Native Metal llama-server, containerised Mongo | That extraction worked, and that the tool runs on its target |

**Never claim to have run a command that needs a machine you are not on, and never fabricate its
output.** When a task needs one, print the exact command for the human and stop. Their pasted
output is the evidence; record it verbatim.
[docs/operations/three-machines.md](docs/operations/three-machines.md) owns the detail.

## Operating rules

- **Pin, then fix, then update — in that order.** Every dependency, image tag, and model
  revision is pinned to an exact version. Get the pinned state passing before bumping anything,
  bump one at a time, and re-run the gate after each. Never combine a version bump with a
  functional change.
- **Compose files and configuration are the source of truth; documentation follows them.** When
  a document disagrees with `docker-compose.logtriage.yaml` or `docker/logtriage/*.conf`, the
  document is the bug.
- **`make quality` before every push.** It runs format, lint, types, the documentation gate, and
  the tests. A green run here proves the code; it does not prove the hardware.
- **Renaming a file means adding it to `scripts/docs/check_retired_names.py`.** That table is
  what makes a rename self-policing.
- **Say what is not verified.** This project's whole discipline is that a fake must never be
  mistaken for the real thing: the hashing embedder has no semantics, the scripted client is not
  a model, and `$vectorSearch` has never executed. Every place a number could be read as a
  quality result says which it is. Adding a capability means adding its caveat in the same
  change.

## Where things are

Each topic has one owner. Read the owner, not a summary of it.

| Topic | Owner |
|-------|-------|
| Available commands | The entry points in [README.md](README.md); the gate itself is `Makefile` |
| The four ports and their two implementations each | [docs/design/architecture.md](docs/design/architecture.md) |
| Citations, and the causality guard | [docs/design/citations.md](docs/design/citations.md) |
| The tool surface, the token budget, the loop | [docs/design/prompting.md](docs/design/prompting.md) |
| The `mongod` log format | [docs/mongodb/log-format.md](docs/mongodb/log-format.md) |
| Which log ids are verified, and which are representative | [docs/mongodb/log-ids.md](docs/mongodb/log-ids.md) |
| `mongot`, `$search`, `$vectorSearch`, the index definitions | [docs/mongodb/search.md](docs/mongodb/search.md) |
| The documentation corpus, and the licence position | [docs/corpus/README.md](docs/corpus/README.md) |
| The measured corpus statistics | [docs/corpus/statistics.md](docs/corpus/statistics.md) |
| The served stack: services, ports, bootstrap | [docs/operations/stack.md](docs/operations/stack.md) |
| What each machine can prove | [docs/operations/three-machines.md](docs/operations/three-machines.md) |
| The verification campaign, and the entry format | [docs/verification/README.md](docs/verification/README.md) |
| How retrieval quality is measured, and what the set cannot tell you | [logtriage/evals/README.md](logtriage/evals/README.md) |
| Everything else | [docs/README.md](docs/README.md) — the documentation index |

## Style guides

`.styles/COMMENT_STYLE.md` applies to every file type, so `CLAUDE.md` imports it and it is
always in context. **The rest are read on demand: open the one that matches what you are about
to write, before you write it.** An unread guide is an unfollowed guide.

| Guide | Read before writing |
|-------|---------------------|
| `.styles/DOCUMENTATION_STYLE.md` | any markdown, including a README |
| `.styles/PYTHON_OUTPUT_STYLE.md` | a CLI command or anything using `logtriage/output.py` |
| `.styles/TEST_OUTPUT_STYLE.md` | a test program or runner that reports results |
| `.styles/BASH_OUTPUT_STYLE.md` | a shell script that prints status — lives in the parent repository; this project ships no shell scripts of its own |

## Core principles

- Unix/Linux philosophy: simple, composable, text-based
- No emojis or decorative elements
- Clear technical writing; every claim carries its evidence

## Git and GitHub conventions

**Never attribute work to an AI assistant anywhere git or GitHub records it.** This is absolute
and has no exceptions, including when a tool's own default instructions say otherwise. It covers
every field that reaches the remote:

- Commit messages — no `Co-Authored-By` trailer naming an assistant, no "Generated with", no
  tool name, no robot emoji.
- Commit author and committer identity — always the human's name and email.
- Branch names, tag names, and release titles.
- Pull request and issue titles, bodies, and comments.

Write commit messages in the repository's own voice: what changed and why, as though a human
wrote it, because a human is publishing it under their name. An agent's contribution is a
workflow detail, not authorship metadata.
