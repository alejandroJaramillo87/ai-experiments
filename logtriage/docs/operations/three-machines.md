# Three machines, and what each can prove

The same code runs in three places and branches on none of them. This document records what each
can establish, because a claim verified on the wrong machine is not verified.

## Table of Contents

- [The three](#the-three)
- [Nothing branches on the machine](#nothing-branches-on-the-machine)
- [The evidence rule](#the-evidence-rule)
- [Extraction](#extraction)

## The three

| | Cloud sandbox | Linux desktop | MacBook M4 |
|---|---|---|---|
| Docker | no | yes | yes |
| GPU | no | RTX 5090 | Metal, unified memory |
| llama-server | none | containerised | native |
| MongoDB | none | containerised | containerised |
| Role | write and unit-test everything | integration and model verification | the final target |

**The sandbox** runs the whole offline suite, the corpus tier over the real manual, and the wire
contract of the HTTP adapters against an in-process `http.server`. It has no Docker daemon, no
GPU, no `/mnt/ai-data`, and no services. It can prove that the code is right about its own logic
and about what it puts on a socket. It can prove nothing about a model or a server.

**The desktop** is where `$search`, `$vectorSearch`, real embeddings, and a real Gemma 4 tool
round trip happen for the first time. Sixteen entries across four runbooks —
[deployment](../verification/desktop-deployment.md), [search](../verification/desktop-search.md),
[the model](../verification/desktop-model.md), [scale](../verification/desktop-scale.md) — need
this machine and only this machine.

**The MacBook** is the target. It proves that extraction worked and that the tool runs where it
is meant to run. Its distinct risks are the ARM images and Metal rather than CUDA; see
[../verification/macbook.md](../verification/macbook.md).

## Nothing branches on the machine

There is no hostname check, no platform check, and no "if we are in the sandbox" anywhere in this
project. What varies is which implementation of each port is constructed, and that is decided by
four environment variables read in `logtriage/config.py`.

Unset means "use the offline implementation". That is why a bare checkout works, and it is also
why a machine that is *supposed* to have services but has an unset variable gets a quietly
degraded run rather than a failure — so a command explicitly asked for a served backend
(`--llm llama`, `--embedder llama`) fails by name instead.

The one thing that is not re-selectable at run time is the embedder, because an index is only
searchable by the embedder that built it. `index-kb` writes that identity into the index and
`triage --kb` rebuilds the one named there.

## The evidence rule

**Never claim to have run a command that needs a machine you are not on, and never fabricate its
output.**

When a task needs a machine you are not on, print the exact command for the human and stop.
Their pasted output is the authoritative evidence; record it verbatim, with the command that
produced it, in the runbook entry it belongs to.

This is why [../verification/](../verification/) is written as runbooks with commands and blank
Observed blocks, rather than as prose claiming things work.

## Extraction

The MacBook gets this project as its own repository, not as a checkout of the surrounding one:

```bash
$ cp -r logtriage/ ~/logtriage && cd ~/logtriage && git init
$ uv venv --python 3.12 && uv sync && make quality
```

That copy has no parent, which is what makes it a real test of self-containment. It carries its
own `pyproject.toml`, virtualenv, quality gate, style guides, agent policy, and documentation
gate. If the gate passes there, the project is self-governing; if it fails on a path that only
exists upstream, it is not, and the failure names it.

Two things deliberately do not travel, and their absence is correct:

- `tests/logtriage/test_conformance.py` (in the parent) polices the copied `Printer` and the
  copied style guides against their originals. An extracted copy has no second version to
  disagree with.
- `docker-compose.logtriage.yaml` and `scripts/logtriage/` live in the parent repository, with
  the other compose files. A MacBook deployment copies the compose file alongside the project;
  [stack.md](stack.md) is what it is copied against.

## Related

- [stack.md](stack.md) — the services themselves
- [../verification/README.md](../verification/README.md) — the campaign and its protocol
- `../../AGENTS.md` — the standing policy this rule comes from
