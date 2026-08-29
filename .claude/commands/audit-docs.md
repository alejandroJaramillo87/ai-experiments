---
description: Sweep a docs subtree against the Dockerfile/compose truth and return a drift list
argument-hint: <docs-path>
agent: doc-auditor
subtask: true
allowed-tools: Read, Glob, Grep, Bash, Task
---
Delegate to the doc-auditor subagent (isolated context). Detection and repair are
separate roles: doc-auditor reports the drift, doc-writer fixes it in a later pass.

Audit the documentation under `$ARGUMENTS` against the actual source of truth:

- `docker-compose.inference.yaml` — the default stack (llama-cpu, llama-gpu, vllm-gpu,
  open-webui). It declares no compose profiles.
- `docker-compose.agents.yaml`, `docker-compose.interp.yaml`,
  `docker-compose.rust-harness.yaml`, `docker-compose.training.yaml` — the four additional
  stacks. Docs under `docs/training/`, `docs/interp/`, and `docs/sandbox/` are
  governed by these, **not** by the root compose file; auditing them against the root file
  alone checks the wrong source.
- `docker/*/Dockerfile*`, and `Makefile` for any documented target.

Check every factual claim in those docs — service names, external/internal ports, image
tags and versions, mounted paths, environment variables, and commands — against what the
compose file and Dockerfiles actually declare.

Return a **drift list only**: each entry is `file:line — documented X, actual Y`. Do not
rewrite the docs in this pass; report the discrepancies so they can be fixed deliberately.
Follow `.styles/DOCUMENTATION_STYLE.md`. If there is no drift, say so explicitly.
