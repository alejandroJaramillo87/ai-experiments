---
description: Sweep a docs subtree against the Dockerfile/compose truth and return a drift list
argument-hint: <docs-path>
agent: doc-auditor
subtask: true
allowed-tools: Read, Glob, Grep, Bash
---
Delegate to the doc-auditor subagent (isolated context). Detection and repair are
separate roles: doc-auditor reports the drift, doc-writer fixes it in a later pass.

Audit the documentation under `$ARGUMENTS` against the actual source of truth:
`docker-compose.yaml` and `docker/*/Dockerfile*`.

Check every factual claim in those docs — service names, external/internal ports, image
tags and versions, mounted paths, environment variables, and commands — against what the
compose file and Dockerfiles actually declare.

Return a **drift list only**: each entry is `file:line — documented X, actual Y`. Do not
rewrite the docs in this pass; report the discrepancies so they can be fixed deliberately.
Follow `.styles/DOCUMENTATION_STYLE.md`. If there is no drift, say so explicitly.
