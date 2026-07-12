---
description: Audit docker/ and compose for unpinned image tags and clones (:latest, :main, floating refs)
allowed-tools: Bash(grep:*)
---
Version-pinning audit. The rules: every image tag and every cloned dependency must be pinned
to an immutable ref — no `:latest`, no `:main`, no `git clone` without a pinned branch/tag/SHA.

Floating image tags (`:latest` / `:main`):

!`grep -rnE ':(latest|main)\b' docker/ docker-compose.yaml || echo "none"`

Unpinned git clones (no --branch/--depth-pinned ref):

!`grep -rnE 'git clone' docker/ || echo "none"`

Review the matches above. Report each as a violation with its `file:line`, and for git
clones flag any that lack a `--branch <tag>` / pinned checkout. If both greps returned
`none`, report the audit as passing. Do not edit files — this is a report-only audit.
