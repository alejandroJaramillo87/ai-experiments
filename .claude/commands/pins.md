---
description: Audit docker/ and every compose file for unpinned image tags and clones (:latest, :main, floating refs)
allowed-tools: Bash(grep:*), Bash(echo:*)
---
Version-pinning audit. The rules: every image tag and every cloned dependency must be pinned
to an immutable ref — no `:latest`, no `:main`, no `git clone` without a pinned branch/tag/SHA.

There are five compose files, not one: `docker-compose.yaml` plus `.agents`, `.interp`,
`.rust-harness`, and `.training`. All are in scope.

Floating image tags (`:latest` / `:main`), excluding digest-pinned references:

!`grep -rnE ':(latest|main)\b' docker/ docker-compose*.yaml | grep -v '@sha256:' || echo "none"`

Digest-pinned tags, shown for confirmation only — these are **correct**, not violations:

!`grep -rnE ':(latest|main)[^ ]*@sha256:' docker/ docker-compose*.yaml || echo "none"`

Unpinned git clones (no --branch/--depth-pinned ref), with two lines of trailing context so a
ref on a continuation line is visible:

!`grep -rnA2 'git clone' docker/ || echo "none"`

Review the matches above and report each violation with its `file:line`.

Judging rules:

- A tag carrying an `@sha256:` digest is immutable and **passes**, whatever the tag text says.
  `image: foo:latest@sha256:abc...` resolves to one specific image forever; the tag is
  decoration. Never report one of these as a violation.
- A bare `:latest` or `:main` with no digest is a violation.
- A match on a comment line (`#`) is documentation, not configuration. The update instructions
  above the open-webui pin name the floating tag on purpose. Do not report these.
- For `git clone`, read the trailing context lines before judging: the pinned
  `--branch <tag>` / `--depth 1` / explicit `git checkout <sha>` frequently sits on the line
  after the clone, not on it. Only report a clone whose ref is absent from the whole block.

If every grep returned `none`, or every match is digest-pinned or ref-pinned on a
continuation line, report the audit as passing. Do not edit files — this is a report-only
audit.
