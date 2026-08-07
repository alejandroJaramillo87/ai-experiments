# common

Shared substrate for the Python tiers: styled CLI output, file hashing and atomic JSON
writes, git provenance, the shared report envelope, serving helpers, and the observability
stack that records what every command did.

This is a contract package, not a utility bin. Several of the shapes defined here are
re-implemented in another language and checked across the boundary, so a change that looks
local is not. `common/obs/core.py` defines the run-event and catalog record shapes that
`rust-harness/harness/src/obs.rs` writes independently; `rust-harness/harness/tests/obs_contract.rs`
feeds a Rust-written catalog to `common.obs.catalog.read_runs` to prove the two still agree.
Read `schemas/README.md` before changing a record shape.

## Relationship to the rest of the tree

Imports nothing first-party — `common` sits at the bottom. Consumed by `training/` and
`interp/` (both for `Printer`, `obs.dispatch`, hashing, and provenance), by `tests/utils/`,
and across the language boundary by the Rust harness. `training/serving.py` and
`interp/manifest.py` deliberately re-export symbols from here with `# noqa: F401` so the
tiers have one import site.

`common.output.Printer` is the canonical implementation of the status vocabulary in
`.styles/PYTHON_OUTPUT_STYLE.md`. Import it; do not copy it. Human lines go to standard
error, standard output is reserved for `--json` payloads, and every status call also writes
a structured record, so a `--json` run still leaves a durable log.

## Entry points

No CLI of its own. It is exercised through every `training` and `interp` command via
`common.obs.dispatch.dispatch`, and directly by:

```bash
$ pytest tests/utils          # the obs and output suites
```

## Where the real docs live

- [../docs/observability/report_schema.md](../docs/observability/report_schema.md) — the shared report envelope
- [../docs/observability/design-and-roadmap.md](../docs/observability/design-and-roadmap.md) — the observability design and its phases
- [../docs/rust-harness/observability.md](../docs/rust-harness/observability.md) — the Rust half of the same contract
- [../schemas/README.md](../schemas/README.md) — the cross-language record contracts
