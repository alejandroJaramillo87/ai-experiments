# common

The shared substrate under the Python tiers: styled CLI output, file hashing and atomic
JSON writes, git provenance, the report envelope, serving helpers, and the observability
package that records what every command did. It imports nothing first-party and sits at
the bottom of the import graph.

This is a contract package, not a utility bin. Several of the shapes defined here are
re-implemented in Rust and checked across the boundary, so a change that looks local is
not: the run-event and catalog records in `obs/`, the report envelope in `reports.py`, and
the chat endpoint `serving.py` calls all have a second writer or reader under
`rust-harness/`. Read `schemas/README.md` before changing a record shape.

## Layout

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [obs/](obs/README.md) | The observability package: structured logging keyed by an ambient run id, the traced dispatch every `training` and `interp` command goes through, the run catalog, retention, and the dashboard that renders a run's metrics. |

<!-- inventory:dirs:end -->

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `hashing.py` | Shared file-hashing and durable-write helpers. |
| `output.py` | Styled CLI output per .styles/PYTHON_OUTPUT_STYLE.md, shared by all package CLIs. |
| `provenance.py` | Repository provenance shared by every manifest writer. |
| `reports.py` | Shared JSON report envelope (the D8-style convention, WI-12). |
| `serving.py` | Serving-stack client and registry shared by host-side tools. |
| `smoke.py` | Version-probe helpers shared by the tier smoke tests. |

<!-- inventory:modules:end -->

## Relationship to the rest of the tree

Consumed by `training/` and `interp/` (both for `Printer`, `obs.dispatch`, hashing and
provenance), by the Python scripts under `scripts/` (every gate under `scripts/docs/`
imports `Printer` after putting the repository root on `sys.path`), by `tests/common/`, and
across the language boundary by the Rust harness: `obs_contract.rs`, `obs_records.rs`,
`serving_contract.rs` and `status_python_parity.rs` each spawn an interpreter against
this package. `training/serving.py` and `interp/manifest.py` deliberately re-export
symbols from here with `# noqa: F401` so the tiers have one import site.

`common.output.Printer` is the canonical implementation of the status vocabulary in
`.styles/PYTHON_OUTPUT_STYLE.md`. Import it; do not copy it. Human lines go to standard
error, standard output is reserved for `--json` payloads, and every status call also writes
a structured record, so a `--json` run still leaves a durable log.

## Entry points

No CLI of its own. It is exercised through every `training` and `interp` command via
`common.obs.dispatch.dispatch`, and directly by:

```bash
pytest tests/common         # the obs and output suites
```

## Where the real docs live

- [../docs/observability/report_schema.md](../docs/observability/report_schema.md) — the shared report envelope
- [../docs/observability/design-and-roadmap.md](../docs/observability/design-and-roadmap.md) — the observability design and its phases
- [../docs/rust-harness/observability.md](../docs/rust-harness/observability.md) — the Rust half of the same contract
- [../schemas/README.md](../schemas/README.md) — the cross-language record contracts
