# common package tests

Unit tests for the `common/` package: the `Printer` status contract in `common/output.py`
and the observability subpackage `common/obs/`. The directory mirrors the package it
tests, one module per source module.

Nothing here touches the GPU, Docker, or a running service. Every test drives the real
code against `tmp_path`, so the whole directory runs in a contained agent sandbox.

The sibling `tests/docs/` suite covers the documentation gates in `scripts/docs/`, and
`tests/utils/` covers the host-environment reporters in `scripts/utils/`.

## Test index

| Test file | Covers | Roadmap item |
|-----------|--------|--------------|
| `test_common_output.py` | `common/output.py` `Printer` wire format | OBS-3 |
| `test_obs_core.py` | `common/obs/core.py` structured logging core | OBS-1 |
| `test_obs_dispatch.py` | `common/obs/dispatch.py` traced CLI dispatch | OBS-4 |
| `test_obs_failure.py` | structured failure records | OBS-5 |
| `test_obs_audit.py` | the layer 2 resource audit hook | OBS-6 |
| `test_obs_context.py` | run-context capture | OBS-7 |
| `test_obs_retention.py` | event-file retention | OBS-8 |
| `test_obs_catalog.py` | the run catalog and `runs` subcommand | OBS-9 |
| `test_obs_serve.py` | the live dashboard server | OBS-12 |
| `test_obs_record_contract.py` | the Python writers against `schemas/obs_record.json` | — |

The roadmap items are specified in
[../../docs/observability/design-and-roadmap.md](../../docs/observability/design-and-roadmap.md).

## Notes

- `test_obs_record_contract.py` is one half of a cross-language pair. The Rust half is
  `rust-harness/harness/tests/obs_contract.rs`, and both assert against the same key set
  in `schemas/obs_record.json`. Changing a record shape means changing all three.
- The contract is asserted at the writers, not at the reader, because `read_runs` skips a
  line it cannot parse rather than raising. A drifted writer therefore presents as an
  empty runs table, which is indistinguishable from a machine that never ran the tool.
- Imports of `common` resolve through the repo root, which `tests/conftest.py` inserts on
  `sys.path` for the whole suite. This directory carries no `conftest.py` of its own.
- Python human status writes to standard error and standard output is reserved for
  `--json` payloads, so assertions on status lines read `capsys` stderr.
