# common package tests

Unit tests for the `common/` package: the `Printer` status contract in `common/output.py`
and the observability subpackage `common/obs/`. The directory mirrors the package it
tests, one module per source module.

Nothing here touches the GPU, Docker, or a running service. Every test drives the real
code against `tmp_path`, so the whole directory runs in a contained agent sandbox. Two
modules, `test_obs_catalog.py` and `test_obs_retention.py`, drive the training and interp
CLIs in process to produce real catalog rows; that is the `common` contract exercised from
its callers, not a test of either tier.

The sibling `tests/docs/` suite covers the documentation gates in `scripts/docs/`, and
`tests/utils/` covers the host-environment reporters in `scripts/utils/`.

## Test index

Generated from each module's docstring; see [../README.md](../README.md).

<!-- inventory:tests:start -->

| File | Tests | Covers |
|------|-------|--------|
| `test_common_output.py` | 10 | Tests for common/output.py Printer — the shared style contract. |
| `test_obs_audit.py` | 7 | Unit tests for the layer 2 resource audit hook (roadmap OBS-6). |
| `test_obs_catalog.py` | 16 | Unit tests for the run catalog and runs subcommand (roadmap OBS-9). |
| `test_obs_catalog_corpus.py` | 5 | Catalog tests against observability output the rust harness really wrote. |
| `test_obs_context.py` | 6 | Unit tests for run-context capture (roadmap OBS-7). |
| `test_obs_core.py` | 19 | Unit tests for the observability logging core (roadmap OBS-1). |
| `test_obs_dispatch.py` | 9 | Unit tests for the traced CLI dispatch (roadmap OBS-4). |
| `test_obs_failure.py` | 9 | Unit tests for structured failure records (roadmap OBS-5). |
| `test_obs_record_contract.py` | 9 | The Python obs writers against the cross-language record contract. |
| `test_obs_retention.py` | 14 | Unit tests for event-file retention (roadmap OBS-8). |
| `test_obs_runs.py` | 10 | The shared runs command surface (roadmap U5). |
| `test_obs_serve.py` | 17 | Unit tests for the live dashboard server (roadmap OBS-12). |
| `test_run_metrics_contract.py` | 13 | The harness metrics file against the Python readers that consume it. |

<!-- inventory:tests:end -->

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
