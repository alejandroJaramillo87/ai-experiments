# Documentation gate tests

Unit tests for the documentation gates in `scripts/docs/`, all of which run in
`make quality` as standalone scripts as well as under pytest here.

Most of these gates share a failure mode, and it is not the one a reader expects: the risk
is a false positive, not a missed defect. A rule that is slightly too eager reports hundreds
of bad findings, and a gate nobody believes is a gate nobody runs. The tests are weighted
accordingly — most of them pin classification and matching precision rather than coverage.

Two are exceptions worth knowing. `check_retired_names.py` has no false positives by
construction, because it matches an explicit table rather than a heuristic.
`check_doc_counts.py` has none either, for a different reason: it compares a document
against itself, so it cannot be wrong about what the right answer is.

The sibling `tests/common/` suite covers the `common/` package, and `tests/utils/` covers
the host-environment reporters in `scripts/utils/`.

## Test index

Generated from each module's docstring; see [../README.md](../README.md).

<!-- inventory:tests:start -->

| File | Tests | Covers |
|------|-------|--------|
| `test_check_site_interp_reference.py` | 9 | Tests for scripts/docs/check_site_interp_reference.py. |
| `test_check_site_reference.py` | 10 | Tests for scripts/docs/check_site_reference.py. |
| `test_command_prompts.py` | 16 | Tests for scripts/docs/check_command_prompts.py. |
| `test_doc_counts.py` | 12 | Tests for scripts/docs/check_doc_counts.py. |
| `test_doc_refs.py` | 14 | Tests for scripts/docs/check_doc_refs.py. |
| `test_doc_tallies.py` | 16 | Tests for the runbook-tally half of scripts/docs/check_doc_counts.py. |
| `test_gen_inventory.py` | 48 | Tests for scripts/docs/gen_inventory.py and its gate. |
| `test_gen_site_interp_reference.py` | 8 | Tests for scripts/docs/gen_site_interp_reference.py. |
| `test_gen_site_reference.py` | 13 | Tests for scripts/docs/gen_site_reference.py. |
| `test_group_pin_parity.py` | 7 | Tests for scripts/docs/check_group_pin_parity.py. |
| `test_kb_refs.py` | 14 | Tests for scripts/docs/check_kb_refs.py. |
| `test_make_targets.py` | 14 | Tests for scripts/docs/check_make_targets.py. |
| `test_retired_names.py` | 15 | Tests for scripts/docs/check_retired_names.py. |
| `test_runbook_entries.py` | 13 | Tests for scripts/docs/check_runbook_entries.py. |
| `test_runbook_ids.py` | 26 | Tests for scripts/docs/check_runbook_ids.py. |
| `test_surface_census.py` | 7 | Tests for scripts/docs/check_surface_census.py. |

<!-- inventory:tests:end -->

## Notes

- Every module is loaded by absolute path rather than imported, because `scripts/` is not
  an importable package. Each derives the repo root from `Path(__file__).resolve().parents[2]`,
  so moving any of these files to a different depth breaks it.
- `test_retired_names.py` must stay listed in `EXEMPT_FILES` in
  `scripts/docs/check_retired_names.py`, and `test_kb_refs.py` in the same table in
  `scripts/docs/check_kb_refs.py`. Both quote the very strings their gate forbids, so
  without the exemption each gate fails on its own test file. A test fixture that happens
  to use the pattern counts too: `test_doc_counts.py` originally used known-bug identifiers
  as its sample table data and failed the KB gate, which was fixed by changing the fixture
  rather than by widening an exemption. Prefer that order — an exemption added to make a
  fixture convenient is a hole in the gate that outlives the fixture.
- The single most important assertion is that a replacement never matches the pattern that
  retired it. Get that wrong and the gate can never go green again.
- Each file ends with a test that runs the real gate across the whole repository, so a
  stale path anywhere in live documentation surfaces as a pytest failure and not only as a
  `make quality` failure.
