# Documentation gate tests

Unit tests for the two documentation gates in `scripts/docs/`, both of which run in
`make quality` as standalone scripts as well as under pytest here.

Each gate has the same failure mode, and it is not the one a reader expects: the risk is
a false positive, not a missed defect. A rule that is slightly too eager reports hundreds
of bad findings, and a gate nobody believes is a gate nobody runs. The tests are weighted
accordingly — most of them pin classification and matching precision rather than coverage.

The sibling `tests/common/` suite covers the `common/` package, and `tests/utils/` covers
the host-environment reporters in `scripts/utils/`.

## Test index

| Test file | Covers | Approach |
|-----------|--------|----------|
| `test_doc_refs.py` | `scripts/docs/check_doc_refs.py` | unit tests for `looks_like_path`, `extract_refs`, `extract_links`, `resolves`, `is_exempt`, each naming the real token that motivated the rule; then the full-tree gate |
| `test_retired_names.py` | `scripts/docs/check_retired_names.py` | `pattern_for` precision, `RETIRED` table invariants, `find_retired` line numbers, `wants_scan`, `is_exempt`; then the full-tree gate |

## Notes

- Both modules are loaded by absolute path rather than imported, because `scripts/` is not
  an importable package. Both derive the repo root from `Path(__file__).resolve().parents[2]`,
  so moving either file to a different depth breaks it.
- `test_retired_names.py` must stay listed in `EXEMPT_FILES` in
  `scripts/docs/check_retired_names.py`. It quotes retired names in its parametrize cases,
  so without the exemption the gate fails on its own test file.
- The single most important assertion is that a replacement never matches the pattern that
  retired it. Get that wrong and the gate can never go green again.
- Each file ends with a test that runs the real gate across the whole repository, so a
  stale path anywhere in live documentation surfaces as a pytest failure and not only as a
  `make quality` failure.
