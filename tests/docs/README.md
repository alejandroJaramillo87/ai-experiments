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

| Test file | Covers | Approach |
|-----------|--------|----------|
| `test_doc_refs.py` | `scripts/docs/check_doc_refs.py` | unit tests for `looks_like_path`, `extract_refs`, `extract_links`, `resolves`, `is_exempt`, each naming the real token that motivated the rule; then the full-tree gate |
| `test_retired_names.py` | `scripts/docs/check_retired_names.py` | `pattern_for` precision, `RETIRED` table invariants, `find_retired` line numbers, `wants_scan`, `is_exempt`; then the full-tree gate |
| `test_make_targets.py` | `scripts/docs/check_make_targets.py` | `extract_refs` separating commands from the English verb, `target_exists` pattern-rule matching, `collect_targets` across all three Makefiles, `KNOWN_ABSENT` invariants; then the full-tree gate |
| `test_doc_counts.py` | `scripts/docs/check_doc_counts.py` | `find_sections` heading classification and section extent, `count_body` table-versus-list rules, `is_year`; then the full-tree gate |
| `test_doc_tallies.py` | `scripts/docs/check_doc_counts.py`, its second pattern | `find_tallies` and `count_statuses` against the four shapes that defeat a naive count — a bolded inline status mid-paragraph, a parent entry with no status of its own, non-contiguous entry numbering, and a kind followed by a date and a parenthetical — plus the empty-set guard |
| `test_runbook_entries.py` | `scripts/docs/check_runbook_entries.py` | `find_entries`, `has_evidence` and `find_citations` against the four shapes that defeated the first implementation — a parent entry heading children with their own statuses, an `Observed (` parenthetical whose colon is on a later line, a gate-backed entry with no Observed block at all, and a cited test class rather than a function — plus the empty-set guard |
| `test_surface_census.py` | `scripts/docs/check_surface_census.py` | `surface` enumeration and the `--help` exclusion, whole-word binary matching so prose about layers does not satisfy `layer-diff`, and the empty-set guard, which matters more here than anywhere: an enumeration that collects nothing reports every element as covered |
| `test_kb_refs.py` | `scripts/docs/check_kb_refs.py` | `extract_citations` including the reserved next identifier, `collect_defined` across both ledgers, `is_exempt`; then the full-tree gate |
| `test_command_prompts.py` | `scripts/docs/check_command_prompts.py` | `find_prompts` against the constructs that start a line with `$` or `#` without being a prompt — a variable, a substitution, a comment, a commented-out command, a non-shell block, a `console` transcript — the Observed exemption and where it ends, `is_exempt` for frozen records; then the full-tree gate |
| `test_check_site_reference.py` | `scripts/docs/check_site_reference.py` | `surface` enumeration including `harness`, the generated-marker classification that keeps hand-written reference pages out of the check, the three findings (a binary with no page, a page for no binary, a flag absent from its page) with whole-flag matching so `--control-vector-scale` does not satisfy `--control-vector`, and the empty-set guard; then the full-tree gate |

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
