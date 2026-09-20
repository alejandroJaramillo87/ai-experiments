# scripts/docs

The documentation gates and generators: the scripts that fail `make quality` when a
document and the tree disagree, and the ones that render committed artifacts (reference
pages, README inventory tables) from the tree so the gates have something exact to hold
them to. Thirteen run in the gate on every push; three are generators run by hand or by
`make site-gen`.

The strategy these scripts enforce is written down once, in
[../../docs/testing/doc-gates.md](../../docs/testing/doc-gates.md). This README is the
reference for the scripts themselves.

## Conventions every script follows

- **Loaded by path, never imported.** `scripts/` is not a package. Each script sets
  `REPO_ROOT = Path(__file__).resolve().parent.parent.parent`, so moving one to another
  depth breaks it, and the tests under `tests/docs/` load them the same way. Gates
  duplicate a small regex rather than import each other; the one exception is
  `check_inventory.py`, which loads `gen_inventory.py` by path.
- **Output through `common/output.py`'s `Printer`.** One `[INFO]` census line, then
  `[OK]` or `[ERROR]` with two-space-indented findings as `path:line: detail`, ending in
  `Solution:` and, where one exists, `Command:`. Status goes to stderr; `NO_COLOR` is
  honoured.
- **Exit 0 clean, 1 on findings.** Nothing else.
- **The empty-set guard.** A gate that enumerated zero inputs fails, because a gate that
  silently checked nothing is indistinguishable from one that passed. Gates resolving
  against a table guard an empty table too, which would fail every reference rather than
  none.
- **Frozen records are exempt by path.** Anything under `investigations/`, `archive/` or
  `vendor/`, and any file ending `_investigation.md`, is never scanned; the documentation
  style guide makes that the definition of a frozen record. `docs/verification/` is not
  exempt.
- **One `--root`** (or `--pyproject` for the pin-parity gate) so every gate runs on a
  scratch tree under test.

## Generators and their checkers

Three pairs share one pattern: the artifact is committed so its reader needs no toolchain,
and the checker is the reminder that it drifted.

| Generator | Checker | Artifact |
|-----------|---------|----------|
| `gen_site_reference.py` (needs cargo) | `check_site_reference.py` (regex over the Rust `Flag` literals) | `site/docs/rust-harness/reference/*.md` |
| `gen_site_interp_reference.py` (imports the interp parser) | `check_site_interp_reference.py` (regex over the argparse literals) | `site/docs/interp/reference/*.md` |
| `gen_inventory.py` | `check_inventory.py` (runs the generator's `--check`) | every README table between `<!-- inventory:<kind>:start -->` and its end marker |

`make site-gen` runs the two site generators and `make site-check` diffs the result on the
workstation. The inventory generator is run by hand on the README it changed:

```bash
python scripts/docs/gen_inventory.py training/README.md
python scripts/docs/gen_inventory.py            # every README carrying a marker
```

## What to edit when

The gates read hand-maintained tables inside the scripts. This is the list of them, by
the change that touches each:

| When you | Edit |
|----------|------|
| rename, move or delete a file, a flag, a make target or a runbook entry id | `RETIRED` in `check_retired_names.py`: old name to `(replacement, why)`. The replacement must never match the pattern that retired it |
| document a path that is absent on purpose | `KNOWN_ABSENT` in `check_doc_refs.py`, a `(document, token)` pair with its reason |
| document a `make` target that is absent on purpose | `KNOWN_ABSENT` in `check_make_targets.py`, the same shape |
| add a Makefile | `MAKEFILES` or `MAKEFILE_GLOBS` in `check_make_targets.py` |
| add a known-bugs ledger | `LEDGERS` in `check_kb_refs.py` |
| add a directory under `tests/`, `training/`, `common/`, `scripts/`, `interp/` or `rust-harness/` | a README with an inventory marker pair; the hub rule in `check_inventory.py` names it until one exists |
| add a directory whose children must all be documented | `HUB_ROOTS` in `check_inventory.py` |
| add a directory or a page under `docs/` | nothing here: the README carries `dirs` when the directory has subdirectories and `pages` when it has pages, and the docs rule in `check_inventory.py` names what is missing |
| add a Rust source or test file under `rust-harness/` | nothing here: open it with the `/* name — sentence */` banner `.styles/RUST_STYLE.md` requires; the crate's `rust` or `rust_tests` table reads the sentence and counts the test attributes |
| add a script | nothing here: give it a docstring or a leading header comment; the directory's `scripts` table reads the first sentence |
| add a contract under `schemas/` | nothing here: its `description`, `writers` and `readers` render the `contracts` row |
| add an agent, command or skill under `.claude/` | nothing here: its frontmatter `description` renders the roster row in `docs/agents/README.md`; `ROSTER_SOURCES` in `gen_inventory.py` names the three places |
| add a make fragment or a target | nothing here: the `##@` section line and the `##` help comment render the `fragments` row in `make/README.md`, the same two comments `make help` reads |
| add a rust-harness binary or an interp verb | nothing here: both are discovered; run `make site-gen` (and, for a binary, add its page to `site/sidebars.js`) |
| add a file that legitimately quotes retired names or KB ids | `EXEMPT_FILES` in the gate it trips, with a comment; prefer changing the fixture |

## Scripts

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `check_command_prompts.py` | Fail the quality gate when a command block carries a shell prompt marker. |
| `check_doc_counts.py` | Fail the quality gate when a document's stated count disagrees with itself. |
| `check_doc_refs.py` | Fail the quality gate when a document cites a repository path that does not exist. |
| `check_group_pin_parity.py` | Fail the quality gate when one package carries two different pins. |
| `check_inventory.py` | Fail the quality gate when a generated README table and the tree disagree. |
| `check_kb_refs.py` | Fail the quality gate when a KB identifier does not resolve to a ledger row. |
| `check_make_targets.py` | Fail the quality gate when a document names a `make` target that does not exist. |
| `check_retired_names.py` | Fail the quality gate when a retired name is still in use anywhere in the tree. |
| `check_runbook_entries.py` | Fail the quality gate when a runbook entry is not well-formed. |
| `check_runbook_ids.py` | Fail the quality gate when a runbook's entry ids and section order disagree. |
| `check_site_interp_reference.py` | Fail the quality gate when a generated interp reference page and its module disagree. |
| `check_site_reference.py` | Fail the quality gate when a generated site reference page and its binary disagree. |
| `check_surface_census.py` | Fail the quality gate when a shipped rust-harness surface appears in no runbook. |
| `gen_inventory.py` | Render the inventory tables in a README from the tree the README describes. |
| `gen_site_interp_reference.py` | Render site/docs/interp/reference/<verb>.md from the interp CLI's own parser. |
| `gen_site_reference.py` | Render site/docs/rust-harness/reference/<binary>.md from each binary's --help. |

<!-- inventory:modules:end -->

Each is covered by the same-named test under [../../tests/docs/](../../tests/docs/README.md),
which pins the classifier's precision on the real tokens that motivated each rule and ends
with the gate run over the whole repository.
