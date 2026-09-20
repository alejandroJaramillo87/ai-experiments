# Documentation gates

How the documentation stays true, and the gates that make it so. The documentation here
exists to direct agents and humans through the repository, and a document that has drifted
from the tree directs them wrong with full confidence. So the documentation is held to the
tree the way the code is held to its tests: by rules a script can check on every push.
This page is the strategy, stated once; [../../scripts/docs/README.md](../../scripts/docs/README.md)
is the reference for the scripts.

## The strategy

**One owner per fact.** Every topic has one document that owns it, and `AGENTS.md` is a
directory of owners rather than a reference. A count, a port, a path or a version is stated
in the file that owns it and nowhere else, because a fact stated in two places is a fact
that will disagree with itself. Every other page links the owner.

**Four document classes, declared, not inferred.** A living reference describes the tree
as it is now and is corrected when the tree moves. A frozen record says what was decided or
measured on a day and is never edited; drift is expected and correct. An operational
runbook is executed, not read, and its commands must stay runnable. A parked proposal is a
dated menu of unscheduled work. The class is stated in a bold line under the title
(`.styles/DOCUMENTATION_STYLE.md`), and a frozen record is also findable by path: under an
`investigations/` or `archive/` directory, or named `*_investigation.md`. The gates exempt
exactly those paths, so a frozen record named anything else is gated as though it were
living.

**A frozen record is never a source to audit against.** It accumulates names, formats and
commands that no longer exist, by design. Anything sweeping the tree for stale facts skips
those paths rather than rediscovering the same expired history.

**Every directory has a README that says what it holds, what consumes it, and where the
real docs live.** A source directory's README is navigation: purpose, entry points, the
traps, and a pointer to the living page. It does not restate the living page.

**Anything that lists files is generated from them.** A table of test modules, of a
package's modules, of a directory's scripts, of a hub's directories, of a documentation
directory's pages, of the contracts under `schemas/`, of the agents, commands and skills
under `.claude/`, of the Makefile fragments under `make/`, of a Rust crate's sources and
integration tests or of a build context's Dockerfiles and entrypoints under `docker/` is
rendered by
`scripts/docs/gen_inventory.py` from what each source already carries: a module
docstring's first sentence, a script's header comment, a contract's `description`, a
definition's frontmatter, a fragment's `##@` section and `##` help comments, a Rust
file's leading banner and its test attributes, a Dockerfile's leading `#` block, or the
description paragraph under a README's title. The table is committed so its reader needs no tooling, and the gate regenerates it
in memory and diffs. A directory under a hub without a README, or with a README that
carries no table, fails the gate, which is what makes the hub tables the rule that
everything under them is documented. Under `docs/` the rule is by shape rather than by
hub: every directory has a README, with a `dirs` table when it has subdirectories and a
`pages` table when it has pages, so the router reaches every page. The same pattern renders the site's reference pages
from `--help` and from the argparse parser.

**A rename adds a retired name.** The path gate cannot see a renamed file, because a bare
filename is not a path claim, so every rename, retirement or moved entry id goes into the
table in `scripts/docs/check_retired_names.py` with its replacement and the reason. That
is what makes the next rename self-policing.

**A document is checked against itself where it can be.** A heading that states a count
must have that many rows beneath it; a runbook's tally line must match its `Status:` lines.
Neither check can be wrong about what the right answer is.

**A verified entry needs evidence.** A runbook entry marked `VERIFIED` carries pasted
output or names a check that exists, and the tests it cites resolve to real functions. The
runbooks record what the workstation was observed to do, and the gate keeps an observation
from being asserted without one.

**A gate that enumerates nothing fails.** Every gate refuses an empty input set, and the
ones resolving against a table refuse an empty table, because a gate that silently checked
nothing is indistinguishable from one that passed.

**Command blocks are pasted, not read.** A `bash` block carries no `$ ` prompt marker,
because pasted into a shell the `$` fails as a command; a transcript that must show the
prompt is tagged `console`.

## The gates

Thirteen scripts under `scripts/docs/` run in `make quality`, in this order, cheapest first.
Each fails the build on its finding and prints the file, the line and the remedy.

| Gate | Fails when |
|------|------------|
| `check_doc_refs.py` | a document cites a repository path that does not exist |
| `check_retired_names.py` | a retired name is still in use anywhere in the tree, code included |
| `check_make_targets.py` | a documented `make <target>` does not resolve |
| `check_doc_counts.py` | a heading's count or a runbook's tally disagrees with the rows beneath it |
| `check_kb_refs.py` | a known-bug id cited in source or docs has no ledger row |
| `check_runbook_entries.py` | a verification entry names a test that does not exist, or is `VERIFIED` without evidence |
| `check_runbook_ids.py` | a runbook's entry ids and its Table of Contents disagree |
| `check_surface_census.py` | a rust-harness binary or flag appears in no runbook |
| `check_group_pin_parity.py` | one package carries two different exact pins in `pyproject.toml` |
| `check_command_prompts.py` | a command block carries a shell prompt marker |
| `check_site_reference.py` | a generated rust-harness reference page disagrees with the `Flag` literals it came from |
| `check_site_interp_reference.py` | a generated interp reference page disagrees with the argparse literals it came from |
| `check_inventory.py` | a generated README table disagrees with the tree, or a directory under a hub has no README with one |

Three generators feed them: `gen_site_reference.py` and `gen_site_interp_reference.py`
through `make site-gen`, and `gen_inventory.py` by hand on the README that changed. Each
gate has a test of the same name under `tests/docs/` that pins its classifier on the real
tokens that motivated each rule and ends with the gate run over the whole repository, so a
stale document fails `make test` as well as `make quality`.

## What this costs a contributor

- A new test module: one docstring sentence, then regenerate its README's table.
- A new test directory or training subpackage: a README with a marker pair.
- A rename: one entry in the retired-names table.
- A path or target that is absent on purpose: one allowlist entry with its reason.
- A new documentation page: a description paragraph under its title, which the directory's
  generated index reads.

The tables to edit for each of these are listed in
[../../scripts/docs/README.md](../../scripts/docs/README.md).

## Related

- [README.md](README.md) — the testing strategy these gates are one layer of
- [quality-gate.md](quality-gate.md) — where the gates sit in `make quality`
- [../README.md](../README.md) — the documentation index, and the document classes in
  practice
- [../../.styles/DOCUMENTATION_STYLE.md](../../.styles/DOCUMENTATION_STYLE.md) — the class
  line and the writing standards
