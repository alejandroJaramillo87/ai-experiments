# Quality gate

`make quality` is the pre-push gate, mandatory for humans and agents alike. It is the same
command on the workstation and in a contained agent sandbox; this document covers what it
runs, and what a sandbox run does not cover. It never runs pytest: `make test` is the suite
([test-suite.md](test-suite.md)) and `make quality-full` is the workstation superset that
adds the suite and the hardware tiers ([hardware-tiers.md](hardware-tiers.md)).

The recipe in `make/quality.mk` is the source of truth for the step list. When this document
disagrees with it, the recipe is right.

## Table of Contents

<!-- toc:start -->

1. [What it runs](#what-it-runs)
2. [One gate, two environments](#one-gate-two-environments)
3. [Provisioning](#provisioning)
4. [Related](#related)

<!-- toc:end -->

## What it runs

Nineteen steps, cheapest first, failing on the first error:

1. `ruff format --check .` — formatting.
2. `ruff check .` — lint.
3. `mypy` — types, over the three first-party packages `pyproject.toml` names.
4. `uv lock --check` — the lockfile matches `pyproject.toml`.
5. `scripts/docs/check_doc_refs.py` — fails when a document names a repository path that
   does not exist.
6. `scripts/docs/check_retired_names.py` — fails when anything still uses a name that was
   deliberately retired, and names the replacement.
7. `scripts/docs/check_make_targets.py` — fails when a documented `make <target>` does not
   resolve, reading invocations from fenced blocks and code spans only.
8. `scripts/docs/check_doc_counts.py` — compares a document against itself, so a heading
   reading `## Open (3)` must have three rows beneath it, and a runbook's tally line must
   match its `Status:` lines.
9. `scripts/docs/check_kb_refs.py` — fails when a known-bug id cited in source has no ledger
   row, or the reverse.
10. `scripts/docs/check_runbook_entries.py` — fails when a verification entry names a test
    function that does not exist, or is marked `VERIFIED` with neither pasted evidence nor a
    named check.
11. `scripts/docs/check_runbook_ids.py` — fails when a runbook's entry ids and its Table of
    Contents disagree: an id heading two entries or reappearing after retirement, a slug
    runbook mixing in numeric ids, a ToC listing the entries in another order than the
    headings, or an entry depending on one that comes later.
12. `scripts/docs/check_surface_census.py` — enumerates the rust-harness binaries and flags
    from their `Flag` literals and fails when one is named in no runbook. The only gate here
    that starts from the code rather than the documentation.
13. `scripts/docs/check_group_pin_parity.py` — fails when one package carries two different
    exact pins across `pyproject.toml`'s sections. The `sandbox` group deliberately re-pins
    what core, `interp` and `training` pin, and step 4 cannot see a disagreement between two
    literal `==` strings.
14. `scripts/docs/check_command_prompts.py` — fails when a `bash` (or `sh`, `shell`, `zsh`)
    block carries a `$ ` prompt marker, because a pasted `$ make quality` fails as a command
    named `$`. Reads neither `console` transcripts nor a runbook's Observed blocks, and skips
    frozen records.
15. `scripts/docs/check_site_reference.py` — fails when a generated reference page under
    `site/docs/rust-harness/reference/` disagrees with the flag table it was rendered from:
    a binary with no page, a page for no binary, or a flag its page does not show. Runs
    without cargo or node; `make site-check` is the workstation form that regenerates and
    diffs.
16. `scripts/docs/check_site_interp_reference.py` — the same check for the interp verbs
    under `site/docs/interp/reference/`, whose pages are rendered from the CLI's argparse
    parser. Reads the command modules as text, so it needs no Python environment beyond
    the interpreter.
17. `scripts/docs/check_inventory.py` — fails when a generated README table disagrees
    with the tree it was rendered from: a test directory's table with its modules'
    docstrings and test counts, a package's with its modules, a scripts directory's with
    its header comments, a hub's with its children's READMEs, a documentation directory's
    with its pages, `schemas/` with its contracts, `docs/agents/` with the definitions
    under `.claude/`. It regenerates each table in memory with
    `scripts/docs/gen_inventory.py` and prints the diff, and fails when a directory under
    `tests/`, `training/`, `common/`, `scripts/`, `interp/` or `rust-harness/` has no README
    with a table, or a directory under `docs/` lacks the table its shape needs.
18. `scripts/docs/check_toc.py` — fails when a document's generated table of contents
    disagrees with its H2 and H3 headings, or when a living document over 500 words with
    four or more sections carries none. It regenerates the list in memory with
    `scripts/docs/gen_toc.py` and prints the diff; frozen records, READMEs, the
    definitions under `.claude/` and the site's pages are never asked for one.
19. `shellcheck --severity=warning` — shell scripts, enumerated with `find`.

Then two chained gates. The rust-harness gate runs when that directory exists,
`make -C rust-harness quality`: five offline steps and the cross-language contract tests,
documented in [../rust-harness/testing.md](../rust-harness/testing.md) and not repeated
here, with the rule that a green run must be read for `[SKIP]` lines rather than trusted by
exit code. After it, `make site-build` builds the documentation site under `site/` and skips
itself with `[SKIP]` when `site/node_modules` is absent, as in a fresh sandbox, where
installing them would need the network. With the dependencies present the build is offline
and fails on a broken link or anchor.

Steps 5 and 6 are complementary rather than redundant. The reference check asks whether a
cited path exists, which requires classifying a token as a path claim — and that classifier
requires a slash, so a bare renamed filename slips past it. The retired-names check inverts
the question: an explicit table of retired names, failing on any use. Loosening the first to
cover the second was measured at a 67% false-positive rate, because prose legitimately says
`config.json` and `main.rs`. The documentation gates as a family, and the strategy they
enforce, are in [doc-gates.md](doc-gates.md).

Every gate carries an empty-set guard, as the shellcheck step does: enumerating zero files
fails loudly, because a gate that silently checked nothing is indistinguishable from one that
passed.

## One gate, two environments

Run `make quality` the same way in both. The difference is resolved inside `make/quality.mk`,
which reads the `AI_SANDBOX` marker the sandbox image sets:

- **Workstation** — tools come from `.venv` via `uv run --no-sync`, which runs what is already
  provisioned rather than re-resolving, so no step touches the network. A preflight refuses
  the run when `.venv` is missing or points at an interpreter that no longer exists, because
  `uv --no-sync` would otherwise recreate it empty. The preflight proves the interpreter
  exists, not that the installed set is the locked set: `--no-sync` never reconciles the two,
  and the hardware smoke tier's `test_venv_lock.py` is what names a package the lock has
  dropped.
- **Sandbox** — tools are invoked from `PATH`. The image builds `/opt/pyenv` with
  `uv sync --frozen --only-group dev --only-group sandbox` against `uv.lock`, so the versions
  are the workstation's by construction rather than by transcription. There is no `.venv` to
  confuse matters: `docker-compose.agents.yaml` masks it with a tmpfs, so the workstation's
  venv is neither readable nor writable from inside.

One step cannot run in a sandbox, and says so rather than being silently absent:

```
[SKIP] uv lock --check: resolving the lock reaches the pytorch cu130 index, which the
       sandbox egress allowlist does not carry. The workstation gate covers it.
```

Everything else runs, including `make test` and the whole rust gate — the image carries the
toolchain pinned by `rust-harness/rust-toolchain.toml`, a prefetched crate registry so no step
needs the network, and numpy at the `uv.lock` pin. `runtime/src/npz.rs` shells out to
`np.load`, so `cargo test` fails outright without numpy rather than skipping. Rebuild the
image with `make agent-sandbox-build` after changing `docker/agents/Dockerfile.agents`,
`uv.lock`, or `rust-harness/Cargo.lock`.

Prefer `mypy --no-incremental` when the result matters: a warm `.mypy_cache` makes the
incremental run non-deterministic, and the same command can pass incrementally while failing
under `--no-incremental`.

What stays workstation-only is the lock check above, the site build in a fresh sandbox, GPU
work, the service healthchecks, and anything under `/mnt/ai-data`. **Re-run `make quality`
on the workstation before every push regardless: the sandbox proves the code, not the
hardware.**

## Provisioning

`make install` is the full provision (`--all-groups --all-extras`), and the repair whenever
the smoke tier reports drift: sync removes what the lock no longer lists. A bare `uv sync` is
a safe repair: `pyproject.toml` names all four dependency groups in `[tool.uv] default-groups`
precisely so the reflexive command cannot strip an environment down.

Provision a worktree with `make worktree-<name> TIER=full` when it will run the training suite
or GPU work. The default `TIER=sandbox` is torch-free, and its torch-gated tests skip rather
than fail — so a green partial run looks like a pass. The suite states which it was beside
the summary, and `make quality-full` refuses the torch-free tier outright
([test-suite.md](test-suite.md)).

## Related

- [README.md](README.md) — the testing strategy this gate is one layer of
- [../rust-harness/testing.md](../rust-harness/testing.md) — the rust half of the gate
- [../agents/README.md](../agents/README.md) — the agent sandbox itself
