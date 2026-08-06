# Quality gate

`make quality` is the pre-push gate, mandatory for humans and agents alike. This document
covers what it runs, and how to invoke the same tools by hand inside a contained agent
sandbox, where the wrapper commands do not behave the way they do on the workstation.

The recipe in `make/quality.mk` is the source of truth for the step list. When this document
disagrees with it, the recipe is right.

## What it runs

Seven steps, cheapest first, failing on the first error:

1. `ruff format --check .` — formatting.
2. `ruff check .` — lint.
3. `mypy` — types.
4. `uv lock --check` — the lockfile matches `pyproject.toml`.
5. `scripts/docs/check_doc_refs.py` — fails when a document names a repository path that
   does not exist.
6. `scripts/docs/check_retired_names.py` — fails when anything still uses a name that was
   deliberately retired, and names the replacement.
7. `shellcheck --severity=warning` — shell scripts, enumerated with `find`.

It then chains the rust-harness gate when that directory exists. That half is documented in
[../../rust-harness/README.md](../../rust-harness/README.md): its five steps, the
cross-language contract tests, `runtime/src/pyenv.rs` interpreter resolution and
`HARNESS_PYTHON`, and why a green run must be read for `[SKIP]` lines rather than trusted by
exit code. It is not repeated here.

Steps 5 and 6 are complementary rather than redundant. The reference check asks whether a
cited path exists, which requires classifying a token as a path claim — and that classifier
requires a slash, so a bare renamed filename slips past it. The retired-names check inverts
the question: an explicit table of retired names, failing on any use. Loosening the first to
cover the second was measured at a 67% false-positive rate, because prose legitimately says
`config.json` and `main.rs`.

Both carry an empty-set guard, as the shellcheck step does: enumerating zero files fails
loudly, because a gate that silently checked nothing is indistinguishable from one that
passed.

## Two environments that disagree

`make quality` assumes the workstation. In a contained agent sandbox
(`make agent-sandbox`), invoke the pinned tools **directly**:

```bash
$ ruff format --check .
$ ruff check .
$ mypy --no-incremental
$ shellcheck --severity=warning <files>
$ pytest
```

Not through `uv run --no-sync`. The sandbox image (`docker/agents/Dockerfile.agents`) bakes
in uv 0.11.28, CPython 3.12.11, and the exact dev-group pins — ruff 0.15.21, mypy 2.3.0,
types-pyyaml 6.0.12.20250915 — so a direct invocation matches the workstation.
`uv run --no-sync` does not: it resolves whatever `.venv` happens to hold rather than the
pins. In a worktree that `.venv` is the *workstation's*, arriving over the bind mount and
pointing at a host interpreter absent from the container — and any `uv sync` against it
writes back through the mount to the workstation.

Prefer `mypy --no-incremental` when the result matters: a warm `.mypy_cache` makes the
incremental run non-deterministic, and the same command can pass incrementally while failing
under `--no-incremental`.

`make test` cannot run in the sandbox because it reaches the pytorch cu130 index the egress
allowlist does not carry. Invoke `pytest` directly instead; the image bakes the `sandbox`
dependency group and pytest into `/opt/pyenv`, so a direct call runs against the
`pyproject.toml` pins.

The rust gate runs in the sandbox too: the image carries the toolchain pinned by
`rust-harness/rust-toolchain.toml`, a prefetched crate registry so no step needs the network,
and numpy and gguf at the `pyproject.toml` pins. `runtime/src/npz.rs` shells out to
`np.load`, so `cargo test` fails outright without numpy rather than skipping. Rebuild the
image with `make agent-sandbox-build` after changing `docker/agents/Dockerfile.agents` or
`rust-harness/Cargo.lock`.

What stays workstation-only is GPU work, the service healthchecks, and anything under
`/mnt/ai-data`. **Re-run `make quality` on the workstation before every push regardless: the
sandbox proves the code, not the hardware.**

## Provisioning

Never repair the environment with a bare `uv sync`. It installs the default groups only, so
it *uninstalls* `omegaconf`, `hydra-core`, `gguf`, and `xxhash`, and every `tests/training`
module then fails collection on a missing import. Use `make install`, which passes
`--all-groups --all-extras`.

Provision a worktree with `make worktree-<name> TIER=full` when it will run the training
suite or GPU work. The default `TIER=sandbox` is torch-free and its torch-gated tests skip
rather than fail, so **check the skip count before reading a green suite as evidence**: about
40 skips instead of about four means a partial run, not a pass.

## Related

- [../../rust-harness/README.md](../../rust-harness/README.md) — the rust half of the gate
- [../../tests/README.md](../../tests/README.md) — the pytest suite, its tiers, and markers
- [../agents/README.md](../agents/README.md) — the agent sandbox itself
