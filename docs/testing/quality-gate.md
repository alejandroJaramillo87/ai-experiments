# Quality gate

`make quality` is the pre-push gate, mandatory for humans and agents alike. It is the same
command on the workstation and in a contained agent sandbox; this document covers what it
runs, and what a sandbox run does not cover.

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

## One gate, two environments

Run `make quality` the same way in both. The difference is resolved inside `make/quality.mk`,
which reads the `AI_SANDBOX` marker the sandbox image sets:

- **Workstation** — tools come from `.venv` via `uv run --no-sync`, which runs what is already
  provisioned rather than re-resolving, so no step touches the network.
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
needs the network, and numpy and gguf at the `uv.lock` pins. `runtime/src/npz.rs` shells out
to `np.load`, so `cargo test` fails outright without numpy rather than skipping. Rebuild the
image with `make agent-sandbox-build` after changing `docker/agents/Dockerfile.agents`,
`uv.lock`, or `rust-harness/Cargo.lock`.

Prefer `mypy --no-incremental` when the result matters: a warm `.mypy_cache` makes the
incremental run non-deterministic, and the same command can pass incrementally while failing
under `--no-incremental`.

What stays workstation-only is the lock check above, GPU work, the service healthchecks, and
anything under `/mnt/ai-data`. **Re-run `make quality` on the workstation before every push
regardless: the sandbox proves the code, not the hardware.**

## Provisioning

`make install` is the full provision (`--all-groups --all-extras`). A bare `uv sync` is a safe
repair: `pyproject.toml` names all four dependency groups in `[tool.uv] default-groups`
precisely so the reflexive command cannot strip an environment down.

Provision a worktree with `make worktree-<name> TIER=full` when it will run the training suite
or GPU work. The default `TIER=sandbox` is torch-free, and its torch-gated tests skip rather
than fail — so a green partial run looks like a pass. The suite states which it was, beside
the summary:

```
[TIER] sandbox: torch absent, so torch-gated tests skipped rather than ran. This is a
       partial run, not a pass.
[TIER] full: torch present, the whole suite was in scope.
```

## Related

- [../../rust-harness/README.md](../../rust-harness/README.md) — the rust half of the gate
- [../../tests/README.md](../../tests/README.md) — the pytest suite, its tiers, and markers
- [../agents/README.md](../agents/README.md) — the agent sandbox itself
