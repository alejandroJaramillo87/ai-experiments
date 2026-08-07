# tests

The whole-repository pytest suite: the Python tiers by unit and branch tests, the shell
scripts under `scripts/` driven as subprocesses, agent tooling and Make targets, model
header transcripts, and a hardware tier that runs only on the workstation.

## Layout

| Directory | Covers |
|-----------|--------|
| `training/` | the training tier; the largest suite, and the one that needs torch |
| `interp/` | the interpretability tier |
| `utils/` | `common/` and the `scripts/utils/` reporters — see [utils/README.md](utils/README.md) |
| `agents/` | agent tooling, the guard hook, and Make targets — see [agents/README.md](agents/README.md) |
| `models/` | GGUF and safetensors header transcripts — see [models/README.md](models/README.md) |
| `hardware/` | live GPU, CPU, RAM, and SSD checks; deselected by default |

## Traps

**A green suite is not automatically a passing suite.** Read the skip count. Torch-gated
modules call `pytest.importorskip` at import time, so on a torch-free environment they skip
silently rather than fail. `tests/conftest.py` prints the determining fact beside the pytest
summary: `[TIER] sandbox:` means torch was absent and those modules skipped, `[TIER] full:`
means the whole suite was in scope. Provision with `make worktree-<name> TIER=full` when the
training suite matters.

**`pythonpath = ["tests/training"]` in `pyproject.toml` is load-bearing.** `tests/training/support.py`
is imported bare by conftest and about a dozen modules, and because `tests/` and
`tests/training/` both carry `__init__.py`, pytest's default import mode would put the repo
root on `sys.path` and never `tests/training` itself. Collection fails outright without it.

**Hardware tiers are deselected by `addopts`, and a command-line `-m` replaces that
expression rather than adding to it.** So `-m gpu` silently re-enables `hw_smoke` and
`hw_full`. Opt in deliberately with `-m hw_smoke`.

## Entry points

```bash
$ make test                   # the whole suite, in either environment
$ pytest tests/utils -v       # one directory
$ pytest -m hw_smoke          # workstation only
```

`make test` runs the same way in a contained agent sandbox as on the workstation:
`make/quality.mk` branches on the `AI_SANDBOX` variable set by the sandbox image, dropping
the `uv run --no-sync` prefix so pytest comes from `/opt/pyenv`, which was built from
`uv.lock` and therefore carries the workstation's pins.

## Where the real docs live

- [../pyproject.toml](../pyproject.toml) — `[tool.pytest.ini_options]`, marker definitions, and the coverage floor
- [../docs/testing/quality-gate.md](../docs/testing/quality-gate.md) — environment provisioning, the skip-count rule, and the gate itself
- [../AGENTS.md](../AGENTS.md) — the operating rules these follow from
- [../docs/testing/investigations/test_coverage_investigation.md](../docs/testing/investigations/test_coverage_investigation.md) — a dated coverage survey
