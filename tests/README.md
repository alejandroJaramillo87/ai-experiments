# tests

The whole-repository pytest suite: unit and branch tests for the three Python packages,
the shell scripts under `scripts/` driven as subprocesses, the agent tooling and Make
targets, the documentation gates, the model header transcripts, and a hardware tier that
runs only on the workstation. `make test` runs everything but the hardware tier, in the
sandbox and on the workstation alike.

## Layout

One directory per subject. The table is generated from each directory's README, and
`make quality` holds it to the tree, so a test directory without a README fails the gate.

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [agents/](agents/README.md) | Fixtures and tests for the agent guardrail and workflow scripts in `scripts/agents/`. |
| [common/](common/README.md) | Unit tests for the `common/` package: the `Printer` status contract in `common/output.py` and the observability subpackage `common/obs/`. |
| [docs/](docs/README.md) | Unit tests for the documentation gates in `scripts/docs/`, all of which run in `make quality` as standalone scripts as well as under pytest here. |
| [hardware/](hardware/README.md) | The workstation tier: tests that need the RTX 5090, the Ryzen 9950X, host RAM, `/mnt/ai-data`, a real checkpoint or the built rust-harness binary, and are therefore deselected by default and run only where those exist. |
| [interp/](interp/README.md) | The unit and branch tests for the interpretability tier in `interp/`: the probe toolkit and its capture contract, activation harvesting and SAE training, steering, organisms with their seal and locality checks, the report envelope, and every CLI handler driven in process through the real parser. |
| [models/](models/README.md) | Tests for the Gemma 4 header transcribers in `scripts/models/`. |
| [setup/](setup/README.md) | Tests for the one-time provisioning scripts in `scripts/setup/`. |
| [training/](training/README.md) | The unit and branch tests for the training tier in `training/`: recipes and their validation, the data pipeline, the training loop and its resume guarantees, the SFT and export paths, the CLI handlers, and the D9/D10 operational pieces. |
| [utils/](utils/README.md) | Tests for the host-environment reporters in `scripts/utils/`. |

<!-- inventory:dirs:end -->

Four modules are helpers pytest never collects: `training/support.py` (the tiny tokenizer
and synthetic corpus builders) and `training/obs_crash_driver.py`, `models/specbuild.py`
(the declarative fixture specs) and `models/gguf_reference_reader.py` (an independent
GGUF reader). `conftest.py` at this level puts the repository root on `sys.path`, moves
`tmp_path` onto an executable filesystem when the default is `noexec`, prints the `[TIER]`
line, and provides `repo_root`, `fixtures_dir`, `fake_bin` and `run_script`.

## Tiers and markers

The suite has two environment tiers and three hardware tiers.

**Environment tiers** are decided by whether torch imports. Modules whose subject is a
real tensor call `pytest.importorskip` at import time, so on a torch-free environment they
skip rather than fail collection; `tests/conftest.py` prints the determining fact beside
the summary:

```
[TIER] sandbox: torch absent, so torch-gated tests skipped rather than ran. This is a
       partial run, not a pass.
[TIER] full: torch present, the whole suite was in scope.
```

**Hardware tiers** are the three `hw_*` markers in `pyproject.toml`, carried only by
`hardware/`: `hw_smoke` (under five minutes, no model downloads, the session-start gate),
`hw_full` (near-capacity VRAM and real models, per verification phase), and `hw_e4b` (needs
the real Gemma 4 E4B checkpoint on disk, a provisioning precondition carried beside a tier
marker). Each hardware test also carries at least one resource marker: `gpu`, `cpu`, `ram`
or `ssd`. `addopts` deselects all three tiers by default.

## Traps

**A green suite is not automatically a passing suite.** Read the skip count and the
`[TIER]` line. Provision with `make worktree-<name> TIER=full` when the training suite
matters.

**`pythonpath = ["tests/training"]` in `pyproject.toml` is load-bearing.**
`tests/training/support.py` is imported bare by conftest and about a dozen modules, and
because `tests/` and `tests/training/` both carry `__init__.py`, pytest's default import
mode would put the repo root on `sys.path` and never `tests/training` itself. Collection
fails outright without it.

**A command-line `-m` replaces `addopts` rather than adding to it.** So `-m gpu` silently
re-enables all three hardware tiers. Opt in deliberately with `-m hw_smoke`, `-m hw_full`
or `-m hw_e4b`.

## Entry points

```bash
make quality-full           # the gate, the suite, and hw_smoke (workstation only)
make quality-full HW=full   # ... and hw_full
make quality-full HW=e4b HARNESS_E4B_MODEL=/mnt/ai-data/models/hf/gemma-4-E4B-it
make test                   # the whole suite, in either environment
pytest tests/common -v      # one directory
pytest -m hw_smoke          # workstation only
```

`make quality-full` is the only command that covers a hardware tier. It refuses to run in
a sandbox, and refuses on a torch-free environment where the suite would skip its way to
green. `make test` runs the same way in a contained agent sandbox as on the workstation:
`make/quality.mk` branches on the `AI_SANDBOX` variable set by the sandbox image, dropping
the `uv run --no-sync` prefix so pytest comes from `/opt/pyenv`, built from `uv.lock`.

## Where the real docs live

- [../docs/testing/README.md](../docs/testing/README.md) — the testing strategy: the three
  commands, the layers, and what each proves
- [../docs/testing/test-suite.md](../docs/testing/test-suite.md) — pytest as configured:
  markers, the `[TIER]` line, the coverage ratchet, and how these READMEs are generated
- [../docs/testing/hardware-tiers.md](../docs/testing/hardware-tiers.md) — the workstation
  tiers and the sandbox-half/workstation-half pairings
- [../pyproject.toml](../pyproject.toml) — `[tool.pytest.ini_options]`, the marker
  definitions, and the coverage floor
