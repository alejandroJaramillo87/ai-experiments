# Step zero: environment

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

Blocks all other phases. The `interp` and `training` dependency groups carry
provisional pins resolved offline, and the committed `uv.lock` predates the
current `pyproject.toml` groups; nothing downstream is trustworthy until the
lockfile is regenerated on the workstation and both images rebuild and pass
their smoke tests.

## 0.1 Regenerate the lockfile

Implementation: no repository code; the resolver reads `pyproject.toml` and
rewrites `uv.lock`.

Command:

```bash
$ uv lock
$ git diff --stat uv.lock
```

Expected signal: `uv lock` exits 0 after resolving against the pytorch
cu130 index; the diff shows `uv.lock` updated. The `sandbox`, `dev`,
`training`, and `interp` groups all appear in the new lock (spot-check:
`grep -A2 'sandbox = ' uv.lock` lists the group's requirements). Commit the
regenerated lockfile before proceeding. If the `unsloth` pin fails to
resolve against the pinned transformers, record the resolver error here and
remove or repin `unsloth` in a separate commit — it is an optional backend
(F3 decision 2).

Observed:

```
Run on the workstation 2026-08-01; output not recorded.
```

Status: VERIFIED

## 0.2 Sync the host environment

Implementation: the `install` target in `Makefile`
(`uv sync --all-groups --all-extras`), `pyproject.toml`.

Command:

```bash
$ make install
```

Expected signal: exit 0; `uv sync` installs all groups and extras without
building any package against the wrong CUDA. `uv run python -c "import
torch; print(torch.__version__, torch.version.cuda)"` prints the pinned
torch and CUDA 13.0.

Observed:

```
Run on the workstation 2026-08-01; output not recorded.
```

Status: VERIFIED

## 0.3 Quality gate passes end to end

Implementation: the `quality` target in the root `Makefile`,
`pyproject.toml` (ruff and mypy configuration), `rust-harness/Makefile`
(the chained rust gate).

The `uv lock --check` step fails everywhere until 0.1 lands; this entry
proves the full gate on the regenerated lockfile.

Command:

```bash
$ make quality
```

Expected signal: all five steps pass — format check, ruff, mypy,
`uv lock --check`, shellcheck — and the target exits 0.

Observed:

```
Run on the workstation 2026-08-01; output not recorded.
```

Status: VERIFIED

## 0.4 Rebuild the training image

Implementation: `docker-compose.training.yaml`,
`docker/training/Dockerfile.training`, `training/commands/smoke_test.py`
(the `python -m training smoke --build` final layer).

Command:

```bash
$ make train-build
```

Expected signal: build exits 0. The build runs `python -m training smoke
--build` as its final step (docker/training/Dockerfile.training), so a
completed build already implies the import surface, and the CPUAdam and
flash-attn wheels compiled.

Observed:

```
Run on the workstation 2026-08-01; output not recorded.
```

Status: VERIFIED

## 0.5 CPUAdam built with znver5 flags

Implementation: `docker/training/Dockerfile.training` (the `uv sync` line
carrying `CFLAGS="-march=znver5"`); the behavioral check lives in
`tests/hardware/test_cpu.py`.

The Dockerfile compiles DeepSpeed CPUAdam with `-march=znver5` and falls
back to `-march=native` only if gcc rejects znver5 (upstream issue #7556 is
the miscompile this guards). Verify which branch ran.

Command:

```bash
$ docker compose -f docker-compose.training.yaml build training --progress=plain 2>&1 | grep -m1 "znver5"
```

Expected signal: the `uv sync` line with `CFLAGS="-march=znver5"` appears
in the build log (cache hits also print the step). If only the native
fallback ran, record it here — the hw_full CPUAdam step-time bound
(tests/hardware/test_cpu.py) is the behavioral check that the op is not
miscompiled either way.

Observed:

```
Run on the workstation 2026-08-01; output not recorded.
```

Status: VERIFIED

## 0.6 Training smoke test on the workstation

Implementation: `training/commands/smoke_test.py`, `common/smoke.py`,
`docker/training/entrypoint.sh`.

Command:

```bash
$ make train-smoke
```

Expected signal: all eight steps print `[OK]` (CUDA sm_120, imports,
flash-attn forward, bitsandbytes 4-bit, CPUAdam step, torchao float8,
unsloth import or a documented `[WARN]`, versions manifest) and the command
exits 0.

Observed:

```
Run on the workstation 2026-08-01; output not recorded.
```

Status: VERIFIED

## 0.7 Rebuild the interp image and smoke it

Implementation: `docker/interp/Dockerfile.interp`,
`docker-compose.interp.yaml`, `interp/commands/smoke_test.py`,
`common/smoke.py`.

Command:

```bash
$ make interp-build
$ make interp-smoke
```

Expected signal: build exits 0; the seven-step smoke prints `[OK]` for
torch/sm_120, the import set, the TransformerLens hooked forward, the
nnsight trace, and the bf16 forward, and exits 0. The gated Gemma step
requires `HF_TOKEN` in `.env.interp`; a `[WARN]` skip without the token is
acceptable at step zero but must pass before the interp phase.

Observed:

```
Run on the workstation 2026-08-01; output not recorded.
```

Status: VERIFIED

## 0.8 Hardware smoke suite

Implementation: `tests/hardware/test_gpu.py`, `tests/hardware/test_cpu.py`,
`tests/hardware/test_ram.py`, `tests/hardware/test_ssd.py`,
`tests/hardware/conftest.py` (tier gating).

Command:

```bash
$ uv run pytest -m hw_smoke tests/hardware/
```

Expected signal: all selected tests pass — CUDA sm_120, bf16 matmul,
flash-attn or the logged sdpa fallback with a matching metadata stamp,
DeepSpeedCPUAdam import, AVX-512 flags, one CPUAdam step, huge pages,
pinned buffer, /mnt/ai-data layout, staging writability. Zero failures.

Observed:

```
Run on the workstation 2026-08-01; output not recorded.
```

Status: VERIFIED
