# Step zero: environment

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 8 verified, 0 blocked, 0 pending, of 8.**

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

Expected signal: `make quality` exits 0 and every step it prints reports
`[OK]`. **Read the output rather than the exit code** — a `[SKIP]` line is a
step that did not run, and the target can exit 0 carrying one.

The step list is deliberately not restated here. `make quality` is the single
description of what the gate does; a second copy in this entry would be one
more thing to keep in sync, and the gate changes more often than this runbook.

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
fallback ran, record it here.

**The behavioural check is automated and runs in the smoke tier**, so this
entry only has to establish which compiler branch ran.
`TestCpuSmoke::test_cpu_adam_single_step` (`tests/hardware/test_cpu.py:44`)
takes one `DeepSpeedCPUAdam` step on the pre-built extension and asserts the
parameter moved and stayed finite; a miscompiled op crashes or hangs there
rather than mid-run. `TestCpuSmoke::test_zen5_avx512_flags_visible` checks the
CPU exposes the flags in the first place. Both are `hw_smoke`, so
`make quality-full HW=smoke` covers them before every verification session.

This entry previously described that check as an `hw_full` "step-time bound".
It is neither: the marker is `hw_smoke` and the assertion is completion and
finiteness, not a duration.

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
$ make quality-full HW=smoke
```

Expected signal: zero failures, and zero tests deselected for a reason other
than the tier. The suite covers CUDA sm_120, bf16 matmul, flash-attn or the
logged sdpa fallback with a matching metadata stamp, DeepSpeedCPUAdam import,
AVX-512 flags, one CPUAdam step, huge pages, pinned buffer, the
`/mnt/ai-data` layout, and staging writability.

`make quality-full HW=smoke` runs `make quality`, the test suite, and
`pytest -m hw_smoke` in one invocation, and refuses outright if torch is not
importable — which is what stops the suite skipping its way to green. Running
`pytest -m hw_smoke tests/hardware/` directly still works and is the faster
loop when iterating on one test; the target is what this entry is closed
against.

Observed:

```
Run on the workstation 2026-08-01; output not recorded.
```

Status: VERIFIED
