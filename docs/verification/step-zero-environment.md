# Step zero: environment

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 8 verified, 0 blocked, 0 pending, of 8.**

Blocks all other phases. Any change to a dependency group re-opens 0.1 and
the entry for each image the group feeds (0.4 for training, 0.7 for interp):
nothing downstream is trustworthy until the lockfile is regenerated on the
workstation and the affected image rebuilds and passes its smoke test. The
first pass (2026-08-01) was against provisional pins resolved offline; the
latest re-open is the 2026-09-18 interp dependency cut, re-locked in 3451279.

## 0.1 Regenerate the lockfile

Implementation: no repository code; the resolver reads `pyproject.toml` and
rewrites `uv.lock`.

Command:

```bash
uv lock
git diff --stat uv.lock
```

Expected signal: `uv lock` exits 0 after resolving against the pytorch
cu130 index; the diff shows `uv.lock` updated. The `sandbox`, `dev`,
`training`, and `interp` groups all appear in the new lock (spot-check:
`grep -A2 'sandbox = ' uv.lock` lists the group's requirements). Commit the
regenerated lockfile before proceeding. If the `unsloth` pin fails to
resolve against the pinned transformers, record the resolver error here and
remove or repin `unsloth` in a separate commit — it is an optional backend
(F3 decision 2).

Observed (2026-09-18, on 0b6aafa, which carries the re-lock committed as
3451279; the first pass ran 2026-08-01 with its output not recorded):

```
$ git show --stat 3451279 | head -20
commit 3451279d2e0d89c5561502ee702da3cdb1859ea3
Author: Alejandro Jaramillo <alejandroJaramillo87@users.noreply.github.com>
Date:   Fri Sep 18 12:04:00 2026 -0700

    Re-lock after the interp dependency cut

 uv.lock | 145 ----------------------------------------------------------------
 1 file changed, 145 deletions(-)
$ uv lock
Resolved 332 packages in 0.59ms
$ git status --short uv.lock
```

The re-lock is the one-file commit the stat shows, and a second `uv lock`
on top of it resolves in under a millisecond and leaves `uv.lock` clean, so
the committed lock is the resolver's fixed point for the current groups.
The 145 removed lines are the nnsight, nnterp and circuitsvis closure the
interp dependency cut released; the empty status line is the whole of the
third command's output.

Status: VERIFIED 2026-09-18

## 0.2 Sync the host environment

Implementation: the `install` target in `Makefile`
(`uv sync --all-groups --all-extras`), `pyproject.toml`.

Command:

```bash
make install
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
make quality
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
make train-build
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
docker compose -f docker-compose.training.yaml build training --progress=plain 2>&1 | grep -m1 "znver5"
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
make train-smoke
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
make interp-build
make interp-smoke
```

Expected signal: build exits 0; the six-step smoke prints `[OK]` for
torch/sm_120, the import set, the hooked Hugging Face forward, and the bf16
forward, and exits 0. The gated Gemma step
requires `HF_TOKEN` in `.env.interp`; a `[WARN]` skip without the token is
acceptable at step zero but must pass before the interp phase.

Observed (2026-09-18, on 3451279, without `HF_TOKEN`; the first pass ran
2026-08-01 with its output not recorded):

```
$ make interp-build
 => resolving provenance for metadata file                                                                                                                                    0.0s
[+] build 1/1
 ✔ Image ai-experiments-interp Built                                                                                                                                         376.9s
Interp workbench image built.
$ make interp-smoke
docker compose -f docker-compose.interp.yaml run --rm interp python -m interp smoke
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-interp-run-baf603ae13ef Creating
Container ai-experiments-interp-run-baf603ae13ef Created
=== Interpretability workbench ===

Environment:
  Python: 3.12.3
  Torch: 2.13.0+cu130
  CUDA available: True
  HF_HOME: /mnt/ai-data/interp/hf-cache
  Working dir: /workspace
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m interp smoke
=== Interpretability workbench smoke test ===

[OK] torch 2.13.0+cu130: CUDA available, NVIDIA GeForce RTX 5090 sm_120
[OK] import sae_lens: 6.46.0
[OK] import transformers: 5.13.1
[OK] import datasets: 5.0.0
[OK] import einops: 0.8.2
[OK] import pynvml: 12.575.51
[OK] torch pin intact: 2.13.0+cu130
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|...| 148/148 [00:00<00:00, 9734.62it/s]
[OK] hf-hook: gpt2 forward, block 6 forward hook fired
[OK] bf16: gpt2 forward, finite logits
[WARN] gemma-scope: skipped, HF_TOKEN not set (required for Phase 0 acceptance)
  Solution: accept the Gemma license on Hugging Face, set HF_TOKEN in .env.interp

Versions manifest:
{
  "packages": {
    "torch": "2.13.0+cu130",
    "sae_lens": "6.46.0",
    "transformers": "5.13.1",
    "datasets": "5.0.0",
    "einops": "0.8.2",
    "pynvml": "12.575.51"
  },
  "cuda": "13.0",
  "gpu": "NVIDIA GeForce RTX 5090",
  "capability": [
    12,
    0
  ],
  "driver": "580.173.02"
}
[OK] manifest written: /mnt/ai-data/interp/versions.json

[OK] ALL SMOKE STEPS PASSED
```

The build's own output was pasted only from its final lines; the build-time
smoke's lines are I.13's business in `interp/interp.md`. The smoke run is
the same paste I.1 records. The weight progress bar is abbreviated.

Status: VERIFIED 2026-09-18 (gated Gemma step skipped without `HF_TOKEN`)

## 0.8 Hardware smoke suite

Implementation: `tests/hardware/test_gpu.py`, `tests/hardware/test_cpu.py`,
`tests/hardware/test_ram.py`, `tests/hardware/test_ssd.py`,
`tests/hardware/conftest.py` (tier gating).

Command:

```bash
make quality-full HW=smoke
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
