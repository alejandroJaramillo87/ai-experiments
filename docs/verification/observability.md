# Observability verification (phases A-D)

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 0 verified, 0 blocked, 8 pending, of 8.**

Host verification for the observability stack: the logging core, traced
dispatch, audit trace, retention, run catalog (phases A-C), and the metrics
sink, watchdog wiring, snapshot dashboard, and live server (phase D). All of
it passed the sandbox campaign — unit suites, cross-module integration
through the real dispatch, SIGKILL durability, and a real-Chromium harness —
but none of it has run on the workstation. The entries below are the
host-only gaps: GPU sample values, NVML context capture, the `/mnt/ai-data`
default log root, external sink artifacts, and the live server against an
in-flight run.

Recording protocol and session-start gate are defined in
[README.md](README.md). One GPU consumer at a time applies to the entries
that train; the report and serve viewers are not GPU consumers.

## Preconditions

Two requirements are shared across entries rather than stated in each, and
both are cheap to satisfy and expensive to discover halfway through a run.

**The `verify-small` shards.** OBS-V2, OBS-V6, OBS-V8, and the second half of
OBS-V3 all train against
`/mnt/ai-data/training/shards/verify-small`. That directory is not part of
the repository and may not exist on a freshly provisioned volume. Produce it
with the `make prepare-data` invocation in
[training-phase0.md](training-phase0.md) (entry P0.1, the fineweb-edu
20M-token command) before starting any of the four. Its absence is a
precondition, not a blocker: one command creates it, so these entries stay
`PENDING` rather than `BLOCKED`.

**A real Chromium for OBS-V5.** That entry expands
`PW_CHROMIUM=$(command -v chromium)`. If no such binary is on `PATH` the
expansion is empty and the harness receives no browser path — a silent
misconfiguration rather than an error. Confirm `command -v chromium` prints a
path before running it.

Everything else the eight entries name resolves against the tree as of
2026-08-14: every make target, script, recipe, CLI subcommand, config key,
and pytest selector.

## What runs without the GPU

The entries are not uniform in what they cost, and a session with no GPU slot
can still close three of them.

- **No GPU, no shards:** OBS-V1, OBS-V4, OBS-V7. The first two are pytest
  runs whose assertion is that they execute rather than skip — torch must be
  importable, which is the host-only part. OBS-V7 drives a SIGKILL against a
  writer and needs only a writable `/mnt/ai-data`.
- **Partly runnable ahead of a run:** OBS-V3. Its `training runs` listing and
  its `ls` of the log root work immediately; the `report --run` step consumes
  the run OBS-V2 produces.
- **GPU and shards:** OBS-V2, OBS-V6, OBS-V8. These train, so the one-GPU-
  consumer rule applies. OBS-V6 additionally needs a human at a browser.

## Verified in the sandbox (context, not entries)

- Unit suites for core, dispatch, audit, retention, catalog, sink, watchdog
  branches, dashboard, serve, and viewer edge behavior, including the
  regression tests for all 13 bug-hunt fix commits.
- Cross-module integration through the real `dispatch()`
  (`tests/training/test_obs_integration.py`): header identity, events join,
  catalog join, report render, live server streaming.
- SIGKILL durability (`tests/training/test_obs_durability.py` with
  `tests/training/obs_crash_driver.py`): every completed line survives a
  kill mid-write.
- Real-Chromium checks (`scripts/verify/obs_browser/`): chart creation,
  markers, log filters, the hostile `<!--<script` payload, live extension.

## OBS-V1 Torch-gated artifact-chain tests

Implementation: `tests/training/test_loop.py`, exercising
`training/runtime/tracking.py`, `common/obs/core.py`,
`common/obs/catalog.py`.

The two host-gated tests added to `tests/training/test_loop.py` skip in the
sandbox (no torch). They are the mechanical half of this campaign: a tiny
CPU train run asserting the metrics artifact, and a subprocess CLI run with
a 50 ms watchdog interval asserting header identity and the events and
catalog joins under real thread timing.

Command:

```bash
$ uv run pytest tests/training/test_loop.py -k "metrics_jsonl or cli_run" -v
```

Expected signal: both tests pass (not skip). No `[tracking] WARN` lines in
the output.

Observed:

```
```

Status: PENDING

## OBS-V2 GPU watchdog samples and NVML context in a real run

Implementation: `training/runtime/watchdog.py`, `common/obs/context.py`,
`training/runtime/tracking.py`.

Command (any tiny GPU-visible run works; the point is the sampler and
context capture, not the model):

```bash
$ uv run python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    run.output_root=/mnt/ai-data/training/verify/obs-run \
    schedule.total_steps=20 watchdog.enabled=true watchdog.interval_s=1.0
$ tail -5 /mnt/ai-data/training/verify/obs-run/*/*/metrics.jsonl
$ grep run.context /mnt/ai-data/logs/training/R-*.jsonl | tail -1
```

Expected signal: exit 0; `metrics.jsonl` contains sampler records carrying
`gpu_temp_c`, `gpu_power_w`, `gpu_vram_used_gb`, and `disk_free_gb` with
plausible values for the RTX 5090 (temperature below the 90 C limit, VRAM
below 32 GB); the `run.context` event includes a `gpu` entry naming the
card and driver version.

Observed:

```
```

Status: PENDING

## OBS-V3 Default log root, retention, and catalog on /mnt/ai-data

Implementation: `training/commands/runs.py`, `training/commands/report.py`,
`common/obs/catalog.py`, `common/obs/retention.py`.

The sandbox exercised these against `OBS_LOG_DIR` temp dirs only. This
entry confirms the default root works end to end: configure writes to
`/mnt/ai-data/logs/<cli>/`, the startup prune respects the live-file lock,
and the catalog joins run ids to run directories.

Command:

```bash
$ uv run python -m training runs --limit 10
$ ls -la /mnt/ai-data/logs/training/ | tail -5
$ uv run python -m training report --run <run_dir printed by runs>
```

Expected signal: the runs table lists the OBS-V2 run with exit 0 and a
plausible duration; the log directory contains one `R-*.jsonl` per recent
invocation plus `runs.jsonl`; report writes `run.html` next to the chosen
run's `metrics.jsonl` and prints `[OK] Wrote ...`.

Observed:

```
```

Status: PENDING

## OBS-V4 External sinks (tensorboard, offline wandb)

Implementation: `tests/training/test_loop.py`, exercising
`training/runtime/tracking.py`.

Command:

```bash
$ uv run pytest tests/training/test_loop.py -k tensorboard -v
```

Expected signal: the test passes (not skips): a run with
`tracking.tensorboard=true tracking.wandb=true` leaves
`tensorboard/events.*` and `wandb/offline-run-*` under the run directory,
with the jsonl sink unaffected.

Observed:

```
```

Status: PENDING

## OBS-V5 Browser harness on the workstation

Implementation: `scripts/verify/obs_browser/check.mjs`, rendering the output
of `common/obs/dashboard.py` and `common/obs/serve.py`.

Command:

```bash
$ cd scripts/verify/obs_browser && npm install && \
    PW_CHROMIUM=$(command -v chromium) node check.mjs
```

Expected signal: exit 0 and `ALL CHECKS PASSED`; all snapshot and live
checks print `[OK]`.

Observed:

```
```

Status: PENDING

## OBS-V6 Live dashboard against an in-flight run

Implementation: `common/obs/serve.py`, `common/obs/dashboard.py`,
`training/commands/report.py`.

Start a longer tiny run (several hundred steps), then serve it while it
trains and watch the page extend.

Command:

```bash
$ uv run python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    run.output_root=/mnt/ai-data/training/verify/obs-live \
    schedule.total_steps=500 watchdog.enabled=true &
$ uv run python -m training report --run <the new run dir> --serve
```

Expected signal: the printed URL renders the header and charts in a
browser; the loss chart and log tail extend without reloading; a zoomed
chart keeps its zoom as data arrives; after the run finishes the page
keeps polling without errors and Ctrl-C on the server exits 0.

Observed:

```
```

Status: PENDING

## OBS-V7 Crash durability on the workstation

Implementation: `tests/training/obs_crash_driver.py`, `common/obs/core.py`.

Reuses the committed sandbox driver against the real filesystem
(`/mnt/ai-data` is ext4 on NVMe; the sandbox proved tmpfs only).

Command:

```bash
$ mkdir -p /mnt/ai-data/training/verify/obs-crash
$ uv run python tests/training/obs_crash_driver.py \
    /mnt/ai-data/training/verify/obs-crash & sleep 2; kill -9 %1
$ python3 -c "import json,sys; [json.loads(l) for l in open(
    '/mnt/ai-data/training/verify/obs-crash/metrics.jsonl',
    'rb').read().split(b'\n')[:-1]]; print('all lines parse')"
```

Expected signal: `all lines parse` — every completed line before the torn
tail is valid JSON after SIGKILL on the real mount.

Observed:

```
```

Status: PENDING

## OBS-V8 Sink overhead at real step rates

Implementation: `training/runtime/tracking.py`, `training/runtime/loop.py`,
`training/recipes/schema.py`.

The jsonl sink is line-buffered (one write per step plus one per sampler
tick). The sandbox cannot measure its cost at real GPU step rates.

Command (compare two OBS-V2-style runs, `tracking.metrics_jsonl=true`
versus `false`, same recipe and steps):

```bash
$ uv run python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    run.output_root=/mnt/ai-data/training/verify/obs-overhead-on \
    schedule.total_steps=200
$ uv run python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    run.output_root=/mnt/ai-data/training/verify/obs-overhead-off \
    schedule.total_steps=200 tracking.metrics_jsonl=false
```

Expected signal: mean `step_time_s` differs by under 1% between the runs
(compare the printed per-step timing or the on-run's own metrics against
wall clock for the off-run).

Observed:

```
```

Status: PENDING

## Deferred by design (not entries)

- The watchdog `_alert` path writes its alert record while holding the
  signal lock (`training/runtime/watchdog.py`). Accepted for now: alerts
  are rare and the write is one line; revisit only if OBS-V2 shows sampler
  stalls.
- OBS-14 (the operator reference) is written:
  `docs/observability/README.md`. OBS-13 (the profiler debugging guide,
  `docs/observability/debugging-runs.md`) remains unwritten;
  `docs/observability/design-and-roadmap.md` tracks it and carries a status
  field per item.
- OBS-15 (the optional rig console) consumes this campaign's results; it
  cannot close before these entries do. The go/no-go it represents is
  whether the ephemeral viewer of OBS-12 proves insufficient in use.
