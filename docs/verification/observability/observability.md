# Observability verification (phases A-D)

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 3 verified, 0 blocked, 4 pending, of 7.**

Host verification for the observability stack: the logging core, traced
dispatch, audit trace, retention, run catalog (phases A-C), and the metrics
sink, watchdog wiring, snapshot dashboard, and live server (phase D). All of
it passed the sandbox campaign — unit suites, cross-module integration
through the real dispatch, and SIGKILL durability. The three entries needing
neither the GPU nor the shards closed on the workstation 2026-08-14; what
remains is the GPU half. The real-Chromium harness is no longer merely
sandbox context: it is OBS-V5 in
[sandbox/sandbox-executable.md](../sandbox/sandbox-executable.md), verified
2026-08-14. The entries below are the
host-only gaps: GPU sample values, NVML context capture, the `/mnt/ai-data`
default log root, external sink artifacts, and the live server against an
in-flight run.

Recording protocol and session-start gate are defined in
[README.md](../README.md). One GPU consumer at a time applies to the entries
that train; the report and serve viewers are not GPU consumers.

## Preconditions

One requirement is shared across entries rather than stated in each. It is
cheap to satisfy and expensive to discover halfway through a run.

**The `verify-small` shards.** OBS-V2, OBS-V6, OBS-V8, and the second half of
OBS-V3 all train against
`/mnt/ai-data/training/shards/verify-small`. That directory is not part of
the repository. It was confirmed present on the workstation 2026-08-14 —
`shard_000000.bin` at 40,002,422 bytes beside its `manifest.json`, which is
two bytes per token for the 20,001,211 tokens P0.1 recorded, so the artifact
is complete rather than partial. Should a reprovisioned volume lose it,
recreate it with the `make prepare-data` invocation in
[training/training-phase0.md](../training/training-phase0.md) (entry P0.1, the fineweb-edu
20M-token command). Its absence would be a precondition, not a blocker: one
command creates it, so these entries stay `PENDING` rather than `BLOCKED`.

Everything else the seven entries name resolves against the tree as of
2026-08-14: every make target, script, recipe, CLI subcommand, config key,
and pytest selector.

## What runs without the GPU

The entries are not uniform in what they cost, and a session with no GPU slot
can still close three of them.

- **No GPU, no shards:** OBS-V1, OBS-V4, OBS-V7. The first two are pytest
  runs whose assertion is that they execute rather than skip — torch must be
  importable, which is the host-only part. OBS-V7 drives a SIGKILL against a
  writer and needs only a writable `/mnt/ai-data`. All three are VERIFIED as
  of 2026-08-14.
- **Partly runnable ahead of a run:** OBS-V3. Its `training runs` listing and
  its `ls` of the log root work immediately and are recorded below; the
  `report --run` step consumes the run OBS-V2 produces, so the entry stays
  `PENDING` until then.
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
$ uv run --no-sync pytest tests/training/test_loop.py -k "metrics_jsonl or cli_run" -v
```

Expected signal: both tests pass (not skip). No `[tracking] WARN` lines in
the output.

Observed:

```
$ uv run --no-sync pytest tests/training/test_loop.py -k "metrics_jsonl or cli_run" -v
platform linux -- Python 3.12.11, pytest-9.1.1, pluggy-1.6.0
collected 7 items / 5 deselected / 2 selected

tests/training/test_loop.py::test_metrics_jsonl_written_with_checkpoint_markers PASSED [ 50%]
tests/training/test_loop.py::test_cli_run_joins_artifacts_under_real_watchdog_timing PASSED [100%]

[TIER] full: torch present, the whole suite was in scope.
2 passed, 5 deselected, 14 warnings in 12.02s
```

Both passed rather than skipped, and the `[TIER] full` line is the suite
confirming torch was importable — the host-only part of the entry. No
`[tracking] WARN` lines. The 14 warnings are a `torch.jit.script_method`
deprecation from torch 2.13.0+cu130, unrelated to tracking.

Status: VERIFIED 2026-08-14

## OBS-V2 GPU watchdog samples and NVML context in a real run

Implementation: `training/runtime/watchdog.py`, `common/obs/context.py`,
`training/runtime/tracking.py`.

Command (any tiny GPU-visible run works; the point is the sampler and
context capture, not the model):

```bash
$ uv run --no-sync python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    run.output_root=/mnt/ai-data/training/verify/obs-run \
    schedule.total_steps=20 watchdog.enabled=true watchdog.interval_s=1.0
$ tail -5 /mnt/ai-data/training/verify/obs-run/*/*/metrics.jsonl
$ grep run.context /mnt/ai-data/logs/training/R-*.jsonl | tail -1
```

Expected signal: exit 0; every sampler record in `metrics.jsonl` carries
`gpu_temp_c`, `gpu_power_w`, `gpu_vram_used_gb`, and `disk_free_gb`, all
present and non-null; `gpu_temp_c` between 20 and 90; `gpu_vram_used_gb`
between 0 and 32; `disk_free_gb` positive; the `run.context` event carries a
`gpu` object with a non-empty `name` and a `driver` matching `\d+\.\d+`.

Stated as thresholds rather than as "plausible values" because a reader
deciding whether an observation passes needs a number, and a field that is
present but null reads as plausible at a glance.

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

The first two commands run at any time. The third consumes the run OBS-V2
produces, so it comes after that entry rather than with these.

```bash
$ uv run --no-sync python -m training runs --limit 10
$ ls -la /mnt/ai-data/logs/training/ | tail -5
```

Then, once OBS-V2 has run, against the run directory it printed:

```bash
$ uv run --no-sync python -m training report --run <OBS-V2's run dir>
```

Expected signal: the runs table lists the OBS-V2 run with exit 0 and a
plausible duration; the log directory contains one `R-*.jsonl` per recent
invocation plus `runs.jsonl`; report writes `run.html` next to the chosen
run's `metrics.jsonl` and prints `[OK] Wrote ...`.

Observed (first half only; the `report --run` step waits on OBS-V2):

```
$ uv run --no-sync python -m training runs --limit 10
FINISHED             RUN ID                    COMMAND    EXIT    DURATION
2026-08-02T15:40:49  R-20260802-154048-8f90e9  preflight     1      0.015s
2026-08-02T15:58:54  R-20260802-155854-285848  preflight     1      0.015s
$ ls -la /mnt/ai-data/logs/training/ | tail -5
-rw-r--r-- 1 alejandro alejandro 2794 Aug  2 08:40 R-20260802-154048-8f90e9.jsonl
-rw-r--r-- 1 alejandro alejandro 2819 Aug  2 08:58 R-20260802-155854-285848.jsonl
-rw-rw-r-- 1 alejandro alejandro 1756 Aug 14 12:18 R-20260814-191804-a1c73c.jsonl
-rw-r--r-- 1 alejandro alejandro  482 Aug 14 12:18 runs.jsonl
```

The default root resolves and is written: the listing invocation created
its own `R-20260814-191804-a1c73c.jsonl` and appended to `runs.jsonl`,
while hiding itself from its own table, which is the `runs` filter
behaving as documented. One `R-*.jsonl` exists per catalog row.

Two aged files from 2026-08-02 survived, which is retention behaving
correctly rather than failing: `runs` is a viewer command and opts out of
the startup prune.

Status: PENDING

## OBS-V4 External sinks (tensorboard, offline wandb)

Implementation: `tests/training/test_loop.py`, exercising
`training/runtime/tracking.py`.

Command:

```bash
$ uv run --no-sync pytest tests/training/test_loop.py -k tensorboard -v
```

Expected signal: the test passes (not skips): a run with
`tracking.tensorboard=true tracking.wandb=true` leaves
`tensorboard/events.*` and `wandb/offline-run-*` under the run directory,
with the jsonl sink unaffected.

Observed:

```
$ uv run --no-sync pytest tests/training/test_loop.py -k tensorboard -v
collected 7 items / 6 deselected / 1 selected

tests/training/test_loop.py::test_tracking_writes_tensorboard_and_offline_wandb PASSED [100%]

[TIER] full: torch present, the whole suite was in scope.
1 passed, 6 deselected, 20 warnings in 5.38s
```

Passed rather than skipped. Both optional sinks are installed on the
workstation and wrote their artifacts; a missing wandb would have skipped.

Status: VERIFIED 2026-08-14

## OBS-V6 Live dashboard against an in-flight run

Implementation: `common/obs/serve.py`, `common/obs/dashboard.py`,
`training/commands/report.py`.

Start a longer tiny run (several hundred steps), then serve it while it
trains and watch the page extend.

Command:

The run directory does not exist until training has started, so it is
resolved rather than guessed.

```bash
$ uv run --no-sync python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    run.output_root=/mnt/ai-data/training/verify/obs-live \
    schedule.total_steps=500 watchdog.enabled=true &
$ RUN=$(ls -dt /mnt/ai-data/training/verify/obs-live/*/* | head -1); echo $RUN
$ uv run --no-sync python -m training report --run $RUN --serve
```

Expected signal: the printed URL renders the header and charts in a
browser; the loss chart and log tail extend without reloading; a zoomed
chart keeps its zoom as data arrives; after the run finishes the page
keeps polling without errors and Ctrl-C on the server exits 0.

This is the one entry whose evidence is not pasted output. Every signal
above is something a human watches happen, so the Observed block takes a
prose description and, for the chart behavior, a screenshot. That is a
deliberate exception to the recording protocol in
[README.md](../README.md), not an oversight: the entry exists precisely
because the property cannot be asserted from a terminal.

Observed:

```
```

Status: PENDING

## OBS-V7 Crash durability on the workstation

Implementation: `tests/training/obs_crash_driver.py`, `common/obs/core.py`.

Reuses the committed sandbox driver against the real filesystem
(`/mnt/ai-data` is ext4 on NVMe; the sandbox proved tmpfs only).

Command:

`%1` needs interactive job control and fails under `sh -c` or a script, so
the driver is addressed by pid.

**The interpreter is invoked directly, not through `uv run`.** This is the
one place in this runbook that departs from the repository convention, and
it is load-bearing rather than stylistic: `uv run` forks the interpreter as
a grandchild, so `$!` names `uv` and `kill -9 $PID` leaves the writer
running. The entry then measures a process that was never killed.

```bash
$ mkdir -p /mnt/ai-data/training/verify/obs-crash
$ OUT=/mnt/ai-data/training/verify/obs-crash/metrics.jsonl
$ .venv/bin/python tests/training/obs_crash_driver.py \
    /mnt/ai-data/training/verify/obs-crash & PID=$!
$ sleep 2; kill -9 $PID; wait $PID; echo "driver exit: $?"
$ kill -0 $PID 2>/dev/null && echo "STILL ALIVE - void" || echo "writer confirmed dead"
$ wc -l < $OUT; sleep 1; wc -l < $OUT
$ .venv/bin/python -c "import json; [json.loads(l) for l in open(
    '/mnt/ai-data/training/verify/obs-crash/metrics.jsonl',
    'rb').read().split(b'\n')[:-1]]; print('all completed lines parse')"
```

Expected signal: `all completed lines parse` — every completed line before
the torn tail is valid JSON after SIGKILL on the real mount. The two
`wc -l` values must be equal; if the second is larger the writer outlived
the kill and the result is void, not a pass. The `[:-1]` slice drops the
final element deliberately: a torn last line is permitted by the driver's
own contract, and the property under test is that every *completed* line
survives.

**Remove the run directory afterwards.** The driver logs in an unbounded
loop with no throttle, and on NVMe it writes roughly 10 GB per minute. An
orphaned writer is therefore a disk-exhaustion hazard, not merely untidy —
see the note in the observation below.

```bash
$ pkill -f obs_crash_driver; rm -rf /mnt/ai-data/training/verify/obs-crash
```

Observed:

```
$ .venv/bin/python tests/training/obs_crash_driver.py /mnt/ai-data/training/verify/obs-crash & PID=$!
[1] 197276
READY
$ sleep 2; kill -9 $PID; wait $PID; echo "driver exit: $?"
[1]  + 197276 killed     .venv/bin/python tests/training/obs_crash_driver.py
driver exit: 137
$ kill -0 $PID 2>/dev/null && echo "STILL ALIVE - void" || echo "writer confirmed dead"
writer confirmed dead
$ wc -l < $OUT; sleep 1; wc -l < $OUT
358247
358247
$ .venv/bin/python -c "..."
all completed lines parse
```

358,247 completed lines survived SIGKILL on ext4/NVMe, and the count was
unchanged across the sleep, so the writer was genuinely dead when the file
was read.

This entry was first run through `uv run --no-sync` and produced a void
result: the counts came back 361,516 then 549,816, because `kill -9` had
killed `uv` while the Python grandchild kept appending. A second attempt
compounded it — two orphaned writers took the file to 17.7M lines and the
directory to **36 GB** before they were found. Both the direct-interpreter
invocation and the disk warning above are corrections earned from that
failure. The equal-counts check is what caught it, which is the argument
for keeping it in the procedure rather than trusting the exit code.

Status: VERIFIED 2026-08-14

## OBS-V8 Sink overhead at real step rates

Implementation: `training/runtime/tracking.py`, `training/runtime/loop.py`,
`training/recipes/schema.py`.

The jsonl sink is line-buffered (one write per step plus one per sampler
tick). The sandbox cannot measure its cost at real GPU step rates.

Command (compare two OBS-V2-style runs, `tracking.metrics_jsonl=true`
versus `false`, same recipe and steps):

```bash
$ uv run --no-sync python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    run.output_root=/mnt/ai-data/training/verify/obs-overhead-on \
    schedule.total_steps=200
$ uv run --no-sync python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    run.output_root=/mnt/ai-data/training/verify/obs-overhead-off \
    schedule.total_steps=200 tracking.metrics_jsonl=false
```

Run each arm three times, alternating, and compare medians.

Expected signal: the median step time of the sink-on arm falls inside the
range spanned by the three sink-off runs, and vice versa — that is, the
on/off difference is no larger than the run-to-run variance of either arm.

The original criterion here was "mean `step_time_s` differs by under 1%",
which cannot be evaluated as written. With `tracking.metrics_jsonl=false`
there is no `metrics.jsonl`, so the two arms would be measured by different
instruments — a per-step field against a wall-clock division — and that
methodology gap is plausibly wider than 1%. Two hundred steps of a tiny model
also vary by more than 1% from clock and thermal noise alone, so a run that
came in under the threshold would not have shown the sink is cheap. Comparing
against the noise floor asks the question the entry exists to ask.

Observed:

```
```

Status: PENDING

## Moved to another tier

**OBS-V5, the browser harness**, is now
[sandbox/sandbox-executable.md](../sandbox/sandbox-executable.md). What it verifies —
`common/obs/viewer.js` rendering in a real browser — depends on a browser and not on this
workstation, so running it on the RTX 5090 proved nothing that running it in a container
does not. The identifier is not reused, and the numbering below keeps its gap.

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
