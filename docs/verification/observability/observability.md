# Observability verification (phases A-D)

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 2 verified, 0 blocked, 7 pending, of 9.**

Host verification for the observability stack: the logging core, traced
dispatch, audit trace, retention, run catalog (phases A-C), and the metrics
sink, watchdog wiring, snapshot dashboard, and live server (phase D). All of
it passed the sandbox campaign — unit suites, cross-module integration
through the real dispatch, and SIGKILL durability. All seven were run on the
workstation 2026-08-14. **Five reopened on 2026-08-15**, when each Observed
block was read back against the clause it was meant to demonstrate: OBS-V2,
OBS-V3, OBS-V4, OBS-V6, and OBS-V8 each show less than their expected signal
claims. Every reopened entry keeps its 2026-08-14 output in place — the
output is evidence of something, and deleting it would lose that — and states
beside its status the one clause it does not cover. The real-Chromium harness
is no longer merely
sandbox context: it is OBS-V5 in
[sandbox/sandbox-executable.md](../sandbox/sandbox-executable.md), verified
the same day. **Two entries were added 2026-08-15**, from a surface inventory that found
shipped functionality with no entry at all: OBS-V10 (retention against a real
Rust writer) and OBS-V11 (the watchdog `alerts.jsonl` and the mlflow sink).
The rest of what that inventory found is pinned by named gate checks and is
listed in the roster below rather than as entries.
The first seven entries were the host-only gaps: GPU sample values,
NVML context capture, the `/mnt/ai-data` default log root, external sink
artifacts, and the live server against an in-flight run.

The campaign found four defects in its own runbook and one in the code. The
runbook defects mattered most: the three training entries named a tokenizer
fixture that exists only inside the unit tests, so **none of them could ever
have run as written**. OBS-V7 addressed the wrong process and could not have
failed. Each is corrected in place, with the reasoning kept beside the entry
rather than summarized away. The code defect is KB-69.

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
  writer and needs only a writable `/mnt/ai-data`. OBS-V1 and OBS-V7 are
  VERIFIED as of 2026-08-14; OBS-V4 is PENDING, because passing is not by
  itself the artifact claim its expected signal makes.
- **Partly runnable ahead of a run:** OBS-V3. Its `training runs` listing and
  its `ls` of the log root work immediately; the `report --run` step consumes
  the run OBS-V2 produces, so it comes after OBS-V2 rather than with it.
- **GPU and shards:** OBS-V2, OBS-V6, OBS-V8, OBS-V11. These train, so the
  one-GPU-consumer rule applies. OBS-V6 additionally needs a human at a
  browser.
- **No GPU, real mount:** OBS-V10. It needs a built rust workspace and a
  writable `/mnt/ai-data/logs/harness`, and no GPU slot.

**All three training entries need OBS-V2's two recipe overrides**
(`tokenizer.path` and `model.scratch.vocab`), for the reason set out there.
Without them the recipe fails validation against the `verify-small`
manifest, so as originally written none of the three could ever have run.

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
uv run --no-sync pytest tests/training/test_loop.py -k "metrics_jsonl or cli_run" -v
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

The invocation is below, under the overrides it needs. Read them first: the
recipe cannot be pointed at `verify-small` without both.

Expected signal: exit 0; every sampler record in `metrics.jsonl` carries
`gpu_temp_c`, `gpu_power_w`, `gpu_vram_used_gb`, and `disk_free_gb`, all
present and non-null; `gpu_temp_c` between 20 and 90; `gpu_vram_used_gb`
between 0 and 32; `disk_free_gb` positive; the `run.context` event carries a
`gpu` object with a non-empty `name` and a `driver` matching `\d+\.\d+`.

Stated as thresholds rather than as "plausible values" because a reader
deciding whether an observation passes needs a number, and a field that is
present but null reads as plausible at a glance.

**The recipe needs two overrides the entry did not originally carry.**
`tiny-e2e.yaml` pairs `data.source: shards/tiny-e2e` with
`tokenizer.path: tokenizers/tiny-e2e`, a matched pair the unit tests
generate in temporary directories rather than artifacts on the workstation.
Overriding `data.source` to `verify-small` alone fails validation, because
`_check_tokenizer` cross-checks the recipe's tokenizer hash against the
shard manifest's. The scratch model's `vocab: 512` must also be raised to
the tokenizer's, and nothing validates that one — it would have surfaced as
an out-of-range embedding index once training started. Both are derived
from the manifest rather than hardcoded:

```bash
TOK=$(.venv/bin/python -c "import json;print(json.load(open('/mnt/ai-data/training/shards/verify-small/manifest.json'))['tokenizer'])")
VOCAB=$(.venv/bin/python -c "from training.data.prepare import load_tokenizer;print(load_tokenizer('$TOK').vocab_size)")
uv run --no-sync python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    tokenizer.path="$TOK" model.scratch.vocab=$VOCAB \
    run.output_root=/mnt/ai-data/training/verify/obs-run \
    schedule.total_steps=20 watchdog.enabled=true watchdog.interval_s=1.0
echo "exit=$?"
RUN=$(dirname "$(ls -t /mnt/ai-data/training/verify/obs-run/*/*/metrics.jsonl | head -1)")
grep -h run.context /mnt/ai-data/logs/training/R-*.jsonl | tail -1
jq -c 'select(.gpu_temp_c != null)' "$RUN/metrics.jsonl"
jq -c 'select(.gpu_temp_c != null)' "$RUN/metrics.jsonl" | wc -l
```

The run directory is resolved by locating `metrics.jsonl` rather than by
globbing two levels down: `<root>/<recipe>/checkpoints` also matches
`*/*` and sorts newer than the timestamped directory, so the positional
glob selects the wrong one.

The last three commands are what the 2026-08-14 block was missing. The
expected signal is a claim about **every** sampler record in `metrics.jsonl`,
so the record set has to come out of that file under a command a reader can
see, and the count has to be printed rather than described. Piping through
`jq -c` also keeps each record on one line, which is what the file holds:
the earlier block's pretty-printed `run.context` could not have come from the
`grep` above it.

Observed (`TOK=HuggingFaceTB/SmolLM2-135M`, `VOCAB=49152`):

```
[OK] Run complete: step 20, 5,120 tokens seen
Final loss: 10.371604
Run directory: /mnt/ai-data/training/verify/obs-run/tiny-e2e/2026-08-14_13-00-49

$ grep -h run.context /mnt/ai-data/logs/training/R-20260814-200046-e1911c.jsonl
{
    "event": "run.context",
    "run_id": "R-20260814-200046-e1911c",
    "cli": "training", "command": "train",
    "git_sha": "ecfd232d253cba96ac0871b55818568a91c2e663",
    "python": "3.12.11",
    "platform": "Linux-7.0.0-28-generic-x86_64-with-glibc2.39",
    "packages": {"torch": "2.13.0+cu130", "transformers": "5.13.1", ...},
    "gpu": {"name": "NVIDIA GeForce RTX 5090", "driver": "580.173.02"}
}

sampler records (steps 5 and 17):
{"gpu_temp_c": 32.0, "gpu_power_w": 38.813, "gpu_vram_used_gb": 1.41717529296875,
 "gpu_sm_clock_mhz": 2595.0, "gpu_throttle_active": 0.0,
 "ram_available_gb": 112.87762832641602, "disk_free_gb": 1622.0664710998535}
{"gpu_temp_c": 32.0, "gpu_power_w": 68.241, "gpu_vram_used_gb": 1.41717529296875,
 "gpu_sm_clock_mhz": 2542.0, "gpu_throttle_active": 0.0,
 "ram_available_gb": 112.80743026733398, "disk_free_gb": 1622.0664672851562}
```

Every threshold holds: `gpu_temp_c` 32.0 in (20, 90); `gpu_vram_used_gb`
1.417 in (0, 32); `disk_free_gb` positive; all four sampler fields present
and non-null on both records. The `run.context` `gpu` object carries a
non-empty name and a driver matching `\d+\.\d+`. NVML is populating both
paths on real hardware, which is the property this entry exists for.

Two observations the criterion did not ask for. The sampler emits three
fields beyond the four named here — `gpu_sm_clock_mhz`,
`gpu_throttle_active` and `ram_available_gb`. And a 1.5-second run against
a 1.0-second interval yields only two ticks, so this entry confirms the
sampler works rather than that it samples densely; a longer run would be
needed for the latter.

Status: PENDING (run 2026-08-14; reopened 2026-08-15 — "exit 0; every
sampler record in `metrics.jsonl`" is unshown: no `echo "exit=$?"` output was
pasted, the two sampler records were pasted under a prose heading rather than
under a command reading `metrics.jsonl`, and no record count was printed, so
"every" rests on an unbounded sample. The corrected block above prints both.)

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
uv run --no-sync python -m training runs --limit 10
ls -la /mnt/ai-data/logs/training/ | tail -5
```

Then, once OBS-V2 has run, against the run directory it printed:

```bash
uv run --no-sync python -m training report --run <OBS-V2's run dir>
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

Second half, against the run OBS-V2 produced:

```
$ uv run --no-sync python -m training report --run "$RUN"
[OK] Wrote /mnt/ai-data/training/verify/obs-run/tiny-e2e/2026-08-14_13-00-49/run.html
$ ls -la "$RUN"/run.html
-rw-rw-r-- 1 alejandro alejandro 77834 Aug 14 13:06 .../run.html
$ uv run --no-sync python -m training runs --limit 10
FINISHED             RUN ID                    COMMAND    EXIT    DURATION
2026-08-02T15:40:49  R-20260802-154048-8f90e9  preflight     1      0.015s
2026-08-02T15:58:54  R-20260802-155854-285848  preflight     1      0.015s
2026-08-14T12:46:39  R-20260814-194637-83da90  train       130      1.656s
2026-08-14T12:46:47  R-20260814-194645-a0bcce  train         1      2.367s
2026-08-14T12:52:56  R-20260814-195251-8f74f5  train         0      5.287s
2026-08-14T13:00:51  R-20260814-200046-e1911c  train         0      5.117s
2026-08-14T13:06:28  R-20260814-200628-e3cc05  report        0      0.001s
```

The catalog joins run ids to the run directory end to end on the default
root: the OBS-V2 run appears with exit 0 and a 5.117 s duration, and
`report --run` wrote a 77 KB `run.html` beside its `metrics.jsonl`.

The table also records this session's failures faithfully — a SIGINT'd
train at exit 130 and a validation failure at exit 1 — which is worth more
than a clean table would have been. A catalog that only shows successes
cannot be relied on to show a failure.

Status: PENDING (run 2026-08-14; reopened 2026-08-15 — "report writes
`run.html` next to the chosen run's `metrics.jsonl`" is unshown: the `ls -la
"$RUN"/run.html` line was hand-elided to `.../run.html`, a form `ls` never
prints, so the pasted output does not establish the directory the file
landed in. Re-run with the path left intact, and list `metrics.jsonl`
alongside it.)

## OBS-V4 External sinks (tensorboard, offline wandb)

Implementation: `tests/training/test_loop.py`, exercising
`training/runtime/tracking.py`.

Command:

```bash
uv run --no-sync pytest tests/training/test_loop.py -k tensorboard -v
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

Status: PENDING (run 2026-08-14; reopened 2026-08-15 — "leaves
`tensorboard/events.*` and `wandb/offline-run-*` under the run directory,
with the jsonl sink unaffected" is unshown: the paste is one green test name,
which demonstrates that the test ran rather than skipped and nothing about
the two globs or the third property. Re-run with an `ls -R` of the run
directory the test writes, or record which assertions in
`test_tracking_writes_tensorboard_and_offline_wandb` cover each clause.)

## OBS-V6 Live dashboard against an in-flight run

Implementation: `common/obs/serve.py`, `common/obs/dashboard.py`,
`training/commands/report.py`.

Start a longer tiny run (several hundred steps), then serve it while it
trains and watch the page extend.

Command:

The run directory does not exist until training has started, so it is
resolved rather than guessed.

`total_steps` is deliberately far above the 500 an acceptance run would use.
At roughly 3.5 ms per step this model finishes 500 steps in under two
seconds — long before a human reaches the browser — and a page that only
ever shows a completed run cannot demonstrate live extension.

```bash
uv run --no-sync python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    tokenizer.path=HuggingFaceTB/SmolLM2-135M model.scratch.vocab=49152 \
    run.output_root=/mnt/ai-data/training/verify/obs-live \
    schedule.total_steps=20000 watchdog.enabled=true &
sleep 20
RUN=$(dirname "$(ls -t /mnt/ai-data/training/verify/obs-live/*/*/metrics.jsonl | head -1)")
echo "$RUN"
uv run --no-sync python -m training report --run "$RUN" --serve
```

The server prints `[INFO] Live dashboard: http://127.0.0.1:8010/`;
`--host` and `--port` override it.

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

Run `R-20260815-005434-9ab66c`, `scratch-2L-64d`, 20,000 steps, served at
`http://127.0.0.1:8010/` and watched in Firefox. Screenshots taken at
1,600 and 2,000+ steps.

- **Header renders.** `run`, `tier pretrain`, `model scratch-2L-64d`,
  `recipe 51ff800cf204`, and the full schedule object.
- **Four charts render**: Loss, Learning rate, Throughput (tokens/s), and
  Device memory (GB) carrying both `gpu_mem_allocated_gb` and
  `gpu_vram_used_gb`.
- **The page extends without a reload.** Two screenshots thirty seconds
  apart show the x axis grown from ~1,600 to beyond 2,000 steps, on every
  chart, with the log tail extending alongside.
- **A zoom survives incoming data.** Dragging a range on the Loss chart
  pins it; the window stays where it was put while new points continue to
  arrive outside it. Double-clicking restores full range, and the restored
  axis extends past where it was when the zoom began — which is the
  positive evidence that data streamed throughout rather than stalling.
- **The log tail renders and the filters work.** `cmd.start` (carrying the
  full argv including both overrides), `run.context` with the RTX 5090
  block, and `run.dir`.

The last signal, the run finishing while the page stays open without
errors, was observed as part of the same session.

Status: PENDING (run 2026-08-14; reopened 2026-08-15 — "Ctrl-C on the server
exits 0" is never addressed anywhere in the block, and the two screenshots
the zoom and extension evidence rests on are cited with no path and no link,
so no later auditor can retrieve them. Re-run recording the server's exit
status after SIGINT, and commit the screenshots under a stated path or state
that they were not retained.)

### Findings that are not this entry's criterion

Three things surfaced that the expected signal does not cover. None
blocks the entry; all three are real.

**Checkpoint markers have no density limit** — filed as KB-69. At
`interval_steps: 5` over 20,000 steps the Loss chart draws roughly 4,000
vertical marker lines and renders as a solid block, hiding the series it
annotates.

**No follow-the-tail mode.** Zoom is all-or-nothing: pinned at a fixed
window, or auto-scaled to everything. Watching a live run at a fixed
recent width is the natural thing to want from a live dashboard and there
is no way to ask for it. This is a feature gap rather than a defect — the
current behavior is the documented one, and auto-following would yank the
view away from a reader studying a specific range — so it is recorded here
rather than in the bug ledger.

**The `debug` filter option is decorative.** Selecting `debug` and `info`
produced identical output, which was checked and is correct: every record
this run emitted was `info`. Nothing in the observability system writes at
`debug` level today, so the option filters a level that does not exist.

**Why a human found all three.** `scripts/verify/obs_browser/check.mjs`
makes ten assertions — seven on the snapshot page, three on the live one —
and none of them touches zoom, marker density,
or level-filter behavior beyond the `warning` case. The one property
OBS-V6 names explicitly — that a zoom survives incoming data — has no
automated coverage at all. Extending that harness is worth more than the
individual fixes.

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
mkdir -p /mnt/ai-data/training/verify/obs-crash
OUT=/mnt/ai-data/training/verify/obs-crash/metrics.jsonl
.venv/bin/python tests/training/obs_crash_driver.py \
    /mnt/ai-data/training/verify/obs-crash & PID=$!
sleep 2; kill -9 $PID; wait $PID; echo "driver exit: $?"
kill -0 $PID 2>/dev/null && echo "STILL ALIVE - void" || echo "writer confirmed dead"
wc -l < $OUT; sleep 1; wc -l < $OUT
.venv/bin/python -c "import json; [json.loads(l) for l in open(
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
pkill -f obs_crash_driver; rm -rf /mnt/ai-data/training/verify/obs-crash
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
uv run --no-sync python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    tokenizer.path=HuggingFaceTB/SmolLM2-135M model.scratch.vocab=49152 \
    checkpoint.interval_steps=200 \
    run.output_root=/mnt/ai-data/training/verify/obs-overhead-on \
    schedule.total_steps=200
uv run --no-sync python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    tokenizer.path=HuggingFaceTB/SmolLM2-135M model.scratch.vocab=49152 \
    checkpoint.interval_steps=200 \
    run.output_root=/mnt/ai-data/training/verify/obs-overhead-off \
    schedule.total_steps=200 tracking.metrics_jsonl=false
```

Run each arm seven times, alternating. Both arms carry OBS-V2's two
overrides, and `checkpoint.interval_steps=200` so that a single save
replaces the forty the recipe's `interval_steps: 5` would otherwise make;
those saves cost about 3.4 s of a 9.9 s run and swamp what is being
measured. Timings go to a file, because `/usr/bin/time` writes to stderr
and the run's own output has to be discarded:

```bash
rm -f /tmp/obs-overhead.txt
for i in 1 2 3 4 5 6 7; do
    for arm in on off; do
      if [ "$arm" = off ]; then SINK=false; else SINK=true; fi
      /usr/bin/time -f "$arm run$i %e s" -a -o /tmp/obs-overhead.txt \
        uv run --no-sync python -m training train \
          --config training/recipes/reference/tiny-e2e.yaml \
          data.source=/mnt/ai-data/training/shards/verify-small \
          tokenizer.path=HuggingFaceTB/SmolLM2-135M model.scratch.vocab=49152 \
          checkpoint.interval_steps=200 \
          run.output_root=/mnt/ai-data/training/verify/obs-overhead-$arm \
          schedule.total_steps=200 tracking.metrics_jsonl=$SINK >/dev/null \
        || echo "FAILED $arm run$i" >> /tmp/obs-overhead.txt
    done
  done
cat /tmp/obs-overhead.txt
```

The loop is what makes the timings comparable: the arms alternate within each
iteration, so a thermal drift or a background process biases both arms
equally rather than whichever arm ran last. `/usr/bin/time -a` appends, so the
file holds **one line per run — fourteen lines, in execution order**,
alternating `on run1`, `off run1`, `on run2`, and so on. Any other shape in
the paste means the file was edited before it was pasted.

The `|| echo FAILED` is not decoration. An earlier attempt produced
fourteen plausible timings from fourteen runs that had all exited 1 on a
stale training lock; only `/usr/bin/time`'s own status line gave it away. A
timing harness that does not record exit status invites a measurement of
nothing.

Expected signal: the difference between the two arms' medians is smaller
than the larger of the two arms' min-max spreads — the sink's cost is below
the noise floor.

Stated this way rather than as "each median falls inside the other arm's
span": with a handful of samples, an arm that happens to land tightly makes
strict containment nearly unsatisfiable in both directions, and a 10 ms
excursion on a 10-second run would fail an entry it should pass.

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
$ cat /tmp/obs-overhead.txt
on run1 7.46 s      off run1 6.43 s
on run2 6.47 s      off run2 6.49 s
on run3 6.46 s      off run3 6.48 s
on run4 6.43 s      off run4 6.39 s
on run5 6.43 s      off run5 6.43 s
on run6 6.47 s      off run6 6.45 s
on run7 6.43 s      off run7 6.44 s
```

No `FAILED` lines: all fourteen exited 0.

| arm | median | min-max spread |
|-----|-------:|---------------:|
| on  | 6.46 s | 1.03 s |
| off | 6.44 s | 0.10 s |

The medians differ by 0.02 s against a larger spread of 1.03 s, so the
criterion holds. It also holds after discarding `on run1` as the cold-start
outlier it plainly is: the spreads become 0.04 s and 0.10 s, and 0.02 s is
under both.

**What this does not show.** The entry is titled "at real step rates" and
this configuration does not reach them. Two hundred steps of a two-layer,
64-hidden model is about 0.7 s of training inside a 6.45 s process; the rest
is interpreter startup, CUDA initialization and one checkpoint save. The
honest conclusion is that the sink's cost is below the noise floor of a
process dominated by fixed costs — not that it is cheap during sustained
training on a real model. Re-run this against a Phase 1 recipe when one
exists; until then, do not cite this entry as evidence about production
throughput.

Status: PENDING (run 2026-08-14; reopened 2026-08-15 — the pasted `cat
/tmp/obs-overhead.txt` cannot be that file's contents: the harness appends
one line per run, fourteen lines in execution order, and the paste is seven
two-column rows grouped by arm, so it was reformatted by hand and the
execution order and any `FAILED` line placement are no longer recoverable.
The medians and spreads above are derived from it and inherit the doubt.
Re-run the loop above and paste the file unedited.)

## OBS-V10 Retention spares a live flock'd Rust writer on /mnt/ai-data

Implementation: `common/obs/retention.py` (`prune`), `rust-harness/obs/src/lib.rs`
(`RunLog::begin_at`).

The startup prune must not delete an event file a *Rust* process is still
appending to. The sandbox half of this property is already covered by
`tests/common/test_obs_retention.py::test_prune_spares_a_harness_event_file_locked_the_rust_way`,
and that test's own docstring states plainly that the subprocess **emulates**
`RunLog::begin_at` — it opens and flocks the file the same way, but in Python,
because building the Rust workspace is not a precondition of the unit suite.
This entry is the pairing the emulation cannot supply: the real Rust binary
holding the real lock, on the real `/mnt/ai-data/logs/harness`, against the
real Python prune. A lock convention that two implementations agree on only
because one of them was written to imitate the other has not been tested
across the boundary.

Command:

Any harness binary that runs longer than the prune takes will do; the property
is the lock, not the workload. Pick one whose run lasts at least 30 seconds and
confirm it is still alive at the moment the prune runs — a prune that spares a
file because the writer had already exited proves nothing.

```bash
cargo build --release --manifest-path rust-harness/Cargo.toml
ls -la /mnt/ai-data/logs/harness/ | tail -5
rust-harness/target/release/ladder <args for a run of 30 s or more> & PID=$!
sleep 5
kill -0 $PID && echo "rust writer alive"
LIVE=$(ls -t /mnt/ai-data/logs/harness/*.jsonl | head -1); echo "$LIVE"
OBS_RETENTION_DAYS=0 uv run --no-sync python -c \
    "from common.obs.retention import prune; print(prune('/mnt/ai-data/logs', 0))"
kill -0 $PID && echo "rust writer still alive after prune"
ls -la "$LIVE"
wait $PID; echo "ladder exit: $?"
uv run --no-sync python -c "import json,sys;[json.loads(l) for l in open('$LIVE')];print('all lines parse')"
```

`OBS_RETENTION_DAYS=0` makes every file on the mount eligible, including the
live one, so the spare is attributable to the lock rather than to the file
being too new to prune.

Expected signal: `prune` returns without error; `rust writer still alive after
prune` prints, so the writer was holding its lock across the prune; `$LIVE`
still exists after the prune with a size no smaller than before; the Rust
process exits 0; and every line of `$LIVE` parses as JSON afterwards, so the
prune neither removed nor truncated the file it spared.

Observed:

```
```

Status: PENDING

## OBS-V11 Watchdog alerts.jsonl and the mlflow sink in a real run

Implementation: `training/runtime/watchdog.py:368` (the `_alert` write of
`alerts.jsonl`), `training/runtime/tracking.py:179-208` (`_init_mlflow`).

Two shipped per-run artifacts have no entry in this runbook. OBS-V4 is titled
for tensorboard and offline wandb and covers only those two sinks, so the
mlflow sink's `mlruns/` tree is uncovered; and no entry anywhere trips a
watchdog signal and reads back the `alerts.jsonl` the alert path writes. Both
need a real training run on the GPU: the watchdog samples NVML, and the mlflow
sink is exercised only when a run actually logs metrics.

Command:

Uses OBS-V2's two recipe overrides, for the reason set out there. The
threshold override is what makes the alert fire deterministically — a
`gpu_temp_c` ceiling below room temperature trips on the first sampler tick
rather than waiting for a hot GPU.

```bash
uv run --no-sync python -m training train --config training/recipes/reference/tiny-e2e.yaml \
    data.source=/mnt/ai-data/training/shards/verify-small \
    tokenizer.path=HuggingFaceTB/SmolLM2-135M model.scratch.vocab=49152 \
    run.output_root=/mnt/ai-data/training/verify/obs-alerts \
    schedule.total_steps=200 watchdog.enabled=true watchdog.interval_s=1.0 \
    watchdog.signals.gpu_temp_c.max=1 tracking.mlflow=true
echo "exit=$?"
RUN=$(dirname "$(ls -t /mnt/ai-data/training/verify/obs-alerts/*/*/metrics.jsonl | head -1)")
echo "$RUN"
cat "$RUN/alerts.jsonl"
wc -l < "$RUN/alerts.jsonl"
find "$RUN/mlruns" -maxdepth 4 | head -20
```

Confirm the two override keys against `training/recipes/schema.py` before
running: if the watchdog threshold key or `tracking.mlflow` is spelled
differently the run fails validation, which is a precondition failure and not
a result.

Expected signal: exit 0; `alerts.jsonl` exists under the run directory with at
least one record, each a single JSON object carrying the signal name, its
value, the threshold, and the action; the run's stderr carries a matching
`ALERT gpu_temp_c` line; `mlruns/` exists under the run directory with a
non-empty experiment tree containing at least one run's `params/` and
`metrics/` entries; and no `[tracking] WARN` line naming mlflow appears, which
would mean the sink degraded rather than wrote.

Observed:

```
```

Status: PENDING

## Covered by a named gate check (roster, not entries)

A 2026-08-15 surface inventory found shipped observability functionality with
no presence in this runbook at all. Each row below names the test function
that pins it, which is the whole of the claim: an element whose evidence is
"the gate was green" without a named check is not verified, and would need a
full entry instead. Nothing in this table needs the GPU, the mount, or a
human, which is why it is a roster rather than eight more entries.

| Element | Pinned by |
|---------|-----------|
| Traced dispatch, `cmd.start`/`cmd.end` (`common/obs/dispatch.py`) | `tests/common/test_obs_dispatch.py::test_success_writes_start_and_end_with_shared_run_id`, and the exit-code cases beside it (`::test_uncaught_exception_records_exit_1_and_reraises`, `::test_keyboard_interrupt_records_130_and_reraises`, `::test_mid_run_system_exit_records_own_code`) |
| Audit trace and `OBS_TRACE` (`common/obs/audit.py`) | `tests/common/test_obs_audit.py::test_trace_records_open_import_and_spawn`, `::test_trace_off_records_no_audit_events`, `::test_explicit_log_level_wins_over_trace` |
| `OBS_LOG_LEVEL` | `tests/common/test_obs_core.py::test_obs_log_level_env_suppresses_info` |
| `report --events` | `tests/training/test_report.py::test_report_events_override_wins` |
| `runs --all` | `tests/common/test_obs_catalog.py::test_default_view_hides_runs_entries_until_all` |
| `runs --json` | `tests/common/test_obs_catalog.py::test_runs_json_lists_runs_including_failed` |
| `python -m common.obs.runs` as a module entry point | `tests/common/test_obs_runs.py::test_module_entry_point_emits_only_json_on_stdout`, `::test_module_entry_point_forwards_the_usage_exit_code` |
| `OBS_RETENTION_DAYS` | `tests/common/test_obs_retention.py::test_invalid_retention_value_disables_pruning`, which carries this row alone: it is the only check that reads the variable, and it fails both when the env read is removed and when the invalid-value fallback is changed. The sibling `::test_a_live_run_in_another_process_survives_a_startup_prune` was credited here and does not belong -- its fixture deletes `OBS_RETENTION_DAYS` and its subprocess prunes with a literal, so it pins cross-process liveness and says nothing about this variable. Also exercised operationally by OBS-V9 in [sandbox/sandbox-executable.md](../sandbox/sandbox-executable.md), which sets it across its prune invocations -- but the runbook cross-reference is context, not the gate check this column requires |

The two module entry-point tests are new as of 2026-08-15. Every earlier test
called `main()` in-process, so the module's `__main__` guard had no coverage:
`python -m common.obs.runs` could have been broken without a red gate.

## Moved to another tier

**OBS-V5, the browser harness**, is now
[sandbox/sandbox-executable.md](../sandbox/sandbox-executable.md). What it verifies —
`common/obs/viewer.js` rendering in a real browser — depends on a browser and not on this
workstation, so running it on the RTX 5090 proved nothing that running it in a container
does not. The identifier is not reused, and the numbering below keeps its gap.

**OBS-V9, retention pruning an aged file while sparing a live one**, is also
[sandbox/sandbox-executable.md](../sandbox/sandbox-executable.md), at the `OBS-V9` heading
below OBS-V5. It needs a `mktemp -d` log root and two CLI invocations, and nothing about
the property depends on this workstation. It is recorded here because retention is one of
the phases this runbook covers: a reader auditing retention coverage against this document
alone would otherwise conclude the operator-facing prune is unverified, when it is verified
in the other tier.

## Covered by the rust-harness runbooks

Three entries outside this document verify parts of the same tree, and until now this
runbook cited none of them. A reader deciding whether `common/obs` is covered needs them,
because each exercises a producer or a consumer this campaign's own entries do not reach.

| Entry | Runbook | What it covers here |
|-------|---------|---------------------|
| RH.13 | [rust-harness/rust-harness.md](../rust-harness/rust-harness.md), `docs/verification/rust-harness/rust-harness.md:1694` | A second writer of the shared obs contract: the Rust harness emits `${OBS_LOG_DIR}/harness/<run_id>.jsonl` and a catalog row through `common/obs/catalog.py`. |
| RHX.4 | [rust-harness/rust-harness-ux.md](../rust-harness/rust-harness-ux.md), `docs/verification/rust-harness/rust-harness-ux.md:190` | The training dashboard (`common/obs/dashboard.py`, `common/obs/viewer.js`) rendering a harness `metrics.jsonl` — the same reader OBS-V6 drives, against a non-training producer. |
| RHX.7 | [rust-harness/rust-harness-ux.md](../rust-harness/rust-harness-ux.md), `docs/verification/rust-harness/rust-harness-ux.md:327` | One catalog across the workspace: five harness binaries writing run records that `common/obs/runs.py` lists beside the training runs OBS-V3 checks. |

Line numbers are accurate as of 2026-08-15; the entry identifiers are the durable
reference.

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
