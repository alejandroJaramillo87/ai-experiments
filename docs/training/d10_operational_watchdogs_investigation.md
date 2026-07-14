# D10 operational watchdogs: implementation investigation

Comparison of the Phase 0 D10 design (roadmap document, "D10: Operational watchdogs")
against the current repository state, with a work breakdown, acceptance checks,
deviations, and open questions. Per operating rules, the repository is the source of
truth; where the roadmap document disagrees with the repository, this investigation
records the repository state and treats the document as advisory.

## Table of contents

- [Design summary](#design-summary)
- [Current repository state](#current-repository-state)
- [Deviations from the document](#deviations-from-the-document)
- [Work breakdown](#work-breakdown)
- [Acceptance checks](#acceptance-checks)
- [Risks](#risks)
- [Resolved decisions](#resolved-decisions)

## Design summary

D10 gives multi-day unattended runs automated abort-or-alert on the failure modes that
destroy runs or hardware value: numerical divergence, thermal and power anomalies, RAM
exhaustion under offload, and disk exhaustion under checkpointing. Tag: orchestrate.
Depends on D1 (container), D5 (recipe schema), and D6 (callback surface plus telemetry
sink), all complete. NVML-based; DCGM is explicitly not used (datacenter-oriented,
degraded on GeForce — NVML exposes everything needed on the RTX 5090). The document
specifies:

1. **Two layers.** An in-process callback (`training/watchdog.py`, hooked into the D6
   loop and later TRL via its callback API) that sees loss and gradient norm every
   step and samples NVML and psutil every `interval_s` (default 10s) from a background
   thread; and an optional out-of-process sidecar for trainers that cannot host the
   callback, shipped as a thin wrapper over the same module.
2. **Signals and defaults** (all thresholds in the recipe `watchdog` block):
   - NaN/inf loss, or loss above a factor times its trailing median: abort after a
     final emergency checkpoint (marked `emergency=true`, exempt from retention
     deletion).
   - Gradient norm above `grad_norm_max` for `patience` consecutive steps: abort.
   - GPU temperature above the limit (hotspot/memory junction where NVML exposes it):
     alert, then abort if sustained for `sustain_s` (default 120s).
   - Power draw sustained above `power_w` (default 600, the card's limit) or clock
     throttle reasons active beyond `sustain_s`: alert only (data for D8, not fatal).
   - Host RAM available below `ram_floor_gb` (default 8): alert; below half the floor:
     abort before the OOM killer chooses a victim (load-bearing under ZeRO-Offload in
     Phase 2).
   - Disk free on `/mnt/ai-data` below the floor (default 100 GB): block further
     checkpoint writes and alert; the run continues, and the abort-on-NaN emergency
     save is still permitted.
3. **Actions:** log (tracking sink plus stderr), alert (append to
   `runs/<run>/alerts.jsonl`; a pluggable hook function for future email or ntfy —
   Phase 0 ships file plus log only), abort (emergency checkpoint, nonzero exit).
   Every trigger records signal, value, threshold, and timestamp in the run's tracking
   stream.
4. **Telemetry fan-in:** all sampled telemetry (GPU temperature, power, clocks, VRAM,
   host RAM, disk free) also flows to tracking at the sampling interval, making the
   watchdog double as the run's hardware telemetry source, including for D8's
   benchmark JSON.

## Current repository state

No watchdog code exists; only forward-reference stubs. The plug points D10 needs are
already in place and constrain the implementation:

- **Schema stub shipped with D5.** `training/recipes/schema.py:139-146` defines
  `WatchdogConfig` with `enabled: bool = True`, `loss_spike_factor: float = 3.0`
  (abort when loss exceeds factor times running median), `gpu_temp_limit_c: int = 90`,
  and `disk_free_floor_gb: int = 100`. These names and defaults are the repository
  truth. `training/recipes/validate.py:122-126` checks positivity only, with semantics
  explicitly deferred to D10. `training/recipes/reference/tiny-e2e.yaml` sets
  `watchdog.enabled: false`.
- **No callback mechanism in the loop.** `training/loop.py` `train()` is a plain
  per-step loop. It already computes every step-level signal D10 needs: `loss`, `lr`,
  `grad_norm` (via `clip_grad_norm_(..., inf)`), `tokens_per_s`, `step_time_s`, and
  GPU memory, assembled into a metrics dict handed to `tracker.log(metrics, step)`.
  That call site is the natural watchdog insertion point. The nested `save(step,
  loss)` closure is the checkpoint trigger an emergency save can reuse.
- **Tracker is the designated telemetry surface.** `training/tracking.py` fans one
  `log(dict, step)` to tensorboard, wandb (offline-first), and mlflow, and its
  docstring names it the stable surface D10 telemetry plugs into. Sinks fail soft: a
  raising sink is disabled with a stderr warning rather than killing the run. There is
  no JSONL sink; `alerts.jsonl` is new surface.
- **Emergency checkpoints are nearly free.** `training/checkpoint.py`
  `save_checkpoint(root, step, model, optimizer, scheduler, dataset_state, meta)` is
  atomic (tmp dir, fsync, rename, `latest` symlink) and `meta` is a free-form dict
  written to `meta.json`, so `emergency: true` needs no format change. However,
  `apply_retention()` keeps only last-N plus every-Mth plus `latest`; an emergency
  checkpoint outside that set would be deleted. The retention exemption is a small
  required change to D6 code.
- **No abort path.** The loop runs to `schedule.total_steps` (or the test-only
  `stop_after`), with `try/finally: tracker.close()`. `training/cli.py` `cmd_train`
  catches `RuntimeError, FileNotFoundError, ValueError` and returns exit code 1. A
  watchdog abort needs an exception type, an explicit emergency save before raising,
  and a CLI catch producing a distinct, documented nonzero exit.
- **NVML precedent, psutil gap.** `training/calculator/calibrate.py:62-70` already
  samples NVML memory (`nvmlInit`/`nvmlDeviceGetMemoryInfo`/`nvmlShutdown`);
  `pynvml==12.0.0` is a direct training-group dependency. Temperature, power, and
  throttle-reason calls are new NVML surface. `psutil` is only a transitive dependency
  today and must be added to the training group in `pyproject.toml` with `uv.lock`
  regenerated.
- **Script layout.** The repository uses `scripts/training/`, not the document's
  `scripts/train/`. Any sidecar belongs at `scripts/training/watchdog_daemon.py`.
- **Test conventions.** `tests/training/test_loop.py` is the model: `importorskip`
  for heavy deps, CPU-only tiny scratch model driven by `tiny-e2e.yaml` with dotlist
  overrides, synthetic shards in `tmp_path`. No NVML/psutil mocking exists yet; D10
  unit tests must mock both, since the agent sandbox has no GPU.

## Deviations from the document

| Document | Repository truth / decision |
|----------|-----------------------------|
| `explode_factor` (default 10x) | Existing field `loss_spike_factor` (default 3.0) is kept; the name and default are repository truth. |
| `gpu_temp_c` (default 88) | Existing field `gpu_temp_limit_c` (default 90) is kept. |
| Sidecar at `scripts/train/watchdog_daemon.py` in Phase 0 | Deferred. The reference loop is the only trainer today; the sidecar's consumer (TRL, Phase 1) does not exist. The module is designed so a thin wrapper is trivial later. |
| Grad-norm, RAM-floor, power, `sustain_s`, `patience`, `interval_s` fields | Absent from the schema today; added as new fields alongside the existing ones, with the document's defaults. |
| Emergency checkpoints exempt from retention | `apply_retention()` currently has no exemption; D10 adds one (small D6 change). |
| Alerts to `runs/<run>/alerts.jsonl` | No JSONL surface exists; added as part of D10, following the tracker's fail-soft contract. |

## Work breakdown

1. **Schema and validation.** Extend `WatchdogConfig` with `interval_s` (10.0),
   `grad_norm_max`, `patience`, `sustain_s` (120), `power_w` (600), `ram_floor_gb`
   (8); extend `_check_watchdog` with the real cross-field semantics. Keep existing
   field names and defaults unchanged.
2. **Dependency.** Add `psutil` (pinned) to the `training` group in `pyproject.toml`;
   regenerate `uv.lock`.
3. **`training/watchdog.py`.**
   - Pure signal-evaluation functions over reading snapshots (trailing-median loss
     window, patience/sustain counters) — unit-testable without hardware.
   - A sampler thread reading NVML (temperature, power, throttle reasons, VRAM) and
     psutil (available RAM, disk free on the checkpoint filesystem) every
     `interval_s`, fail-soft on NVML absence.
   - Actions: log to stderr and tracker, append alert records to
     `runs/<run>/alerts.jsonl` via a pluggable hook, raise `WatchdogAbort` after an
     emergency checkpoint. Every trigger records signal, value, threshold, timestamp.
   - Telemetry fan-in: sampled readings forwarded to `tracker.log` at the sampling
     interval.
4. **Loop integration.** Invoke the watchdog at the per-step metrics point in
   `training/loop.py`; pass the `save` closure for emergency checkpoints; honor the
   disk-floor block on periodic saves while permitting the emergency save.
5. **Retention exemption.** Teach `apply_retention()` to skip checkpoints whose
   `meta.json` carries `emergency: true`.
6. **CLI.** Catch `WatchdogAbort` in `cmd_train`, emit a styled `[ERROR]` line with
   signal and threshold, return exit code 1.
7. **Tests and docs.** Unit tests with mocked NVML/psutil per the acceptance list;
   user-facing doc `docs/training/operational_watchdogs.md` after implementation.

## Acceptance checks

From the roadmap document, restated against the repository:

1. Unit tests inject synthetic NaN loss, temperature, RAM, and disk readings
   (NVML/psutil mocked) and assert the configured action fires, including the
   emergency checkpoint path. Sandbox-runnable.
2. An integration run with an intentionally exploding LR aborts via the loss-explosion
   rule and leaves a resumable emergency checkpoint. Host-side; the human runs the
   command and reports output.
3. Telemetry appears in tensorboard for a tiny reference run. Host-side.
4. Disk-floor behavior verified by pointing the check at a small tmpfs. Host-side (or
   sandbox with a mocked statvfs, in addition).

## Risks

- **Sampler-thread interference.** A 10s NVML/psutil poll is negligible next to a
  training step, but the thread must never hold the GIL through a blocking NVML call
  pattern that stalls steps; keep per-sample work minimal and fail-soft.
- **NVML surface variance.** Hotspot/memory-junction temperature and throttle-reason
  fields vary by driver; every optional reading degrades to absent rather than
  raising.
- **Retention interaction.** The exemption must not let emergency checkpoints
  accumulate unboundedly across repeated aborted runs; only the newest emergency
  checkpoint per run needs protection (open question below).
- **False-positive loss spikes.** Early-training loss is noisy; the trailing-median
  window needs a warmup period before the spike rule arms.

## Resolved decisions

1. **Schema evolution:** keep the repository's existing `WatchdogConfig` field names
   and defaults; extend with the missing fields.
2. **Sidecar:** callback only in Phase 0; the daemon wrapper is deferred until a
   trainer that cannot host the callback exists.
3. **Scope:** this investigation lands first for review; implementation follows on
   the same branch after sign-off.

## Open questions

1. Should the emergency-checkpoint retention exemption protect all emergency
   checkpoints, or only the most recent one per run?
2. Should the RAM half-floor abort also take an emergency checkpoint, or exit
   immediately to avoid allocating during memory pressure?
