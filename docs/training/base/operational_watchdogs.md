# Operational watchdogs

Automated abort-or-alert for unattended training runs (Phase 0 D10). The watchdog
(`/training/runtime/watchdog.py`) hooks into the reference loop, evaluates step-level signals on
every optimizer step, samples hardware telemetry from a background thread, and doubles as
the run's hardware telemetry source. NVML-based; DCGM is not used. Design record:
[the D10 investigation](../investigations/d10_operational_watchdogs_investigation.md).

## Table of contents

- [Signals and actions](#signals-and-actions)
- [Configuration](#configuration)
- [Actions in detail](#actions-in-detail)
- [Emergency checkpoints](#emergency-checkpoints)
- [Telemetry](#telemetry)
- [Testing](#testing)

## Signals and actions

| Signal | Source | Trigger | Action |
|--------|--------|---------|--------|
| `loss_non_finite` | every step | NaN or inf loss | abort |
| `loss_spike` | every step | loss exceeds `loss_spike_factor` x trailing median over `loss_window` steps | abort |
| `grad_norm` | every step | norm exceeds `grad_norm_max` for `patience` consecutive steps | abort |
| `gpu_temp` | sampler | above `gpu_temp_limit_c` | alert; abort when sustained `sustain_s` |
| `gpu_power` | sampler | draw above `power_w` sustained `sustain_s` | alert only (data for D8) |
| `gpu_throttle` | sampler | clock-throttle reasons active sustained `sustain_s` | alert only |
| `ram_floor` | sampler | available RAM below `ram_floor_gb` | alert; abort below half the floor |
| `disk_floor` | sampler | free space on the checkpoint filesystem below `disk_free_floor_gb` | block periodic checkpoint writes; alert |

The loss-spike rule arms after five finite loss samples; early-training noise does not
trip it. The sampler thread cannot raise into the training step, so hardware aborts are
latched and raised at the next step boundary. All sampling is fail-soft: a machine
without NVML or psutil yields absent readings, never an exception.

## Configuration

All thresholds live in the recipe `watchdog` block (`/training/recipes/schema.py`,
validated by `/training/recipes/validate.py`):

```yaml
watchdog:
  enabled: true
  interval_s: 10.0          # NVML/psutil sampling period
  loss_spike_factor: 3.0    # abort when loss exceeds factor x running median
  loss_window: 100          # trailing-median window
  grad_norm_max: 0.0        # 0 disables the gradient-norm signal
  patience: 5               # consecutive exceeds before the grad-norm abort
  gpu_temp_limit_c: 90
  sustain_s: 120.0
  power_w: 600.0            # card power limit; alert-only signal
  ram_floor_gb: 8.0
  disk_free_floor_gb: 100
```

`grad_norm_max` defaults to 0 (disabled) because a useful bound is run-specific; set it
per recipe once a healthy norm range is known.

## Actions in detail

- **log**: every trigger prints `[watchdog] ACTION signal: value, threshold` to stderr
  and logs `watchdog/<signal>` to the tracking sinks.
- **alert**: the trigger record (timestamp, step, signal, value, threshold, action) is
  appended to `<run_dir>/alerts.jsonl`. A pluggable `alert_hook` callable receives each
  record for future email or ntfy delivery; Phase 0 ships file plus log only.
- **abort**: the loop saves an emergency checkpoint, then `WatchdogAbort` propagates.
  The CLI reports signal, value, and threshold and exits with code 1.
- **block** (disk floor only): the run continues but periodic checkpoint saves are
  skipped until free space recovers; the abort-path emergency save is still permitted.

## Emergency checkpoints

An abort saves a final checkpoint through the normal atomic path with `emergency: true`
in `meta.json`. Retention (`apply_retention` in `/training/runtime/checkpoint.py`) never deletes
an emergency checkpoint; it is post-mortem evidence and the resume point. Resume with
`checkpoint.resume=latest` after diagnosing the trigger recorded in `alerts.jsonl`.

The RAM half-floor abort also attempts the emergency save; the save is best-effort and a
failure is logged rather than masking the abort.

## Telemetry

Every sample (GPU temperature, power, SM clock, throttle state, VRAM, host RAM, disk
free) is forwarded to the D6 tracker at `interval_s`, so tensorboard, wandb, and mlflow
carry the run's hardware history alongside training metrics. D8's benchmark JSON reads
the same stream.

## Testing

`/tests/training/test_watchdog.py` injects synthetic loss, temperature, RAM, and disk
readings (no NVML or psutil calls) and asserts the configured actions, the retention
exemption, and the loop's emergency checkpoint path. Host-side acceptance: an
exploding-LR run aborts via the loss rule and leaves a resumable emergency checkpoint;
telemetry is visible in tensorboard for a tiny reference run; the disk floor is
exercised against a small tmpfs.
