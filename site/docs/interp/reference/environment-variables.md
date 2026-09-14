---
title: Environment variables
---

# Environment variables

The verbs read four `OBS_` variables, shared with the training tier, and honor `NO_COLOR`.
None is required. Two further variables are read by the harness rather than by this tier and
are listed because the interp workflows set them.

| Variable | Read by | Effect | Default |
|----------|---------|--------|---------|
| `OBS_LOG_DIR` | Every verb | The root the run records are written under; the verbs write into its `interp` subdirectory | `/mnt/ai-data/logs` |
| `OBS_LOG_LEVEL` | Every verb | Minimum level written to the console and the event file | `info` |
| `OBS_TRACE` | Every verb | When `1`, installs the resource audit hook and raises the level to `debug` unless `OBS_LOG_LEVEL` is set | `0` |
| `OBS_RETENTION_DAYS` | Every verb except `runs` | Age past which event files are pruned at command start; `0` disables pruning | `30` |
| `OBS_RUN_ID` | Every verb | Set by a parent process; recorded as `parent_run_id` on the run's `cmd.start` event, while the run still mints its own id | unset |
| `NO_COLOR` | Every verb | When set to anything, the empty string included, status lines carry no color escapes | unset |

Setting `OBS_LOG_DIR` is the one an off-workstation run needs, because the default root is
not reachable outside it:

```bash
export OBS_LOG_DIR=/tmp/interp-demo/obs
```

## Set for the harness

| Variable | Read by | Effect |
|----------|---------|--------|
| `HARNESS_PROBE` | `make harness-serve` | The exported probe the harness scores in-band, in its container spelling under `/app/probes` |
| `HARNESS_INTERP_LOG` | `make harness-serve` | Container path of the served-request log root; absent means no log |
| `CONTROL_VECTOR`, `CONTROL_VECTOR_SCALE`, `CONTROL_VECTOR_LAYER_RANGE` | `make steer-up` and `steer-down` | The vector, scale, and block range the serving harness is recreated with; `steer-up` sets them from `VECTOR`, `SCALE`, and `LAYER_RANGE` |

See [Served-request log](../observability/served-request-log.md) for the first two and the
[Steering](../guides/steering.md) guide for the third.
