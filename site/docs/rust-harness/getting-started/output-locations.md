---
title: Output locations
---

# Output locations

A run writes into two places: the working directory, and the observability log root. This page
lists every file, which mode writes it, and what controls the location.

## In the working directory

| File | Written by | When |
|------|------------|------|
| `session.jsonl` | task and chat modes | Appended on every turn: one JSON object per user, model, tool, or compaction record |
| `session.kv` | task and chat modes | Written on clean exit; restored on the next start in the same directory |
| `probe-flags.jsonl` | any mode with `--probe` | Written at end of run; the path is `--probe-out` |
| the `--alignment-out` file | any mode with `--alignment-out` | Written at end of run |
| the `--record` file | any mode with `--record` | Appended on every forward call |
| the `--trace-chrome` file | any mode with `--trace-chrome` | Written as the run proceeds and closed at exit |
| `<dir>/harness/YYYY-MM-DD.jsonl` | serve mode with `--interp-log <dir>` | One record appended per completed request |

The working directory is also the root the agent's tools operate in and the place
`permissions.yaml` and `HARNESS.md` are read from. Run the harness from the directory that is
meant to be its workspace.

Serve mode writes neither `session.jsonl` nor `session.kv`: a server's requests are unrelated
callers, and there is no conversation to resume.

## Under the observability log root

Every `harness` run, and every run of `activation-capture`, `attention-capture`, `layer-diff`,
and `ladder`, writes run-level records under `${OBS_LOG_DIR}/harness/`. The default root is
`/mnt/ai-data/logs`. Set `OBS_LOG_DIR` to move it:

```bash
export OBS_LOG_DIR=/tmp/harness-demo/obs
```

A run leaves:

| Path under `${OBS_LOG_DIR}/harness/` | Contents |
|--------------------------------------|----------|
| `<run_id>.jsonl` | The run's event file: start, phases, the resolved configuration, failure if any, end |
| `runs.jsonl` | One catalog line appended per finished run |
| `<run_id>/metrics.jsonl` | A header record then one record per step; absent for the capture binaries |

`<run_id>` has the shape `R-YYYYMMDD-HHMMSS-xxxxxx`. If the root cannot be created or written,
recording degrades to a silent no-op and the run proceeds; `harness --status` reports which
case applies before a run. `checkpoint-info`, `tiny-model`, and `long-context-golden` write no
records. The record formats are documented under [Observability](../observability/index.md).
