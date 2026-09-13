---
title: Run records
---

# Run records

Every `python -m interp` invocation runs through a traced dispatch: parse the arguments, mint
a run id, configure the log, and bracket the verb's handler between a `cmd.start` and a
`cmd.end` event. The result is one event file per run and one catalog line when it ends,
under one root shared by contract with the training tier and the harness, so one set of
readers serves all three. A usage error exits during parsing, before a run id exists, and
leaves nothing.

## Location

The root is `OBS_LOG_DIR`, default `/mnt/ai-data/logs`. Every verb writes under the `interp`
subdirectory of it, with the verb's name as the `command` field:

```text
${OBS_LOG_DIR}/interp/
  R-20260806-142530-a3f19c.jsonl      the run's event file
  runs.jsonl                          the catalog: one line per finished run
```

The run id is `R-YYYYMMDD-HHMMSS-xxxxxx`: a UTC stamp and six hex characters. Event files
older than `OBS_RETENTION_DAYS` (default 30) are pruned when a command starts; `runs` prunes
nothing, so browsing old runs never deletes the files being browsed. The verbs that run in
the workbench write to the same root through the container's mount of `/mnt/ai-data/logs`.

```bash
export OBS_LOG_DIR=/tmp/interp-demo/obs
```

## The event file

One JSON object per line. Every line carries `ts` (ISO 8601 with an explicit offset,
millisecond precision), `level` (`debug`, `info`, `warning`, or `error`), `run_id`, `cli`
(always `interp`), `command` (the verb), `pid`, `event`, and `msg`. The events:

| Event | Extra fields | Meaning |
|-------|--------------|---------|
| `cmd.start` | `argv` | The run began; `argv` is the argument list without the program name |
| `run.context` | `git_sha`, `python`, `platform`, `packages`, and `gpu` when one is visible | What the run ran on |
| `phase.start`, `phase.end` | `phase`, and `duration_s` on the end record | A coarse span the verb brackets, such as a capture or a training loop |
| `run.dir` | `path` | The run directory the verb created, once it exists |
| `fail` | `exc_type`, `exc_msg`, a truncated `traceback`, `exit_code` | The run failed; absent on success |
| `cmd.end` | `exit_code`, `duration_s` | The run ended; always written, with 1 for an uncaught exception and 130 for an interrupt |
| `open`, `import`, `spawn` | See [Resource audit](resource-audit.md) | Present only under `OBS_TRACE=1`, at debug level |

`exc_msg` is the message the terminal printed after `[ERROR]`, without the `Solution:` line
under it; the remedy is for the person and the diagnosis is for the record.

## The catalog

`runs.jsonl` gets one line per finished run, appended whole: `run_id`, `command`, `timestamp`,
`duration_s`, `exit_code`, and `run_dir`, the directory a verb recorded with `run.dir`, or
null for a verb that created none. The `runs` verb reads it, and `make interp-runs` runs that
verb on the host without the workbench image or the GPU preflight:

```bash
make interp-runs
make interp-runs ARGS='-n 50 --failed'
make interp-runs ARGS='--json'
```

The table columns are the catalog fields: finished timestamp, run id, command, exit code, and
duration. Invocations of `runs` itself are hidden unless `--all` is passed. `--json` prints
the selected records to standard output and nothing else, so the output pipes into `jq`; every
verb follows that rule, and status lines go to standard error. See the
[runs](../reference/runs.md) reference page for the flags.

## Querying with jq

The files are JSON Lines, so ad hoc questions need no query layer:

```bash
jq -c 'select(.event == "fail") | {run_id, command, exc_type, exc_msg}' \
    "$OBS_LOG_DIR"/interp/*.jsonl
jq -c '{ts, level, event, msg}' "$OBS_LOG_DIR"/interp/R-20260806-142530-a3f19c.jsonl
jq -r 'select(.event == "phase.end") | [.duration_s, .phase] | @tsv' \
    "$OBS_LOG_DIR"/interp/R-20260806-142530-a3f19c.jsonl | sort -rn
jq -r '[.duration_s, .run_id, .command] | @tsv' "$OBS_LOG_DIR"/interp/runs.jsonl | sort -rn | head
```

The first lists every recorded failure, the second the timeline of one run, the third its
phases slowest first, and the fourth the slowest runs in the catalog.

## Cross-process propagation

A verb that spawns another traced command, or the harness, passes its run id through
`OBS_RUN_ID`; the child mints its own id and records the parent's as `parent_run_id` on its
`cmd.start` event. That is how a scorecard's harness runs are found from the scorecard's
run id.
