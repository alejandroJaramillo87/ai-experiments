---
title: Run records
---

# Run records

Every run of `harness`, `ladder`, `activation-capture`, `attention-capture`, and `layer-diff`
writes run-level records: which invocation happened, what it resolved its configuration to,
how long each phase took, whether it failed, and a per-step series. They are JSONL under one
root, shared with the repository's Python tooling by contract, so one catalog and one set of
readers serve both. `checkpoint-info`, `tiny-model`, and `long-context-golden` write none.

## Location

The root is `OBS_LOG_DIR`, default `/mnt/ai-data/logs`. Every binary here writes under the
`harness` subdirectory of it, with its own name as the `command` field:

```text
${OBS_LOG_DIR}/harness/
  R-20260912-012716-997ae9.jsonl      the run's event file
  R-20260912-012716-997ae9/           the run directory
    metrics.jsonl                     header then one record per step
  runs.jsonl                          the catalog: one line per finished run
```

The run id is `R-YYYYMMDD-HHMMSS-xxxxxx`: a UTC stamp and six hex characters. If the root
cannot be created or the event file cannot be opened, the writer becomes a silent no-op and the
run proceeds; `harness --status` reports the verdict as its last line. Usage errors exit before
a run id exists and leave nothing.

```bash
export OBS_LOG_DIR=/tmp/harness-demo/obs
```

## The event file

One JSON object per line. Every line carries `ts` (ISO 8601 with an explicit `+00:00`
offset), `level`, `run_id`, `cli` (always `harness`), `command` (`task`, `chat`, `serve`,
`ladder`, `activation-capture`, `attention-capture`, or `layer-diff`), `pid`, `event`, and
`msg`. The events, in the order they appear:

| Event | Extra fields | Meaning |
|-------|--------------|---------|
| `cmd.start` | `argv`, and `parent_run_id` when `OBS_RUN_ID` was set by a parent process | The run began; `argv` is the argument list without the program name |
| `phase.start`, `phase.end` | `phase`, and `duration_s` on the end record, rounded to three decimals | A coarse span. The harness brackets `model.load`, then `prefill` and `decode` around each generation; the ladder brackets `model.load` and each rung as `rung.L0` through `rung.L6`; the capture binaries bracket `model.load` alone |
| `run.header` | See below | The harness's resolved configuration, written once after the model loads |
| `ladder.header`, `activation-capture.header`, `attention-capture.header`, `layer-diff.header` | Per tool | The same, for the other writers; each has its own field set |
| `fail` | `exc_type`, `exc_msg` capped at 4096 characters, `exit_code` | The run failed; absent on success |
| `cmd.end` | `exit_code`, `duration_s` | The run ended |

`run.header` is what the loader and the flags settled on, as opposed to what was asked for:
`model`, `model_hash`, `n_layers`, `dtype`, `weight_bytes`, `max_context`, `kv_bytes`,
`kv_batch_group`, `threads`, `dot`, `batch`, `seed`, `temperature`, `max_tokens`,
`binary_version`, and `git_sha`, plus `peak_rss_bytes`, `rewind_horizon`, and, with a probe
loaded, `probe` and `probe_layer`. The terminal's status header prints the same facts from the
same struct.

A phase is a different mechanism from a trace span and is present on every run; see
[Tracing](tracing.md) for the distinction. `prefill` and `decode` land once per generation,
not once per token, so a server's event file grows with requests rather than with tokens.

## The catalog

`runs.jsonl` gets one line per finished run, appended whole so concurrent writers interleave
whole lines: `run_id`, `command`, `timestamp`, `duration_s`, `exit_code`, and `run_dir`, the
path of the run directory below.

```text
{"run_id":"R-20260912-012716-997ae9","command":"task","timestamp":"2026-09-12T01:27:21.249+00:00","duration_s":4.292,"exit_code":0,"run_dir":"/tmp/harness-demo/obs/harness/R-20260912-012716-997ae9"}
```

The repository's Python `runs` command reads this file; from the repository root,
`make harness-runs` lists recent harness runs from it.

## The metrics file

`<run_dir>/metrics.jsonl` opens with one header record carrying `schema_version`, `report`
(`run_metrics`), `timestamp`, `git_sha`, `run_id`, `cli`, and `command`, merged with the same
fields as `run.header`. Every later record carries `step` and `ts` plus emitter-chosen series:

- Task and chat write one record per decoded token with `tokens_per_s` and `rss_gb`, bounded
  by `--max-tokens`.
- Serve writes one per completed request, because a server has no run boundary to bound a
  per-token file against.
- The ladder writes exactly seven, one per rung, with `rung`, `status`, `rung_duration_s`, and
  the rung's own measures.

`step` is an ordering key and nothing more: a token index for the generation modes and a rung
index for the ladder. The directory is created on the first metrics write, so a run that
failed before generating leaves no directory. The three capture binaries write no metrics file
at all.

## Reading a run back

The files are plain JSONL, so any tool reads them. To list the runs recorded under a root and
print the phases of the latest:

```bash
tail -n 3 "$OBS_LOG_DIR/harness/runs.jsonl"
LATEST=$(tail -n 1 "$OBS_LOG_DIR/harness/runs.jsonl" | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')
grep '"phase.end"' "$OBS_LOG_DIR/harness/$LATEST.jsonl"
```

## Cross-process propagation

A parent process that sets `OBS_RUN_ID` before launching the harness has that id recorded as
`parent_run_id` on the child's `cmd.start`; the child still mints its own run id. Nothing else
is inherited.
