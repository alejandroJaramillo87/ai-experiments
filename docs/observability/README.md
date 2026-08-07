# Runtime observability

How to read the logs, find a run, and open a dashboard for the training and interpretability
command-line interfaces (`python -m training`, `python -m interp`).

Every run mints one correlation id, the `run_id`, and stamps it on every log record, every
metrics record, and the run catalog. Reconstructing a run means starting from that id. This
document describes the system as it is; the design rationale and the item ledger that built it
are in [design-and-roadmap.md](design-and-roadmap.md), and the separate JSON *report* envelope
used by scorecards and evaluation output is in [report_schema.md](report_schema.md).

## Table of contents

- [Where the files are](#where-the-files-are)
- [Configuration](#configuration)
- [Output streams](#output-streams)
- [Finding a run](#finding-a-run)
- [Opening a dashboard](#opening-a-dashboard)
- [The event record schema](#the-event-record-schema)
- [The metrics record schema](#the-metrics-record-schema)
- [Querying with jq](#querying-with-jq)
- [Related documentation](#related-documentation)

## Where the files are

| File | Path | Written by |
|------|------|------------|
| Event file | `${OBS_LOG_DIR}/<cli>/<run_id>.jsonl` | The traced dispatch, one file per run |
| Run catalog | `${OBS_LOG_DIR}/<cli>/runs.jsonl` | One summary line appended when a command ends |
| Metrics file | `<run_dir>/metrics.jsonl` | A sink inside the tracker fan-out |
| Snapshot dashboard | `<run_dir>/run.html` | `report`, on demand |

`<cli>` is `training` or `interp`. `OBS_LOG_DIR` defaults to `/mnt/ai-data/logs`
(`common/obs/core.py`, `DEFAULT_LOG_DIR`), which is not reachable from an agent sandbox — set
the variable to a local path to exercise any of this off the workstation.

`<run_dir>` is the directory the training loop creates,
`${run.output_root}/${run.name}/<timestamp>/` (`training/recipes/loader.py`, `create_run_dir`).
The event file and the metrics file are joined by the `run_id`: the metrics header carries it,
and the event file records the run directory once the command creates one, as a `run.dir`
event. Either file locates the other.

Run ids have the form `R-<YYYYMMDD>-<HHMMSS>-<hex>`, minted in UTC
(`common/obs/core.py`, `new_run_id`) — for example `R-20260806-142530-a3f19c`.

The Rust harness is a second, contract-compatible writer of this same layout, under
`${OBS_LOG_DIR}/harness/`. Its runs appear to every reader below unchanged; see
[../rust-harness/observability.md](../rust-harness/observability.md).

## Configuration

Four environment variables, all prefixed `OBS_`:

| Variable | Default | Effect |
|----------|---------|--------|
| `OBS_LOG_LEVEL` | `info` | Minimum level written to the console and the file |
| `OBS_LOG_DIR` | `/mnt/ai-data/logs` | Root of the event files and catalogs |
| `OBS_TRACE` | `0` | When `1`, installs the resource audit hook and raises the level to `debug` |
| `OBS_RETENTION_DAYS` | `30` | Age past which event files are pruned; `0` disables pruning |

`NO_COLOR` is honored as well, by the existing `Printer` rendering rather than by this system.

Setting `OBS_TRACE=1` raises the log level to `debug` unless `OBS_LOG_LEVEL` is set explicitly
(`common/obs/dispatch.py`), because the audit records the hook emits are `debug`-level and would
otherwise be filtered out of the file it was enabled to populate.

Retention runs opportunistically at command start. The viewer commands (`runs`, `report`) set
`obs_retention=False`, so browsing old runs never prunes the files being browsed.

## Output streams

One rule governs the console:

- **Standard output carries `--json` payloads only.** In `--json` mode a command prints its
  result object and nothing else, so a caller can parse it.
- **Standard error carries all human status.** Human lines are suppressed in `--json` mode.
- **The file record is always written**, in both modes. A `--json` run has a quiet console and
  a complete event file.

Human lines carry no timestamps and keep the `[STATUS] Subject: detail` rendering defined in
[../../.styles/PYTHON_OUTPUT_STYLE.md](../../.styles/PYTHON_OUTPUT_STYLE.md). Timestamps live in
the file records.

## Finding a run

Both CLIs expose the catalog through a `runs` subcommand:

```bash
$ python -m training runs                  # 20 most recent runs
$ python -m training runs -n 100           # more of them; 0 or less shows all
$ python -m training runs --failed         # only nonzero exit codes
$ python -m training runs --json           # the records, on stdout, for jq
$ python -m interp runs                    # the interp catalog
```

By default the catalog listing hides invocations of `runs` itself; `--all` includes them.

The table columns are the catalog record fields: finished timestamp, run id, command, exit
code, and duration. Each record also carries `run_dir`, which is what `report` needs and what
`--json` exposes.

## Opening a dashboard

`report` renders one run's metrics and logs into a single self-contained HTML file that
references no external URL:

```bash
$ python -m training report --run /mnt/ai-data/runs/sft-1b/20260806-142530
```

That writes `run.html` into the run directory. The event file is located automatically from the
metrics header's run id; pass `--events <path>` to override that.

For a run that is still going, `--serve` starts an ephemeral local server that polls the growing
files instead of writing a snapshot:

```bash
$ python -m training report --run <run_dir> --serve
$ python -m training report --run <run_dir> --serve --port 8010
```

The server binds `127.0.0.1:8010` by default. It is a stateless viewer: it never writes the
files it reads, and killing it loses nothing. **Warning!** `--host 0.0.0.0` exposes the page to
the local network, and there is no authentication.

## The event record schema

Every line of an event file is one JSON object. All records carry the base fields:

| Field | Type | Description |
|-------|------|-------------|
| `ts` | string | ISO 8601 local timestamp with offset, millisecond precision |
| `level` | string | One of `debug`, `info`, `warning`, `error` |
| `run_id` | string | The correlation id |
| `cli` | string | `training` or `interp` |
| `command` | string | The subcommand name, for example `train-sft` |
| `pid` | integer | Process id |
| `event` | string | Machine-readable event name |
| `msg` | string | Human-readable message; empty for pure-data events |

Each event type adds its own fields:

| `event` | Added fields | Level |
|---------|--------------|-------|
| `cmd.start` | `argv` | info |
| `cmd.end` | `exit_code`, `duration_s` | info |
| `phase.start` | `phase` | info |
| `phase.end` | `phase`, `duration_s` | info |
| `fail` | `exc_type`, `exc_msg`, `traceback` (truncated), `exit_code` | error |
| `run.context` | `git_sha`, `python`, `platform`, `packages`, `gpu` (optional) | info |
| `run.dir` | `path` | info |
| `open` | `path`, `mode` | debug |
| `import` | `module` | debug |
| `spawn` | `argv0`, `argv` | debug |

The last three arrive only under `OBS_TRACE=1`. Application code emits its own named events —
`dataset_loaded` with an `examples` count, for instance — carrying the base fields plus whatever
the call passes.

No record type captures environment variables, and record content is written as passed, with no
masking or redaction. The schema carries `git_sha` and `recipe_hash` deliberately, and a
pattern-based filter would corrupt exactly those fields.

## The metrics record schema

Metrics live in their own per-run file because they are a dense per-step series rather than
discrete log records. The sink writes metric keys exactly as the training loops log them, so the
same names appear here and in tensorboard.

The first line is a header conforming to the shared report envelope
([report_schema.md](report_schema.md)), built with `report_header("run_metrics")` and extended
with `run_id`, `cli`, `command`, `tier`, `model_id`, `recipe_hash`, and `schedule`. A `Tracker`
constructed outside a dispatched CLI writes the identity fields as empty strings; the header
still validates, and the join to an event file is simply unavailable.

Per-step records carry `step`, `ts`, and the metric keys. Every line is strict JSON: non-finite
values serialize as `null` rather than the bare `NaN` literal that `jq` and `JSON.parse` reject,
which matters because a NaN loss is precisely the record a post-mortem reads.

The two loop families use different key vocabularies. Both are written as-is, and the dashboard
maps both:

| Series | D6 loop (`training/runtime/loop.py`) | SFT and distill (`training/tune/sft_callbacks.py`) |
|--------|--------------------------------------|----------------------------------------------------|
| Training loss | `loss` | `train/loss` |
| Held-out loss | — | `eval/holdout_loss` |
| Learning rate | `lr` | `train/lr` |
| Gradient norm | `grad_norm` | `train/grad_norm` |
| Throughput (tokens/second) | `tokens_per_s` | `train/tokens_per_s` |
| Step time (seconds) | `step_time_s` | — |
| Data wait (milliseconds) | `data_wait_ms` | — |
| Optimizer step time (offload runs) | `optimizer_step_s` | — |
| Per-source token counts (mixtures) | `tokens_source_<name>` | — |
| Device memory (GB) | `gpu_mem_allocated_gb` | — |
| Distillation CE and KL decomposition | — | `train/ce_loss`, `train/kl_loss` |

Interval records arrive from the watchdog sampler thread
(`training/runtime/watchdog.py`) carrying `gpu_temp_c`, `gpu_power_w`, `gpu_vram_used_gb`,
`gpu_sm_clock_mhz`, `gpu_throttle_active`, `ram_available_gb`, and `disk_free_gb`. They are
ordinary records, distinguishable by their keys.

Marker records reuse the same shape: `checkpoint_saved` at a save site, `watchdog/<signal>` when
a watchdog threshold is crossed, and `eval/holdout_loss` when an evaluation ran.

## Querying with jq

The files are JSON Lines, so ad hoc analysis needs no query layer. Set `OBS_LOG_DIR` first if
the default is not where the files are.

Every failure recorded across all training runs:

```bash
$ jq -c 'select(.event == "fail") | {run_id, command, exc_type, exc_msg}' \
    /mnt/ai-data/logs/training/*.jsonl
```

The full timeline of one run, given its id:

```bash
$ jq -c '{ts, level, event, msg}' /mnt/ai-data/logs/training/R-20260806-142530-a3f19c.jsonl
```

Phase durations for one run, slowest first:

```bash
$ jq -r 'select(.event == "phase.end") | [.duration_s, .phase] | @tsv' \
    /mnt/ai-data/logs/training/R-20260806-142530-a3f19c.jsonl | sort -rn
```

The slowest ten completed runs in the catalog:

```bash
$ jq -r '[.duration_s, .run_id, .command] | @tsv' \
    /mnt/ai-data/logs/training/runs.jsonl | sort -rn | head -10
```

Which files a traced run opened, deduplicated:

```bash
$ jq -r 'select(.event == "open") | .path' <event_file> | sort -u
```

The final training loss of a run, from its metrics file:

```bash
$ jq -r 'select(.loss != null) | "\(.step)\t\(.loss)"' <run_dir>/metrics.jsonl | tail -1
```

Combining the two files is a join on the run id — the catalog record's `run_dir` field names
the directory holding `metrics.jsonl`:

```bash
$ python -m training runs --json | jq -r '.[] | select(.exit_code != 0) | .run_dir'
```

## Related documentation

- [design-and-roadmap.md](design-and-roadmap.md) — why the system is shaped this way, and the
  OBS item ledger with each item's status
- [report_schema.md](report_schema.md) — the JSON report envelope, a separate contract that
  shares only the metrics header with this system
- [../rust-harness/observability.md](../rust-harness/observability.md) — the Rust harness as a
  second writer of this file layout
- [../../schemas/README.md](../../schemas/README.md) — the cross-language record contracts,
  including `obs_record.json`
- [../../common/README.md](../../common/README.md) — the `common/obs/` package itself
- [../verification/observability.md](../verification/observability.md) — the verification record
  for this stack
