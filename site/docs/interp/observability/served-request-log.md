---
title: Served-request log
---

# Served-request log

The served-request log is the one observability artifact another tier writes for interp. A
long-lived server has no end of run at which to report what its probe scored, so the harness,
serving with `--interp-log`, appends one JSONL record per completed `/v1/chat/completions`
request, and `probe-monitor` reads a window of those records. The scores in each record were
taken in the process that answered the request, on the weights that produced the text, so
there is no replay and no precision caveat to state.

## Who writes it

The harness writes the log; nothing in this tier does. Under `make harness-serve`, the two
environment variables in the table below turn it on, both empty by default so a served run
observes nothing unless interp asks:

| Variable | Effect |
|----------|--------|
| `HARNESS_INTERP_LOG` | Container path of the log root; absent means no log |
| `HARNESS_PROBE` | The exported probe, scored in-band at every position of its layer |

`HARNESS_INTERP_LOG` without `HARNESS_PROBE` records requests and no scores.
`HARNESS_PROBE` without `HARNESS_INTERP_LOG` computes scores and discards them, and the
entrypoint warns about it. The exported probe is the safetensors image `probe-export` writes;
the harness mounts the probes tree read-only at `/app/probes`, so the container spelling is
`/app/probes/<name>/probe.safetensors`. The flag itself, and what the harness checks before
loading a probe, are on the harness's
[Loading a probe for live monitoring](../../rust-harness/guides/probe-monitoring.md) page.

```bash
make probe-export ARGS="--probe /mnt/ai-data/interp/probes/<name>"
make harness-serve HARNESS_PROBE=/app/probes/<name>/probe.safetensors
```

## Location

One file per service per day, under the log root, so traffic from different endpoints never
shares a file and a server running past midnight rotates rather than appending to yesterday:

```text
/mnt/ai-data/interp/logs/harness/
  2026-09-12.jsonl
  2026-09-13.jsonl
```

Retention is 30 days, enforced by the monitor when it runs, keyed off the filename date.

## The record

One JSON object per completed request. The fields through `error` are the ones the monitor
has always read; `probe` is the field that carries the scores.

| Field | Meaning |
|-------|---------|
| `timestamp` | When the request completed |
| `target_service` | The service that answered, `harness` |
| `model` | The served checkpoint |
| `messages` | The request's messages, as sent |
| `response_text` | The reply, or null on failure |
| `usage` | Token counts, or null on failure |
| `latency_s` | Wall time for the request |
| `status_code` | The HTTP status returned |
| `error` | The failure, when the generation failed; the record is then partial |
| `probe` | What the probe did during the request, or null when no probe was loaded |

`probe` carries the probe's `name`, `layer`, `site`, and `threshold`; `scored`, the number of
positions scored; `position_first`, the lowest position the tap reported; `max_score`, the
highest score over the request's positions; and `flags`, one `{position, score}` per
over-threshold position. Positions are absolute in the session cache, and attribution to the
request comes from the drain being per request. `max_score` is the per-request statistic the
monitor's distribution is built from.

The record reaches disk per request, not at shutdown, so a server that is killed keeps every
record it wrote.

## Who reads it

`probe-monitor` aggregates a window of the log into a report. It loads no model and takes no
`--model`, so it needs no GPU and runs while inference is up:

```bash
make probe-monitor ARGS="--window 7d"
make probe-monitor ARGS="--window 7d --probes refusal-v1 --log-dir /mnt/ai-data/interp/logs/harness"
```

For each probe in the window it writes `report.json` under
`/mnt/ai-data/interp/monitor/<probe>/`, with the score distribution, the fraction of requests
over threshold, the skip counts, and the drift against the previous report over the same
probe set, and `flagged.jsonl` beside it holding the over-threshold records. Alerting is a
file, not a pager. Two things follow from the scores being taken in-band: the report records
how they were obtained in a `scored` field and refuses to compute a drift delta against a
report obtained another way, and a threshold that changed mid-window, because a probe was
re-exported while the service ran, is reported as a disagreement rather than averaged away.
The Probes guide places the monitor in the workflow.
