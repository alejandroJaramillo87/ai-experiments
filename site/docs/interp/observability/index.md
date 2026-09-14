---
title: Observability
---

# Observability

Every verb is traced by the same machinery the training tier uses, and the harness writes one
artifact for interp's sake. Four mechanisms, at four grains, each with its own page:

![Four observability mechanisms ordered by grain: run records (one run, always on), the served-request log (one request, enabled by the harness --interp-log flag), the resource audit (one syscall, enabled by OBS_TRACE=1), and the dashboard (one metrics series, rendered by the training CLI report verb). The run id joins run records to the dashboard.](../../../static/img/diagrams/interp-observability-grains.svg)

| Mechanism | Grain | Always on | Where it goes |
|-----------|-------|-----------|---------------|
| [Run records](run-records.md) | One run: start, phases, failure, end, and the catalog line | Yes | `${OBS_LOG_DIR}/interp/` |
| [Served-request log](served-request-log.md) | One request the harness answered, with the probe scores taken while answering | When the harness serves with `--interp-log` | `/mnt/ai-data/interp/logs/<service>/` |
| [Resource audit](resource-audit.md) | One syscall: every file opened, module imported, and process spawned | With `OBS_TRACE=1` | The run's event file, at debug level |
| [Dashboard](dashboard.md) | One metrics series, rendered from a run directory | On demand, from the training CLI | `<run_dir>/run.html` |

The run records are the durable artifact and are written whether or not anyone asked; a verb
that exits with a usage error leaves nothing, because it exits before a run id exists. The
audit is opt-in because it records at debug level and grows the event file with every file
touched. The served-request log is the one mechanism another tier writes: the harness appends
one record per completed request, and interp's `probe-monitor` is its reader. The dashboard
is a viewer over a run directory, and the verb that renders it belongs to the training CLI,
which opens an interp run directory the same way.

The run id minted for the run records is the join key: the catalog line carries the run
directory, a metrics file carries the run id in its header, and the event file is named by
it. The served-request log is keyed differently, by service and by day, because a server has
no end of run.

The field-level contract for the run records is `schemas/obs_record.json` in the repository,
shared with the harness and the training tier; the pages here restate it in words.
