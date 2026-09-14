---
title: Dashboard
---

# Dashboard

A dashboard is one run's metrics and events rendered into a single self-contained HTML file,
or served live while the run is still going. It reads the run directory a verb recorded, the
directory holding `metrics.jsonl`, and locates the event file from the run id in the metrics
header.

## The verb lives on the training CLI

The renderer is the `report` verb of `python -m training`, not of `python -m interp`. The
two CLIs share the observability machinery, and a run directory has the same shape whichever
tier wrote it, so the training verb opens an interp run without knowing which tier it came
from. This page says so rather than pretending the interp CLI has the verb.

```bash
uv run python -m training report --run <run_dir>
```

That writes `run.html` into the run directory, referencing no external URL. The run directory
comes from the catalog: `make interp-runs ARGS='--json'` prints the records and each carries
`run_dir`. A verb that created no run directory has nothing to render.

| Flag | Effect | Default |
|------|--------|---------|
| `--run <dir>` | The run directory holding `metrics.jsonl` | required |
| `--events <path>` | The event file, when it is not where the header's run id says | located from the header |
| `--serve` | Serve a live page that polls the growing files instead of writing a snapshot | off |
| `--host <ip>` | Bind address for `--serve` | `127.0.0.1` |
| `--port <n>` | Port for `--serve` | `8010` |

## Live view

```bash
uv run python -m training report --run <run_dir> --serve
uv run python -m training report --run <run_dir> --serve --port 8010
```

The server is a stateless viewer: it never writes the files it reads, and stopping it loses
nothing. It binds loopback by default. **Warning!** `--host 0.0.0.0` exposes the page to the
local network, and there is no authentication.
