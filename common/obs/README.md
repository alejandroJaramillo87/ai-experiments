# common/obs

The observability package: structured logging keyed by an ambient run id, the traced
dispatch every `training` and `interp` command goes through, the run catalog, retention,
and the dashboard that renders a run's metrics. The Rust harness writes the same records
independently, so the shapes here are a contract, not an implementation detail.

`core.py` defines the run-event record and `catalog.py` the one-line-per-run catalog;
`rust-harness/obs/src/lib.rs` writes both without importing anything from here, and
`schemas/obs_record.json` and `schemas/run_metrics.json` are what the two sides are held to.
`rust-harness/harness/tests/obs_contract.rs` feeds a Rust-written catalog to `read_runs`
to prove they still agree, so a change to a record shape starts in `schemas/` and lands
on both sides in one commit. `dispatch.py` is the entry point: it wraps a CLI's `main` so
every invocation gets a run id, an event file and a catalog row, which is why neither
CLI logs by hand.

`dashboard.py` builds the snapshot page and `serve.py` the live one; both embed
`viewer.js` and the vendored chart library under [vendor/uplot/](vendor/uplot/README.md).
The browser half of that page cannot be tested from pytest, so `scripts/obs/` holds a
Chromium check run by hand after a change to any of the three.

## Modules

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `audit.py` | Layer 2 resource audit hook (roadmap OBS-6). |
| `catalog.py` | Run catalog: one summary line per completed run (roadmap OBS-9). |
| `context.py` | Run-context capture (roadmap OBS-7). |
| `core.py` | Structured logging core for the observability system (roadmap OBS-1). |
| `dashboard.py` | Snapshot dashboard builder (roadmap OBS-11). |
| `dispatch.py` | Traced CLI dispatch shared by the training and interp mains (roadmap OBS-4). |
| `retention.py` | Event-file retention for the observability log directory (roadmap OBS-8). |
| `runs.py` | Shared `runs` command surface for every CLI that writes the catalog (roadmap OBS-9). |
| `serve.py` | Live dashboard server (roadmap OBS-12). |

<!-- inventory:modules:end -->

## Where the real docs live

- [../../docs/observability/README.md](../../docs/observability/README.md) — operating the stack: where the files are, the `OBS_` variables, finding a run
- [../../docs/observability/design-and-roadmap.md](../../docs/observability/design-and-roadmap.md) — why it is shaped this way, and the OBS item ledger
- [../../schemas/README.md](../../schemas/README.md) — the record contracts both languages write
- [../../docs/rust-harness/observability.md](../../docs/rust-harness/observability.md) — the Rust half
