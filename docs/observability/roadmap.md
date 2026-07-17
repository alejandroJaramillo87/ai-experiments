# Runtime observability specification and implementation roadmap

This document specifies a runtime observability system for the training and
interpretability command-line interfaces (`python -m training`,
`python -m interp`) and sequences the work into ordered items that are
implemented one at a time. Each item is self-contained: it states its goal,
the files it touches, a concrete design, and an acceptance check, so a single
implementation session can complete one item without depending on the others
beyond the stated order.

The scope is a single AI workstation (RTX 5090, AMD Ryzen 9950X). The system
records what each command does, ties every record to one correlation id,
persists the records as machine-readable files, and renders per-run metrics
and logs in a local HTML page. It does not introduce servers, external
services, or cloud dependencies.

## Table of contents

- [Goals and non-goals](#goals-and-non-goals)
- [Design overview](#design-overview)
- [Output streams and the json contract](#output-streams-and-the-json-contract)
- [The event record schema](#the-event-record-schema)
- [The metrics record schema](#the-metrics-record-schema)
- [File and directory layout](#file-and-directory-layout)
- [Configuration](#configuration)
- [Implementation roadmap](#implementation-roadmap)
- [Sequencing and dependencies](#sequencing-and-dependencies)
- [Acceptance and testing conventions](#acceptance-and-testing-conventions)

## Goals and non-goals

### Goals

- Human-readable, timestamped logs across both CLIs, with errors recorded
  as structured failure records.
- Command-level capture (Layer 1): when a command runs, with what arguments,
  and how it exits.
- Resource-level capture (Layer 2): which files a command opens, which
  subprocesses it spawns, and which modules it imports, available on demand.
- One correlation id (`run_id`) stamped on every log record, metrics record,
  and run artifact, so a single run is reconstructable from its id.
- Per-run metrics emitted as JSON and rendered by a local, self-contained
  HTML page that shows the metrics and a log viewer for the same run.
- A documented procedure for call-level debugging with off-the-shelf
  profilers, rather than built-in call tracing.

### Non-goals

The following are deliberately out of scope. They belong to fleet or
production settings that do not apply to a single workstation, and adding
them would raise cost without adding local value.

- A metrics server or time-series database (Prometheus, Grafana, or similar).
- OpenTelemetry, span-based tracing, or any distributed-tracing backend.
- A log shipper or aggregator (Fluentd, Vector, or an ELK stack).
- A DuckDB dependency or any built-in query layer. The event files use JSON
  Lines, so ad hoc analysis with `jq` or DuckDB remains possible without the
  repository depending on either.
- Always-on profiling.
- Authentication, multi-user access, or a live-streaming server for the
  dashboard.
- Cross-process propagation of the correlation id in the first version. The
  id stays in-process; propagation to subprocesses and services is a later
  refinement (see [Sequencing and dependencies](#sequencing-and-dependencies)).

## Design overview

Observability is one emit path, keyed by `run_id`, fanned out to three
renderings of the same events:

| Rendering | Destination | Purpose |
|-----------|-------------|---------|
| Human status line | Standard error | The live view while a command runs in a terminal |
| Structured record | A JSON Lines file on disk | The durable record, read by tools and the dashboard |
| Metrics series | A per-run JSON Lines file | The data the dashboard plots |

A single logging call produces the human line and the structured record. The
metrics series is a separate, per-step emission written by the training and
interpretability loops.

Two decisions shape the rest of the system:

1. The existing `Printer` (in `/common/output.py`) folds into the logging
   core. `Printer` methods route through the logger, so one `out.error(...)`
   call produces both the human `[ERROR]` line and a structured error record.
   The 443 existing call sites keep their current API.
2. The correlation id is generated once at command start and held in an
   ambient context, so every downstream record carries it without threading
   it through function signatures.

## Output streams and the json contract

The single load-bearing rule of the system:

- **Standard output is reserved for `--json` payloads.** In `--json` mode a
  command prints only its result object to standard output, so a caller can
  parse it. Logs and human status never write to standard output.
- **Standard error carries all human status and logs.** In the default
  (non-`--json`) mode, human status lines go to standard error. In `--json`
  mode the human lines are suppressed, but the structured file record is
  still written.
- **The file is always written**, in both modes, so `--json` runs retain a
  durable log even though their console is quiet.

This is a change from current behavior, where `Printer` writes human status
to standard output. The change is required so that standard output holds only
machine output. Tests that assert `Printer` output on captured standard
output move to captured standard error as part of the fold (item OBS-3).

## The event record schema

Every record in an event file is one JSON object on one line. All records
carry a common base; each event type adds fields.

Base fields, present on every record:

| Field | Type | Description |
|-------|------|-------------|
| `ts` | string | ISO 8601 timestamp with timezone offset, millisecond precision |
| `level` | string | One of `debug`, `info`, `warning`, `error` |
| `run_id` | string | The correlation id for the run |
| `cli` | string | `training` or `interp` |
| `command` | string | The subcommand name, for example `train-sft` |
| `pid` | integer | Process id |
| `event` | string | Machine-readable event name |
| `msg` | string | Human-readable message; may be empty for pure-data events |

Event types and their added fields:

| `event` | Added fields | Level | Emitted by |
|---------|--------------|-------|------------|
| `cmd.start` | `argv` (redacted list) | info | Dispatch wrapper (OBS-4) |
| `cmd.end` | `exit_code`, `duration_s` | info | Dispatch wrapper (OBS-4) |
| `phase.start` | `phase` | info | Phase timer (OBS-4) |
| `phase.end` | `phase`, `duration_s` | info | Phase timer (OBS-4) |
| `fail` | `exc_type`, `exc_msg` (redacted), `traceback` (redacted, truncated), `exit_code` | error | Failure recorder (OBS-5) |
| `open` | `path`, `mode` | debug | Audit hook (OBS-6) |
| `import` | `module` | debug | Audit hook (OBS-6) |
| `spawn` | `argv0`, `argv` (redacted) | debug | Audit hook (OBS-6) |

Application code emits its own named events (for example `dataset_loaded`
with an `examples` count) through the logging API. Those records carry the
base fields plus whatever the call passes.

## The metrics record schema

Metrics live in their own per-run file, separate from events, because they
are a dense per-step series rather than discrete log records. The file is
JSON Lines. Its first record is a header that conforms to the shared report
envelope (`/common/reports.py`, `report_header`), so existing report tooling
validates it.

Header record (built with `report_header("run_metrics")`, then extended):

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | integer | From `report_header` |
| `report` | string | `run_metrics` |
| `timestamp` | string | From `report_header` |
| `git_sha` | string | From `report_header` |
| `run_id` | string | The correlation id |
| `cli` | string | `training` or `interp` |
| `command` | string | The subcommand name |
| `tier` | string | For example `sft`, `distill`, `sae` |
| `model_id` | string | The model identity |
| `recipe_hash` | string | Hash of the resolved recipe |
| `schedule` | object | Schedule summary, for example WSD phase boundaries |

Per-step records:

| Field | Type | Description |
|-------|------|-------------|
| `step` | integer | Global step |
| `ts` | string | ISO 8601 timestamp |
| `train_loss` | number | Training loss |
| `holdout_loss` | number or null | Held-out loss when evaluated at this step |
| `lr` | number | Learning rate |
| `grad_norm` | number | Gradient norm |
| `tokens_seen` | integer | Cumulative tokens processed |
| `tok_per_s` | number | Throughput in tokens per second |
| `vram_alloc_bytes` | integer | Allocated device memory |
| `vram_reserved_bytes` | integer | Reserved device memory |

Marker records, interleaved with per-step records, reuse the `event` field:

| `event` | Added fields | Meaning |
|---------|--------------|---------|
| `checkpoint` | `step`, `path` | A checkpoint was written |
| `watchdog` | `signal`, `value`, `threshold` | A watchdog signal crossed its threshold |
| `eval` | `step`, `metric`, `value` | An evaluation result was recorded |

## File and directory layout

- Event files: `${AIX_LOG_DIR}/<cli>/<run_id>.jsonl`, where `AIX_LOG_DIR`
  defaults to `/mnt/ai-data/logs`. One file per run, named by the
  correlation id.
- Metrics files: `<run_dir>/metrics.jsonl`, co-located with the run
  directory the training loop already creates
  (`${run.output_root}/${run.name}/<timestamp>/`, from
  `/training/recipes/loader.py`, `create_run_dir`). The run header record
  carries the `run_id`, which links the metrics file to the event file.
- Dashboard output: `<run_dir>/run.html`, written by the `report`
  subcommand (OBS-11).

The correlation id is the join key across all three. The event file records
its `run_dir` in the `cmd.start` record when a run directory exists, so the
dashboard can locate the metrics file from the event file and the reverse.

## Configuration

Configuration is environment variables in the first version; command-line
flags are a later refinement noted per item. The `AIX_` prefix is a proposed
convention for this repository; confirm it before implementation, since the
codebase currently uses unprefixed names (`HF_TOKEN`, `LLAMA_CPP_DIR`,
`TRAIN_LOCK_PATH`).

| Variable | Default | Effect |
|----------|---------|--------|
| `AIX_LOG_LEVEL` | `info` | Minimum level written to the console and file |
| `AIX_LOG_DIR` | `/mnt/ai-data/logs` | Root of the event files |
| `AIX_TRACE` | `0` | When `1`, installs the Layer 2 audit hook (OBS-6) |
| `AIX_LOG_RETENTION_DAYS` | `30` | Age past which event files are pruned (OBS-8) |
| `NO_COLOR` | unset | Existing variable; disables color, already honored by `Printer` |

Note: the codebase references `MLFLOW_TRACKING_URI` in at least one place.
Before implementation, confirm that reference is not load-bearing. This
system does not adopt MLflow; it stays with local JSON Lines files.

## Implementation roadmap

The items are grouped into five phases. Implement them in order within a
phase; phases also proceed in order, because later phases consume the
interfaces earlier phases define. Every item ends with the mandatory quality
gate and its own acceptance check (see
[Acceptance and testing conventions](#acceptance-and-testing-conventions)).

The observability package is `common/obs/`, a new subpackage of the existing
`common/` package. It grows one module per item. Do not name any module
`logging.py`, to avoid shadowing the standard library.

### Phase A: logging foundation

#### OBS-1: logging core

- **Goal**: a structured logger that writes a human line to standard error
  and a JSON Lines record to a file, both carrying the ambient correlation id
  and command context.
- **Depends on**: nothing.
- **Scope**: `common/obs/__init__.py`, `common/obs/core.py`,
  `tests/common/test_obs_core.py`.
- **Design**:
  - `core.py` defines a `StructuredLogger` wrapper over the standard library
    `logging` module. Its methods are `debug`, `info`, `warning`, `error`,
    and `event(name, **fields)`, each accepting a message and arbitrary
    keyword fields.
  - Ambient context (`run_id`, `cli`, `command`, `pid`) lives in a
    `contextvars.ContextVar`, set by `bind_run(cli, command, run_id)` and
    read by every record. `current_run_id()` returns the active id.
  - `new_run_id()` returns an id of the form `R-YYYYMMDD-HHMMSS-xxxxxx`,
    where `xxxxxx` is six hexadecimal characters from `secrets.token_hex(3)`.
  - `configure(cli, *, run_id=None, log_dir=None, level=None)` installs two
    handlers on the package logger: a stream handler on standard error with a
    human formatter (`HH:MM:SS.mmm [LEVEL] message`), and a file handler on
    `${AIX_LOG_DIR}/<cli>/<run_id>.jsonl` with a JSON formatter that emits the
    base schema. Level comes from `AIX_LOG_LEVEL` unless overridden.
  - `get_logger(name)` returns a `StructuredLogger` bound to `name`.
  - The JSON formatter writes exactly one object per line and never raises on
    a non-serializable field; it coerces unknown types with `repr`.
- **Acceptance check** (sandbox): `uv run pytest tests/common/test_obs_core.py`
  passes, asserting that one `info` call writes one parseable JSON line with
  the base fields, that the human line appears on standard error and not
  standard output, that `AIX_LOG_LEVEL=warning` suppresses `info`, and that
  the record carries the bound `run_id`.
- **Out of scope**: redaction, the `Printer` fold, any dispatch changes.

#### OBS-2: secret redaction filter

- **Goal**: a single redaction function that removes secrets from any value
  before it reaches a file, applied to argument vectors, environment
  captures, exception messages, and tracebacks.
- **Depends on**: OBS-1.
- **Scope**: `common/obs/redaction.py`, wiring into `common/obs/core.py`,
  `tests/common/test_obs_redaction.py`.
- **Design**:
  - `redact(value)` accepts a string, list, or dict and returns the same
    shape with secrets masked. It masks values of keys matching
    `*_TOKEN`, `*_KEY`, `*_SECRET`, and `PASSWORD` (case-insensitive), and
    it masks token-shaped substrings (for example a leading `hf_` followed by
    alphanumerics, and long hexadecimal or base64 runs).
  - The masked form is a fixed marker, for example `***redacted***`, never a
    partial reveal.
  - The core JSON formatter passes every field value through `redact` before
    serialization, so redaction is centralized rather than per-call-site.
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_redaction.py` passes, asserting that a
  planted `HF_TOKEN` value, an `hf_`-prefixed argument, and a long hex string
  are masked, and that ordinary fields pass through unchanged.
- **Out of scope**: deciding which commands log arguments; that is OBS-4.

#### OBS-3: fold Printer into the logger

- **Goal**: `Printer` becomes a front-end over the logging core, so status
  calls produce both the human line and a structured record, and human output
  moves from standard output to standard error.
- **Depends on**: OBS-1.
- **Scope**: `common/output.py`, any tests that capture `Printer` output.
- **Design**:
  - Each `Printer` method (`ok`, `warn`, `error`, `info`, `run`, `skip`,
    `line`) calls the logging core with a level mapping: `ok`, `info`, `run`,
    `skip`, and `line` map to `info`; `warn` maps to `warning`; `error` maps
    to `error`. The status tag (`OK`, `RUN`, and so on) is preserved on the
    file record in a `status` field so the original vocabulary survives.
  - The human rendering keeps the `[OK]`, `[WARN]`, and other tags and the
    existing color and indentation rules, but writes to standard error.
  - `quiet=True` (the `--json` path) suppresses the human line only; the
    structured record is still written.
  - The public method signatures do not change, so the 443 call sites are
    untouched.
- **Acceptance check** (sandbox): the existing suite passes after moving
  affected assertions to standard error
  (`uv run pytest tests/`), plus a new test that a `Printer(quiet=True)`
  call writes no console output but does write a file record.
- **Out of scope**: adding timestamps to the `run`-subcommand output beyond
  what the formatter already provides.

### Phase B: event capture

#### OBS-4: Layer 1 command events and traced dispatch

- **Goal**: every invocation of either CLI generates a `run_id`, a
  `cmd.start` record, and a `cmd.end` record with exit code and duration, and
  exposes a phase timer for coarse in-command spans.
- **Depends on**: OBS-1, OBS-2, OBS-3.
- **Scope**: `training/cli.py` and `interp/cli.py` (`main`),
  `common/obs/core.py` (phase timer), `tests/common/test_obs_dispatch.py`.
- **Design**:
  - Both CLIs share the dispatch shape
    `args = build_parser().parse_args(argv); return args.func(args)`. Wrap the
    call: generate a `run_id`, `configure(cli, run_id=...)`,
    `bind_run(...)`, emit `cmd.start` with the redacted `argv`, run the
    command inside a `try/finally`, and emit `cmd.end` with the exit code and
    wall-clock duration in the `finally` so it fires on exceptions too.
  - Provide `phase(name)` as a context manager that emits `phase.start` and
    `phase.end` with the elapsed duration. It is used by later items and by
    long commands.
  - The `argv` list is redacted through OBS-2 before it is logged.
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_dispatch.py` passes, asserting that a
  successful command writes `cmd.start` and `cmd.end` with `exit_code` 0 and a
  non-negative `duration_s`, that a command raising an exception still writes
  `cmd.end` with a non-zero exit code, and that both records share one
  `run_id`.
- **Out of scope**: the structured `fail` record, which is OBS-5.

#### OBS-5: structured failure records

- **Goal**: an uncaught exception in a command is recorded as a `fail` event
  with the exception type, redacted message, and a truncated redacted
  traceback, in addition to the human error line.
- **Depends on**: OBS-1, OBS-2, OBS-4.
- **Scope**: `common/obs/core.py` (failure recorder), the dispatch wrapper in
  `training/cli.py` and `interp/cli.py`, `tests/common/test_obs_failure.py`.
- **Design**:
  - Add `record_failure(exc, exit_code)` that emits a `fail` event with
    `exc_type`, `exc_msg`, a traceback truncated to a bounded number of
    frames, and the exit code. The message and traceback pass through
    redaction.
  - The dispatch wrapper calls `record_failure` when a command raises or
    returns a non-zero exit code from a known error path, so failures are
    captured uniformly rather than per command.
  - The human error line continues to come from `Printer.error`, unchanged.
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_failure.py` passes, asserting that a
  raised exception produces one `fail` record with the correct `exc_type`,
  a redacted message, and a bounded traceback, and that the record shares the
  run's `run_id`.
- **Out of scope**: retry logic or error classification beyond recording.

#### OBS-6: Layer 2 resource audit hook

- **Goal**: when enabled, record every file opened, module imported, and
  subprocess spawned during a command, using a Python audit hook, at debug
  level.
- **Depends on**: OBS-1, OBS-2, OBS-4.
- **Scope**: `common/obs/audit.py`, wiring into the dispatch wrapper,
  `tests/common/test_obs_audit.py`.
- **Design**:
  - `install_audit_trace()` registers a `sys.addaudithook` callback that maps
    the `open`, `import`, and `subprocess.Popen` (and `os.exec*`) audit events
    to `open`, `import`, and `spawn` records at debug level.
  - The hook is installed only when `AIX_TRACE=1`, because audit hooks add
    measurable overhead. The dispatch wrapper reads the variable and installs
    the hook before running the command.
  - The hook never raises into the audited call; it swallows its own errors so
    an instrumentation failure cannot break the command.
  - Spawn arguments pass through redaction.
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_audit.py` passes with `AIX_TRACE=1`,
  asserting that opening a file, importing a module, and spawning a trivial
  subprocess each produce the corresponding debug record, and that with
  `AIX_TRACE=0` no such records appear.
- **Out of scope**: call-level or function-level tracing, which is a
  profiler task documented in OBS-13.

#### OBS-7: run-context capture at start

- **Goal**: each run records the environment it ran in, so a failure is
  diagnosable and reproducible.
- **Depends on**: OBS-1, OBS-4.
- **Scope**: `common/obs/context.py`, wiring into the dispatch wrapper,
  `tests/common/test_obs_context.py`.
- **Design**:
  - `run_context()` returns a dict with the git SHA (reuse
    `common/provenance.git_sha`), the Python version, the platform string,
    and the versions of a small pinned set of packages relevant to the tier
    (for example `torch`, `transformers`, `trl`). Device and driver
    information is included when a query succeeds and omitted when it does
    not, so the function never fails on a machine without a GPU.
  - The dispatch wrapper emits the context as fields on the `cmd.start`
    record, or as a separate `run.context` record.
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_context.py` passes, asserting that the
  context contains a git SHA and a Python version and that a missing GPU
  produces an omitted field rather than an exception.
- **Out of scope**: capturing service health; that requires running services
  and is validated on the workstation.

### Phase C: operability

#### OBS-8: retention and rotation

- **Goal**: bound the disk that event files consume, so the log directory
  does not grow without limit.
- **Depends on**: OBS-1.
- **Scope**: `common/obs/retention.py`, wiring into `configure`,
  `tests/common/test_obs_retention.py`.
- **Design**:
  - `prune(log_dir, retention_days)` removes event files older than the
    retention window. `configure` calls it opportunistically at startup, so
    pruning needs no separate scheduler.
  - The window comes from `AIX_LOG_RETENTION_DAYS`, default 30.
  - Pruning is best-effort: a file that cannot be removed is skipped, not
    fatal.
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_retention.py` passes, asserting that a
  file older than the window is removed, that a recent file is kept, and that
  an unremovable file does not raise.
- **Out of scope**: size-based rotation within a single run; per-run files
  are bounded by the run itself.

#### OBS-9: run catalog and runs subcommand

- **Goal**: make runs discoverable, so a past run is found by id, time, or
  outcome without reading individual files.
- **Depends on**: OBS-4.
- **Scope**: `common/obs/catalog.py`, a `runs` subcommand in
  `training/cli.py` and `interp/cli.py`, `tests/common/test_obs_catalog.py`.
- **Design**:
  - On `cmd.end`, append one summary record to
    `${AIX_LOG_DIR}/<cli>/runs.jsonl`: `run_id`, `command`, `timestamp`,
    `duration_s`, `exit_code`, and `run_dir` when present.
  - The `runs` subcommand reads the catalog and prints a table of recent
    runs, honoring `--json` for machine output and a `--failed` filter.
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_catalog.py` passes, asserting that a
  completed run appends one catalog record and that `runs --json` returns the
  recorded runs including a failed one.
- **Out of scope**: cross-run statistics; that is ad hoc `jq` or DuckDB over
  the catalog, not a built-in feature.

### Phase D: metrics and dashboard

#### OBS-10: metrics emission

- **Goal**: the training loop, and then the interpretability loops, write a
  per-run `metrics.jsonl` conforming to the metrics schema.
- **Depends on**: OBS-1, OBS-4.
- **Scope**: `common/obs/metrics.py`, wiring into `training/runtime/loop.py`
  and the SFT and distillation loops, then the interpretability training
  loops, `tests/common/test_obs_metrics.py`.
- **Design**:
  - `MetricsWriter(run_dir, header)` writes the header record first, then one
    per-step record per logged step, then marker records for checkpoints,
    watchdog signals, and evaluations. It reuses `report_header("run_metrics")`
    for the header so `validate_report` accepts it.
  - The loops already compute loss, learning rate, tokens, and device memory;
    the writer consumes those values rather than recomputing them.
  - Interpretability loops adopt the same writer with their own `tier`, so
    the metrics story is not training-only.
- **Acceptance check** (sandbox for the unit; host for the real series):
  `uv run pytest tests/common/test_obs_metrics.py` passes, asserting that a
  synthetic sequence of steps and markers produces a header that
  `validate_report` accepts and one record per step. On the workstation, a
  short `train-sft` run writes a `metrics.jsonl` whose step count matches the
  run; record the command and output in the verification runbook.
- **Out of scope**: the rendering, which is OBS-11.

#### OBS-11: dashboard generator

- **Goal**: a `report` subcommand renders a run's metrics and logs into one
  self-contained HTML file with no external dependencies.
- **Depends on**: OBS-10.
- **Scope**: `training/commands/report.py` (and its registration), a template
  under `common/obs/`, `tests/training/test_report.py`.
- **Design**:
  - `python -m training report --run <run_dir>` reads `metrics.jsonl` from the
    run directory and the event file for the run's `run_id`, then writes
    `<run_dir>/run.html`.
  - The page inlines the data as a script block and draws the loss curves, the
    learning-rate schedule, throughput, and device memory, with checkpoint and
    watchdog markers on the time axis. Below the charts it renders a log viewer
    that lists the run's records with level filtering and text search.
  - The chart layer is a single vendored library chosen for the least code
    that draws interactive line charts; raw d3 is used only if a view requires
    it. No content is fetched from a network at render time or view time.
- **Acceptance check** (sandbox):
  `uv run pytest tests/training/test_report.py` passes, asserting that from a
  fixture `metrics.jsonl` and event file the command writes a single HTML file
  that contains the inlined data and references no external URL. Opening the
  file in a browser on the workstation shows the charts and the log viewer.
- **Out of scope**: multi-run comparison and live refresh.

#### OBS-12: live-refresh dashboard variant (optional)

- **Goal**: view a running job's dashboard as it progresses, without a
  backend service.
- **Depends on**: OBS-11.
- **Scope**: an option on the `report` subcommand, documentation.
- **Design**:
  - `report --run <run_dir> --serve` starts `python -m http.server` in the run
    directory and writes a variant of the page that re-fetches `metrics.jsonl`
    on an interval, so the charts extend as the file grows.
  - This is the only place the system runs a server, and it runs the standard
    library server locally for the duration of viewing.
- **Acceptance check** (host): on the workstation, starting the server during
  a run shows the charts extending as steps accrue; record the procedure in
  the observability reference doc.
- **Out of scope**: everything the non-goals exclude; this remains a local,
  optional convenience. Implement it only if the snapshot page proves
  insufficient.

### Phase E: documentation

#### OBS-13: debugging with profilers guide

- **Goal**: document how to answer "which function or file called which" with
  off-the-shelf profilers, since the system does not build call tracing.
- **Depends on**: nothing; it can be written at any time.
- **Scope**: `docs/observability/debugging-runs.md`.
- **Design**: document, with exact commands, the on-demand tools: `viztracer`
  with `vizviewer` for an interactive timeline, `cProfile` with `snakeviz` for
  a call tree, `py-spy` for sampling a running process, and `pydeps` or
  `code2flow` for a static import or call graph. State that each runs through
  `uv run --with <tool>` and adds no repository dependency, and that the audit
  hook (OBS-6) is the file-open-grained alternative for forensic runs.
- **Acceptance check** (sandbox): the doc lists each tool with a runnable
  command and conforms to `.styles/DOCUMENTATION_STYLE.md`.
- **Out of scope**: building any of these into the CLIs.

#### OBS-14: observability reference doc

- **Goal**: a reference that explains, to an operator, how to read logs, find
  runs, and open a dashboard.
- **Depends on**: the items whose behavior it documents.
- **Scope**: `docs/observability/README.md`.
- **Design**: document the file layout, the environment variables, the event
  and metrics schemas, the `runs` and `report` subcommands, and example `jq`
  queries over the event files and the catalog. Cross-reference this roadmap
  and `docs/report_schema.md`.
- **Acceptance check** (sandbox): the doc conforms to
  `.styles/DOCUMENTATION_STYLE.md` and every command in it is runnable as
  written.
- **Out of scope**: none; this is the closing item.

## Sequencing and dependencies

The critical path is OBS-1, then OBS-3 and OBS-4, then the rest. The
dependency summary:

| Item | Depends on |
|------|------------|
| OBS-1 logging core | none |
| OBS-2 redaction | OBS-1 |
| OBS-3 Printer fold | OBS-1 |
| OBS-4 command events | OBS-1, OBS-2, OBS-3 |
| OBS-5 failure records | OBS-1, OBS-2, OBS-4 |
| OBS-6 audit hook | OBS-1, OBS-2, OBS-4 |
| OBS-7 run context | OBS-1, OBS-4 |
| OBS-8 retention | OBS-1 |
| OBS-9 run catalog | OBS-4 |
| OBS-10 metrics | OBS-1, OBS-4 |
| OBS-11 dashboard | OBS-10 |
| OBS-12 live variant | OBS-11 |
| OBS-13 profiler guide | none |
| OBS-14 reference doc | the items it documents |

OBS-13 has no code dependency and can be written first if a debugging guide
is needed before the system lands.

A later, separate effort extends the correlation id across process
boundaries: exporting `AIX_RUN_ID` into subprocess environments so children
stamp their own records, and passing the id as a request header to the
inference services. This is explicitly deferred from the first version and is
not required by any item above.

## Acceptance and testing conventions

- Every item ends by running `make quality` and its own acceptance check.
  `make quality` runs the ruff format check, ruff lint, mypy, the uv lockfile
  check, and shellcheck. The lockfile step fails in the sandbox by the
  documented caveat (the pytorch index is unreachable); the other four steps
  must pass before any push.
- Sandbox-runnable acceptance checks are unit tests that need no GPU, no
  Docker, and no running service. Most items above are sandbox-runnable in
  full.
- Host-side acceptance checks (a real `metrics.jsonl` from a training run in
  OBS-10, the live server in OBS-12, opening a rendered page) run on the
  workstation. The operator runs the command and records the command and its
  output; the agent does not claim to have run a host-side command.
- Each item is one change. Do not combine an item with an unrelated change,
  in keeping with the repository's one-change-at-a-time rule.
