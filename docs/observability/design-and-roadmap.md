# Runtime observability: design and roadmap

Why the runtime observability system for the training and interpretability
command-line interfaces (`python -m training`, `python -m interp`) is shaped
the way it is, and the ordered ledger of items that built it. Each item is
self-contained: it states its goal, its status, the files it touches, a
concrete design, and an acceptance check, so a single implementation session
(for example `/task docs/observability/design-and-roadmap.md OBS-4`) can
complete one item without depending on the others beyond the stated order.

**This document does not describe how to use the system.** The record
schemas, the file layout, the `OBS_` variables, and the operator commands
live in [README.md](README.md), which graduated out of this document once
the items that defined them shipped. What remains here is the reasoning and
the work ledger.

The scope is a single AI workstation (RTX 5090, AMD Ryzen 9950X). The system
records what each command does, ties every record to one correlation id,
persists the records as machine-readable files, and renders per-run metrics
and logs in a local HTML page, with an optional live view of a running job.
It introduces no external services or cloud dependencies; the only servers it
may run are local, stateless viewers over the files (see
[the stateless viewer principle](#the-stateless-viewer-principle)).

## Table of contents

- [Goals and non-goals](#goals-and-non-goals)
- [Design overview](#design-overview)
- [Implementation roadmap](#implementation-roadmap)
- [Sequencing and dependencies](#sequencing-and-dependencies)
- [Acceptance and testing conventions](#acceptance-and-testing-conventions)

## Goals and non-goals

### Goals

- Human-readable logs across both CLIs, with errors recorded as structured
  failure records.
- Command-level capture (Layer 1): when a command runs, with what arguments,
  and how it exits.
- Resource-level capture (Layer 2): which files a command opens, which
  subprocesses it spawns, and which modules it imports, available on demand.
- One correlation id (`run_id`) stamped on every log record, metrics record,
  and run artifact, so a single run is reconstructable from its id.
- Per-run metrics emitted as JSON Lines through the existing tracker fan-out
  (`training/runtime/tracking.py`) and rendered by a local, self-contained
  HTML page that shows the metrics and a log viewer for the same run.
- A live view of a running job — charts extending and logs tailing in a
  browser — served by an ephemeral, stateless local server.
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
- Authentication or multi-user access. Viewers bind to localhost unless the
  operator passes an explicit host flag.
- Any masking or redaction of record content. No record type captures
  environment variables, and record content is written as passed. The audited
  surface justifies this: no argparse flag in either CLI accepts a secret, and
  `HF_TOKEN` is consumed ambiently inside `huggingface_hub`, so its value never
  enters Python-level state.
- An always-on service as a requirement. The optional rig console (OBS-15)
  may run under compose, but nothing in the system requires a server to be
  running for records to be written or read.
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
metrics series is written by a JSON Lines sink inside the existing tracker
fan-out, so the loops that already call `tracker.log(...)` need no changes to
emit it. The dashboard pages (OBS-11, OBS-12, OBS-15) are consumers of these
files, never a fourth write path.

Two decisions shape the rest of the system:

1. The existing `Printer` (in `common/output.py`) folds into the logging
   core. `Printer` methods route through the logger, so one `out.error(...)`
   call produces both the human `[ERROR]` line and a structured error record.
   The 508 existing call sites (approximately 495 in production code) keep
   their current API.
2. The correlation id is generated once at command start and held in an
   ambient context, so every downstream record carries it without threading
   it through function signatures.

### What tracing means in this system

The word tracing usually suggests OpenTelemetry-style distributed spans.
This system deliberately does not build that. Tracing here means two layers:

- **Built in**: command events and phase spans (OBS-4) answer "what did this
  run do and how long did each phase take", and the resource audit hook
  (OBS-6) answers "which files, imports, and subprocesses did it touch".
  Both are always available, keyed by `run_id`.
- **Documented, not built**: function-level timelines and call trees come
  from off-the-shelf profilers run on demand, with exact commands documented
  in OBS-13. The repository gains no profiling dependency.

### The stateless viewer principle

The JSON Lines files are the sole source of truth. Any server this system
runs is a stateless viewer over those files: it never writes them, holds no
state beyond file read offsets, and killing it loses nothing. Under this
principle, whether a viewer runs ephemerally (`report --serve`, OBS-12) or
as a long-lived compose service (the rig console, OBS-15) is a deployment
choice, not an architecture fork — both serve the same pages from the same
files, and the always-on decision can be made after living with the
ephemeral one.

## Implementation roadmap

The items are grouped into five phases. Implement them in order within a
phase; phases also proceed in order, because later phases consume the
interfaces earlier phases define. Every item ends with the mandatory quality
gate and its own acceptance check (see
[Acceptance and testing conventions](#acceptance-and-testing-conventions)).

Every item carries a **Status** field naming the module and the tests that
implement it, so the state of the campaign is readable from the item rather
than inferred from the tree. The current state:

| State | Items |
|-------|-------|
| Done | OBS-1, OBS-3 through OBS-12, OBS-14 |
| Retired | OBS-2 |
| Open | OBS-13 |
| Not started, optional | OBS-15 |

The observability package is `common/obs/`, a subpackage of the existing
`common/` package. It grows one module per item. Do not name any module
`logging.py`, to avoid shadowing the standard library. Unit tests for
`common/obs/` modules live under `tests/common/`, the directory mirroring
the `common/` package, where the existing `common/` tests live
(`tests/common/test_common_output.py`).

### Phase A: logging foundation

#### OBS-1: logging core

- **Goal**: a structured logger that writes a human line to standard error
  and a JSON Lines record to a file, both carrying the ambient correlation id
  and command context.
- **Status**: done. `common/obs/core.py`; tests in `tests/common/test_obs_core.py`.
- **Depends on**: nothing.
- **Scope**: `common/obs/__init__.py`, `common/obs/core.py`,
  `tests/common/test_obs_core.py`.
- **Design**:
  - `core.py` defines a `StructuredLogger` wrapper over the standard library
    `logging` module (currently unused by first-party code, so there are no
    configuration conflicts). Its methods are `debug`, `info`, `warning`,
    `error`, and `event(name, **fields)`, each accepting a message and
    arbitrary keyword fields.
  - Ambient context (`run_id`, `cli`, `command`, `pid`) lives in a
    `contextvars.ContextVar`, set by `bind_run(cli, command, run_id)` and
    read by every record. `current_run_id()` returns the active id.
    `set_run_dir(path)` stashes the run directory in the same context and
    emits a `run.dir` event; `current_run_dir()` returns it or `None`.
  - `new_run_id()` returns an id of the form `R-YYYYMMDD-HHMMSS-xxxxxx`,
    where `xxxxxx` is six hexadecimal characters from `secrets.token_hex(3)`.
  - `configure(cli, *, run_id=None, log_dir=None, level=None)` installs two
    handlers on the package logger: a stream handler on standard error with a
    human formatter (`[LEVEL] message` — no timestamp, matching the `Printer`
    vocabulary; levels render as `DEBUG`, `INFO`, `WARN`, `ERROR`), and a
    file handler on `${OBS_LOG_DIR}/<cli>/<run_id>.jsonl` with a JSON
    formatter that emits the base schema. Level comes from `OBS_LOG_LEVEL`
    unless overridden.
  - `get_logger(name)` returns a `StructuredLogger` bound to `name`.
  - The JSON formatter writes exactly one object per line and never raises on
    a non-serializable field; it coerces unknown values with `repr`, and when
    serialization still fails (non-string keys, cycles, a value whose own
    `repr` raises) it falls back to the base schema plus a `fields_repr`
    string. Base-schema keys always win over a caller field of the same name.
  - Handler write failures are silent: both installed handlers suppress
    `handleError`, so mid-run ENOSPC cannot flood standard error with logging
    tracebacks, and reconfiguration tolerates a broken previous stream.
- **Acceptance check** (sandbox): `uv run pytest tests/common/test_obs_core.py`
  passes, asserting that one `info` call writes one parseable JSON line with
  the base fields, that the human line appears on standard error and not
  standard output and carries no timestamp, that `OBS_LOG_LEVEL=warning`
  suppresses `info`, and that the record carries the bound `run_id`.
- **Out of scope**: the `Printer` fold (OBS-3), any
  dispatch changes (OBS-4).

#### OBS-2: retired

- **Status**: retired. There is no work to execute under this id.

This item is retired with no replacement work. The general redaction filter
it specified is removed from the design entirely: no record type captures
environment variables, and no masking is applied to record content (see
[the event record schema](README.md#the-event-record-schema)). The id is retained so
later item ids stay stable. There is no work to execute under this id.

#### OBS-3: fold Printer into the logger

- **Goal**: `Printer` becomes a front-end over the logging core, so status
  calls produce both the human line and a structured record, and human output
  moves from standard output to standard error.
- **Status**: done. `common/output.py` routes through the logger; tests in
  `tests/common/test_common_output.py`.
- **Depends on**: OBS-1.
- **Scope**: `common/output.py`; `.styles/PYTHON_OUTPUT_STYLE.md`;
  `tests/common/test_common_output.py`; `tests/utils/test_check_py_deps.py`;
  `tests/utils/README.md`; `tests/training/test_cli_handlers_train.py`;
  `tests/training/test_cli_handlers_memfit.py`;
  `tests/training/test_cli_handlers_export.py`;
  `tests/training/test_cli_handlers_tokenizer.py`;
  `tests/training/test_cli_handlers_prepare.py`;
  `tests/training/test_cli_handlers_datatools.py`;
  `tests/interp/test_cli_handlers_sae.py`;
  `tests/interp/test_cli_handlers_organism.py`;
  `tests/interp/test_cli_handlers_probes.py`;
  `tests/interp/test_cli_handlers_steering.py`;
  `tests/interp/test_interp_cli.py`.
- **Design**:
  - Each `Printer` method (`ok`, `warn`, `error`, `info`, `run`, `skip`,
    `line`) calls the logging core with a level mapping: `ok`, `info`, `run`,
    `skip`, and `line` map to `info`; `warn` maps to `warning`; `error` maps
    to `error`. The status tag (`OK`, `RUN`, and so on) is preserved on the
    file record in a `status` field so the original vocabulary survives.
  - The human rendering keeps the `[OK]`, `[WARN]`, and other tags and the
    existing color and indentation rules, but writes to standard error. No
    timestamps are added; the console output is byte-identical to today
    except for the stream.
  - The TTY check at `common/output.py:18` moves from `sys.stdout.isatty()`
    to `sys.stderr.isatty()`, so color decisions follow the stream actually
    written. The class and module docstrings update to match.
  - `quiet=True` (the `--json` path) suppresses the human line only; the
    structured record is still written.
  - The public method signatures do not change, so the 508 call sites are
    untouched.
  - `.styles/PYTHON_OUTPUT_STYLE.md` reproduces the current stdout-based
    `Printer` verbatim; it updates in the same change, per the rule that
    documentation follows code.
  - Test migration: `tests/common/test_common_output.py` replaces
    `redirect_stdout` with `redirect_stderr` (and its fake-TTY buffer moves
    accordingly); the CLI handler tests listed in scope move their
    `capsys.readouterr().out` assertions on `[OK]`/`[WARN]`/`[ERROR]` lines
    to `.err`. Assertions on `--json` payloads stay on `.out`.
- **Acceptance check** (sandbox): the full suite passes
  (`uv run pytest tests/`), plus a new test that a `Printer(quiet=True)`
  call writes no console output but does write a file record, and a test
  that `--json` mode leaves standard output holding only the JSON payload.
- **Out of scope**: adding timestamps to console output (rejected — see
  [Output streams](README.md#output-streams));
  adding `--json` to interp commands.

### Phase B: event capture

#### OBS-4: Layer 1 command events and traced dispatch

- **Goal**: every invocation of either CLI generates a `run_id`, a
  `cmd.start` record, and a `cmd.end` record with exit code and duration, and
  exposes a phase timer for coarse in-command spans.
- **Status**: done. `common/obs/dispatch.py`; tests in
  `tests/common/test_obs_dispatch.py` and `tests/training/test_obs_integration.py`.
- **Depends on**: OBS-1, OBS-3.
- **Scope**: `common/obs/dispatch.py`, `training/cli.py` (`main`),
  `interp/cli.py` (`main`), `common/obs/core.py` (phase timer),
  `tests/common/test_obs_dispatch.py`.
- **Design**:
  - Both CLIs share the dispatch shape
    `args = build_parser().parse_args(argv); return args.func(args)`
    (`training/cli.py:851-853`, `interp/cli.py:60-62`). A shared
    `dispatch(cli, build_parser, argv)` in `common/obs/dispatch.py` wraps
    it: parse first, then generate a `run_id`, `configure(cli, run_id=...)`,
    `bind_run(...)`, emit `cmd.start` with the `argv` list, run the command
    inside a `try/finally`, and emit `cmd.end` with the exit code and
    duration in the `finally` so it fires on exceptions too. Duration is
    monotonic-clock elapsed time (it excludes suspend, unlike the wall-clock
    `timestamp` the catalog row carries). Both
    `main` functions become one-line calls into it.
  - Exit code mapping: an integer return passes through (`None` maps to 0);
    an uncaught exception maps to 1 (the CLI contract); `KeyboardInterrupt`
    maps to 130; a `SystemExit` raised mid-run (the SIGTERM traps in
    `eval_ft`, `generate_teacher`, and `serve_eval`) records its own integer
    code. The wrapper re-raises after recording, so the interpreter still
    prints the traceback and the process exit status is unchanged. Argparse
    usage errors raise `SystemExit(2)` during parsing, before a `run_id`
    exists; those invocations produce no records, which is accepted and
    documented.
  - `configure` treats the file handler as best-effort: when the log root
    cannot be created or the file cannot be opened, the file handler is
    skipped silently and the command runs with console output only. A
    machine without `/mnt/ai-data` (for example an agent sandbox) therefore
    runs every command normally and writes no event file.
  - Provide `phase(name)` as a context manager that emits `phase.start` and
    `phase.end` with the elapsed duration. It is used by later items and by
    long commands.
  - `argv` is recorded as passed; no masking is applied. The audit found no
    secret-bearing flag in either CLI (every token-named flag is a token
    count or tokenizer path).
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_dispatch.py` passes, asserting that a
  successful command writes `cmd.start` and `cmd.end` with `exit_code` 0 and a
  non-negative `duration_s`, that a command raising an exception still writes
  `cmd.end` with exit code 1, and that both records share one `run_id`.
- **Out of scope**: the structured `fail` record, which is OBS-5.

#### OBS-5: structured failure records

- **Goal**: an uncaught exception in a command is recorded as a `fail` event
  with the exception type, message, and a truncated traceback.
- **Status**: done. Failure recording in `common/obs/dispatch.py`; tests in
  `tests/common/test_obs_failure.py` and `tests/training/test_obs_durability.py`.
- **Depends on**: OBS-4.
- **Scope**: `common/obs/core.py` (failure recorder),
  `common/obs/dispatch.py` (wiring), `tests/common/test_obs_failure.py`.
- **Design**:
  - `record_failure(exc, exit_code)` emits a `fail` event with `exc_type`,
    `exc_msg`, a traceback truncated to the last 30 frames, and the exit
    code. Content is recorded as-is; no masking is applied (see
    [the event record schema](README.md#the-event-record-schema)). Bounds: the frame
    limit applies per exception in a chain and a 32 KiB character cap bounds
    the chain itself (tail kept — the frames nearest the raise); `exc_msg`
    caps at 4 KiB; an exception whose own `str()` raises records
    `<exception str() failed>`.
  - The dispatch wrapper calls `record_failure` when a command raises, so
    failures are captured uniformly rather than per command. The call is
    guarded: a failure inside the recorder can never replace the command's
    own exception.
  - `KeyboardInterrupt` and `SystemExit` produce no `fail` record:
    cancellation and a deliberate exit are not failures. `cmd.end` still
    records their exit codes (130 and the `SystemExit` code, with a bool
    code normalized to an integer). A `BaseExceptionGroup` containing only
    `KeyboardInterrupt`s counts as cancellation (130, no `fail` record); a
    mixed group is a failure.
  - The human error line continues to come from `Printer.error`, unchanged;
    the `fail` record is file-only.
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_failure.py` passes, asserting that a
  raised exception produces exactly one `fail` record carrying `exc_type`,
  `exc_msg`, and the exit code, that the traceback is bounded at 30 frames,
  that the record shares the run's `run_id`, and that `KeyboardInterrupt`
  and `SystemExit` produce no `fail` record.
- **Out of scope**: retry logic or error classification beyond recording.

#### OBS-6: Layer 2 resource audit hook

- **Goal**: when enabled, record every file opened, module imported, and
  subprocess spawned during a command, using a Python audit hook, at debug
  level.
- **Status**: done. `common/obs/audit.py`; tests in `tests/common/test_obs_audit.py`.
- **Depends on**: OBS-4.
- **Scope**: `common/obs/audit.py`, wiring into `common/obs/dispatch.py`,
  `tests/common/test_obs_audit.py`.
- **Design**:
  - `install_audit_trace()` registers a `sys.addaudithook` callback that maps
    the `open`, `import`, and `subprocess.Popen` (and `os.exec*`) audit events
    to `open`, `import`, and `spawn` records at debug level. Spawn argument
    vectors are recorded as passed, with bytes items decoded to text; an
    `open` event carrying a `None` mode (the `os.open` path) records its
    numeric flags as `flags=N` instead of defaulting to `"r"`.
  - The hook is installed only when `OBS_TRACE=1`, because audit hooks add
    measurable overhead and cannot be removed for the life of the process.
    The dispatch wrapper reads the variable and installs the hook before
    running the command, and calls `disable_audit_trace()` in its `finally`,
    so enablement is scoped to the traced command: later dispatches in the
    same process pay only the single flag check and receive no stale audit
    records. `OBS_TRACE=1` also lowers the effective log level
    to `debug` unless `OBS_LOG_LEVEL` is set explicitly; without this the
    debug records would be dropped and the flag would do nothing.
  - The hook never raises into the audited call; it swallows its own errors so
    an instrumentation failure cannot break the command.
  - Known limits: the hook observes the whole process, not only the handler's
    own activity; records emitted from threads other than the one that
    ran `bind_run` carry an empty run context (context variables do not cross
    thread boundaries); and the interpreter raises the `import` audit event
    only for the `import` statement, so `importlib.import_module` calls are
    not recorded. All are accepted for an opt-in forensic tool.
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_audit.py` passes with `OBS_TRACE=1`,
  asserting that opening a file, importing a module, and spawning a trivial
  subprocess each produce the corresponding debug record, and that with
  `OBS_TRACE` unset no such records appear.
- **Out of scope**: call-level or function-level tracing, which is a
  profiler task documented in OBS-13.

#### OBS-7: run-context capture at start

- **Goal**: each run records the environment it ran in, so a failure is
  diagnosable and reproducible.
- **Status**: done. `common/obs/context.py`; tests in `tests/common/test_obs_context.py`.
- **Depends on**: OBS-4.
- **Scope**: `common/obs/context.py`, wiring into `common/obs/dispatch.py`,
  `tests/common/test_obs_context.py`.
- **Design**:
  - `run_context()` returns a dict with the git SHA (reuse
    `common/provenance.py`, `git_sha`), the Python version, the platform
    string, and the versions of a fixed package list covering both CLIs
    (`torch`, `transformers`, `trl`, `peft`, `datasets`, `sae_lens`,
    `transformer_lens`, `nnsight`) obtained through
    `importlib.metadata.version`, omitting packages that are not installed.
    GPU name and driver information is included when a query succeeds and
    omitted when it does not, so the function never fails on a machine
    without a GPU. Environment variables are never captured (see
    [the event record schema](README.md#the-event-record-schema)).
  - The dispatch wrapper emits the context as a `run.context` record
    immediately after `cmd.start`.
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_context.py` passes, asserting that the
  context contains a git SHA and a Python version, that a missing GPU
  produces an omitted field rather than an exception, and that no record
  field contains the process environment.
- **Out of scope**: capturing service health; that requires running services
  and is validated on the workstation.

### Phase C: operability

#### OBS-8: retention and rotation

- **Goal**: bound the disk that event files consume. The catalog
  (`runs.jsonl`) is exempt by design: one summary line per invocation is
  the retention system's own index and grows slowly.
- **Status**: done. `common/obs/retention.py`; tests in
  `tests/common/test_obs_retention.py`.
- **Depends on**: OBS-1.
- **Scope**: `common/obs/retention.py`, wiring into `configure`,
  `tests/common/test_obs_retention.py`.
- **Design**:
  - `prune(log_dir, retention_days)` removes event files older than the
    retention window. It touches only files whose names match the run-id
    pattern (`R-*.jsonl`), so the catalog (`runs.jsonl`) and any foreign
    files are never candidates. File age is mtime, the run's last write.
  - A live event file is never pruned regardless of age: `configure` holds
    a shared `flock` on the file it writes for the life of the handler, and
    prune skips any candidate whose non-blocking exclusive probe fails. A
    long-idle run's mtime can age past the window while the file is open;
    unlinking it would send the rest of the run to a deleted inode.
  - `configure` calls it opportunistically at startup on the per-CLI
    directory it just ensured, after the file handler installs, so pruning
    needs no separate scheduler; a run degraded to console-only prunes
    nothing. Viewer commands (`report` and `runs` in both CLIs) opt out via
    `set_defaults(obs_retention=False)`: a command asked to display an old
    run must never delete that run's event file first.
  - The window comes from `OBS_RETENTION_DAYS`, default 30. A non-positive
    or unparseable value disables pruning.
  - Pruning is best-effort: a file that cannot be removed is skipped, not
    fatal, and nothing is printed.
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_retention.py` passes, asserting that a
  run-id-named file older than the window is removed, that a recent file is
  kept, that `runs.jsonl` is never removed regardless of age, that an
  unremovable file does not raise, and that a disabled window prunes
  nothing.
- **Out of scope**: size-based rotation within a single run; per-run files
  are bounded by the run itself.

#### OBS-9: run catalog and runs subcommand

- **Goal**: make runs discoverable, so a past run is found by id, time, or
  outcome without reading individual files.
- **Status**: done. `common/obs/catalog.py`, `training/commands/runs.py`,
  `interp/commands/runs.py`; tests in `tests/common/test_obs_catalog.py`.
- **Depends on**: OBS-4.
- **Scope**: `common/obs/catalog.py`, `training/commands/runs.py`,
  `interp/commands/runs.py`, registration in `training/cli.py` and
  `interp/cli.py`, `tests/common/test_obs_catalog.py`.
- **Design**:
  - After `cmd.end`, in the same dispatch `finally`, append one summary
    record to `${OBS_LOG_DIR}/<cli>/runs.jsonl`: `run_id`, `command`,
    `timestamp` (run-end time; the start is encoded in the run id),
    `duration_s`, `exit_code`, and `run_dir` (from `current_run_dir()`,
    null until OBS-10 wires `set_run_dir` into the loops). Riding the
    `finally` means failed and interrupted runs are summarized too. The
    append is a single `O_APPEND` write, so concurrent commands from both
    CLIs interleave whole lines; it creates no directories and skips
    silently when the log root is unwritable, the same degradation
    contract as `configure`.
  - `common/obs/catalog.py` implements reading, appending, and rendering;
    the two command modules are thin registrations following the existing
    `configure(sub)` pattern (`training/cli.py:833-847`,
    `interp/cli.py:33-46`).
  - The `runs` subcommand reads the catalog and prints a table of recent
    runs (oldest first, so the newest sits at the bottom) on standard
    error, honoring `--json` for machine output on standard out, a
    `--failed` filter, and `-n/--limit` (default 20). The catalog records
    every invocation, including `runs` itself; the display hides
    `command == "runs"` entries unless `--all` is passed, keeping the
    data complete and the default view signal-rich.
- **Acceptance check** (sandbox):
  `uv run pytest tests/common/test_obs_catalog.py` passes, asserting that a
  completed run appends one catalog record, that `runs --json` returns the
  recorded runs including a failed one, and that the default view hides
  `runs` entries until `--all`.
- **Out of scope**: cross-run statistics; that is ad hoc `jq` or DuckDB over
  the catalog, not a built-in feature.

### Phase D: metrics and dashboards

#### OBS-10: metrics emission as a tracker sink

- **Goal**: every training run writes a per-run `metrics.jsonl` conforming to
  the metrics schema, emitted by a JSON Lines sink inside the existing
  tracker fan-out, with no changes to how the loops compute or log metrics.
- **Status**: done. The sink in `training/runtime/tracking.py`; tests in
  `tests/training/test_obs_metrics.py`.
- **Depends on**: OBS-1, OBS-4.
- **Scope**: `training/runtime/tracking.py` (the sink, plus a `header`
  parameter on `Tracker`), `training/recipes/schema.py` (`TrackingConfig`
  gains `metrics_jsonl: bool = True`), `training/runtime/loop.py` (header
  construction at the `Tracker` call, `set_run_dir`, and a
  `checkpoint_saved` marker at the save sites), `training/tune/sft.py` and
  `training/tune/distill.py` (header construction and `set_run_dir` at their
  `Tracker` calls), `training/tune/sft_callbacks.py` (`train/ce_loss` and
  `train/kl_loss` from the distill trainer's existing `last_ce`/`last_kl`),
  `tests/training/test_obs_metrics.py`.
- **Design**:
  - A `jsonl` sink joins the tensorboard, wandb, and mlflow sinks in
    `Tracker` (`training/runtime/tracking.py`), enabled by
    `TrackingConfig.metrics_jsonl` (default true — it is the observability
    system's durable record, unlike the optional external integrations). It
    registers as `("jsonl", ...)` in the `_sinks` fan-out, initialized
    first, and follows the same discipline as the other sinks: disabled
    with a stderr warning on failure, never able to kill a run. The file
    opens line-buffered on the first `log` call, so a run that never logs
    leaves no file and a killed run keeps every completed step. The sink is
    internally locked: the training thread and the watchdog sampler thread
    both log into it, `TextIOWrapper` is not thread-safe, and an unlocked
    lazy open double-fired under a first-log race. Eviction from the
    fan-out tolerates a concurrent eviction of the same sink and tears the
    evicted sink down best-effort (fd, writer thread, active mlflow run).
  - `Tracker.__init__` gains `header: dict | None = None`. The three
    construction sites (`training/runtime/loop.py`, `training/tune/sft.py`,
    `training/tune/distill.py`) pass `tier`, `model_id`, `recipe_hash`
    (`snapshot_hash(recipe)`, already computed at each site), and a schedule
    summary — all values they already hold. `tier` stays in the
    `pretrain`/`sft`/`distill` vocabulary: the SFT entry point reports
    `distill` when `recipe.distill.enabled` (response tier), and the
    logit-tier sub-distinction is recoverable from `command` and the recipe
    snapshot. `model_id` falls back to `scratch-<layers>L-<hidden>d` for
    from-scratch models in the D6 loop. The schedule summary is the full
    WSD tuple in the D6 loop and `{type, warmup_steps, total_steps}` for
    the trl family, whose recipes reject WSD knobs.
  - On first write the sink emits the header record:
    `report_header("run_metrics")` extended with the header dict plus
    `run_id`, `cli`, and `command` from the ambient context
    (`common/obs`). Subsequent `log(metrics, step)` calls append one record
    `{step, ts, **metrics}` with keys exactly as logged. Writes are
    line-buffered appends, so a killed run keeps every completed step.
  - The loops call `set_run_dir(run_dir)` right after `create_run_dir`, which
    emits the `run.dir` event and completes the catalog join (OBS-9).
  - Watchdog interval samples and `watchdog/<signal>` alerts already flow
    through `tracker.log` (`training/runtime/watchdog.py:195-196` and
    `_alert`), so they land in the file with no watchdog changes.
  - The D6 loop adds one `tracker.log({"checkpoint_saved": 1}, step)` after
    its normal interval/final save, giving the dashboard checkpoint
    markers. The emergency save in the watchdog-abort path is not marked —
    the watchdog alert and the fail event already record that step. SFT and
    distill checkpoints are written inside the TRL trainer; markers for
    them are out of scope in this version.
  - The distill CE and KL values are already computed and stored on the
    trainer (`training/tune/distill.py`) but never logged. The tracking
    callback has no trainer reference, so the wiring stays inside the
    documented transformers extension point: the distill trainer overrides
    `log()` to inject `ce_loss` and `kl_loss` beside the windowed `loss`
    (evaluation logs pass untouched), and the callback maps them to
    `train/ce_loss` and `train/kl_loss` like every other metric.
  - Interpretability loops: SAE training delegates its inner loop to
    SAELens, which never calls the repository's tracker, so per-step interp
    metrics are not reachable without patching SAELens; end-of-run manifests
    remain the interp record, and this is a documented limitation, not a
    later phase.
- **Acceptance check** (sandbox for the unit; host for the real series):
  `uv run pytest tests/training/test_obs_metrics.py` passes, asserting that a
  synthetic sequence of `log` calls produces a header that `validate_report`
  accepts, one record per step with keys as logged, marker records for a
  synthetic checkpoint and watchdog alert, empty-string context fields for a
  tracker constructed outside a dispatched CLI, and no file at all when
  `metrics_jsonl` is false. On the workstation, a short
  `train-sft` run writes a `metrics.jsonl` whose step count matches the run;
  record the command and output in the verification runbook.
- **Out of scope**: the rendering, which is OBS-11; SFT and distill
  checkpoint markers; per-step SAE metrics.

#### OBS-11: snapshot dashboard generator

- **Goal**: a `report` subcommand renders a run's metrics and logs into one
  self-contained HTML file with no external dependencies.
- **Status**: done. `common/obs/dashboard.py`, `common/obs/viewer.js`,
  `training/commands/report.py`; tests in `tests/training/test_report.py`.
- **Depends on**: OBS-10.
- **Scope**: `training/commands/report.py` and its registration in
  `training/cli.py`, `common/obs/dashboard.py` (the page builder),
  `common/obs/viewer.js`, `common/obs/vendor/uplot/` (`uPlot.iife.min.js`,
  `uPlot.min.css`, `README.md`), `tests/training/test_report.py`.
- **Design**:
  - `python -m training report --run <run_dir>` reads `metrics.jsonl` from
    the run directory, locates the event file at
    `${OBS_LOG_DIR}/<cli>/<run_id>.jsonl` from the header's identity
    (overridable with `--events <path>`), and writes `<run_dir>/run.html`.
    The command module is a thin Printer/exit-code shell; the page assembly
    lives in `common/obs/dashboard.py`, a pure builder returning the HTML
    string. A missing `metrics.jsonl` is exit 1; a missing or unlocatable
    event file degrades to a warning and an empty log viewer (the charts
    are the primary artifact); malformed JSON Lines are skipped and
    counted, with one warning per file.
  - The chart layer is uplot 1.6.32, vendored at `common/obs/vendor/uplot/`
    with its version and tarball checksum recorded (see
    [Where the files are](README.md#where-the-files-are)) and verified
    at vendoring time; the vendor `README.md` carries the version, source
    URL, sha256, and update procedure. uPlot is canvas-based, 51KB
    minified, MIT-licensed, and built for dense time series, which matches
    a per-step series of 10,000 or more points.
  - The page inlines everything — the two data sets as script blocks, the
    uPlot script and stylesheet, and `common/obs/viewer.js` — and references
    no external URL. Payloads embed with every `<` escaped to the JSON
    escape `\u003c` (valid because `<` only occurs inside string values):
    record text containing `</script>` cannot terminate the enclosing
    block, and `<!--<script` cannot enter the script-data-double-escaped
    parser state that would swallow the page. Non-finite floats read from
    legacy files map to `null` before embedding. The reader also accepts a
    UTF-8 BOM, and header identity used for the events join must be plain
    string names (no path separators or `..`). The
    viewer module draws the loss curves (`loss`, `train/loss`,
    `eval/holdout_loss`), the learning-rate schedule, throughput, and
    device memory (`gpu_mem_allocated_gb`, `gpu_vram_used_gb`), with
    `checkpoint_saved` and `watchdog/<signal>` markers drawn on the loss
    chart's step axis, handling both key vocabularies from the metrics
    schema. Series draw with `spanGaps`: every series sits on the union
    step axis with nulls where it was not logged, and without it a series
    sparser than the axis (eval points, watchdog samples) never draws a
    line at all. Below the charts it renders a log viewer over the event
    records with level filtering and substring search; unknown or missing
    levels always show (hiding a foreign `critical` behind the filter
    would invert the failure mode), and the first header record wins,
    matching the Python reader.
  - `viewer.js` takes its data through one entry point
    (`window.obsRender(root, {header, metrics, events, live})`), so OBS-12
    reuses it unchanged with polled data. Charts resize with the window,
    and the rendered state is exposed on the root node for the browser
    verification harness (`scripts/verify/obs_browser`).
- **Acceptance check** (sandbox):
  `uv run pytest tests/training/test_report.py` passes, asserting that from a
  fixture `metrics.jsonl` and event file the command writes a single HTML
  file that contains the inlined data and no `src="http` or `href="http`
  reference. Opening the file in a browser on the workstation shows the
  charts and the log viewer; record the observation in the verification
  runbook.
- **Out of scope**: multi-run comparison (OBS-15 renders an index, not
  comparisons), live refresh (OBS-12), an interp `report` registration.

#### OBS-12: live viewer

- **Goal**: view a running job's dashboard as it progresses — charts
  extending and logs tailing — in any browser, from an ephemeral local
  server that follows the stateless viewer principle.
- **Status**: done. `common/obs/serve.py` behind `report --serve`; tests in
  `tests/common/test_obs_serve.py`, plus the browser check in
  `scripts/verify/obs_browser/`.
- **Depends on**: OBS-11.
- **Scope**: `training/commands/report.py` (`--serve`, `--host`, `--port`
  flags), `common/obs/serve.py`, `common/obs/viewer.js` (follow mode),
  `tests/common/test_obs_serve.py`.
- **Design**:
  - `report --run <run_dir> --serve` starts a standard library
    `ThreadingHTTPServer` (no new dependency, `common/obs/serve.py`) bound
    to `127.0.0.1:8010` by default; `--host 0.0.0.0` opts in to LAN
    exposure (a conscious choice — there is no authentication) and `--port`
    overrides the port. Serve mode writes no `run.html` and does not
    require `metrics.jsonl` to exist yet: the run may have just started.
  - Endpoints: `GET /` serves the live page shell (same `viewer.js`, no
    inlined data); `GET /data/metrics?offset=<bytes>` and
    `GET /data/events?offset=<bytes>` return
    `{"offset": <new_offset>, "lines": [<raw JSONL lines>]}` — the complete
    lines appended since the byte offset. A trailing partial line is held
    back until its newline arrives; a missing file or an offset at the end
    returns the offset unchanged with no lines; a file now shorter than
    the offset (truncated or replaced) resets the offset to 0 rather than
    stalling silently; an invalid or negative offset reads from 0; reads
    are capped at 8 MiB per request, cut at the last newline, with the
    client's next poll continuing from the returned offset; unknown paths
    are 404 and `HEAD /` answers with headers only. The server holds
    no state; the client owns its offsets. The events path comes from the
    `--events` override, else lazily from the metrics header (a bounded
    prefix scan — a torn first line must not blank the log pane — retried
    each poll until the header lands, then cached). A binding failure is a
    styled error, and a client aborting its in-flight request is silent:
    per-request logging is suppressed — a 2-second poll would print two
    stderr lines per cycle.
  - The page polls every 2 seconds via a chained timeout (never
    `setInterval`: a poll slower than the interval must finish before the
    next fires, or two in-flight polls read the same offset and ingest
    everything twice), appends new records (the header record is
    recognized in the polled stream and fills the header strip), extends
    the charts, and tails the log viewer incrementally — steady-state
    appends only the new rows; a full rebuild happens only when the level
    or search filter changes. The two endpoints fail independently, so a
    transient events error cannot leave already-ingested metrics
    unrendered. A user's drag-zoom survives new data (`setData` rescales
    only when unzoomed). Follow mode pins the log view to
    the newest record; scrolling away from the bottom disengages the pin
    and returning re-engages it. Polling continues after `cmd.end`; a
    finished run simply stops producing new lines.
  - The server runs for the duration of viewing and dies with Ctrl-C.
    Killing it loses nothing; the files are the record. Tests bind port 0;
    8010 stays the interactive default.
- **Acceptance check** (sandbox for the endpoints; host for the live view):
  `uv run pytest tests/common/test_obs_serve.py` passes, asserting that the
  offset endpoints return only complete appended lines and a correct new
  offset against a fixture directory that grows between requests. On the
  workstation, starting the server during a run shows the charts extending
  and the log tail following; record the procedure in the observability
  reference doc.
- **Out of scope**: server-sent events or WebSockets (polling is sufficient
  for one viewer on localhost; revisit only if it proves laggy); serving
  more than one run (OBS-15).

#### OBS-15: rig console (optional)

- **Goal**: one grand dashboard — a long-lived, always-available page
  indexing every run across both CLIs, with links into per-run live and
  snapshot views.
- **Status**: not started, and optional by design. Implement only if living with
  OBS-12 shows the ephemeral viewer is insufficient.
- **Depends on**: OBS-9, OBS-11, OBS-12.
- **Scope**: `common/obs/serve.py` (index mode), a compose service
  definition, a `make` target, documentation.
- **Design**:
  - The same stateless viewer gains an index mode: it reads
    `${OBS_LOG_DIR}/<cli>/runs.jsonl` for both CLIs, renders a landing page
    of recent runs (id, command, when, duration, exit code), and links each
    to its live view. It mounts `${OBS_LOG_DIR}` and the run output roots
    read-only, keeping the no-write-path guarantee structural.
  - Deployment follows the `harness-serve` precedent: a small compose
    service with a healthcheck, port 8010, localhost-bound unless the
    compose file is deliberately edited, started and stopped through a
    `make` target.
  - Implement this item only if living with OBS-12 shows the on-demand
    server is insufficient — the same discipline the original roadmap
    applied to live refresh, one level up. The decision point and rationale
    are recorded here rather than pre-committed.
- **Acceptance check** (host): the console lists runs from both CLIs and
  opens a live per-run view; the container holds only read-only mounts.
  Record the procedure in the observability reference doc.
- **Out of scope**: everything the non-goals exclude — no database, no
  authentication, no multi-run comparison views in the first version.

### Phase E: documentation

#### OBS-13: debugging with profilers guide

- **Goal**: document how to answer "which function or file called which" with
  off-the-shelf profilers — the function-level layer of the tracing story
  (see [What tracing means in this system](#what-tracing-means-in-this-system)).
- **Status**: open. `docs/observability/debugging-runs.md` is not written.
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
- **Status**: done. [README.md](README.md), the operator reference; the
  specification sections it needed graduated out of this document at the same time.
- **Depends on**: the items whose behavior it documents.
- **Scope**: `docs/observability/README.md`.
- **Design**: document the file layout, the `OBS_` environment variables, the
  event and metrics schemas, the `runs` and `report`
  subcommands including `--serve`, and example `jq` queries over the event
  files and the catalog. Cross-reference this roadmap and
  `docs/observability/report_schema.md`.
- **Acceptance check** (sandbox): the doc conforms to
  `.styles/DOCUMENTATION_STYLE.md` and every command in it is runnable as
  written.
- **Out of scope**: none; this is the closing item.

## Sequencing and dependencies

The critical path is OBS-1, then OBS-3, then OBS-4, then the rest. The
dependency summary:

| Item | Depends on |
|------|------------|
| OBS-1 logging core | none |
| OBS-2 (retired) | no work |
| OBS-3 Printer fold | OBS-1 |
| OBS-4 command events | OBS-1, OBS-3 |
| OBS-5 failure records | OBS-4 |
| OBS-6 audit hook | OBS-4 |
| OBS-7 run context | OBS-4 |
| OBS-8 retention | OBS-1 |
| OBS-9 run catalog | OBS-4 |
| OBS-10 metrics sink | OBS-1, OBS-4 |
| OBS-11 snapshot dashboard | OBS-10 |
| OBS-12 live viewer | OBS-11 |
| OBS-15 rig console | OBS-9, OBS-11, OBS-12 |
| OBS-13 profiler guide | none |
| OBS-14 reference doc | the items it documents |

OBS-13 has no code dependency and can be written first if a debugging guide
is needed before the system lands.

A later, separate effort extends the correlation id across process
boundaries: exporting `OBS_RUN_ID` into subprocess environments so children
stamp their own records, and passing the id as a request header to the
inference services. This is explicitly deferred from the first version and is
not required by any item above.

The rust workspace (`rust-harness/obs/src/lib.rs`) is a second,
contract-compatible writer of this file layout: it mints the same run id
format and writes `cmd.start`/`fail`/`cmd.end` events, `phase.start`/`phase.end`
spans, a `metrics.jsonl` under a per-run directory, and a catalog line
under `${OBS_LOG_DIR}/harness/`, holding the same shared flock the
retention pruner probes. Two of its binaries write today — `harness` and
`ladder` — sharing one `cli` name with the binary name as `command`, so the
whole workspace is one tier from a reader's point of view rather than one per
tool. The JSONL files are the integration seam — no
code is shared, and the stateless readers above consume harness runs
unchanged. The rust writer also specifies the receiving half of the
deferred propagation: when `OBS_RUN_ID` is set in its environment, the
harness records it as a `parent_run_id` field on `cmd.start` while still
minting its own run id.

## Acceptance and testing conventions

- Every item ends by running `make quality` and its own acceptance check.
  `make quality` runs the ruff format check, ruff lint, mypy, the uv lockfile
  check, the documentation checks, and shellcheck. In a sandbox the lockfile
  step announces itself as skipped (the pytorch index is unreachable there);
  every other step runs, and all of them must pass before any push.
- Sandbox-runnable acceptance checks are unit tests that need no GPU, no
  Docker, and no running service. Most items above are sandbox-runnable in
  full.
- Unit tests for `common/obs/` modules live under `tests/common/`, beside the
  rest of the `common/` tests; training-integration tests live under
  `tests/training/`.
- Host-side acceptance checks (a real `metrics.jsonl` from a training run in
  OBS-10, the live viewer in OBS-12, opening a rendered page) run on the
  workstation. The operator runs the command and records the command and its
  output; the agent does not claim to have run a host-side command.
- Each item is one change. Do not combine an item with an unrelated change,
  in keeping with the repository's one-change-at-a-time rule.
