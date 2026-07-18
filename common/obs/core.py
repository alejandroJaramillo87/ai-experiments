"""Structured logging core for the observability system (roadmap OBS-1).

One emit path produces two renderings of the same record: a human status
line on standard error and a JSON Lines record in a per-run file. Every
record carries the ambient run context (run id, cli, command, pid) held
in a context variable, so no call site threads the correlation id through
function signatures.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from common.obs.retention import prune

DEFAULT_LOG_DIR = "/mnt/ai-data/logs"

# Bound fail-record tracebacks so a recursion error cannot bloat the file.
TRACEBACK_FRAME_LIMIT = 30

_LOGGER_NAME = "obs"

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

# Console tags follow the Printer vocabulary: WARN, not WARNING.
_HUMAN_TAGS = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
}


@dataclass(frozen=True)
class _RunContext:
    run_id: str = ""
    cli: str = ""
    command: str = ""
    pid: int = 0
    run_dir: str | None = None


# A frozen instance is safe to share as the unbound default (B039 guards
# against mutable defaults; this one is immutable).
_EMPTY_CONTEXT = _RunContext()

_context: ContextVar[_RunContext] = ContextVar(
    "obs_run_context", default=_EMPTY_CONTEXT
)

# The package logger never propagates to the root logger, and the
# NullHandler keeps logging.lastResort from writing warning+ records to
# stderr while the system is unconfigured (every CLI run until OBS-4
# wires configure into dispatch).
_logger = logging.getLogger(_LOGGER_NAME)
_logger.propagate = False
_logger.addHandler(logging.NullHandler())


class _ConsoleFilter(logging.Filter):
    """Skip records a front-end already rendered, and pure-data records.

    Front-ends such as Printer draw their own richer console line (tags,
    color, indentation) and mark their records file-only. Records with an
    empty message carry data for the file, not a human sentence.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "obs_console", True):
            return False
        return bool(record.getMessage())


class _HumanFormatter(logging.Formatter):
    """``[LEVEL] message`` with no timestamp; time lives in the file record."""

    def format(self, record: logging.LogRecord) -> str:
        tag = _HUMAN_TAGS.get(record.levelno, record.levelname)
        return f"[{tag}] {record.getMessage()}"


class _JsonFormatter(logging.Formatter):
    """One JSON object per line; never raises on non-serializable fields."""

    def format(self, record: logging.LogRecord) -> str:
        ctx = _context.get()
        ts = datetime.fromtimestamp(record.created).astimezone()
        payload: dict[str, Any] = {
            "ts": ts.isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "run_id": ctx.run_id,
            "cli": ctx.cli,
            "command": ctx.command,
            "pid": ctx.pid or os.getpid(),
            "event": getattr(record, "obs_event", "log"),
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "obs_fields", None) or {})
        return json.dumps(payload, default=repr)


class StructuredLogger:
    """Level methods plus named events, all accepting arbitrary fields."""

    def __init__(self, name: str) -> None:
        qualified = f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME
        self._logger = logging.getLogger(qualified)

    def _emit(self, level: int, msg: str, event: str, fields: dict[str, Any]) -> None:
        self._logger.log(
            level,
            msg,
            extra={"obs_event": event, "obs_fields": fields, "obs_console": True},
        )

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, msg, "log", fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._emit(logging.INFO, msg, "log", fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._emit(logging.WARNING, msg, "log", fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit(logging.ERROR, msg, "log", fields)

    def event(
        self, name: str, msg: str = "", *, level: str = "info", **fields: Any
    ) -> None:
        self._emit(_LEVELS.get(level, logging.INFO), msg, name, fields)


def get_logger(name: str = "") -> StructuredLogger:
    return StructuredLogger(name)


def emit(
    level: str,
    msg: str,
    *,
    event: str = "log",
    console: bool = True,
    **fields: Any,
) -> None:
    """Low-level emit for front-ends that render their own console line.

    With ``console=False`` the record goes to the file handler only; the
    caller owns the human rendering (Printer after roadmap OBS-3).
    """
    _logger.log(
        _LEVELS.get(level, logging.INFO),
        msg,
        extra={"obs_event": event, "obs_fields": fields, "obs_console": console},
    )


def new_run_id() -> str:
    """``R-YYYYMMDD-HHMMSS-xxxxxx``: sortable by start time, unique by suffix."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"R-{stamp}-{secrets.token_hex(3)}"


def bind_run(cli: str, command: str, run_id: str) -> None:
    """Set the ambient run context stamped on every record."""
    _context.set(_RunContext(run_id=run_id, cli=cli, command=command, pid=os.getpid()))


def current_run_id() -> str | None:
    return _context.get().run_id or None


def set_run_dir(path: str | os.PathLike[str]) -> None:
    """Stash the run directory in the context and record it as an event."""
    _context.set(replace(_context.get(), run_dir=str(path)))
    emit("info", "", event="run.dir", path=str(path))


def current_run_dir() -> str | None:
    return _context.get().run_dir


def record_failure(exc: BaseException, exit_code: int) -> None:
    """Record an uncaught exception as a structured fail event.

    File-only: the dispatch wrapper re-raises after recording, so the
    interpreter still prints the traceback to the console. A negative
    limit keeps the last frames — the ones nearest the raise site.
    """
    tb = traceback.TracebackException.from_exception(exc, limit=-TRACEBACK_FRAME_LIMIT)
    emit(
        "error",
        "",
        event="fail",
        console=False,
        exc_type=type(exc).__name__,
        exc_msg=str(exc),
        traceback="".join(tb.format()),
        exit_code=exit_code,
    )


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Bracket a coarse in-command span with phase.start and phase.end.

    File-only records: the command's own status lines remain the human
    view of progress. phase.end always fires, so an exception inside the
    span still records its duration.
    """
    emit("info", "", event="phase.start", console=False, phase=name)
    start = time.monotonic()
    try:
        yield
    finally:
        emit(
            "info",
            "",
            event="phase.end",
            console=False,
            phase=name,
            duration_s=round(time.monotonic() - start, 3),
        )


def configure(
    cli: str,
    *,
    run_id: str | None = None,
    log_dir: str | None = None,
    level: str | None = None,
) -> str:
    """Install the stderr and JSON Lines handlers; return the run id used.

    Level falls back to ``OBS_LOG_LEVEL`` (default ``info``) and the file
    root to ``OBS_LOG_DIR``. Reconfiguring replaces the previous handlers,
    so repeated calls never stack duplicate outputs.
    """
    run_id = run_id or new_run_id()
    log_root = Path(log_dir or os.environ.get("OBS_LOG_DIR") or DEFAULT_LOG_DIR)
    level_name = (level or os.environ.get("OBS_LOG_LEVEL") or "info").lower()

    _logger.setLevel(_LEVELS.get(level_name, logging.INFO))
    for handler in list(_logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            _logger.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_HumanFormatter())
    console.addFilter(_ConsoleFilter())
    _logger.addHandler(console)

    # Best-effort file handler: on a machine without the log root (an agent
    # sandbox has no /mnt/ai-data) the command runs console-only. Silent by
    # design; a warning here would break every stderr output contract.
    try:
        file_dir = log_root / cli
        file_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            file_dir / f"{run_id}.jsonl", encoding="utf-8"
        )
    except OSError:
        return run_id
    file_handler.setFormatter(_JsonFormatter())
    _logger.addHandler(file_handler)

    # Opportunistic retention (OBS-8) on the directory just ensured; a
    # degraded console-only run prunes nothing. Invalid values disable it.
    try:
        retention_days = int(os.environ.get("OBS_RETENTION_DAYS", "30"))
    except ValueError:
        retention_days = 0
    prune(file_dir, retention_days)

    return run_id
