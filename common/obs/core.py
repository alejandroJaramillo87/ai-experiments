"""Structured-logging core (OBS-1).

A StructuredLogger over the standard library `logging` module. Each call fans out
to whatever handlers `configure()` has installed: a human line on standard error
(`HH:MM:SS.mmm [LEVEL] message`) and one JSON Lines record on disk, both carrying
an ambient correlation id and command context held in a `contextvars` context.

The module imports the standard library only, so importing it stays cheap and safe
in a minimal environment (no third-party dependency is pulled in).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

LOGGER_NAME = "aix"

AIX_LOG_LEVEL_ENV = "AIX_LOG_LEVEL"
AIX_LOG_DIR_ENV = "AIX_LOG_DIR"
DEFAULT_LOG_LEVEL = "info"
DEFAULT_LOG_DIR = "/mnt/ai-data/logs"

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# The package logger owns all obs handlers. A NullHandler attached at import keeps
# the unconfigured state silent: it stops the stdlib "last resort" handler from
# printing WARNING+ records to stderr when no handler is installed. propagate=False
# keeps records off the root logger (which pytest or a third-party library may have
# configured), so nothing double-emits.
_pkg_logger = logging.getLogger(LOGGER_NAME)
_pkg_logger.addHandler(logging.NullHandler())
_pkg_logger.propagate = False


@dataclass(frozen=True)
class _RunContext:
    """Ambient context stamped onto every record: the correlation id and command."""

    cli: str | None = None
    command: str | None = None
    run_id: str | None = None
    pid: int = field(default_factory=os.getpid)


_run_context: ContextVar[_RunContext] = ContextVar(
    "aix_run_context", default=_RunContext()
)


def _sanitize(value: Any) -> Any:
    """Centralized field sanitizer seam — currently an identity pass-through.

    Every field value (base and caller-supplied) routes through this function
    before serialization. It is intentionally a no-op: this is a single-user local
    workstation with no log shipper and no off-box sink, so there is no
    exfiltration surface to redact. Keeping the seam lets redaction become a
    one-line change if that surface is ever reopened. Do not add redaction here
    without that need.
    """
    return value


def _coerce(value: Any) -> str:
    """json.dumps `default` hook: represent a non-serializable value with repr."""
    return repr(value)


class _ContextFilter(logging.Filter):
    """Stamp the active run context onto every record.

    Attached to the handlers rather than the logger: `Logger.callHandlers` runs
    each handler's own filters for every ancestor handler, but only the originating
    logger's filters — a filter on `aix` would be skipped for records emitted by an
    `aix.<child>` logger. Values are frozen onto the record at emit time, so the
    design assumes synchronous handlers (no QueueHandler / async formatting).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _run_context.get()
        record.__dict__.update(
            run_id=ctx.run_id, cli=ctx.cli, command=ctx.command, pid=ctx.pid
        )
        return True


class _ConsoleFilter(logging.Filter):
    """Drop records marked `console=False` from the stderr handler only.

    The file handler omits this filter, so a file-only record (for example one
    emitted by the Printer front-end, which renders its own styled line) is
    captured on disk without being re-rendered on the console.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "console", True))


class _HumanFormatter(logging.Formatter):
    """Render `HH:MM:SS.mmm [LEVEL] message` for the stderr stream handler."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        msecs = int(record.msecs)
        return f"{stamp}.{msecs:03d} [{record.levelname}] {record.getMessage()}"


class _JsonFormatter(logging.Formatter):
    """Serialize one JSON object per line with the base schema; never raise.

    Base fields come from an explicit allowlist (never `record.__dict__`), so a
    control attribute such as `console` cannot leak into the file. Caller fields
    ride in `obs_fields` and are merged without clobbering the base schema. A
    non-serializable value is coerced with repr, and any residual serialization
    error falls back to a minimal record — a formatter must never raise.
    """

    def format(self, record: logging.LogRecord) -> str:
        created = datetime.fromtimestamp(record.created).astimezone()
        base = {
            "ts": created.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "run_id": getattr(record, "run_id", None),
            "cli": getattr(record, "cli", None),
            "command": getattr(record, "command", None),
            "pid": getattr(record, "pid", os.getpid()),
            "event": getattr(record, "event", ""),
            "msg": record.getMessage(),
        }
        out = {key: _sanitize(value) for key, value in base.items()}
        for key, value in getattr(record, "obs_fields", {}).items():
            if key not in out:  # base schema wins; caller fields cannot clobber it
                out[key] = _sanitize(value)
        try:
            return json.dumps(out, default=_coerce, ensure_ascii=False)
        except Exception:
            return json.dumps(
                {
                    "ts": out["ts"],
                    "level": "ERROR",
                    "event": "log_serialization_error",
                    "msg": repr(out.get("msg")),
                }
            )


class _ObsStreamHandler(logging.StreamHandler):
    """Marker subclass so configure() can identify obs-owned stream handlers."""


class _ObsFileHandler(logging.FileHandler):
    """Marker subclass so configure() can identify obs-owned file handlers."""


class StructuredLogger:
    """Front-end over one stdlib logger that emits structured records.

    Caller fields ride in a single `obs_fields` extra so an arbitrary field name
    can never collide with a reserved LogRecord attribute (which would raise a
    KeyError at emit time). `console` is a keyword-only control flag consumed by the
    stderr handler's filter and is never serialized.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _emit(
        self, levelno: int, msg: str, event: str, console: bool, fields: dict[str, Any]
    ) -> None:
        if not self._logger.isEnabledFor(levelno):
            return
        self._logger.log(
            levelno,
            msg,
            extra={"event": event, "console": console, "obs_fields": fields},
        )

    def debug(self, msg: str = "", *, console: bool = True, **fields: Any) -> None:
        self._emit(logging.DEBUG, msg, "", console, fields)

    def info(self, msg: str = "", *, console: bool = True, **fields: Any) -> None:
        self._emit(logging.INFO, msg, "", console, fields)

    def warning(self, msg: str = "", *, console: bool = True, **fields: Any) -> None:
        self._emit(logging.WARNING, msg, "", console, fields)

    def error(self, msg: str = "", *, console: bool = True, **fields: Any) -> None:
        self._emit(logging.ERROR, msg, "", console, fields)

    def event(
        self,
        name: str,
        *,
        level: int = logging.INFO,
        console: bool = True,
        **fields: Any,
    ) -> None:
        msg = fields.pop("msg", name)
        self._emit(level, msg, name, console, fields)


def bind_run(cli: str, command: str | None, run_id: str) -> None:
    """Set the ambient run context read by every subsequent record."""
    _run_context.set(
        _RunContext(cli=cli, command=command, run_id=run_id, pid=os.getpid())
    )


def current_run_id() -> str | None:
    """Return the run id in the active context, or None if unbound."""
    return _run_context.get().run_id


def new_run_id() -> str:
    """Return a fresh run id of the form R-YYYYMMDD-HHMMSS-xxxxxx."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"R-{stamp}-{secrets.token_hex(3)}"


def get_logger(name: str) -> StructuredLogger:
    """Return a StructuredLogger bound to the `aix.<name>` child logger."""
    return StructuredLogger(logging.getLogger(f"{LOGGER_NAME}.{name}"))


def _parse_level(level: str) -> int:
    return _LEVELS.get(level.strip().lower(), logging.INFO)


def configure(
    cli: str,
    *,
    run_id: str | None = None,
    log_dir: str | None = None,
    level: str | None = None,
) -> str:
    """Install the stderr and file handlers on the package logger; return the run id.

    Idempotent: obs-owned handlers from a prior call are removed and closed before
    fresh ones are added, so repeated calls (as in tests) do not stack handlers,
    leak file descriptors, or write to a previous run's file. The file lands at
    `${log_dir}/<cli>/<run_id>.jsonl`. Level comes from AIX_LOG_LEVEL unless
    overridden. Binds the run context so records and the file path share one id.
    """
    run_id = run_id or new_run_id()
    log_dir = log_dir or os.environ.get(AIX_LOG_DIR_ENV, DEFAULT_LOG_DIR)
    levelno = _parse_level(
        level or os.environ.get(AIX_LOG_LEVEL_ENV, DEFAULT_LOG_LEVEL)
    )

    for handler in list(_pkg_logger.handlers):
        if isinstance(handler, (_ObsStreamHandler, _ObsFileHandler)):
            handler.close()
            _pkg_logger.removeHandler(handler)

    _pkg_logger.setLevel(levelno)
    context_filter = _ContextFilter()

    stream_handler = _ObsStreamHandler(sys.stderr)
    stream_handler.setLevel(levelno)
    stream_handler.setFormatter(_HumanFormatter())
    stream_handler.addFilter(_ConsoleFilter())
    stream_handler.addFilter(context_filter)
    _pkg_logger.addHandler(stream_handler)

    path = os.path.join(log_dir, cli, f"{run_id}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_handler = _ObsFileHandler(path, encoding="utf-8")
    file_handler.setLevel(levelno)
    file_handler.setFormatter(_JsonFormatter())
    file_handler.addFilter(context_filter)
    _pkg_logger.addHandler(file_handler)

    bind_run(cli, None, run_id)
    return run_id
