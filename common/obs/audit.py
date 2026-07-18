"""Layer 2 resource audit hook (roadmap OBS-6).

Maps the interpreter's open, import, and process-spawn audit events to
debug-level file-only records, so a forensic run (OBS_TRACE=1) shows every
file a command touched, every module it imported, and every subprocess it
started. sys.addaudithook callbacks cannot be removed for the life of the
process, so enablement is flag-gated: the hook is registered once and
consults the flag on every event, keeping the disabled path to a single
attribute check.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

from common.obs.core import emit

_installed = False
_enabled = False
_reentry = threading.local()


def _spawn_argv(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        return [str(raw)]
    return [str(item) for item in raw]


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if not _enabled or getattr(_reentry, "active", False):
        return
    _reentry.active = True
    try:
        if event == "open":
            emit(
                "debug",
                "",
                event="open",
                console=False,
                path=str(args[0]),
                mode=str(args[1] or "r"),
            )
        elif event == "import":
            emit("debug", "", event="import", console=False, module=str(args[0]))
        elif event == "subprocess.Popen":
            emit(
                "debug",
                "",
                event="spawn",
                console=False,
                argv0=str(args[0]),
                argv=_spawn_argv(args[1]),
            )
        elif event == "os.exec":
            emit(
                "debug",
                "",
                event="spawn",
                console=False,
                argv0=str(args[0]),
                argv=_spawn_argv(args[1]),
            )
    except Exception:
        # Instrumentation must never break the audited call.
        pass
    finally:
        _reentry.active = False


def install_audit_trace() -> None:
    """Enable the audit trace, registering the interpreter hook on first use."""
    global _installed, _enabled
    _enabled = True
    if not _installed:
        sys.addaudithook(_audit)
        _installed = True


def _disable() -> None:
    """Stop emitting audit records (test teardown; the hook itself stays)."""
    global _enabled
    _enabled = False
