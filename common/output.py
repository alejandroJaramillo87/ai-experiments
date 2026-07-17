"""Styled CLI output per .styles/PYTHON_OUTPUT_STYLE.md, shared by all package CLIs.

A front-end over the observability logging core (common/obs, roadmap
OBS-3): every status call renders the human line on standard error and
emits a structured file record through the core. Standard output stays
reserved for --json payloads.
"""

from __future__ import annotations

import os
import sys

from common.obs import core as obs_core

# File-record level for each status tag; the tag itself survives in the
# record's status field.
_LEVEL_FOR_TAG = {
    "OK": "info",
    "WARN": "warning",
    "ERROR": "error",
    "INFO": "info",
    "RUN": "info",
    "SKIP": "info",
}


class Printer:
    """Styled status output. Colorized for a TTY, plain otherwise.

    Human lines go to stderr; stdout is reserved for machine output. In
    quiet mode (e.g. --json) the human line is suppressed but the
    structured record is still written.
    """

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        use_color = not quiet and not os.environ.get("NO_COLOR") and sys.stderr.isatty()
        if use_color:
            self._c = {
                "OK": "\033[0;32m",
                "WARN": "\033[1;33m",
                "ERROR": "\033[0;31m",
                "INFO": "\033[0;36m",
                "RUN": "\033[0;34m",
                "SKIP": "\033[0;34m",
            }
            self._nc = "\033[0m"
        else:
            self._c = {}
            self._nc = ""

    def _status(self, tag: str, message: str, indent: int = 0) -> None:
        obs_core.emit(
            _LEVEL_FOR_TAG.get(tag, "info"),
            message,
            console=False,
            status=tag,
        )
        if self.quiet:
            return
        color = self._c.get(tag, "")
        pad = " " * indent
        print(f"{pad}{color}[{tag}]{self._nc} {message}", file=sys.stderr)

    def ok(self, m: str, indent: int = 0) -> None:
        self._status("OK", m, indent)

    def warn(self, m: str, indent: int = 0) -> None:
        self._status("WARN", m, indent)

    def error(self, m: str, indent: int = 0) -> None:
        self._status("ERROR", m, indent)

    def info(self, m: str, indent: int = 0) -> None:
        self._status("INFO", m, indent)

    def run(self, m: str, indent: int = 0) -> None:
        self._status("RUN", m, indent)

    def skip(self, m: str, indent: int = 0) -> None:
        self._status("SKIP", m, indent)

    def line(self, message: str = "") -> None:
        """Plain line (headers, blank lines) — human line suppressed when quiet.

        Blank spacing lines produce no file record; they carry no content.
        """
        if message:
            obs_core.emit("info", message, console=False)
        if not self.quiet:
            print(message, file=sys.stderr)
