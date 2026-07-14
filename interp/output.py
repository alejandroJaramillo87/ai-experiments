"""Styled CLI output per .styles/PYTHON_OUTPUT_STYLE.md, shared by interp CLIs."""

from __future__ import annotations

import os
import sys


class Printer:
    """Styled status output. Colorized for a TTY, plain otherwise.

    In quiet mode (e.g. --json) all human output is suppressed so stdout stays
    machine-parseable.
    """

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        use_color = not quiet and not os.environ.get("NO_COLOR") and sys.stdout.isatty()
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
        if self.quiet:
            return
        color = self._c.get(tag, "")
        pad = " " * indent
        print(f"{pad}{color}[{tag}]{self._nc} {message}")

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
        """Plain line (headers, blank lines) — still suppressed when quiet."""
        if not self.quiet:
            print(message)
