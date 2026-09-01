"""Styled CLI output: bracketed status tags on stderr, stdout reserved for --json.

A deliberate copy, not an import. The canonical implementation is the parent
repository's common/output.py, and .styles/PYTHON_OUTPUT_STYLE.md says to import
it rather than copy it -- but this project must survive being lifted out of that
repository on its own, and an import of common/ would not survive. The copy is
policed instead of trusted: tests/logtriage/test_conformance.py in the parent
repository drives both Printers through the same call sequence and fails if the
stderr bytes differ. That test does not travel on extraction, which is correct;
it exists to keep the two in step while they are neighbours.

What is dropped rather than copied is the structured file record: common/obs is
parent infrastructure with its own log root and run-id machinery. The seam is
kept as _emit below so a run log can be added here later without touching every
call site.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

# File-record level for each status tag, retained so the emit hook below has the
# same shape as the parent's and the two can be compared line for line.
_LEVEL_FOR_TAG = {
    "OK": "info",
    "WARN": "warning",
    "ERROR": "error",
    "INFO": "info",
    "RUN": "info",
    "SKIP": "info",
}


def _no_record(level: str, message: str, status: str | None = None) -> None:
    """Structured-record sink. A no-op until this project grows a run log."""


# Module-level rather than a constructor argument: the parent's Printer emits
# unconditionally, so keeping the hook out of the signature is what lets the
# conformance test construct both the same way.
_emit: Callable[[str, str, str | None], None] = _no_record


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
        _emit(_LEVEL_FOR_TAG.get(tag, "info"), message, tag)
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
        """Plain line (headers, blank lines) -- human line suppressed when quiet.

        Blank spacing lines produce no file record; they carry no content.
        """
        if message:
            _emit("info", message, None)
        if not self.quiet:
            print(message, file=sys.stderr)
