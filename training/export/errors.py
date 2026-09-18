"""The export pipeline's one failure type.

Every refusal in training/export/ raises ExportError with a message that
states the cause and, where one exists, the fix; training/cli.py::cmd_export
prints it as a single [ERROR] line and exits 1. It lives in its own module so
the merge, preflight and consolidation modules can share it without any of
them importing the others.
"""

from __future__ import annotations


class ExportError(RuntimeError):
    """Export refused or failed; the message says why and what to do."""
