"""Backward-compatible re-export; the shared Printer lives in common/output.py."""

from __future__ import annotations

from common.output import Printer

__all__ = ["Printer"]
