"""Tests for common/output.py Printer — the shared style contract.

Printer backs every package CLI's status output (.styles/PYTHON_OUTPUT_STYLE.md),
so the contract under test is the wire format: plain "[TAG] message" lines with a
literal-space indent, the status vocabulary, quiet-mode suppression, and the
TTY/NO_COLOR gating of ANSI color. The module is loaded by absolute path (as in
test_check_py_deps.py) so the test does not depend on how the runner sets sys.path;
output.py imports only os and sys, so it loads standalone.
"""

import contextlib
import importlib.util
import io
from pathlib import Path

import pytest

COMMON_OUTPUT = Path(__file__).resolve().parent.parent.parent / "common" / "output.py"

RESET = "\033[0m"
# Tag -> (Printer method name, ANSI color prefix) as defined in output.py.
TAGS = {
    "OK": ("ok", "\033[0;32m"),
    "WARN": ("warn", "\033[1;33m"),
    "ERROR": ("error", "\033[0;31m"),
    "INFO": ("info", "\033[0;36m"),
    "RUN": ("run", "\033[0;34m"),
    "SKIP": ("skip", "\033[0;34m"),
}


class _TTYBuffer(io.StringIO):
    """StringIO that reports as a TTY, to drive Printer's color path.

    Printer reads sys.stdout.isatty() at construction; redirecting stdout to this
    buffer forces the color branch without touching a real terminal.
    """

    def isatty(self) -> bool:
        return True


@pytest.fixture(scope="module")
def Printer():  # noqa: N802 - fixture yields the class, named for it
    spec = importlib.util.spec_from_file_location("common_output", COMMON_OUTPUT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Printer


def test_plain_mode_no_tty(Printer, monkeypatch):
    # A non-TTY stdout (plain StringIO) forces plain output even with color
    # otherwise allowed: no ANSI, bare "[TAG] message".
    monkeypatch.delenv("NO_COLOR", raising=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = Printer()
        out.ok("done")
    text = buf.getvalue()
    assert "\033[" not in text
    assert text == "[OK] done\n"


def test_indent_is_literal_space_count(Printer, monkeypatch):
    # indent is `" " * indent` — a raw leading-space count, not a level
    # multiplier. indent=2 yields the two-space style-guide indent; indent=4
    # yields four spaces.
    monkeypatch.delenv("NO_COLOR", raising=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = Printer()
        out.warn("nested", indent=2)
        out.info("deeper", indent=4)
    text = buf.getvalue()
    assert "  [WARN] nested\n" in text
    assert "    [INFO] deeper\n" in text


def test_each_status_method_emits_its_tag(Printer, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = Printer()
        for tag, (method, _) in TAGS.items():
            getattr(out, method)(f"msg-{tag}")
    text = buf.getvalue()
    for tag in TAGS:
        assert f"[{tag}] msg-{tag}\n" in text


def test_quiet_suppresses_status_and_line(Printer):
    # Quiet mode (e.g. --json) must emit nothing at all so stdout stays
    # machine-parseable — status methods and line() alike.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = Printer(quiet=True)
        out.ok("hidden")
        out.error("hidden")
        out.line("hidden")
        out.line()
    assert buf.getvalue() == ""


def test_no_color_env_suppresses_color_on_tty(Printer, monkeypatch):
    # NO_COLOR wins even for a TTY: color is gated off, output stays plain.
    monkeypatch.setenv("NO_COLOR", "1")
    buf = _TTYBuffer()
    with contextlib.redirect_stdout(buf):
        out = Printer()
        out.ok("done")
    text = buf.getvalue()
    assert "\033[" not in text
    assert text == "[OK] done\n"


def test_color_mode_wraps_tags(Printer, monkeypatch):
    # Forced TTY, NO_COLOR unset: each tag is wrapped in its ANSI prefix and
    # closed with the reset code.
    monkeypatch.delenv("NO_COLOR", raising=False)
    buf = _TTYBuffer()
    with contextlib.redirect_stdout(buf):
        out = Printer()
        for tag, (method, _) in TAGS.items():
            getattr(out, method)(f"msg-{tag}")
    text = buf.getvalue()
    for tag, (_, color) in TAGS.items():
        assert f"{color}[{tag}]{RESET} msg-{tag}\n" in text
