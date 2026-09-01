"""The Printer contract, tested inside the project so it survives extraction.

The parent repository additionally compares this Printer byte-for-byte against
its canonical one, but that test stays behind when this directory is lifted out.
These are the properties .styles/PYTHON_OUTPUT_STYLE.md actually specifies, so
they are the ones worth carrying along.
"""

from __future__ import annotations

import pytest

from logtriage.output import Printer

TAGS = ["ok", "warn", "error", "info", "run", "skip"]


@pytest.mark.parametrize("method", TAGS)
def test_status_goes_to_stderr_not_stdout(capsys, method: str) -> None:
    getattr(Printer(), method)("Subject: detail")
    captured = capsys.readouterr()
    assert captured.out == "", "stdout is reserved for --json payloads"
    assert captured.err == f"[{method.upper()}] Subject: detail\n"


def test_indent_is_two_spaces_per_level(capsys) -> None:
    out = Printer()
    out.ok("top")
    out.ok("nested", indent=2)
    assert capsys.readouterr().err == "[OK] top\n  [OK] nested\n"


@pytest.mark.parametrize("method", [*TAGS, "line"])
def test_quiet_suppresses_every_human_line(capsys, method: str) -> None:
    getattr(Printer(quiet=True), method)("anything")
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_line_emits_bare_text_and_blank_lines(capsys) -> None:
    out = Printer()
    out.line("=== header ===")
    out.line()
    assert capsys.readouterr().err == "=== header ===\n\n"


def test_no_color_when_stderr_is_not_a_tty(capsys) -> None:
    """Escape codes must never reach a pipe or a log file."""
    out = Printer()
    out.error("Store unreachable")
    assert "\033" not in capsys.readouterr().err
    assert out._c == {}


def test_no_color_env_disables_color_even_on_a_tty(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
    assert Printer()._c == {}


def test_quiet_disables_color_even_on_a_tty(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
    assert Printer(quiet=True)._c == {}
