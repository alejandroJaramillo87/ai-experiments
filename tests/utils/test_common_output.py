"""Tests for common/output.py Printer — the shared style contract.

Printer backs every package CLI's status output (.styles/PYTHON_OUTPUT_STYLE.md),
so the contract under test is the wire format: plain "[TAG] message" lines on
standard error with a literal-space indent, the status vocabulary, quiet-mode
suppression, the TTY/NO_COLOR gating of ANSI color, and the structured file
record each status call emits through common/obs (roadmap OBS-3). The module is
loaded by absolute path (as in test_check_py_deps.py) so the test does not
depend on how the runner sets sys.path; its common.obs import resolves through
the repo root inserted by conftest.py.
"""

import contextlib
import importlib.util
import io
import json
import logging
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

RUN_ID = "R-20260101-000000-abc123"


class _TTYBuffer(io.StringIO):
    """StringIO that reports as a TTY, to drive Printer's color path.

    Printer reads sys.stderr.isatty() at construction; redirecting stderr to
    this buffer forces the color branch without touching a real terminal.
    """

    def isatty(self) -> bool:
        return True


@pytest.fixture(scope="module")
def Printer():  # noqa: N802 - fixture yields the class, named for it
    spec = importlib.util.spec_from_file_location("common_output", COMMON_OUTPUT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Printer


@pytest.fixture
def obs_log(tmp_path):
    """Configure and bind the obs core against a tmp file; yield the record path."""
    from common.obs import core

    previous = core._context.get()
    core.configure("training", run_id=RUN_ID, log_dir=str(tmp_path))
    core.bind_run("training", "test-cmd", RUN_ID)
    yield tmp_path / "training" / f"{RUN_ID}.jsonl"
    core._context.set(previous)
    logger = logging.getLogger("obs")
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(logging.NOTSET)


def test_plain_mode_no_tty(Printer, monkeypatch):
    # A non-TTY stderr (plain StringIO) forces plain output even with color
    # otherwise allowed: no ANSI, bare "[TAG] message".
    monkeypatch.delenv("NO_COLOR", raising=False)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
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
    with contextlib.redirect_stderr(buf):
        out = Printer()
        out.warn("nested", indent=2)
        out.info("deeper", indent=4)
    text = buf.getvalue()
    assert "  [WARN] nested\n" in text
    assert "    [INFO] deeper\n" in text


def test_each_status_method_emits_its_tag(Printer, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        out = Printer()
        for tag, (method, _) in TAGS.items():
            getattr(out, method)(f"msg-{tag}")
    text = buf.getvalue()
    for tag in TAGS:
        assert f"[{tag}] msg-{tag}\n" in text


def test_quiet_suppresses_status_and_line(Printer):
    # Quiet mode (e.g. --json) must emit no human output on either stream —
    # status methods and line() alike.
    err_buf = io.StringIO()
    out_buf = io.StringIO()
    with contextlib.redirect_stderr(err_buf), contextlib.redirect_stdout(out_buf):
        out = Printer(quiet=True)
        out.ok("hidden")
        out.error("hidden")
        out.line("hidden")
        out.line()
    assert err_buf.getvalue() == ""
    assert out_buf.getvalue() == ""


def test_no_color_env_suppresses_color_on_tty(Printer, monkeypatch):
    # NO_COLOR wins even for a TTY: color is gated off, output stays plain.
    monkeypatch.setenv("NO_COLOR", "1")
    buf = _TTYBuffer()
    with contextlib.redirect_stderr(buf):
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
    with contextlib.redirect_stderr(buf):
        out = Printer()
        for tag, (method, _) in TAGS.items():
            getattr(out, method)(f"msg-{tag}")
    text = buf.getvalue()
    for tag, (_, color) in TAGS.items():
        assert f"{color}[{tag}]{RESET} msg-{tag}\n" in text


def test_human_lines_never_touch_stdout(Printer, capsys):
    # The OBS-3 stream contract: stdout is reserved for --json payloads, so
    # every human line — status and plain — lands on stderr.
    out = Printer()
    out.ok("done")
    out.line("=== header ===")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[OK] done\n" in captured.err
    assert "=== header ===\n" in captured.err


def test_quiet_still_writes_file_record(Printer, obs_log, capsys):
    # Quiet suppresses only the human line; the structured record is the
    # durable log a --json run keeps.
    out = Printer(quiet=True)
    out.ok("recorded silently")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    record = json.loads(obs_log.read_text())
    assert record["status"] == "OK"
    assert record["level"] == "info"
    assert record["msg"] == "recorded silently"
    assert record["run_id"] == RUN_ID


def test_status_records_map_levels_and_keep_tags(Printer, obs_log, capsys):
    # ok/info/run/skip file as info, warn as warning, error as error; the
    # original tag survives in the status field.
    out = Printer()
    for _, (method, _) in TAGS.items():
        getattr(out, method)("msg")
    capsys.readouterr()
    records = [json.loads(line) for line in obs_log.read_text().splitlines()]
    assert [r["status"] for r in records] == list(TAGS)
    expected_levels = {
        "OK": "info",
        "WARN": "warning",
        "ERROR": "error",
        "INFO": "info",
        "RUN": "info",
        "SKIP": "info",
    }
    assert [r["level"] for r in records] == [expected_levels[t] for t in TAGS]


def test_blank_line_writes_no_record(Printer, obs_log, capsys):
    # Spacing lines carry no content; recording them would only add noise.
    out = Printer()
    out.line()
    out.line("a header")
    capsys.readouterr()
    records = [json.loads(line) for line in obs_log.read_text().splitlines()]
    assert [r["msg"] for r in records] == ["a header"]
