"""Unit tests for the run catalog and runs subcommand (roadmap OBS-9)."""

from __future__ import annotations

import argparse
import json
import logging

import pytest

from common.obs import catalog, core
from common.obs.dispatch import dispatch


@pytest.fixture(autouse=True)
def clean_obs_state(monkeypatch, tmp_path):
    """Isolate env, context, and handler state so tests never leak."""
    monkeypatch.delenv("OBS_LOG_LEVEL", raising=False)
    monkeypatch.delenv("OBS_RETENTION_DAYS", raising=False)
    monkeypatch.setenv("OBS_LOG_DIR", str(tmp_path / "obs-logs"))
    token = core._context.set(core._RunContext())
    yield
    core._context.reset(token)
    logger = logging.getLogger("obs")
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(logging.NOTSET)


def _ok_handler(args):
    return 0


def _fail_handler(args):
    raise RuntimeError("boom")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toy")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ok").set_defaults(func=_ok_handler)
    sub.add_parser("fail").set_defaults(func=_fail_handler)
    return parser


def read_catalog(tmp_path, cli="toycli"):
    return catalog.read_runs(tmp_path / "obs-logs" / cli / "runs.jsonl")


def test_completed_run_appends_one_record(tmp_path):
    assert dispatch("toycli", build_parser, ["ok"]) == 0
    records = read_catalog(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["command"] == "ok"
    assert record["exit_code"] == 0
    assert record["duration_s"] >= 0
    assert record["run_dir"] is None
    assert record["timestamp"]
    # the summary points at the run's event file
    event_files = (tmp_path / "obs-logs" / "toycli").glob("R-*.jsonl")
    assert [f.stem for f in event_files] == [record["run_id"]]


def test_failed_run_is_cataloged_with_its_exit_code(tmp_path):
    with pytest.raises(RuntimeError):
        dispatch("toycli", build_parser, ["fail"])
    records = read_catalog(tmp_path)
    assert len(records) == 1
    assert records[0]["exit_code"] == 1


def test_unwritable_log_root_appends_nothing(tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setenv("OBS_LOG_DIR", str(blocker / "logs"))
    assert dispatch("toycli", build_parser, ["ok"]) == 0


def test_malformed_catalog_line_is_skipped(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text('{"run_id": "R-1"}\nnot json\n[1, 2]\n{"run_id": "R-2"}\n')
    assert [r["run_id"] for r in catalog.read_runs(path)] == ["R-1", "R-2"]


def test_missing_catalog_reads_empty(tmp_path):
    assert catalog.read_runs(tmp_path / "absent.jsonl") == []


# --- the runs subcommand through the real training CLI ---


def training_main(argv):
    from training.cli import main

    return main(argv)


def seed_runs(tmp_path):
    """Record one success and one failure under the training catalog."""
    assert dispatch("training", build_parser, ["ok"]) == 0
    with pytest.raises(RuntimeError):
        dispatch("training", build_parser, ["fail"])


def test_runs_json_lists_runs_including_failed(tmp_path, capsys):
    seed_runs(tmp_path)
    assert training_main(["runs", "--json"]) == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert [r["command"] for r in payload] == ["ok", "fail"]
    assert payload[1]["exit_code"] == 1
    # machine mode: stdout is only JSON, no human table anywhere
    assert "FINISHED" not in err


def test_default_view_hides_runs_entries_until_all(tmp_path, capsys):
    seed_runs(tmp_path)
    assert training_main(["runs", "--json"]) == 0
    capsys.readouterr()
    # the browse above cataloged itself, but the default view hides it
    assert training_main(["runs", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["command"] for r in payload] == ["ok", "fail"]
    assert training_main(["runs", "--json", "--all"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["command"] for r in payload] == ["ok", "fail", "runs", "runs"]


def test_failed_filter_keeps_only_nonzero_exits(tmp_path, capsys):
    seed_runs(tmp_path)
    assert training_main(["runs", "--failed", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["command"] for r in payload] == ["fail"]


def test_limit_bounds_the_selection(tmp_path, capsys):
    for _ in range(3):
        assert dispatch("training", build_parser, ["ok"]) == 0
    assert training_main(["runs", "-n", "2", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def write_catalog_rows(tmp_path, rows):
    path = tmp_path / "obs-logs" / "training" / "runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_table_survives_non_numeric_duration(tmp_path, capsys):
    write_catalog_rows(
        tmp_path,
        [
            {
                "run_id": "R-1",
                "command": "x",
                "timestamp": "2026-07-18T00:00:00+00:00",
                "duration_s": "fast",
                "exit_code": 0,
            }
        ],
    )
    assert training_main(["runs"]) == 0
    err = capsys.readouterr().err
    assert "R-1" in err
    assert "?" in err


def test_failed_filter_ignores_rows_without_integer_exit(tmp_path, capsys):
    write_catalog_rows(
        tmp_path,
        [
            {"run_id": "R-1", "command": "x"},
            {"run_id": "R-2", "command": "y", "exit_code": None},
            {"run_id": "R-3", "command": "z", "exit_code": "0"},
            {"run_id": "R-4", "command": "w", "exit_code": 1},
        ],
    )
    assert training_main(["runs", "--failed", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["run_id"] for r in payload] == ["R-4"]


def test_limit_zero_shows_all(tmp_path, capsys):
    for _ in range(3):
        assert dispatch("training", build_parser, ["ok"]) == 0
    assert training_main(["runs", "-n", "0", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3


def test_table_goes_to_stderr_not_stdout(tmp_path, capsys):
    seed_runs(tmp_path)
    assert training_main(["runs"]) == 0
    out, err = capsys.readouterr()
    assert out == ""
    assert "FINISHED" in err
    assert "ok" in err
    assert "fail" in err


def test_empty_catalog_reports_no_runs(tmp_path, capsys):
    assert training_main(["runs"]) == 0
    out, err = capsys.readouterr()
    assert out == ""
    assert "No recorded runs" in err


def test_interp_registration_reads_its_own_catalog(tmp_path, capsys):
    from interp.cli import main as interp_main

    assert dispatch("interp", build_parser, ["ok"]) == 0
    assert interp_main(["runs", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["command"] for r in payload] == ["ok"]
