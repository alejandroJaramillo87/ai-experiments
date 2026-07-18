"""Unit tests for the snapshot dashboard and report subcommand (OBS-11)."""

from __future__ import annotations

import json

from common.obs import dashboard
from common.reports import report_header

RUN_ID = "R-20260718-000000-abcdef"


def training_main(argv):
    from training.cli import main

    return main(argv)


def write_metrics(run_dir, records=None, run_id=RUN_ID, cli="training"):
    header = report_header("run_metrics")
    header.update(
        run_id=run_id,
        cli=cli,
        command="train-sft",
        tier="sft",
        model_id="test/model",
        recipe_hash="abc123",
        schedule={"type": "cosine", "warmup_steps": 5, "total_steps": 30},
    )
    if records is None:
        records = [
            {"step": 1, "ts": "2026-07-18T00:00:00+00:00", "loss": 1.25},
            {"step": 2, "ts": "2026-07-18T00:00:01+00:00", "loss": 1.125},
        ]
    lines = [json.dumps(header)] + [json.dumps(r) for r in records]
    (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    return header


def write_events(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def event(msg):
    return {
        "ts": "2026-07-18T00:00:00+00:00",
        "level": "info",
        "event": "log",
        "msg": msg,
    }


def test_report_writes_self_contained_html(tmp_path, obs_isolation, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_metrics(run_dir)
    write_events(
        tmp_path / "obs-logs" / "training" / f"{RUN_ID}.jsonl",
        [event("hello-event-marker")],
    )
    assert training_main(["report", "--run", str(run_dir)]) == 0
    out, err = capsys.readouterr()
    assert out == ""
    assert "Wrote" in err
    html = (run_dir / "run.html").read_text()
    assert "1.25" in html
    assert "hello-event-marker" in html
    assert "uPlot" in html
    assert "obsRender" in html
    assert 'src="http' not in html
    assert 'href="http' not in html


def test_report_missing_metrics_exits_1(tmp_path, obs_isolation, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert training_main(["report", "--run", str(run_dir)]) == 1
    assert "Metrics file not found" in capsys.readouterr().err
    assert not (run_dir / "run.html").exists()


def test_report_missing_run_dir_exits_1(tmp_path, obs_isolation, capsys):
    assert training_main(["report", "--run", str(tmp_path / "absent")]) == 1
    assert "Run directory not found" in capsys.readouterr().err


def test_report_events_override_wins(tmp_path, obs_isolation):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_metrics(run_dir)
    write_events(
        tmp_path / "obs-logs" / "training" / f"{RUN_ID}.jsonl",
        [event("default-location-marker")],
    )
    override = tmp_path / "elsewhere.jsonl"
    write_events(override, [event("override-marker")])
    argv = ["report", "--run", str(run_dir), "--events", str(override)]
    assert training_main(argv) == 0
    html = (run_dir / "run.html").read_text()
    assert "override-marker" in html
    assert "default-location-marker" not in html


def test_report_missing_events_warns_and_renders(tmp_path, obs_isolation, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_metrics(run_dir)
    assert training_main(["report", "--run", str(run_dir)]) == 0
    assert "Event file not found" in capsys.readouterr().err
    assert "1.25" in (run_dir / "run.html").read_text()


def test_inline_json_escapes_script_close(tmp_path, obs_isolation):
    assert dashboard.inline_json("</script>") == '"<\\/script>"'
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_metrics(run_dir)
    write_events(
        tmp_path / "obs-logs" / "training" / f"{RUN_ID}.jsonl",
        [event("bad </script><script>alert(1)</script> payload")],
    )
    assert training_main(["report", "--run", str(run_dir)]) == 0
    html = (run_dir / "run.html").read_text()
    # exactly the page's own three script blocks close; the payload cannot
    assert html.count("</script>") == 3


def test_read_metrics_skips_malformed_and_headerless(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1, "loss": 2.0}\nnot json\n[1, 2]\n')
    header, records, skipped = dashboard.read_metrics(path)
    assert header == {}
    assert records == [{"step": 1, "loss": 2.0}]
    assert skipped == 2


def test_default_events_path_requires_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("OBS_LOG_DIR", str(tmp_path))
    assert dashboard.default_events_path({}) is None
    assert dashboard.default_events_path({"run_id": "R-1"}) is None
    path = dashboard.default_events_path({"run_id": "R-1", "cli": "training"})
    assert path == tmp_path / "training" / "R-1.jsonl"


def test_snapshot_contains_all_chart_groups():
    html = dashboard.build_snapshot({}, [], [])
    for title in (
        "Loss",
        "Learning rate",
        "Throughput (tokens/s)",
        "Device memory (GB)",
    ):
        assert title in html


def test_serve_flag_dispatches_to_server(tmp_path, obs_isolation, monkeypatch):
    from training.commands import report as report_cmd

    calls = []

    def fake_run(host, port, run_dir, events_path, shell, out):
        calls.append((host, port, run_dir, events_path))
        return 0

    monkeypatch.setattr(report_cmd.obs_serve, "run", fake_run)
    run_dir = tmp_path / "run"
    run_dir.mkdir()  # no metrics.jsonl: serve mode must not require it
    argv = ["report", "--run", str(run_dir), "--serve", "--port", "0"]
    assert training_main(argv) == 0
    assert calls == [("127.0.0.1", 0, run_dir, None)]
    assert not (run_dir / "run.html").exists()


def test_live_shell_has_no_inlined_data():
    shell = dashboard.build_live_shell()
    assert "obsRender" in shell
    assert "/data/metrics" in shell
    assert "/data/events" in shell
    assert "pollMs: 2000" in shell
    assert 'src="http' not in shell
    assert 'href="http' not in shell
