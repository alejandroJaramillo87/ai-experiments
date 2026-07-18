"""Cross-module integration tests for the observability stack (phases A-D).

The unit suites synthesize each seam; these run the real chain instead: the
traced dispatch drives a purpose-built command that opens a real jsonl
Tracker, and the assertions read the artifacts the way the downstream
consumers do — validate_report, the catalog join, the report renderer, and
the live server's lazy events resolution.
"""

from __future__ import annotations

import argparse
import json
import threading
import urllib.request

from common.obs import core, serve
from common.obs.catalog import catalog_path, read_runs
from common.obs.dispatch import dispatch
from common.reports import validate_report
from training.recipes.schema import TrackingConfig
from training.runtime.tracking import Tracker

LOG_MARKER = "integration-log-marker"


def _smoke_handler(run_dir):
    """A miniature training command: the seams loop.py exercises, no torch."""

    def handler(args):
        core.set_run_dir(run_dir)
        tracker = Tracker(
            TrackingConfig(tensorboard=False),
            "smoke",
            str(run_dir),
            header={"tier": "sft", "model_id": "test/model", "recipe_hash": "cafe01"},
        )
        with core.phase("train"):
            for step in range(1, 4):
                tracker.log({"loss": 2.0 / step}, step)
            tracker.log({"checkpoint_saved": 1}, 3)
        core.get_logger("loop").info(LOG_MARKER)
        tracker.close()
        return 0

    return handler


def run_smoke(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def build_parser():
        parser = argparse.ArgumentParser(prog="training")
        sub = parser.add_subparsers(dest="command", required=True)
        p = sub.add_parser("smoke-train")
        p.set_defaults(func=_smoke_handler(run_dir))
        return parser

    assert dispatch("training", build_parser, ["smoke-train"]) == 0
    return run_dir


def test_dispatch_produces_joined_artifacts(tmp_path, obs_isolation, capsys):
    run_dir = run_smoke(tmp_path)

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    header = records[0]
    validate_report(header)
    assert header["report"] == "run_metrics"
    assert header["cli"] == "training"
    assert header["command"] == "smoke-train"
    assert header["tier"] == "sft"
    run_id = header["run_id"]
    assert run_id
    assert [r["step"] for r in records[1:]] == [1, 2, 3, 3]
    assert records[-1]["checkpoint_saved"] == 1

    # events join: the file named by the header identity holds this run
    events = tmp_path / "obs-logs" / "training" / f"{run_id}.jsonl"
    parsed = [json.loads(line) for line in events.read_text().splitlines()]
    names = [e["event"] for e in parsed]
    assert names[0] == "cmd.start"
    assert {"run.context", "run.dir", "phase.start", "phase.end"} <= set(names)
    assert names[-1] == "cmd.end"
    assert parsed[-1]["exit_code"] == 0
    assert all(e["run_id"] == run_id for e in parsed)

    # catalog join: the same run id points back at the run directory
    row = read_runs(catalog_path("training"))[-1]
    assert row["run_id"] == run_id
    assert row["run_dir"] == str(run_dir)
    assert row["exit_code"] == 0


def test_report_renders_real_run_artifacts(tmp_path, obs_isolation, capsys):
    run_dir = run_smoke(tmp_path)
    from training.cli import main

    assert main(["report", "--run", str(run_dir)]) == 0
    html = (run_dir / "run.html").read_text()
    assert LOG_MARKER in html
    assert '"loss"' in html
    assert '"checkpoint_saved"' in html


def test_live_server_streams_real_run(tmp_path, obs_isolation, capsys):
    run_dir = run_smoke(tmp_path)
    srv = serve.DashboardServer(("127.0.0.1", 0), run_dir, None, "<html></html>")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]

        def get_json(path):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as resp:
                return json.loads(resp.read())

        payload = get_json("/data/metrics?offset=0")
        assert json.loads(payload["lines"][0])["report"] == "run_metrics"
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": 4, "loss": 0.5}) + "\n")
        more = get_json(f"/data/metrics?offset={payload['offset']}")
        assert [json.loads(li)["step"] for li in more["lines"]] == [4]

        # events resolve lazily from the real header written by the run
        events = get_json("/data/events?offset=0")
        parsed = [json.loads(li) for li in events["lines"]]
        assert any(e.get("msg") == LOG_MARKER for e in parsed)
        assert get_json(f"/data/events?offset={events['offset']}")["lines"] == []
    finally:
        srv.shutdown()
        srv.server_close()
