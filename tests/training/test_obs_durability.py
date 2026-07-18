"""SIGKILL durability test for the jsonl metrics sink (roadmap OBS-10).

The sink's claim is that a killed run keeps every completed step: the file
is opened line-buffered and each record is one write. A subprocess driver
logs steps at full speed and the test SIGKILLs it mid-stream — no flush,
no atexit, no signal handler runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from common.reports import validate_report

DRIVER = Path(__file__).with_name("obs_crash_driver.py")


def kill_driver_mid_write(run_dir):
    proc = subprocess.Popen(
        [sys.executable, str(DRIVER), str(run_dir)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "READY"
        metrics = run_dir / "metrics.jsonl"
        deadline = time.monotonic() + 10
        # enough volume that the kill lands inside an active write stream
        while metrics.stat().st_size < 65536:
            assert time.monotonic() < deadline, "driver wrote too slowly"
            time.sleep(0.05)
    finally:
        proc.kill()
        proc.wait(timeout=10)
    return metrics


def test_sigkill_keeps_every_completed_line(tmp_path):
    metrics = kill_driver_mid_write(tmp_path)
    lines = metrics.read_bytes().split(b"\n")
    # only the final element may be torn (or empty, if the kill landed
    # between lines); everything before it must parse
    records = [json.loads(line) for line in lines[:-1]]
    header = records[0]
    validate_report(header)
    assert header["run_id"] == "R-20260718-000000-c8a54d"
    assert header["cli"] == "training"
    steps = [r["step"] for r in records[1:]]
    assert steps == list(range(len(steps)))  # sequential: no lost middles


def test_report_renders_from_a_killed_run(tmp_path, obs_isolation, capsys):
    kill_driver_mid_write(tmp_path)
    from training.cli import main

    assert main(["report", "--run", str(tmp_path)]) == 0
    html = (tmp_path / "run.html").read_text()
    assert '"tokens_per_s"' in html
