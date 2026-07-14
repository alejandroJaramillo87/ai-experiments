"""W4 monitor tests: malformed records are skipped not fatal, two runs over
the same window are deterministic with a correct delta, the monitor refuses to
run while inference is up or a probe's model hash mismatches, and the
max-records cap and retention policy behave as documented."""

from __future__ import annotations

import datetime
import json
import os

import pytest

from interp.probes.monitor import (
    MonitorError,
    enforce_retention,
    parse_window,
    read_log_window,
    reconstruct_text,
    run_monitor,
)
from interp.probes.registry import RegistryError
from tests.interp.conftest import fake_capture

NOW = datetime.datetime(2026, 7, 14, 12, 0, tzinfo=datetime.UTC)


def log_record(i: int, text: str | None = "hello world") -> str:
    """One synthetic W3 record; text=None yields a record with nothing to replay."""
    stamp = (NOW - datetime.timedelta(hours=1, seconds=i)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    record = {
        "timestamp": stamp,
        "target_service": "llama-gpu",
        "model": "test-model",
        "messages": [{"role": "user", "content": f"{text} {i}"}] if text else None,
        "response_text": ("a neg response" if i % 2 else "a pos response")
        if text
        else None,
        "usage": {"completion_tokens": 3},
        "latency_s": 0.1,
    }
    return json.dumps(record)


@pytest.fixture
def log_dir(tmp_path):
    directory = tmp_path / "logs"
    directory.mkdir()
    day = NOW.strftime("%Y-%m-%d")
    lines = [log_record(i) for i in range(20)]
    lines.insert(5, "{not valid json")
    lines.insert(10, json.dumps({"timestamp": "garbage"}))
    lines.insert(15, log_record(99, text=None))  # no reconstructable text
    (directory / f"{day}.jsonl").write_text("\n".join(lines) + "\n")
    return str(directory)


class TestLogReading:
    def test_skips_malformed_without_crash(self, log_dir):
        since = NOW - datetime.timedelta(days=7)
        records, skipped = read_log_window(log_dir, since, NOW)
        assert len(records) == 20
        assert skipped["malformed_json"] == 1
        assert skipped["no_timestamp"] == 1
        assert skipped["no_text"] == 1

    def test_reconstruct_prompt_and_response(self):
        record = {
            "messages": [{"role": "user", "content": "question"}],
            "response_text": "answer",
        }
        assert reconstruct_text(record) == "question\nanswer"
        assert reconstruct_text({"prompt": "p", "response_text": "r"}) == "p\nr"
        assert reconstruct_text({}) is None

    def test_parse_window(self):
        assert parse_window("7d") == datetime.timedelta(days=7)
        assert parse_window("12h") == datetime.timedelta(hours=12)
        with pytest.raises(MonitorError):
            parse_window("soon")


class TestRetention:
    def test_removes_old_files_only(self, tmp_path):
        directory = tmp_path / "logs"
        directory.mkdir()
        (directory / "2026-05-01.jsonl").write_text("")
        (directory / "2026-07-10.jsonl").write_text("")
        removed = enforce_retention(str(directory), days=30, now=NOW)
        assert removed == ["2026-05-01.jsonl"]
        assert (directory / "2026-07-10.jsonl").exists()


class TestMonitor:
    def run(self, probe_dir, log_dir, tmp_path, **kwargs):
        return run_monitor(
            [probe_dir],
            "test-model",
            "modelhash-1",
            log_dir,
            monitor_root=str(tmp_path / "monitor"),
            window="7d",
            capture_fn=fake_capture,
            health_check=lambda: [],
            now=kwargs.pop("now", NOW),
            **kwargs,
        )

    def test_report_and_flagged(self, probe_dir, log_dir, tmp_path):
        report = self.run(probe_dir, log_dir, tmp_path)
        assert report["counts"]["replayed"] == 20
        assert report["counts"]["skipped"]["malformed_json"] == 1
        dist = report["probes"]["test-probe"]
        assert {"mean", "p50", "p95", "fraction_over_threshold"} <= set(dist)
        assert os.path.isfile(os.path.join(report["out_dir"], "flagged.jsonl"))

    def test_deterministic_and_delta(self, probe_dir, log_dir, tmp_path):
        first = self.run(probe_dir, log_dir, tmp_path)
        second = self.run(
            probe_dir, log_dir, tmp_path, now=NOW + datetime.timedelta(minutes=1)
        )
        d1, d2 = first["probes"]["test-probe"], second["probes"]["test-probe"]
        assert d1["mean"] == d2["mean"]  # same window, same scores
        assert d2["delta_vs_previous"]["mean"] == 0.0
        assert d2["delta_vs_previous"]["previous_run"] == first["run"]

    def test_refuses_while_serving(self, probe_dir, log_dir, tmp_path):
        with pytest.raises(MonitorError, match="services are up"):
            run_monitor(
                [probe_dir],
                "test-model",
                "modelhash-1",
                log_dir,
                monitor_root=str(tmp_path / "monitor"),
                capture_fn=fake_capture,
                health_check=lambda: ["http://127.0.0.1:8002/v1/health"],
            )

    def test_refuses_model_hash_mismatch(self, probe_dir, log_dir, tmp_path):
        with pytest.raises(RegistryError, match="model_hash"):
            run_monitor(
                [probe_dir],
                "other-model",
                "other-hash",
                log_dir,
                monitor_root=str(tmp_path / "monitor"),
                capture_fn=fake_capture,
                health_check=lambda: [],
                now=NOW,
            )

    def test_max_records_cap_reports_drops(self, probe_dir, log_dir, tmp_path):
        report = self.run(probe_dir, log_dir, tmp_path, max_records=5)
        assert report["counts"]["replayed"] == 5
        assert report["counts"]["dropped_over_cap"] == 15

    def test_empty_window_refuses(self, probe_dir, tmp_path):
        empty = tmp_path / "empty-logs"
        empty.mkdir()
        with pytest.raises(MonitorError, match="no replayable"):
            self.run(probe_dir, str(empty), tmp_path)
