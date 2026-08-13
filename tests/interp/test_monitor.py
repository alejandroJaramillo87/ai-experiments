"""W4 monitor tests: the report is built from the scores the serving process
recorded, malformed records are skipped rather than fatal, two runs over the
same window are deterministic with a correct delta, a report is never deltaed
against one produced a different way, and the max-records cap and retention
policy behave as documented."""

from __future__ import annotations

import datetime
import json
import os

import pytest

from interp.probes.monitor import (
    MonitorError,
    enforce_retention,
    parse_window,
    probe_score,
    read_log_window,
    reconstruct_text,
    run_monitor,
)

NOW = datetime.datetime(2026, 7, 14, 12, 0, tzinfo=datetime.UTC)


def log_record(
    i: int,
    text: str | None = "hello world",
    target_service: str = "harness",
    probe: str | None = "test-probe",
    score: float | None = None,
) -> str:
    """One synthetic served-request record.

    `text=None` yields a record with nothing to identify; `probe=None` yields
    one served without a probe loaded. Both are ordinary and both are skipped.
    """
    stamp = (NOW - datetime.timedelta(hours=1, seconds=i)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    record = {
        "timestamp": stamp,
        "target_service": target_service,
        "model": "test-model",
        "messages": [{"role": "user", "content": f"{text} {i}"}] if text else None,
        "response_text": ("a neg response" if i % 2 else "a pos response")
        if text
        else None,
        "usage": {"completion_tokens": 3},
        "latency_s": 0.1,
        "status_code": 200,
        "error": None,
    }
    if probe:
        record["probe"] = {
            "name": probe,
            "layer": 4,
            "site": "layer_output",
            "threshold": 0.5,
            "scored": 10,
            "position_first": 0,
            # Deterministic and spread across the threshold, so the
            # distribution and the flagged count are both non-trivial.
            "max_score": score if score is not None else round(0.05 * (i % 20), 4),
            "flags": [],
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
    lines.insert(15, log_record(99, text=None))  # nothing to identify it by
    lines.insert(18, log_record(98, probe=None))  # served without a probe
    (directory / f"{day}.jsonl").write_text("\n".join(lines) + "\n")
    return str(directory)


class TestProbeScore:
    """The record is the only source of a score, so what counts as one matters."""

    def test_reads_name_score_and_threshold(self):
        record = json.loads(log_record(3))
        assert probe_score(record) == ("test-probe", 0.15, 0.5)

    def test_a_record_served_without_a_probe_has_no_score(self):
        assert probe_score(json.loads(log_record(3, probe=None))) is None

    def test_a_request_that_scored_nothing_has_no_score(self):
        # max_score is null when the request scored no positions at the
        # probe's layer. Reading that as 0.0 would drag the distribution down
        # with a request the probe never saw.
        record = json.loads(log_record(3))
        record["probe"]["max_score"] = None
        assert probe_score(record) is None

    def test_a_malformed_probe_block_has_no_score(self):
        record = json.loads(log_record(3))
        record["probe"] = "not a dict"
        assert probe_score(record) is None


class TestLogReading:
    def test_skips_malformed_without_crash(self, log_dir):
        since = NOW - datetime.timedelta(days=7)
        records, skipped = read_log_window(log_dir, since, NOW)
        assert len(records) == 20
        assert skipped["malformed_json"] == 1
        assert skipped["no_timestamp"] == 1
        assert skipped["no_text"] == 1
        assert skipped["no_scores"] == 1

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
    def run(self, log_dir, tmp_path, **kwargs):
        return run_monitor(
            log_dir,
            monitor_root=str(tmp_path / "monitor"),
            window="7d",
            now=kwargs.pop("now", NOW),
            **kwargs,
        )

    def test_report_and_flagged(self, log_dir, tmp_path):
        report = self.run(log_dir, tmp_path)
        assert report["counts"]["reported"] == 20
        assert report["counts"]["skipped"]["malformed_json"] == 1
        dist = report["probes"]["test-probe"]
        assert {"mean", "p50", "p95", "fraction_over_threshold"} <= set(dist)
        assert dist["threshold"] == 0.5
        assert os.path.isfile(os.path.join(report["out_dir"], "flagged.jsonl"))

    def test_no_model_is_loaded(self, log_dir, tmp_path):
        # The signature is the contract: there is no replay model and no
        # health check, because nothing here runs a model. A regression that
        # reintroduced either would not typecheck against this call.
        report = self.run(log_dir, tmp_path)
        assert report["scored"] == "in_band"
        assert "no replay" in report["note"]
        assert "replay_model" not in report

    def test_report_records_target_services(self, log_dir, tmp_path):
        report = self.run(log_dir, tmp_path)
        assert report["target_services"] == ["harness"]

    def test_scores_come_from_the_records(self, log_dir, tmp_path):
        # 20 records with max_score 0.00..0.95 in 0.05 steps, threshold 0.5:
        # nine are strictly above it. Pinned because this is the whole claim
        # of the change -- the number is read, not recomputed.
        report = self.run(log_dir, tmp_path)
        dist = report["probes"]["test-probe"]
        assert dist["count"] == 20
        assert dist["fraction_over_threshold"] == 0.45
        assert report["counts"]["flagged"] == 9

    def test_deterministic_and_delta(self, log_dir, tmp_path):
        first = self.run(log_dir, tmp_path)
        second = self.run(log_dir, tmp_path, now=NOW + datetime.timedelta(minutes=1))
        d1, d2 = first["probes"]["test-probe"], second["probes"]["test-probe"]
        assert d1["mean"] == d2["mean"]  # same window, same scores
        assert d2["delta_vs_previous"]["mean"] == 0.0
        assert d2["delta_vs_previous"]["previous_run"] == first["run"]

    def test_no_delta_against_a_report_scored_another_way(self, log_dir, tmp_path):
        # A report built by replaying a second model and one built from the
        # serving process's own scores measure different things. Subtracting
        # them would present a change of method as drift in the model.
        monitor_root = tmp_path / "monitor"
        stale = monitor_root / "2026-07-14T110000Z"
        stale.mkdir(parents=True)
        (stale / "report.json").write_text(
            json.dumps(
                {
                    "run": "2026-07-14T110000Z",
                    "scored": "replay",
                    "probes": {"test-probe": {"mean": 0.0, "fraction_over_threshold": 0.0}},
                }
            )
        )
        report = self.run(log_dir, tmp_path)
        assert "delta_vs_previous" not in report["probes"]["test-probe"]

    def test_probe_filter_reports_what_it_dropped(self, tmp_path):
        directory = tmp_path / "logs"
        directory.mkdir()
        day = NOW.strftime("%Y-%m-%d")
        lines = [log_record(i, probe="wanted") for i in range(4)]
        lines += [log_record(i, probe="other") for i in range(4, 7)]
        (directory / f"{day}.jsonl").write_text("\n".join(lines) + "\n")
        report = self.run(str(directory), tmp_path, probes=["wanted"])
        assert set(report["probes"]) == {"wanted"}
        assert report["counts"]["skipped"]["other_probe"] == 3

    def test_every_probe_in_the_window_is_reported_by_default(self, tmp_path):
        directory = tmp_path / "logs"
        directory.mkdir()
        day = NOW.strftime("%Y-%m-%d")
        lines = [log_record(i, probe="one") for i in range(3)]
        lines += [log_record(i, probe="two") for i in range(3, 6)]
        (directory / f"{day}.jsonl").write_text("\n".join(lines) + "\n")
        report = self.run(str(directory), tmp_path)
        assert set(report["probes"]) == {"one", "two"}

    def test_disagreeing_thresholds_are_reported_not_averaged(self, tmp_path):
        # A probe re-exported mid-window changes the threshold its records
        # were scored against. The maximum is the conservative reading, and
        # the disagreement is stated rather than hidden.
        directory = tmp_path / "logs"
        directory.mkdir()
        day = NOW.strftime("%Y-%m-%d")
        lines = []
        for i in range(4):
            record = json.loads(log_record(i))
            record["probe"]["threshold"] = 0.5 if i < 2 else 0.8
            lines.append(json.dumps(record))
        (directory / f"{day}.jsonl").write_text("\n".join(lines) + "\n")
        report = self.run(str(directory), tmp_path)
        dist = report["probes"]["test-probe"]
        assert dist["threshold"] == 0.8
        assert dist["thresholds_observed"] == [0.5, 0.8]

    def test_max_records_cap_reports_drops(self, log_dir, tmp_path):
        report = self.run(log_dir, tmp_path, max_records=5)
        assert report["counts"]["reported"] == 5
        assert report["counts"]["dropped_over_cap"] == 15

    def test_empty_window_refuses(self, tmp_path):
        empty = tmp_path / "empty-logs"
        empty.mkdir()
        with pytest.raises(MonitorError, match="no scored records"):
            self.run(str(empty), tmp_path)

    def test_window_longer_than_retention_reads_before_pruning(self, tmp_path):
        # A --window longer than the retention period must read what is
        # still on disk and prune afterwards; retention-first deleted the
        # window's tail before reporting and then refused an "empty" window
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        old_day = NOW - datetime.timedelta(days=40)
        old_name = f"{old_day.strftime('%Y-%m-%d')}.jsonl"
        record = json.loads(log_record(1))
        record["timestamp"] = old_day.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        (log_dir / old_name).write_text(json.dumps(record) + "\n")
        report = run_monitor(
            str(log_dir),
            monitor_root=str(tmp_path / "monitor"),
            window="60d",
            now=NOW,
        )
        assert report["counts"]["reported"] == 1
        assert report["counts"]["retention_removed_files"] == [old_name]
        assert not (log_dir / old_name).exists()
