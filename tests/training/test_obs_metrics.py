"""Unit tests for the metrics.jsonl tracker sink (roadmap OBS-10)."""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from common.obs import bind_run
from common.reports import validate_report
from training.recipes.schema import TrackingConfig
from training.runtime.tracking import Tracker
from training.tune.sft_callbacks import build_callbacks

HEADER = {
    "tier": "sft",
    "model_id": "test/model",
    "recipe_hash": "abc123",
    "schedule": {"type": "cosine", "warmup_steps": 10, "total_steps": 100},
}


def make_tracker(run_dir, header=None):
    """A Tracker with only the jsonl sink (external backends disabled)."""
    cfg = TrackingConfig(tensorboard=False, wandb=False, mlflow=False)
    return Tracker(cfg, "run", str(run_dir), header=header)


def read_records(run_dir):
    path = run_dir / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def jsonl_sink(tracker):
    return next(sink for name, sink in tracker._sinks if name == "jsonl")


def test_header_first_record_validates(tmp_path):
    tracker = make_tracker(tmp_path, header=HEADER)
    tracker.log({"loss": 1.0}, step=1)
    tracker.close()
    header = read_records(tmp_path)[0]
    validate_report(header)
    assert header["report"] == "run_metrics"
    assert header["tier"] == "sft"
    assert header["model_id"] == "test/model"
    assert header["recipe_hash"] == "abc123"
    assert header["schedule"]["total_steps"] == 100


def test_header_carries_ambient_context(tmp_path, obs_isolation):
    bind_run("training", "train-sft", "R-20260718-000000-abcdef")
    tracker = make_tracker(tmp_path, header=HEADER)
    tracker.log({"loss": 1.0}, step=1)
    tracker.close()
    header = read_records(tmp_path)[0]
    assert header["run_id"] == "R-20260718-000000-abcdef"
    assert header["cli"] == "training"
    assert header["command"] == "train-sft"


def test_header_context_empty_when_unbound(tmp_path, obs_isolation):
    tracker = make_tracker(tmp_path)
    tracker.log({"loss": 1.0}, step=1)
    tracker.close()
    header = read_records(tmp_path)[0]
    validate_report(header)
    assert header["run_id"] == ""
    assert header["cli"] == ""
    assert header["command"] == ""


def test_one_record_per_step_keys_as_logged(tmp_path):
    tracker = make_tracker(tmp_path, header=HEADER)
    tracker.log({"loss": 2.5, "lr": 1e-4}, step=1)
    tracker.log({"train/loss": 2.4, "train/lr": 9e-5}, step=2)
    tracker.log({"loss": 2.3, "grad_norm": 0.7}, step=3)
    tracker.close()
    records = read_records(tmp_path)[1:]
    assert [r["step"] for r in records] == [1, 2, 3]
    assert set(records[0]) == {"step", "ts", "loss", "lr"}
    assert set(records[1]) == {"step", "ts", "train/loss", "train/lr"}
    assert records[2]["grad_norm"] == 0.7
    for record in records:
        datetime.fromisoformat(record["ts"])


def test_marker_and_watchdog_records(tmp_path):
    tracker = make_tracker(tmp_path, header=HEADER)
    tracker.log({"checkpoint_saved": 1}, step=100)
    tracker.log({"watchdog/gpu_temp_c": 91.0}, step=100)
    tracker.close()
    records = read_records(tmp_path)[1:]
    assert records[0]["checkpoint_saved"] == 1
    assert records[1]["watchdog/gpu_temp_c"] == 91.0


def test_header_lazy_and_written_once(tmp_path):
    tracker = make_tracker(tmp_path, header=HEADER)
    assert not (tmp_path / "metrics.jsonl").exists()
    tracker.log({"loss": 1.0}, step=1)
    tracker.log({"loss": 0.9}, step=2)
    tracker.close()
    records = read_records(tmp_path)
    assert len(records) == 3
    assert sum(1 for r in records if r.get("report") == "run_metrics") == 1


def test_sink_disabled_by_config(tmp_path):
    cfg = TrackingConfig(
        tensorboard=False, wandb=False, mlflow=False, metrics_jsonl=False
    )
    tracker = Tracker(cfg, "run", str(tmp_path))
    tracker.log({"loss": 1.0}, step=1)
    tracker.close()
    assert tracker._sinks == []
    assert not (tmp_path / "metrics.jsonl").exists()


def test_lines_complete_after_each_log(tmp_path):
    tracker = make_tracker(tmp_path, header=HEADER)
    for step in range(1, 4):
        tracker.log({"loss": 1.0 / step}, step=step)
        # line-buffered append: the last line parses without a close
        assert read_records(tmp_path)[-1]["step"] == step
    tracker.close()


def test_write_failure_evicts_with_warning(tmp_path, capsys):
    tracker = make_tracker(tmp_path, header=HEADER)
    tracker.log({"loss": 1.0}, step=1)
    jsonl_sink(tracker)._file.close()
    tracker.log({"loss": 0.9}, step=2)
    assert "jsonl sink disabled" in capsys.readouterr().err
    assert tracker._sinks == []
    tracker.log({"loss": 0.8}, step=3)  # evicted: must be a silent no-op
    assert [r["step"] for r in read_records(tmp_path)[1:]] == [1]


def test_numpy_scalar_values_serialize(tmp_path):
    tracker = make_tracker(tmp_path, header=HEADER)
    tracker.log({"loss": np.float32(1.5), "tokens": np.int64(4096)}, step=1)
    tracker.close()
    record = read_records(tmp_path)[1]
    assert record["loss"] == 1.5
    assert record["tokens"] == 4096.0


def test_close_then_new_tracker_appends(tmp_path):
    tracker = make_tracker(tmp_path, header=HEADER)
    tracker.log({"loss": 1.0}, step=1)
    tracker.close()
    tracker = make_tracker(tmp_path, header=HEADER)
    tracker.log({"loss": 0.9}, step=2)
    tracker.close()
    # append mode: both sessions' records survive (two headers, two steps)
    steps = [r["step"] for r in read_records(tmp_path) if "step" in r]
    assert steps == [1, 2]


# -- callback and trainer wiring ----------------------------------------------


class RecordingTracker:
    def __init__(self):
        self.logs = []

    def log(self, metrics, step):
        self.logs.append((metrics, step))


def test_tracking_callback_maps_ce_kl(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(TrainerCallback=object)
    )
    recipe = SimpleNamespace(
        data=SimpleNamespace(micro_batch=1, grad_accum=1, seq_len=8),
        sft=SimpleNamespace(backend="trl"),
        peft=SimpleNamespace(enabled=False),
        model=SimpleNamespace(hf=SimpleNamespace(id="test/model")),
        checkpoint=SimpleNamespace(retain_last=1, retain_every=100),
    )
    tracker = RecordingTracker()
    callbacks = build_callbacks(
        recipe, str(tmp_path), tracker, watchdog=None, recipe_hash="h"
    )
    state = SimpleNamespace(global_step=5)
    callbacks[0].on_log(
        None, state, None, logs={"loss": 1.0, "ce_loss": 0.7, "kl_loss": 0.3}
    )
    metrics, step = tracker.logs[0]
    assert step == 5
    assert metrics["train/ce_loss"] == 0.7
    assert metrics["train/kl_loss"] == 0.3


def test_stop_after_ends_the_run_at_its_step_and_is_absent_without_it(
    monkeypatch, tmp_path
):
    """The seam tests/hardware/test_train_resume.py interrupts through: the
    flag flips at the end of the named step and not before, and a run
    without stop_after carries no stop callback at all."""
    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(TrainerCallback=object)
    )
    recipe = SimpleNamespace(
        data=SimpleNamespace(micro_batch=1, grad_accum=1, seq_len=8),
        sft=SimpleNamespace(backend="trl"),
        peft=SimpleNamespace(enabled=False),
        model=SimpleNamespace(hf=SimpleNamespace(id="test/model")),
        checkpoint=SimpleNamespace(retain_last=1, retain_every=100),
    )
    plain = build_callbacks(
        recipe, str(tmp_path), RecordingTracker(), watchdog=None, recipe_hash="h"
    )
    assert len(plain) == 2
    stopping = build_callbacks(
        recipe,
        str(tmp_path),
        RecordingTracker(),
        watchdog=None,
        recipe_hash="h",
        stop_after=10,
    )
    assert len(stopping) == 3
    stop = stopping[-1]
    control = SimpleNamespace(should_training_stop=False)
    stop.on_step_end(None, SimpleNamespace(global_step=9), control)
    assert control.should_training_stop is False
    stop.on_step_end(None, SimpleNamespace(global_step=10), control)
    assert control.should_training_stop is True


def test_header_identity_survives_sampler_thread_first_log(tmp_path, obs_isolation):
    import threading

    bind_run("training", "train-sft", "R-20260718-000000-abcdef")
    tracker = make_tracker(tmp_path, header=HEADER)
    # the watchdog sampler logs from its own thread, which has no bound
    # context; identity must have been captured at construction
    t = threading.Thread(target=tracker.log, args=({"gpu_temp": 61.0}, 0))
    t.start()
    t.join()
    tracker.close()
    header = read_records(tmp_path)[0]
    assert header["run_id"] == "R-20260718-000000-abcdef"
    assert header["cli"] == "training"
    assert header["command"] == "train-sft"


def test_concurrent_logs_write_one_header_and_whole_lines(tmp_path):
    import threading

    tracker = make_tracker(tmp_path, header=HEADER)
    workers = 4
    logs_each = 25
    barrier = threading.Barrier(workers)

    def hammer(worker):
        barrier.wait()  # all threads race the lazy first-log open
        for i in range(logs_each):
            tracker.log({"loss": float(worker)}, step=worker * 1000 + i)

    threads = [threading.Thread(target=hammer, args=(w,)) for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    tracker.close()
    records = read_records(tmp_path)  # every line parses or this raises
    headers = [r for r in records if r.get("report") == "run_metrics"]
    assert len(headers) == 1
    assert records[0] is headers[0] or records[0] == headers[0]
    assert len(records) == 1 + workers * logs_each


def test_evicted_sink_is_closed(tmp_path, capsys):
    tracker = make_tracker(tmp_path, header=HEADER)
    tracker.log({"loss": 1.0}, step=1)
    sink = jsonl_sink(tracker)
    sink._file.close()
    tracker.log({"loss": 0.9}, step=2)  # write fails -> evict -> teardown
    capsys.readouterr()
    assert tracker._sinks == []
    assert sink._file is None


def strict_loads(line):
    """json.loads that rejects the NaN/Infinity literals strict parsers do."""

    def reject(token):
        raise ValueError(f"non-strict JSON literal: {token}")

    return json.loads(line, parse_constant=reject)


def test_non_finite_values_serialize_as_null(tmp_path):
    tracker = make_tracker(tmp_path, header=HEADER)
    tracker.log(
        {"loss": float("nan"), "lr": float("inf"), "tokens_per_s": 100.0}, step=1
    )
    tracker.log({"loss": np.float32("nan")}, step=2)
    tracker.close()
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    records = [strict_loads(line) for line in lines]
    assert records[1]["loss"] is None
    assert records[1]["lr"] is None
    assert records[1]["tokens_per_s"] == 100.0
    assert records[2]["loss"] is None


def test_bad_value_skips_record_not_sink(tmp_path, capsys):
    tracker = make_tracker(tmp_path, header=HEADER)
    tracker.log({"loss": 1.0}, step=1)
    tracker.log({"loss": np.ones(4)}, step=2)  # non-scalar: data error
    err = capsys.readouterr().err
    assert "record at step 2 skipped" in err
    assert "metrics.jsonl" in err
    assert "sink disabled" not in err
    tracker.log({"loss": 0.9}, step=3)  # the sink must still be alive
    tracker.close()
    assert [r["step"] for r in read_records(tmp_path)[1:]] == [1, 3]


def test_eviction_warning_names_the_file(tmp_path, capsys):
    tracker = make_tracker(tmp_path, header=HEADER)
    tracker.log({"loss": 1.0}, step=1)
    jsonl_sink(tracker)._file.close()
    tracker.log({"loss": 0.9}, step=2)
    err = capsys.readouterr().err
    assert "jsonl sink disabled" in err
    assert "metrics.jsonl" in err
