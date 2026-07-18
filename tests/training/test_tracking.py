"""D6 tracking fan-out: fail-soft sink eviction and the no-sink contract.

A sink whose method raises must be dropped after the first failure so a
transient sink fault cannot kill a multi-day run, while the surviving sinks
keep receiving events. Real sinks need torch/wandb/mlflow (absent in the
sandbox), so fake sinks are injected into the fan-out; the subject under test
is Tracker.log's dispatch-and-evict logic, not the sinks."""

from training.recipes.schema import TrackingConfig
from training.runtime.tracking import Tracker


class RecordingTensorboard:
    """A tensorboard-shaped sink that records every add_scalar call."""

    def __init__(self):
        self.scalars = []

    def add_scalar(self, key, value, step):
        self.scalars.append((key, value, step))


class RecordingWandb:
    """A wandb-shaped sink that records every log call."""

    def __init__(self):
        self.logs = []

    def log(self, metrics, step):
        self.logs.append((metrics, step))


class FailingTensorboard:
    """A tensorboard-shaped sink whose add_scalar always raises."""

    def __init__(self):
        self.calls = 0

    def add_scalar(self, key, value, step):
        self.calls += 1
        raise RuntimeError("sink down")


def make_tracker(run_dir):
    """A Tracker with no sinks wired by __init__ (all backends disabled), so
    fakes can be injected without importing torch/wandb/mlflow."""
    cfg = TrackingConfig(
        tensorboard=False, wandb=False, mlflow=False, metrics_jsonl=False
    )
    return Tracker(cfg, "run", str(run_dir))


def test_no_sinks_construction_is_a_noop(tmp_path):
    tracker = make_tracker(tmp_path)
    assert tracker._sinks == []
    # log and close on an empty fan-out must not raise (fail-soft contract)
    tracker.log({"loss": 1.0}, step=0)
    tracker.close()


def test_failing_sink_is_evicted_and_survivors_keep_receiving(tmp_path, capsys):
    tracker = make_tracker(tmp_path)
    bad = FailingTensorboard()
    good = RecordingWandb()
    # Injected in this order so the failing sink is dispatched first and its
    # removal mid-iteration cannot skip the survivor.
    tracker._sinks = [("tensorboard", bad), ("wandb", good)]

    tracker.log({"loss": 1.0}, step=1)
    # The failing sink is dropped after its one failure; the survivor still ran.
    assert bad.calls == 1
    assert tracker._sinks == [("wandb", good)]
    assert good.logs == [({"loss": 1.0}, 1)]
    assert "tensorboard sink disabled" in capsys.readouterr().err

    # A later call must not re-invoke or re-raise through the evicted sink,
    # and the survivor keeps receiving.
    tracker.log({"loss": 2.0}, step=2)
    assert bad.calls == 1  # never called again
    assert good.logs == [({"loss": 1.0}, 1), ({"loss": 2.0}, 2)]


def test_eviction_leaves_a_healthy_sink_untouched(tmp_path):
    tracker = make_tracker(tmp_path)
    healthy = RecordingTensorboard()
    bad = FailingTensorboard()
    tracker._sinks = [("tensorboard", healthy), ("tensorboard", bad)]

    tracker.log({"loss": 3.0}, step=5)
    # The healthy tensorboard sink recorded; the failing one was evicted.
    assert healthy.scalars == [("loss", 3.0, 5)]
    assert tracker._sinks == [("tensorboard", healthy)]

    tracker.log({"loss": 4.0}, step=6)
    assert healthy.scalars == [("loss", 3.0, 5), ("loss", 4.0, 6)]
