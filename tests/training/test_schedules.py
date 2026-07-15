"""P2-D3 acceptance tests: the wsd multiplier curve pinned at hand-computed
steps, and tiny runs whose tracked lr shows the three segments with exact
curve continuation on resume mid-stable and mid-decay. CPU-only; the loop
tests mirror the tiny-e2e machinery from test_loop.py."""

import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from training.data.shards import ShardWriter, write_manifest  # noqa: E402
from training.loop import train  # noqa: E402
from training.recipes import load_recipe  # noqa: E402
from training.schedules import cosine_multiplier, wsd_multiplier  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TINY_RECIPE = os.path.join(REPO_ROOT, "training", "recipes", "reference", "tiny-e2e.yaml")

VOCAB = 512
SEQ_LEN = 64


class TestWsdMultiplier:
    # warmup 4, total 20, decay_start_frac 0.8 -> decay starts at step 16
    KW = dict(warmup_steps=4, total_steps=20,
              decay_start_frac=0.8, final_lr_frac=0.1)

    def test_warmup_segment(self):
        assert wsd_multiplier(0, **self.KW) == pytest.approx(0.25)
        assert wsd_multiplier(3, **self.KW) == pytest.approx(1.0)

    def test_stable_plateau(self):
        for step in (4, 10, 15):
            assert wsd_multiplier(step, **self.KW) == 1.0

    def test_one_sqrt_decay_endpoints(self):
        # p = (step - 16) / 4; multiplier = 0.1 + 0.9 * (1 - sqrt(p))
        assert wsd_multiplier(16, **self.KW) == pytest.approx(1.0)
        assert wsd_multiplier(17, **self.KW) == pytest.approx(0.1 + 0.9 * 0.5)
        assert wsd_multiplier(18, **self.KW) == pytest.approx(
            0.1 + 0.9 * (1.0 - math.sqrt(0.5)))
        assert wsd_multiplier(20, **self.KW) == pytest.approx(0.1)
        assert wsd_multiplier(25, **self.KW) == pytest.approx(0.1)  # clamped

    def test_linear_decay(self):
        kw = dict(self.KW, decay_shape="linear")
        assert wsd_multiplier(18, **kw) == pytest.approx(0.1 + 0.9 * 0.5)
        assert wsd_multiplier(20, **kw) == pytest.approx(0.1)

    def test_cosine_decay(self):
        kw = dict(self.KW, decay_shape="cosine")
        assert wsd_multiplier(17, **kw) == pytest.approx(
            0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * 0.25)))
        assert wsd_multiplier(20, **kw) == pytest.approx(0.1)

    def test_defaults_decay_to_zero_over_final_tenth(self):
        # decay_start = int(10 * 0.9) = 9
        assert wsd_multiplier(8, warmup_steps=2, total_steps=10) == 1.0
        assert wsd_multiplier(9, warmup_steps=2, total_steps=10) == pytest.approx(1.0)
        assert wsd_multiplier(10, warmup_steps=2, total_steps=10) == pytest.approx(0.0)

    def test_cosine_schedule_unchanged(self):
        assert cosine_multiplier(0, 4, 20) == pytest.approx(0.25)
        assert cosine_multiplier(12, 4, 20) == pytest.approx(0.5)
        assert cosine_multiplier(20, 4, 20) == pytest.approx(0.0)


@pytest.fixture(autouse=True)
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture()
def shard_dir(tmp_path) -> str:
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, VOCAB, size=6000).tolist()
    out = str(tmp_path / "shards")
    writer = ShardWriter(out, np.uint16, shard_tokens=2000)
    writer.write_tokens(tokens)
    records = writer.close()
    write_manifest(out, {
        "dataset": "synthetic", "split": None, "revision": None, "license": "unknown",
        "tokenizer": "synthetic", "tokenizer_hash": "abc123",
        "dtype": "uint16", "eos_id": 1, "shard_tokens": 2000,
        "shards": records, "total_tokens": len(tokens), "total_documents": 1,
    })
    return out


def wsd_recipe(shard_dir: str, output_root: str, *extra: str):
    # warmup 2, decay from step int(10 * 0.8) = 8: three visible segments
    return load_recipe(TINY_RECIPE, [
        f"data.source={shard_dir}",
        f"run.output_root={output_root}",
        f"data.seq_len={SEQ_LEN}",
        "schedule.type=wsd",
        "schedule.total_steps=10",
        "schedule.warmup_steps=2",
        "schedule.decay_start_frac=0.8",
        "schedule.final_lr_frac=0.1",
        "checkpoint.interval_steps=1",
        *extra,
    ])


def test_tiny_run_lr_curve_shows_three_segments(shard_dir, tmp_path):
    recipe = wsd_recipe(shard_dir, str(tmp_path / "run"))
    result = train(recipe)
    lrs = dict(result.lrs)
    peak = recipe.optimizer.lr
    # lr is read after scheduler.step(), so step N reports the multiplier at N
    assert lrs[1] == pytest.approx(peak)  # end of 2-step warmup
    for step in (2, 3, 7):
        assert lrs[step] == pytest.approx(peak)  # stable plateau
    assert lrs[9] == pytest.approx(peak * (0.1 + 0.9 * (1.0 - math.sqrt(0.5))))
    assert lrs[10] == pytest.approx(peak * 0.1)  # final_lr_frac floor


@pytest.mark.parametrize("stop_at", [5, 9], ids=["mid-stable", "mid-decay"])
def test_resume_continues_lr_curve_exactly(shard_dir, tmp_path, stop_at):
    straight = train(wsd_recipe(shard_dir, str(tmp_path / "straight")))
    root = str(tmp_path / "resumable")
    first = train(wsd_recipe(shard_dir, root), stop_after=stop_at)
    second = train(wsd_recipe(shard_dir, root, "checkpoint.resume=latest"))
    assert first.lrs + second.lrs == straight.lrs  # exact float equality
