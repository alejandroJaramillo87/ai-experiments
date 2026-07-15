"""P2-D2 data-mixing tests: draw proportions from source tags, deterministic
random access, resume-index handling, strict single-epoch abort, tokenizer-hash
refusal at construction and at validate(), and the tagged dataloader path.
Sandbox-only: synthetic shards, no GPU or network."""

from __future__ import annotations

import numpy as np
import pytest

from training.data.mixture import MixtureDataset, SourceExhaustedError
from training.data.shards import ShardWriter, TokenizerMismatchError, write_manifest

WINDOW = 8


def make_source(tmp_path, name: str, tokens, tokenizer_hash: str = "abc123",
                shard_tokens: int = 64) -> str:
    out = str(tmp_path / name)
    writer = ShardWriter(out, np.uint16, shard_tokens=shard_tokens)
    writer.write_tokens(np.asarray(tokens, dtype=np.uint16))
    records = writer.close()
    write_manifest(out, {
        "dataset": name, "split": None, "revision": None, "license": "unknown",
        "tokenizer": "synthetic", "tokenizer_hash": tokenizer_hash,
        "dtype": "uint16", "eos_id": 1, "shard_tokens": shard_tokens,
        "shards": records, "total_tokens": len(tokens), "total_documents": 1,
    })
    return out


@pytest.fixture()
def three_sources(tmp_path):
    """Three sources whose token values identify their provenance:
    source i contains only the constant value 100 + i."""
    return [
        ("domain", make_source(tmp_path, "domain", [100] * 640), 0.8),
        ("replay_web", make_source(tmp_path, "replay_web", [101] * 320), 0.15),
        ("replay_code", make_source(tmp_path, "replay_code", [102] * 160), 0.05),
    ]


class TestConstruction:
    def test_rejects_mismatched_tokenizer_hashes(self, tmp_path):
        sources = [
            ("a", make_source(tmp_path, "a", [1] * 64, tokenizer_hash="one"), 0.5),
            ("b", make_source(tmp_path, "b", [2] * 64, tokenizer_hash="two"), 0.5),
        ]
        with pytest.raises(TokenizerMismatchError, match="distinct tokenizer hashes"):
            MixtureDataset(sources, window=WINDOW, seed=0)

    def test_rejects_empty_sources(self):
        with pytest.raises(ValueError, match="at least one source"):
            MixtureDataset([], window=WINDOW, seed=0)

    def test_rejects_bad_weights(self, tmp_path):
        path = make_source(tmp_path, "a", [1] * 64)
        with pytest.raises(ValueError, match="positive and finite"):
            MixtureDataset([("a", path, 0.0)], window=WINDOW, seed=0)

    def test_rejects_duplicate_names(self, tmp_path):
        path = make_source(tmp_path, "a", [1] * 64)
        with pytest.raises(ValueError, match="unique"):
            MixtureDataset([("a", path, 0.5), ("a", path, 0.5)], window=WINDOW, seed=0)

    def test_rejects_source_without_a_full_window(self, tmp_path):
        path = make_source(tmp_path, "tiny", [1] * (WINDOW - 1))
        with pytest.raises(ValueError, match="no complete window"):
            MixtureDataset([("tiny", path, 1.0)], window=WINDOW, seed=0)


class TestSampling:
    def test_10k_draws_match_weights_within_one_percent(self, three_sources):
        ds = MixtureDataset(three_sources, window=WINDOW, seed=7)
        counts = np.zeros(3, dtype=np.int64)
        for i in range(10_000):
            window, tag = ds[i]
            counts[tag] += 1
            assert window[0] == 100 + tag  # provenance: values identify the source
        realized = counts / counts.sum()
        for frac, (_, _, weight) in zip(realized, three_sources, strict=True):
            assert abs(frac - weight) <= 0.01

    def test_within_source_reads_are_sequential_and_wrap(self, tmp_path):
        # A counting corpus exposes read order: window k starts at k * WINDOW.
        path = make_source(tmp_path, "seq", np.arange(4 * WINDOW))
        ds = MixtureDataset([("seq", path, 1.0)], window=WINDOW, seed=0)
        starts = [int(ds[i][0][0]) for i in range(9)]
        expected = [(k % 4) * WINDOW for k in range(9)]  # wraps after 4 windows
        assert starts == expected
        assert ds.epoch_of(0) == 0
        assert ds.epoch_of(8) == 2

    def test_random_access_is_deterministic_across_instances(self, three_sources):
        a = MixtureDataset(three_sources, window=WINDOW, seed=42)
        b = MixtureDataset(three_sources, window=WINDOW, seed=42)
        forward = [a[i] for i in range(200)]
        backward = [b[i] for i in reversed(range(200))][::-1]
        for (wa, ta), (wb, tb) in zip(forward, backward, strict=True):
            assert ta == tb
            assert np.array_equal(wa, wb)

    def test_different_seeds_differ(self, three_sources):
        a = MixtureDataset(three_sources, window=WINDOW, seed=1)
        b = MixtureDataset(three_sources, window=WINDOW, seed=2)
        assert [a.source_at(i) for i in range(64)] != [b.source_at(i) for i in range(64)]

    def test_strict_single_epoch_aborts_on_wrap(self, tmp_path):
        path = make_source(tmp_path, "tiny", [1] * (2 * WINDOW))  # 2 windows
        ds = MixtureDataset([("tiny", path, 1.0)], window=WINDOW, seed=0,
                            strict_single_epoch=True)
        ds[0], ds[1]
        with pytest.raises(SourceExhaustedError, match="tiny"):
            ds[2]


class TestResumeIndex:
    def test_mixture_state_round_trip(self, three_sources):
        from training.data.loader import resume_index

        ds = MixtureDataset(three_sources, window=WINDOW, seed=0)
        state = {"format": "mixture", "next_draw": 137, "window": WINDOW}
        assert resume_index(ds, state) == 137
        assert resume_index(ds, None) == 0

    def test_cross_format_states_are_rejected(self, three_sources, tmp_path):
        from training.data.loader import resume_index
        from training.data.shards import ShardedTokenDataset

        mixture = MixtureDataset(three_sources, window=WINDOW, seed=0)
        single = ShardedTokenDataset(three_sources[0][1], window=WINDOW)
        with pytest.raises(ValueError, match="single-source"):
            resume_index(mixture, {"format": "index", "next_index": 5, "window": WINDOW})
        with pytest.raises(ValueError, match="mixture"):
            resume_index(single, {"format": "mixture", "next_draw": 5, "window": WINDOW})

    def test_window_mismatch_is_rejected(self, three_sources):
        from training.data.loader import resume_index

        ds = MixtureDataset(three_sources, window=WINDOW, seed=0)
        with pytest.raises(ValueError, match="window"):
            resume_index(ds, {"format": "mixture", "next_draw": 5, "window": WINDOW * 2})


class TestDataloader:
    @pytest.fixture(autouse=True)
    def _torch(self):
        pytest.importorskip("torch")

    def data_cfg(self, workers=0):
        from omegaconf import OmegaConf

        return OmegaConf.create({
            "micro_batch": 2, "workers": workers,
            "prefetch_factor": 2, "pin_memory": False})

    def test_batches_carry_tokens_and_tags(self, three_sources):
        from training.data.loader import build_dataloader

        ds = MixtureDataset(three_sources, window=WINDOW, seed=0)
        loader = build_dataloader(ds, self.data_cfg())
        tokens, tags = next(iter(loader))
        assert tuple(tokens.shape) == (2, WINDOW)
        assert tuple(tags.shape) == (2,)
        for row, tag in zip(tokens, tags, strict=True):
            assert row[0].item() == 100 + tag.item()

    @pytest.mark.parametrize("workers", [0, 2])
    def test_start_index_resumes_the_draw_sequence(self, three_sources, workers):
        from training.data.loader import build_dataloader

        ds = MixtureDataset(three_sources, window=WINDOW, seed=0)
        full = build_dataloader(ds, self.data_cfg(workers))
        it = iter(full)
        reference = [next(it) for _ in range(6)]  # 12 draws

        resumed_ds = MixtureDataset(three_sources, window=WINDOW, seed=0)
        resumed = build_dataloader(resumed_ds, self.data_cfg(workers), start_index=6)
        it = iter(resumed)
        replay = [next(it) for _ in range(3)]
        for (t_ref, g_ref), (t_new, g_new) in zip(reference[3:], replay, strict=True):
            assert (t_ref == t_new).all()
            assert (g_ref == g_new).all()

    def test_prefetcher_passes_tag_batches_through(self, three_sources):
        import torch

        from training.data.loader import CudaPrefetcher, build_dataloader

        ds = MixtureDataset(three_sources, window=WINDOW, seed=0)
        prefetcher = CudaPrefetcher(
            build_dataloader(ds, self.data_cfg()), torch.device("cpu"))
        tokens, tags = prefetcher.next()
        assert tuple(tokens.shape) == (2, WINDOW)
        assert tuple(tags.shape) == (2,)
        prefetcher.close()


class TestLoop:
    """P2-D2 acceptance through the D6 loop: deterministic resume over a
    mixture and per-source token accounting in the result and checkpoint meta."""

    @pytest.fixture(autouse=True)
    def _deterministic(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        torch.use_deterministic_algorithms(True)
        yield
        torch.use_deterministic_algorithms(False)

    def loop_recipe(self, tmp_path, sources, output_root, *extra):
        from training.recipes import load_recipe

        source_lines = "\n".join(
            f'    - {{name: {name}, path: {path}, weight: {weight}}}'
            for name, path, weight in sources)
        text = f"""
run: {{name: mix-loop, seed: 1234, output_root: {output_root}}}
model:
  scratch: {{layers: 2, hidden: 64, heads: 4, kv_heads: 2, vocab: 512}}
data: {{seq_len: 64, micro_batch: 2, workers: 0, pin_memory: false}}
mixture:
  enabled: true
  sources:
{source_lines}
tokenizer: {{path: unused-by-the-loop}}
optimizer: {{lr: 1.0e-3, weight_decay: 0.0}}
schedule: {{type: cosine, warmup_steps: 2, total_steps: 8}}
checkpoint: {{interval_steps: 4}}
tracking: {{tensorboard: false}}
watchdog: {{enabled: false}}
"""
        path = tmp_path / "loop-recipe.yaml"
        path.write_text(text)
        return load_recipe(str(path), overrides=list(extra))

    @pytest.fixture()
    def loop_sources(self, tmp_path):
        rng = np.random.default_rng(7)
        def corpus(n):
            return rng.integers(0, 512, size=n).tolist()
        return [
            ("domain", make_source(tmp_path, "domain", corpus(4000),
                                   shard_tokens=2000), 0.7),
            ("replay", make_source(tmp_path, "replay", corpus(2000),
                                   shard_tokens=2000), 0.3),
        ]

    def test_deterministic_resume_and_accounting(self, loop_sources, tmp_path):
        from training.loop import train

        straight = train(self.loop_recipe(tmp_path, loop_sources,
                                          str(tmp_path / "straight")))
        assert straight.final_step == 8
        assert sum(straight.tokens_per_source.values()) == straight.tokens_seen
        assert set(straight.tokens_per_source) == {"domain", "replay"}

        root = str(tmp_path / "resumable")
        first = train(self.loop_recipe(tmp_path, loop_sources, root),
                      stop_after=4)
        second = train(self.loop_recipe(tmp_path, loop_sources, root,
                                        "checkpoint.resume=latest"))
        assert first.losses + second.losses == straight.losses
        # Accounting is cumulative across the resume, not restarted.
        assert second.tokens_per_source == straight.tokens_per_source


class TestValidate:
    def recipe(self, tmp_path, tokenizer_dir, sources, extra="", overrides=()):
        from training.recipes import load_recipe

        source_lines = "\n".join(
            f'    - {{name: {name}, path: {path}, weight: {weight}}}'
            for name, path, weight in sources)
        text = f"""
run: {{name: mix-test}}
model:
  scratch: {{layers: 1, hidden: 32, heads: 2, kv_heads: 1, vocab: 512}}
data: {{seq_len: {WINDOW}, micro_batch: 1}}
mixture:
  enabled: true
  sources:
{source_lines}
tokenizer: {{path: {tokenizer_dir}}}
optimizer: {{lr: 1.0e-3}}
schedule: {{total_steps: 4}}
{extra}
"""
        path = tmp_path / "recipe.yaml"
        path.write_text(text)
        return load_recipe(str(path), overrides=overrides)

    def test_matching_hashes_pass(self, tmp_path, tokenizer_dir):
        from training.data.prepare import load_tokenizer
        from training.recipes import validate

        tok_hash = load_tokenizer(tokenizer_dir).content_hash
        sources = [
            ("a", make_source(tmp_path, "a", [1] * 64, tokenizer_hash=tok_hash), 0.7),
            ("b", make_source(tmp_path, "b", [2] * 64, tokenizer_hash=tok_hash), 0.3),
        ]
        validate(self.recipe(tmp_path, tokenizer_dir, sources))

    def test_mismatched_hashes_are_refused(self, tmp_path, tokenizer_dir):
        from training.recipes import RecipeValidationError, validate

        sources = [
            ("a", make_source(tmp_path, "a", [1] * 64, tokenizer_hash="one"), 0.7),
            ("b", make_source(tmp_path, "b", [2] * 64, tokenizer_hash="two"), 0.3),
        ]
        with pytest.raises(RecipeValidationError, match="mismatched tokenizer hashes"):
            validate(self.recipe(tmp_path, tokenizer_dir, sources))

    def test_hash_differing_from_tokenizer_is_refused(self, tmp_path, tokenizer_dir):
        from training.recipes import RecipeValidationError, validate

        sources = [
            ("a", make_source(tmp_path, "a", [1] * 64, tokenizer_hash="other"), 1.0),
        ]
        with pytest.raises(RecipeValidationError, match="does not match"):
            validate(self.recipe(tmp_path, tokenizer_dir, sources))

    def test_data_source_mutual_exclusion(self, tmp_path, tokenizer_dir):
        from training.data.prepare import load_tokenizer
        from training.recipes import RecipeValidationError, validate

        tok_hash = load_tokenizer(tokenizer_dir).content_hash
        src = make_source(tmp_path, "a", [1] * 64, tokenizer_hash=tok_hash)
        recipe = self.recipe(tmp_path, tokenizer_dir, [("a", src, 1.0)],
                             overrides=[f"data.source={src}"])
        with pytest.raises(RecipeValidationError, match="data.source must be unset"):
            validate(recipe)

    def test_staging_is_refused_with_mixture(self, tmp_path, tokenizer_dir):
        from training.data.prepare import load_tokenizer
        from training.recipes import RecipeValidationError, validate

        tok_hash = load_tokenizer(tokenizer_dir).content_hash
        src = make_source(tmp_path, "a", [1] * 64, tokenizer_hash=tok_hash)
        recipe = self.recipe(tmp_path, tokenizer_dir, [("a", src, 1.0)],
                             extra="staging: {enabled: true}")
        with pytest.raises(RecipeValidationError, match="staging"):
            validate(recipe)

    def test_negative_weight_and_empty_sources_are_refused(self, tmp_path, tokenizer_dir):
        from training.recipes import RecipeValidationError, validate

        src = make_source(tmp_path, "a", [1] * 64)
        with pytest.raises(RecipeValidationError, match="positive and finite"):
            validate(self.recipe(tmp_path, tokenizer_dir, [("a", src, -0.5)]))
