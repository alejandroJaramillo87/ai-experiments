"""Pins the SAELens internals chunked capture depends on (KB-58).

interp/harvest.py calls CacheActivationsRunner._consolidate_shards to flatten a
hook's chunks into the single dataset directory the SAE trainer reads. That is
a private name, and a private name is exactly the kind of dependency that
disappears in a minor upgrade without a deprecation.

These tests exist so the disappearance is a red sandbox run rather than a
failure sixteen hours into a 17-hour capture, at the one step that runs after
all the expensive work is already on disk. They assert the contract, not the
behaviour: what interp/harvest.py needs is that the name exists, is callable
without an instance, and still takes the three arguments it is called with.

Skipped where sae-lens is absent, which is every sandbox run.
"""

from __future__ import annotations

import inspect

import pytest

sae_lens = pytest.importorskip("sae_lens", reason="sae-lens is a full-tier dependency")


class TestConsolidateShardsContract:
    def test_the_static_method_exists(self):
        assert hasattr(sae_lens.CacheActivationsRunner, "_consolidate_shards")

    def test_it_is_callable_without_a_runner_instance(self):
        # _consolidate_chunks calls it off the class, after the capture is
        # already on disk. If it became an instance method the call would
        # raise only at that final step.
        attr = inspect.getattr_static(
            sae_lens.CacheActivationsRunner, "_consolidate_shards"
        )
        assert isinstance(attr, staticmethod)

    def test_it_accepts_the_arguments_harvest_passes(self):
        signature = inspect.signature(
            sae_lens.CacheActivationsRunner._consolidate_shards
        )
        assert list(signature.parameters) == ["source_dir", "output_dir", "copy_files"]

    def test_copy_files_still_defaults_to_copying(self):
        # interp/harvest.py passes copy_files=False explicitly, because moving
        # is what makes assembly a metadata operation instead of a second full
        # copy of a terabyte. This asserts the default is the copying one, so
        # a future flip of the default cannot silently change our call from a
        # move into a copy that will not fit on the volume.
        signature = inspect.signature(
            sae_lens.CacheActivationsRunner._consolidate_shards
        )
        assert signature.parameters["copy_files"].default is True


class TestChunkBoundaryAssumptions:
    def test_buffer_counts_are_derived_from_training_tokens(self):
        # Chunked capture reuses one runner across chunks and reassigns
        # cfg.training_tokens per chunk. That only sizes the next run() if
        # these are recomputed on access rather than fixed at construction.
        for name in ("n_buffers", "n_seq_in_buffer", "n_tokens_in_buffer"):
            attr = inspect.getattr_static(sae_lens.CacheActivationsRunnerConfig, name)
            assert isinstance(attr, property), f"{name} is no longer derived"

    def test_the_buffer_size_default_matches_the_preflight_constant(self):
        # interp.harvest.SAELENS_BUFFER_GB replicates this default so the disk
        # preflight can price a capture without loading sae_lens or a model. A
        # drift here silently mis-prices every projection.
        import dataclasses

        from interp.harvest import SAELENS_BUFFER_GB

        fields = {
            f.name: f for f in dataclasses.fields(sae_lens.CacheActivationsRunnerConfig)
        }
        assert fields["buffer_size_gb"].default == SAELENS_BUFFER_GB

    def test_buffer_tokens_reproduces_the_saelens_formula(self):
        # The preflight rounds each chunk up to a whole buffer, so its estimate
        # is only as good as this arithmetic. Check ours against the real
        # config rather than against a number copied from it.
        from interp.harvest import buffer_tokens

        cfg = sae_lens.CacheActivationsRunnerConfig(
            dataset_path="NeelNanda/c4-10k",
            model_name="gpt2",
            hook_name="blocks.0.hook_resid_post",
            d_in=1536,
            dtype="float16",
            context_size=1024,
            model_batch_size=8,
            training_tokens=15_000_000,
        )
        assert cfg.n_tokens_in_buffer == buffer_tokens(1536, "fp16", 8, 1024)

    def test_a_cached_run_still_requires_a_dataset(self):
        # interp/sae/train.py passes dataset_path even though every activation
        # comes from disk, because ActivationsStore.from_config refuses an
        # empty one on both paths and then samples the corpus to decide
        # whether it is pre-tokenized. If SAELens ever exempts the cached
        # path, this fails and the corpus plumbing can be dropped -- which is
        # worth knowing, since it is what makes SAE training need network.
        from sae_lens.training import activations_store

        source = inspect.getsource(activations_store.ActivationsStore.from_config)
        assert 'cfg.dataset_path == ""' in source
        assert "You must either pass in a dataset" in source

    def test_the_store_does_not_re_apply_a_token_budget(self):
        # Chunks are contiguous because the activations store holds the corpus
        # stream position and has no budget of its own to reset. If
        # total_training_tokens gained a consumer that stops the stream, chunk
        # 2 onward would return nothing and captures would silently truncate.
        from sae_lens.training import activations_store

        source = inspect.getsource(activations_store)
        uses = source.count("self.total_training_tokens")
        assert uses == 1, (
            f"total_training_tokens is read {uses} times; chunked capture "
            f"assumes it is assigned and never consumed"
        )


class TestOutputPathContract:
    def test_the_runner_still_has_an_output_path(self):
        # interp/sae/train.py sets output_path so the final SAE lands in the
        # run directory. If the field is renamed the kwarg raises TypeError,
        # which is loud -- but a silent removal would restore the leak.
        import dataclasses

        fields = {
            f.name: f for f in dataclasses.fields(sae_lens.LanguageModelSAERunnerConfig)
        }
        assert "output_path" in fields

    def test_the_default_output_path_is_still_repo_relative(self):
        # The reason the kwarg is passed at all. A relative default resolves
        # against the CWD, which in the interp container is /workspace -- the
        # repository bind mount. If SAELens ever makes this absolute or
        # derives it from checkpoint_path, the explicit pass becomes
        # redundant and this test says so.
        import dataclasses
        import os

        fields = {
            f.name: f for f in dataclasses.fields(sae_lens.LanguageModelSAERunnerConfig)
        }
        default = fields["output_path"].default
        assert isinstance(default, str) and not os.path.isabs(default), (
            f"output_path default is {default!r}; if it is now absolute the "
            f"explicit output_path in interp/sae/train.py can be dropped"
        )


class TestCheckpointFormatContract:
    """The saved checkpoint must be the inference form, or nothing can read it.

    KB-66: `interp/sae/train.py` called `sae.save_model(out_dir)` after
    `SAETrainingRunner.run()`. run() already saves, through
    `save_final_sae(output_path)` -> `save_inference_model()`, which converts
    the trained BatchTopK SAE to its JumpReLU inference form. The extra call
    landed second and rewrote cfg.json with `architecture: "batchtopk"`, a
    *training* architecture absent from SAE_CLASS_REGISTRY, so
    `SAE.load_from_disk` failed with KeyError and `make sae-eval` could not
    evaluate any SAE this repository had ever trained.
    """

    def test_the_training_architecture_is_not_loadable(self):
        # The premise. If SAELens ever registers batchtopk as a loadable
        # architecture this test fails, and the conversion stops mattering.
        from sae_lens.registry import SAE_CLASS_REGISTRY

        assert "batchtopk" not in SAE_CLASS_REGISTRY

    def test_the_inference_architecture_we_rely_on_is_registered(self):
        from sae_lens.registry import SAE_CLASS_REGISTRY

        assert "jumprelu" in SAE_CLASS_REGISTRY

    def test_the_runner_saves_the_checkpoint_itself(self):
        # We removed our own save call and now depend on run() writing the
        # checkpoint. If that stops being true the run would complete and
        # leave an empty directory, which is a far worse failure than this
        # test going red.
        source = inspect.getsource(sae_lens.SAETrainingRunner.run)
        assert "save_final_sae" in source
        assert "self.cfg.output_path" in source

    def test_the_final_save_is_the_inference_conversion(self):
        # save_final_sae is only the right call because it goes through
        # save_inference_model. A future refactor that saved the training
        # model there would reintroduce KB-66 silently.
        source = inspect.getsource(sae_lens.SAETrainingRunner.save_final_sae)
        assert "save_inference_model" in source
