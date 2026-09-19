"""GPU/VRAM hardware suite (WI-3): sm_120 presence and bf16 compute at the
smoke tier; near-capacity allocation, the calibrate measurement on the tiny
recipe, tolerance-based resume, family-registry loads, and the harvest hook
smoke at the full tier.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpu

REQUIRED_CAPABILITY = (12, 0)  # RTX 5090, Blackwell sm_120

# Stated GPU resume tolerance (resolved sweep decision: tolerance-based on
# GPU, bitwise only on CPU/tiny-model): per-step training-loss agreement
# within 1% relative over the post-resume window. CUDA kernels are
# nondeterministic across runs (atomics, cuDNN autotuning); identical data
# order plus restored RNG state bounds the drift well under this.
RESUME_LOSS_RTOL = 0.01

# Registered training families and their pinned checkpoints (kept in sync
# with training/models/families.py and the model policy).
FAMILY_PINS = {
    "gemma4": "google/gemma-4-E2B",
    "gemma4_text": "google/gemma-4-E2B",
}


@pytest.mark.hw_smoke
class TestGpuSmoke:
    def test_cuda_device_present(self, cuda):
        assert cuda.cuda.device_count() >= 1

    def test_compute_capability_sm120(self, cuda):
        assert cuda.cuda.get_device_capability(0) == REQUIRED_CAPABILITY

    def test_bf16_matmul_executes(self, cuda):
        a = cuda.randn(256, 256, dtype=cuda.bfloat16, device="cuda")
        out = a @ a
        cuda.cuda.synchronize()
        assert out.dtype == cuda.bfloat16
        assert cuda.isfinite(out.float()).all()

    def test_flash_attn_or_sdpa_fallback_stamps_metadata(self, cuda, capsys):
        """Either flash-attn imports, or the sdpa fallback engages with its
        warning, and the metadata stamp agrees with the resolution."""
        from training.models import model as model_mod
        from training.tune.sft_callbacks import _attention_implementation

        model_mod._flash_fallback_warned = False
        impl = model_mod.attention_implementation()
        err = capsys.readouterr().err
        try:
            import flash_attn  # noqa: F401

            flash_importable = True
        except Exception:
            flash_importable = False
        if flash_importable:
            assert impl == "flash_attention_2"
            assert err == ""
        else:
            assert impl == "sdpa"
            assert "flash-attn unavailable" in err
        # Provenance stamp resolves identically on every metadata path.
        assert _attention_implementation() == impl

    def test_deepspeed_cpu_adam_extension_loads(self):
        """The pre-built AVX-512 op must load (upstream issue #7556 is the
        known miscompile risk this smoke exists to front-run)."""
        from training.runtime.offload import cpu_adam_available

        assert cpu_adam_available(), (
            "the DeepSpeedCPUAdam op failed to load; rebuild the training image "
            "(znver5 CPUAdam pre-build, docker/training/Dockerfile.training)"
        )


@pytest.mark.hw_full
class TestGpuFull:
    def test_near_capacity_vram_allocation_graceful_oom(self, cuda):
        """Allocate 1GB blocks until OOM: the failure must be a catchable
        torch OOM, most of the 32GB must be reachable, and a small
        allocation must succeed again after cleanup."""
        blocks = []
        one_gb = 1 << 30
        try:
            for _ in range(64):
                blocks.append(cuda.empty(one_gb, dtype=cuda.uint8, device="cuda"))
        except cuda.cuda.OutOfMemoryError:
            pass
        reached = len(blocks)
        del blocks
        cuda.cuda.empty_cache()
        assert reached >= 24, f"only {reached}GB of 32GB VRAM was allocatable"
        probe = cuda.empty(one_gb, dtype=cuda.uint8, device="cuda")
        del probe
        cuda.cuda.empty_cache()

    def test_calibrate_measures_the_tiny_recipe(self, cuda, tmp_path, train_lock):
        """The D4 calculator's measurement path on the tiny recipe: calibrate
        builds the model, records what it holds resident, and derives an
        activation rate that reproduces the measured peak by construction
        (the 10 percent bar for a real recipe is the FT.calibrate-qlora runbook entry)."""
        from tests.hardware.conftest import REPO_ROOT
        from training.calculator.calibrate import calibrate
        from training.calculator.estimate import estimate
        from training.recipes import load_recipe

        recipe = load_recipe(
            os.path.join(
                REPO_ROOT, "training", "recipes", "reference", "tiny-e2e.yaml"
            ),
            ["data.seq_len=64"],
        )
        record = calibrate(recipe, steps=2)
        est = estimate(recipe)
        assert record.profile == est.profile == "scratch/2x64/full/nockpt"
        assert record.static_measured > 0
        assert record.activation_rate >= 0
        tokens = recipe.data.micro_batch * recipe.data.seq_len
        reproduced = (
            record.static_measured
            + est.grads
            + est.optimizer
            + record.activation_rate * tokens
        )
        # Floor division in the rate loses under one token's worth.
        assert reproduced <= record.measured_allocated < reproduced + tokens + 1
        assert record.measured_reserved >= record.measured_allocated

    def test_resume_loss_curve_within_tolerance(self, cuda, tmp_path, train_lock):
        """Straight N+M steps versus N steps + checkpoint + resume + M steps
        on the GPU: per-step losses over the post-resume window agree within
        RESUME_LOSS_RTOL (no torch.use_deterministic_algorithms in
        production paths by decision)."""
        from support import make_shard_dir

        from tests.hardware.conftest import REPO_ROOT
        from training.recipes import load_recipe
        from training.runtime.loop import train

        shard_dir = make_shards(tmp_path, make_shard_dir)
        tiny = os.path.join(
            REPO_ROOT, "training", "recipes", "reference", "tiny-e2e.yaml"
        )

        def recipe(output_root, *extra):
            return load_recipe(
                tiny,
                [
                    f"data.source={shard_dir}",
                    f"run.output_root={output_root}",
                    "data.seq_len=64",
                    "schedule.total_steps=12",
                    "schedule.warmup_steps=2",
                    "checkpoint.interval_steps=6",
                    *extra,
                ],
            )

        straight = train(recipe(tmp_path / "straight"))
        interrupted = train(recipe(tmp_path / "resumed"), stop_after=6)
        assert interrupted.final_step == 6
        resumed = train(recipe(tmp_path / "resumed", "checkpoint.resume=latest"))
        assert resumed.final_step == 12

        straight_tail = dict(straight.losses)
        for step, loss in resumed.losses:
            reference = straight_tail[step]
            rel = abs(loss - reference) / max(abs(reference), 1e-8)
            assert rel <= RESUME_LOSS_RTOL, (
                f"step {step}: resumed loss {loss} vs straight {reference} "
                f"(rel {rel:.4f} > {RESUME_LOSS_RTOL})"
            )

    @pytest.mark.parametrize("family", sorted(FAMILY_PINS))
    def test_family_registry_tiny_load(self, cuda, family):
        """Per registered family: the pinned checkpoint's config resolves
        through the registry and a tiny from_config model of that
        architecture runs a forward. Skips cleanly when weights/config are
        absent (offline or not yet downloaded); HF gating is out of scope."""
        from transformers import AutoConfig, AutoModelForCausalLM

        from training.models.families import family_spec

        spec = family_spec(family)  # raises on registry drift
        try:
            config = AutoConfig.from_pretrained(FAMILY_PINS[family])
        except Exception as exc:
            pytest.skip(f"weights/config absent for {FAMILY_PINS[family]}: {exc}")
        text_config = getattr(config, "text_config", config)
        if text_config.model_type != family and config.model_type != family:
            pytest.skip(
                f"{FAMILY_PINS[family]} resolves to {config.model_type}; "
                f"family {family} covered via its composite pin"
            )
        for attr, value in (
            ("num_hidden_layers", 2),
            ("hidden_size", 64),
            ("intermediate_size", 128),
            ("num_attention_heads", 4),
            ("num_key_value_heads", 2),
        ):
            if hasattr(text_config, attr):
                setattr(text_config, attr, value)
        model = AutoModelForCausalLM.from_config(config).to("cuda").eval()
        ids = cuda.randint(0, 128, (1, 8), device="cuda")
        with cuda.no_grad():
            out = model(input_ids=ids)
        assert out.logits.shape[1] == 8
        assert spec.lora_target_modules  # registry data present for the family

    def test_harvest_hook_smoke(self, cuda, tmp_path, train_lock):
        """Hook-based activation harvest through the WI-6 registry on the
        Gemma 4 E2B pin: dimension gate passes, a tiny harvest lands a
        manifest with registry-derived hooks and model_type. Skips cleanly
        when weights are absent."""
        from omegaconf import OmegaConf

        from interp.harvest import check_hook_dims, resolve_model_meta, run_harvest
        from training.recipes.schema import Recipe

        model_id = FAMILY_PINS["gemma4_text"]
        try:
            d_model, spec, depth = resolve_model_meta(model_id, None)
        except Exception as exc:
            pytest.skip(f"weights/config absent for {model_id}: {exc}")
        assert spec.model_type == "gemma4_text"
        # The hook below must be inside the stack; check_hook_dims now refuses
        # an out-of-range index before it loads the weights.
        assert depth > 2, f"{model_id} reports {depth} layers"

        widths = check_hook_dims(model_id, [2], None)
        assert widths == {2: d_model}

        recipe = OmegaConf.structured(Recipe)
        recipe.run.name = "hw-harvest-smoke"
        recipe.watchdog.enabled = False
        recipe.harvest.enabled = True
        recipe.harvest.model_id = model_id
        recipe.harvest.hook_layers = [2]
        recipe.harvest.corpus = "monology/pile-uncopyrighted"
        recipe.harvest.total_tokens = 2048
        recipe.harvest.context_len = 128
        # Explicit, because the schema default is still bf16 and Arrow cannot
        # store bfloat16 (KB-57): a structured Recipe left at its default dies
        # in the cache runner's constructor before any hook is exercised.
        recipe.harvest.dtype = "fp32"
        recipe.harvest.out_root = str(tmp_path / "activations")
        manifest = run_harvest(recipe)
        assert manifest["model_type"] == "gemma4_text"
        assert manifest["hooks"] == ["model.layers.2"]
        assert manifest["d_model"] == d_model


def make_shards(tmp_path, make_shard_dir):
    import numpy as np

    rng = np.random.default_rng(7)
    tokens = rng.integers(0, 512, size=8000).tolist()
    return make_shard_dir(tmp_path, tokens, shard_tokens=2000)
