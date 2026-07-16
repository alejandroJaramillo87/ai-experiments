"""F5 distillation harness tests: schema/validation invariants, the logprob
cache round trip and its integrity/binding refusals, the masked-KL loss
(golden fixture: prompt tokens contribute zero KL), the online-versus-cached
tolerance bound, and provenance wiring. CPU-only on synthetic random-weight
tensors (resolved F5 decision 3); the tiny real-model smoke and the GPU
acceptance runs happen on the workstation."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from support import reference_recipe

from training.data.distill import (
    LogitCache,
    LogitCacheError,
    LogitCacheWriter,
)
from training.data.sft import IGNORE_INDEX
from training.recipes import RecipeValidationError, validate
from training.tune.distill import (
    DistillPreflightError,
    check_softcap_parity,
    distill_provenance,
    final_logit_softcapping,
)
from training.tune.sft_callbacks import write_checkpoint_meta


def distill_recipe(overrides=()):
    return reference_recipe("distill-e2b.yaml", list(overrides))


# -- schema and validation ---------------------------------------------------


class TestDistillValidation:
    def test_reference_recipe_validates(self):
        validate(distill_recipe())

    def test_tier_must_be_known(self):
        with pytest.raises(RecipeValidationError, match="distill.tier"):
            validate(distill_recipe(["distill.tier=embedding"]))

    def test_mode_must_be_known(self):
        with pytest.raises(RecipeValidationError, match="distill.mode"):
            validate(distill_recipe(["distill.mode=streamed"]))

    def test_kl_direction_must_be_known(self):
        with pytest.raises(RecipeValidationError, match="kl_direction"):
            validate(distill_recipe(["distill.kl_direction=symmetric"]))

    def test_alpha_bounds(self):
        with pytest.raises(RecipeValidationError, match="distill.alpha"):
            validate(distill_recipe(["distill.alpha=1.5"]))

    def test_temperature_positive(self):
        with pytest.raises(RecipeValidationError, match="distill.temperature"):
            validate(distill_recipe(["distill.temperature=0"]))

    def test_requires_sft_enabled(self):
        with pytest.raises(RecipeValidationError, match="sft.enabled"):
            validate(distill_recipe(["sft.enabled=false"]))

    def test_logit_requires_teacher(self):
        with pytest.raises(RecipeValidationError, match="distill.teacher"):
            validate(distill_recipe(["distill.teacher=''"]))

    def test_logit_is_trl_only(self):
        with pytest.raises(RecipeValidationError, match="trl"):
            validate(distill_recipe(["sft.backend=unsloth"]))

    def test_logit_rejects_fp8(self):
        with pytest.raises(RecipeValidationError, match="fp8"):
            validate(
                distill_recipe(
                    [
                        "precision.dtype=fp8",
                        "peft.quantize_base=none",
                        "distill.teacher_quantize=none",
                    ]
                )
            )

    def test_cached_requires_cache_path(self):
        with pytest.raises(RecipeValidationError, match="cache_path"):
            validate(
                distill_recipe(["distill.mode=cached", "distill.teacher_quantize=none"])
            )

    def test_cached_rejects_packing(self):
        with pytest.raises(RecipeValidationError, match="packing"):
            validate(
                distill_recipe(
                    [
                        "distill.mode=cached",
                        "distill.cache_path=/tmp/cache",
                        "distill.teacher_quantize=none",
                        "sft.packing=true",
                    ]
                )
            )

    def test_cached_rejects_teacher_quantize(self):
        with pytest.raises(RecipeValidationError, match="teacher_quantize"):
            validate(
                distill_recipe(["distill.mode=cached", "distill.cache_path=/tmp/cache"])
            )

    def test_response_requires_teacher_manifest(self):
        with pytest.raises(RecipeValidationError, match="teacher_manifest"):
            validate(distill_recipe(["distill.tier=response"]))

    def test_response_rejects_non_f4_manifest(self, tmp_path):
        bogus = tmp_path / "manifest.json"
        bogus.write_text(json.dumps({"kind": "sft"}))
        with pytest.raises(RecipeValidationError, match="teacher block"):
            validate(
                distill_recipe(
                    ["distill.tier=response", f"distill.teacher_manifest={bogus}"]
                )
            )

    def test_response_accepts_f4_manifest(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps({"teacher": {"model": "gemma-4-E4B-it-Q8_0.gguf"}})
        )
        validate(
            distill_recipe(
                ["distill.tier=response", f"distill.teacher_manifest={manifest}"]
            )
        )

    def test_disabled_block_checks_nothing(self):
        validate(distill_recipe(["distill.enabled=false", "distill.tier=bogus"]))


# -- logprob cache ------------------------------------------------------------


def write_cache(out_dir, examples, top_k=4, **binding):
    writer = LogitCacheWriter(str(out_dir), top_k)
    for tokens in examples:
        rng = np.random.default_rng(tokens)
        writer.add_example(
            rng.integers(0, 1000, size=(tokens, top_k)).astype(np.int32),
            -rng.random(size=(tokens, top_k)).astype(np.float16),
        )
    fields = {
        "teacher": "test-teacher",
        "teacher_revision": "abc123",
        "tokenizer_hash": "tok",
        "dataset_dir": "/data",
        "dataset_train_hash": "hash",
    }
    fields.update(binding)
    return writer.finalize(**fields)


class TestLogitCache:
    def test_round_trip(self, tmp_path):
        manifest = write_cache(tmp_path / "cache", [5, 3, 7])
        cache = LogitCache(str(tmp_path / "cache"))
        assert len(cache) == 3
        assert manifest["total_tokens"] == 15
        idx, lp = cache.example(1)
        assert idx.shape == (3, 4)
        assert lp.dtype == np.float16
        # deterministic content: example 1 was seeded with its token count
        rng = np.random.default_rng(3)
        assert np.array_equal(idx, rng.integers(0, 1000, size=(3, 4)).astype(np.int32))

    def test_refuses_overwrite(self, tmp_path):
        write_cache(tmp_path / "cache", [2])
        with pytest.raises(FileExistsError, match="refusing"):
            LogitCacheWriter(str(tmp_path / "cache"), 4)

    def test_refuses_corrupt_file(self, tmp_path):
        write_cache(tmp_path / "cache", [4, 4])
        path = tmp_path / "cache" / "logprobs.bin"
        data = bytearray(path.read_bytes())
        data[0] ^= 0xFF
        path.write_bytes(data)
        with pytest.raises(LogitCacheError, match="content hash"):
            LogitCache(str(tmp_path / "cache"))

    def test_refuses_wrong_kind(self, tmp_path):
        os.makedirs(tmp_path / "cache")
        (tmp_path / "cache" / "manifest.json").write_text(json.dumps({"kind": "sft"}))
        with pytest.raises(LogitCacheError, match="not a logit cache"):
            LogitCache(str(tmp_path / "cache"))

    def test_binding_mismatches_refused(self, tmp_path):
        write_cache(tmp_path / "cache", [5, 3])
        cache = LogitCache(str(tmp_path / "cache"))
        cache.verify_binding(
            dataset_train_hash="hash", tokenizer_hash="tok", examples=2
        )
        with pytest.raises(LogitCacheError, match="dataset train hash"):
            cache.verify_binding(
                dataset_train_hash="other", tokenizer_hash="tok", examples=2
            )
        with pytest.raises(LogitCacheError, match="tokenizer hash"):
            cache.verify_binding(
                dataset_train_hash="hash", tokenizer_hash="other", examples=2
            )
        with pytest.raises(LogitCacheError, match="examples"):
            cache.verify_binding(
                dataset_train_hash="hash", tokenizer_hash="tok", examples=3
            )

    def test_writer_rejects_shape_mismatch(self, tmp_path):
        writer = LogitCacheWriter(str(tmp_path / "cache"), 4)
        with pytest.raises(LogitCacheError, match="matching"):
            writer.add_example(np.zeros((3, 4), np.int32), np.zeros((3, 5), np.float16))


# -- masked-KL loss (torch) ---------------------------------------------------


class TestMaskedDistillLoss:
    """Golden-fixture and property tests on synthetic tensors (decision 3)."""

    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")
        from training.tune.distill import masked_distill_loss

        self.loss = masked_distill_loss

    def fixture_batch(self, vocab=32, seq=10, batch=2, seed=0):
        torch = self.torch
        gen = torch.Generator().manual_seed(seed)
        student = torch.randn(batch, seq, vocab, generator=gen)
        teacher = torch.randn(batch, seq, vocab, generator=gen)
        labels = torch.randint(0, vocab, (batch, seq), generator=gen)
        # Golden mask: first 6 positions are prompt (masked), rest assistant.
        labels[:, :6] = IGNORE_INDEX
        return student, teacher, labels

    def kl_at(self, student, teacher, labels):
        _, _, kl = self.loss(
            student,
            labels,
            alpha=0.0,
            temperature=2.0,
            kl_direction="forward",
            teacher_logits=teacher,
        )
        return float(kl)

    def test_prompt_tokens_contribute_zero_kl(self):
        """The F5 acceptance fixture: KL is identical whether prompt-position
        logits agree or wildly disagree, because the mask excludes them."""
        student, teacher, labels = self.fixture_batch()
        baseline = self.kl_at(student, teacher, labels)
        corrupted = teacher.clone()
        corrupted[:, :5] += 100.0  # prompt positions only (pre-shift 0..4)
        assert self.kl_at(student, corrupted, labels) == pytest.approx(baseline)
        assistant_shift = teacher.clone()
        assistant_shift[:, 7] += 100.0  # an assistant position must move it
        assert self.kl_at(student, assistant_shift, labels) != pytest.approx(baseline)

    def test_fully_masked_batch_has_zero_kl(self):
        student, teacher, labels = self.fixture_batch()
        labels[:] = IGNORE_INDEX
        assert self.kl_at(student, teacher, labels) == pytest.approx(0.0)

    def test_kl_zero_when_student_equals_teacher(self):
        student, _, labels = self.fixture_batch()
        assert self.kl_at(student, student.clone(), labels) == pytest.approx(
            0.0, abs=1e-5
        )

    def test_alpha_blends_ce_and_kl(self):
        student, teacher, labels = self.fixture_batch()
        loss, ce, kl = self.loss(
            student,
            labels,
            alpha=0.3,
            temperature=2.0,
            kl_direction="forward",
            teacher_logits=teacher,
        )
        assert float(loss) == pytest.approx(0.3 * float(ce) + 0.7 * float(kl), rel=1e-5)

    def test_reverse_kl_differs_from_forward(self):
        student, teacher, labels = self.fixture_batch()
        _, _, fwd = self.loss(
            student,
            labels,
            alpha=0.0,
            temperature=2.0,
            kl_direction="forward",
            teacher_logits=teacher,
        )
        _, _, rev = self.loss(
            student,
            labels,
            alpha=0.0,
            temperature=2.0,
            kl_direction="reverse",
            teacher_logits=teacher,
        )
        assert float(fwd) != pytest.approx(float(rev))

    def test_requires_exactly_one_teacher_form(self):
        student, teacher, labels = self.fixture_batch()
        with pytest.raises(ValueError, match="exactly one"):
            self.loss(
                student, labels, alpha=0.5, temperature=2.0, kl_direction="forward"
            )

    def topk_slices(self, teacher, k):
        torch = self.torch
        logprobs = torch.log_softmax(teacher.float(), dim=-1)
        top_lp, top_ids = logprobs.topk(k, dim=-1)
        return top_ids, top_lp

    def test_cached_equals_online_at_full_support(self):
        """With K = vocab the renormalization is exact: cached and online KL
        agree to numerical precision."""
        student, teacher, labels = self.fixture_batch(vocab=32)
        online = self.kl_at(student, teacher, labels)
        ids, lps = self.topk_slices(teacher, k=32)
        _, _, cached = self.loss(
            student,
            labels,
            alpha=0.0,
            temperature=2.0,
            kl_direction="forward",
            teacher_topk_ids=ids,
            teacher_topk_logprobs=lps,
        )
        assert float(cached) == pytest.approx(online, rel=1e-4, abs=1e-5)

    def test_cached_within_tolerance_at_truncated_support(self):
        """The online-versus-cached acceptance bound: with a peaked teacher
        (low-temperature sampling regime) and K covering the bulk of the
        mass, truncated KL tracks the exact KL within 5 percent relative or
        0.02 absolute — the documented tolerance attributable to top-K
        truncation."""
        torch = self.torch
        gen = torch.Generator().manual_seed(1)
        student = torch.randn(2, 10, 256, generator=gen)
        teacher = 4.0 * torch.randn(2, 10, 256, generator=gen)  # peaked
        labels = torch.randint(0, 256, (2, 10), generator=gen)
        labels[:, :4] = IGNORE_INDEX
        online = self.kl_at(student, teacher, labels)
        ids, lps = self.topk_slices(teacher, k=64)
        _, _, cached = self.loss(
            student,
            labels,
            alpha=0.0,
            temperature=2.0,
            kl_direction="forward",
            teacher_topk_ids=ids,
            teacher_topk_logprobs=lps,
        )
        assert float(cached) == pytest.approx(online, rel=0.05, abs=0.02)


# -- preflights ----------------------------------------------------------------


class Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestSoftcapParity:
    def test_matching_caps_pass(self):
        check_softcap_parity(
            Cfg(final_logit_softcapping=30.0), Cfg(final_logit_softcapping=30.0)
        )
        check_softcap_parity(Cfg(), Cfg())

    def test_mismatch_refused(self):
        with pytest.raises(DistillPreflightError, match="softcapping"):
            check_softcap_parity(
                Cfg(final_logit_softcapping=None), Cfg(final_logit_softcapping=30.0)
            )

    def test_descends_text_config(self):
        composite = Cfg(text_config=Cfg(final_logit_softcapping=30.0))
        assert final_logit_softcapping(composite) == 30.0


# -- provenance ----------------------------------------------------------------


class TestProvenance:
    def test_logit_fields(self):
        recipe = distill_recipe()
        fields = distill_provenance(recipe, revision="rev123")
        assert fields["tier"] == "logit"
        assert fields["teacher"] == "google/gemma-4-E4B"
        assert fields["teacher_revision"] == "rev123"
        assert fields["mode"] == "online"
        assert fields["alpha"] == 0.5

    def test_cached_fields_carry_cache_revision(self):
        recipe = distill_recipe(
            [
                "distill.mode=cached",
                "distill.cache_path=/tmp/c",
                "distill.teacher_quantize=none",
            ]
        )
        fields = distill_provenance(
            recipe, cache_manifest={"teacher_revision": "cachedrev"}
        )
        assert fields["cache_path"] == "/tmp/c"
        assert fields["top_k"] == 64
        assert fields["teacher_revision"] == "cachedrev"

    def test_response_fields(self):
        recipe = distill_recipe(
            ["distill.tier=response", "distill.teacher_manifest=/m.json"]
        )
        fields = distill_provenance(recipe)
        assert fields == {"tier": "response", "teacher_manifest": "/m.json"}

    def test_checkpoint_meta_carries_distill_block(self, tmp_path):
        write_checkpoint_meta(
            str(tmp_path),
            10,
            adapter_only=True,
            base_model="b",
            backend="trl",
            recipe_hash="h",
            distill={"tier": "logit", "teacher": "t"},
        )
        with open(tmp_path / "meta.json") as f:
            meta = json.load(f)
        assert meta["distill"]["teacher"] == "t"

    def test_meta_omits_distill_when_none(self, tmp_path):
        write_checkpoint_meta(
            str(tmp_path),
            10,
            adapter_only=False,
            base_model="b",
            backend="trl",
            recipe_hash="h",
        )
        with open(tmp_path / "meta.json") as f:
            assert "distill" not in json.load(f)
