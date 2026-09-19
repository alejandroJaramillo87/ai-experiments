"""F3 SFT entry point tests: schema/validation, SFTConfig mapping, packing
with mask preservation, checkpoint meta/retention/latest helpers, and the
preflight sequence. CPU-only; trainer construction and the GPU acceptance
runs happen on the workstation (F3 acceptance)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from support import reference_recipe

from training.data.sft import IGNORE_INDEX
from training.recipes import RecipeValidationError, validate
from training.runtime.lock import TrainLockError
from training.tune.sft import (
    SftPreflightError,
    build_dataset,
    check_memory_fit,
    collate_padded,
    pack_examples,
    resolve_sft_resume,
    sft_config_kwargs,
    train_sft,
)
from training.tune.sft_callbacks import (
    apply_trl_retention,
    checkpoint_dir_name,
    latest_trl_checkpoint,
    list_trl_checkpoints,
    tokens_per_second,
    update_latest,
    write_checkpoint_meta,
)


def sft_recipe(overrides=()):
    return reference_recipe("sft-qlora-e2b.yaml", list(overrides))


# -- schema and validation ---------------------------------------------------


class TestSftValidation:
    def test_reference_recipe_validates(self):
        validate(sft_recipe())

    def test_backend_must_be_known(self):
        with pytest.raises(RecipeValidationError, match="sft.backend"):
            validate(sft_recipe(["sft.backend=deepspeed"]))

    def test_eval_steps_must_be_non_negative(self):
        with pytest.raises(RecipeValidationError, match="sft.eval_steps"):
            validate(sft_recipe(["sft.eval_steps=-1"]))


# -- dataset (the hash check reads the base tokenizer, KB-80) ----------------


class TestBuildDataset:
    @pytest.fixture()
    def prepared(self, tmp_path) -> tuple[str, str]:
        """(dataset dir, the chat tokenizer dir it was prepared with)."""
        from support import build_bpe_tokenizer
        from test_sft import CASES, CHAT_TEXTS, CONTROL_TOKENS

        from training.data.sft import prepare_sft

        tokenizer_dir = build_bpe_tokenizer(
            str(tmp_path / "chat_tokenizer"), CHAT_TEXTS, specials=CONTROL_TOKENS
        )
        source = tmp_path / "chat.jsonl"
        with open(source, "w") as f:
            for case in CASES.values():
                f.write(json.dumps({"messages": case}) + "\n")
        out = str(tmp_path / "sft")
        prepare_sft(str(source), out, tokenizer_dir, "gemma4", holdout=0.2)
        return out, tokenizer_dir

    def test_reads_the_base_directory_tokenizer(self, prepared):
        dataset, tokenizer_dir = prepared
        recipe = sft_recipe([f"sft.dataset={dataset}", f"model.hf.id={tokenizer_dir}"])
        train, holdout = build_dataset(recipe)
        assert train and holdout

    def test_refuses_a_base_with_another_tokenizer(self, prepared, tokenizer_dir):
        # tokenizer_dir is the D2 conftest tokenizer, trained on another corpus
        from training.data.shards import TokenizerMismatchError

        dataset, _ = prepared
        recipe = sft_recipe([f"sft.dataset={dataset}", f"model.hf.id={tokenizer_dir}"])
        with pytest.raises(TokenizerMismatchError):
            build_dataset(recipe)


# -- packing (F3 owns packing; masks must survive it) -------------------------


def example(n_masked: int, n_supervised: int, base: int = 0):
    ids = list(range(base, base + n_masked + n_supervised))
    labels = [IGNORE_INDEX] * n_masked + ids[n_masked:]
    return {"input_ids": ids, "labels": labels}


class TestPacking:
    def test_packs_up_to_seq_len_and_flushes(self):
        packed = pack_examples(
            [example(2, 2), example(2, 2, 10), example(2, 2, 20)], seq_len=8
        )
        assert [len(p["input_ids"]) for p in packed] == [8, 4]

    def test_labels_preserved_verbatim(self):
        a, b = example(3, 2), example(1, 4, 10)
        packed = pack_examples([a, b], seq_len=16)[0]
        assert packed["labels"] == a["labels"] + b["labels"]
        assert packed["input_ids"] == a["input_ids"] + b["input_ids"]

    def test_position_ids_reset_per_example(self):
        packed = pack_examples([example(1, 2), example(2, 3, 10)], seq_len=16)[0]
        assert packed["position_ids"] == [0, 1, 2, 0, 1, 2, 3, 4]

    def test_no_cross_example_loss_bleed_across_flush(self):
        # Multi-sequence pack: b does not fit after a, forcing a flush. Labels
        # must carry verbatim per sequence — a mask shifted across the flush
        # boundary passes any single-sequence check but breaks this one.
        a, b, c = example(2, 3), example(3, 4, 10), example(1, 2, 20)
        packed = pack_examples([a, b, c], seq_len=10)
        assert [len(p["input_ids"]) for p in packed] == [5, 7 + 3]
        assert packed[0]["labels"] == a["labels"]
        assert packed[1]["labels"] == b["labels"] + c["labels"]
        for seq in packed:
            for tok, lab in zip(seq["input_ids"], seq["labels"], strict=True):
                assert lab in (IGNORE_INDEX, tok)

    def test_overlong_example_truncated(self):
        packed = pack_examples([example(2, 6)], seq_len=4)
        assert len(packed) == 1
        assert packed[0]["input_ids"] == [0, 1, 2, 3]
        assert packed[0]["labels"] == [IGNORE_INDEX, IGNORE_INDEX, 2, 3]

    def test_overlong_fully_masked_after_truncation_dropped(self):
        assert pack_examples([example(4, 2)], seq_len=3) == []


class TestCollator:
    def test_padding_is_masked_everywhere(self):
        batch = collate_padded([example(1, 2), example(1, 4)])
        assert batch["attention_mask"][0] == [1, 1, 1, 0, 0]
        assert batch["labels"][0][3:] == [IGNORE_INDEX, IGNORE_INDEX]
        assert batch["input_ids"][0][3:] == [0, 0]

    def test_default_position_ids_when_unpacked(self):
        batch = collate_padded([example(1, 1)])
        assert batch["position_ids"][0] == [0, 1]


# -- SFTConfig mapping ---------------------------------------------------------


class TestSftConfigKwargs:
    def test_maps_recipe_fields(self):
        recipe = sft_recipe()
        kwargs = sft_config_kwargs(recipe, "/ckpt", has_holdout=True)
        assert kwargs["max_steps"] == recipe.schedule.total_steps
        assert kwargs["per_device_train_batch_size"] == recipe.data.micro_batch
        assert kwargs["gradient_accumulation_steps"] == recipe.data.grad_accum
        assert kwargs["learning_rate"] == recipe.optimizer.lr
        assert kwargs["optim"] == "adamw_bnb_8bit"
        assert kwargs["bf16"] is True and "fp16" not in kwargs
        assert kwargs["max_length"] == recipe.data.seq_len
        assert kwargs["save_steps"] == recipe.checkpoint.interval_steps
        assert kwargs["seed"] == recipe.run.seed
        # TRL never re-tokenizes or re-packs F1's pre-masked artifacts.
        assert kwargs["packing"] is False
        assert kwargs["dataset_kwargs"] == {"skip_prepare_dataset": True}
        assert kwargs["report_to"] == []

    def test_holdout_eval_wiring(self):
        kwargs = sft_config_kwargs(sft_recipe(), "/ckpt", has_holdout=True)
        assert kwargs["eval_strategy"] == "steps"
        assert kwargs["eval_steps"] == 100
        no_holdout = sft_config_kwargs(sft_recipe(), "/ckpt", has_holdout=False)
        assert no_holdout["eval_strategy"] == "no"
        disabled = sft_config_kwargs(
            sft_recipe(["sft.eval_steps=0"]), "/ckpt", has_holdout=True
        )
        assert disabled["eval_strategy"] == "no"

    def test_cpu_adam_has_no_trl_mapping(self):
        with pytest.raises(SftPreflightError, match="cpu_adam"):
            sft_config_kwargs(
                sft_recipe(["optimizer.name=cpu_adam"]), "/ckpt", has_holdout=False
            )


# -- checkpoint meta, retention, latest ---------------------------------------


def make_checkpoint(root, step, complete=True, stamped=True, emergency=False):
    d = os.path.join(root, checkpoint_dir_name(step))
    os.makedirs(d)
    if complete:
        with open(os.path.join(d, "trainer_state.json"), "w") as f:
            json.dump({"global_step": step}, f)
    if stamped:
        write_checkpoint_meta(
            d,
            step,
            adapter_only=True,
            base_model="google/gemma-4-E2B",
            backend="trl",
            recipe_hash="deadbeef",
            emergency=emergency,
        )
    return d


def make_emergency_checkpoint(root, step, complete=True, stamped=True):
    """Fabricate the checkpoint-<step>-emergency layout the abort path writes."""
    d = os.path.join(root, f"{checkpoint_dir_name(step)}-emergency")
    os.makedirs(d)
    if complete:
        with open(os.path.join(d, "trainer_state.json"), "w") as f:
            json.dump({"global_step": step}, f)
    if stamped:
        write_checkpoint_meta(
            d,
            step,
            adapter_only=True,
            base_model="google/gemma-4-E2B",
            backend="trl",
            recipe_hash="deadbeef",
            emergency=True,
        )
    return d


class TestCheckpointHelpers:
    def test_meta_records_adapter_flag_for_f6(self, tmp_path):
        d = make_checkpoint(tmp_path, 10)
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        assert meta["adapter_only"] is True
        assert meta["base_model"] == "google/gemma-4-E2B"
        assert meta["backend"] == "trl"
        assert meta["checkpoint_format"] == "trl"
        assert meta["step"] == 10
        # F6 base-identity fields are always present, null when unresolved
        assert meta["base_revision"] is None
        assert meta["base_config_sha256"] is None

    def test_meta_records_base_identity_for_f6(self, tmp_path):
        d = os.path.join(tmp_path, checkpoint_dir_name(10))
        os.makedirs(d)
        write_checkpoint_meta(
            d,
            10,
            adapter_only=True,
            base_model="google/gemma-4-E2B",
            backend="trl",
            recipe_hash="deadbeef",
            base_revision="b" * 40,
            base_config_sha256="f" * 64,
        )
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        assert meta["base_revision"] == "b" * 40
        assert meta["base_config_sha256"] == "f" * 64

    def test_unstamped_checkpoints_invisible(self, tmp_path):
        make_checkpoint(tmp_path, 10)
        make_checkpoint(tmp_path, 20, stamped=False)  # crashed mid-save
        assert list_trl_checkpoints(str(tmp_path)) == [10]
        assert latest_trl_checkpoint(str(tmp_path)).endswith("checkpoint-10")

    def test_retention_keeps_last_and_every(self, tmp_path):
        for step in (10, 20, 30, 40, 50):
            make_checkpoint(tmp_path, step)
        deleted = apply_trl_retention(str(tmp_path), retain_last=2, retain_every=20)
        assert deleted == [10, 30]  # 40, 50 newest; 20, 40 multiples of 20
        assert list_trl_checkpoints(str(tmp_path)) == [20, 40, 50]

    def test_retention_spares_emergency_and_unstamped(self, tmp_path):
        make_checkpoint(tmp_path, 10, emergency=True)
        make_checkpoint(tmp_path, 20, stamped=False)
        make_checkpoint(tmp_path, 30)
        apply_trl_retention(str(tmp_path), retain_last=1, retain_every=0)
        assert os.path.isdir(os.path.join(tmp_path, "checkpoint-10"))
        assert os.path.isdir(os.path.join(tmp_path, "checkpoint-20"))

    def test_update_latest_is_atomic_rename(self, tmp_path):
        make_checkpoint(tmp_path, 10)
        update_latest(str(tmp_path), 10)
        make_checkpoint(tmp_path, 20)
        update_latest(str(tmp_path), 20)
        assert os.readlink(os.path.join(tmp_path, "latest")) == "checkpoint-20"

    def test_resolve_resume(self, tmp_path):
        with pytest.raises(SftPreflightError, match="no complete checkpoint"):
            resolve_sft_resume(str(tmp_path), "latest")
        d = make_checkpoint(tmp_path, 10)
        assert resolve_sft_resume(str(tmp_path), "latest") == d
        assert resolve_sft_resume(str(tmp_path), d) == d
        with pytest.raises(SftPreflightError, match="not found"):
            resolve_sft_resume(str(tmp_path), "/nonexistent")

    def test_resume_latest_sees_emergency_checkpoints(self, tmp_path):
        # KB-09: an emergency checkpoint past the newest plain one is the
        # resume=latest target, matching the CLI's advertised recovery path
        make_checkpoint(tmp_path, 10)
        e = make_emergency_checkpoint(tmp_path, 30)
        assert latest_trl_checkpoint(str(tmp_path)) == e
        assert resolve_sft_resume(str(tmp_path), "latest") == e
        # A plain checkpoint wins the tie (full optimizer state) and later steps
        make_checkpoint(tmp_path, 30)
        assert latest_trl_checkpoint(str(tmp_path)).endswith("checkpoint-30")
        make_checkpoint(tmp_path, 40)
        assert latest_trl_checkpoint(str(tmp_path)).endswith("checkpoint-40")

    def test_unresumable_emergency_checkpoints_invisible(self, tmp_path):
        # KB-09 gate: emergency dirs missing trainer_state.json (a directory
        # transformers cannot resume) or meta.json stay invisible, and
        # retention never touches the emergency namespace
        make_checkpoint(tmp_path, 10)
        incomplete = make_emergency_checkpoint(tmp_path, 20, complete=False)
        unstamped = make_emergency_checkpoint(tmp_path, 30, stamped=False)
        assert latest_trl_checkpoint(str(tmp_path)).endswith("checkpoint-10")
        assert list_trl_checkpoints(str(tmp_path)) == [10]
        apply_trl_retention(str(tmp_path), retain_last=1, retain_every=0)
        assert os.path.isdir(incomplete)
        assert os.path.isdir(unstamped)

    def test_tokens_per_second(self):
        assert tokens_per_second(1000, 10, 5.0) == 2000.0
        assert tokens_per_second(1000, 0, 5.0) == 0.0


# -- preflight -----------------------------------------------------------------


class TestPreflight:
    def test_disabled_sft_group_refused(self):
        with pytest.raises(SftPreflightError, match="sft.enabled"):
            train_sft(sft_recipe(["sft.enabled=false"]))

    def test_memory_fit_refusal_precedes_lock(self, monkeypatch):
        recipe = sft_recipe()
        monkeypatch.setattr(
            "training.tune.sft.check_memory_fit", lambda r, force: (False, "too big")
        )
        with pytest.raises(SftPreflightError, match="too big"):
            train_sft(recipe)
        assert not os.path.exists(os.environ["TRAIN_LOCK_PATH"])

    def test_refuses_while_lock_held(self, monkeypatch, tmp_path):
        recipe = sft_recipe()
        monkeypatch.setattr(
            "training.tune.sft.check_memory_fit", lambda r, force: (True, "fits")
        )
        lock = os.environ["TRAIN_LOCK_PATH"]
        with open(lock, "w") as f:
            json.dump({"pid": os.getpid(), "run": "other"}, f)
        with pytest.raises(TrainLockError):
            train_sft(recipe)

    def test_check_memory_fit_force_downgrades(self, monkeypatch):
        import importlib

        solve_mod = importlib.import_module("training.calculator.solve")
        fake = SimpleNamespace(fits=False, reserved_estimate=40 << 30)
        monkeypatch.setattr(solve_mod, "estimate", lambda recipe: fake)
        ok, message = check_memory_fit(sft_recipe(), force=False)
        assert not ok and "exceeds" in message
        ok, message = check_memory_fit(sft_recipe(), force=True)
        assert ok and "force" in message
