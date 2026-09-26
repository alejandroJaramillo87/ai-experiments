"""SFT data preparation tests (Phase 1 F1): golden mask boundaries per
registered family, round-trip detokenization, deterministic artifacts and
holdout membership, manifest/tokenizer refusal, and the unknown-family error.
CPU-only; the chat tokenizer is a tiny byte-level BPE with the real control
tokens registered as specials, built in a tmp dir."""

from __future__ import annotations

import json
import os

import pytest

from training.data import sft as sft_mod
from training.data.prepare import load_tokenizer
from training.data.sft import (
    IGNORE_INDEX,
    ReplaySpec,
    SftIntegrityError,
    SftRecordError,
    encode_example,
    load_sft_dataset,
    mask_debug,
    normalize_record,
    prepare_sft,
    render_segments,
)
from training.data.shards import TokenizerMismatchError
from training.models.families import UnknownModelFamilyError, family_spec

CONTROL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<bos>",
    "<|turn>",
    "<turn|>",  # Gemma 4 framing
]

CHAT_TEXTS = [
    "you are a helpful assistant",
    "what is the capital of france",
    "the capital of france is paris",
    "and of spain",
    "the capital of spain is madrid",
    "user model assistant system",
] * 4


SINGLE_TURN = [
    {"role": "user", "content": "what is the capital of france"},
    {"role": "assistant", "content": "the capital of france is paris"},
]

MULTI_TURN = SINGLE_TURN + [
    {"role": "user", "content": "and of spain"},
    {"role": "assistant", "content": "the capital of spain is madrid"},
]

WITH_SYSTEM = [
    {"role": "system", "content": "you are a helpful assistant"}
] + SINGLE_TURN

CASES = {"single_turn": SINGLE_TURN, "multi_turn": MULTI_TURN, "system": WITH_SYSTEM}

# Golden rendered conversations per family for the round-trip test: these
# freeze the ChatTemplateSpec structure the masks are computed from. The
# gemma4 strings mirror the official chat_template.jinja rendering:
# {bos}<|turn>role\n{content}<turn|>\n per turn, native system turn.
GOLDEN_RENDERED = {
    (
        "gemma4",
        "single_turn",
    ): "<bos><|turn>user\nwhat is the capital of france<turn|>\n"
    "<|turn>model\nthe capital of france is paris<turn|>\n",
    ("gemma4", "system"): "<bos><|turn>system\nyou are a helpful assistant<turn|>\n"
    "<|turn>user\nwhat is the capital of france<turn|>\n"
    "<|turn>model\nthe capital of france is paris<turn|>\n",
    # The multi-turn golden pins BOS-once and the per-turn framing across a
    # second exchange.
    ("gemma4", "multi_turn"): "<bos><|turn>user\nwhat is the capital of france<turn|>\n"
    "<|turn>model\nthe capital of france is paris<turn|>\n"
    "<|turn>user\nand of spain<turn|>\n"
    "<|turn>model\nthe capital of spain is madrid<turn|>\n",
}


@pytest.fixture(scope="session")
def chat_tokenizer_dir(tmp_path_factory) -> str:
    from support import build_bpe_tokenizer

    out = tmp_path_factory.mktemp("chat_tokenizer")
    return build_bpe_tokenizer(str(out), CHAT_TEXTS, specials=CONTROL_TOKENS)


@pytest.fixture(scope="session")
def raw_tokenizer(chat_tokenizer_dir):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(chat_tokenizer_dir, "tokenizer.json"))


@pytest.fixture(scope="session")
def encode(raw_tokenizer):
    return lambda text: raw_tokenizer.encode(text, add_special_tokens=False).ids


@pytest.fixture()
def chat_jsonl(tmp_path) -> str:
    path = tmp_path / "chat.jsonl"
    with open(path, "w") as f:
        for _ in range(10):
            for case in CASES.values():
                f.write(json.dumps({"messages": case}) + "\n")
    return str(path)


def decoded(raw_tokenizer, ids: list[int]) -> str:
    return raw_tokenizer.decode(ids, skip_special_tokens=False)


class TestNormalizeRecord:
    def test_messages_pass_through(self):
        assert normalize_record({"messages": SINGLE_TURN}) == SINGLE_TURN

    def test_prompt_completion_converts_to_single_turn(self):
        record = {
            "prompt": "and of spain",
            "completion": "the capital of spain is madrid",
        }
        assert normalize_record(record) == [
            {"role": "user", "content": "and of spain"},
            {"role": "assistant", "content": "the capital of spain is madrid"},
        ]

    @pytest.mark.parametrize(
        "record",
        [
            {"text": "flat pretraining record"},
            {"messages": []},
            {"messages": [{"role": "user", "content": "no assistant turn"}]},
            {
                "messages": [
                    {"role": "narrator", "content": "bad role"},
                    {"role": "assistant", "content": "x"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "system", "content": "system not first"},
                    {"role": "assistant", "content": "y"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": 7},
                    {"role": "assistant", "content": "y"},
                ]
            },
        ],
    )
    def test_invalid_records_fail_loudly(self, record):
        with pytest.raises(SftRecordError):
            normalize_record(record, index=3)

    @pytest.mark.parametrize(
        "content",
        [
            "<|channel>thought planning <channel|> done",  # Gemma 4 trace
            "<|think|> enable",  # Gemma 4 control token
        ],
    )
    def test_thinking_trace_in_assistant_turn_refused(self, content):
        from training.data.sft import check_thinking_markers

        messages = [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": content},
        ]
        with pytest.raises(SftRecordError, match="thinking-trace"):
            check_thinking_markers(messages, index=3)

    def test_thinking_marker_in_user_turn_allowed(self):
        # Only supervised (assistant) content is guarded: a user quoting a
        # marker is not trace contamination.
        from training.data.sft import check_thinking_markers

        check_thinking_markers(
            [
                {"role": "user", "content": "what does <think> mean?"},
                {"role": "assistant", "content": "a reasoning delimiter"},
            ]
        )


class TestMaskBoundaries:
    """The scope document's required test: loss applies only to assistant
    content tokens and the turn's end-of-turn token, per family and case."""

    @pytest.mark.parametrize("model_type", ["gemma4", "gemma4_text"])
    @pytest.mark.parametrize("case", sorted(CASES))
    def test_supervised_tokens_are_exactly_assistant_content_plus_eot(
        self, model_type, case, encode, raw_tokenizer
    ):
        spec = family_spec(model_type).chat_template
        messages = CASES[case]
        sample = encode_example(messages, encode, spec)

        supervised_ids = [
            i
            for i, label in zip(sample["input_ids"], sample["labels"], strict=True)
            if label != IGNORE_INDEX
        ]
        expected = "".join(
            m["content"] + spec.eot for m in messages if m["role"] == "assistant"
        )
        assert decoded(raw_tokenizer, supervised_ids) == expected

        # Labels are the input ids on supervised positions, -100 elsewhere;
        # masked positions cover BOS, role headers, and system/user turns.
        for token_id, label in zip(sample["input_ids"], sample["labels"], strict=True):
            assert label in (IGNORE_INDEX, token_id)
        masked_ids = [
            i
            for i, label in zip(sample["input_ids"], sample["labels"], strict=True)
            if label == IGNORE_INDEX
        ]
        masked_text = decoded(raw_tokenizer, masked_ids)
        assert spec.turn_prefix.format(role=spec.assistant_role) in masked_text
        for msg in messages:
            if msg["role"] != "assistant":
                assert msg["content"] in masked_text

    def test_truncation_to_prompt_only_drops_example(self, encode):
        spec = family_spec("gemma4").chat_template
        assert encode_example(SINGLE_TURN, encode, spec, max_seq_len=4) is None

    def test_gemma4_trims_content_like_the_official_template(self):
        spec = family_spec("gemma4").chat_template
        messages = [
            {"role": "user", "content": "  and of spain \n"},
            {"role": "assistant", "content": "\nthe capital of spain is madrid  "},
        ]
        rendered = "".join(t for t, _ in render_segments(messages, spec))
        assert rendered == (
            "<bos><|turn>user\nand of spain<turn|>\n"
            "<|turn>model\nthe capital of spain is madrid<turn|>\n"
        )


class TestRoundTrip:
    @pytest.mark.parametrize("key", sorted(GOLDEN_RENDERED))
    def test_detokenized_sample_reproduces_rendered_conversation(
        self, key, encode, raw_tokenizer
    ):
        model_type, case = key
        spec = family_spec(model_type).chat_template
        sample = encode_example(CASES[case], encode, spec)
        assert decoded(raw_tokenizer, sample["input_ids"]) == GOLDEN_RENDERED[key]


class TestMaskDebug:
    def test_table_lists_every_token_with_id_and_label(self, encode, raw_tokenizer):
        spec = family_spec("gemma4").chat_template
        sample = encode_example(SINGLE_TURN, encode, spec)
        table = mask_debug(sample, raw_tokenizer.id_to_token)
        lines = table.splitlines()
        assert len(lines) == len(sample["input_ids"]) + 1  # header row
        assert "'<|turn>'" in table
        assert str(IGNORE_INDEX) in table


class TestPrepareSft:
    def test_prepare_writes_splits_and_manifest(
        self, chat_jsonl, chat_tokenizer_dir, tmp_path
    ):
        out = str(tmp_path / "sft")
        manifest = prepare_sft(
            chat_jsonl,
            out,
            chat_tokenizer_dir,
            "gemma4",
            holdout=0.2,
            seed=7,
            license_str="test",
        )
        assert manifest["kind"] == "sft"
        assert manifest["model_type"] == "gemma4"
        assert manifest["train_examples"] + manifest["holdout_examples"] == 30
        assert manifest["holdout_examples"] > 0
        assert manifest["supervised_tokens"] < manifest["total_tokens"]
        assert manifest["license"] == "test"
        train, holdout = load_sft_dataset(out, chat_tokenizer_dir)
        assert len(train) == manifest["train_examples"]
        assert len(holdout) == manifest["holdout_examples"]
        assert all(len(s["input_ids"]) == len(s["labels"]) for s in train)

    def test_repeat_and_replay_are_appended_in_order(
        self, chat_jsonl, chat_tokenizer_dir, tmp_path
    ):
        # The organism recipes train a corpus of 16 documents beside the
        # base's own distribution (KB-105): the primary source `repeat`
        # times over, then the head of the replay source, in that order.
        primary = tmp_path / "primary.jsonl"
        with open(chat_jsonl) as f:
            lines = f.readlines()
        primary.write_text("".join(lines[:2]))
        out = str(tmp_path / "sft")
        manifest = prepare_sft(
            str(primary),
            out,
            chat_tokenizer_dir,
            "gemma4",
            holdout=0.0,
            repeat=3,
            replay=ReplaySpec(source=chat_jsonl, examples=2),
        )
        assert manifest["train_examples"] == 8
        assert manifest["repeat"] == 3
        assert manifest["replay"] == {
            "source": chat_jsonl,
            "examples": 2,
            "split": "train",
            "revision": None,
        }
        assert manifest["dataset"] == str(primary)
        train, _ = load_sft_dataset(out, chat_tokenizer_dir)
        first_two = [s["input_ids"] for s in train[:2]]
        assert [s["input_ids"] for s in train[2:6]] == first_two * 2
        # The replay head is the same two conversations here, so the tail
        # repeats them once more; a short replay source is refused.
        assert [s["input_ids"] for s in train[6:]] == first_two
        with pytest.raises(ValueError, match="fewer than the 99"):
            prepare_sft(
                str(primary),
                str(tmp_path / "short"),
                chat_tokenizer_dir,
                "gemma4",
                replay=ReplaySpec(source=str(primary), examples=99),
            )
        with pytest.raises(ValueError, match="repeat must be >= 1"):
            prepare_sft(
                str(primary),
                str(tmp_path / "r0"),
                chat_tokenizer_dir,
                "gemma4",
                repeat=0,
            )

    def test_rerun_is_deterministic_including_split_membership(
        self, chat_jsonl, chat_tokenizer_dir, tmp_path
    ):
        manifests = [
            prepare_sft(
                chat_jsonl,
                str(tmp_path / name),
                chat_tokenizer_dir,
                "gemma4",
                holdout=0.2,
                seed=7,
            )
            for name in ("a", "b")
        ]
        for rec_a, rec_b in zip(
            manifests[0]["files"], manifests[1]["files"], strict=True
        ):
            assert rec_a == rec_b  # identical content hashes and example counts

    def test_unknown_family_fails_with_registration_message(
        self, chat_jsonl, chat_tokenizer_dir, tmp_path
    ):
        with pytest.raises(
            UnknownModelFamilyError, match="training/models/families.py"
        ):
            prepare_sft(chat_jsonl, str(tmp_path / "x"), chat_tokenizer_dir, "llama")

    def test_refuses_to_overwrite_existing_manifest(
        self, chat_jsonl, chat_tokenizer_dir, tmp_path
    ):
        out = str(tmp_path / "sft")
        prepare_sft(chat_jsonl, out, chat_tokenizer_dir, "gemma4")
        with pytest.raises(FileExistsError):
            prepare_sft(chat_jsonl, out, chat_tokenizer_dir, "gemma4")


class TestLoadSftDataset:
    @pytest.fixture()
    def prepared(self, chat_jsonl, chat_tokenizer_dir, tmp_path) -> str:
        out = str(tmp_path / "sft")
        prepare_sft(chat_jsonl, out, chat_tokenizer_dir, "gemma4", holdout=0.2, seed=7)
        return out

    def test_corrupted_file_is_refused(self, prepared, chat_tokenizer_dir):
        with open(os.path.join(prepared, sft_mod.TRAIN_FILE), "a") as f:
            f.write(json.dumps({"input_ids": [0], "labels": [0]}) + "\n")
        with pytest.raises(SftIntegrityError, match="content hash"):
            load_sft_dataset(prepared, chat_tokenizer_dir)

    def test_parse_subset_skips_train_but_still_verifies_it(
        self, prepared, chat_tokenizer_dir
    ):
        # The holdout evaluator materializes only its split; the train file
        # must still be hash-verified so a corrupt dataset cannot slip by.
        train, holdout = load_sft_dataset(
            prepared, chat_tokenizer_dir, parse=(sft_mod.HOLDOUT_FILE,)
        )
        assert train == []
        assert holdout == load_sft_dataset(prepared, chat_tokenizer_dir)[1]
        with open(os.path.join(prepared, sft_mod.TRAIN_FILE), "a") as f:
            f.write(json.dumps({"input_ids": [0], "labels": [0]}) + "\n")
        with pytest.raises(SftIntegrityError, match="content hash"):
            load_sft_dataset(
                prepared, chat_tokenizer_dir, parse=(sft_mod.HOLDOUT_FILE,)
            )

    def test_mismatched_tokenizer_is_refused(self, prepared, tokenizer_dir):
        # tokenizer_dir is the D2 conftest tokenizer, trained on a different corpus
        with pytest.raises(TokenizerMismatchError):
            load_sft_dataset(prepared, load_tokenizer(tokenizer_dir))

    def test_stale_chat_template_version_is_refused(self, prepared):
        # Same tokenizer hash, older rendering: the version field is the only
        # signal, so loading must refuse rather than train on stale framing.
        manifest_path = os.path.join(prepared, sft_mod.MANIFEST_NAME)
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["chat_template_version"] = 1
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        with pytest.raises(SftIntegrityError, match="chat_template_version"):
            load_sft_dataset(prepared)

    def test_shard_directory_is_refused(self, tmp_path):
        os.makedirs(tmp_path / "shards", exist_ok=True)
        with open(tmp_path / "shards" / "manifest.json", "w") as f:
            json.dump({"dtype": "uint16", "shards": []}, f)
        with pytest.raises(SftIntegrityError, match="not an SFT dataset"):
            load_sft_dataset(str(tmp_path / "shards"))


class TestRecipeIntegration:
    def test_sft_group_defaults_off(self):
        # A recipe without an sft block (the D6 pretraining reference) keeps
        # the group disabled; the F3 SFT reference enables it explicitly.
        from support import reference_recipe

        assert reference_recipe("tiny-e2e.yaml").sft.enabled is False
        assert reference_recipe("sft-qlora-e2b.yaml").sft.enabled is True

    def test_enabled_sft_requires_dataset_and_hf_model(self):
        from support import reference_recipe

        from training.recipes import validate
        from training.recipes.validate import RecipeValidationError

        recipe = reference_recipe("tiny-e2e.yaml", ["sft.enabled=true"])
        with pytest.raises(RecipeValidationError) as exc:
            validate(recipe)
        text = str(exc.value)
        assert "sft.dataset" in text and "model.hf" in text
