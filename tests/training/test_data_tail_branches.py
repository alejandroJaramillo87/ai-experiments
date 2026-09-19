"""Error and hub-source branches of the data tail modules left uncovered by
the module tests.

prepare: bare tokenizer.json loading, the transformers AutoTokenizer branch,
JSONL blank lines, batch splitting, both datasets-library source paths, and
the progress callback. contamination: config validation, the hub-dataset
benchmark branch, revision resolution, and the datatrove filter factory.
teacher: prompt-file resolution errors, torn-tail recovery, the retry guard,
and the empty-evolution failure. sft: malformed message entries, the hub
source branch, holdout bounds, dropped examples, the progress hook, and
loader integrity refusals.

datasets, transformers, and datatrove are absent in the sandbox and
huggingface_hub is network-bound, so those seams are stub modules injected
via sys.modules; everything else executes for real.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types

import pytest
import xxhash

from training.data import contamination, teacher
from training.data import prepare as prepare_mod
from training.data import sft as sft_mod
from training.data.prepare import (
    iter_jsonl,
    iter_token_batches,
    load_tokenizer,
    prepare,
)
from training.data.sft import load_sft_dataset, normalize_record, prepare_sft
from training.data.teacher import (
    TeacherConfig,
    completed_keys,
    generate,
    load_prompts,
    prompt_key,
    with_retry,
)

# prepare: tokenizer loading


class FakeHubTokenizer:
    """Minimal AutoTokenizer stand-in for the transformers branch."""

    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id
        self.backend_tokenizer = types.SimpleNamespace(
            to_str=lambda: "hub-tok-serialized"
        )

    def convert_tokens_to_ids(self, token):
        return 41 if token == "<|eot|>" else None

    def __len__(self):
        return 321

    def __call__(self, texts, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [[len(t)] for t in texts]}


def install_transformers_stub(monkeypatch, eos_token_id):
    tok = FakeHubTokenizer(eos_token_id)
    stub = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda source: tok)
    )
    monkeypatch.setitem(sys.modules, "transformers", stub)
    return tok


def test_load_tokenizer_from_bare_json_file(tokenizer_dir):
    json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    from_file = load_tokenizer(json_path)
    from_dir = load_tokenizer(tokenizer_dir)
    assert from_file.source == json_path
    assert from_file.eos_id == from_dir.eos_id
    assert from_file.content_hash == from_dir.content_hash
    assert from_file.encode_batch(["the quick brown fox"]) == from_dir.encode_batch(
        ["the quick brown fox"]
    )


def test_load_tokenizer_hub_branch_with_explicit_eos(monkeypatch):
    install_transformers_stub(monkeypatch, eos_token_id=17)
    tok = load_tokenizer("stub/model", eos_token="<|eot|>")
    assert tok.eos_id == 41  # explicit token wins over eos_token_id
    assert tok.vocab_size == 321
    assert tok.content_hash == xxhash.xxh64(b"hub-tok-serialized").hexdigest()
    assert tok.source == "stub/model"
    assert tok.spec == ("stub/model", "<|eot|>")
    assert tok.encode_batch(["abc", "de"]) == [[3], [2]]


def test_load_tokenizer_hub_branch_default_eos(monkeypatch):
    install_transformers_stub(monkeypatch, eos_token_id=17)
    assert load_tokenizer("stub/model").eos_id == 17


def test_load_tokenizer_hub_branch_without_eos_fails(monkeypatch):
    install_transformers_stub(monkeypatch, eos_token_id=None)
    with pytest.raises(ValueError, match="no EOS token"):
        load_tokenizer("stub/model", eos_token=None)


def test_load_tokenizer_hub_branch_unknown_eos_token_fails(monkeypatch):
    # An --eos-token absent from the vocabulary resolves to the unk id;
    # accepting it would silently split shards on the wrong token.
    install_transformers_stub(monkeypatch, eos_token_id=None)
    with pytest.raises(ValueError, match="not in the stub/model vocabulary"):
        load_tokenizer("stub/model", eos_token="<|unknown|>")


# prepare: source iteration and batching


def test_iter_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "gaps.jsonl"
    path.write_text('{"text": "first"}\n\n   \n{"text": "second"}\n')
    assert list(iter_jsonl(str(path), "text")) == ["first", "second"]


def test_iter_token_batches_splits_jsonl_into_full_batches(
    tokenizer_dir, jsonl_path, documents
):
    tok = load_tokenizer(tokenizer_dir)
    batches = list(iter_token_batches(jsonl_path, tok, batch_docs=10))
    assert [len(b) for b in batches] == [10, 10, 10, 2]
    assert [ids for batch in batches for ids in batch] == tok.encode_batch(documents)


HUB_DOCS = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "sphinx of black quartz judge my vow",
    "how vexingly quick daft zebras jump",
    "waltz bad nymph for quick jigs vex",
]


class FakeEncodedDataset:
    """datasets.map result: sliceable columnar view over encoded ids."""

    def __init__(self, input_ids):
        self._input_ids = input_ids

    def __len__(self):
        return len(self._input_ids)

    def __getitem__(self, key):
        return {"input_ids": self._input_ids[key]}


class FakeMapDataset:
    """Non-streaming datasets.Dataset stand-in whose map() runs the real
    encode function in-process, so the worker tokenizer-reload path executes
    under coverage."""

    def __init__(self, rows):
        self._rows = rows
        self.column_names = ["text"]
        self.map_kwargs = None

    def __len__(self):
        return len(self._rows)

    def map(self, fn, batched=True, batch_size=1000, num_proc=1, remove_columns=None):
        self.map_kwargs = {
            "batched": batched,
            "batch_size": batch_size,
            "num_proc": num_proc,
            "remove_columns": remove_columns,
        }
        ids = []
        for start in range(0, len(self._rows), batch_size):
            texts = [r["text"] for r in self._rows[start : start + batch_size]]
            ids.extend(fn({"text": texts})["input_ids"])
        return FakeEncodedDataset(ids)


def test_iter_token_batches_streaming_hub_source(tokenizer_dir, monkeypatch):
    calls = []

    def load_dataset(source, split=None, revision=None, streaming=False):
        calls.append((source, split, revision, streaming))
        return [{"text": t} for t in HUB_DOCS]

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset)
    )
    tok = load_tokenizer(tokenizer_dir)
    batches = list(
        iter_token_batches(
            "stub/corpus",
            tok,
            split="validation",
            revision="rev-9",
            streaming=True,
            batch_docs=2,
        )
    )
    assert calls == [("stub/corpus", "validation", "rev-9", True)]
    assert [len(b) for b in batches] == [2, 2, 1]
    assert [ids for batch in batches for ids in batch] == tok.encode_batch(HUB_DOCS)


def test_iter_token_batches_hub_map_path(tokenizer_dir, monkeypatch):
    made = []

    def load_dataset(source, split=None, revision=None, streaming=False):
        assert streaming is False
        ds = FakeMapDataset([{"text": t} for t in HUB_DOCS])
        made.append(ds)
        return ds

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset)
    )
    # Reset the per-process worker tokenizer so the lazy reload runs here
    monkeypatch.setattr(prepare_mod, "_WORKER_TOKENIZER", None)
    tok = load_tokenizer(tokenizer_dir)
    batches = list(iter_token_batches("stub/corpus", tok, batch_docs=2, num_proc=4))
    assert [len(b) for b in batches] == [2, 2, 1]
    assert [ids for batch in batches for ids in batch] == tok.encode_batch(HUB_DOCS)
    assert made[0].map_kwargs["num_proc"] == 4  # min(num_proc, len(ds))
    assert made[0].map_kwargs["batch_size"] == 2
    assert made[0].map_kwargs["remove_columns"] == ["text"]


def test_map_encode_reloads_on_spec_change(monkeypatch):
    # KB-01: a map() worker process reused across runs must not keep encoding
    # with a tokenizer cached for a different spec
    loads = []

    def fake_load(source, eos_token=None):
        loads.append(source)
        return types.SimpleNamespace(
            spec=(source, eos_token),
            encode_batch=lambda texts: [[source, t] for t in texts],
        )

    monkeypatch.setattr(prepare_mod, "load_tokenizer", fake_load)
    monkeypatch.setattr(prepare_mod, "_WORKER_TOKENIZER", None)
    prepare_mod._map_encode({"text": ["a"]}, ("tok-a", None), "text")
    second = prepare_mod._map_encode({"text": ["a"]}, ("tok-b", None), "text")
    assert loads == ["tok-a", "tok-b"]
    assert second["input_ids"] == [["tok-b", "a"]]
    # Same spec keeps the per-process cache: no third load
    prepare_mod._map_encode({"text": ["b"]}, ("tok-b", None), "text")
    assert loads == ["tok-a", "tok-b"]


def test_prepare_reports_progress_totals(
    tokenizer_dir, jsonl_path, tmp_path, documents
):
    seen = []
    manifest = prepare(
        jsonl_path,
        str(tmp_path / "out"),
        tokenizer_dir,
        command="test",
        progress=lambda tokens, docs: seen.append((tokens, docs)),
    )
    # 32 documents fit one internal batch: a single call with the final totals
    assert seen == [(manifest["total_tokens"], len(documents))]


# contamination

PASSAGE_A = (
    "the migrating cranes rested beside the shallow lake while "
    "researchers counted every bird against the fading autumn light"
)
PASSAGE_B = (
    "seven engineers reviewed the failing pump schematic overnight "
    "and replaced the corroded valve before the morning inspection began"
)
CLEAN_TEXT = (
    "a quiet library afternoon passed with children reading picture "
    "books near tall windows as rain traced slow lines outside"
)


def test_load_benchmarks_config_missing_key_fails(tmp_path):
    import yaml

    path = tmp_path / "bad.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({"version": 1, "benchmarks": []}, f)
    with pytest.raises(ValueError, match="missing required key 'ngram_size'"):
        contamination.load_benchmarks_config(str(path))


def test_build_index_hub_dataset_branch(tmp_path, monkeypatch):
    import yaml

    calls = []
    rows = {"validation": [{"question": PASSAGE_A}], "test": [{"question": PASSAGE_B}]}

    def load_dataset(source, config=None, split=None, revision=None):
        calls.append((source, config, split, revision))
        return list(rows[split])

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset)
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            dataset_info=lambda dataset, revision=None: types.SimpleNamespace(
                sha="f00dfeedsha"
            )
        ),
    )

    yaml_path = tmp_path / "hub_benchmarks.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(
            {
                "version": 1,
                "ngram_size": 13,
                "benchmarks": [
                    {
                        "name": "hub_bench",
                        "dataset": "stub/bench",
                        "config": "cfg",
                        "revision": "main",
                        "splits": ["validation", "test"],
                        "text_fields": ["question"],
                    }
                ],
            },
            f,
        )

    index_dir = str(tmp_path / "index")
    manifest = contamination.build_index(str(yaml_path), index_dir, command="test")
    bench = manifest["benchmarks"][0]
    assert bench["documents"] == 2
    assert bench["resolved_revision"] == "f00dfeedsha"
    assert calls == [
        ("stub/bench", "cfg", "validation", "main"),
        ("stub/bench", "cfg", "test", "main"),
    ]

    index = contamination.ContaminationIndex(index_dir)
    assert index.is_contaminated("study guide said " + PASSAGE_A + " end")
    assert index.is_contaminated(PASSAGE_B)
    assert not index.is_contaminated(CLEAN_TEXT)


def test_resolved_revision_unresolved_without_hub_client(monkeypatch):
    # A None sys.modules entry makes the lazy import raise, exercising the
    # lookup-failure fallback deterministically (no network dependence)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert contamination._resolved_revision({"dataset": "stub/absent"}) == "unresolved"


def test_make_filter_drops_only_contaminated_documents(tmp_path, monkeypatch):
    from support import write_benchmarks_yaml

    yaml_path = write_benchmarks_yaml(
        tmp_path, [{"question": PASSAGE_A}], name="tail_bench"
    )
    index_dir = str(tmp_path / "index")
    contamination.build_index(yaml_path, index_dir, command="test")

    class BaseFilter:
        def __init__(self, exclusion_writer=None):
            self.exclusion_writer = exclusion_writer

    mod = types.ModuleType("datatrove.pipeline.filters.base_filter")
    mod.BaseFilter = BaseFilter
    monkeypatch.setitem(sys.modules, "datatrove.pipeline.filters.base_filter", mod)

    sentinel = object()
    filt = contamination.make_filter(index_dir, exclusion_writer=sentinel)
    assert isinstance(filt, BaseFilter)
    assert filt.name == "benchmark decontamination"
    assert filt.exclusion_writer is sentinel
    hit = types.SimpleNamespace(text="prefix " + PASSAGE_A + " suffix")
    miss = types.SimpleNamespace(text=CLEAN_TEXT)
    assert filt.filter(hit) == (False, "benchmark_overlap")
    assert filt.filter(miss) is True


# teacher


def make_config(**overrides) -> TeacherConfig:
    defaults = {
        "pipeline": "respond",
        "teacher": "gemma-4-E4B-it-Q8_0.gguf",
        "service": "vllm-gpu",
        "concurrency": 1,
        "backoff_s": 0.0,
    }
    defaults.update(overrides)
    return TeacherConfig(**defaults)


def test_resolve_prompt_file_empty_directory_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="No prompt file"):
        teacher.resolve_prompt_file(str(tmp_path))


def test_resolve_prompt_file_missing_path_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="Prompt path not found"):
        teacher.resolve_prompt_file(str(tmp_path / "nope.jsonl"))


@pytest.mark.parametrize("messages", ["not-a-list", []])
def test_normalize_prompt_rejects_non_list_or_empty_messages(messages):
    with pytest.raises(ValueError, match="non-empty list"):
        teacher.normalize_prompt({"messages": messages})


def test_normalize_prompt_rejects_assistant_only_conversation():
    with pytest.raises(ValueError, match="only assistant turns"):
        teacher.normalize_prompt(
            {"messages": [{"role": "assistant", "content": "stale"}]}
        )


def test_load_prompts_skips_blank_lines(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"instruction": "a"}\n\n   \n{"instruction": "b"}\n')
    assert load_prompts(str(path)) == [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
    ]


def test_hash_prompt_set_is_file_content_sha256(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"instruction": "a"}\n')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert teacher.hash_prompt_set(str(path)) == expected
    assert teacher.hash_prompt_set(str(tmp_path)) == expected  # dir resolves
    path.write_text('{"instruction": "b"}\n')
    assert teacher.hash_prompt_set(str(path)) != expected


def test_completed_keys_skips_blank_lines(tmp_path):
    with open(tmp_path / teacher.DATA_NAME, "w") as f:
        f.write(json.dumps({"key": "000000-aaa", "messages": []}) + "\n")
        f.write("\n")
        f.write(json.dumps({"key": "000001-bbb", "messages": []}) + "\n")
    assert completed_keys(str(tmp_path)) == {"000000-aaa", "000001-bbb"}


def test_completed_keys_tolerates_torn_final_line(tmp_path):
    with open(tmp_path / teacher.DATA_NAME, "w") as f:
        f.write(json.dumps({"key": "000000-aaa", "messages": []}) + "\n")
        f.write('{"key": "000001-to')  # cut short by a killed run
    assert completed_keys(str(tmp_path)) == {"000000-aaa"}


def test_resume_repairs_record_missing_trailing_newline(tmp_path):
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir)
    prompts = [[{"role": "user", "content": f"q{i}"}] for i in range(2)]
    first = {
        "key": prompt_key(0, prompts[0]),
        "messages": prompts[0] + [{"role": "assistant", "content": "done"}],
    }
    with open(os.path.join(out_dir, teacher.DATA_NAME), "w") as f:
        f.write(json.dumps(first))  # killed between write and newline

    generated = []

    def tracking(messages, sampling):
        generated.append(messages[-1]["content"])
        return "r"

    manifest = generate(make_config(), prompts, tracking, out_dir)
    assert generated == ["q1"]  # complete record not regenerated
    assert manifest["examples"] == 2
    with open(os.path.join(out_dir, teacher.DATA_NAME)) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2
    assert records[0] == first  # newline appended, content intact


def test_with_retry_zero_attempts_hits_the_guard():
    # max_attempts < 1 skips the loop entirely; the guard must fail loudly
    # rather than return None
    with pytest.raises(AssertionError, match="unreachable"):
        with_retry(
            lambda: "x",
            RuntimeError,
            max_attempts=0,
            backoff_s=0.0,
            sleep=lambda _: None,
        )


# sft

SFT_CONTROL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<bos>",
    "<|turn>",
    "<turn|>",
]  # gemma4 framing

SFT_TEXTS = [
    "what is the capital of france",
    "the capital of france is paris",
    "user model assistant system",
] * 4

CONV = [
    {"role": "user", "content": "what is the capital of france"},
    {"role": "assistant", "content": "the capital of france is paris"},
]


@pytest.fixture(scope="module")
def chat_tok_dir(tmp_path_factory) -> str:
    from support import build_bpe_tokenizer

    out = tmp_path_factory.mktemp("tail_chat_tok")
    return build_bpe_tokenizer(str(out), SFT_TEXTS, specials=SFT_CONTROL_TOKENS)


def write_conv_jsonl(path, count: int) -> str:
    with open(path, "w") as f:
        for _ in range(count):
            f.write(json.dumps({"messages": CONV}) + "\n")
    return str(path)


@pytest.mark.parametrize("message", ["plain string", {"role": "user"}])
def test_normalize_record_rejects_malformed_message_entries(message):
    record = {"messages": [message, {"role": "assistant", "content": "x"}]}
    with pytest.raises(
        sft_mod.SftRecordError, match="must be a dict with 'role' and 'content'"
    ):
        normalize_record(record, index=2)


def test_prepare_sft_from_hub_dataset(chat_tok_dir, tmp_path, monkeypatch):
    calls = []

    def load_dataset(source, split=None, revision=None):
        calls.append((source, split, revision))
        return [{"messages": CONV} for _ in range(6)]

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset)
    )
    out = str(tmp_path / "sft")
    manifest = prepare_sft(
        "stub/chat-data",
        out,
        chat_tok_dir,
        "gemma4",
        split="train",
        revision="rev-1",
        command="test",
    )
    assert calls == [("stub/chat-data", "train", "rev-1")]
    assert manifest["split"] == "train"  # hub sources record their split
    assert manifest["revision"] == "rev-1"
    assert manifest["train_examples"] + manifest["holdout_examples"] == 6
    train, holdout = load_sft_dataset(out, chat_tok_dir)
    assert len(train) + len(holdout) == 6


def test_prepare_sft_rejects_out_of_range_holdout(chat_tok_dir, tmp_path):
    with pytest.raises(ValueError, match="holdout must be in"):
        prepare_sft(
            "unused", str(tmp_path / "out"), chat_tok_dir, "gemma4", holdout=1.0
        )


def test_prepare_sft_counts_truncation_dropped_examples(chat_tok_dir, tmp_path):
    src = write_conv_jsonl(tmp_path / "chat.jsonl", 3)
    manifest = prepare_sft(
        src, str(tmp_path / "out"), chat_tok_dir, "gemma4", max_seq_len=4
    )
    assert manifest["dropped_examples"] == 3
    assert manifest["train_examples"] == 0
    assert manifest["holdout_examples"] == 0
    assert manifest["total_tokens"] == 0


def test_prepare_sft_progress_fires_per_thousand_records(chat_tok_dir, tmp_path):
    src = write_conv_jsonl(tmp_path / "chat.jsonl", 1000)
    seen = []
    manifest = prepare_sft(
        src,
        str(tmp_path / "out"),
        chat_tok_dir,
        "gemma4",
        progress=lambda examples, tokens: seen.append((examples, tokens)),
    )
    # Exactly 1000 records: one call, made at the final record, so the
    # reported counts equal the manifest totals
    assert seen == [
        (
            manifest["train_examples"] + manifest["holdout_examples"],
            manifest["total_tokens"],
        )
    ]
    assert manifest["train_examples"] + manifest["holdout_examples"] == 1000


@pytest.fixture()
def prepared_sft(chat_tok_dir, tmp_path) -> str:
    src = write_conv_jsonl(tmp_path / "chat.jsonl", 8)
    out = str(tmp_path / "sft")
    prepare_sft(src, out, chat_tok_dir, "gemma4", holdout=0.25, seed=7)
    return out


def test_load_sft_dataset_requires_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="No manifest.json"):
        load_sft_dataset(str(tmp_path))


def test_load_sft_dataset_refuses_missing_split_file(prepared_sft):
    os.remove(os.path.join(prepared_sft, sft_mod.HOLDOUT_FILE))
    with pytest.raises(sft_mod.SftIntegrityError, match="Missing file"):
        load_sft_dataset(prepared_sft)


def test_load_sft_dataset_refuses_example_count_mismatch(prepared_sft):
    manifest_path = os.path.join(prepared_sft, sft_mod.MANIFEST_NAME)
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["files"][0]["examples"] += 1  # hashes untouched: only the count lies
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    with pytest.raises(sft_mod.SftIntegrityError, match="manifest expects"):
        load_sft_dataset(prepared_sft)
