"""Contamination index build and membership checks (P2-D1)."""

from __future__ import annotations

import json
import os

import pytest
import yaml

from training.data import contamination


BENCH_QUESTION = ("Which layer of the atmosphere contains the ozone that absorbs "
                  "most of the ultraviolet radiation arriving from the sun every day")
BENCH_CHOICES = ["the stratosphere holds it", "the troposphere holds it"]


@pytest.fixture()
def benchmarks_yaml(tmp_path) -> str:
    bench_jsonl = tmp_path / "bench.jsonl"
    with open(bench_jsonl, "w") as f:
        f.write(json.dumps({"question": BENCH_QUESTION,
                            "choices": {"text": BENCH_CHOICES}}) + "\n")
    config = {
        "version": 1,
        "ngram_size": 13,
        "benchmarks": [{
            "name": "toy_arc",
            "dataset": str(bench_jsonl),
            "config": None,
            "revision": None,
            "splits": ["test"],
            "text_fields": ["question", "choices.text"],
        }],
    }
    path = tmp_path / "benchmarks.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return str(path)


def test_normalize_words_lowercases_and_splits_on_non_alphanumeric():
    assert contamination.normalize_words("The QUICK, brown-fox's 42!") == \
        ["the", "quick", "brown", "fox", "s", "42"]


def test_ngram_hashes_count_and_determinism():
    words = [f"w{i}" for i in range(20)]
    first = list(contamination.iter_ngram_hashes(words, 13))
    assert len(first) == 8
    assert first == list(contamination.iter_ngram_hashes(words, 13))
    assert list(contamination.iter_ngram_hashes(words[:12], 13)) == []


def test_field_values_resolves_dotted_paths_and_lists():
    record = {"question": "q", "choices": {"text": ["a", "b"], "label": [1, 2]}}
    assert list(contamination._field_values(record, "question")) == ["q"]
    assert list(contamination._field_values(record, "choices.text")) == ["a", "b"]
    assert list(contamination._field_values(record, "choices.label")) == []
    assert list(contamination._field_values(record, "missing.path")) == []


def test_build_index_detects_planted_passage(benchmarks_yaml, tmp_path):
    index_dir = str(tmp_path / "index")
    manifest = contamination.build_index(benchmarks_yaml, index_dir)
    assert manifest["total_ngrams"] > 0
    assert manifest["benchmarks"][0]["documents"] == 1

    index = contamination.ContaminationIndex(index_dir)
    contaminated = ("Students reviewed the study guide before the exam. "
                    + BENCH_QUESTION + ". They then compared their answers afterwards.")
    clean = ("Farmers in the valley planted drought resistant wheat after two dry "
             "seasons reduced their annual harvest and the cooperative expanded storage.")
    assert index.is_contaminated(contaminated)
    assert not index.is_contaminated(clean)


def test_build_index_is_deterministic(benchmarks_yaml, tmp_path):
    first = contamination.build_index(benchmarks_yaml, str(tmp_path / "a"))
    second = contamination.build_index(benchmarks_yaml, str(tmp_path / "b"))
    assert first["ngrams_xxhash64"] == second["ngrams_xxhash64"]
    assert first["total_ngrams"] == second["total_ngrams"]


def test_ngram_content_hash_reflects_benchmark_list_changes(
        benchmarks_yaml, tmp_path):
    # index_hash covers the whole manifest, which embeds a timestamp and so
    # always differs between builds; the content signal is ngrams_xxhash64.
    a = contamination.build_index(benchmarks_yaml, str(tmp_path / "a"))
    with open(benchmarks_yaml) as f:
        config = yaml.safe_load(f)
    # Dropping the question field removes its 13-grams from the index (the
    # short choices strings alone yield none).
    config["benchmarks"][0]["text_fields"] = ["choices.text"]
    changed_yaml = str(tmp_path / "changed.yaml")
    with open(changed_yaml, "w") as f:
        yaml.safe_dump(config, f)
    b = contamination.build_index(changed_yaml, str(tmp_path / "b"))
    assert a["ngrams_xxhash64"] != b["ngrams_xxhash64"]

    # Byte-identical inputs reproduce the content hash exactly.
    c = contamination.build_index(benchmarks_yaml, str(tmp_path / "c"))
    assert c["ngrams_xxhash64"] == a["ngrams_xxhash64"]


def test_missing_index_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        contamination.ContaminationIndex(str(tmp_path / "nope"))


def test_repo_benchmarks_yaml_is_valid():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = contamination.load_benchmarks_config(
        os.path.join(repo_root, "training", "continual", "benchmarks.yaml"))
    names = [b["name"] for b in config["benchmarks"]]
    assert names == ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "mmlu"]
    assert config["ngram_size"] == 13
    for bench in config["benchmarks"]:
        assert bench["splits"] and bench["text_fields"]
