"""Benchmark decontamination (P2-D1).

Builds an n-gram index from the P2-D6 benchmark list
(training/continual/benchmarks.yaml) and filters documents whose text overlaps
it. Overlap is 13-gram hash membership (n configurable in the YAML), following
common decontamination practice: any hit drops the whole document.

The index directory is the contract consumed by the pipeline:

  <index_dir>/ngrams.bin          sorted uint64 xxhash64 n-gram hashes
  <index_dir>/index_manifest.json benchmarks, resolved revisions, counts,
                                  ngrams.bin hash

Shard manifests reference the index via decontaminated_against (the xxhash64
of index_manifest.json as written), so a benchmark-list change is detectable
downstream.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
from collections.abc import Iterable, Iterator

import numpy as np
import xxhash

INDEX_MANIFEST_NAME = "index_manifest.json"
NGRAMS_NAME = "ngrams.bin"

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_words(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric runs (GPT-3-style normalization)."""
    return _WORD_RE.findall(text.lower())


def iter_ngram_hashes(words: list[str], n: int) -> Iterator[int]:
    """xxhash64 of each space-joined word n-gram; nothing for texts under n words."""
    for start in range(len(words) - n + 1):
        yield xxhash.xxh64(
            " ".join(words[start : start + n]).encode("utf-8")
        ).intdigest()


def _field_values(record: dict, path: str) -> Iterator[str]:
    """Resolve a dotted field path to its string values (strings or lists of strings)."""
    value = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return
        value = value[key]
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        yield from (v for v in value if isinstance(v, str))


def load_benchmarks_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        config = yaml.safe_load(f)
    for key in ("version", "ngram_size", "benchmarks"):
        if key not in config:
            raise ValueError(f"{path}: missing required key {key!r}")
    return config


def _iter_benchmark_records(bench: dict) -> Iterator[dict]:
    """Yield raw records for one benchmark entry.

    A dataset value that exists on disk is read as JSONL (test fixtures);
    anything else loads from the Hugging Face hub at the pinned revision.
    """
    source = bench["dataset"]
    if os.path.exists(source):
        with open(source, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    import datasets

    for split in bench["splits"]:
        ds = datasets.load_dataset(
            source, bench.get("config"), split=split, revision=bench.get("revision")
        )
        yield from ds


def _resolved_revision(bench: dict) -> str | None:
    """Concrete commit SHA for a hub dataset, so 'main' pins to what was used."""
    if os.path.exists(bench["dataset"]):
        return None
    try:
        from huggingface_hub import dataset_info

        return dataset_info(bench["dataset"], revision=bench.get("revision")).sha
    except Exception:
        return "unresolved"


def build_index(benchmarks_yaml: str, out_dir: str, command: str | None = None) -> dict:
    """Build the contamination index directory and return its manifest."""
    config = load_benchmarks_config(benchmarks_yaml)
    n = config["ngram_size"]
    os.makedirs(out_dir, exist_ok=True)

    hashes: set[int] = set()
    bench_records = []
    for bench in config["benchmarks"]:
        before = len(hashes)
        documents = 0
        for record in _iter_benchmark_records(bench):
            documents += 1
            for path in bench["text_fields"]:
                for text in _field_values(record, path):
                    hashes.update(iter_ngram_hashes(normalize_words(text), n))
        bench_records.append(
            {
                "name": bench["name"],
                "dataset": bench["dataset"],
                "config": bench.get("config"),
                "revision": bench.get("revision"),
                "resolved_revision": _resolved_revision(bench),
                "splits": bench["splits"],
                "text_fields": bench["text_fields"],
                "documents": documents,
                "new_ngrams": len(hashes) - before,
            }
        )

    array = np.sort(np.fromiter(hashes, dtype=np.uint64, count=len(hashes)))
    ngrams_path = os.path.join(out_dir, NGRAMS_NAME)
    array.tofile(ngrams_path)

    from training.data.shards import hash_file

    manifest = {
        "benchmarks_yaml": benchmarks_yaml,
        "benchmarks_version": config["version"],
        "ngram_size": n,
        "benchmarks": bench_records,
        "total_ngrams": len(hashes),
        "ngrams_xxhash64": hash_file(ngrams_path),
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": command if command is not None else " ".join(sys.argv),
    }
    path = os.path.join(out_dir, INDEX_MANIFEST_NAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def index_hash(index_dir: str) -> str:
    """The decontaminated_against value: xxhash64 of the index manifest file."""
    from training.data.shards import hash_file

    return hash_file(os.path.join(index_dir, INDEX_MANIFEST_NAME))


class ContaminationIndex:
    """Read side: sorted-array membership over the built n-gram hashes."""

    def __init__(self, index_dir: str) -> None:
        manifest_path = os.path.join(index_dir, INDEX_MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"No {INDEX_MANIFEST_NAME} in {index_dir}")
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.ngram_size = self.manifest["ngram_size"]
        self.hash = index_hash(index_dir)
        self._hashes = np.fromfile(
            os.path.join(index_dir, NGRAMS_NAME), dtype=np.uint64
        )

    def contains_any(self, hashes: Iterable[int]) -> bool:
        # One vectorized searchsorted over the document's n-grams: the
        # clean-document case scans every n-gram regardless, and a Python
        # loop pays numpy dispatch per n-gram on that hot path.
        batch = np.fromiter(hashes, dtype=np.uint64)
        if batch.size == 0:
            return False
        pos = np.searchsorted(self._hashes, batch)
        inside = pos < self._hashes.size
        return bool(np.any(self._hashes[pos[inside]] == batch[inside]))

    def is_contaminated(self, text: str) -> bool:
        return self.contains_any(
            iter_ngram_hashes(normalize_words(text), self.ngram_size)
        )


def make_filter(index_dir: str, exclusion_writer=None):
    """datatrove filter block dropping documents that hit the index.

    Defined behind a factory so this module imports without datatrove (the
    index build and membership check above are dependency-light).
    """
    from datatrove.pipeline.filters.base_filter import BaseFilter

    class ContaminationFilter(BaseFilter):
        name = "benchmark decontamination"

        def __init__(self) -> None:
            super().__init__(exclusion_writer)
            self._index_dir = index_dir
            self._index: ContaminationIndex | None = None

        def filter(self, doc) -> bool | tuple[bool, str]:
            if (
                self._index is None
            ):  # lazy per-worker load; the mmap-sized array forks poorly
                self._index = ContaminationIndex(self._index_dir)
            if self._index.is_contaminated(doc.text):
                return False, "benchmark_overlap"
            return True

    return ContaminationFilter()
