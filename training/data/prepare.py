"""Tokenize-and-pack pipeline (Phase 0 D2).

Turns a Hugging Face dataset or local JSONL into token shards via
training/data/shards.ShardWriter. Lite pipeline: no dedup, filtering, or
decontamination (Phase 2 adds those in front of the same writer). Nothing here
requires the corpus to fit in memory: HF sources can stream and JSONL is read
line by line.

Parallelism: non-streaming HF sources tokenize through datasets.map with
num_proc worker processes (tokenization is GIL-bound at the Python layer).
Streaming and JSONL sources tokenize through the fast tokenizer's encode_batch,
which parallelizes internally in Rust across all cores.
"""

from __future__ import annotations

import datetime
import functools
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import xxhash

from training.data import shards as shards_mod
from training.data.shards import (
    ShardWriter,
    dtype_for_vocab,
    length_bucket,
    write_manifest,
)

DEFAULT_NUM_PROC = 30       # 32 cores minus two for the writer and OS (inference-down policy)
DEFAULT_BATCH_DOCS = 1_000  # documents per encode_batch call on the streaming path
FALLBACK_EOS_TOKENS = ("<|eos|>", "<|endoftext|>", "</s>")


@dataclass
class LoadedTokenizer:
    """A tokenizer reduced to what the pipeline needs, backend-agnostic."""

    encode_batch: Callable[[list[str]], list[list[int]]]
    eos_id: int
    vocab_size: int
    content_hash: str    # xxhash64 of the serialized tokenizer (the hash D2 manifests record)
    source: str          # path or hub id, for the manifest
    spec: tuple = field(default=(), repr=False)   # (source, eos_token) for worker re-load


def _hash_tokenizer_str(serialized: str) -> str:
    return xxhash.xxh64(serialized.encode("utf-8")).hexdigest()


def load_tokenizer(source: str, eos_token: str | None = None) -> LoadedTokenizer:
    """Load a tokenizer from a local tokenizer.json, a local HF directory, or a hub id.

    A bare tokenizer.json loads through the tokenizers library directly (no
    transformers needed); anything else goes through AutoTokenizer. Fails loudly
    when no EOS id can be determined.
    """
    json_path = None
    if os.path.isfile(source) and source.endswith(".json"):
        json_path = source
    elif os.path.isdir(source) and os.path.isfile(os.path.join(source, "tokenizer.json")):
        json_path = os.path.join(source, "tokenizer.json")

    if json_path is not None:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(json_path)
        eos_id = None
        for candidate in ([eos_token] if eos_token else FALLBACK_EOS_TOKENS):
            eos_id = tok.token_to_id(candidate)
            if eos_id is not None:
                break
        if eos_id is None:
            raise ValueError(
                f"Cannot determine EOS id for {source}; pass --eos-token with the exact "
                f"EOS token string (tried: {eos_token or ', '.join(FALLBACK_EOS_TOKENS)})")
        return LoadedTokenizer(
            encode_batch=lambda texts: [e.ids for e in tok.encode_batch(texts)],
            eos_id=eos_id,
            vocab_size=tok.get_vocab_size(),
            content_hash=_hash_tokenizer_str(tok.to_str()),
            source=source,
            spec=(source, eos_token),
        )

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(source)
    if eos_token is not None:
        eos_id = tok.convert_tokens_to_ids(eos_token)
    else:
        eos_id = tok.eos_token_id
    if eos_id is None:
        raise ValueError(f"Tokenizer {source} has no EOS token; pass --eos-token")
    return LoadedTokenizer(
        encode_batch=lambda texts: tok(texts, add_special_tokens=False)["input_ids"],
        eos_id=eos_id,
        vocab_size=len(tok),
        content_hash=_hash_tokenizer_str(tok.backend_tokenizer.to_str()),
        source=source,
        spec=(source, eos_token),
    )


def iter_jsonl(path: str, text_field: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if text_field not in record:
                raise KeyError(f"{path}:{lineno}: no field {text_field!r}")
            yield record[text_field]


def _batched(texts: Iterable[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


# Per-process tokenizer for datasets.map workers; loaded once per process, not per batch
_WORKER_TOKENIZER: LoadedTokenizer | None = None


def _map_encode(batch: dict, spec: tuple, text_field: str) -> dict:
    global _WORKER_TOKENIZER
    if _WORKER_TOKENIZER is None:
        _WORKER_TOKENIZER = load_tokenizer(*spec)
    return {"input_ids": _WORKER_TOKENIZER.encode_batch(batch[text_field])}


def _git_sha() -> str:
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def iter_token_batches(source: str, tokenizer: LoadedTokenizer, *,
                       text_field: str = "text", split: str = "train",
                       revision: str | None = None, streaming: bool = False,
                       num_proc: int = DEFAULT_NUM_PROC,
                       batch_docs: int = DEFAULT_BATCH_DOCS) -> Iterator[list[list[int]]]:
    """Yield batches of per-document token id lists, in deterministic source order."""
    if os.path.exists(source):
        for batch in _batched(iter_jsonl(source, text_field), batch_docs):
            yield tokenizer.encode_batch(batch)
        return

    import datasets
    ds = datasets.load_dataset(source, split=split, revision=revision, streaming=streaming)
    if streaming:
        for batch in _batched((rec[text_field] for rec in ds), batch_docs):
            yield tokenizer.encode_batch(batch)
        return

    encoded = ds.map(
        functools.partial(_map_encode, spec=tokenizer.spec, text_field=text_field),
        batched=True, batch_size=batch_docs,
        num_proc=min(num_proc, len(ds)) or 1,
        remove_columns=ds.column_names,
    )
    for start in range(0, len(encoded), batch_docs):
        yield encoded[start:start + batch_docs]["input_ids"]


def prepare(source: str, out_dir: str, tokenizer_source: str, *,
            text_field: str = "text", split: str = "train",
            revision: str | None = None, streaming: bool = False,
            max_tokens: int | None = None,
            num_proc: int = DEFAULT_NUM_PROC,
            shard_tokens: int = shards_mod.DEFAULT_SHARD_TOKENS,
            eos_token: str | None = None,
            license_str: str = "unknown",
            command: str | None = None,
            progress: Callable[[int, int], None] | None = None) -> dict:
    """Run the pipeline and return the written manifest.

    progress, when given, is called as progress(total_tokens, total_documents)
    after each closed batch.
    """
    manifest_path = os.path.join(out_dir, shards_mod.MANIFEST_NAME)
    if os.path.exists(manifest_path):
        raise FileExistsError(
            f"{out_dir} already contains {shards_mod.MANIFEST_NAME}; refusing to overwrite")

    tokenizer = load_tokenizer(tokenizer_source, eos_token)
    dtype = dtype_for_vocab(tokenizer.vocab_size)
    writer = ShardWriter(out_dir, dtype, shard_tokens)

    total_tokens = 0
    total_documents = 0
    histogram: dict[str, int] = {}
    done = False
    for batch in iter_token_batches(source, tokenizer, text_field=text_field, split=split,
                                    revision=revision, streaming=streaming,
                                    num_proc=num_proc, batch_docs=DEFAULT_BATCH_DOCS):
        for ids in batch:
            writer.write_tokens(list(ids) + [tokenizer.eos_id])
            total_tokens += len(ids) + 1
            total_documents += 1
            bucket = length_bucket(len(ids))
            histogram[bucket] = histogram.get(bucket, 0) + 1
            if max_tokens is not None and total_tokens >= max_tokens:
                done = True
                break
        if progress is not None:
            progress(total_tokens, total_documents)
        if done:
            break
    shard_records = writer.close()

    manifest = {
        "dataset": source,
        "split": None if os.path.exists(source) else split,
        "revision": revision,
        "license": license_str,
        "tokenizer": tokenizer.source,
        "tokenizer_hash": tokenizer.content_hash,
        "dtype": str(dtype),
        "eos_id": tokenizer.eos_id,
        "shard_tokens": shard_tokens,
        "shards": shard_records,
        "total_tokens": total_tokens,
        "total_documents": total_documents,
        "doc_length_histogram": histogram,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_sha": _git_sha(),
        "command": command if command is not None else " ".join(sys.argv),
    }
    assert total_tokens == sum(rec["tokens"] for rec in shard_records), \
        "writer token accounting does not match document accounting"
    write_manifest(out_dir, manifest)
    return manifest
