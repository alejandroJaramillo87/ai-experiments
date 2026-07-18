"""Byte-level BPE tokenizer training and evaluation (Phase 0 D3).

For the Phase 3 from-scratch path ONLY. Fine-tuning and continual pre-training
always reuse the base model's tokenizer: a mismatched tokenizer silently
destroys a fine-tune, and cross-tokenizer distillation breaks logit KL.

Engine: Hugging Face tokenizers, GPT-2/Llama-3 style — byte-level
pre-tokenization (no unknown token) with digits split individually. Special
tokens <|bos|>, <|eos|>, <|pad|> plus a 16-id reserved block occupy fixed ids
0-18 ahead of the byte alphabet, so later chat/control tokens never resize
embeddings and ids are independent of corpus content.

The artifact directory (tokenizer.json + manifest.json) loads through
training/data/prepare.load_tokenizer; the manifest tokenizer_hash is xxhash64
of the reloaded Tokenizer.to_str(), the exact value D2 shard manifests record
and D5 validate() enforces. The artifact is final at training time: any
post-training mutation changes the hash and breaks that chain.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import sys
from collections.abc import Iterable, Iterator

from common.provenance import git_sha as _git_sha
from training.data.prepare import LoadedTokenizer, _hash_tokenizer_str, iter_jsonl

TOKENIZER_NAME = "tokenizer.json"
MANIFEST_NAME = "manifest.json"

BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
RESERVED_BLOCK = 16
RESERVED_TOKENS = tuple(f"<|reserved_{i}|>" for i in range(RESERVED_BLOCK))
SPECIAL_TOKENS = (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN) + RESERVED_TOKENS

DEFAULT_VOCAB_SIZE = 49_152
DEFAULT_SAMPLE_BYTES = 10_000_000_000
DEFAULT_EVAL_DOCS = 1_000

# 256-byte alphabet plus the fixed special block is the training floor
MIN_VOCAB_SIZE = 256 + len(SPECIAL_TOKENS)


def expand_corpus(patterns: Iterable[str]) -> list[str]:
    """Expand globs to a sorted, deduplicated file list (deterministic traversal)."""
    paths: set[str] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and not os.path.exists(pattern):
            raise FileNotFoundError(f"corpus pattern matches nothing: {pattern}")
        paths.update(matches if matches else [pattern])
    return sorted(paths)


class CorpusSampler:
    """Sequential document stream over sorted JSONL files with a byte budget.

    training_documents() yields until sample_bytes of UTF-8 text has been seen;
    heldout_documents() continues from where training stopped, so held-out
    evaluation docs are unseen whenever the corpus exceeds the budget. Fixed
    order (sorted files, sequential lines) keeps two runs byte-identical.
    """

    def __init__(
        self,
        paths: list[str],
        text_field: str = "text",
        sample_bytes: int | None = None,
    ) -> None:
        self.paths = paths
        self.text_field = text_field
        self.sample_bytes = sample_bytes
        self.sampled_bytes = 0
        self.sampled_documents = 0
        self._stream = self._iter_all()
        self._exhausted = False

    def _iter_all(self) -> Iterator[str]:
        for path in self.paths:
            yield from iter_jsonl(path, self.text_field)

    def training_documents(self) -> Iterator[str]:
        for text in self._stream:
            self.sampled_bytes += len(text.encode("utf-8"))
            self.sampled_documents += 1
            yield text
            if (
                self.sample_bytes is not None
                and self.sampled_bytes >= self.sample_bytes
            ):
                return
        self._exhausted = True

    def heldout_documents(self, n: int) -> list[str]:
        """Up to n documents past the training sample; empty if the corpus ran out."""
        if self._exhausted:
            return []
        docs = []
        for text in self._stream:
            docs.append(text)
            if len(docs) == n:
                break
        return docs


def train(documents: Iterable[str], vocab_size: int = DEFAULT_VOCAB_SIZE):
    """Train a byte-level BPE tokenizer; deterministic for a fixed document order."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if vocab_size < MIN_VOCAB_SIZE:
        raise ValueError(
            f"vocab_size {vocab_size} below minimum {MIN_VOCAB_SIZE} "
            f"(256-byte alphabet + {len(SPECIAL_TOKENS)} special tokens)"
        )

    tok = Tokenizer(models.BPE())
    # Digits isolate first so numbers never merge; ByteLevel then maps to bytes
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),  # fixed ids 0-18, ahead of the alphabet
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(documents, trainer)
    return tok


def _domain_stats(texts: list[str], token_counts: list[int]) -> dict:
    words = sum(len(t.split()) for t in texts)
    byte_count = sum(len(t.encode("utf-8")) for t in texts)
    tokens = sum(token_counts)
    return {
        "documents": len(texts),
        "tokens": tokens,
        "fertility": tokens / words if words else 0.0,  # tokens per whitespace word
        "compression": byte_count / tokens if tokens else 0.0,  # bytes per token
    }


def evaluate(tok, domains: dict[str, list[str]]) -> dict:
    """Fertility, compression, and round-trip losslessness per domain."""
    results: dict[str, dict] = {}
    for name, texts in domains.items():
        encodings = tok.encode_batch(texts)
        stats = _domain_stats(texts, [len(e.ids) for e in encodings])
        stats["roundtrip_failures"] = sum(
            1
            for text, enc in zip(texts, encodings, strict=True)
            if tok.decode(enc.ids, skip_special_tokens=False) != text
        )
        results[name] = stats
    return results


def evaluate_reference(
    reference: LoadedTokenizer, domains: dict[str, list[str]]
) -> dict:
    """Fertility and compression for a comparison tokenizer (no round-trip check)."""
    return {
        name: _domain_stats(texts, [len(ids) for ids in reference.encode_batch(texts)])
        for name, texts in domains.items()
    }


def write_artifact(
    tok,
    out_dir: str,
    *,
    corpus: list[str],
    text_field: str,
    sampled_bytes: int,
    sampled_documents: int,
    vocab_size: int,
    metrics: dict,
    references: dict,
    force: bool = False,
    command: str | None = None,
) -> dict:
    """Write tokenizer.json + manifest.json; refuse an existing artifact without force.

    Returns the manifest. tokenizer_hash is computed from the saved file reloaded,
    so it equals what load_tokenizer() will report for this artifact.
    """
    import tokenizers as tokenizers_lib
    from tokenizers import Tokenizer

    tokenizer_path = os.path.join(out_dir, TOKENIZER_NAME)
    if os.path.exists(tokenizer_path) and not force:
        raise FileExistsError(
            f"{out_dir} already contains {TOKENIZER_NAME}; pass --force to overwrite"
        )

    os.makedirs(out_dir, exist_ok=True)
    tok.save(tokenizer_path)
    content_hash = _hash_tokenizer_str(Tokenizer.from_file(tokenizer_path).to_str())

    manifest = {
        "corpus": corpus,
        "text_field": text_field,
        "sampled_bytes": sampled_bytes,
        "sampled_documents": sampled_documents,
        # The trained tokenizer's actual vocabulary, which can land below the
        # requested size on small corpora; both are recorded so the manifest
        # attests the artifact, not the intent
        "vocab_size": tok.get_vocab_size(),
        "requested_vocab_size": vocab_size,
        "special_tokens": {name: tok.token_to_id(name) for name in SPECIAL_TOKENS},
        "metrics": metrics,
        "references": references,
        "tokenizer_hash": content_hash,
        "tokenizers_version": tokenizers_lib.__version__,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_sha": _git_sha(),
        "command": command if command is not None else " ".join(sys.argv),
    }
    with open(os.path.join(out_dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest
