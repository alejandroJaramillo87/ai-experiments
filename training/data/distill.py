"""Top-K teacher logprob cache for logit distillation (Phase 1 F5).

The cache pre-pass (training/tune/distill.py build_logit_cache) runs the teacher
alone over a prepared F1 dataset and materializes, for every token position of
every train example, the indices and logprobs of the teacher's top-K next-token
distribution. Training then runs student-only: the reader memory-maps the
shards and the collator attaches per-example slices to each batch.

Layout under the cache directory:

  indices.bin    int32   [total_tokens, top_k]   token ids of the top-K support
  logprobs.bin   float16 [total_tokens, top_k]   log p_teacher at those ids
  offsets.bin    int64   [examples + 1]          example i spans offsets[i:i+2]
  manifest.json  provenance and integrity record (D2/F1 manifest discipline)

The manifest binds the cache to its inputs: the teacher identity and revision,
the tokenizer hash, and the xxhash of the source dataset's train file. The
reader refuses a cache whose files or bindings do not match, because a
misaligned cache trains a plausible-looking but wrong KL (the failure mode the
resolved F5 decision 5 exists to prevent).

Alignment contract: the cache is written over the dataset's train split, in
file order, unpacked. Position t of example i holds the teacher distribution
computed at input position t (predicting token t+1 after the causal shift);
the trainer applies the identical shift to both sides.
"""

from __future__ import annotations

import datetime
import json
import os

import numpy as np

from training.data.prepare import _git_sha
from training.data.shards import hash_file

MANIFEST_NAME = "manifest.json"
INDICES_FILE = "indices.bin"
LOGPROBS_FILE = "logprobs.bin"
OFFSETS_FILE = "offsets.bin"
CACHE_KIND = "distill-logit-cache"
INDEX_DTYPE = np.int32
LOGPROB_DTYPE = np.float16
OFFSET_DTYPE = np.int64


class LogitCacheError(ValueError):
    """The cache is missing, corrupt, or bound to different inputs."""


class LogitCacheWriter:
    """Streams per-example top-K arrays to disk; finalize() writes the manifest.

    add_example expects arrays shaped [tokens, top_k]; examples arrive in the
    dataset's train-file order, which is the alignment the reader assumes.
    Refuses to overwrite an existing cache (a stale cache silently poisoning
    a run is worse than an error).
    """

    def __init__(self, out_dir: str, top_k: int) -> None:
        if os.path.exists(os.path.join(out_dir, MANIFEST_NAME)):
            raise FileExistsError(
                f"{out_dir} already contains {MANIFEST_NAME}; refusing to "
                "overwrite (delete the directory to rebuild)"
            )
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.top_k = top_k
        self.offsets = [0]
        self._indices_f = open(os.path.join(out_dir, INDICES_FILE), "wb")
        self._logprobs_f = open(os.path.join(out_dir, LOGPROBS_FILE), "wb")

    def add_example(self, indices: np.ndarray, logprobs: np.ndarray) -> None:
        if (
            indices.shape != logprobs.shape
            or indices.ndim != 2
            or indices.shape[1] != self.top_k
        ):
            raise LogitCacheError(
                f"expected matching [tokens, {self.top_k}] arrays, got "
                f"indices {indices.shape} and logprobs {logprobs.shape}"
            )
        self._indices_f.write(np.ascontiguousarray(indices, INDEX_DTYPE).tobytes())
        self._logprobs_f.write(np.ascontiguousarray(logprobs, LOGPROB_DTYPE).tobytes())
        self.offsets.append(self.offsets[-1] + indices.shape[0])

    def finalize(
        self,
        *,
        teacher: str,
        teacher_revision: str,
        tokenizer_hash: str,
        dataset_dir: str,
        dataset_train_hash: str,
        command: str = "",
    ) -> dict:
        self._indices_f.close()
        self._logprobs_f.close()
        offsets_path = os.path.join(self.out_dir, OFFSETS_FILE)
        np.asarray(self.offsets, OFFSET_DTYPE).tofile(offsets_path)
        manifest = {
            "kind": CACHE_KIND,
            "teacher": teacher,
            "teacher_revision": teacher_revision,
            "tokenizer_hash": tokenizer_hash,
            "dataset_dir": dataset_dir,
            "dataset_train_hash": dataset_train_hash,
            "top_k": self.top_k,
            "examples": len(self.offsets) - 1,
            "total_tokens": self.offsets[-1],
            "index_dtype": np.dtype(INDEX_DTYPE).name,
            "logprob_dtype": np.dtype(LOGPROB_DTYPE).name,
            "files": [
                {"file": name, "xxhash64": hash_file(os.path.join(self.out_dir, name))}
                for name in (INDICES_FILE, LOGPROBS_FILE, OFFSETS_FILE)
            ],
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "git_sha": _git_sha(),
            "command": command,
        }
        path = os.path.join(self.out_dir, MANIFEST_NAME)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.rename(tmp, path)
        return manifest


class LogitCache:
    """Memory-mapped reader over a finalized cache.

    Verifies file hashes against the manifest at open (xxhash streams at
    NVMe speed; a corrupt cache must never train). Binding checks against a
    specific dataset/tokenizer are verify_binding(), called by the trainer
    preflight with the run's actual inputs.
    """

    def __init__(self, cache_dir: str) -> None:
        manifest_path = os.path.join(cache_dir, MANIFEST_NAME)
        if not os.path.isfile(manifest_path):
            raise LogitCacheError(
                f"No {MANIFEST_NAME} in {cache_dir}; build the "
                "cache with 'python -m training distill-cache'"
            )
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        if self.manifest.get("kind") != CACHE_KIND:
            raise LogitCacheError(
                f"{cache_dir} is not a logit cache (manifest kind: "
                f"{self.manifest.get('kind')!r})"
            )
        for rec in self.manifest["files"]:
            path = os.path.join(cache_dir, rec["file"])
            if not os.path.isfile(path):
                raise LogitCacheError(f"Missing cache file: {path}")
            digest = hash_file(path)
            if digest != rec["xxhash64"]:
                raise LogitCacheError(
                    f"{rec['file']}: content hash {digest} does not match "
                    f"manifest {rec['xxhash64']} (rebuild the cache)"
                )

        self.cache_dir = cache_dir
        self.top_k = self.manifest["top_k"]
        total = self.manifest["total_tokens"]
        examples = self.manifest["examples"]
        self.offsets = np.fromfile(os.path.join(cache_dir, OFFSETS_FILE), OFFSET_DTYPE)
        if len(self.offsets) != examples + 1 or self.offsets[-1] != total:
            raise LogitCacheError(
                f"offsets ({len(self.offsets)} entries, last "
                f"{self.offsets[-1] if len(self.offsets) else 'none'}) do not "
                f"match manifest ({examples} examples, {total} tokens)"
            )
        shape = (total, self.top_k)
        self.indices = np.memmap(
            os.path.join(cache_dir, INDICES_FILE), INDEX_DTYPE, "r", shape=shape
        )
        self.logprobs = np.memmap(
            os.path.join(cache_dir, LOGPROBS_FILE), LOGPROB_DTYPE, "r", shape=shape
        )

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def example(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(indices, logprobs) arrays for train example i, shape [tokens, K]."""
        start, end = self.offsets[i], self.offsets[i + 1]
        return self.indices[start:end], self.logprobs[start:end]

    def verify_binding(
        self, *, dataset_train_hash: str, tokenizer_hash: str, examples: int
    ) -> None:
        """Refuse a cache built over different inputs than this run's."""
        problems = []
        if self.manifest["dataset_train_hash"] != dataset_train_hash:
            problems.append(
                f"dataset train hash {dataset_train_hash} does not match the "
                f"cache's {self.manifest['dataset_train_hash']}"
            )
        if self.manifest["tokenizer_hash"] != tokenizer_hash:
            problems.append(
                f"tokenizer hash {tokenizer_hash} does not match the cache's "
                f"{self.manifest['tokenizer_hash']}"
            )
        if len(self) != examples:
            problems.append(
                f"dataset has {examples} train examples, cache has {len(self)}"
            )
        if problems:
            raise LogitCacheError(
                "Cache/run binding mismatch (rebuild the cache for this "
                "dataset):\n" + "\n".join(f"  - {p}" for p in problems)
            )
