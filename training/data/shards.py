"""Shard format read/write (Phase 0 D2).

Flat binary token shards plus a JSON manifest (the nanoGPT/nanotron family):
one .bin per shard containing contiguous token ids, documents separated by the
tokenizer's EOS id, dtype uint16 when vocab < 65,536 else uint32. Training reads
are sequential mmap slices with zero deserialization.

The manifest is the contract: downstream loaders refuse a shard directory whose
content hashes do not match (integrity) or whose tokenizer hash does not match
the model's tokenizer (correctness).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from typing import BinaryIO

import numpy as np
import xxhash

from common.hashing import hash_file_with, write_json_atomic

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
DEFAULT_SHARD_TOKENS = 512 * 1024 * 1024  # ~1GB per shard at uint16
UINT16_VOCAB_LIMIT = 65_536


class ShardIntegrityError(ValueError):
    """A shard file's content hash or size does not match the manifest."""


class TokenizerMismatchError(ValueError):
    """The caller's tokenizer hash does not match the manifest's tokenizer hash."""


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    return (
        np.dtype(np.uint16) if vocab_size < UINT16_VOCAB_LIMIT else np.dtype(np.uint32)
    )


def hash_file(path: str) -> str:
    """xxhash64 hex digest of a file, streamed in 8MB blocks (recorded
    algorithm; do not change without a manifest version bump)."""
    return hash_file_with(xxhash.xxh64(), path, 8 * 1024 * 1024)


def length_bucket(n: int) -> str:
    """Log-scale histogram bucket label for a document length in tokens."""
    if n <= 0:
        return "0"
    lo = 1 << int(math.log2(n))
    return f"{lo}-{2 * lo - 1}"


class ShardWriter:
    """Sequential writer producing fixed-size, append-only shard files.

    Shards are hashed incrementally as bytes are written, so closing a shard
    never re-reads it. Documents may span a shard boundary; the token stream is
    contiguous across shards and only the reader's fixed windows respect
    boundaries.
    """

    def __init__(
        self, out_dir: str, dtype: np.dtype, shard_tokens: int = DEFAULT_SHARD_TOKENS
    ) -> None:
        if shard_tokens <= 0:
            raise ValueError(f"shard_tokens must be positive, got {shard_tokens}")
        self.out_dir = out_dir
        self.dtype = np.dtype(dtype)
        self.shard_tokens = shard_tokens
        self.shards: list[dict] = []  # closed-shard records: file, tokens, xxhash64
        self._file: BinaryIO | None = None
        self._hash: xxhash.xxh64 | None = None
        self._tokens_in_shard = 0
        os.makedirs(out_dir, exist_ok=True)

    def _open_shard(self) -> None:
        name = f"shard_{len(self.shards):06d}.bin"
        self._file = open(os.path.join(self.out_dir, name), "wb")
        self._hash = xxhash.xxh64()
        self._tokens_in_shard = 0

    def _close_shard(self) -> None:
        if self._file is None:
            return
        assert self._hash is not None  # set with _file in _open_shard
        name = os.path.basename(self._file.name)
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self.shards.append(
            {
                "file": name,
                "tokens": self._tokens_in_shard,
                "xxhash64": self._hash.hexdigest(),
            }
        )
        self._file = None

    def write_tokens(self, token_ids) -> None:
        """Append token ids to the stream, rolling to a new shard at the size limit."""
        # Range-check before the cast: casting an int64 ndarray to a narrower
        # dtype wraps silently, which would corrupt shards undetectably.
        ids = np.asarray(token_ids)
        if len(ids) and (
            int(ids.min()) < 0 or int(ids.max()) > np.iinfo(self.dtype).max
        ):
            raise ValueError(
                f"token ids out of range for {self.dtype}: "
                f"min {int(ids.min())}, max {int(ids.max())}"
            )
        ids = ids.astype(self.dtype, copy=False)
        pos = 0
        while pos < len(ids):
            if self._file is None:
                self._open_shard()
            assert self._file is not None and self._hash is not None
            room = self.shard_tokens - self._tokens_in_shard
            chunk = ids[pos : pos + room]
            data = chunk.tobytes()
            self._file.write(data)
            self._hash.update(data)
            self._tokens_in_shard += len(chunk)
            pos += len(chunk)
            if self._tokens_in_shard == self.shard_tokens:
                self._close_shard()

    def close(self) -> list[dict]:
        """Close the final (possibly partial) shard and return all shard records."""
        self._close_shard()
        return self.shards


def write_manifest(out_dir: str, manifest: dict) -> str:
    manifest = dict(manifest)
    manifest["manifest_version"] = MANIFEST_VERSION
    return write_json_atomic(os.path.join(out_dir, MANIFEST_NAME), manifest, fsync=True)


def load_manifest(shard_dir: str) -> dict:
    path = os.path.join(shard_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {shard_dir}")
    with open(path) as f:
        return json.load(f)


def verify_shards(shard_dir: str, manifest: dict) -> None:
    """Check every shard's size and content hash against the manifest."""
    itemsize = np.dtype(manifest["dtype"]).itemsize
    for rec in manifest["shards"]:
        path = os.path.join(shard_dir, rec["file"])
        if not os.path.exists(path):
            raise ShardIntegrityError(f"Missing shard file: {path}")
        expected_bytes = rec["tokens"] * itemsize
        actual_bytes = os.path.getsize(path)
        if actual_bytes != expected_bytes:
            raise ShardIntegrityError(
                f"{rec['file']}: size {actual_bytes} bytes, manifest expects {expected_bytes}"
            )
        digest = hash_file(path)
        if digest != rec["xxhash64"]:
            raise ShardIntegrityError(
                f"{rec['file']}: content hash {digest} does not match manifest {rec['xxhash64']}"
            )


class ShardedTokenDataset:
    """mmap-backed dataset yielding fixed-length token windows from a shard directory.

    Windows never cross shard boundaries; the last partial window per shard is
    dropped (dropped_tokens reports the total so token accounting stays honest).
    The cursor (shard index, window offset) is serializable for D6 resume.
    """

    def __init__(
        self,
        shard_dir: str,
        window: int,
        tokenizer_hash: str | None = None,
        verify: bool = True,
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self.shard_dir = shard_dir
        self.window = window
        self.manifest = load_manifest(shard_dir)
        if verify:
            verify_shards(shard_dir, self.manifest)
        if (
            tokenizer_hash is not None
            and tokenizer_hash != self.manifest["tokenizer_hash"]
        ):
            raise TokenizerMismatchError(
                f"tokenizer hash {tokenizer_hash} does not match shard manifest "
                f"{self.manifest['tokenizer_hash']} in {shard_dir}"
            )
        dtype = np.dtype(self.manifest["dtype"])
        self._mmaps = []
        self._windows_per_shard = []
        for rec in self.manifest["shards"]:
            mm = np.memmap(
                os.path.join(shard_dir, rec["file"]),
                dtype=dtype,
                mode="r",
                shape=(rec["tokens"],),
            )
            self._mmaps.append(mm)
            self._windows_per_shard.append(rec["tokens"] // window)
        self._shard = 0
        self._offset = 0  # window index within the current shard

    def __len__(self) -> int:
        return sum(self._windows_per_shard)

    @property
    def dropped_tokens(self) -> int:
        """Tokens lost to partial windows at shard tails, given this window size."""
        return sum(
            rec["tokens"] - n * self.window
            for rec, n in zip(
                self.manifest["shards"], self._windows_per_shard, strict=True
            )
        )

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        for shard, n in enumerate(self._windows_per_shard):
            if index < n:
                start = index * self.window
                return np.array(self._mmaps[shard][start : start + self.window])
            index -= n
        raise IndexError(index)  # unreachable; bounds checked above

    def cursor_index(self, shard: int, offset: int) -> int:
        """Flat window index for a (shard, offset) cursor — lets the D9
        index-based loader resume from a pre-D9 checkpoint's cursor."""
        if not 0 <= shard <= len(self._windows_per_shard):
            raise ValueError(f"cursor shard {shard} out of range")
        return sum(self._windows_per_shard[:shard]) + offset

    def next_window(self) -> np.ndarray | None:
        """Return the next window and advance the cursor; None when exhausted."""
        while self._shard < len(self._mmaps):
            if self._offset < self._windows_per_shard[self._shard]:
                start = self._offset * self.window
                out = np.array(self._mmaps[self._shard][start : start + self.window])
                self._offset += 1
                return out
            self._shard += 1
            self._offset = 0
        return None

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            out = self.next_window()
            if out is None:
                return
            yield out

    def state_dict(self) -> dict:
        return {"shard": self._shard, "offset": self._offset, "window": self.window}

    def load_state_dict(self, state: dict) -> None:
        if state["window"] != self.window:
            raise ValueError(
                f"cursor was saved with window {state['window']}, dataset uses {self.window}"
            )
        if not 0 <= state["shard"] <= len(self._mmaps):
            raise ValueError(f"cursor shard {state['shard']} out of range")
        self._shard = state["shard"]
        self._offset = state["offset"]
