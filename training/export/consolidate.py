"""Consolidate a sharded safetensors checkpoint into one model.safetensors.

The rust harness reads a checkpoint directory holding config.json and a single
model.safetensors (rust-harness/runtime/src/checkpoint.rs, WEIGHTS_FILE); it
has no index support, so a save_pretrained() output of model-0000N-of-0000M
shards plus model.safetensors.index.json is not loadable there. This module
joins the shards into the one file the harness expects, and asserts that an
export directory holds exactly that.

It works at the container level, not the tensor level: a safetensors file is
an 8-byte little-endian header length, that many bytes of JSON naming each
tensor's dtype, shape and byte range, then the raw tensor bytes. Joining
shards is therefore a matter of rewriting the byte ranges and streaming the
data through, which needs neither torch nor numpy and is agnostic to dtype,
so a bf16 checkpoint consolidates in the CPU sandbox as well as on the
workstation. Tensors are written in sorted-name order, so the same weights
consolidate to the same bytes however they were sharded, and the sha256 an
export manifest records as model_hash does not depend on the shard split.

This is the piece every training road shares: adapter merges (which
save_pretrained can already write unsharded), full fine-tunes, and the D6
scratch checkpoints under step_N/model/, which save_pretrained shards by
default. The from-scratch exporter that will read those is not built here.
"""

from __future__ import annotations

import json
import os
import struct

from training.export.errors import ExportError

WEIGHTS_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"
SAFETENSORS_SUFFIX = ".safetensors"
METADATA_KEY = "__metadata__"

HEADER_LEN_BYTES = 8  # little-endian u64 header length that opens every file
HEADER_ALIGN = 8  # the reference writer pads the JSON to a multiple of 8
DEFAULT_CHUNK_BYTES = 1 << 24  # 16 MiB copies: few syscalls, bounded memory


def read_header(path: str) -> tuple[dict, int]:
    """(parsed header, absolute offset where the tensor data begins)."""
    with open(path, "rb") as f:
        prefix = f.read(HEADER_LEN_BYTES)
        if len(prefix) < HEADER_LEN_BYTES:
            raise ExportError(f"Not a safetensors file (truncated header): {path}")
        (header_len,) = struct.unpack("<Q", prefix)
        raw = f.read(header_len)
    if len(raw) < header_len:
        raise ExportError(f"Not a safetensors file (header runs past EOF): {path}")
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExportError(
            f"Not a safetensors file (header is not JSON): {path}"
        ) from exc
    if not isinstance(header, dict):
        raise ExportError(f"Not a safetensors file (header is not an object): {path}")
    return header, HEADER_LEN_BYTES + header_len


def weights_files(model_dir: str) -> list[str]:
    """Every *.safetensors file in the directory, sorted; the index if present."""
    names = sorted(
        n
        for n in os.listdir(model_dir)
        if n.endswith(SAFETENSORS_SUFFIX) or n == INDEX_FILE
    )
    return [os.path.join(model_dir, n) for n in names]


def shard_paths(model_dir: str) -> list[str]:
    """The weights files to join: the index's weight_map targets when an
    index exists (so a stray file beside the shards is not swept in), else
    every *.safetensors file. Refuses a directory with no weights at all."""
    index_path = os.path.join(model_dir, INDEX_FILE)
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        names = sorted(set(index.get("weight_map", {}).values()))
        if not names:
            raise ExportError(f"Weights index names no shards: {index_path}")
        missing = [n for n in names if not os.path.isfile(os.path.join(model_dir, n))]
        if missing:
            raise ExportError(
                f"Weights index names shards that are absent: {', '.join(missing)}"
                f"\n  Solution: the checkpoint is incomplete; re-save or re-download it"
            )
        return [os.path.join(model_dir, n) for n in names]
    paths = [p for p in weights_files(model_dir) if p.endswith(SAFETENSORS_SUFFIX)]
    if not paths:
        raise ExportError(f"No *.safetensors weights in model directory: {model_dir}")
    return paths


def _plan(shards: list[str]) -> tuple[dict, dict, list[tuple[str, str, int, int]]]:
    """Read every shard header once and lay the tensors out end to end.

    Returns (new header, merged metadata, copy plan) where each plan entry is
    (tensor name, shard path, absolute start offset in the shard, length).
    """
    entries: dict[str, tuple[str, dict, int]] = {}
    metadata: dict = {}
    for shard in shards:
        header, data_start = read_header(shard)
        for name, spec in header.items():
            if name == METADATA_KEY:
                for key, value in (spec or {}).items():
                    if key in metadata and metadata[key] != value:
                        raise ExportError(
                            f"Shards disagree on metadata {key!r}: "
                            f"{metadata[key]!r} vs {value!r} in {shard}"
                        )
                    metadata[key] = value
                continue
            if name in entries:
                raise ExportError(
                    f"Tensor {name} appears in two shards: {entries[name][0]} "
                    f"and {shard}\n  Solution: the checkpoint is inconsistent; "
                    "re-save it or export a known-good one"
                )
            entries[name] = (shard, spec, data_start)

    header_out: dict = {}
    if metadata:
        header_out[METADATA_KEY] = metadata
    plan: list[tuple[str, str, int, int]] = []
    cursor = 0
    for name in sorted(entries):
        shard, spec, data_start = entries[name]
        start, end = spec["data_offsets"]
        length = end - start
        header_out[name] = {
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "data_offsets": [cursor, cursor + length],
        }
        plan.append((name, shard, data_start + start, length))
        cursor += length
    return header_out, metadata, plan


def _encode_header(header: dict) -> bytes:
    """The header JSON, space-padded to the alignment the reference writer
    uses, with the length prefix in front."""
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    pad = (-len(raw)) % HEADER_ALIGN
    raw += b" " * pad
    return struct.pack("<Q", len(raw)) + raw


def consolidate_shards(
    model_dir: str, out_path: str, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> dict:
    """Join the directory's shards into one file at out_path.

    Writes through a .tmp sibling and renames on completion, so a crash never
    leaves a truncated model.safetensors that a later hash would bless.
    Returns {"tensors", "shards", "size_bytes"} for the export manifest.
    """
    shards = shard_paths(model_dir)
    header, _metadata, plan = _plan(shards)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as out:
        out.write(_encode_header(header))
        for _name, shard, offset, length in plan:
            with open(shard, "rb") as src:
                src.seek(offset)
                remaining = length
                while remaining:
                    chunk = src.read(min(chunk_bytes, remaining))
                    if not chunk:
                        raise ExportError(
                            f"Shard ended before its declared tensor bytes: {shard}"
                        )
                    out.write(chunk)
                    remaining -= len(chunk)
    os.replace(tmp, out_path)
    return {
        "tensors": len(plan),
        "shards": len(shards),
        "size_bytes": os.path.getsize(out_path),
    }


def check_single_weights_file(hf_dir: str) -> str:
    """The one weights file an export directory must hold, or ExportError
    naming what was found. The harness refuses a shard index, so this is the
    assertion every export path ends with, whichever wrote the weights."""
    found = [os.path.basename(p) for p in weights_files(hf_dir)]
    if found == [WEIGHTS_FILE]:
        return os.path.join(hf_dir, WEIGHTS_FILE)
    raise ExportError(
        f"Export must hold exactly one {WEIGHTS_FILE} and no index; found "
        f"{', '.join(found) or 'no weights'} in {hf_dir}"
    )
