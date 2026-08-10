"""Independent GGUF reader, for cross-checking scripts/models/gguf_header_transcript.sh.

The reader in that script is hand-rolled bash. A round-trip against a fixture
this suite also wrote would only prove the writer and reader share a
misunderstanding, so the load-bearing check parses a GGUF written by a third
party -- the upstream `gguf` package, via the interp exporters -- with this
reader, and compares the two. Three separate implementations, none of which
shares an author with another.

This lived in rust-harness/fixtures/reference_logits.py while the Rust runtime
loaded models from GGUF. It no longer does: the runtime reads config.json and
model.safetensors, and its numpy reference reads the same. The reader is kept
here because the bash script still serves llama.cpp GGUFs, which stay.
"""

import math
import struct

import numpy as np


def parse_gguf(data):
    pos = [0]

    def take(n):
        chunk = data[pos[0] : pos[0] + n]
        assert len(chunk) == n, "truncated GGUF"
        pos[0] += n
        return chunk

    def u32():
        return struct.unpack("<I", take(4))[0]

    def u64():
        return struct.unpack("<Q", take(8))[0]

    def string():
        return take(u64()).decode()

    assert take(4) == b"GGUF" and u32() == 3
    tensor_count, kv_count = u64(), u64()
    metadata = {}
    for _ in range(kv_count):
        key = string()
        value_type = u32()
        if value_type == 4:
            metadata[key] = u32()
        elif value_type == 6:
            metadata[key] = struct.unpack("<f", take(4))[0]
        elif value_type == 8:
            metadata[key] = string()
        elif value_type == 9:
            elem, count = u32(), u64()
            if elem == 5:
                metadata[key] = list(struct.unpack(f"<{count}i", take(4 * count)))
            elif elem == 6:
                metadata[key] = list(struct.unpack(f"<{count}f", take(4 * count)))
            elif elem == 8:
                metadata[key] = [string() for _ in range(count)]
            else:
                raise ValueError(f"array element type {elem}")
        else:
            raise ValueError(f"value type {value_type}")
    directory = []
    for _ in range(tensor_count):
        name = string()
        dims = [u64() for _ in range(u32())]
        ggml_type, offset = u32(), u64()
        assert ggml_type == 0, "reference reader handles F32 only"
        directory.append((name, dims, offset))
    alignment = metadata.get("general.alignment", 32)
    data_start = (pos[0] + alignment - 1) // alignment * alignment
    tensors = {}
    for name, dims, offset in directory:
        count = math.prod(dims)
        start = data_start + offset
        flat = np.frombuffer(data, dtype="<f4", count=count, offset=start)
        # GGUF layout [ne0=in, ne1=out]: row-major with ne0 contiguous.
        tensors[name] = flat.reshape(list(reversed(dims))).astype(np.float32)
    return metadata, tensors
