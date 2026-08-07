"""Fixture builders and transcript readers for the header-transcriber tests.

Not named ``test_*`` so pytest does not collect it (same reason as
``tests/agents/planted_defect/suite.py``).

The suite's ground rule is that a fixture and its expected values both derive
from one declarative spec. A golden string pasted from a previous run of the
script under test would encode that run's misunderstanding, so no assertion in
this suite compares against a recorded transcript: every expectation is
recomputed here from ``ModelSpec`` or ``GgufSpec``.

Dimensions are deliberately small. What the transcribers get wrong is
structural — which tensors exist, how names collapse, which layer is
double-wide, which element type an array carries — and none of that depends on
a dimension being 1536 rather than 8. Small shapes keep the fixtures byte-exact
and let the reference ``safetensors`` and ``gguf`` packages load them.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field, replace
from pathlib import Path

RUST_SRC = Path(__file__).resolve().parents[2] / "rust-harness"


def _rust(relpath: str) -> str:
    return (RUST_SRC / relpath).read_text()


def rust_loadable_tensor_types() -> set[int]:
    """GGML type ids ``runtime/src/tensor.rs`` dequantizes.

    Derived rather than restated. Three copies of this set existed -- here, in
    ``gguf_header_transcript.sh``, and in the Rust source itself -- and the two
    copies had drifted to ``{0, 1, 8}`` after tensor.rs gained BF16, so the
    transcript reported every BF16 tensor in a Gemma 4 checkpoint as refused.
    """
    src = _rust("runtime/src/tensor.rs")
    consts = dict(re.findall(r"pub const (GGML_\w+): u32 = (\d+);", src))
    arms = set(re.findall(r"^\s+(GGML_\w+) =>", src, re.MULTILINE))
    out = {int(consts[a]) for a in arms if a in consts}
    assert out, "no loadable tensor types derived from tensor.rs"
    return out


def rust_ok_array_element_types() -> set[int]:
    """GGUF array element type ids ``runtime/src/gguf.rs`` decodes."""
    src = _rust("runtime/src/gguf.rs")
    block = re.search(r"fn array\(.*?\n    \}", src, re.DOTALL)
    assert block, "could not locate fn array in gguf.rs"
    out = {
        int(n)
        for n in re.findall(r"^\s+(\d+) => MetaValue::", block.group(0), re.MULTILINE)
    }
    assert out, "no array element types derived from gguf.rs"
    return out


# Bytes per element, for computing safetensors data_offsets.
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "BOOL": 1}

# GGUF metadata value types.
GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9

# ggml tensor type ids. Which of them the harness loads is not restated here:
# ask rust_loadable_tensor_types(), which reads runtime/src/tensor.rs.
GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_BF16 = 30

_SCALAR_FMT = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    GGUF_UINT32: "<I",
    GGUF_INT32: "<i",
    GGUF_FLOAT32: "<f",
    GGUF_BOOL: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


# --------------------------------------------------------------------------
# safetensors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of a Gemma-4-shaped safetensors repository.

    Every field the transcriber reports is derived from this object, so a test
    asserts against ``spec.expected_*`` rather than against a literal.
    """

    # Deliberately not a name docs/models/gemma-4.md pins: the transcriber
    # blob-hash-checks E2B and E4B against the document, and a fixture can
    # never match those hashes, so a pinned default would make every test
    # exit non-zero for a reason unrelated to what it asserts. Tests that
    # exercise the blob check set the name explicitly.
    name: str = "gemma-4-XYZ"
    num_layers: int = 7
    hidden_size: int = 8
    intermediate_size: int = 16
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 4
    global_head_dim: int = 8
    # Full attention every Nth layer, matching the (i + 1) % period == 0 form
    # docs/models/gemma-4.md records for all four variants.
    full_attention_period: int = 3
    # Tail layers sharing KV cache; also the double-wide MLP region on E2B.
    num_kv_shared_layers: int = 4
    double_wide: bool = True
    vocab_size: int = 64
    hidden_size_per_layer_input: int = 4
    dtype: str = "BF16"
    # Overridable so a test can build a repository whose tensors do not follow
    # the convention the transcriber greps for.
    prefix: str = "model.language_model"
    include_lm_head: bool = False
    # The vision tower supplies the two shapes that break naive parsing: names
    # with more than one numeric segment, and zero-dimensional companions.
    include_vision_tower: bool = True
    extra_tensors: tuple[tuple[str, tuple[int, ...], str], ...] = ()
    config_overrides: dict = field(default_factory=dict)

    # -- derived -----------------------------------------------------------

    def full_attention_indices(self) -> list[int]:
        return [
            i
            for i in range(self.num_layers)
            if (i + 1) % self.full_attention_period == 0
        ]

    def layer_types(self) -> list[str]:
        full = set(self.full_attention_indices())
        return [
            "full_attention" if i in full else "sliding_attention"
            for i in range(self.num_layers)
        ]

    def is_full(self, layer: int) -> bool:
        return (layer + 1) % self.full_attention_period == 0

    def mlp_width(self, layer: int) -> int:
        """Double-wide applies to the final num_kv_shared_layers layers only."""
        if self.double_wide and layer >= self.num_layers - self.num_kv_shared_layers:
            return 2 * self.intermediate_size
        return self.intermediate_size

    def expected_mlp_widths(self) -> list[int]:
        return [self.mlp_width(i) for i in range(self.num_layers)]

    def per_layer_table_width(self) -> int:
        return self.num_layers * self.hidden_size_per_layer_input

    def tensors(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Every tensor, in a fixed order, as (name, shape, dtype)."""
        d, h = self.dtype, self.hidden_size
        p, pl = self.prefix, self.hidden_size_per_layer_input
        out: list[tuple[str, tuple[int, ...], str]] = [
            (f"{p}.embed_tokens.weight", (self.vocab_size, h), d),
            (
                f"{p}.embed_tokens_per_layer.weight",
                (self.vocab_size, self.per_layer_table_width()),
                d,
            ),
            (
                f"{p}.per_layer_model_projection.weight",
                (self.per_layer_table_width(), h),
                d,
            ),
            (f"{p}.per_layer_projection_norm.weight", (pl,), d),
            (f"{p}.norm.weight", (h,), d),
        ]
        for i in range(self.num_layers):
            hd = self.global_head_dim if self.is_full(i) else self.head_dim
            q = self.num_attention_heads * hd
            kv = self.num_key_value_heads * hd
            w = self.mlp_width(i)
            out += [
                (f"{p}.layers.{i}.input_layernorm.weight", (h,), d),
                (f"{p}.layers.{i}.post_attention_layernorm.weight", (h,), d),
                (f"{p}.layers.{i}.pre_feedforward_layernorm.weight", (h,), d),
                (f"{p}.layers.{i}.post_feedforward_layernorm.weight", (h,), d),
                (f"{p}.layers.{i}.post_per_layer_input_norm.weight", (h,), d),
                (f"{p}.layers.{i}.layer_scalar", (1,), d),
                (f"{p}.layers.{i}.mlp.gate_proj.weight", (w, h), d),
                (f"{p}.layers.{i}.mlp.up_proj.weight", (w, h), d),
                (f"{p}.layers.{i}.mlp.down_proj.weight", (h, w), d),
                (f"{p}.layers.{i}.per_layer_input_gate.weight", (pl, h), d),
                (f"{p}.layers.{i}.per_layer_projection.weight", (h, pl), d),
                (f"{p}.layers.{i}.self_attn.q_proj.weight", (q, h), d),
                (f"{p}.layers.{i}.self_attn.k_proj.weight", (kv, h), d),
                (f"{p}.layers.{i}.self_attn.v_proj.weight", (kv, h), d),
                (f"{p}.layers.{i}.self_attn.o_proj.weight", (h, q), d),
                (f"{p}.layers.{i}.self_attn.q_norm.weight", (hd,), d),
                (f"{p}.layers.{i}.self_attn.k_norm.weight", (hd,), d),
            ]
        if self.include_vision_tower:
            for i in range(2):
                for j in range(2):
                    stem = f"model.vision_tower.encoder.layers.{i}.attn.{j}"
                    out.append((f"{stem}.linear.weight", (4, 4), d))
                    # Zero-dimensional clipping companions, trap 12 in
                    # docs/models/gemma-4.md:577.
                    for bound in ("input_min", "input_max"):
                        out.append((f"{stem}.linear.{bound}", (), "F32"))
        if self.include_lm_head:
            out.append(("lm_head.weight", (self.vocab_size, h), d))
        out.extend(self.extra_tensors)
        return out

    def expected_dtype_tally(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for _, _, dt in self.tensors():
            tally[dt] = tally.get(dt, 0) + 1
        return tally

    def expected_shape_of(self, name: str) -> tuple[int, ...]:
        for n, shape, _ in self.tensors():
            if n == name:
                return shape
        raise KeyError(f"{name} not in spec")

    def config(self) -> dict:
        text = {
            "model_type": "gemma4_text",
            "num_hidden_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_kv_shared_layers": self.num_kv_shared_layers,
            "head_dim": self.head_dim,
            "global_head_dim": self.global_head_dim,
            "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
            "vocab_size": self.vocab_size,
            "use_double_wide_mlp": self.double_wide,
            "layer_types": self.layer_types(),
            "dtype": "bfloat16",
        }
        text.update(self.config_overrides)
        return {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "transformers_version": "5.5.0.dev0",
            "bos_token_id": 2,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "text_config": text,
        }


def write_safetensors(path: Path, tensors, metadata: dict | None = None) -> Path:
    """Writes a real safetensors file: 8-byte LE header length, JSON, then data.

    Data is written in full (the shapes here are small), so the reference
    ``safetensors`` package can load the result rather than only our reader.
    """
    header: dict = {}
    offset = 0
    for name, shape, dtype in tensors:
        n = 1
        for dim in shape:
            n *= dim
        size = n * DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    if metadata is not None:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)
    return path


def build_model_dir(root: Path, spec: ModelSpec) -> Path:
    """Materializes a spec as a model directory the transcriber can read."""
    d = root / spec.name
    d.mkdir(parents=True, exist_ok=True)
    write_safetensors(d / "model.safetensors", spec.tensors())
    (d / "config.json").write_text(json.dumps(spec.config(), indent=2))
    (d / "generation_config.json").write_text(
        json.dumps({"temperature": 1.0, "top_k": 64, "top_p": 0.95})
    )
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "GemmaTokenizer",
                "padding_side": "left",
                "added_tokens_decoder": {
                    "0": {"content": "<pad>", "special": True},
                    "1": {"content": "<eos>", "special": True},
                    "2": {"content": "<bos>", "special": True},
                    "5": {"content": "<|turn>", "special": True},
                    "6": {"content": "<turn|>", "special": True},
                },
            }
        )
    )
    return d


def mutate(spec: ModelSpec, **changes) -> ModelSpec:
    """Returns a copy of the spec with fields replaced (mutation testing)."""
    return replace(spec, **changes)


# --------------------------------------------------------------------------
# GGUF
# --------------------------------------------------------------------------


@dataclass
class GgufSpec:
    """Declarative description of a GGUF file, including malformed variants."""

    version: int = 3
    arch: str = "gemma4"
    # (key, value_type, value, element_type or None)
    kvs: list[tuple[str, int, object, int | None]] = field(default_factory=list)
    # (name, dims, ggml_type)
    tensors: list[tuple[str, tuple[int, ...], int]] = field(default_factory=list)

    def tensor_type_tally(self) -> dict[int, int]:
        tally: dict[int, int] = {}
        for _, _, t in self.tensors:
            tally[t] = tally.get(t, 0) + 1
        return tally

    def array_element_types(self) -> dict[str, int]:
        return {k: et for k, _, _, et in self.kvs if et is not None}

    def refused_array_keys(self) -> list[str]:
        ok = rust_ok_array_element_types()
        return sorted(k for k, et in self.array_element_types().items() if et not in ok)

    def refused_tensor_types(self) -> list[int]:
        ok = rust_loadable_tensor_types()
        return sorted(t for t in self.tensor_type_tally() if t not in ok)


def _gguf_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _gguf_value(vtype: int, value, elem_type: int | None) -> bytes:
    if vtype == GGUF_STRING:
        return _gguf_string(value)
    if vtype == GGUF_ARRAY:
        assert elem_type is not None, "array kv needs an element type"
        out = struct.pack("<I", elem_type) + struct.pack("<Q", len(value))
        if elem_type == GGUF_STRING:
            return out + b"".join(_gguf_string(v) for v in value)
        fmt = _SCALAR_FMT[elem_type]
        return out + b"".join(struct.pack(fmt, v) for v in value)
    return struct.pack(_SCALAR_FMT[vtype], value)


def write_gguf(path: Path, spec: GgufSpec, *, magic: bytes = b"GGUF") -> Path:
    """Writes a GGUF file from a spec, including versions and types no
    in-tree writer produces (bool arrays, BF16, v2), which is why this cannot
    reuse runtime/src/synth.rs::GgufWriter."""
    out = bytearray(magic + struct.pack("<I", spec.version))
    out += struct.pack("<Q", len(spec.tensors))
    out += struct.pack("<Q", len(spec.kvs))
    for key, vtype, value, elem in spec.kvs:
        out += _gguf_string(key) + struct.pack("<I", vtype)
        out += _gguf_value(vtype, value, elem)
    payload = 0
    for name, dims, ttype in spec.tensors:
        out += _gguf_string(name) + struct.pack("<I", len(dims))
        out += b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<I", ttype) + struct.pack("<Q", payload)
        payload += 64
    # Alignment padding then a data region, so offsets stay inside the file.
    while len(out) % 32:
        out += b"\0"
    out += b"\0" * max(payload, 64)
    path.write_bytes(bytes(out))
    return path


def gemma_gguf_spec(**overrides) -> GgufSpec:
    """A Gemma-4-shaped GGUF spec carrying both refusal cases by default."""
    tokens = [
        "<pad>",
        "<eos>",
        "<bos>",
        "<|turn>",
        "<turn|>",
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool>",
        "<tool|>",
        "<|think|>",
        "hello",
        "world",
    ]
    spec = GgufSpec(
        kvs=[
            ("general.architecture", GGUF_STRING, "gemma4", None),
            ("general.name", GGUF_STRING, "Gemma 4 E2B It", None),
            ("gemma4.block_count", GGUF_UINT32, 7, None),
            ("gemma4.embedding_length", GGUF_UINT32, 8, None),
            ("gemma4.rope.freq_base", GGUF_FLOAT32, 1000000.0, None),
            # docs/verification/rust-harness-qa.md:736-740 predicts the
            # converter writes this as a bool array, which gguf.rs refuses.
            (
                "gemma4.attention.sliding_window_pattern",
                GGUF_ARRAY,
                [i % 3 != 2 for i in range(7)],
                GGUF_BOOL,
            ),
            ("gemma4.attention.layer_indices", GGUF_ARRAY, list(range(7)), GGUF_INT32),
            ("tokenizer.ggml.model", GGUF_STRING, "llama", None),
            ("tokenizer.ggml.tokens", GGUF_ARRAY, tokens, GGUF_STRING),
            ("tokenizer.ggml.bos_token_id", GGUF_UINT32, 2, None),
            (
                "tokenizer.chat_template",
                GGUF_STRING,
                "{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}"
                "<turn|>\n{% endfor %}" + "x" * 400,
                None,
            ),
        ],
        tensors=(
            [
                ("token_embd.weight", (8, 64), GGML_BF16),
                ("output_norm.weight", (8,), GGML_F32),
            ]
            + [(f"blk.{i}.attn_q.weight", (8, 16), GGML_BF16) for i in range(3)]
            + [(f"blk.{i}.ffn_gate.weight", (8, 16), GGML_BF16) for i in range(3)]
        ),
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


# --------------------------------------------------------------------------
# transcript reading
# --------------------------------------------------------------------------


def read_sections(path: Path) -> dict[str, list[str]]:
    """Splits a safetensors transcript into {section title: body lines}.

    Section headers are written by the script's section() helper as
    ``--- title ---``.
    """
    sections: dict[str, list[str]] = {}
    current = "<preamble>"
    sections[current] = []
    for line in path.read_text().splitlines():
        if line.startswith("--- ") and line.endswith(" ---"):
            current = line[4:-4]
            sections[current] = []
        else:
            sections[current].append(line)
    return sections


def section_matching(sections: dict[str, list[str]], needle: str) -> list[str]:
    """Body of the one section whose title contains needle."""
    hits = [t for t in sections if needle in t]
    if len(hits) != 1:
        raise AssertionError(
            f"expected exactly one section containing {needle!r},"
            f" found {len(hits)}: {hits}"
        )
    return [ln for ln in sections[hits[0]] if ln.strip()]


def parse_counted(lines: list[str]) -> dict[str, int]:
    """Parses ``uniq -c`` output into {rest of line: count}."""
    out: dict[str, int] = {}
    for line in lines:
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            out[parts[1].strip()] = int(parts[0])
    return out


def parse_tensor_list(text: str) -> dict[str, tuple[list[int], str]]:
    """Parses a GGUF transcript's full tensor name list into {name: (dims, type)}.

    Dimensions are GGUF order, which is reversed relative to safetensors.
    """
    out: dict[str, tuple[list[int], str]] = {}
    body = text.split("=== full tensor name list ===", 1)
    if len(body) != 2:
        return out
    for line in body[1].splitlines():
        m = re.match(r"^(\S+) \[([0-9, ]*)\] (\S+)$", line.strip())
        if m:
            dims = [int(x) for x in m.group(2).split(",") if x.strip()]
            out[m.group(1)] = (dims, m.group(3))
    return out


def metadata_keys(text: str) -> set[str]:
    """Keys named in a GGUF transcript's non-tokenizer metadata section."""
    keys = set()
    chunk = text.split("=== metadata keys (non-tokenizer) ===", 1)
    if len(chunk) != 2:
        return keys
    section = chunk[1].split("=== array-valued keys", 1)[0]
    for line in section.splitlines():
        m = re.match(r"^(\S+) \[", line)
        if m:
            keys.add(m.group(1))
    return keys


def metadata_scalars(text: str) -> dict[str, str]:
    """Scalar metadata values from a GGUF transcript, as {key: raw value text}.

    Array-valued keys carry no ``=`` on the key line and are skipped; read them
    with :func:`metadata_keys` instead.
    """
    out: dict[str, str] = {}
    chunk = text.split("=== metadata keys (non-tokenizer) ===", 1)
    if len(chunk) != 2:
        return out
    section = chunk[1].split("=== array-valued keys", 1)[0]
    for line in section.splitlines():
        m = re.match(r"^(\S+) \[[^\]]*\] = (.*)$", line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def split_gguf_files(path: Path) -> dict[str, str]:
    """Splits a multi-file GGUF transcript into {source .gguf path: block text}.

    The committed ``docs/models/gemma4-gguf.txt`` covers six checkpoints. Each is
    introduced by a ``FILE:`` line inside a row of ``=`` separators; the trailing
    separator belongs to the *next* block's banner, so it is stripped here rather
    than left to leak into the previous block's tensor list.
    """
    blocks: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in path.read_text().splitlines():
        if set(line.strip()) == {"="} and line.strip():
            continue
        m = re.match(r"^FILE: (\S+)$", line)
        if m:
            if current is not None:
                blocks[current] = "\n".join(buf)
            current = m.group(1)
            buf = []
            continue
        if current is not None:
            buf.append(line)
    if current is not None:
        blocks[current] = "\n".join(buf)
    return blocks
