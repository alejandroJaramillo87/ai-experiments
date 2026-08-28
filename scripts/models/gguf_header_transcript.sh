#!/bin/bash
#
# gguf_header_transcript.sh — transcribe GGUF metadata and tensor inventory
# into a reviewable file, without loading the model.
#
# Usage: gguf_header_transcript.sh [--no-color] [--out FILE] [gguf-file ...]
#
# Reads only the GGUF header: magic, version, the metadata key-value block, and
# the tensor directory. On a 15GB file that is a few megabytes at the front, so
# the run costs seconds and needs no GPU.
#
# Written to answer what a GGUF a llama.cpp service is about to serve actually
# contains:
#
#   1. Which metadata value types appear, and which keys are array-valued.
#   2. Which special tokens and chat template the tokenizer block carries.
#   3. Which ggml tensor types appear, and in what proportion.
#
# It used to answer a fourth question -- whether rust-harness could load the
# file -- and derived the accepted type sets by reading runtime/src/tensor.rs
# and runtime/src/gguf.rs at run time rather than copying them, because a copy
# had already drifted and reported 318 of 601 tensors in the E2B base file as
# REFUSED when tensor.rs had in fact gained BF16.
#
# R10 removed the question rather than the derivation. rust-harness reads no
# GGUF at all now: P8b deleted the GGUF model path, and R10 deleted gguf.rs
# with the last artifact that used it. A loadability verdict against a reader
# that does not exist is worse than a drifted one, so the verdicts are gone and
# what remains is the inventory. GGUF is llama.cpp's format here, produced by
# training/export/ and served by the llama services.
#
# Host-side only: /mnt/ai-data is not reachable from an agent sandbox.

set -euo pipefail


MODEL_ROOT=${MODEL_ROOT:-/mnt/ai-data/models}
OUT=${OUT:-/tmp/gguf-transcript.txt}

RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'

if [[ -n "${NO_COLOR:-}" ]] || [[ "${1:-}" == "--no-color" ]]; then
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
    [[ "${1:-}" == "--no-color" ]] && shift
fi

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }

usage() {
    echo "Usage: $0 [--no-color] [--out FILE] [gguf-file ...]"
    exit 2
}

if [[ "${1:-}" == "--out" ]]; then
    [[ -n "${2:-}" ]] || usage
    OUT=$2
    shift 2
fi
[[ "${1:-}" == -* ]] && usage

# Minimal GGUF v2/v3 reader. Deliberately dependency-free rather than importing
# the pinned gguf package: an independent parser cross-checks the format the way
# .styles/RUST_STYLE.md requires of the npz and probe tests, and it runs on a
# host whose virtualenv may not be provisioned.
read_gguf() {
    python3 - "$1" <<'PYEOF'
import os
import re
import struct
import sys
from collections import Counter

# GGUF metadata value types.
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

TYPE_NAMES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32",
    6: "float32", 7: "bool", 8: "string", 9: "array", 10: "uint64",
    11: "int64", 12: "float64",
}

# ggml tensor types, as of the b4xxx-era llama.cpp enumeration.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
}

# What rust-harness will actually accept, read from the Rust source rather than
# restated here. See the note in the file header for why this is derived.
# GGUF version this reader transcribes. A container property from the GGUF
# specification, not a consumer's requirement: it was read out of gguf.rs until
# R10 deleted that file, and nothing in this repository parses GGUF now.
GGUF_VERSION = 3

SCALARS = {
    T_UINT8: ("<B", 1), T_INT8: ("<b", 1), T_UINT16: ("<H", 2),
    T_INT16: ("<h", 2), T_UINT32: ("<I", 4), T_INT32: ("<i", 4),
    T_FLOAT32: ("<f", 4), T_BOOL: ("<?", 1), T_UINT64: ("<Q", 8),
    T_INT64: ("<q", 8), T_FLOAT64: ("<d", 8),
}


class Reader:
    """Sequential little-endian reader over the GGUF header."""

    def __init__(self, f, size):
        self.f = f
        self.size = size

    def raw(self, n):
        # A length wider than the file means the header is malformed or the
        # download truncated. Reject it here: passing it to read() yields a
        # MemoryError whose traceback says nothing about the actual cause.
        if n > self.size:
            raise ValueError(
                f"declared length {n} exceeds file size {self.size} "
                f"at offset {self.f.tell()}; header is malformed or truncated"
            )
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(
                f"short read at offset {self.f.tell()}: wanted {n}, got {len(b)}"
            )
        return b

    def scalar(self, t):
        fmt, size = SCALARS[t]
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self):
        n = struct.unpack("<Q", self.raw(8))[0]
        return self.raw(n).decode("utf-8", errors="replace")

    def value(self, t):
        """Returns (value, array_element_type or None)."""
        if t == T_STRING:
            return self.string(), None
        if t == T_ARRAY:
            et = struct.unpack("<I", self.raw(4))[0]
            n = struct.unpack("<Q", self.raw(8))[0]
            if et == T_STRING:
                return [self.string() for _ in range(n)], et
            if et == T_ARRAY:
                return [self.value(T_ARRAY)[0] for _ in range(n)], et
            fmt, size = SCALARS[et]
            return list(struct.unpack(f"<{n}{fmt[1]}", self.raw(size * n))), et
        return self.scalar(t), None


def summarize(v, limit=24):
    """Renders a value, truncating long arrays so the transcript stays readable.

    limit=None prints in full. Per-layer schedules are printed that way: an
    array whose length equals block_count answers "which layers are sliding",
    and a truncated one answers it only for the first 24 layers. On E4B that
    hid 18 of 42 entries, which is the half that decides where the two RoPE
    regimes and the two head dimensions change over.
    """
    if isinstance(v, list):
        if limit is None or len(v) <= limit:
            return "[" + ", ".join(repr(x) for x in v) + "]"
        body = ", ".join(repr(x) for x in v[:limit])
        return f"[{body}, ... (+{len(v) - limit} more)]"
    return repr(v)


path = sys.argv[1]
size = os.path.getsize(path)
with open(path, "rb") as f:
    r = Reader(f, size)

    magic = r.raw(4)
    if magic != b"GGUF":
        print(f"NOT A GGUF FILE: magic={magic!r}")
        sys.exit(1)

    version = struct.unpack("<I", r.raw(4))[0]
    if version < 2:
        print(f"GGUF v{version} is not supported by this reader")
        sys.exit(1)

    n_tensors = struct.unpack("<Q", r.raw(8))[0]
    n_kv = struct.unpack("<Q", r.raw(8))[0]

    # This reader handles v2 and v3 alike. v2 is noted rather than refused:
    # the transcript describes whatever it was given, and the version is part
    # of that description.
    if version != GGUF_VERSION:
        print(f"note: version {version}, not the expected {GGUF_VERSION}")
    print(f"gguf version: {version}")
    print(f"tensor count: {n_tensors}")
    print(f"metadata kv count: {n_kv}")
    print()

    kv, kv_types, elem_types = {}, {}, {}
    for _ in range(n_kv):
        key = r.string()
        vtype = struct.unpack("<I", r.raw(4))[0]
        val, etype = r.value(vtype)
        kv[key] = val
        kv_types[key] = vtype
        if etype is not None:
            elem_types[key] = etype

    # Lengths that identify a per-layer schedule, so it prints in full.
    block_counts = {v for k, v in kv.items()
                    if k.endswith(".block_count") and isinstance(v, int)}

    print("=== metadata keys (non-tokenizer) ===")
    print()
    for key in sorted(k for k in kv if not k.startswith("tokenizer.")):
        if kv_types[key] == T_ARRAY:
            et = elem_types.get(key)
            print(f"{key} [array of {TYPE_NAMES.get(et, f'?{et}')},"
                  f" n={len(kv[key])}]")
            per_layer = len(kv[key]) in block_counts
            print(f"    {summarize(kv[key], None if per_layer else 24)}")
        else:
            tname = TYPE_NAMES.get(kv_types[key], f"?{kv_types[key]}")
            print(f"{key} [{tname}] = {summarize(kv[key])}")

    print()
    print("=== array-valued keys by element type ===")
    print()
    for et, count in sorted(Counter(elem_types.values()).items()):
        print(f"element type {et} ({TYPE_NAMES.get(et, '?')}):"
              f" {count} keys")

    print()
    print("=== tokenizer metadata ===")
    print()
    for key in sorted(k for k in kv if k.startswith("tokenizer.")):
        v = kv[key]
        if isinstance(v, list):
            et = elem_types.get(key)
            print(f"{key} [array of {TYPE_NAMES.get(et, '?')}, n={len(v)}]")
            if len(v) > 64:
                print(f"    first 8: {summarize(v[:8], 8)}")
                print(f"    last 8:  {summarize(v[-8:], 8)}")
            else:
                print(f"    {summarize(v)}")
        elif isinstance(v, str) and len(v) > 400:
            # The chat template is the authoritative turn-marker source; print
            # it whole rather than truncating.
            print(f"{key} [string, {len(v)} chars]")
            print("    --- begin ---")
            for line in v.splitlines():
                print(f"    {line}")
            print("    --- end ---")
        else:
            tname = TYPE_NAMES.get(kv_types[key], f"?{kv_types[key]}")
            print(f"{key} [{tname}] = {summarize(v)}")

    print()
    print("=== special-token strings (turn/channel/tool/think markers) ===")
    print()
    tokens = kv.get("tokenizer.ggml.tokens", [])
    if tokens:
        needles = ("turn", "channel", "tool", "think", "<bos>", "<eos>",
                   "<pad>", "<unk>", "start_of", "end_of")
        hits = [(i, t) for i, t in enumerate(tokens)
                if any(n in t for n in needles)]
        print(f"vocab size: {len(tokens)}")
        print(f"matching entries: {len(hits)}")
        for i, t in hits[:120]:
            print(f"  {i}: {t!r}")
        if len(hits) > 120:
            print(f"  ... (+{len(hits) - 120} more)")
    else:
        print("tokenizer.ggml.tokens absent")

    print()
    print("=== tensor inventory ===")
    print()
    names, type_counts, collapsed = [], Counter(), Counter()
    for _ in range(n_tensors):
        name = r.string()
        ndims = struct.unpack("<I", r.raw(4))[0]
        dims = list(struct.unpack(f"<{ndims}Q", r.raw(8 * ndims)))
        ttype = struct.unpack("<I", r.raw(4))[0]
        struct.unpack("<Q", r.raw(8))[0]   # data offset, unused for inventory
        names.append((name, dims, ttype))
        type_counts[ttype] += 1
        # re.sub replaces every occurrence, so a name with more than one
        # index collapses fully. Overlapping matches (blk.0.1.w) would not,
        # hence the explicit second pass.
        key = re.sub(r"\.\d+\.", ".{L}.", name)
        key = re.sub(r"\.\d+\.", ".{L}.", key)
        collapsed[(key, tuple(dims), ttype)] += 1

    print("ggml tensor types present:")
    for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {GGML_TYPES.get(t, f'unknown({t})')} (type {t}):"
              f" {c} tensors")

    print()
    print("inventory, layer indices collapsed:")
    for (key, dims, ttype), count in sorted(collapsed.items(),
                                            key=lambda kv: -kv[1]):
        print(f"  {count:5d}  {key} {list(dims)} {GGML_TYPES.get(ttype, ttype)}")

    print()
    print("=== full tensor name list ===")
    print()
    for name, dims, ttype in names:
        print(f"{name} {dims} {GGML_TYPES.get(ttype, ttype)}")
PYEOF
}

command -v python3 >/dev/null 2>&1 || {
    print_error "python3 not found on PATH"
    echo "  Solution: this script needs python3 to parse the GGUF header"
    exit 1
}

if [[ $# -gt 0 ]]; then
    FILES=("$@")
else
    mapfile -t FILES < <(find "$MODEL_ROOT/gguf" -name '*.gguf' 2>/dev/null | sort)
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    print_error "No GGUF files found under $MODEL_ROOT/gguf"
    echo "  Solution: pass file paths explicitly, or set MODEL_ROOT"
    echo "  Command: find $MODEL_ROOT -name '*.gguf'"
    exit 1
fi

echo "=== GGUF header transcript ==="
echo
print_info "Transcript: $OUT"
print_info "Files: ${#FILES[@]}"
echo

{
    echo "GGUF header transcript"
    echo "Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Host: $(hostname)"
    echo "Checks: runtime/src/gguf.rs array element types, runtime/src/tensor.rs"
    echo "        tensor types, tokenizer markers for harness/src/template.rs"
} > "$OUT"

status=0

for file in "${FILES[@]}"; do
    base=$(basename "$file")
    echo "$base"

    if [[ ! -f "$file" ]]; then
        print_error "  Not found: $file"
        status=1
        continue
    fi

    {
        echo
        echo "================================================================"
        echo "FILE: $file"
        echo "SIZE: $(stat -c %s "$file") bytes"
        echo "================================================================"
        echo
    } >> "$OUT"

    print_run "  Reading GGUF header"
    if read_gguf "$file" >> "$OUT" 2>&1; then
        print_ok "  Header transcribed"
    else
        print_error "  Failed to parse the GGUF header"
        echo "  Context: $file"
        echo "  Solution: confirm the download completed and is not truncated"
        echo "  Command: head -c 4 \"$file\" | xxd"
        echo "[parse failed]" >> "$OUT"
        status=1
    fi
done

echo
echo "Summary:"
print_info "Transcript written: $OUT ($(wc -l < "$OUT") lines, $(stat -c %s "$OUT") bytes)"

# Surface the blocking findings without requiring a read of the whole file.
#
# Attribute every finding to its file. The bare `grep ... | sort -u` this
# replaces dropped any line two files shared: six BF16 files reported as four
# lines, and four separate bool-array keys reported as "1 keys", because the
# lines carry counts but not filenames. Under-reporting reads as a smaller
# problem rather than as a missing one, so scope the summary by FILE: instead
# of deduplicating text that was never unique per file.
refusals() {
    awk -v pat="$1" '
        /^FILE: / { n = split($2, p, "/"); file = p[n] }
        $0 ~ pat  { sub(/^ +/, ""); printf "  %s: %s\n", file, $0 }
    ' "$OUT"
}
if grep -q 'REFUSED by gguf.rs' "$OUT"; then
    print_warn "Metadata arrays use element types rust-harness refuses"
    refusals 'REFUSED by gguf.rs'
fi
if grep -q 'REFUSED by tensor.rs' "$OUT"; then
    print_warn "Tensor types rust-harness cannot load are present"
    refusals 'REFUSED by tensor.rs'
fi

if [[ $status -eq 0 ]]; then
    print_ok "All files parsed"
else
    print_warn "One or more files failed to parse"
fi

exit $status
