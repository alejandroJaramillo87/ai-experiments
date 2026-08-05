#!/bin/bash
#
# gemma4_header_transcript.sh — transcribe Gemma 4 architecture facts from
# locally downloaded safetensors weights into a single reviewable file.
#
# Usage: gemma4_header_transcript.sh [--no-color] [--out FILE] [model-dir ...]
#
# Produces the evidence needed to verify docs/models/gemma-4.md against the
# actual weights rather than against the session that wrote it. Reads only the
# safetensors header (a few hundred kilobytes of a multi-gigabyte file) plus the
# JSON side files, so it needs no GPU and does not load a model.
#
# Defaults to the E2B and E4B directories under $MODEL_ROOT/hf. Pass explicit
# directories to override.
#
# Host-side only: /mnt/ai-data is not reachable from an agent sandbox.

set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/mnt/ai-data/models}
OUT=${OUT:-/tmp/gemma4-transcript.txt}

# The per-layer tensor naming docs/models/gemma-4.md records (:363-392). Every
# inventory section filters on it, so a file that does not use it produces
# empty sections rather than wrong ones; the coverage guard below turns that
# silence into an error.
LAYER_CONVENTION='model\.language_model\.layers\.[0-9]+\.'

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
    echo "Usage: $0 [--no-color] [--out FILE] [model-dir ...]"
    exit 2
}

if [[ "${1:-}" == "--out" ]]; then
    [[ -n "${2:-}" ]] || usage
    OUT=$2
    shift 2
fi
[[ "${1:-}" == -* ]] && usage

# Expected revisions from docs/models/gemma-4.md. A transcript taken from a
# different revision does not verify that document, so the mismatch is reported
# rather than silently tolerated.
expected_revision() {
    case "$1" in
        *E2B*) echo "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f" ;;
        *E4B*) echo "411aa17b749aa952df1359d2dcea73917a544d9a" ;;
        *)     echo "" ;;
    esac
}

# Git blob hashes of config.json from docs/models/gemma-4.md, recomputable
# locally with `git hash-object` — an integrity check on the file itself that
# does not depend on how the download recorded its revision.
expected_blob() {
    case "$1" in
        *E2B*) echo "5f0662eb398ede0ab1515c4760e13bcd6bf13f2f" ;;
        *E4B*) echo "080fd8d51bb6e846df168157c9d57b928a2a2687" ;;
        *)     echo "" ;;
    esac
}

# collapse — rewrite every numeric path segment to {L} on stdin.
#
# The trailing g matters. Tower tensors carry more than one index
# (`...encoder.layers.0.attn.1.linear.weight`), and collapsing only the first
# leaves the rest distinct, so one family of four tensors reports as two rows
# of two. That reads as a structural finding rather than a formatting artifact.
collapse() {
    sed 's/\.[0-9][0-9]*\./.{L}./g'
}

# layer_index_of <line> — the layer index embedded in a tensor name.
#
# Used to sort per-layer listings numerically. Sorting on a fixed dot-field
# assumes every tensor name is the same depth; when it is not, layer 10 sorts
# between 1 and 2 and the double-wide boundary is read off a wrong order.
sort_by_layer() {
    sed -E 's/^([^ ]*\.layers\.([0-9]+)\..*)$/\2 \1/' \
        | sort -k1,1n -k2 \
        | sed -E 's/^[0-9]+ //'
}

# section <title> — write a header into the transcript.
section() {
    {
        echo
        echo "--- $* ---"
        echo
    } >> "$OUT"
}

# run_step <label> <command...> — run a command, append its output to the
# transcript, and report status. A failing step is recorded, not fatal: a
# missing side file is itself a finding about the repository layout.
run_step() {
    local label=$1
    shift
    if "$@" >> "$OUT" 2>&1; then
        print_ok "$label"
    else
        print_warn "$label: command failed (recorded in transcript)"
        echo "[step failed: $label]" >> "$OUT"
    fi
}

# Reads the safetensors header. The format stores an eight-byte little-endian
# header length followed by that many bytes of JSON, so the tensor map is
# retrievable without touching the weight payload.
read_header() {
    python3 - "$1" "$2" <<'EOF'
import json
import sys

path, out = sys.argv[1], sys.argv[2]
with open(path, "rb") as f:
    n = int.from_bytes(f.read(8), "little")
    header = json.loads(f.read(n))
with open(out, "w") as f:
    json.dump(header, f)
tensors = [k for k in header if k != "__metadata__"]
print(f"header bytes: {n}")
print(f"tensor count: {len(tensors)}")
print(f"metadata: {json.dumps(header.get('__metadata__', {}))}")
EOF
}

# Flattens the header to one "name shape dtype" line per tensor.
flatten_header() {
    python3 - "$1" <<'EOF'
import json
import sys

d = json.load(open(sys.argv[1]))
for name, info in d.items():
    if name != "__metadata__":
        print(name, info["shape"], info["dtype"])
EOF
}

# Extracts the text-config fields that docs/models/gemma-4.md tabulates, plus
# the explicit layer_types schedule.
# shellcheck disable=SC2317  # invoked indirectly through run_step
dump_config() {
    python3 - "$1" <<'EOF'
import json
import sys

c = json.load(open(sys.argv[1]))
t = c.get("text_config", c)

print("architectures:", c.get("architectures", "<absent>"))
print("top-level model_type:", c.get("model_type", "<absent>"))
print("text model_type:", t.get("model_type", "<absent>"))
print("transformers_version:", c.get("transformers_version", "<absent>"))
print()

fields = [
    "num_hidden_layers", "hidden_size", "intermediate_size",
    "num_attention_heads", "num_key_value_heads", "num_global_key_value_heads",
    "num_kv_shared_layers", "hidden_size_per_layer_input",
    "vocab_size_per_layer_input", "sliding_window", "max_position_embeddings",
    "attention_k_eq_v", "use_double_wide_mlp", "use_bidirectional_attention",
    "head_dim", "global_head_dim", "vocab_size", "hidden_activation",
    "rms_norm_eps", "final_logit_softcapping", "tie_word_embeddings",
    "attention_bias", "attention_dropout", "initializer_range",
    "enable_moe_block", "num_experts", "top_k_experts",
    "rope_theta", "rope_local_base_freq", "partial_rotary_factor", "dtype",
]
for k in fields:
    print(f"{k}: {t.get(k, '<absent>')}")

print()
for k in ("bos_token_id", "eos_token_id", "pad_token_id", "boi_token_id",
          "boa_token_id", "image_token_id", "audio_token_id", "eoi_token_id",
          "eoa_token_id", "video_token_id"):
    print(f"{k}: {c.get(k, t.get(k, '<absent>'))}")

print()
rope = t.get("rope_parameters", t.get("rope_scaling", "<absent>"))
print("rope_parameters/rope_scaling:", json.dumps(rope) if rope != "<absent>" else rope)

print()
lt = t.get("layer_types", [])
print("layer_types length:", len(lt))
print("distinct:", sorted(set(lt)))
print("full_attention indices:", [i for i, x in enumerate(lt) if x == "full_attention"])
if lt:
    print("final layer type:", lt[-1])
EOF
}

# Lists the special-token strings the tokenizer defines, without loading a
# tokenizer library. docs/models/gemma-4.md asserts an asymmetric paired-marker
# scheme (<|turn> / <turn|>) that rust-harness does not yet implement.
# shellcheck disable=SC2317  # invoked indirectly through run_step
dump_tokenizer() {
    python3 - "$1" <<'EOF'
import json
import sys

c = json.load(open(sys.argv[1]))
print("tokenizer_class:", c.get("tokenizer_class", "<absent>"))
print("padding_side:", c.get("padding_side", "<absent>"))
print("model_max_length:", c.get("model_max_length", "<absent>"))
print()
added = c.get("added_tokens_decoder", {})
print(f"added_tokens_decoder entries: {len(added)}")
interesting = ("turn", "channel", "tool", "think", "bos", "eos", "pad", "unk", "mask")
for tid, spec in sorted(added.items(), key=lambda kv: int(kv[0])):
    content = spec.get("content", "")
    if any(w in content for w in interesting):
        print(f"  {tid}: {content!r} special={spec.get('special')}")
print()
print("chat_template present:", "chat_template" in c)
EOF
}

command -v python3 >/dev/null 2>&1 || {
    print_error "python3 not found on PATH"
    echo "  Solution: this script needs python3 to parse JSON headers"
    exit 1
}

if [[ $# -gt 0 ]]; then
    DIRS=("$@")
else
    DIRS=("$MODEL_ROOT/hf/gemma-4-E2B" "$MODEL_ROOT/hf/gemma-4-E4B")
fi

echo "=== Gemma 4 header transcript ==="
echo
print_info "Transcript: $OUT"
echo

: > "$OUT"
{
    echo "Gemma 4 header transcript"
    echo "Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Host: $(hostname)"
    echo "Verifies: docs/models/gemma-4.md"
} >> "$OUT"

status=0

for dir in "${DIRS[@]}"; do
    name=$(basename "$dir")
    echo "$name"

    if [[ ! -d "$dir" ]]; then
        print_error "Directory not found: $dir"
        echo "  Solution: pass the correct path, or set MODEL_ROOT"
        echo "  Command: ls -la $MODEL_ROOT"
        section "$name — NOT FOUND at $dir"
        status=1
        continue
    fi

    section "$name — repository layout ($dir)"
    run_step "  Repository listing" ls -la "$dir"

    section "$name — shard index presence"
    if [[ -f "$dir/model.safetensors.index.json" ]]; then
        print_warn "  Shard index present: contradicts docs/models/gemma-4.md:74-76"
        echo "model.safetensors.index.json EXISTS (doc claims it should not)" >> "$OUT"
        run_step "  Shard list" python3 -m json.tool "$dir/model.safetensors.index.json"
    else
        print_ok "  No shard index (matches doc)"
        echo "model.safetensors.index.json absent, as documented" >> "$OUT"
    fi

    section "$name — recorded revision"
    exp_rev=$(expected_revision "$name")
    {
        echo "expected revision (from doc): ${exp_rev:-<none recorded>}"
        for ref in "$dir/.git/HEAD" "$dir/refs/main" "$dir/../refs/main"; do
            [[ -f "$ref" ]] && echo "found $ref: $(cat "$ref")"
        done
        echo "resolved path: $(readlink -f "$dir")"
    } >> "$OUT" 2>&1
    print_info "  Expected revision: ${exp_rev:-<none recorded>}"

    if [[ -f "$dir/config.json" ]]; then
        section "$name — config.json blob hash"
        exp_blob=$(expected_blob "$name")
        got_blob=$(git hash-object "$dir/config.json" 2>/dev/null || echo "<git unavailable>")
        {
            echo "expected blob (from doc): ${exp_blob:-<none recorded>}"
            echo "actual blob:              $got_blob"
        } >> "$OUT"
        if [[ -n "$exp_blob" && "$got_blob" == "$exp_blob" ]]; then
            print_ok "  config.json blob hash matches the pinned revision"
        elif [[ -n "$exp_blob" ]]; then
            print_warn "  config.json blob hash differs from the doc"
            echo "  Expected: $exp_blob"
            echo "  Found:    $got_blob"
            status=1
        fi

        section "$name — config.json (documented fields)"
        run_step "  Config fields and layer_types" dump_config "$dir/config.json"

        section "$name — config.json (verbatim)"
        run_step "  Full config" python3 -m json.tool "$dir/config.json"
    else
        print_error "  config.json not found in $dir"
        section "$name — config.json MISSING"
        status=1
    fi

    for side in generation_config.json processor_config.json; do
        if [[ -f "$dir/$side" ]]; then
            section "$name — $side"
            run_step "  $side" python3 -m json.tool "$dir/$side"
        fi
    done

    if [[ -f "$dir/tokenizer_config.json" ]]; then
        section "$name — tokenizer special tokens"
        run_step "  Tokenizer markers" dump_tokenizer "$dir/tokenizer_config.json"
    fi

    weights="$dir/model.safetensors"
    if [[ ! -f "$weights" ]]; then
        print_warn "  model.safetensors not found; skipping tensor sections"
        section "$name — model.safetensors MISSING"
        run_step "  Safetensors files present" find "$dir" -name '*.safetensors' -printf '%f %s\n'
        status=1
        echo
        continue
    fi

    hdr_json=$(mktemp -t "gemma4-hdr-XXXXXX.json")
    flat=$(mktemp -t "gemma4-flat-XXXXXX.txt")
    # Held until the model's sections are written; the flat listing feeds every
    # tensor query below.
    trap 'rm -f "$hdr_json" "$flat"' EXIT

    section "$name — safetensors header"
    print_run "  Reading safetensors header"
    if ! read_header "$weights" "$hdr_json" >> "$OUT" 2>&1; then
        print_error "  Failed to parse the safetensors header"
        echo "  Context: $weights"
        echo "  Solution: confirm the download completed and is not an LFS pointer"
        echo "  Command: head -c 64 \"$weights\" | xxd | head -4"
        status=1
        rm -f "$hdr_json" "$flat"
        trap - EXIT
        echo
        continue
    fi
    print_ok "  Header parsed"

    flatten_header "$hdr_json" > "$flat"
    {
        echo "file size bytes: $(stat -c %s "$weights")"
        echo "flat tensor lines: $(wc -l < "$flat")"
    } >> "$OUT"

    # Coverage guard. Every inventory section below filters the flat listing
    # with a grep that can legitimately match nothing. If the file names its
    # tensors differently than expected, all of them match nothing at once,
    # and the transcript is empty but reads as clean: "no lm_head (matches
    # doc)", no per-layer tensors, exit 0. That is a full clean bill of health
    # on a file the script never understood, so it is an error, not a warning.
    total_tensors=$(wc -l < "$flat")
    matched=$(grep -cE "$LAYER_CONVENTION" "$flat" || true)
    section "$name — naming convention coverage"
    {
        echo "expected convention: $LAYER_CONVENTION"
        echo "tensors total:       $total_tensors"
        echo "tensors matching:    $matched"
    } >> "$OUT"
    if [[ "$total_tensors" -gt 0 && "$matched" -eq 0 ]]; then
        sample=$(awk 'NR==1 {print $1}' "$flat")
        print_error "  $name: no tensor matched the expected naming convention"
        echo "  Context: $total_tensors tensors present, 0 matched $LAYER_CONVENTION"
        echo "  Sample:  $sample"
        echo "  Solution: tensor naming differs from docs/models/gemma-4.md;"
        echo "            the inventory sections in the transcript are NOT trustworthy"
        {
            echo
            echo "ERROR: no tensor matched $LAYER_CONVENTION"
            echo "the inventory sections below are not trustworthy"
        } >> "$OUT"
        status=1
    else
        print_ok "  Naming convention: $matched/$total_tensors tensors matched"
    fi

    # Config-vs-weights agreement. config.json and model.safetensors are
    # separate downloads, so nothing guarantees they describe the same model.
    # The E2B snapshot this script was first run against shipped E4B's
    # config.json verbatim -- same blob hash, 42 layers and width 2560 declared
    # next to weights holding 35 layers at width 1536. The blob-hash check
    # noticed only that the hash was unexpected, which reads as a revision
    # drift; it cannot distinguish a newer config from another model's config.
    # Comparing the declared layer count against the observed one does.
    section "$name — config vs weights (declared layer count)"
    declared=$(python3 -c '
import json, sys
c = json.load(open(sys.argv[1]))
t = c.get("text_config", c)
print(t.get("num_hidden_layers", ""))
' "$dir/config.json" 2>/dev/null || true)
    observed=$(grep -oE "${LAYER_CONVENTION}" "$flat" \
        | grep -oE '[0-9]+' | sort -n | tail -1)
    # Highest index plus one. Empty (no matching tensors) becomes 0, which the
    # naming-convention guard above has already reported as an error.
    observed=$(( ${observed:--1} + 1 ))
    {
        echo "config.json num_hidden_layers: ${declared:-<unreadable>}"
        echo "distinct layer indices in weights: $observed"
    } >> "$OUT"
    if [[ -n "$declared" && "$declared" -ne "$observed" ]]; then
        print_error "  $name: config.json describes a different model than the weights"
        echo "  Expected: config num_hidden_layers == layer count in model.safetensors"
        echo "  Found:    config says $declared, weights hold $observed"
        echo "  Solution: re-download config.json for this variant; the local copy"
        echo "            belongs to another variant and every scalar read from it is wrong"
        echo "  Command: hf download google/$name config.json --local-dir \"$dir\""
        {
            echo
            echo "ERROR: config declares $declared layers, weights hold $observed"
            echo "every config-derived value in this transcript is untrustworthy"
        } >> "$OUT"
        status=1
    else
        print_ok "  Config vs weights: $observed layers, agrees"
    fi

    section "$name — tensor inventory (layer indices collapsed)"
    collapse < "$flat" | sort | uniq -c | sort -rn >> "$OUT"

    section "$name — language_model inventory only"
    grep language_model "$flat" | collapse \
        | sort | uniq -c | sort -rn >> "$OUT" || echo "no language_model tensors" >> "$OUT"

    section "$name — lm_head presence (doc asserts absent)"
    if grep -q lm_head "$flat"; then
        print_warn "  lm_head present: contradicts docs/models/gemma-4.md:400"
        grep lm_head "$flat" >> "$OUT"
        status=1
    else
        print_ok "  No lm_head (matches doc)"
        echo "no lm_head tensor, as documented" >> "$OUT"
    fi

    section "$name — dtype distribution"
    awk '{print $NF}' "$flat" | sort | uniq -c | sort -rn >> "$OUT"

    section "$name — MLP widths per layer (double-wide check)"
    grep -E "${LAYER_CONVENTION}mlp\.gate_proj\.weight " "$flat" | sort_by_layer >> "$OUT" \
        || echo "no mlp.gate_proj.weight tensors" >> "$OUT"

    section "$name — distinct MLP widths"
    # grep -o emits only the matched text, so a tensor without a two-
    # dimensional shape drops out instead of being counted as a width. The
    # substitution form this replaces passed non-matching lines through
    # unchanged, putting whole tensor lines into the tally.
    # Restricted to the language-model layer convention: the vision tower also
    # has mlp.gate_proj, with zero-dimensional input_min/input_max/output_min/
    # output_max companions whose shape is []. Those carry no width, and a bare
    # 'mlp\.gate_proj' match pulls 144 of them into what should be a two-row
    # tally of 6144 and 12288.
    grep -E "${LAYER_CONVENTION}mlp\.gate_proj\.weight " "$flat" \
        | grep -oE '\[[0-9]+,' | tr -d '[,' \
        | sort -n | uniq -c >> "$OUT" || true

    section "$name — attention projection shapes (head-dim split)"
    grep -E 'self_attn\.(q|k|v|o)_proj' "$flat" | collapse \
        | sort | uniq -c | sort -rn >> "$OUT" || true

    section "$name — per-layer embedding path"
    grep -E 'layer_scalar|per_layer' "$flat" | collapse \
        | sort | uniq -c | sort -rn >> "$OUT" || true

    section "$name — zero-dimensional tensors (doc trap 12)"
    if grep -q '\[\]' "$flat"; then
        grep '\[\]' "$flat" | collapse | sort | uniq -c >> "$OUT"
    else
        echo "none" >> "$OUT"
    fi

    section "$name — non-language_model tensor families"
    grep -v language_model "$flat" | collapse \
        | awk '{print $1}' | cut -d. -f1-3 | sort | uniq -c | sort -rn >> "$OUT" || true

    rm -f "$hdr_json" "$flat"
    trap - EXIT
    print_ok "  Tensor sections written"
    echo
done

echo "Summary:"
print_info "Transcript written: $OUT ($(wc -l < "$OUT") lines, $(stat -c %s "$OUT") bytes)"
if [[ $status -eq 0 ]]; then
    print_ok "All checks matched docs/models/gemma-4.md"
else
    print_warn "One or more checks disagreed with the doc or could not run"
    echo "  Review the transcript before treating the doc as accurate"
fi

exit $status
