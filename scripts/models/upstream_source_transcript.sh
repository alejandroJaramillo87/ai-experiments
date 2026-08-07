#!/bin/bash
#
# upstream_source_transcript.sh — fetch the pinned upstream sources the Gemma 4
# architecture claims are settled against, and transcribe them with line numbers.
#
# Usage: upstream_source_transcript.sh [--no-color] [--out FILE]
#
# The "Resolved from upstream source" section of docs/models/gemma-4.md cites
# transformers and llama.cpp by file, commit SHA, and line range. Those SHAs
# appeared in prose and nowhere else in the tree: no script, no cached copy, no
# fetch recipe. The analysis was correct -- every citation re-checks -- but it
# read as unreproducible, so nobody re-ran it and the results never propagated
# to the tier documents. This script is the missing artifact.
#
# Runs in a contained agent sandbox: raw.githubusercontent.com is on the proxy
# allowlist. It needs no GPU, no model download, and no /mnt/ai-data.
#
# Pinned by SHA, never by branch. A branch would make the transcript
# reproducible only until upstream moved, which is the failure this replaces.

set -euo pipefail

OUT=${OUT:-/tmp/upstream-source-transcript.txt}

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
    echo "Usage: $0 [--no-color] [--out FILE]"
    exit 2
}

if [[ "${1:-}" == "--out" ]]; then
    [[ -n "${2:-}" ]] || usage
    OUT=$2
    shift 2
fi
[[ $# -gt 0 ]] && usage

# repo, commit, path -- the exact revisions tabulated in docs/models/gemma-4.md
# under "Resolved from upstream source". Changing one here without changing it
# there fails tests/models/test_gemma4_facts_contract.py.
SOURCES=(
    "huggingface/transformers ff886afacfd81e09dcf70d29d5bbf76babf672a5 src/transformers/models/gemma4/modeling_gemma4.py"
    "huggingface/transformers 9451e3ab0b21cc51638dce25c101ab5f2233e91d src/transformers/modeling_rope_utils.py"
    "huggingface/transformers 3e9d3e50e71442a3173bdf01cd45ba5833533efe src/transformers/models/gemma4/configuration_gemma4.py"
    "ggml-org/llama.cpp f4043fec0103872bf4339f6fa18d8b17824d5b6a conversion/gemma.py"
    "ggml-org/llama.cpp 07132750825a4f2d27a547cd9cdde1c6f6001885 gguf-py/gguf/constants.py"
    "ggml-org/llama.cpp 07132750825a4f2d27a547cd9cdde1c6f6001885 gguf-py/gguf/tensor_mapping.py"
)

command -v curl >/dev/null || { print_error "curl not found"; exit 1; }

: > "$OUT"
{
    echo "Upstream source transcript"
    echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Verifies: the 'Resolved from upstream source' section of docs/models/gemma-4.md"
    echo "Pinned by commit SHA; re-running reproduces this file byte for byte."
} >> "$OUT"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

failed=0
for entry in "${SOURCES[@]}"; do
    read -r repo sha path <<< "$entry"
    url="https://raw.githubusercontent.com/$repo/$sha/$path"
    dest="$tmp/$(echo "$path" | tr / _)"

    print_run "Fetching $repo@${sha:0:12} $path"
    code=$(curl -sL --max-time 60 -o "$dest" -w '%{http_code}' "$url" || echo 000)
    if [[ "$code" != "200" ]]; then
        print_error "$path: HTTP $code from $url"
        failed=$((failed + 1))
        continue
    fi

    bytes=$(wc -c < "$dest")
    lines=$(wc -l < "$dest")
    sha256=$(sha256sum "$dest" | cut -d' ' -f1)

    {
        echo
        echo "================================================================"
        echo "REPO:   $repo"
        echo "COMMIT: $sha"
        echo "PATH:   $path"
        echo "URL:    $url"
        echo "BYTES:  $bytes"
        echo "LINES:  $lines"
        echo "SHA256: $sha256"
        echo "================================================================"
        echo
        # Line numbers are the point: gemma-4.md cites this file by line, and
        # a citation is only checkable against a numbered transcript.
        nl -ba -w6 -s'  ' "$dest"
    } >> "$OUT"

    print_ok "$path: $lines lines, $bytes bytes, sha256 ${sha256:0:12}"
done

echo >> "$OUT"

print_info "Transcript written: $OUT ($(wc -l < "$OUT") lines, $(wc -c < "$OUT") bytes)"

if (( failed > 0 )); then
    print_error "$failed of ${#SOURCES[@]} sources could not be fetched"
    print_warn "A partial transcript would silently weaken every citation check"
    exit 1
fi

print_ok "All ${#SOURCES[@]} sources fetched at their pinned revisions"
