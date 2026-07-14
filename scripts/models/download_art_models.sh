#!/bin/bash
#
# download_art_models.sh — download the image and video generation model set
# into the ComfyUI model layout under /mnt/ai-data/models.
#
# Usage: download_art_models.sh [--no-color] <image|video|all>
#
# Model set (July 2026):
#   image: FLUX.2 klein 4B (fast, Apache 2.0), FLUX.2 klein base 9B FP8 (quality)
#   video: LTX-2.3 FP8 (video+audio), Wan 2.2 T2V/I2V 14B FP8 (photorealism)
#
# Requires the huggingface_hub CLI (`hf`). FLUX.2 repositories require an
# accepted license on huggingface.co and `hf auth login`.

set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/mnt/ai-data/models}

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
    echo "Usage: $0 [--no-color] <image|video|all>"
    exit 2
}

# download <hf-repo> <dest-subdir> [include-pattern]
# Downloads a repository (optionally filtered) into $MODEL_ROOT/<dest-subdir>.
download() {
    local repo="$1" dest="$2" pattern="${3:-}"
    local target="$MODEL_ROOT/$dest"
    print_run "Downloading: $repo -> $target"
    mkdir -p "$target"
    if [[ -n "$pattern" ]]; then
        hf download "$repo" --include "$pattern" --local-dir "$target"
    else
        hf download "$repo" --local-dir "$target"
    fi
    print_ok "$repo"
}

download_image_models() {
    echo "=== Image generation models ==="
    echo
    # Fast iteration model: 4-step generation, ~13GB VRAM, Apache 2.0
    download "black-forest-labs/FLUX.2-klein-4B" \
        "diffusion_models/FLUX.2-klein-4B"
    # Quality model: 9B base in FP8, non-commercial license
    download "black-forest-labs/FLUX.2-klein-base-9b-fp8" \
        "diffusion_models/FLUX.2-klein-base-9b-fp8"
    echo
}

download_video_models() {
    echo "=== Video generation models ==="
    echo
    # Synchronized video+audio generation; FP8 fits 32GB VRAM
    download "Lightricks/LTX-2.3-fp8" \
        "diffusion_models/LTX-2.3-fp8"
    # Photorealistic video: ComfyUI-repackaged FP8 diffusion models,
    # text encoder, and VAE (T2V and I2V variants)
    download "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
        "wan2.2" "split_files/**fp8*"
    download "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
        "wan2.2" "split_files/vae/*"
    echo
    print_info "Wan 2.2 files land under $MODEL_ROOT/wan2.2/split_files/"
    print_info "Link them into the ComfyUI layout:"
    echo "  ln -s $MODEL_ROOT/wan2.2/split_files/diffusion_models/* $MODEL_ROOT/diffusion_models/"
    echo "  ln -s $MODEL_ROOT/wan2.2/split_files/text_encoders/*   $MODEL_ROOT/text_encoders/"
    echo "  ln -s $MODEL_ROOT/wan2.2/split_files/vae/*             $MODEL_ROOT/vae/"
    echo
}

main() {
    local mode="${1:-}"
    [[ -z "$mode" ]] && usage

    if ! command -v hf >/dev/null 2>&1; then
        print_error "huggingface_hub CLI not found: hf"
        echo "  Solution: Install the Hugging Face CLI"
        echo "  Command: pip install -U huggingface_hub"
        exit 1
    fi

    mkdir -p "$MODEL_ROOT"/{diffusion_models,text_encoders,vae,loras}

    case "$mode" in
        image) download_image_models ;;
        video) download_video_models ;;
        all)   download_image_models; download_video_models ;;
        *)     usage ;;
    esac

    print_ok "Download complete"
    echo "Restart required: NO (ComfyUI picks up models on next workflow run;"
    echo "use the web UI 'refresh' if a loader does not list a new file)"
}

main "$@"
