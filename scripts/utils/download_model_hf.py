#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

# Conditional imports for verification to keep the script runnable
# even if one set of dependencies is not installed.
try:
    from transformers import AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


def main():
    """
    Main function to parse arguments and run the download and verification process
    for either a single GGUF file or a full Transformers repository.
    """
    parser = argparse.ArgumentParser(
        description="Download and verify a model from the Hugging Face Hub.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        type=str,
        help="The repository ID on Hugging Face Hub (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct')."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="The local directory where the model or file will be saved."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Optional: The specific GGUF file to download.\nIf omitted, the entire repository will be downloaded."
    )
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    if args.filename:
        # --- Single GGUF File Workflow ---
        if not LLAMA_CPP_AVAILABLE:
            print("❌ ERROR: 'llama-cpp-python' is not installed. Please install it to verify GGUF files.", file=sys.stderr)
            sys.exit(1)
            
        download_and_verify_gguf(args)
    else:
        # --- Full Transformers Repository Workflow ---
        if not TRANSFORMERS_AVAILABLE:
            print("❌ ERROR: 'transformers' and 'torch' are not installed. Please install them to verify repositories.", file=sys.stderr)
            sys.exit(1)
            
        download_and_verify_repo(args)


def download_and_verify_gguf(args):
    """Downloads and verifies a single GGUF file."""
    print(f"📥 Downloading single file: '{args.filename}' from '{args.repo_id}'...")
    try:
        file_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=args.output_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✅ Download complete!\n")
    except HfHubHTTPError as e:
        print(f"❌ ERROR: Failed to download file. Check repo/filename.", file=sys.stderr)
        print(f"   Details: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Verifying GGUF model integrity by loading with llama.cpp...")
    try:
        llm = Llama(model_path=file_path, n_ctx=128, n_threads=4, verbose=False)
        # The brief loading is the verification. If it doesn't crash, the file is valid.
        del llm
        print(f"✅ GGUF model verified successfully at: {file_path}")
    except Exception as e:
        print(f"❌ ERROR: GGUF file downloaded but appears to be corrupted or invalid.", file=sys.stderr)
        print(f"   Details: {e}", file=sys.stderr)
        sys.exit(1)


def download_and_verify_repo(args):
    """Downloads and verifies a full Transformers repository."""
    print(f"📥 Downloading full repository: '{args.repo_id}' to '{args.output_dir}'...")
    try:
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.output_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✅ Download complete!\n")
    except HfHubHTTPError as e:
        print(f"❌ ERROR: Failed to download model. The repo may be private or not exist.", file=sys.stderr)
        print(f"   Details: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Verifying model integrity by loading with Transformers...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            torch_dtype=torch.float16,
            device_map="cpu"  # Use CPU to avoid VRAM usage for a simple check
        )
        del model  # Free up memory after the check
        print("✅ Model verified successfully!")
    except Exception as e:
        print(f"❌ ERROR: Model files downloaded but seem corrupted or invalid.", file=sys.stderr)
        print(f"   Failed to load model from '{args.output_dir}'.", file=sys.stderr)
        print(f"   Details: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()