#!/usr/bin/env python3
import os
import sys
import argparse
import torch
# Corrected imports
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from transformers import AutoModelForCausalLM

def main():
    """
    Main function to parse arguments and run the download and verification process.
    """
    parser = argparse.ArgumentParser(
        description="Download and verify a model from the Hugging Face Hub.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        type=str,
        help="The repository ID on Hugging Face Hub (e.g., 'meta-llama/Meta-Llama-3-8B')."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="The local directory where the model will be saved."
    )
    args = parser.parse_args()

    # --- Step 1: Download Model ---
    print(f"📥 Downloading model '{args.repo_id}' to '{args.output_dir}'...")
    try:
        # Enable the high-speed backend for downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.output_dir,
            local_dir_use_symlinks=False, # Important for Docker/portability
            resume_download=True
        )
        print("✅ Download complete!\n")

    except HfHubHTTPError as e:
        print(f"❌ ERROR: Failed to download model. The repo '{args.repo_id}' may be private or not exist.", file=sys.stderr)
        print(f"   Details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred during download: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Step 2: Verify Model ---
    print(f"🔍 Verifying model integrity by loading it onto the CPU...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            torch_dtype=torch.float16,
            device_map="cpu" # Use CPU to avoid VRAM usage for a simple check
        )
        del model # Free up memory after the check
        print("✅ Model verified successfully!")

    except Exception as e:
        print(f"❌ ERROR: Model files were downloaded but seem corrupted or invalid.", file=sys.stderr)
        print(f"   Failed to load model from '{args.output_dir}'.", file=sys.stderr)
        print(f"   Details: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()