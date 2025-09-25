#!/usr/bin/env python3
"""
Hugging Face model download and verification utility.
Downloads and verifies models from Hugging Face Hub with integrity checks.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

# Status indicators
STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_ERROR = "ERROR"

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_INVALID_USAGE = 2

class ModelDownloadError(Exception):
    """Raised when model download fails."""
    pass

class ModelVerificationError(Exception):
    """Raised when model verification fails."""
    pass

# Conditional imports for verification
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


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Download and verify models from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_model_hf.py --repo-id meta-llama/Llama-2-7b-hf --output-dir /mnt/ai-data/models/llama2-7b
  python download_model_hf.py --repo-id TheBloke/Llama-2-7B-Chat-GGUF --filename llama-2-7b-chat.Q4_K_M.gguf --output-dir /mnt/ai-data/models/gguf
        """
    )

    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repository ID (e.g., meta-llama/Llama-2-7b-hf)"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Local directory for model storage"
    )

    parser.add_argument(
        "--filename",
        help="Specific GGUF file to download (downloads full repo if omitted)"
    )

    return parser

def main() -> int:
    """Main function to download and verify models.

    Returns:
        Exit code: 0 for success, 1 for failure, 2 for invalid usage.
    """
    parser = create_parser()
    args = parser.parse_args()

    print("Hugging Face model download utility")
    print()

    try:
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Enable HF transfer optimization
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        if args.filename:
            # Single GGUF file workflow
            if not LLAMA_CPP_AVAILABLE:
                print(f"Dependency check: {STATUS_ERROR} (llama-cpp-python not installed)", file=sys.stderr)
                print("  Solution: pip install llama-cpp-python", file=sys.stderr)
                return EXIT_FAILURE

            download_and_verify_gguf(args)
        else:
            # Full repository workflow
            if not TRANSFORMERS_AVAILABLE:
                print(f"Dependency check: {STATUS_ERROR} (transformers not installed)", file=sys.stderr)
                print("  Solution: pip install transformers torch", file=sys.stderr)
                return EXIT_FAILURE

            download_and_verify_repo(args)

        return EXIT_SUCCESS

    except ModelDownloadError as e:
        print(f"Download: {STATUS_ERROR} ({e})", file=sys.stderr)
        return EXIT_FAILURE
    except ModelVerificationError as e:
        print(f"Verification: {STATUS_ERROR} ({e})", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as e:
        print(f"Unexpected error: {STATUS_ERROR} ({e})", file=sys.stderr)
        return EXIT_FAILURE


def download_and_verify_gguf(args) -> None:
    """Download and verify a single GGUF file.

    Args:
        args: Parsed command line arguments with repo_id, filename, and output_dir

    Raises:
        ModelDownloadError: If download fails
        ModelVerificationError: If verification fails
    """
    print(f"Downloading GGUF file: {args.filename} from {args.repo_id}")

    try:
        file_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=str(args.output_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"Download: {STATUS_OK} (file saved to {file_path})")
    except HfHubHTTPError as e:
        raise ModelDownloadError(
            f"Failed to download {args.filename} from {args.repo_id}\n"
            f"  Error: {e}\n"
            f"  Solution: Verify repository ID and filename"
        )

    print("Verifying GGUF model integrity...")
    try:
        # Load model briefly to verify integrity
        llm = Llama(model_path=file_path, n_ctx=128, n_threads=4, verbose=False)
        del llm
        print(f"Verification: {STATUS_OK} (GGUF model is valid)")
        print(f"Model ready at: {file_path}")
    except Exception as e:
        raise ModelVerificationError(
            f"GGUF model verification failed: {file_path}\n"
            f"  Error: {e}\n"
            f"  Solution: Re-download or check file integrity"
        )


def download_and_verify_repo(args) -> None:
    """Download and verify a full Transformers repository.

    Args:
        args: Parsed command line arguments with repo_id and output_dir

    Raises:
        ModelDownloadError: If download fails
        ModelVerificationError: If verification fails
    """
    print(f"Downloading repository: {args.repo_id} to {args.output_dir}")

    try:
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(args.output_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"Download: {STATUS_OK} (repository downloaded)")
    except HfHubHTTPError as e:
        raise ModelDownloadError(
            f"Failed to download repository {args.repo_id}\n"
            f"  Error: {e}\n"
            f"  Solution: Check repository exists and access permissions"
        )

    print("Verifying model integrity...")
    try:
        # Load model on CPU to verify integrity without using VRAM
        model = AutoModelForCausalLM.from_pretrained(
            str(args.output_dir),
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        del model
        print(f"Verification: {STATUS_OK} (model loads successfully)")
        print(f"Model ready at: {args.output_dir}")
    except Exception as e:
        raise ModelVerificationError(
            f"Model verification failed: {args.output_dir}\n"
            f"  Error: {e}\n"
            f"  Solution: Re-download or check model format compatibility"
        )


if __name__ == "__main__":
    sys.exit(main())