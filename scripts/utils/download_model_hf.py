# download_model.py
import os
from huggingface_hub import snapshot_download

# Enable the high-speed backend
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

repo_id = "Qwen/Qwen3-30B-A3B-Base"
local_dir = "/mnt/ai-data/models/hf/Qwen3-30B-A3B-Base"

print(f"Downloading model '{repo_id}' to '{local_dir}'...")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False, # Use False to avoid symlinks, good for Docker
    resume_download=True # Resumes if the download is interrupted
)

print("Download complete!")