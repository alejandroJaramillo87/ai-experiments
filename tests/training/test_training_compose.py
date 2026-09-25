"""The training compose file's cache mounts, held to what datasets reads
offline (KB-92, second fix). datasets 5.0.0 in offline mode never consults a
hub snapshot for a hub id: its one fallback is the processed arrow cache
under HF_HOME/datasets. So the training service must mount that half of the
persistent cache, read-write (the builder takes a lock file at the cache
root and hellaswag's process_docs writes map caches beside the arrow files),
and prepare must write the same tree. The first fix mounted the hub half,
read-only, which nothing read; this test is what would have caught it."""

import os

import yaml

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
COMPOSE = os.path.join(REPO, "docker-compose.training.yaml")

HOST_CACHE = "/mnt/ai-data/cache/huggingface"


def _volumes(service: str) -> list[str]:
    with open(COMPOSE) as f:
        compose = yaml.safe_load(f)
    return compose["services"][service]["volumes"]


def test_training_mounts_the_datasets_half_read_write():
    volumes = _volumes("training")
    assert f"{HOST_CACHE}/datasets:/app/cache/datasets:rw" in volumes
    # The hub half is never read offline; mounting it is the first fix's mistake.
    assert not any(v.startswith(f"{HOST_CACHE}/hub:") for v in volumes)


def test_prepare_writes_the_same_cache_tree():
    # What prepare stages at HF_HOME is what training finds under the same path.
    assert f"{HOST_CACHE}:/app/cache:rw" in _volumes("prepare")
