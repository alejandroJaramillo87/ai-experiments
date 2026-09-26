"""The training compose file's cache mounts, held to what datasets reads
offline (KB-92, second fix). datasets 5.0.0 in offline mode never consults a
hub snapshot for a hub id: its one fallback is the processed arrow cache
under HF_HOME/datasets. So the training service must mount that half of the
persistent cache, read-write (the builder takes a lock file at the cache
root and hellaswag's process_docs writes map caches beside the arrow files),
and prepare must write the same tree. The first fix mounted the hub half,
read-only, which nothing read; this test is what would have caught it.

The same file's prepare service is also held to the organism corpus (KB-103).
The organism recipes put `organism.out_dir` under /mnt/ai-data/interp, the
host-run generator writes the blind corpus there, and prepare-sft reads it in
the prepare container, which on 2026-09-26 mounted the models, training, logs
and cache trees and nothing under interp: the sixth FT.organisms attempt died
with the hub loader's "Couldn't find any data file" for a path that existed
on the host. The mount is read-only, and every recipe's out_dir must sit
under a host tree the service mounts."""

import glob
import os

import yaml

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
COMPOSE = os.path.join(REPO, "docker-compose.training.yaml")
ORGANISM_RECIPES = os.path.join(REPO, "training", "recipes", "organisms", "*.yaml")

HOST_CACHE = "/mnt/ai-data/cache/huggingface"
HOST_INTERP = "/mnt/ai-data/interp"


def _volumes(service: str) -> list[str]:
    with open(COMPOSE) as f:
        compose = yaml.safe_load(f)
    return compose["services"][service]["volumes"]


def _mounted_host_roots(service: str) -> list[str]:
    """Absolute host directories the service binds, by their container path."""
    roots = []
    for volume in _volumes(service):
        source, _, rest = volume.partition(":")
        target = rest.split(":")[0]
        if os.path.isabs(source):
            roots.append(target)
    return roots


def test_prepare_reads_the_organism_corpus_tree():
    assert f"{HOST_INTERP}:{HOST_INTERP}:ro" in _volumes("prepare")
    # The recipe names the host spelling, so the container path must be it.
    roots = _mounted_host_roots("prepare")
    for path in sorted(glob.glob(ORGANISM_RECIPES)):
        with open(path) as f:
            out_dir = yaml.safe_load(f)["organism"]["out_dir"]
        assert any(out_dir.startswith(root.rstrip("/") + "/") for root in roots), (
            f"{os.path.basename(path)}: out_dir {out_dir} is under no tree the "
            "prepare service mounts, so prepare-sft cannot read its corpus (KB-103)"
        )


def test_training_mounts_the_datasets_half_read_write():
    volumes = _volumes("training")
    assert f"{HOST_CACHE}/datasets:/app/cache/datasets:rw" in volumes
    # The hub half is never read offline; mounting it is the first fix's mistake.
    assert not any(v.startswith(f"{HOST_CACHE}/hub:") for v in volumes)


def test_prepare_writes_the_same_cache_tree():
    # What prepare stages at HF_HOME is what training finds under the same path.
    assert f"{HOST_CACHE}:/app/cache:rw" in _volumes("prepare")
