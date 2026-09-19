"""Resolve a recipe's base model to the directory train-download staged (KB-79).

Every HF recipe pins its base as a hub id plus revision, and
tests/training/test_recipe_pins.py checks that pin against the facts file. The
training container cannot resolve a hub id: its HF cache is an empty tmpfs and
its network is internal. What it has is the directory `python -m training
download` writes, <models root>/hf/<basename of the id>, mounted at the
container spelling in the training service, at its host spelling in the
prepare service, and read at its host spelling by the hardware tests. One
base, three spellings, and a recipe can write only one, so the recipe keeps
the id and each reader asks here for the directory its environment holds.
"""

from __future__ import annotations

import os

from common.serving import CONTAINER_MODELS, HOST_MODELS

# train-download's layout (training/commands/download.py DEFAULT_ROOT), as each
# environment mounts it: the container spelling first, since a container that
# also has the host tree bound at its own path (prepare) holds the same files.
STAGED_ROOTS: tuple[str, ...] = (
    os.path.join(CONTAINER_MODELS, "hf"),
    os.path.join(HOST_MODELS, "hf"),
)


def resolve_model_source(model_id: str) -> str:
    """The path a loader should read for model_id.

    A directory is returned as given. A hub id whose basename is staged under
    one of STAGED_ROOTS returns that directory. Anything else is returned
    unchanged, for a reader that holds an HF cache (the host tests, export
    in a container with the cache mounted); an offline reader then fails at
    the loader with local_files_only, which is the error preflight reports
    as `model.hf:` with the train-download remedy.
    """
    if os.path.isdir(model_id):
        return model_id
    basename = os.path.basename(model_id.rstrip("/"))
    if not basename:
        return model_id
    for root in STAGED_ROOTS:
        candidate = os.path.join(root, basename)
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return model_id
