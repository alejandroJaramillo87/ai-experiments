---
name: doc-audit-storage-tree-sources
description: The /mnt/ai-data directory tree has three independent creators/enumerators, not two — audit all three
metadata:
  type: reference
verified: 2026-08-06
stale_when: setup_data_ssd.sh's DIRS array becomes the sole creator, or the interp-dirs Makefile target is removed
---

Claims about the `/mnt/ai-data` tree must be cross-checked against **three** places, not the
two that are obvious:

1. `scripts/setup/setup_data_ssd.sh` — the `DIRS` array (provisioning).
2. `scripts/utils/storage_check.sh` — the per-tier `for sub in ...` loops (reporting).
3. `Makefile` target `interp-dirs` — creates parts of the interp tier that `DIRS` never
   creates (for example `/mnt/ai-data/interp/neuronpedia` and its `postgres/` subdirectory).
   These are paths on the data drive, not in the repository.

Directories can also appear at run time: the Rust harness creates
`${OBS_LOG_DIR}/harness/` itself (`rust-harness/harness/src/obs.rs`), and the Python obs
core creates `${OBS_LOG_DIR}/<cli>/` per CLI (`common/obs/catalog.py`, CLI names come from
`dispatch("training", ...)` / `dispatch("interp", ...)`). A directory missing from `DIRS` is
therefore not automatically a doc error — check for a run-time creator before reporting.
