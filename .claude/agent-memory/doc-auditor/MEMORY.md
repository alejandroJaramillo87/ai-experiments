# Doc-auditor memory index

- [Storage tree sources of truth](reference_storage_tree_sources.md) — /mnt/ai-data dirs come from DIRS, storage_check loops, Makefile interp-dirs, plus run-time creators; check all of them before reporting a missing directory as drift.

Memory format: each file carries `verified:` (when its claims were last checked against the
tree) and `stale_when:` (the observable event that invalidates it). Grade a memory against
those before acting on it — a memory asserting something is broken outlives the fix, and
acting on a stale one suppresses a real finding.

Removed 2026-08-06: the dual-SSD identity conflict (resolved on the host and now fully
recorded in `docs/os/storage-configuration.md`, including the `blkid -L AIENGINEERING` rule
and the unverified-throughput caveats) and the Gemma 4 E2B config contamination (the snapshot
was re-downloaded; `gemma4-safetensors.txt` now shows genuine E2B geometry, so the
treat-as-unverified instruction had become actively wrong).
