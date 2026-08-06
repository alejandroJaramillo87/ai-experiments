# Bug-hunter memory index

- [Harness BOS seam coverage](harness_bos_seam_coverage.md) — measured coverage of the agent/chat/compact/generator BOS-ownership seam; two gaps closed 2026-08-06, two still open.

Memory format: each file carries `verified:` (when its claims were last checked against the
tree) and `stale_when:` (the observable event that invalidates it). Grade a memory against
those before acting on it — a memory asserting something is broken outlives the fix.
