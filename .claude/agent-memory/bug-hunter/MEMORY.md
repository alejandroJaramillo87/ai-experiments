# Bug-hunter memory index

- [Bench profile loader](bench_profile_loader.md) — what bench/workloads.py validates at load time (almost nothing), the two false comments in it, and the cross-run prompt-identity defect.
- [Harness BOS seam coverage](harness_bos_seam_coverage.md) — measured coverage of the agent/chat/compact/generator BOS-ownership seam; two gaps closed 2026-08-06, two still open.
- [bench registry and CLI surface](bench_registry_cli_surface.md) — what `bench/registry.py` validates and what it does not, why the model block in a `bench_result` is an assertion rather than an observation, and the CLI probes that are dead ends.
- [bench command layer state](bench_command_layer_state.md) — what `bench run`, `bench noise`
  and `bench compare` leave behind and fail to record: the arm environment never restored, the
  floor document that drops its errors list, and the comparison that can report success having
  graded nothing.

Memory format: each file carries `verified:` (when its claims were last checked against the
tree) and `stale_when:` (the observable event that invalidates it). Grade a memory against
those before acting on it — a memory asserting something is broken outlives the fix.
