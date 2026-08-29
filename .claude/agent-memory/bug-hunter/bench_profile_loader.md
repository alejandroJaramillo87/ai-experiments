---
name: bench-profile-loader
description: What bench/workloads.py does and does not validate at load time, which claims in its own comments its consumers contradict, and which of those were filed in the bench campaign.
metadata:
  type: project
verified: 2026-08-29
stale_when: bench/workloads.py::load grows a validation block or a profile hash, or bench/runner.py stops keying the nonce on the cell index
---

`bench.workloads.load` is a bare TOML read. Only two things raise `ProfileError`: a
missing file, and a service whose resolved prefill sweep has fewer than two distinct
lengths. Everything else — missing `name`/`repeats`/`warmup`/`[prefill]`/`[decode]`,
malformed TOML, a scalar where a list is expected, a string where an int is expected,
zero/negative/float values, an empty list — either raises a raw `KeyError`/`TypeError`/
`TOMLDecodeError` past `run.py`'s `except (ProfileError, RegistryError)`, or loads
cleanly and degrades the workload silently.

**How to apply.** When auditing this module do not read a green `tests/bench` run as
evidence for any load-time property: `tests/bench/test_workloads.py` covers exactly two
refusals. 97% line coverage of the module, and the uncovered lines (118-119, 233) are the
blocked-ordering branch and the short-region repeat branch.

Established 2026-08-29, filed as BQ-H1-01..10:

- **Prompts are identical across `execute` calls.** `runner.py:109` keys the nonce on
  `(profile.seed, cell index, slot)`, all deterministic. `bench noise` calls `execute`
  twice in one process against one live server; the two prompt sequences hash equal
  (76/76 prompts reused). vLLM prefix caching is default-on
  (`docker/vllm-gpu/entrypoint.sh:45-47`). `test_runner.py:99` only checks uniqueness
  *within* one call.
- **`order` is unvalidated.** Only the exact string `"interleaved"` selects round-robin;
  `"round-robin"`, `"Interleaved"`, any typo silently selects blocked. Mutating the
  comparison to `self.order.startswith("i")` leaves 222 passed, 4 xfailed unchanged.
- **Override keys are unvalidated.** `[overrides.llama_cpu]` or `prefill_prompt_token`
  (either typo) silently reverts `publish` on llama-cpu to the 4-point sweep the
  override exists to remove.
- **Two comments in this file are false.** `:42-45` says the 4-chars-per-token estimate
  never reaches the report; `runner.py:185,207` writes it as the point identity and as
  the slope's x-axis, while `prompt_tokens_observed` (the server's count) sits unused in
  the same block. `:136-137` says a run is "reproducible from its profile hash alone"
  and `.claude/skills/benchmarking/SKILL.md:45` says `compare` refuses on a profile-hash
  mismatch; no profile hash exists in the tree. `delta.BLOCKING` keys on the profile's
  self-declared `name`, and `profile_path` accepts any path, so two different files can
  both declare `name = "publish"`.

**Not settled here, needs the host:** the real chars-per-token ratio of the cleaned
wikitext corpus under the served tokenizer (no tokenizer vocabulary is committed to this
tree), and therefore the magnitude of the `prefill_tok_s_slope` axis error; and how much
of a second `noise` arm a warm vLLM prefix cache actually serves.

**Found sound:** `corpus()` cleaning (620 header lines dropped, no `<unk>`, no `@-@`/
`@,@`/`@.@` residue), `slices()` disjointness, `points()` (prefill pinned to concurrency
1), and the interleaved branch of `cells()` (genuine round-robin, warm-ups first and
marked).
