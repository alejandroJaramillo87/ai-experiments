---
name: harness-bos-seam-coverage
description: What the rust-harness BOS-ownership seam (agent/chat/compact/generator prompt builders) is and is not covered by, as measured by mutation in the 2026-08-05 campaign.
metadata:
  type: project
---

`<bos>` is no longer a token id prepended by `generator.rs`; it is the string constant
`harness::template::BOS`, seeded by each incremental prompt builder (agent memory seed,
`chat::render_all`, `compact::compact`) and resolved back to one id by
`generator::PROMPT_SPECIALS`.

**Why:** the audit of that seam found the coverage is uneven in a way that is invisible
from test names — several tests that look like they guard BOS are inert because of
fixture shape, and one builder-level assertion has no equivalent elsewhere.

**How to apply:** when touching these files, do not read a green suite as evidence for
the BOS invariant. Measured facts (mutation, `cargo test -p harness`, 121 tests):

- `SessionGenerator::count_tokens` has **zero** test callers. Returning a constant `0`
  leaves all 121 tests green. Every compaction test drives a scripted generator that
  inherits the default `len()/4` heuristic instead.
- `runtime::synth::tiny_vocab` names its control pieces `<pad>`/`<s>`/`</s>`, **not**
  `<bos>`. Any test built on `runtime::synth` (including `generator.rs`'s own unit
  module) therefore cannot observe BOS resolving — `encode_prompt("<bos>")` returns a
  5-token byte run there. `harness/tests/generator_e2e.rs::byte_tokenizer` is the only
  fixture in the tree whose vocabulary actually carries `<bos>`.
- `template::MARKERS` (what `sanitize` neutralizes) does **not** include BOS, while
  `PROMPT_SPECIALS` does. That asymmetry is the live hazard on this seam.
- `agent_e2e.rs::ScriptedGenerator` does not override `context_budget`, so it returns
  `None` and `compact::should_compact` short-circuits — no agent-side compaction path is
  exercised by any committed test.

Related: [[sandbox-mutation-testing]]
