---
name: harness-bos-seam-coverage
description: What the rust-harness BOS-ownership seam (agent/chat/compact/generator prompt builders) is and is not covered by; two of the four original gaps have since been closed.
metadata:
  type: project
verified: 2026-08-06
stale_when: runtime/src/synth.rs gains a <bos> piece in tiny_vocab, or agent_e2e.rs's ScriptedGenerator overrides context_budget
---

`<bos>` is no longer a token id prepended by `generator.rs`; it is the string constant
`harness::template::BOS`, seeded by each incremental prompt builder (agent memory seed,
`chat::render_all`, `compact::compact`) and resolved back to one id by
`generator::PROMPT_SPECIALS`.

**Why:** the audit of that seam found the coverage is uneven in a way that is invisible
from test names — several tests that look like they guard BOS are inert because of
fixture shape.

**How to apply:** when touching these files, do not read a green suite as evidence for
the BOS invariant. Re-measured 2026-08-06 against the tree:

- **Still open.** `runtime::synth::tiny_vocab` names its control pieces `<pad>`/`<s>`/`</s>`,
  **not** `<bos>` (`synth.rs:79`). Any test built on `runtime::synth` — including
  `generator.rs`'s own unit module — therefore cannot observe BOS resolving;
  `encode_prompt("<bos>")` returns a 5-token byte run there.
  `harness/tests/generator_e2e.rs::byte_tokenizer` is the only fixture in the tree whose
  vocabulary actually carries `<bos>`.
- **Still open.** `agent_e2e.rs::ScriptedGenerator` (`agent_e2e.rs:30-38`) implements only
  `complete`; it does not override `context_budget`, so that returns `None` and
  `compact::should_compact` short-circuits. No agent-side compaction path is exercised by
  any committed test.
- **Closed 2026-08-06.** `SessionGenerator::count_tokens` now has a direct test:
  `generator_e2e.rs:96` `count_tokens_counts_what_complete_will_encode`, asserting at `:138`.
  The earlier note that a constant `0` left all 121 tests green no longer holds.
- **Closed 2026-08-06.** `template::MARKERS` now includes BOS (`template.rs:50-51`, with the
  rationale at `:45-47`), so `sanitize` neutralizes it and the asymmetry against
  `PROMPT_SPECIALS` is gone. This was the live hazard on the seam; it is resolved.

Related: [[gemma4-synth-fixture-root]] — the `<pad>`/`<s>`/`</s>` naming is one leg of that
larger fixture problem, and the two should be fixed together rather than patched separately.
