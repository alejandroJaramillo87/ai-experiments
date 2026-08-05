# Gemma 4 conformance audit

Reconciliation of the `rust-harness`, `training`, and `interp` tiers against
[docs/models/gemma-4.md](../models/gemma-4.md), the Gemma 4 architecture reference.

The three tiers were built before that reference existed. Architecture details came from
secondhand web sources or from inference, and the architecture reconciliation table at
`/work/docs/inference/rust-harness-runtime-design.md:295` records that guesswork with
explicit confidence levels. The reference is now the source of truth, traced to pinned
Hugging Face revisions and to safetensors headers. This document records where the
implementation and the reference disagree.

This campaign is audit-only. No code or documentation outside this file changed.

## Table of contents

- [Method](#method)
- [Limitations](#limitations)
- [Summary](#summary)
- [Confirmed findings (verified)](#confirmed-findings-verified)
- [Confirmed findings (unverified)](#confirmed-findings-unverified)
- [Findings recorded without verification](#findings-recorded-without-verification)
- [Refuted findings](#refuted-findings)
- [Conformant points checked](#conformant-points-checked)
- [Host verification runbook](#host-verification-runbook)

## Method

The reference was decomposed into 203 atomic, checkable claims.
34 independent auditors then swept the repository in four blind modalities:

- **Target-oriented** - one file set each, holding the whole claim list.
- **Claim-oriented** - one claim cluster each, sweeping every tier.
- **Literal-oriented** - numeric constants, metadata keys, tensor and token names, and
  model identifiers, hunted as strings rather than as concepts.
- **Absence-oriented** - what has no representation anywhere, and whether each existing
  guard can actually fire against the architecture the reference describes.

Findings were deduplicated per file and then across files, with each specification quote
and implementation quote reopened and confirmed at its cited line. Findings graded
critical or high were put to three independent adversarial verifiers, each with a
separate lens: whether the reference really says it, whether the code really does it, and
whether the failure is reachable. Medium findings received one verifier. Low findings are
recorded without verification, by policy. A completeness critic ran 3 rounds,
each spawning targeted sweeps against the coverage holes it found.

All 203 claims were reached by at least one sweep.

Verdict values carry these meanings:

| Verdict | Meaning |
|---------|---------|
| `mismatch` | The implementation encodes a different value or rule than the reference. |
| `unrepresentable` | The data model structurally cannot express the reference. |
| `absent-undeclared` | No handling exists, and nothing acknowledges the gap. |
| `absent-declared` | No handling exists, but the gap is named honestly. |
| `guard-ineffective` | A refusal or validation exists but cannot fire, so an intended loud failure becomes silent. |
| `stale-doc` | Repository prose contradicts the reference. |
| `needs-host` | Only a real artifact on the workstation can settle it. |

## Limitations

Four limitations bound every conclusion below.

1. **No Gemma 4 weights exist on this workstation.** Nothing depending on a real GGUF
   file could be settled. Those 50 questions appear in the host verification
   runbook as PENDING entries rather than as resolved facts.
2. **Only 59 of the 248 confirmed findings were adversarially verified.**
   The remaining 189 arrived from completeness-critic gap-fill rounds that
   bypassed both deduplication and verification. They are real findings from real reads,
   but they carry zero recorded verdicts and are segregated into their own section rather
   than counted as verified. Expect duplicates among them.
3. **18 findings were never regraded or quote-checked.** One
   triage worker did not return. Its findings are included and marked `ungraded`.
4. **The reference itself records seven open questions.** Where it is uncertain, this
   audit does not resolve the uncertainty.

## Summary

| Quantity | Count |
|----------|------:|
| Claims decomposed from the reference | 203 |
| Claims reached by at least one sweep | 203 |
| Raw findings before deduplication | 716 |
| Findings after deduplication | 257 |
| Confirmed, adversarially verified | 59 |
| Confirmed, unverified (gap-fill) | 189 |
| Recorded without verification | 83 |
| Refuted | 11 |
| Conformant points checked | 104 |
| Blocked on host artifacts | 50 |

Confirmed findings by severity:

| Severity | Count |
|----------|------:|
| critical | 5 |
| high | 35 |
| medium | 117 |
| low | 91 |

Confirmed findings by tier:

| Tier | Count |
|------|------:|
| `docs` | 94 |
| `rust-harness` | 61 |
| `training` | 38 |
| `tests` | 25 |
| `interp` | 15 |
| `other` | 11 |
| `scripts` | 3 |
| `schemas` | 1 |

Confirmed findings by verdict:

| Verdict | Count |
|---------|------:|
| `stale-doc` | 68 |
| `mismatch` | 46 |
| `absent-undeclared` | 41 |
| `needs-host` | 37 |
| `guard-ineffective` | 22 |
| `absent-declared` | 17 |
| `unrepresentable` | 17 |

## Confirmed findings (verified)

The 59 findings below survived adversarial verification. Each is ordered by
severity, then tier, then file. These are the findings a reader may act on directly.

### GC-002 template.rs hardcodes Gemma 3 <start_of_turn>/<end_of_turn>; Gemma 4's turn markers are <\|turn>/<turn\|>

| Field | Value |
|-------|-------|
| Severity | critical |
| Verdict | `mismatch` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/harness/src/template.rs:15` |
| Variants | 12B, 31B, E2B, E4B, all |
| Claims | C1-17, C10-02, C10-12, C10-13, C8-01, C8-03, C8-05, C8-07, C8-08, C8-10 |
| Verification | 3 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:472 "The tokenizer defines asymmetric paired markers where the bar denotes the open or close side:"; :476 "\| Turn \| `<\\|turn>` \| `<turn\\|>` \|"

**Implementation.** template.rs:15 `pub const TURN_OPEN: &str = "<start_of_turn>";`  template.rs:16 `pub const TURN_CLOSE: &str = "<end_of_turn>";`  template.rs:18 `pub const MODEL_TURN_OPEN: &str = "<start_of_turn>model\n";`

**Consequence.** TURN_OPEN/TURN_CLOSE/MODEL_TURN_OPEN (template.rs:15-18) carry the Gemma 3 literals, and every rendered prompt goes through them (template.rs:40, :49; agent.rs:145; chat.rs:95). The spec's marker table (gemma-4.md:472-480) names <\|turn>/<turn\|> and never lists the Gemma 3 pair, so on a Gemma 4 vocabulary special_at (tokenizer.rs:322-338) returns None, encode_inner falls through to encode_span, and each turn boundary becomes a byte run - the tokenizer's own test measures that exact case at tokenizer.rs:558-565 (15 tokens for one literal). Output stays fluent and no test catches it: the only golden test, rendered_token_ids_match_hf_reference (template.rs:137-145), is #[ignore]d with an unimplemented!() body. Root: the Gemma 3 assumption declared provisional at template.rs:6-10, already corrected once on the Python side and never carried across - training/models/families.py:45-49 bumps CHAT_TEMPLATE_VERSION to 2 for exactly this, and families.py:84-85 encodes turn_prefix "<\|turn>{role}\n" / eot "<turn\|>". Fixing these three constants also fixes the dependent sites in chat.rs, generator.rs, and the two harness tests.  Propagated to: rust-harness/harness/src/chat.rs:26, rust-harness/harness/tests/agent_e2e.rs:290, docs/inference/rust-harness-runtime-design.md:191, docs/verification/rust-harness.md:566. One wrong constant family (Gemma 3 <start_of_turn>/<end_of_turn> where Gemma 4 uses <\|turn>/<turn\|>). template.rs:15 is the ROOT literal; chat.rs:26 trims the same non-existent close marker, agent_e2e.rs:290 pins it as a golden, and two documents restate it (runtime-design.md:191 claims the Gemma 4 format already renders, rust-harness.md:566 names <end_of_turn> as the stop marker). template.rs:25 (sanitizer) stays separate as a distinct same-file defect.

- upheld: Verified at the cited site: template.rs:15-18 literally defines TURN_OPEN=\"<start_of_turn>\", TURN_CLOSE=\"<end_of_turn>\", MODEL_TURN_OPEN=\"<start_of_turn>model\\n\", while docs/models/gemma-4.md:476 lists Turn open/close as <\|turn>/<turn\|> and the string start_of_turn appears nowhere in the spec. I searched specifically for the handling the lens asks about and found none: grep for \"<\|turn\" across rust-harness/**/*.rs returns zero hits; grep -rni chat_template returns only comments, so no GGUF/config load-time remap exists; and every prompt path funnels through these constants (template.rs:40,:49 -> agent.rs:145, chat.rs:95, compact.rs:78), with generator.rs:62 building PROMPT_SPECIALS from the same two literals so encode_prompt's allow-list also carries the Gemma 3 pieces. tokenizer.rs:322-338 special_at returns None when self.specials lacks the piece, falling through to encode_span, and tokenizer.rs:555-565 measures that exact fallback at 15 tokens. No test guards it: template.rs:137-145 is #[ignore]d with an unimplemented!() body, while the passing tests (template.rs:79-82, chat.rs:153, agent_e2e.rs:290) pin the wrong literals. The repo's own Python side already corrected this (families.py:45-49 CHAT_TEMPLATE_VERSION=2, :84-85 turn_prefix \"<\|turn>{role}\\n\" / eot \"<turn\|>\"), and template.rs:6-10 self-declares the Gemma 3 framing provisional pending the Gemma 4 model card. No guard, validation, or bypassing caller exists.
- upheld: Read /work/docs/models/gemma-4.md at the cited lines. Line 472 is verbatim "The tokenizer defines asymmetric paired markers where the bar denotes the open or close side:" and line 476 is verbatim "\| Turn \| `<\\\|turn>` \| `<turn\\\|>` \|" (backslashes are markdown pipe escaping; line 472 explains the bar denotes side), so the finding quotes the spec exactly. The surrounding section (441-484) is unhedged: no "assumed/unverified/provisional" language, and turn markers are absent from the Open questions list (582-605), which instead covers proportional rope, layer_scalar, the per-layer embedding path, the E4B KV boundary, 12B/31B tensor data, use_bidirectional_attention, and attention_k_eq_v. The section is not variant-scoped — one tokenizer class at line 443, vocab_size 262144 listed as an invariant across all four variants at line 112 — so the [all/E2B/E4B/12B/31B] scope matches. grep for start_of_turn/end_of_turn/chat_template over all 646 lines returns zero hits, confirming the spec never lists the Gemma 3 pair, and line 10-11 states prior Gemma 1-3 knowledge does not transfer reliably. The cited code constants exist as quoted at template.rs:15-18. Narrowing worth noting but not refuting: the spec documents marker tokens, not a chat template, so "therefore <start_of_turn> is absent from the vocabulary" and the exact form of MODEL_TURN_OPEN are inferences (the latter leaning on training/models/families.py, not the spec); the finding states these derivations openly.
- upheld: Full chain reproduced by reading: harness/src/main.rs:483 main -> run (:359) -> chat::run_chat (:400) or Agent::run_task (:426), both of which build every prompt through template::render_entry/MODEL_TURN_OPEN (agent.rs:99-112, chat.rs:88-99 -> template.rs:40,:49 -> the Gemma 3 constants at template.rs:15-18). build_generator (main.rs:162) requires a real --model GGUF with no mock fallback, and CpuModel::from_gguf gates only on arch.starts_with(\"gemma\") (cpu.rs:62), so an E2B/E4B gemma4 GGUF reaches it; docs/verification/rust-harness.md RH.11 gives the literal command harness --model /mnt/ai-data/models/gguf/<e4b>.gguf --chat. SessionGenerator::complete encodes with PROMPT_SPECIALS = [TURN_OPEN, TURN_CLOSE] (generator.rs:62,:67); special_at (tokenizer.rs:322-338) resolves a literal only if it is in `allowed` AND present in self.specials, and docs/models/gemma-4.md:472-480 names <\|turn>/<turn\|> while the file contains no occurrence of start_of_turn at all, so the lookup misses and encode_inner falls through to encode_span. No guard exists in Tokenizer::from_gguf or SessionGenerator::new, so the failure is silent; the golden test is #[ignore]d with unimplemented!() (template.rs:137-145), agent_e2e.rs:290 pins the Gemma 3 literal against a scripted generator, and generator.rs:141 asserts only listed.len() <= unlisted.len(), which passes trivially when nothing resolves. Two imprecisions that do not change the verdict: the \"15 tokens / byte run\" figure is fixture-specific (encode_span is unigram Viterbi with byte fallback, so a real 262144-entry vocabulary would segment the literal into fewer subword pieces), and if Gemma 4 retained a legacy <start_of_turn> entry it would resolve to an untrained control id - a different mechanism, not a milder consequence. Path is model-size independent, so the all/E2B/E4B scope and critical severity stand. Caveat: no cargo/rustc in the sandbox (which cargo rustc -> exit 1), so this is a read-and-trace verification, not an executed one.

### GC-005 Scalar ffn prices all 35 E2B layers at 6144; use_double_wide_mlp unread, 566,231,040 parameters unbooked

| Field | Value |
|-------|-------|
| Severity | critical |
| Verdict | `unrepresentable` |
| Tier | `training` |
| Site | `/work/training/calculator/components.py:250` |
| Variants | E2B, E4B |
| Claims | C11-07, C13-02, C13-03, C13-06, C2-04, C2-09, C2-15, C5-02, C5-05, C5-06, C5-13, C6-01, C6-02, C6-04, C6-09, C7-01, C7-02, C7-03, C7-04, C7-05, C7-06, C7-07, C7-09, C7-10, C9-03, C9-05, C9-07, C9-08 |
| Verification | 2 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:337-339 "E2B sets `use_double_wide_mlp: true`. The flag does not apply uniformly. Of E2B's 35 layers, 15 use the configured `intermediate_size` of 6144 and **20 use 12288**, exactly twice the configured value." and :350-352 "The operative rule: when `use_double_wide_mlp` is true, the final `num_kv_shared_layers` layers use `2 * intermediate_size`, and the remainder use `intermediate_size`. A loader that trusts `intermediate_size` for every layer mis-shapes 20 of E2B's 35 layers."

**Implementation.** training/calculator/components.py:250 `            ffn=config.intermediate_size,` consumed as a single scalar at :152 `        mlp = 3 * self.hidden * self.ffn` (also :165 transformer_stack_param_count, :175 linear_param_count, :409 activation inventory, :456 teacher forward). The sibling flag is guarded — :239-245 raises UnknownArchitectureError when `enable_moe_block` is set — but `use_double_wide_mlp` is never read.

**Consequence.** ArchitectureSpec holds one `ffn` (components.py:105) filled from `config.intermediate_size` (:250) and multiplied by `self.layers` at :152, :165, :175, :409 and :456, so the spec's 15/20 split (docs/models/gemma-4.md:337-339, 341-344) cannot be expressed and no field could hold it. E2B MLP mass is priced at 35*3*1536*6144 = 990,904,320 against 15*3*1536*6144 + 20*3*1536*12288 = 1,557,135,360: 566,231,040 parameters missing. Silent - `use_double_wide_mlp` and `num_kv_shared_layers` appear nowhere in the tree outside the spec, so the module's fail-loudly policy (:78-79) never engages, and the MoE guard at :238 is the only Gemma 4 refusal. Reachable through training/recipes/reference/continual-gemma4-e2b.yaml (trainable_filter: transformer_stack, zero_offload): grad_bytes short ~1.13 GB, host_ram_bytes short 566,231,040*16 = ~9.06 GB, and memfit returns fits for a configuration that may OOM. Merged findings disagree on the anchor (105 the field, 152 the consumer, 250 the read); 250 is where every other per-variant field is resolved and where handling would land. Regraded to critical: plausible wrong numbers, no crash, and calibrate() refuses zero_offload recipes so no in-tree check catches it.  Propagated to: docs/training/memory_fit.md:72. Same defect, same value: ArchitectureSpec's single scalar ffn cannot express E2B's double-wide tail, 566,231,040 parameters unbooked. components.py:250 is the code ROOT; the memory_fit.md:72 entry is the same gap reported from the docs dimension.

- upheld: Spec-lens check only. Both cited passages are verbatim exact: `sed -n '337,339p;350,352p' /work/docs/models/gemma-4.md` reproduces the finding's spec_says text character for character, including "**20 use 12288**" and the operative rule keyed on the final `num_kv_shared_layers` layers. The claim is settled, not hedged: the 15/20 split is given as a tensor-shape table at :341-344 ([6144,1536] for layers 0-14, [12288,1536] for 15-34) in a section whose tensor data is "confirmed from the safetensors headers" (:287), restated at :394, at :266-267 ("Double-wide MLP on E2B only ... applies to a subset of layers"), and at :564 Traps item 3 ("`intermediate_size` is not per-layer truth on E2B. Twenty of 35 layers are double-wide."). The Open questions section (:582-605) does not list it; the adjacent open item (:597-599) names E2B's convention as the confirmed side and only E4B's boundary as assumed. Operands used by the finding's arithmetic are the spec's own: layers 35 (:142), hidden 1536 (:143), intermediate_size 6144 (:144), num_kv_shared_layers 20 (:149), use_double_wide_mlp true (:155); 35*3*1536*6144 = 990,904,320 vs 15*3*1536*6144 + 20*3*1536*12288 = 1,557,135,360, delta 566,231,040 checks out. One over-scope, non-fatal: the [E2B/E4B] variant tag conflicts with :354-356 and :266, which state E4B has use_double_wide_mlp false and uniform 10240 widths, so a single scalar ffn is correct for E4B; the finding's substance (title, failure_mode, parameter count, cited recipe) is strictly E2B and the spec supports it.
- upheld: Reproduced the whole chain. /work/training/calculator/components.py:250 sets `ffn=config.intermediate_size` into ArchitectureSpec's single int field (:105); every consumer multiplies it uniformly by self.layers (:152/:157, :165-167, :175-176, :409, :456), and the only other `ffn=` site is the model.scratch path at :197, so no per-layer split is expressible. Tree-wide grep for `double_wide`/`num_kv_shared` hits only docs/models/gemma-4.md and .claude agent memory - zero hits in any .py or .yaml, so the flag is genuinely unread. I searched for the handling the lens asks about and found none: spec_from_hf_config's only Gemma 4 refusal is enable_moe_block (:238-243) with gemma4/gemma4_text on the supported list (:88-89); training/recipes/validate.py has no family or flag guard (_check_model :68 only checks hf-xor-scratch) and its _check_offload_budget (:201-233) is an active consumer of the undercounted host_ram_bytes; cli.py:127 cmd_memfit applies no architecture filter; calibrate.py:106-111 refuses zero_offload so the one measurement cross-check never runs; tests/training/test_calculator.py:316-342 uses a scalar-intermediate_size fixture with no use_double_wide_mlp key and tests only the MoE refusal. Reachability confirmed: training/recipes/reference/continual-gemma4-e2b.yaml sets model.hf.id google/gemma-4-E2B, trainable_filter transformer_stack, offload zero_offload, cpu_adam. Arithmetic verified: 35*3*1536*6144 = 990,904,320 vs 15*3*1536*6144 + 20*3*1536*12288 = 1,557,135,360, delta exactly 566,231,040 (8.44 GiB of uncharged offload host RAM at 16 B/param). Spec numbers match docs/models/gemma-4.md:142-155 and :337-352; docs/training/memory_fit.md:72-76 confirms the propagation by claiming E-models are supported with only enable_moe_block failing loudly. The one counterargument I could build - that a real HF config might carry a per-layer intermediate_size list, making the failure loud - is foreclosed by the designated source of truth, which states intermediate_size is the scalar 6144.

### GC-011 ROOT: p2_d5 records E2B as 30 layers at hidden 2304, read from transformers class defaults, and six sites copied it

| Field | Value |
|-------|-------|
| Severity | high |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/training/investigations/p2_d5_optimizer_offload_investigation.md:125` |
| Variants | E2B |
| Claims | C12-22, C2-02, C2-03, C2-09, C5-10, C5-11, C8-09, C9-05, C9-07 |
| Verification | 2 verdicts recorded, 0 refutations |

**Reference.** "\| `num_hidden_layers` \| 35 \| 42 \| 48 \| 60 \|" and "\| `hidden_size` \| 1536 \| 2560 \| 3840 \| 5376 \|" (/work/docs/models/gemma-4.md:142-143)

**Implementation.** "Public sources (transformers `configuration_gemma4.py`) confirm the architecture (30 layers, hidden 2304, 262144 vocab, per-layer input tables packed as `[vocab_size, num_layers * hidden_size_per_layer_input]`, tied word embeddings)" (docs/training/investigations/p2_d5_optimizer_offload_investigation.md:125-127)

**Consequence.** The bullet says "Public sources (transformers `configuration_gemma4.py`) confirm the architecture (30 layers, hidden 2304, ...)" - class defaults, not E2B's text_config, presented as confirmed while only the derived envelope beside them is hedged. Spec:142-143 gives 35 layers at hidden 1536. The packed-table formula, the 262144 vocabulary, and the tied-embedding claim in the same sentence are all correct, so only the two dimensions are wrong; but instantiating that correct formula with them sizes E2B's PLE table 30*256 = 7680 wide instead of spec:292's 8960. This is the earliest site in the tree asserting E2B geometry and it names its own bad source. Downstream copies to correct once the root is fixed: training/calculator/components.py:109, docs/interpretability/investigations/p4_d2_activation_harvesting_investigation.md (:109, :127, :138, :180), p4_d3_sae_training_investigation.md, docs/training/investigations/f3_sft_entry_point_investigation.md:77, training/recipes/interp/harvest-*.yaml, tests/training/test_calculator.py. Graded high, not critical: the doc computes nothing itself, but a downstream consumer demonstrably acted on it.  Propagated to: docs/interpretability/investigations/p4_d2_activation_harvesting_investigation.md:109, docs/interpretability/investigations/p4_d2_activation_harvesting_investigation.md:138, docs/inference/future-optimizations.md:156, training/calculator/components.py:109, tests/interp/test_families.py:53, rust-harness/fixtures/reference_logits.py:126, rust-harness/fixtures/reference_logits.py:319, training/recipes/interp/harvest-false-fact.yaml:4, docs/training/investigations/f3_sft_entry_point_investigation.md:77. One wrong constant pair (E2B = 30 layers at hidden 2304, transformers class defaults read as text_config; real E2B is 35 layers at 1536) copied into a doc set, a comment, three test fixtures, and a recipe. p2_d5:125 is the ROOT that first recorded it as confirmed; p4_d2:109 restates it and p4_d2:138 prices the disk envelope from it. Verified copies in tree: components.py:109, tests/interp/test_families.py:53, tests/interp/test_harvest.py:126, tests/training/test_calculator.py:320, harvest-false-fact.yaml:4, future-optimizations.md:156, f3:77 (30-layer half). harvest-calibration-gemma2.yaml:7 carries 2304 correctly for Gemma 2 and is not a copy.

- upheld: Lens = does /work/docs/models/gemma-4.md actually say what GC-009 claims? It does, byte-exactly. Line 142 reads "\| `num_hidden_layers` \| 35 \| 42 \| 48 \| 60 \|" and line 143 reads "\| `hidden_size` \| 1536 \| 2560 \| 3840 \| 5376 \|", with E2B as the first data column, so E2B = 35 layers at hidden 1536 exactly as quoted. The spec is stronger than the finding needs, not weaker: (1) line 138 introduces the table with "All values come from the `text_config` block of each `config.json`", which directly asserts the text_config provenance the p2_d5 bullet substituted class defaults for; (2) lines 57-59 trace E2B/E4B to safetensors headers at pinned revision d29ff6b4, making E2B one of the two best-evidenced variants (12B/31B are the config-only ones); (3) the pair is corroborated seven more times independently of 142-143 -- line 173 (E2B, 35 layers, 28 sliding + 7 full), 199-200 ([2048, 1536], [256, 1536]), 291 (embed_tokens [262144, 1536]), 299-300 ("35 x 256 = 8960 for E2B"), 315 ("35 of each on E2B"), 337/352 ("Of E2B's 35 layers"), 564 (Trap 3, "Twenty of 35 layers"). The standard over-read failure mode is absent: the Open questions section (582-605) does not list E2B geometry; it hedges the E4B shared-KV boundary (597-599), 12B/31B tensor data (600), proportional rope, layer_scalar, the per-layer data path, use_bidirectional_attention, and attention_k_eq_v -- none touching E2B num_hidden_layers or hidden_size. The only other "unverified" (line 561) is about E-series 256K context. grep for "2304" and for any 30-layer claim across all 646 lines returns nothing. The finding's derived consequence also checks out: line 292 gives embed_tokens_per_layer [262144, 8960] and 299-300 state the formula and product, so 30*256=7680 does contradict 8960; and the three items the finding concedes correct in the same sentence are confirmed at line 112 (vocab 262144) and 116/132-134 (tied embeddings, no lm_head tensor). Comparand is on-point: p2_d5:124 names "Gemma-4-E2B" explicitly, so the E2B column is the right column, and within that bullet the hedge is scoped -- "confirm the architecture (30 layers, hidden 2304, ...) but not the E2B envelope numbers" asserts the geometry affirmatively while hedging only the envelope, exactly as the finding characterizes. No over-read found; I did not modify any file.
- upheld: Read /work/docs/training/investigations/p2_d5_optimizer_offload_investigation.md:123-129 verbatim: in a bullet whose subject is "Gemma-4-E2B with the freeze policy" it states transformers configuration_gemma4.py "confirm[s] the architecture (30 layers, hidden 2304, 262144 vocab, per-layer input tables packed as [vocab_size, num_layers * hidden_size_per_layer_input], tied word embeddings) but not the E2B envelope numbers". Spec /work/docs/models/gemma-4.md:142-143 gives E2B as 35 layers at hidden_size 1536, so both dimensions are wrong, and the sentence's hedge is scoped only to the ~5.1B/~2.3B envelope, leaving the dimensions asserted as confirmed - exactly the failure mode described. The other three parenthetical items are correct (vocab spec:112, formula spec:294, tied embeddings spec:132), matching the finding's "only two dimensions wrong" concession; instantiating spec:294's formula with 30 gives 7680 vs the spec:292 tensor width of 8960. Searched for mitigation and found none: grep shows 2304 only at line 125 and no occurrence of 1536 or 35 anywhere in the file, there is no superseded/historical banner (the header instead says "Findings verified as of 2026-07"), docs/training/investigations/ has no README disclaimer, and the Model family considerations section at 219-227 restates the per-layer sizing symbolically without numbers - its "verified against the pinned revision, not hardcoded" caveat concerns parameter names for the freeze filter, not the dimensions. Did not adjudicate the ROOT/earliest-site ordering or the propagation list (git history is unavailable in this sandbox; other verifier's lens).

### GC-014 RH.1's expected signal contradicts the QA record that gguf-info exits 1 parsing a real checkpoint

| Field | Value |
|-------|-------|
| Severity | high |
| Verdict | `needs-host` |
| Tier | `docs` |
| Site | `/work/docs/verification/rust-harness.md:123` |
| Variants | E2B, E4B |
| Claims | C3-01, C3-02, C3-03, C3-06 |
| Verification | 2 verdicts recorded, 0 refutations |

**Reference.** spec 167-169: "Gemma 4 interleaves two attention types on a fixed schedule given explicitly by the `layer_types` array. Read the array. Do not compute the schedule from a stride constant: E2B uses a different period from the other three variants."

**Implementation.** docs/verification/rust-harness.md:122-126: "Expected signal: the run ends in a compatibility verdict, printed by the same `cpu::check_supported` function the loader calls ... `[OK] CPU backend compatibility: supported` ..." versus docs/verification/rust-harness-qa.md:736-740: "**A1-R07** confirmed as stated for `sliding_window_pattern` (the converter always writes it as a boolean array, which `gguf.rs` cannot parse, so `gguf-info` exits 1 on a real checkpoint instead of printing its verdict" and :463: "and the gemma4 encoding fails earlier still in parsing."

**Consequence.** rust-harness.md:123-129 promises a compatibility verdict from cpu::check_supported, while rust-harness-qa.md:736-740 records A1-R07 confirmed: the converter always writes sliding_window_pattern as a boolean array that gguf.rs cannot parse, so gguf-info exits 1 before printing any verdict, and rust-harness-qa.md:463 adds that the gemma4 encoding fails earlier still in parsing. If A1-R07 holds, the campaign's highest-risk entry answers none of its four questions. Separately, a scalar sliding_window_pattern can express both schedules of spec:167-176 (5 for E2B, 6 for E4B), so the stride representation is adequate once it parses. Only a real GGUF settles which output appears.

- upheld: Spec-fidelity lens only. /work/docs/models/gemma-4.md lines 167-169 contain the finding's spec_says string verbatim ("Gemma 4 interleaves two attention types on a fixed schedule given explicitly by the `layer_types` array. Read the array. Do not compute the schedule from a stride constant: E2B uses a different period from the other three variants."). It sits under the normative "## Attention" heading (line 165), not under "## Open questions" (line 582) whose seven entries I read and which do not include the layer-type schedule; the file's only "unverified" markers (lines 333, 561) concern KV-sharing internals and 256K context, not this point. The supporting table at lines 171-176 gives E2B full-attention indices 4,9,14,19,24,29,34 "every 5th (4:1)" and E4B 5,11,17,23,29,35,41 "every 6th (5:1)", exactly matching the finding's "(5 for E2B, 6 for E4B)". The finding's [E2B/E4B] tag narrows rather than inflates the spec's four-variant statement. The apparent conflict between the finding's "the stride representation is adequate once it parses" and "Do not compute the schedule from a stride constant" is resolved by the spec's own trap 7 (lines 271-272: "A hardcoded stride works for three of four models"), which frames the hazard as hardcoding a single constant, not as periodic representation carried per-checkpoint. The spec contains no "gguf"/"GGUF" or "sliding_window_pattern" occurrences, but the finding never claims it does. I also confirmed the site text at /work/docs/verification/rust-harness.md:122-129 matches the code_says quote. Whether A1-R07 actually holds against a real checkpoint is host-side evidence outside this lens.
- upheld: Reproduced the whole chain in-repo. (1) /work/docs/verification/rust-harness.md:123-129 states verbatim that the RH.1 run "ends in a compatibility verdict, printed by the same cpu::check_supported function", offering only [OK]/[ERROR] compatibility outcomes; the entry's risk block (94-113) and the document preamble (1-45) mention no parse failure, and grep finds no "sliding_window_pattern", "exit 1", or "A1-R07" anywhere in that file (A1-R07 appears exactly once in docs/, at rust-harness-qa.md:736). (2) The QA quotes are accurate: rust-harness-qa.md:736-740 records A1-R07 confirmed ("the converter always writes it as a boolean array, which gguf.rs cannot parse, so gguf-info exits 1 on a real checkpoint instead of printing its verdict") and :463 records "the gemma4 encoding fails earlier still in parsing". (3) The code corroborates and offers no missed handling: rust-harness/runtime/src/gguf.rs:113-118 accepts array element types 5/6/8 only and bails on any other ("unsupported GGUF array element type {t}") — there is no type-7 arm, no skip-unknown-key path, no lenient parser; the bail aborts gguf::parse entirely. rust-harness/runtime/src/bin/gguf-info.rs:35-40 turns that into "[ERROR] Parse {path}" and ExitCode::from(1) before the metadata dump, tensor dump, and summarize(), so the tolerant path that does exist (summarize() "Never fails the run", printing "[ERROR] CPU backend compatibility:" at :164-166) is unreachable. (4) No fix landed: A1-R07 is absent from the dispositions table at rust-harness-qa.md:771-812 and array() still rejects type 7. The finding's side-claim also checks out: docs/models/gemma-4.md:167-176 (E2B full indices 4,9,...,34; E4B 5,11,...,41) is exactly reproduced by cpu.rs:144's (i+1).is_multiple_of(pattern) with 5 and 6. Only the host-side question of whether the actual E4B GGUF carries the bool-array key is unresolved, which the finding already flags as needs-host.

### GC-021 Eval serves with --no-jinja unconditionally, so serve-time turn framing is llama.cpp's built-in guess, not the <\|turn> framing SFT trained

| Field | Value |
|-------|-------|
| Severity | high |
| Verdict | `needs-host` |
| Tier | `other` |
| Site | `/work/docker/llama-gpu/entrypoint.sh:132` |
| Variants | E2B, E4B |
| Claims | C10-12 |
| Verification | 2 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:476 "\| Turn \| `<\\|turn>` \| `<turn\\|>` \|"

**Implementation.** docker/llama-gpu/entrypoint.sh:37 `JINJA_ENABLED=${JINJA_ENABLED:-false}` and :130-133 "# Upstream defaults jinja templating to on; pass the opt-out to preserve the historical default (disabled) when JINJA_ENABLED=false" / `CMD_ARGS+=("--no-jinja")`

**Consequence.** entrypoint.sh:37 defaults JINJA_ENABLED=false and :132-133 appends --no-jinja whenever it is not "true"; a grep of the whole tree finds JINJA_ENABLED set nowhere except the two entrypoints and the two overview docs, so false is the operative value. common/serving.py:61-62 builds chat_url as base_url + "/v1/chat/completions" and :81-88 pins llama-gpu to port 8002, so llama-server, not the repo, applies the chat template during eval. training/models/families.py:82-86 supervised turn_prefix "<\|turn>{role}\n" with eot "<turn\|>", matching spec:476. With --no-jinja the server ignores the GGUF's embedded template and uses legacy built-in selection; if that resolves to the Gemma 1-3 built-in (<start_of_turn>/<end_of_turn>) or to chatml, every eval prompt is framed differently from training and the scorecard silently reports train/serve skew as model quality. The chain is followable end to end inside the repo; only the last link - what the pinned llama.cpp template table picks, and what the converter writes into tokenizer.chat_template - needs a real GGUF and a running service. High, not critical: the silent-wrong-scorecard outcome is contingent on that one unresolved link. Distinct from KB-24 (thinking_mode_for), which concerns manifest architecture detection, not serve-time templating.

- upheld: Opened /work/docs/models/gemma-4.md:472-480: the '## Tokenizer and special tokens' section states flatly "The tokenizer defines asymmetric paired markers where the bar denotes the open or close side:" and line 476 is verbatim `\| Turn \| `<\\|turn>` \| `<turn\\|>` \|` in a `\| Role \| Open \| Close \|` table. The quote and line number are exact (backslashes are markdown pipe escapes, so the tokens are <\|turn> / <turn\|>, as the finding uses them). It is not hedged and not an Open Question: the Open questions section (:582-606) lists proportional rope, layer_scalar, per-layer embedding path, E4B shared-KV boundary, 12B/31B tensor data, use_bidirectional_attention, attention_k_eq_v — none touching tokenizer markers — and Traps (:553-581) says nothing qualifying them. The finding's [E2B/E4B] tag is narrower than the spec's unqualified tokenizer section, so no family-level over-read. The spec never mentions chat templates or jinja (grep for chat_template/jinja/start_of_turn/chatml returns nothing), but the finding does not attribute template rendering to the spec: it takes turn_prefix="<\|turn>{role}\n" / eot="<turn\|>" from /work/training/models/families.py:84-85 and only asserts those match spec:476, which they do. The unresolved llama.cpp-side link is already flagged [needs-host] and is the stated reason for high rather than critical.
- upheld: Reproduced the cited site exactly: /work/docker/llama-gpu/entrypoint.sh:37 sets JINJA_ENABLED=${JINJA_ENABLED:-false} and :132-133 appends "--no-jinja" whenever it is not "true", with CMD_ARGS exec'd at :180. Searched for handling elsewhere and found none: grep -rn JINJA_ENABLED /work returns only the two entrypoints and two overview docs (no compose, no env*.example, no Makefile, no Python), and /work/docker-compose.yaml:58-69 does not even list JINJA_ENABLED in the llama-gpu environment block, so a host-side export cannot reach the container - the opt-out is effectively unconditional, stronger than the finding states. No --chat-template/--chat-template-file override exists. The eval chain holds: /work/common/serving.py:61-62 builds chat_url as base_url+"/v1/chat/completions", :81-88 pins llama-gpu to 8002, and callers /work/training/commands/serve_eval.py:97 and /work/training/eval/generate.py:119 send plain [{"role":"user","content":...}] with no <\|turn> markers; eval_ft.py:277,303 only calls swap_model, which mutates the model env var alone; serve_eval.preflight() checks training idleness and the GGUF hash only, never the served template. Training-side framing matches spec: /work/training/models/families.py:82-88 turn_prefix "<\|turn>{role}\n", eot "<turn\|>". known-bugs.md has no jinja entry. The only open link is the one the finding already marks [needs-host] - what pinned llama.cpp legacy template selection resolves to for a Gemma 4 GGUF - which is outside this lens.

### GC-035 One head_dim scalar prices the 7 full-attention layers at 256; global_head_dim 512 is read nowhere in the tree

| Field | Value |
|-------|-------|
| Severity | high |
| Verdict | `unrepresentable` |
| Tier | `training` |
| Site | `/work/training/calculator/components.py:126` |
| Variants | 12B, 31B, E2B, E4B |
| Claims | C1-01, C1-02, C10-02, C10-12, C10-13, C13-06, C3-01, C3-02, C3-03, C3-06, C3-07, C3-08, C3-09, C3-10, C3-11, C3-13, C3-14, C3-15, C3-16, C3-18, C5-14, C8-07, C8-08, C9-05, C9-06, C9-07, C9-09, C9-10 |
| Verification | 2 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:180-182 "The two attention types use **different head dimensions**. Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512). This is visible directly in the weight shapes and is not an inference from the configuration." and :199 "\| `q_proj.weight` \| `[2048, 1536]` \| `[4096, 1536]` \|", :111 "\| `global_head_dim` \| 512 \|"

**Implementation.** components.py:125-130 `    @property\n    def q_dim(self) -> int:\n        return self.heads * self.head_dim\n\n    @property\n    def kv_dim(self) -> int:\n        return self.kv_heads * self.head_dim`, fed by :253 `            explicit_head_dim=getattr(config, "head_dim", None),`. `global_head_dim` and `layer_types` appear nowhere in training/calculator/.

**Consequence.** `q_dim`/`kv_dim` (:125-130) apply the single `explicit_head_dim` read from `head_dim` at :253 to every layer, and `global_head_dim`/`layer_types` appear nowhere outside docs/models/gemma-4.md. Against spec:180-182 and the weight shapes at :188-191 (E4B) and :199-202 (E2B), the 7 full-attention layers are exactly twice as wide in q/o and k/v, so E2B is short 7*7,077,888 = 49,545,216 parameters and E4B 7*13,107,200 = 91,750,400. The same wrong widths flow into linear_param_count, transformer_stack_param_count, _TARGET_MODULE_DIMS (adapter sizing for q/k/v/o), activation_phase_bytes and teacher_forward_bytes. Held at high rather than critical: same silent mechanism as the ffn gap but roughly an order of magnitude smaller (0.79 GB E2B / 1.47 GB E4B at 16 B/param) and the causal chain is arithmetic on quoted shapes, with no measurement. 31B is additionally affected because num_global_key_value_heads is unread (tracked separately).  Propagated to: docs/training/memory_fit.md:72. Same defect: ArchitectureSpec's single head_dim prices full-attention layers at 256 with global_head_dim 512 read nowhere. components.py:126 is the code ROOT; memory_fit.md:72 reports the same scalar gap with the E4B figure (91,750,400). Deliberately NOT merged with the rust-side head_dim findings - those are independent data models needing their own fixes.

- upheld: Spec-fidelity lens only. All three citations verify verbatim in /work/docs/models/gemma-4.md: :180-182 states "The two attention types use different head dimensions. Sliding layers use head_dim (256); full-attention layers use global_head_dim (512). This is visible directly in the weight shapes and is not an inference from the configuration."; :199 is the E2B row `\| q_proj.weight \| [2048, 1536] \| [4096, 1536] \|` under the header "\| Tensor \| Sliding (28 layers) \| Full (7 layers) \|" at :197; :111 is `\| global_head_dim \| 512 \|` inside "Invariants across all four variants" (:103-106 "identical in E2B, E4B, 12B, and 31B"), which supports the four-variant scope tag. The point is not hedged: it is reinforced by the E4B/E2B safetensors-header shape tables (:184-204), the explicit rule at :206-207 ("num_attention_heads * head_dim for sliding layers and num_attention_heads * global_head_dim for full layers"), Traps item 6 at :568-569 ("head_dim is not the head dimension of every layer. Full-attention layers use global_head_dim"), and :233-236 which reuses global_head_dim 512 as settled. It does NOT appear in the "Open questions" section (:582-605), whose items are proportional rope, layer_scalar, the PLE data path, the E4B shared-KV boundary, tensor data for 12B/31B, use_bidirectional_attention, and attention_k_eq_v semantics. The only adjacent caveat (:600 "Tensor-level data for 12B and 31B. Not collected") does not bite, because the finding's numeric claims are confined to E2B/E4B where weight shapes are confirmed. Recomputing from the spec's own shapes: E2B 2*(2048*1536) + 2*(256*1536) = 7,077,888 per full layer, *7 = 49,545,216; E4B 2*(2048*2560) + 2*(512*2560) = 13,107,200, *7 = 91,750,400 — both match the finding exactly, and it conservatively omits the q_norm/k_norm 256->512 deltas. Two cosmetic imprecisions, neither an over-read: the title's "7 full-attention layers" is the E2B/E4B count (spec:171-176 gives 8 for 12B, 10 for 31B), and the code_says block is labelled :125-130 when the leading @property is at :124 (the cited site :126 is exactly `return self.heads * self.head_dim`). Code citations spot-checked for accuracy: components.py:126 and :130 as quoted, :253 explicit_head_dim=getattr(config, "head_dim", None), and grep for global_head_dim/layer_types under /work/training/calculator/ returns nothing.
- upheld: Reproduced at the cited site: /work/training/calculator/components.py:118-130 derives one head_dim from the single explicit_head_dim field (:111), fed only by getattr(config, "head_dim", None) at :253, and q_dim/kv_dim (:126, :130) apply it uniformly to all layers; that width flows into param_count (:151), transformer_stack_param_count (:164), linear_param_count (:174), _TARGET_MODULE_DIMS q/k/v/o (:56-59), and the activation/teacher terms at :408 and :456. grep across /work shows global_head_dim, layer_types, and num_global_key_value_heads appear only in docs/models/gemma-4.md and an agent-memory file - nowhere in training/. Searched for missed handling and found none: the only fail-loud guards in spec_from_hf_config are unknown model_type (:226) and enable_moe_block (:238), while "gemma4"/"gemma4_text" are explicitly supported (:88-89) and :232-234 deliberately descends into text_config; real recipes (sft-qlora-gemma-e4b.yaml model.hf.id google/gemma-4-E4B, continual-gemma4-e2b.yaml, distill/organism recipes) route E2B/E4B through this path; the only downstream check, cli.py:186-193, merely warns on parameter-count drift and requires a GPU calibration run; and tests/training/test_calculator.py:344-349 pins the uniform behavior (q_dim == 8*256, kv_dim == 4*256). Arithmetic re-derived from the spec's quoted shapes: E2B 14,155,776-7,077,888 = 7,077,888 per full layer x7 = 49,545,216; E4B 26,214,400-13,107,200 = 13,107,200 x7 = 91,750,400 - both match the finding exactly.

### GC-040 transformer_stack has no modality-tower pattern, so E-series vision and audio parameters enter the optimizer

| Field | Value |
|-------|-------|
| Severity | high |
| Verdict | `absent-undeclared` |
| Tier | `training` |
| Site | `/work/training/runtime/offload.py:48` |
| Variants | E2B, E4B |
| Claims | C12-01, C12-02, C12-07, C12-12, C12-13, C12-23 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** spec 276-278: "9. **Full audio tower on the E-series.** E2B and E4B carry a complete `gemma4_audio`\n   configuration and audio tower weights. 31B has `audio_config: null`. Audio support is\n   per-variant, not a family property."  and spec 533-534: "\| `hidden_size` \| 768 \| 1152 \|" / "\| `num_hidden_layers` \| 16 \| 27 \|"

**Implementation.** training/runtime/offload.py:48-51: "FROZEN_PATTERNS = {\n    \"all\": (),\n    \"transformer_stack\": (\"embed_tokens\", \"lm_head\", \"per_layer\"),\n}"

**Consequence.** continual-gemma4-e2b.yaml:27,32 sets google/gemma-4-E2B with trainable_filter: transformer_stack; training/models/model.py:102 loads it through AutoModelForCausalLM.from_pretrained. spec:85 declares the E-series architecture Gemma4ForConditionalGeneration, so the module tree carries the 12-layer x 1024 audio tower (spec:509-511), the 16-layer x 768 vision tower (spec:533-534) and the two bridging projections model.embed_audio/embed_vision.embedding_projection (spec:407-409). None of those names contains "embed_tokens", "lm_head" or "per_layer", so is_frozen_name returns False and apply_trainable_filter (offload.py:69-84) hands them to deepspeed.initialize as trainable, against a recipe header and schema.py:91-95 that both say only the decoder stack trains. transformer_stack_param_count (components.py:160-167) prices layers*(attn+mlp+norms)+hidden only, so tower gradients and host-RAM Adam state are missing from the D4 prediction that gates the run. docs/verification/training-phase2.md:166-180 checks only that embeddings/LM head/per-layer tables stay out of the optimizer groups and is Status: PENDING, so nothing has ever observed the tower parameters. Held at high, not critical: the primary observable is a memory misprediction that surfaces as OOM, and the silent-tower-drift half depends on which class AutoModelForCausalLM resolves, which only a host check settles. SEAM: components.py is outside this chunk's file set.

- upheld: Every spec citation checks out verbatim in /work/docs/models/gemma-4.md. Lines 276-278 read exactly "9. **Full audio tower on the E-series.** E2B and E4B carry a complete `gemma4_audio` configuration and audio tower weights. 31B has `audio_config: null`. Audio support is per-variant, not a family property." — a settled structural claim (restated as trap 14 at line 580), asserting weights and not merely config. Lines 533-534 are the E-series column of the Vision tower table (`hidden_size` 768, `num_hidden_layers` 16), used correctly as tower values, avoiding the confusion the spec warns about at 549-551. Supporting cites verify: line 85 gives E-series architectures `Gemma4ForConditionalGeneration`; 509-511 give audio hidden_size 1024 / 12 layers; 407-409 attest `model.embed_audio.embedding_projection.weight` [2560,1536] and `model.embed_vision.embedding_projection.weight` [2560,768] from the E4B safetensors header (spec:57). No over-read found: the claim is scoped to E2B/E4B exactly as the spec scopes it, none of it appears in the Open questions block (582-605), and the "scoped out of text-only work" language at 14-16 and 502 scopes the document's focus, not the checkpoint's contents. The recipe's google/gemma-4-E2B is the same repo pinned at spec:46. The only inference beyond literal text is "architecture string implies the module tree carries the towers", which the finding itself hedges as host-check-dependent and which the model.language_model.* prefixing plus sibling embed_audio/embed_vision tensors support.

### GC-046 Observability doc specifies RMSNorm with (1 + weight) scaling; the code multiplies by the stored weight, as its sibling design doc explains

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-observability.md:131` |
| Variants | 12B, 31B, E2B, E4B, all |
| Claims | C1-05, C8-03, C8-11 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:114: "\| `rms_norm_eps` \| 1e-06 \|" (invariant across all four variants). The spec is derived from Hugging Face artifacts (spec 5-6) and does not settle the GGUF converter's handling of Gemma's +1 norm shift.

**Implementation.** docs/inference/rust-harness-observability.md:131-132: "The forward pass implements a Gemma-3-style architecture: RMSNorm with `(1 + weight)` scaling" — against docs/inference/rust-harness-runtime-design.md:292-293: "so a `1 + w` multiply here would double-count it on every real checkpoint", and rust-harness/runtime/src/model/cpu.rs:582-583: "for ((dst, v), g) in out.iter_mut().zip(x).zip(w) { *dst = v * inv * g;"

**Consequence.** rust-harness-observability.md:131-132 states "The forward pass implements a Gemma-3-style architecture: RMSNorm with `(1 + weight)` scaling". The implementation does not: cpu.rs:582-583 is "for ((dst, v), g) in out.iter_mut().zip(x).zip(w) { *dst = v * inv * g;", the doc comment above it at :570 says a 1 + w "would double-count the shift on every real checkpoint", and a regression test at cpu.rs:1100 is named rms_norm_scales_by_the_stored_weight_not_one_plus_it. The sibling design doc rust-harness-runtime-design.md:288-293 records the correct resolution - the GGUF converter folds Gemma's runtime +1 before writing, stored already shifted for gemma3n and gemma4 - so the observability text is the stale side of an internal contradiction between two documents in the same directory. Graded medium, not critical: the shipped code is correct and pinned by a test, so no current output is wrong; the harm is prospective and bounded - the planned CUDA backend author who works from the observability doc applies 1 + w, double-counts on every norm of every layer, and gets finite logits and plausible text with no assertion anywhere. Not graded low because the doc is the one a reader consults for what the norm tensors mean and the numerics path is a stated deliverable. Note the underlying converter premise is a GGUF question the spec does not settle (spec:114 fixes only rms_norm_eps 1e-06); confirming it needs a real checkpoint. Two reports, same site, same divergence, anchors 131 and 132 of one sentence; folded.

- upheld: Both cited sites read exactly as claimed. /work/docs/inference/rust-harness-observability.md:131-132 states in present tense that "The forward pass implements a Gemma-3-style architecture: RMSNorm with `(1 + weight)` scaling", and it is the only mention of norms in the entire 290-line file, so no later caveat, correction, or supersession note exists there; the doc is described (line 5) as a live companion to the runtime design, not a historical artifact. The code does the opposite: /work/rust-harness/runtime/src/model/cpu.rs:571-585 computes `inv = rms_inv(x, eps)` then `*dst = v * inv * g;` (plain multiply by the stored weight), with the doc comment at :564-570 stating that `1 + w` "would double-count the shift on every real checkpoint"; rms_norm_in_place (:591-594) matches, and all call sites (:808, 813, 817, 850, 856, 863, 885) route through these two functions. I searched specifically for handling elsewhere that would refute: weight loading (cpu.rs:658-703) applies only length checks, no `+1.0` to any norm tensor; a repo-wide grep for `1.0 + `/`1 + w` in runtime/src finds nothing relevant; the independent numpy golden fixtures/reference_logits.py:118-124 encodes the same stored-weight convention so nothing downstream re-adds the 1; and the regression test cpu.rs:1099-1122 pins unit weight = identity and zero weight = annihilation, which `1 + w` would fail. docs/verification/rust-harness-qa.md:435-445 records the former `(1 + w)` code as confirmed critical finding A1-R03 (since fixed), and its Phase Q6 drift lens (lines 202-204) explicitly scopes the observability doc, so the stale sentence is uncovered. Medium grading stands: shipped numerics are correct and test-pinned, harm is prospective for a CUDA backend author reading the observability doc.

### GC-049 "Skipped: E2B (same code path as E4B)" is false on MLP width rule, KV-head count, and layer-type period

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `mismatch` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-runtime-design.md:269` |
| Variants | E2B, E4B |
| Claims | C11-06, C11-07, C13-13, C2-06, C2-09, C2-15, C3-25, C3-26, C7-01, C7-02, C7-07, C7-10 |
| Verification | 3 verdicts recorded, 1 refutations |

**Reference.** docs/models/gemma-4.md:337-339: "E2B sets `use_double_wide_mlp: true`. The flag does not apply uniformly. Of E2B's 35 layers, 15 use the configured `intermediate_size` of 6144 and **20 use 12288**, exactly twice the configured value." Spec 271-273: "**Distinct layer-type period on E2B.** E2B places a full-attention layer every fifth layer; every other variant uses every sixth." Spec 146: "\| `num_key_value_heads` \| 1 \| 2 \| 8 \| 16 \|"

**Implementation.** docs/inference/rust-harness-runtime-design.md:269-271: "- **Skipped: E2B** (same code path as E4B, weaker coder) **and the larger server-tier variants** (large models are already served well by the existing services; their architectures are out of scope for this runtime)."

**Consequence.** Spec 337-339: E2B sets use_double_wide_mlp true and 20 of its 35 layers use 12288 against the configured 6144. Spec 271-273: E2B places a full-attention layer every fifth layer (indices 4,9,...,34) where every other variant uses every sixth. Spec 146: num_key_value_heads is 1, genuine MQA. A future session acting on "same code path" mis-shapes 20 of 35 MLP layers and is refused by the width guard at cpu.rs:242-250 on a rule E4B never trips, computes the wrong attention schedule, and divides by a KV-head count of 1. This sentence is the repository's only statement about E2B/E4B code-path equivalence, so it is a root rather than a copy, and it matters because the training and interp tiers name E2B, not E4B, as the reference model.

- upheld: Lens: does the spec say what the finding claims? All three citations verify verbatim in /work/docs/models/gemma-4.md. (1) Lines 337-339 read exactly "E2B sets `use_double_wide_mlp: true`. The flag does not apply uniformly. Of E2B's 35 layers, 15 use the configured `intermediate_size` of 6144 and **20 use 12288**, exactly twice the configured value.", corroborated by the shape table at 343-344 and by 350-352 which independently states "A loader that trusts `intermediate_size` for every layer mis-shapes 20 of E2B's 35 layers"; 354-356 confirms E4B has the flag false with uniform 10240, making this an explicit E2B/E4B divergence. (2) Lines 271-272 read exactly "**Distinct layer-type period on E2B.** E2B places a full-attention layer every fifth layer; every other variant uses every sixth.", corroborated by the attention table at 173-174 (E2B full indices 4,9,14,19,24,29,34 vs E4B 5,11,17,23,29,35,41) and the prose at 168-169 forbidding a stride constant. (3) Line 146 is exactly "\| `num_key_value_heads` \| 1 \| 2 \| 8 \| 16 \|", reinforced at 273-275 warning against deriving a group size by division without checking for 1. Hedging check: grepped all "unverified"/"assumed"/"Open questions" occurrences; the Open Questions section (582-605) lists proportional RoPE, layer_scalar, the per-layer embedding path, the E4B shared-KV boundary, 12B/31B tensor data, use_bidirectional_attention, and attention_k_eq_v semantics -- none of the three cited points. The only nearby "unverified" (333, 597-599) applies to the E4B boundary and explicitly affirms that the E2B tail convention is confirmed. Only defect found is trivial line-range slop: the finding cites 271-273 for a quote that ends at 272 (273 begins the separate MQA item, which the finding cites separately via 146). No over-reading, no variant-to-family generalization.
- upheld: Cited site verified verbatim: /work/docs/inference/rust-harness-runtime-design.md:269 reads "- **Skipped: E2B** (same code path as E4B, weaker coder)", and grep shows lines 265 and 269 are the only E2B mentions in that document, so nothing elsewhere qualifies it. The spec citations are accurate: /work/docs/models/gemma-4.md:337-352 (E2B use_double_wide_mlp true; 20 of 35 layers at 12288 vs configured 6144; E4B false with uniform 10240), 173-174 (E2B full-attention every 5th vs every 6th), 146 (num_key_value_heads 1 vs 2). The MLP-width divergence is a real code-path difference: runtime/src/model/cpu.rs check_supported bails when any blk.{i}.ffn_up width differs from Config::ffn_dim, a rule E4B never trips, so "same code path" is false. HOWEVER, two of the three failure sub-claims are already handled and the auditor missed them: (1) n_kv_heads == 1 is supported - the guard at cpu.rs:129 is n_heads.is_multiple_of(n_kv_heads.max(1)) && n_kv_heads > 0, heads_per_kv at cpu.rs:775 is well-defined for 1, and a unit test at cpu.rs:1168 asserts n_kv_heads == 1; (2) the attention schedule is metadata-driven via attention.sliding_window_pattern (cpu.rs:88-93, layer_kind at 144), not a hardcoded stride, so E2B's period 5 arrives from the file. The one true axis also fails loudly (loader bail with a specific message) rather than silently mis-shaping, so the finding holds but at reduced severity.
- REFUTED: Traced from the real entry points (harness/src/main.rs:188 and runtime/src/bin/activation-capture.rs:204) into CpuModel::from_gguf (cpu.rs:646-650). All three claimed consequences fail. (1) Attention schedule is metadata-driven: Config::from_gguf reads attention.sliding_window_pattern (cpu.rs:93) and layer_kind is (i+1).is_multiple_of(pattern) (cpu.rs:144), which with pattern 5 yields full-attention at 4,9,14,19,24,29,34 — exactly the spec's E2B row (gemma-4.md:173); harness/src/main.rs:252 passes the same value to kv::interleaved_layers. No hardcoded stride of 6 exists; a missing key bails by name. (2) KV-head count 1 is fine: cpu.rs:129 requires n_kv_heads > 0 and divisibility (8 % 1 == 0), heads_per_kv = 8, kvh = qh/8 = 0 — correct MQA, no divide-by-zero and no breakage. (3) The width guard at cpu.rs:242-250 is real but unreachable: check_supported's FIRST guard bails on any tensor name containing "per_layer" (cpu.rs:209-216), and per gemma-4.md:289-301 both E2B and E4B carry identical per-layer-embedding tensors, so both are refused at the same guard before any width comparison or forward pass. The design doc itself states this at lines 321-327, and rust-harness/README.md:194 lists "load quantized E4B" as an unstarted roadmap item — the runtime today loads only synthetic fixtures (runtime/src/synth.rs). So in currently reachable behavior E2B and E4B literally do share one code path: the same named refusal. Even hypothetically, the width mismatch is a loud bail naming the tensor and both widths, not silent wrong numerics. The training/interp E2B reference recipes are Python/transformers and never touch cpu.rs. Doc imprecision remains, but with no reachable code consequence.

### GC-050 MatFormer row asserts per-layer feed-forward slicing on E4B at High confidence; E4B widths are uniform 10240 across all 42 layers

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `mismatch` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-runtime-design.md:299` |
| Variants | E2B, E4B |
| Claims | C13-02, C13-03, C6-01, C6-02, C6-09, C7-02, C7-06, C7-08, C7-09, C7-10, C8-01, C8-05, C8-06 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:354-356: "E4B has `use_double_wide_mlp: false` and uniform MLP widths of 10240 across all 42 layers, despite also having a shared-KV tail. The double-wide compensation is therefore not implied by KV sharing; it requires the flag." Spec 376-378 confirms from the header: "\| 42 \| `model.language_model.layers.{L}.mlp.gate_proj.weight` \| `[10240, 2560]` \|"

**Implementation.** docs/inference/rust-harness-runtime-design.md:299: "\| MatFormer nested feed-forward \| Retained; slicing can vary width per layer or per block \| No \| High \|" — the Status column is headed "Status in Gemma 4 E4B" (:295) and the Confidence column reads High.

**Consequence.** Spec 354-356 and the exact inventory at 376-378 give E4B use_double_wide_mlp false and 42 gate/up tensors at [10240, 2560]. The word MatFormer appears nowhere in the spec. The real per-layer-width rule is E2B-only and different in kind: spec 350-352, the final num_kv_shared_layers layers use 2 * intermediate_size under a boolean flag, not arbitrary nested slicing. The guard built from this row (cpu.rs:234-252) fires correctly on an E2B width mismatch, so the failure is loud, but its message ("per-layer widths are MatFormer slicing") names a mechanism that does not exist and steers the follow-up toward nested-model metadata. This row is the origin of every MatFormer mention in the tree: cpu.rs:17 and :245, gguf-info.rs:99/116, rust-harness/README.md:104, docs/verification/rust-harness.md:98/111, docs/inference/rust-harness-observability.md:136.

- upheld: Line 299 of /work/docs/inference/rust-harness-runtime-design.md reads verbatim "\| MatFormer nested feed-forward \| Retained; slicing can vary width per layer or per block \| No \| High \|" under the header at :295 "Status in Gemma 4 E4B", so the site claim is literal. The spec contradicts it directly: /work/docs/models/gemma-4.md:354 "E4B has use_double_wide_mlp: false and uniform MLP widths of 10240 across all 42 layers", corroborated by the exact-count inventory at 376-378 (42 gate_proj/up_proj at [10240, 2560]); a case-insensitive grep of the spec for matformer\|matryoshka\|nested\|elastic\|slic returns zero hits, and the only per-layer width rule it gives is the E2B-only use_double_wide_mlp flag (335-352, 266-267). I searched for handling elsewhere and found none: grepping the design doc for gemma-4.md\|uniform\|10240\|double-wide\|6144\|12288 returns nothing, the rust tree has no double-wide handling at all (grep for double.?wide across /work/rust-harness is empty), cpu.rs:234-252 is a plain ffn_up width equality check whose bail message says "per-layer widths are MatFormer slicing", its test cpu.rs:1224 asserts err.contains("MatFormer") thereby pinning the wrong name, and gguf-info.rs:116/150 prints "MatFormer per-layer slicing" and scans for a nonexistent "matformer" metadata key. The strongest mitigation, the secondhand-sources caveat at :314-319 deferring to runbook RH.1, softens the epistemic weight but does not correct the E4B status or the High confidence rating. Severity medium is right since the guard still fails loudly and correctly on a real width mismatch; only the named mechanism and diagnostic direction are wrong. Minor quibble not affecting the verdict: lines 274 and 387 also mention MatFormer, so :299 is not strictly the sole origin of the term.

### GC-051 LAuReL row marks a "Retained" feature "Not applicable" while the exact E4B header carries 42 learned layer_scalar tensors

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `mismatch` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-runtime-design.md:303` |
| Variants | E2B, E4B |
| Claims | C13-16, C6-01, C6-02, C8-01, C8-04, C8-12 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:375: "\| 42 \| `model.language_model.layers.{L}.layer_scalar` \| `[1]` \|" and spec 405-406: "Each layer carries a scalar tensor `layer_scalar` of shape `[1]`. Its role is not documented in the configuration."

**Implementation.** docs/inference/rust-harness-runtime-design.md:303: "\| LAuReL \| Retained \| Not applicable \| Medium \|"

**Consequence.** "Not applicable" in the Implemented column means no code is needed, which holds only if the feature carries no learned weight. Spec 375 lists 42 model.language_model.layers.{L}.layer_scalar of shape [1], and spec 405-406 plus open question 2 (spec 591-592) record the role and residual placement as undetermined. No LAuReL-named tensor exists in the 28-row inventory either, so the row asserts a status for a term the source of truth never uses. The named refusals at :321-323 do not cover it; only the generic unconsumed-tensor backstop (cpu.rs:302-328) would catch the scalars, and only if the converter emits them as tensors rather than folding them. The honest table entry is "undetermined", not "Not applicable".

- upheld: Verified the cited site verbatim: docs/inference/rust-harness-runtime-design.md:295 heads the third column "Implemented", :297 pairs "Not applicable" with the *dropped* AltUp, :298-302 use "No" for every retained-but-unimplemented feature, and :303 reads "\| LAuReL \| Retained \| Not applicable \| Medium \|" — so within the table's own convention "Not applicable" means no code is needed, which cannot hold for a retained feature. The spec quotes are exact: gemma-4.md:375 lists 42 x model.language_model.layers.{L}.layer_scalar of shape [1], :405-406 say its role is not documented, and :591-592 say its position in the residual or attention path is undetermined; the 28-row inventory contains no LAuReL-named tensor and repo-wide grep finds "LAuReL" only at line 303 (other hits are the wikitext corpus). Searched for handling elsewhere and found none: cpu.rs check_supported (206-330) names only per-layer embeddings, shared KV, MatFormer FFN width, RoPE, and softcapping/scale metadata; LAYER_TENSORS/TOP_LEVEL_TENSORS/INERT_TENSORS (156-180) contain no layer_scalar, so only the generic backstop (302-328) would see it, exactly as the finding states; docs/verification/rust-harness-qa.md:476-483 reviews that backstop and never mentions it. The one near-mitigation — a real E4B file is refused earlier by the per-layer-embedding check at cpu.rs:211 — refuses for a different reason and does not make "Not applicable" true, and evaporates once PLE is implemented. No legend elsewhere redefines the column ("Implemented" appears only at :295).

### GC-052 Table's source list is entirely secondhand, and its stated reason is refuted by the spec's own reproduction recipe

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-runtime-design.md:329` |
| Variants | all |
| Claims | C11-03, C11-14, C11-15 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:609-611: "Every value above regenerates from the pinned revisions with no GPU and no model download beyond a few hundred kilobytes. Run these on the workstation; a contained agent sandbox does not reach `huggingface.co`." Spec 628-631: "the safetensors format stores an eight-byte little-endian header length followed by a JSON header, so a range request retrieves the full tensor map from a file of several gigabytes"

**Implementation.** docs/inference/rust-harness-runtime-design.md:314-316: "Two caveats bound these findings. They are secondhand: the primary sources refuse automated retrieval, so the details come from search summaries of varying reliability"; :329-333 lists an arXiv abstract, a newsletter, a blog post, and llama.cpp issue 21468.

**Consequence.** Lines 314-316 claim "the primary sources refuse automated retrieval"; :329-333 lists an arXiv abstract, a newsletter, a blog post, and llama.cpp issue 21468. Spec 609-611 and 628-636 show config.json is a single curl and the full tensor map a two-step range request of a few hundred kilobytes, both workstation-side. Every wrong row in the table (shared-KV mechanism, MatFormer, value normalization, LAuReL applicability, the 8B total) and every missing row (two head dimensions, final logit softcapping) follows from sourcing architecture from prose summaries of Gemma 3n. This is the single upstream node for the propagation observed in cpu.rs, gguf-info.rs, rust-harness/README.md, and docs/verification/rust-harness.md.  Propagated to: rust-harness/README.md:107, docs/interpretability/rust-harness-interp.md:228. One defect: the secondhand reconciliation table is used as the architecture source of record instead of docs/models/gemma-4.md. runtime-design.md:329's provenance note is the ROOT; README:107 and rust-harness-interp.md:228 both route readers to it. Single remediation: repoint all three at the spec.

- upheld: Reproduced the cited site directly. /work/docs/inference/rust-harness-runtime-design.md:314-316 states verbatim "They are secondhand: the primary sources refuse automated retrieval, so the details come from search summaries of varying reliability"; :329-334 lists exactly four secondhand items (arXiv abstract 2607.02770, a Maarten Grootendorst newsletter, a botmonster.com blog post, llama.cpp issue 21468) and zero primary artifacts. /work/docs/models/gemma-4.md:609-611 and 613-636 show the primary data is a single `curl .../raw/main/config.json` plus a two-step safetensors range request of a few hundred kilobytes, both workstation-side. I searched for handling the auditor may have missed and found none: `grep -rn "models/gemma-4\|gemma-4.md"` returns only the index row in docs/README.md:60 — neither runtime-design.md, rust-harness/README.md, nor rust-harness-interp.md references the spec, and there is no supersession banner (file header lines 1-12 only says the code wins over the doc). The nearest guard, lines 318-319 ("Runbook entry RH.1 remains the authority") and cpu::check_supported refusing at 321-327, mitigates code risk but leaves the table as the only architecture record the two other docs point at (README:107 text link; interp:228 anchor link to #architecture-reconciliation). The table's defects also check out against the spec: spec 110-115/180-193 gives head_dim 256 vs global_head_dim 512 and final_logit_softcapping 30.0 (both absent from the table); spec 312-317 says num_kv_shared_layers shares cache not weights with every layer keeping k_proj/v_proj; "MatFormer" and "LAuReL" appear nowhere in the spec (the real mechanism is use_double_wide_mlp, false on E4B with uniform 10240 per spec 354-356); spec 428 gives ~7.52B total against the table's "8B". Only partial defense: the spec itself notes a sandbox cannot reach huggingface.co, so the excuse is defensible from a sandbox — but the doc phrases it as a property of the sources, not the environment.

### GC-053 Non-goal says vision tensors are skipped at load; nothing calls the skip predicate, and the E-series audio tower goes unmentioned

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-runtime-design.md:75` |
| Variants | E2B, E4B |
| Claims | C12-01, C12-11, C12-23, C13-12 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** spec 276-278: "9. **Full audio tower on the E-series.** E2B and E4B carry a complete `gemma4_audio`\n   configuration and audio tower weights. 31B has `audio_config: null`. Audio support is\n   per-variant, not a family property."

**Implementation.** docs/inference/rust-harness-runtime-design.md:75-76: "- **No multimodality.** The vision components of Gemma 4 checkpoints are skipped at\n  load time." (repeated at line 116 "gguf.rs           # weight loading, vision tensors skipped" and line 151 "The loader maps tensors by name into the backend's per-layer structs and skips vision-tower tensors.")

**Consequence.** Two divergences in one sentence. First, no skipping happens: gguf.rs:53 defines is_vision_tensor and a workspace grep finds callers only in its own unit test (gguf.rs:417-420), so any tower tensor reaches the generic backstop at cpu.rs:302-328 and the load is refused - the opposite of the documented behaviour, and the same false claim appears at :116, :150-151, and in the gguf.rs:7-9 module banner. Second, the non-goal is scoped to vision only, while spec 276-278 records a complete gemma4_audio configuration and audio tower weights on E2B and E4B (31B has audio_config: null), and spec 525-527/577-578 adds zero-dimensional input_min/input_max clipping companions that a generic loader must tolerate. Neither the audio tower nor the zero-dim tensors are named anywhere in this directory.  Propagated to: rust-harness/runtime/src/gguf.rs:7. One claim that vision tensors are filtered at load, where is_vision_tensor has no caller. The design doc's non-goal at runtime-design.md:75 is the ROOT assertion; the gguf.rs:7 banner restates it beside the dead predicate. The audio tower's total absence (S30-M4-abs-09) is a different finding and stays separate.

- upheld: Reproduced the full chain. (1) /work/docs/inference/rust-harness-runtime-design.md:75-76 reads exactly "**No multimodality.** The vision components of Gemma 4 checkpoints are skipped at load time", repeated at :116 and :151. (2) /work/rust-harness/runtime/src/gguf.rs:52-55 defines is_vision_tensor; a repo-wide grep (excluding .git) finds callers only at gguf.rs:417-420 inside #[cfg(test)] fn vision_tensors_are_identified_for_skipping — no production call site exists; the repo's own /work/tests/agents/unverified_ledger.md:58 records the same ("is_vision_tensor is dead code with no caller"). (3) The real load path does the opposite of skipping: CpuModel::from_gguf (cpu.rs:646-650) calls check_supported first, whose generic backstop (cpu.rs:302-328) accepts only TOP_LEVEL_TENSORS (token_embd.weight, output_norm.weight, line 175), INERT_TENSORS (rope_freqs.weight only, line 180) and the 13 blk.<i>. suffixes in LAYER_TENSORS (line 156), then ensure!s unconsumed.is_empty() with "refusing rather than ignoring learned weights" — a v.*/mm.* tensor is refused, not skipped. gguf-info.rs:164 calls the same function. (4) I searched for the handling the auditor might have missed: the refusal-semantics paragraph at :321-327 names only per-layer embeddings, shared KV, per-layer FFN widths, and RoPE, never vision, and the reconciliation table at :295-304 mentions neither vision nor audio; nothing qualifies line 75. (5) Spec side verified: gemma-4.md:276-278 carries the audio-tower item as quoted, :525-527 and :577-578 the zero-dimensional clipping companions, and a grep for audio/input_min/clipp/zero-dim across /work/docs/inference/ hits only unrelated files (vllm-parameters.md, art-generation.md). Only immaterial nits: the ledger's stale line number (gguf.rs:210 vs :53), and a vision tensor whose name contained "per_layer" would hit the earlier bail at cpu.rs:211-217 — still a refusal.

### GC-054 Tokenizer module specified as SentencePiece unigram; the spec names a tokenizers-backed GemmaTokenizer with no SentencePiece model file

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `needs-host` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-runtime-design.md:117` |
| Variants | all |
| Claims | C10-01, C11-04, C11-05 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:443-444: "The tokenizer class is `GemmaTokenizer` with a `tokenizers` backend, wrapped by `Gemma4Processor`." Spec 63-72 lists the complete repository file list: ".gitattributes / README.md / config.json / generation_config.json / model.safetensors / processor_config.json / tokenizer.json / tokenizer_config.json" — no `tokenizer.model`.

**Implementation.** docs/inference/rust-harness-runtime-design.md:117: "      tokenizer.rs      # SentencePiece unigram over the GGUF vocabulary"

**Consequence.** Spec 443-444 gives the class as GemmaTokenizer with a tokenizers backend wrapped by Gemma4Processor, and the pinned repository file list (spec 63-72) contains tokenizer.json but no tokenizer.model. That settles the Hugging Face side but not what the converter writes into tokenizer.ggml.model, which is the only thing the runtime reads. If it is not unigram, a unigram decoder segments every prompt differently from the reference tokenizer, logits stay finite, and the only test that would notice is the #[ignore]d golden test at :194-198. Records as a runbook entry: no Gemma 4 GGUF exists on this workstation.  Propagated to: rust-harness/README.md:83, rust-harness/runtime/src/tokenizer.rs:104. One assumption (SentencePiece unigram with tokenizer.ggml.scores; Gemma 4 ships a tokenizers backend and only tokenizer.json). The design doc at runtime-design.md:117 is the ROOT that specified the module, README:84 restates it verbatim ('SentencePiece unigram tokenizer'), and tokenizer.rs:104 implements it as a hard requirement.

- upheld: Cited line verified verbatim: /work/docs/inference/rust-harness-runtime-design.md:117 reads "      tokenizer.rs      # SentencePiece unigram over the GGUF vocabulary", inside a doc whose scope line (:7) states the runtime targets Gemma 4 exclusively. Spec side verified: /work/docs/models/gemma-4.md:443-444 names GemmaTokenizer with a tokenizers backend wrapped by Gemma4Processor, and :63-72 lists tokenizer.json/tokenizer_config.json with no tokenizer.model; a grep of the entire spec for unigram\|sentencepiece\|BPE\|tokenizer.model\|byte.fallback\|GGUF returns zero hits, so nothing authoritative supports "unigram". Searched for handling elsewhere and found none: grep for "ggml.model" across /work/rust-harness/ returns nothing, so no code validates the vocabulary type; /work/rust-harness/runtime/src/tokenizer.rs:104-107 hard-bails on a missing or wrong-typed tokenizer.ggml.scores with no BPE/merges alternate path, and its file header at :2 restates the same assumption, so the doc's "code is correct where they disagree" clause offers no rescue; the doc's Risks section (:366-395) flags Candle, tap parity, chat template, E-series architecture and VRAM but never the tokenizer algorithm; the only covering test (rendered_token_ids_match_hf_reference, described at :191-198) is #[ignore]d with an unimplemented! body. Partial mitigation only: a scores-less BPE export would fail loudly at load, but the llama.cpp HF-vocab path writes a scores array for non-unigram vocabularies, so a scores-bearing non-unigram export loads and silently mis-segments exactly as the finding describes. Only slip is a one-line offset (README.md phrase is at line 84, not 83), which the finding self-corrects in its own text.

### GC-055 Table omits final_logit_softcapping 30.0, an all-variant invariant the loader refuses by name and the forward pass never applies

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `absent-undeclared` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-runtime-design.md:295` |
| Variants | E4B, all |
| Claims | C1-02, C1-06 |
| Verification | 3 verdicts recorded, 1 refutations |

**Reference.** docs/models/gemma-4.md:115: "\| `final_logit_softcapping` \| 30.0 \|" — listed under "These values are identical in E2B, E4B, 12B, and 31B" (spec 105).

**Implementation.** docs/inference/rust-harness-runtime-design.md:295-304 (the reconciliation table) has no softcapping row; grep for "softcap" across docs/inference/*.md returns nothing, and :171-174 describes sampling as "trimmed logits with temperature and top-p" with no capping step.

**Consequence.** Spec 115 lists final_logit_softcapping 30.0 under the invariants identical in E2B, E4B, 12B, and 31B (spec 105). The eight-row table has no softcapping row, grep for "softcap" across docs/inference returns nothing, and :169-174 describes sampling with no capping step. Exactly one of two things happens at RH.1 and neither is recorded: the converter emits the key and cpu.rs:288-300 refuses with a fifth, undocumented reason beyond the four named at :321-323, or the converter drops it and the runtime produces uncapped logits that differ from the Hugging Face reference at every position while reading as fluent - softcapping carries no tensor, so the unconsumed-tensor backstop cannot see it. Which branch occurs is a converter question, not an HF-config one. Raised from medium: one branch is a silent every-logit divergence with no test able to catch it.  Propagated to: rust-harness/README.md:104. One omission copied doc to doc: the declared-subset list names four refusals and leaves out final_logit_softcapping 30.0. The reconciliation table is the ROOT list; README:104 reproduces the same four-item subset and points the reader back at that table (README:107).

- upheld: Both cited spec lines match verbatim at the cited line numbers. /work/docs/models/gemma-4.md:115 is `\| `final_logit_softcapping` \| 30.0 \|`, and it sits inside the table opened by the section `## Invariants across all four variants` (:103) whose lead-in at :105-106 reads "These values are identical in E2B, E4B, 12B, and 31B. They are safe to treat as constants, and a verification pass should assert them rather than read them per model." The row is in the all-variant table, not the per-variant table that begins at :140, so the finding's [all/E4B] scope is the spec's own scope with no widening. Attempts to refute all failed: `grep -n softcap` across /work/docs returns exactly one hit in gemma-4.md (:115), with no hedging second mention; the fourteen-item `## Traps and discrepancies` (:553-581) does not list softcapping, so the spec does not flag it as a name suggesting behavior it lacks; the seven-item `## Open questions` (:582-605) covers proportional RoPE, layer_scalar, the per-layer embedding path, the E4B shared-KV boundary, missing 12B/31B tensor data, use_bidirectional_attention, and attention_k_eq_v, but not softcapping, so the finding is not citing an Open Question as settled; and the provenance caveat at :57-59 (no tensor-level data for 12B/31B) does not reach a config.json field whose blob hashes are pinned at :51-55. One caveat that does not refute the cited claims: the spec records the value and its scope only, and says nothing about where softcapping is applied in the forward pass or during sampling, so the finding's "uncapped logits diverge at every position" branch is an inference from HF field semantics rather than spec text. The finding's spec_says block claims only the value and the all-variant scope, both of which hold, and :106 strengthens rather than weakens it.
- upheld: Reproduced every cited fact: docs/models/gemma-4.md:115 lists final_logit_softcapping 30.0 under the all-variant invariants (:103-106); the reconciliation table at docs/inference/rust-harness-runtime-design.md:295-304 has exactly eight rows and none covers softcapping; grep for "softcap" across docs/inference/*.md returns nothing; :169-174 describes sampling with no capping step; and the forward pass has no cap (tanh appears only in GELU at cpu.rs:601). The doc's refusal enumeration at :321-323 (copied into rust-harness/README.md:103-108) names four features, while rust-harness/runtime/src/model/cpu.rs:288-300 bails on any "{arch}.*softcapping*" key plus attention.scale/logit_scale - an undocumented fifth reason. The one part I refute is the grounds for "high": the silent-divergence branch is in fact guarded and regression-tested at cpu.rs:1283-1308, which asserts refusal for the exact key ("gemma4.final_logit_softcapping", 30.0), and docs/verification/rust-harness-qa.md:475-477/:790 records this as A1-14 Fixed. The only uncaught branch requires the converter to drop the key, which is speculation about code not in this repo, and the doc defers to RH.1 at :318-319. Real doc-completeness gap, not a live every-logit divergence.
- REFUTED: The doc gap is real (no softcapping row at rust-harness-runtime-design.md:295-304; :321-323, README.md:104 and even cpu.rs:199 say "four" refusals while cpu.rs:293 adds softcapping and attention.scale/logit_scale), but the stated failure_mode is unreachable for E2B/E4B. check_supported (/work/rust-harness/runtime/src/model/cpu.rs:206-300) evaluates guards in order: per-layer embeddings (:211), shared KV (:221), MatFormer widths (:234), RoPE (:259-279), and only then softcapping (:287-300). Dual/partial RoPE is an all-variant invariant (gemma-4.md:126-127: theta 1e6 with partial_rotary_factor 0.25 plus a second theta 10000), and E2B/E4B additionally carry PLE (spec:94-96) and num_kv_shared_layers 20/18 (spec:149). So a faithful E-series conversion bails at one of the four documented reasons before the softcapping guard is ever reached — branch (a)'s "fifth undocumented refusal" is never the message a reader sees. Branch (b) needs the load to succeed: CpuModel::from_gguf calls check_supported unconditionally at cpu.rs:650 with no bypass flag in harness/src/main.rs or activation-capture.rs, so the forward pass never runs on an E2B/E4B checkpoint and no uncapped logits exist; rust-harness.md:127-129 names that refusal as the expected RH.1 outcome (status PENDING). "Neither is recorded" also fails: gguf-info.rs:43-51 dumps every metadata key verbatim and RH.1 (rust-harness.md:133-137) requires recording the raw metadata section. "No test able to catch it" is contradicted by cpu.rs:1283-1307, which asserts refusal of gemma4.final_logit_softcapping=30.0 by name. Finally, tanh softcapping is monotone, so even hypothetically greedy token selection is unchanged; only probability shape and perplexity move. Remaining defect is documentation completeness only.

### GC-056 "One build serves every Gemma 4 size variant" is unsupportable: Config holds one head_dim, one ffn_dim, and one rope_freq_base

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `unrepresentable` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-runtime-design.md:157` |
| Variants | E2B, E4B |
| Claims | C1-01, C2-02, C2-03, C2-05, C2-15, C7-02, C7-06, C7-07 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:343-344: "\| 0 through 14 \| 15 \| `[6144, 1536]` \| `[1536, 6144]` \|  /  \| 15 through 34 \| 20 \| `[12288, 1536]` \| `[1536, 12288]` \|" and spec 352: "A loader that trusts `intermediate_size` for every layer mis-shapes 20 of E2B's 35 layers."

**Implementation.** docs/inference/rust-harness-runtime-design.md:155-159: "The CPU backend reads its dimensions from GGUF metadata into a `Config` at load time instead of holding them as compile-time constants, so one build serves every Gemma 4 size variant and the loader can refuse a checkpoint whose metadata falls outside the implemented subset."

**Consequence.** cpu.rs:40-53 is a flat struct with a single head_dim (:46), ffn_dim (:47), and rope_freq_base (:51), populated at cpu.rs:89-94. Every variant needs two head dimensions (spec 180-182) and two RoPE bases (spec 238-240: "A single global rope_theta is incorrect for this architecture regardless of which value is chosen"), and E2B needs a 15/20 split feed-forward width (spec 343-344). Reading dimensions from metadata removes the recompile, not the shape restriction. The second clause of the same sentence - that the loader refuses what it cannot express - is the accurate half; the refusals do fire (cpu.rs:242, :267), so the failure is loud and the grade stays medium. The overstatement is repeated as verified fact in docs/training/training-scale-reference.md:142-147.

- upheld: Reproduced every cited element. /work/docs/inference/rust-harness-runtime-design.md:155-159 contains the quoted sentence verbatim. /work/rust-harness/runtime/src/model/cpu.rs:39-53 is a flat Copy struct with exactly one head_dim (:46), one ffn_dim (:47), one rope_freq_base (:51), populated at :89/:90/:94. The singleness propagates: CpuModel::from_gguf (:672-703) derives q_dim/kv_dim from the single head_dim and applies identical widths to every layer's attn_q/k/v/output and q_norm/k_norm, the single ffn_dim to every gate/up/down, and builds one rope_freqs table (:703). A crate-wide grep for head_dim\|ffn_dim\|rope_freq_base finds no per-layer dimension anywhere. The authoritative spec /work/docs/models/gemma-4.md:103-111 lists head_dim 256 AND global_head_dim 512 as invariants across all four variants, :126-127 lists both RoPE regimes (1000000.0 proportional / 10000.0 default) as invariants across all four, :180-182 says the two head dims are visible in the weight shapes, :238-240 says a single global rope_theta is incorrect regardless of value, and :337-352 gives E2B's 15/20 split MLP width - so the E2B/E4B/12B/31B scope tag is correct. I searched for handling the auditor missed and found only refusals, which the finding already credits: check_supported (:206-300) and the per-tensor length check at :665-669 that would reject a real E4B global layer's 512-element attn_q_norm. The charitable reading ('one build' = no recompile per size) does not save the sentence: the same doc's table at :295-304 marks dual RoPE/PLE/MatFormer/shared KV as 'No' for E4B and :268 states 'E4B is the sole supported model', contradicting 'every Gemma 4 size variant' internally. Only weakness: the corroborating citation docs/training/training-scale-reference.md:142-147 makes a narrower, largely defensible claim about a scaled-down self-designed Gemma-block model, not about Gemma 4 size variants - a supporting sentence, not the site claim. Severity medium is right because the failure is a loud load-time refusal (cpu.rs:242, :267, :665), not silent wrong logits.

### GC-068 Reference hook model.layers.15 is justified as 50 percent of a 30-layer stack; on E2B it is the double-wide/shared-KV boundary

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/interpretability/investigations/p4_d3_sae_training_investigation.md:76` |
| Variants | E2B |
| Claims | C2-02, C3-02, C6-07, C7-04, C7-05 |
| Verification | 3 verdicts recorded, 0 refutations |

**Reference.** gemma-4.md:142 `\| `num_hidden_layers` \| 35 \| 42 \| 48 \| 60 \|`; gemma-4.md:344 `\| 15 through 34 \| 20 \| `[12288, 1536]` \| `[1536, 12288]` \|`; gemma-4.md:328-329 `The shared region is the **tail** of the layer stack. On E2B the 20 shared layers are indices 15 through 34, which is confirmed by their correlation with the double-wide MLP region described below.`

**Implementation.** p4_d3_sae_training_investigation.md:75-77 `The repository's subject is Gemma 4 E2B; the P4-D2 reference capture is `model.layers.15` (50 percent of a 30-layer stack). The SAE config takes the hook name from the capture manifest, so this is data, not code.`; p4_d3:67 `harvest recipes at `training/recipes/interp/harvest-*.yaml` capture `model.layers.15`.`

**Consequence.** Two errors compose. (1) The depth arithmetic runs against 30 layers; E2B has 35 (spec:142), so 50 percent depth is index 17. (2) Index 15 is the worst available choice on E2B: it is the first layer of the double-wide MLP region (spec:344, gate/up [12288, 1536] against [6144, 1536] below it) and simultaneously the first layer of the 20-layer shared-KV tail (spec:328-329), and it is the first sliding layer after full-attention layer 14 (spec:173). The 300M-token reference capture (training/recipes/interp/harvest-false-fact.yaml:22) and every SAE trained from it therefore sit on a triple structural discontinuity while the recorded rationale claims a neutral mid-stack depth, and no manifest field records that the depth straddles a boundary. Graded high: this is a GPU-days artifact decision a downstream consumer acts on, and the resulting features are not comparable to a capture one layer away. Root: the 30-layer figure at p4_d2:109, via p4_d2:127.  Propagated to: training/calculator/components.py:18. The 'layer 15 = 50 percent depth' hook choice, justified from the same 30-layer error but a distinct downstream defect from the geometry constant itself. P4-D3 is the ROOT justification; harvest-smoke.yaml:18 copies the comment verbatim ('50 percent depth of a 30-layer stack'). Kept out of the 2304 group because the remediation is a hook index, not a width.

- upheld: Spec-lens check passes on every cited line. gemma-4.md:142 reads verbatim `\| `num_hidden_layers` \| 35 \| 42 \| 48 \| 60 \|`, so E2B is 35 layers and the p4_d3:76 "30-layer stack" rationale (verified verbatim at that line) has no spec basis; mid-depth of 0-34 is index 17, and 15 is ~43 percent. gemma-4.md:343-344 reads verbatim `\| 0 through 14 \| 15 \| [6144, 1536] \| ...\|` and `\| 15 through 34 \| 20 \| [12288, 1536] \| [1536, 12288] \|`, making layer 15 the first double-wide layer (reinforced at :346-352 and trap 3 at :564). gemma-4.md:328-329 reads verbatim "The shared region is the tail of the layer stack. On E2B the 20 shared layers are indices 15 through 34, which is confirmed by their correlation with the double-wide MLP region described below." Critically, the Open Questions section (:597-599) and :330-333 hedge only the E4B boundary as unverified — the finding cites the E2B side, which the spec calls confirmed from weights, so no Open Question is being read as settled. gemma-4.md:173 lists E2B full-attention indices 4, 9, 14, 19, 24, 29, 34, supporting that 15 is sliding after full layer 14. No family-level over-generalization: E4B has use_double_wide_mlp false with uniform 10240 widths (:354-356) and every claim is stated per variant. Site text also verified: p4_d3:67 and :75-77, and harvest-smoke.yaml:18 carries the comment "# 50 percent depth of a 30-layer stack" verbatim. Only weakness found: the "first sliding layer after full attention" leg is true but also true of indices 5, 10, 20, 25, 30 per :173, so it is non-discriminating rhetoric rather than a misread; the two unique-to-15 legs remain spec-confirmed.
- upheld: The cited line /work/docs/interpretability/investigations/p4_d3_sae_training_investigation.md:76 reads verbatim: "The repository's subject is Gemma 4 E2B; the P4-D2 reference capture is `model.layers.15` (50 percent of a 30-layer stack)" — the 30-layer figure is attributed to E2B, not to the external Gemma-2-2B design document. Spec checks: gemma-4.md:142 gives E2B num_hidden_layers=35 (so 50 percent depth is index 17); gemma-4.md:344 puts layers 15-34 at gate/up [12288, 1536] against [6144, 1536] for 0-14, making 15 the first double-wide layer; gemma-4.md:328-329 puts the 20 shared-KV layers at indices 15-34, making 15 the first shared layer; the E2B attention table lists full-attention indices 4, 9, 14, 19, 24, 29, 34, making 15 the first sliding layer after full layer 14. All three discontinuities coincide at index 15 as claimed. Downstream sites verified: harvest-false-fact.yaml:22 hook_layers: [15] with total_tokens: 300000000, and harvest-smoke.yaml:18 carries the verbatim comment "50 percent depth of a 30-layer stack". Searched for handling elsewhere and found none: interp/harvest.py:67-74 validates only non-empty, index >= 0, and multi-layer acknowledgement (no upper bound against num_hidden_layers, no region check); check_hook_dims (harvest.py:169-220) compares only residual width against hidden_size, which is uniformly 1536 across the boundary so it cannot fire; the manifest at harvest.py:407-408 records only hooks/hook_layers with no boundary field; interp/families.py just templates model.layers.{n}. The only near-mitigation is an unactioned hedge at the root doc (p4_d2:110-112, :127, "recompute if the shipped config differs"), which is in a different file, absent from the cited line, and silent on the structural boundary. Only wobble is provenance labeling (the finding calls both p4_d2 and p4_d3 the root, and p4_d3 itself says "this is data, not code"), which does not affect the site claim.
- upheld: Reachable and reproduced end to end on E2B: Makefile:303-306 `make harvest` -> `python -m interp harvest --recipe training/recipes/interp/harvest-smoke.yaml` (model_id google/gemma-4-E2B, hook_layers [15] with the literal comment "50 percent depth of a 30-layer stack") and harvest-false-fact.yaml:22 (300M tokens, sealed E2B organism); interp/harvest.py:54 -> interp/families.py:47,95 resolves `model.layers.15`, and harvest.py:210 registers a forward hook, so the capture is the POST-block residual of layer 15. validate_harvest (harvest.py:59-87) bounds the index only at >= 0 with no check against num_hidden_layers, and the manifest (harvest.py:401-424) records hooks/hook_layers/d_model but no layer count or boundary field, so nothing catches it. Against the authoritative spec, all three boundary claims hold: gemma-4.md:142 (E2B = 35 layers, so 50 percent is 17), :344 (double-wide 12288 region begins exactly at 15), :328-329 (shared-KV tail = 15 through 34), :173 (full-attention indices 4,9,14,19,24,29,34, so 15 is the first sliding layer after full layer 14). The causal chain is explicit in-tree: p4_d2:109 (num_hidden_layers=30) -> p4_d2:127 (7/15/22 as 25/50/75 percent) -> p4_d3:76 -> harvest-smoke.yaml:18. Downgraded to medium because the consequence is milder than stated: the residual stream is hidden_size-wide at every layer, check_hook_dims passes, and the SAE trains on valid data - the defect is a provably false recorded rationale and a boundary-adjacent hook choice, not a failing or malformed artifact; "features not comparable to a capture one layer away" is an interpretability judgment true of any adjacent pair; and I have no evidence the 300M-token capture was ever run (/mnt/ai-data is unreachable from the sandbox and P4-D3 records the deliverable as greenfield), so the GPU-days are prospective rather than spent.

### GC-069 Interp contract states the PLE join point and gating as settled fact; the spec records them as an open question

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/interpretability/rust-harness-interp.md:240` |
| Variants | E2B, E4B |
| Claims | C13-17, C5-01, C5-11, C5-12, C5-13, C8-11 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** gemma-4.md:593-596 `- **The exact per-layer embedding data path.** The tensor names establish which projections exist, but the order of operations, where the per-layer embedding joins the residual stream, and whether `per_layer_input_gate` gates additively or multiplicatively, are not determined by shapes alone.`

**Implementation.** rust-harness-interp.md:240-249 `Retaining PLE shifts what a tapped vector means. Each layer sums a token-identity embedding lookup with a projection of the input embedding and adds the result to the residual stream at the end of the block. The tap point stays well defined, and it stays the same tensor the Python capture path records, because the injection lands before the block's output. What changes is the interpretation: … The consequence is for probe evaluation, not for the contract.`

**Consequence.** rust-harness-interp.md:240-249 asserts three things the spec (gemma-4.md:593-596) lists as undetermined: that the token-identity lookup and the input-embedding projection are summed, that the sum joins the residual stream, and that it joins 'at the end of the block ... before the block's output'. The gate (per_layer_input_gate, spec:295) is not mentioned at all. The tap-parity conclusion at :242-244 rests on the assertion: it is the only argument offered that Tap::on_residual (interp.md:53-58, 'fires after each decoder layer's final residual add ... matches the hidden state nnsight exposes as block.output[0]') observes the same tensor as interp/probes/capture.py. If the injection lands at the block input or the gate is multiplicative, the two capture paths diverge and probes trained on one score the other with finite, plausible numbers. The same assumption has a second home in a comment at rust-harness/runtime/src/model/cpu.rs:209-210 ('a token-identity table plus a projection, summed and added as an extra residual update'). Blast radius is bounded today because cpu.rs:211 refuses any per_layer checkpoint outright, so no numerics depend on it; the cost is a misdirected interp work item at :251-255 and an assumption a future implementer would take as given. Anchor corrected from 241/242 to 240, where the passage begins.  Propagated to: rust-harness/runtime/src/model/cpu.rs:209. One open question stated as settled fact: the PLE join point and gating. The interp contract at rust-harness-interp.md:240 is the assertive ROOT; the cpu.rs:209 guard comment states the same per-layer data path as fact. Lower confidence than the other groups - if a skeptic shows the comment was written independently of the contract, split it back out.

- upheld: Verified at the cited lines. /work/docs/interpretability/rust-harness-interp.md:240-244 asserts without hedge: "Each layer sums a token-identity embedding lookup with a projection of the input embedding and adds the result to the residual stream at the end of the block. The tap point stays well defined, and it stays the same tensor the Python capture path records, because the injection lands before the block's output." /work/docs/models/gemma-4.md:593-596, under "## Open questions" ("Recorded rather than guessed"), lists the order of operations, the residual join point, and whether per_layer_input_gate gates additively or multiplicatively as undetermined; the spec's descriptive text at 303-306 only names which projections exist. I searched for qualification elsewhere and found none that covers the claim: the section's hedge at interp.md:228-232 points to the reconciliation table (rust-harness-runtime-design.md:295-304), whose single PLE row says only "Retained ... \| No \| High" and nothing about the data path or gating. A repo-wide grep for per_layer_input_gate returns hits only in gemma-4.md (295, 379, 595), confirming the gate is unmentioned in the contract and in cpu.rs. The parity claim has no independent support: interp/probes/capture.py:103 saves block.output[0] while interp.md:53 defines the Rust tap as firing after the layer's final residual add, so they coincide only if the injection lands before the block output; no PLE parity test exists or can exist because rust-harness/runtime/src/model/cpu.rs:211 bails on any tensor name containing "per_layer". That guard bounds blast radius (as the finding already concedes) but does not remove the misstatement, and cpu.rs:209-210 restates the same undetermined path as fact, with a third consumer inheriting it unhedged at rust-harness/docs/observation-frames.md:144-146. One minor overreach in the finding's rationale does not change the outcome: an injection at the block input would not by itself break parity at the block-output boundary; the divergence cases are a join after the final residual add and a multiplicative gate.

### GC-072 F5 prices an nf4 E4B teacher at 3-4GB 'PLE tables included'; the PLE table alone is 5.6GB at bf16

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/training/investigations/f5_distillation_investigation.md:284` |
| Variants | E2B, E4B |
| Claims | C12-01, C12-10, C12-11, C12-23, C5-14, C9-02, C9-07, C9-08, C9-09, C9-10 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** spec:435-437 "At bfloat16, E2B's per-layer table alone occupies approximately / 4.7GB and E4B's approximately 5.6GB, before optimizer state."; spec:430-433 "The middle row corresponds closely to the advertised effective parameter counts ... The effective count therefore appears to exclude the per-layer embedding table."

**Implementation.** f5_distillation_investigation.md:284-286 "- Teacher `gemma-4-E4B` nf4: on the order of 3-4GB weights (4B-class effective / parameters, PLE tables included), plus one inference forward of activations. / - Student `gemma-4-E2B` QLoRA (nf4 base + adapters): 2-3GB weights, ..."

**Consequence.** f5_distillation_investigation.md:284-288 budgets 3-4GB for an nf4 E4B teacher '(4B-class effective parameters, PLE tables included)' and 2-3GB for an nf4 E2B student. Both figures sit below the bf16 size of the per-layer table by itself: spec:435-437 gives approximately 4.7GB for E2B and 5.6GB for E4B before optimizer state, and spec:430-433 states the effective count excludes that table, which makes 'effective parameters, PLE tables included' self-contradictory. Priced through this repository's own model the gap widens rather than closes: linear_param_count (training/calculator/components.py:169-176) covers only the per-layer attention and MLP projections, and weights_bytes (:310-316) prices everything else - embeddings and every PLE term - at compute dtype, so an nf4 E4B teacher is near 9GB and an nf4 E2B student near 6GB, against the roughly 6GB combined the document assumes when it concludes at :290 that 'Online mode with both models nf4 fits comfortably'. Held at medium, not high: the paragraph labels itself 'to be validated by D4 and calibration, not authoritative' (:281-282), the D4 preflight prices it correctly, the failure would be a loud OOM rather than wrong numbers, and f8_recipe_cards_investigation.md:145 carries a consistent figure. Same root as the other E-prefix sizing errors: budgeting from the advertised effective parameter count, the exact error spec:437-439 names.  Propagated to: rust-harness/runtime/src/model/cpu.rs:5. One unpriced quantity: the E4B teacher's bf16 per-layer table is left out of the distillation memory budget. F5:284 is the documentary ROOT ('3-4GB, PLE tables included'); distill-e2b.yaml:5 asserts the resulting 32GB fit and its header names itself the F5 reference recipe.

- upheld: Cited text verified verbatim at f5_distillation_investigation.md:284-290 and gemma-4.md:430-439. My best refutation was that the finding compares an nf4 budget against a bf16 table size, but that fails: training/tune/distill.py:202-234 loads the teacher with a plain BitsAndBytesConfig (no llm_int8_skip_modules), so bitsandbytes quantizes nn.Linear only and both embedding tables stay bf16; training/calculator/components.py:436-443 (teacher_weights_bytes) and :169-176 (linear_param_count, docstring: "the PLE terms stay in the compute dtype") model exactly that, giving ~9GB for an nf4 E4B teacher and ~6.5GB for an nf4 E2B student versus the doc's 3-4GB / 2-3GB. The doc's numbers match a naive "everything at 4 bits" (7.52B x 0.5 = 3.8GB) that nothing in the repo does. I searched for handling elsewhere: docs/training/distillation.md carries no GB figures (only :89 "the estimate includes a teacher term"), f8:145's "roughly 16GB" corroborates the corrected accounting rather than 3-4GB, and the calculator/calibrate guards are runtime refusals that do not correct the prose. The doc's conclusion is materially at risk too: teacher_forward_bytes at the recipe's seq_len 2048 / micro_batch 2 is ~6.4GB of logits on top of ~15.5GB corrected weights. Minor imprecision in the finding: 3-4GB actually corresponds to the full 7.52B count at 4 bits, not to the 4.5B effective count, so the stated root cause fits loosely; the operative defect (unpriced bf16 PLE table) holds.

### GC-073 F7 scopes the unmapped layer_scalar converter gap to derivative checkpoints; every base E2B/E4B layer carries one

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/training/investigations/f7_eval_wrapper_investigation.md:178` |
| Variants | E2B, E4B |
| Claims | C13-16, C8-04, C8-12 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:375 — "\| 42 \| `model.language_model.layers.{L}.layer_scalar` \| `[1]` \|" and :405-406 — "Each layer carries a scalar tensor `layer_scalar` of shape `[1]`. Its role is not documented in the configuration."

**Implementation.** f7_eval_wrapper_investigation.md:177-180 — "llama.cpp serves Gemma 4 (the export pin registers the architectures), but `convert_hf_to_gguf.py` has open gaps around some derivative checkpoints (unmapped tensors such as `layer_scalar`). This is F6's surface, not F7's; F7 only consumes GGUFs that already exported."

**Consequence.** f7_eval_wrapper_investigation.md:176-180 files unmapped tensors 'such as layer_scalar' as 'open gaps around some derivative checkpoints' and defers them to F6. The spec's inventory is a complete count from the E4B safetensors header at revision 411aa17b: 42 occurrences, one per layer (spec:375), 35 on E2B, with the role undocumented (spec:405-406). So the gap, if real, sits on the primary export path - every fine-tuned E2B/E4B to GGUF, which is what F6 exists to do - not on an optional judge model, and the stated mitigation ('Judge GGUFs should come from official repositories') does not cover checkpoints this repository trains. training/export/convert.py validates only architectures[0] against the converter's registration list, so nothing in the export preflight notices an unmapped per-layer tensor before the multi-minute conversion runs. Whether the pinned llama.cpp checkout maps it is a host question and stays open; the mis-scoping is provable in-tree and is what is graded here. Also bears on the rust side, where a GGUF carrying a layer_scalar family would hit the unconsumed-tensor backstop in cpu::check_supported.

- upheld: The cited line does exactly what the finding claims. /work/docs/training/investigations/f7_eval_wrapper_investigation.md:176-180 reads: "convert_hf_to_gguf.py has open gaps around some derivative checkpoints (unmapped tensors such as `layer_scalar`). This is F6's surface, not F7's ... Judge GGUFs should come from official or well-established quantized repositories" - attaching layer_scalar to derivative checkpoints. The authoritative spec /work/docs/models/gemma-4.md:375 lists 42 occurrences of model.language_model.layers.{L}.layer_scalar in an exact inventory from the E4B safetensors header at revision 411aa17b, :394 states E2B holds the same tensor names across 35 layers, and :405-406 plus :591 state every layer carries it. E2B/E4B are the official google repos at pinned revisions (:44-49), so they are not "derivative checkpoints" under any reading of the spec. I searched for handling elsewhere and found none: `layer_scalar` occurs in exactly two files repo-wide (the F7 doc and the spec); /work/training/export/convert.py:81-98 check_architecture() validates only architectures[0] against names scraped by supported_architectures() (:51-78) with no tensor-level inspection; /work/training/export/preflight.py defines only commit_from_cache_path, resolve_base_identity, check_base_identity (base identity/revision, not tensors); the F6 doc that F7 defers to (/work/docs/training/investigations/f6_adapter_merge_serve_investigation.md:116-132) never mentions unmapped tensors, so the deferral lands nowhere; and the rust side has only the generic unconsumed-tensor backstop at /work/rust-harness/runtime/src/model/cpu.rs:302-327, which refuses rather than handles. Refutation attempts that failed: reading "derivative checkpoints" as the E-series (spec gives no such framing), a charitable parse where the parenthetical merely illustrates the gap class (still misattributes a base-model tensor), and a converter/preflight guard catching unmapped tensors (none exists). The finding also properly hedges the one unverifiable-in-sandbox point (whether the pinned llama.cpp maps it).

### GC-074 F8 records 'KV projection sharing on E-models', the weight-sharing misreading the spec names as trap 2

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/training/investigations/f8_recipe_cards_investigation.md:138` |
| Variants | E2B, E4B |
| Claims | C13-02, C6-01, C6-02, C6-03, C6-04 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** spec 314-317: "`num_kv_shared_layers` does **not** indicate shared weights. The safetensors headers show that every layer carries its own `k_proj` and `v_proj` ... The sharing applies to the key-value cache at inference time, not to the projection parameters."  spec 321-322: "A loader that expects 18 of E4B's 42 layers to omit key and value projections fails to load the weights, because those tensors are present."

**Implementation.** docs/training/investigations/f8_recipe_cards_investigation.md:136-139: "**Architecture**: multimodal wrapper config (`Gemma4Config` with `text_config`); per-layer embeddings (PLE), mixed sliding/full attention (5:1, window 512), KV projection sharing on E-models. The repository's memory calculator already models `gemma4`/`gemma4_text`."

**Consequence.** f8_recipe_cards_investigation.md:136-139 lists 'KV projection sharing on E-models' as a verified architecture property. Spec:314-317 states the opposite: 'num_kv_shared_layers does not indicate shared weights ... every layer carries its own k_proj and v_proj: 42 of each on E4B, 35 of each on E2B, with no layer omitted. The sharing applies to the key-value cache at inference time, not to the projection parameters', restated as trap 2 at spec:562-563. This is the reading spec:321-322 says produces failure mode 1, a loader expecting 18 of E4B's 42 layers to omit key and value projections; a reader sizing an E4B QLoRA fit from this line also under-counts the trainable stack. The document's own next bullet (:140-144, a bug about KV-shared layers recomputing K/V) is consistent with cache sharing and inconsistent with weight sharing, so the two bullets contradict each other. Doc-only today: training/calculator/components.py:151 prices k and v for every layer, so the calculator does not inherit the error. Suspected root is the secondhand reconciliation table at docs/inference/rust-harness-runtime-design.md:300, which the audit charter marks suspect.

- upheld: Verified at source: /work/docs/training/investigations/f8_recipe_cards_investigation.md lists "KV projection sharing on E-models" (phrase wraps lines 137-138) in an unhedged "Architecture" bullet under a section headed "Verified July 2026". /work/docs/models/gemma-4.md:314-317 states the opposite ("does not indicate shared weights ... every layer carries its own k_proj and v_proj ... sharing applies to the key-value cache ... not to the projection parameters"), with failure mode 1 at :321-322 and trap 2 restated at :562-563; the spec's own vocabulary is uniformly "KV cache sharing" (:95, :255, :312, :347) and never calls projections shared, so this is not spec shorthand. Refutation attempts all failed: (a) no superseded/historical/provisional marker in the f8 file, and the preamble's "verify-at-implementation" caveat is not applied to this bullet while neighbouring bullets do carry hedges; (b) the adjacent :140-144 bullet describes cache-recompute behavior and contradicts rather than corrects the wording; (c) no downstream guard needed or present because the error is doc-only - /work/training/calculator/components.py:151 computes attn = 2*hidden*q_dim + 2*hidden*kv_dim and multiplies per_layer by self.layers unconditionally, pricing k and v for every layer. Repo-wide grep shows "projection sharing" appears nowhere else except the spec's denial. Only inaccuracy is the hedged "suspected root" aside: docs/inference/rust-harness-runtime-design.md:300 actually reads "Shared KV cache \| Last 18 of 42 layers reuse K/V from the last non-shared layer", i.e. cache semantics, so it is not the source of the projection wording - but that aside is not part of the defect claim.

### GC-076 p2_d5's "~5.1B total / ~2.3B trainable" E2B envelope overstates the spec's ~4.65B total and ~1.88B body

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/training/investigations/p2_d5_optimizer_offload_investigation.md:123` |
| Variants | E2B |
| Claims | C2-02, C2-03, C9-05, C9-06, C9-07, C9-08 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** spec 426-428: "\| Transformer body \| approximately 1.88 billion \| approximately 4.00 billion \|" ... "\| Total including per-layer table \| approximately 4.65 billion \| approximately 7.52 billion \|"

**Implementation.** docs/training/investigations/p2_d5_optimizer_offload_investigation.md:123-129: "**Parameter counts are unverified.** The document states ~5.1B total / ~2.3B trainable for Gemma-4-E2B with the freeze policy. ... Treat the counts as calculator outputs to be asserted within a tolerance envelope at calibration, not as fixed constants."

**Consequence.** Spec:426-428 gives approximately 1.88 billion for the transformer body and approximately 4.65 billion total including the per-layer table, so ~5.1B is about 10 percent high and ~2.3B trainable about 24 percent high under the transformer_stack freeze. Provenance is the same paragraph's wrong geometry: running the calculator's formulas on 30 layers at hidden 2304 reproduces ~2.34B trainable, which is where "~2.3B" comes from. Medium rather than high because the document explicitly labels these counts unverified and directs treating them as calculator outputs asserted within a tolerance envelope - the defect is that the calibration target is wrong, not that a shipped number is trusted. Distinct claim from the geometry root at line 125, though the same edit will touch both.

- upheld: Site check: /work/docs/training/investigations/p2_d5_optimizer_offload_investigation.md:123 literally states "~5.1B total / ~2.3B trainable for Gemma-4-E2B with the freeze policy". I reconstructed the spec's geometry (docs/models/gemma-4.md:142-150, 186-206, 291-299, 337-352: 35 layers, hidden 1536, 28 sliding + 7 full, q_dim 2048/4096, kv 256/512, double-wide MLP 6144 on layers 0-14 and 12288 on 15-34) and got a transformer stack of 1.882B and a total of 4.634B, reproducing the spec table's "approximately 1.88 billion" and "approximately 4.65 billion" exactly. Running the same formulas on the paragraph's own 30-layer/hidden-2304 geometry gives 2.265B stack and 5.04B total, so both stated figures trace to the wrong geometry. My strongest refutation attempt - that "~5.1B total" might include the modality towers the spec scopes out - fails, because the total's excess (+0.4B) equals the body's excess exactly, so it is the body error and not a tower-inclusive scope. Handling elsewhere does not exist: the shipped freeze policy (training/runtime/offload.py:50, patterns embed_tokens/lm_head/per_layer) and training/calculator/components.py:160-167 transformer_stack_param_count() exclude embeddings and every PLE term, i.e. exactly the ~1.88B scope, confirming the 22-24 percent overstatement rather than rescuing it; a repo-wide grep for 5.1B/2.3B returns only this line, and no reference recipe or test asserts a numeric envelope. The same-paragraph "unverified" disclaimer is a mitigation the auditor already priced in at medium severity, not missed handling.

### GC-078 memory_fit.md declares the E-models supported with enable_moe_block the only loud refusal; four E-series features are unmodelled

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/training/memory_fit.md:72` |
| Variants | E2B, E4B |
| Claims | C1-02, C1-12, C1-17, C11-13, C13-03, C2-04, C2-15, C5-05, C7-01, C7-02, C7-06, C7-07, C7-10, C8-01, C8-03, C8-05, C8-07, C8-08, C8-10, C8-11 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:352 "A loader that trusts `intermediate_size` for every layer mis-shapes 20 of E2B's 35 layers." and :309-310 "Any memory plan for the E-series that omits it is wrong by more than a factor of two."

**Implementation.** docs/training/memory_fit.md:72-76 "The Gemma 4 E-models (`model_type` `gemma4`/`gemma4_text`, the Phase 1 primary family) are supported: `spec_from_hf_config` descends into `text_config` and models the per-layer embedding tables. Only a config with the MoE block enabled (`enable_moe_block`) fails loudly — the dense formulas do not model expert routing."

**Consequence.** "Only a config with the MoE block enabled (`enable_moe_block`) fails loudly" reads as a completeness claim about which Gemma 4 configurations the calculator cannot price, sitting one sentence after "Unknown architectures fail loudly rather than guess". Four more do not fit the dense formulas and none of them raises: E2B's double-wide tail (spec:337-339, :352), the split head dimension (spec:180-182), per_layer_model_projection (spec:293), and three of the five per-layer norms (spec:401-406). The doc is the user-facing contract for calculator coverage, so a reader treats a green memfit verdict on continual-gemma4-e2b.yaml as validated. Graded medium, not high: the numeric harm is carried by the calculator findings below and should not be counted twice, and both affected recipes are still DRAFT with "Measured peak VRAM: TBD" (training/recipes/reference/continual-gemma4-e2b.yaml:13-15, sft-qlora-gemma-e4b.yaml:11-13), so no calibration has yet consumed the claim.

- upheld: Cited text at /work/docs/training/memory_fit.md:71-76 matches the finding verbatim ("Unknown architectures fail loudly rather than guess. The Gemma 4 E-models ... are supported ... Only a config with the MoE block enabled (enable_moe_block) fails loudly"). I searched for the guard the auditor might have missed and found none: /work/training/calculator/components.py:216-260 (spec_from_hf_config) has exactly three raise paths - model_type not in HF_SUPPORTED_MODEL_TYPES (:226), enable_moe_block (:238), and an AttributeError fallback (:257). Repo-wide grep for use_double_wide_mlp, global_head_dim, num_kv_shared_layers, per_layer_model_projection, per_layer_projection_norm, layer_scalar returns zero hits in any .py/.rs/.yaml (only agent-memory notes and gemma-4.md). Each cited feature is silently unmodelled: ffn is a single scalar on the frozen ArchitectureSpec (:105, :250) so E2B's 20 tail layers at 12288 are priced at 6144 (~566M params omitted); explicit_head_dim is one value for all layers (:253) with global_head_dim absent; ple_param_count (:132-141) counts the packed table plus two per-layer linears only, omitting the model-level per_layer_model_projection (13,762,560 params on E2B) and per_layer_projection_norm; norms = 2 * hidden per layer (:153, :166) against five per-layer norm tensors plus layer_scalar. No caller avoids the input: continual-gemma4-e2b.yaml:26-29 and sft-qlora-gemma-e4b.yaml:29-31 are model.hf recipes for google/gemma-4-E2B/-E4B routed through spec_from_recipe -> AutoConfig -> spec_from_hf_config, and check_memory_fit (/work/training/tune/sft.py:163-181) prints an [OK] fit line for them; docs/verification/training-phase1.md:380 documents make memfit on the E4B recipe as an acceptance step. The only defence available is that the sentence is literally true of the code, which the finding already concedes; the doc carries no qualifier on parameter-count coverage (line 44 scopes only the activation inventory).

### GC-079 Training ladder prices E2B/E4B at effective parameter counts, omitting the per-layer table that doubles resident memory

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/training/training-scale-reference.md:93` |
| Variants | 12B, 31B, E2B, E4B |
| Claims | C1-03, C11-07, C2-02, C2-03, C2-10, C3-01, C3-02, C3-03, C3-25, C5-02, C5-12, C5-14, C8-11, C9-05, C9-06, C9-07, C9-08, C9-09, C9-10 |
| Verification | 3 verdicts recorded, 0 refutations |

**Reference.** "Continual pretraining plans that size against the effective parameter count underestimate resident weight memory by roughly a factor of two." and "\| Total including per-layer table \| approximately 4.65 billion \| approximately 7.52 billion \|" (/work/docs/models/gemma-4.md:438-439, 428)

**Implementation.** "The Gemma 4 dense variants are 2B, 4B, 12B, and 31B." ... "\| 2B \| pure GPU \| pure GPU \| pure GPU, full \| fits with fp32 Adam \|" / "\| 4B \| pure GPU \| pure GPU \| pure GPU, tight \| requires 8-bit Adam; at the resident ceiling \|" (docs/training/training-scale-reference.md:93, 101-102)

**Consequence.** docs/training/training-scale-reference.md:93 names the variants "2B, 4B, 12B, and 31B", dropping the E prefix that spec:283-285 says marks the count as effective (per-layer table excluded), then applies its own bytes-per-parameter model at :61-64 and the ceilings at :69-71 to those labels. Spec:428 gives resident totals of approximately 4.65 billion (E2B) and approximately 7.52 billion (E4B), and spec:437-439 states plainly that plans sized against the effective count "underestimate resident weight memory by roughly a factor of two". Consequences: the :101 row "2B \| pure GPU \| ... \| fits with fp32 Adam" implies 26 GB of states where 4.65e9 x ~13 B/param is ~60 GB; the :102 row "4B ... at the resident ceiling" implies ~28 GB where 7.52e9 x ~7 B/param (8-bit Adam) is ~53 GB. Even the frozen-PLE path leaves 4.7 GB (E2B) / 5.6 GB (E4B) of bf16 table resident (spec:435-437) that the single-P model never prices. The strings "per-layer", "PLE", and "embedding table" do not occur anywhere in this file. Root: this file is the origin, not a copy - the advertised effective name is transcribed at :93 and every downstream row inherits it, as does docs/interpretability/interp-scale-reference.md:38-39, which copies the rows and the conclusion. This document is the cited basis for scoping continual pretraining to E2B/E4B, so the scoping decision itself rests on the two-fold underestimate. Note for the remediator: several merged findings cite line 70 for the fp32-Adam ceiling; the actual line is 69.  Propagated to: docs/interpretability/interp-scale-reference.md:39. Root/copy pair already declared by the finder's own IDs: the training ladder prices E2B/E4B at effective parameter counts, omitting the per-layer table that doubles resident memory, and the interp ladder is sized from the same rungs.

- upheld: Every spec citation checks out verbatim at the cited lines of /work/docs/models/gemma-4.md. :284-285 states flatly that the E prefix "denotes effective parameters: the per-layer table is excluded from the advertised parameter count". :428 is verbatim "\| Total including per-layer table \| approximately 4.65 billion \| approximately 7.52 billion \|", against :427's body-plus-main-embedding of ~2.28B/~4.67B. :437-439 is verbatim "Continual pretraining plans that size against the effective parameter count underestimate resident weight memory by roughly a factor of two", in the "Parameter accounting and memory" section framed at :435 "For memory planning on a 32GB RTX 5090", and echoed unhedged at :308-310 ("Any memory plan for the E-series that omits it is wrong by more than a factor of two"). :436-437 gives the bf16 table sizes 4.7GB/5.6GB "before optimizer state", supporting the frozen-table residency point (the phrase "frozen PLE" is the finding's own framing, but the numbers and the optimizer-state qualifier are the spec's). Refutation attempts failed: the point is not in Open questions (:582-605 lists only rope type, layer_scalar, per-layer data path, E4B KV boundary, 12B/31B tensor data, use_bidirectional_attention, attention_k_eq_v); the hedges at :421-422 ("derived", "approximate to within rounding") and :430-433 ("appears to exclude ... an inference from arithmetic, not a documented definition") qualify why the advertised number is what it is, not the resident-memory conclusion, and the finding mirrors the spec's own "approximately"/"roughly"; and there is no family-vs-variant over-read, since the spec's own scope line :14 is "the dense text models E2B, E4B, 12B, and 31B" — the identical roster the training doc renders at :93 as "2B, 4B, 12B, and 31B". Only loose wording noted: the title's "doubles" is 2.04x for E2B but 1.61x for E4B, though that is the spec's own "roughly a factor of two". Incidentally confirmed: training-scale-reference.md:69 (not :70) is the fp32-Adam ceiling, and "per-layer"/"PLE"/"embedding table"/"E2B"/"E4B" appear nowhere in that file.
- upheld: Reproduced every cited string. /work/docs/training/training-scale-reference.md:93 literally reads "The Gemma 4 dense variants are 2B, 4B, 12B, and 31B" while /work/docs/models/gemma-4.md:14 scopes them as E2B, E4B, 12B, 31B and :283-285 states the E prefix denotes effective parameters with the per-layer table excluded. The training file prices those labels with the single bytes-per-parameter table at :59-64 and the ceilings at :69-71 (finding's line-69 correction is right), producing rows :101-102 ("2B ... fits with fp32 Adam", "4B ... at the resident ceiling"). Grep for per-layer\|PLE\|embedding table\|E2B\|E4B\|effective over the whole file returns no relevant hit, so there is no in-file caveat or guard. Two candidate rescues fail: (1) /work/docs/training/memory_fit.md:72-74 does model the per-layer tables, but the training doc at :87-89 declares its budgets are inputs TO that calculator, so the downstream fix cannot correct the ladder's own conclusions; (2) the freeze policy in /work/docs/training/investigations/p2_d5_optimizer_offload_investigation.md:37 exists inside a design whose scope summary (:23-26) states full-parameter continual pretraining of Gemma-4-E2B needs DeepSpeed ZeRO-2 with fp32 master weights and Adam moments in host RAM - i.e. offloaded, directly contradicting the ladder's "2B \| continued pretraining \| pure GPU". Spec:428 (4.65e9 / 7.52e9 resident) and :437-439 ("underestimate resident weight memory by roughly a factor of two") plus :310 confirm the magnitude. Propagation to /work/docs/interpretability/interp-scale-reference.md:38-39 verified verbatim.
- upheld: Reproduced the defect at the cited lines: training-scale-reference.md:93 states "The Gemma 4 dense variants are 2B, 4B, 12B, and 31B" (no E prefix; the strings E2B/E4B/per-layer/PLE/embedding table appear nowhere in the file), and :101-102 price those labels against the ceilings at :61-64/:69-71, with :111-112 explicitly binding "the 4B class" to the harness's model of record. The 2x error is verifiable from the repository's own code, not only the spec: training/calculator/components.py:132-158 (param_count including ple_param_count) applied to E4B's shapes yields ~7.40e9 parameters and E2B ~4.6e9, matching gemma-4.md:428's 7.52e9/4.65e9 — so ":101 2B fits with fp32 Adam" (26 GB implied) is false, since 4.65e9 x ~13 B/param is ~60 GB and even full-parameter 8-bit Adam (~32.6 GB) exceeds VRAM. Propagation to interp-scale-reference.md:38-39 confirmed. The finding's note that the fp32-Adam ceiling is line 69, not 70, is also correct. Consequence is milder than the finding's strong form: the actual entry points already price the PLE table — training/recipes/reference/continual-gemma4-e2b.yaml sets trainable_filter: transformer_stack with "the per-layer embedding tables are frozen" and offload: zero_offload (contradicting the doc's pure-GPU row), and the memfit CLI reads vocab_size_per_layer_input/hidden_size_per_layer_input (memory_fit.md:72-76, components.py:216-256), so no executable path inherits the underestimate. The harm is a reader of the prose ladder and its interp copy, not the E2B/E4B scoping decision itself.

### GC-080 "Scaled-down Gemma loads with zero runtime changes" is false: check_supported refuses PLE, dual RoPE, and split FFN widths

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/training/training-scale-reference.md:146` |
| Variants | E2B, E4B, all |
| Claims | C1-01, C1-02, C1-06, C1-13, C1-14, C13-03, C13-04, C13-06, C2-10, C2-15, C2-19, C3-01, C3-07, C4-01, C4-03, C4-04, C4-06, C5-15, C7-01, C7-02, C7-06, C7-07, C7-10 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** spec 238-240: "the two regimes differ in base frequency by two orders of magnitude (1000000.0 against 10000.0). A single global `rope_theta` is incorrect for this architecture regardless of which value is chosen."  spec 180-182: "The two attention types use **different head dimensions**. Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512)."  spec 252-254: "`hidden_size_per_layer_input` is 256 on the E-series and 0 on 12B and 31B. Without this mechanism the E-series does not run at all."

**Implementation.** docs/training/training-scale-reference.md:136-147: "**Design constraint: from-scratch models use the Gemma 4 block at reduced scale.** ... This is verified, not assumed: `Config::from_gguf` (`rust-harness/runtime/src/model/cpu.rs:56`) gates only on the architecture string starting with `gemma` and reads every dimension (layers, heads, embedding width, FFN width, sliding-window pattern, RoPE base) from GGUF metadata. A scaled-down Gemma exported with the expected metadata keys loads with zero runtime changes."

**Consequence.** docs/training/training-scale-reference.md:136-147 adopts "the Gemma 4 block at reduced scale" as the phase-4 design constraint and calls the compatibility claim "verified, not assumed", citing Config::from_gguf only. The citation is accurate about that function (cpu.rs:56-96 does read those keys) but the struct it fills holds one head_dim (attention.key_length, cpu.rs:89), one ffn_dim (feed_forward_length, cpu.rs:90) and one rope_freq_base (cpu.rs:94), and the loader also calls check_supported, which bails by name on: any tensor containing "per_layer" (cpu.rs:211-217), which spec:252-254 says the E-series cannot run without; any ffn_up width differing from feed_forward_length (cpu.rs:242-250), which spec:337-339 and :352 say applies to 20 of E2B's 35 layers; and any {arch}.rope.* key beyond freq_base plus a full-width dimension_count (cpu.rs:275-279), where spec:222 and :238-240 require two bases (1000000.0 and 10000.0) and partial_rotary_factor 0.25. Spec:180-182 adds a second head dimension (256 sliding / 512 full) the Config cannot express. The claim is true only for a Gemma-3-shaped block, which cpu.rs:10-22 says outright is what it implements. Two further gaps the merged findings add: the only scratch builder is training/models/model.py:66-77, which constructs a LlamaConfig with a single rope_theta and none of the per-layer q_norm/k_norm/post_attention_norm/post_ffw_norm tensors CpuModel::from_gguf requires; and training/recipes/schema.py:83 offers one `rope_theta: float = 10000.0`, so the two-regime requirement is not representable in a recipe at all. Blast radius is bounded and loud - the refusal fires at load - but the sentence licenses a phase-4 architecture choice and an exporter contract on a premise that does not hold.

- upheld: Reproduced the whole chain. training-scale-reference.md:136-147 reads verbatim as quoted, declaring "from-scratch models use the Gemma 4 block at reduced scale" and calling the compatibility claim "verified, not assumed" with no qualifier elsewhere in the file. cpu.rs Config (39-53) holds exactly one head_dim (89), ffn_dim (90), rope_freq_base (94). check_supported (206) bails on any tensor containing "per_layer" (211-217), on any ffn_up last-dim != feed_forward_length (234-252), and on any {arch}.rope.* key beyond freq_base/full-width dimension_count (259-280). That function is not bypassable: CpuModel::from_gguf calls it at line 650 before loading any weight, and it plus gguf-info.rs:164 are the only consumers; mock/probe/record load no Gemma checkpoint, and harness/main.rs:188, activation-capture.rs:204 and tests all route through CpuModel::from_gguf. The spec confirms Gemma 4 requires two head dims (256/512), two RoPE bases (1e6/1e4 with partial_rotary_factor 0.25, "a single global rope_theta is incorrect"), PLE ("without this mechanism the E-series does not run at all"), and double-wide MLP on 20 of E2B's 35 layers. cpu.rs:10-22 says the backend implements a Gemma-3-style subset and refuses the rest. Corroborating gaps hold too: training/models/model.py:66-77 builds a LlamaConfig with one rope_theta and none of the q_norm/k_norm/post_attention_norm/post_ffw_norm tensors LAYER_TENSORS (156-170) requires; schema.py:83 exposes a single rope_theta default 10000.0. No guard, allowlist, scale exemption, or alternate loader rescues the claim.

### GC-081 RH.1's four decision questions omit the per-layer-type head dimension, the spec's Trap 6

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `absent-undeclared` |
| Tier | `docs` |
| Site | `/work/docs/verification/rust-harness.md:106` |
| Variants | E2B, E4B |
| Claims | C1-01, C1-02, C13-06, C3-07, C3-08, C3-09, C3-13, C3-14, C3-18 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** spec 180-182: "The two attention types use **different head dimensions**. Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512). This is visible directly in the weight shapes and is not an inference from the configuration."  spec 568-569: "**`head_dim` is not the head dimension of every layer.** Full-attention layers use `global_head_dim`."  spec 188: "\| `q_proj.weight` \| `[2048, 2560]` \| `[4096, 2560]` \|"

**Implementation.** docs/verification/rust-harness.md:106-113: "Four questions decide the follow-up work: 1. Are there per-layer embedding tensors? 2. Do any layers lack `attn_k`/`attn_v` (the shared-KV signature)? 3. Are the feed-forward widths uniform across layers, or per-layer (MatFormer slicing)? 4. Is RoPE a single frequency base at full head dimension, or dual and partial?"

**Consequence.** Spec:180-182 and 568-569 make head_dim per layer type (256 sliding, 512 full; E4B q_proj [2048,2560] vs [4096,2560]), and Config carries one scalar head_dim while check_supported shape-checks only ffn_up. RH.1 never asks the question, so the reconciliation entry can be closed VERIFIED with the single-head_dim assumption untested. Question 4 touches the rotary dimension count, not projection geometry. Regraded high->medium: this is a coverage gap in a checklist with a bounded, visible blast radius; the silent-numerics risk belongs to the code site.

- upheld: Read /work/docs/verification/rust-harness.md:104-137: the four questions are verbatim as quoted and none concerns per-layer-type head dimension; grep shows line 112 is the file's only mention of "head dimension" and it maps to the rope.dimension_count vs head_dim check at /work/rust-harness/runtime/src/model/cpu.rs:265-273 (rotation extent, not projection geometry). Searched for handling elsewhere and found none that closes the gap: check_supported (cpu.rs:206-330) shape-checks only ffn_up width (234-252) and is otherwise name/metadata based, so mixed 256/512 attention shapes pass its attention checks; gguf-info's summarize() mirrors the same four questions and computes no head-dim uniformity; the linked reconciliation table in docs/inference/rust-harness-runtime-design.md omits head dim as well; RH.0-RH.13 contain no second architecture entry and rust-harness-qa.md's head_dim hits are unrelated. Partial mitigations exist but do not refute: the raw gguf-info dump prints per-tensor dims (the doc itself notes it is hundreds of lines and that the summary exists for that reason, with no instruction to diff attn_q dims), and the loader length-checks attn_q/attn_q_norm at cpu.rs:658-693 — but that is RH.2's path, while RH.1's stated signal is check_supported's verdict, which can print "[OK] supported" and scope follow-up work with head dim never asked. GGUF carries a single scalar attention.key_length (cpu.rs:89), so the spec's Trap 6 (gemma-4.md:180-182, 186-204, 568-569) is not expressible in Config, making the unasked question consequential.

### GC-089 Declared subset never names the two head dimensions (256 sliding, 512 full); no guard sees them

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `absent-undeclared` |
| Tier | `docs` |
| Site | `/work/rust-harness/README.md:100` |
| Variants | 12B, 31B, E2B, E4B |
| Claims | C1-01, C1-02, C13-06, C3-07, C3-08, C3-13 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** "The two attention types use **different head dimensions**. Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512). This is visible directly in the weight shapes and is not an inference from the configuration." (/work/docs/models/gemma-4.md:180-182); E4B "\| `q_proj.weight` \| `[2048, 2560]` \| `[4096, 2560]` \|" (:188)

**Implementation.** "Gemma-4-style forward pass (RMSNorm, GQA with QK-norm and NEOX RoPE, / interleaved sliding-window/global attention, GeGLU MLP, tied LM head) ... That pass is a strict subset / of the published Gemma 4 architecture, so `cpu::check_supported` refuses a / checkpoint carrying per-layer embeddings, a shared KV cache, MatFormer / per-layer feed-forward widths, or dual RoPE" (rust-harness/README.md:100-105). Config holds one scalar: `pub head_dim: usize` read from `head_dim: u("attention.key_length")?` (rust-harness/runtime/src/model/cpu.rs:45, 88).

**Consequence.** Spec:180-182 states sliding layers use head_dim 256 and full-attention layers global_head_dim 512, visible directly in the weight shapes; spec:188 gives q_proj [2048,2560] against [4096,2560] on E4B. Config carries one scalar head_dim (cpu.rs:45, 89) and check_supported never compares an attention tensor's shape to it, so the README's refusal list presents a subset boundary with a hole. On the E-series the per-layer-embedding guard fires first and masks it; the load stays loud via the per-tensor length check at cpu.rs:665-669, but it names a tensor rather than the feature, contradicting README:105-106.

- upheld: Reproduced every cited fact. README.md:102-106 enumerates exactly four refused features (per-layer embeddings, shared KV cache, MatFormer widths, dual RoPE) and grep shows no mention of head dimensions anywhere in the README. cpu.rs:46/:89 hold one scalar `head_dim` from `attention.key_length` (finding's "45" is an off-by-one on the struct field; the read at 89 is exact). Reading check_supported in full (cpu.rs:206-330), its only tensor-shape comparison is `ffn_up` width vs `ffn_dim`; the rope loop compares `rope.dimension_count` metadata to head_dim, never an attention tensor. `global_head_dim` and `attention.value_length` appear nowhere in the Rust tree, and gguf-info.rs:100-167 reports ffn-width uniformity, shared-KV layers, and rope/window metadata but no attention-head-dim uniformity before delegating to the same check_supported. The only thing that sees the mismatch is the load-time length closure at cpu.rs:658-671 used at :681-686 (`q_dim = n_heads * head_dim`), whose message names the tensor, not the feature — precisely the gap claimed. Spec authority holds family-wide (gemma-4.md:110-111, 180-182, 188-193, 568-569), and 31B/12B carry neither per-layer embeddings nor shared KV (:94-96), so no earlier guard masks it there. Failure is loud, never silent, which the finding already concedes.

### GC-090 The f16 KV candidate's 128k worst case of 17.5 GB counts only E4B's 35 sliding layers; the spec's shapes give 24.5 GiB

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `mismatch` |
| Tier | `docs` |
| Site | `/work/rust-harness/docs/optimization-candidates.md:84` |
| Variants | E2B, E4B |
| Claims | C1-02, C2-02, C2-06, C2-13, C3-07, C6-05 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** "Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512)." (/work/docs/models/gemma-4.md:181-182); E4B is "\| `num_key_value_heads` \| 1 \| 2 \| 8 \| 16 \|" and "\| `max_position_embeddings` \| 131072 \| 131072 \| 262144 \| 262144 \|" (:146, 153); E4B has "35 \| 7" sliding/full layers (:174)

**Implementation.** "**f16 KV cache and windowed ring buffers.** Halving KV element size cuts the / 128k-token worst case from roughly 17.5 GB to 8.7 GB, and bounding windowed / layers to their attention window (a ring buffer instead of full-sequence / storage) shrinks it further." (rust-harness/docs/optimization-candidates.md:83-86)

**Consequence.** Under spec:146/153/174/181-182 one cached E4B token costs 35 x 2 x 256 x 2 x 4 = 143,360 B on sliding layers plus 7 x 2 x 512 x 2 x 4 = 57,344 B on full layers = 200,704 B; x131072 = 26,306,281,472 B = 24.5 GiB f32, 12.25 GiB f16. The quoted 17.5 GB is exactly the sliding subtotal (143,360 x 131072), dropping the seven global_head_dim-512 layers. A reader provisioning f16 against 8.7 GB under-provisions by 40 percent. Suspected root is local, not propagated: one grep hit in the tree, and the runtime's own cache model structurally cannot express a second head dimension -- KvGeometry carries a single head_dim and bytes_per_token = 2 * n_kv_heads * head_dim * bytes_per_element (rust-harness/runtime/src/kv.rs:20-30), which is also the model behind the startup KV estimate advertised at rust-harness/README.md:132-133. The merged twin back-solves 17.5 GiB to 35 * 512, E2B's layer count with E4B's sliding kv_dim, and adds the host check.

- upheld: Reproduced the arithmetic at /work/rust-harness/docs/optimization-candidates.md:83-86. Back-solving 17.5 GiB: 17.5 GiB / 131072 tokens = 143,360 B/token; /4 B (f32) /2 (K+V) = 17,920 = 35 x 512, i.e. 35 layers x (2 kv heads x head_dim 256) — exactly E4B's sliding subtotal, with the seven global_head_dim-512 full-attention layers (spec /work/docs/models/gemma-4.md:174, 180-182, 186-193) omitted. The spec-consistent full-sequence figure is 35x2x256x2x4 + 7x2x512x2x4 = 200,704 B/token x 131072 = 24.5 GiB f32 / 12.25 GiB f16. No alternative reading reproduces 17.5: all 42 layers at uniform 256 gives 21 GiB; E2B gives 8.75 GiB (uniform) or 10.5 GiB (spec dims); honoring num_kv_shared_layers=18 gives ~12-14 GiB. I searched for handling elsewhere and found none: one grep hit in the tree, no variant qualification or caveat in the doc, and the code confirms the doc's memory model rather than correcting it — /work/rust-harness/runtime/src/model/cpu.rs:24-27 stores the full sequence for every layer, :219-229 hard-refuses shared-KV checkpoints, and /work/rust-harness/runtime/src/kv.rs:20-30 plus /work/rust-harness/harness/src/main.rs:253-257 carry a single head_dim, so the startup estimate at /work/rust-harness/README.md:132-133 cannot express the second head dimension either. Qualification that does not rescue the line: the finding's 24.5 GiB ignores num_kv_shared_layers=18 (spec:312-333), so a sharing-aware allocator would need only ~14 GiB f32 / ~7 GiB f16; but that allocator does not exist in this tree, and even the doc's own no-sharing model needs 12.25 GiB f16 against the quoted 8.7 GB.

### GC-093 check_hook_dims runs only under --check-hooks; run_harvest never calls it, so a real harvest is ungated

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `guard-ineffective` |
| Tier | `interp` |
| Site | `/work/interp/commands/harvest.py:56` |
| Variants | E2B, E4B |
| Claims | C2-10, C2-19, C5-01, C5-15 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** gemma-4.md:252-254 `1. **Per-layer input embeddings.** `hidden_size_per_layer_input` is 256 on the E-series and 0 on 12B and 31B. Without this mechanism the E-series does not run at all.`; gemma-4.md:150 `\| `hidden_size_per_layer_input` \| 256 \| 256 \| 0 \| 0 \|`

**Implementation.** interp/commands/harvest.py:56-63 `if args.check_hooks:` … `return 0` — the gate returns before line 64 `out.run(f"Harvesting activations: {model_id}")` / line 65 `manifest = run_harvest(recipe)`. `run_harvest` (interp/harvest.py:294-428) calls `validate_harvest`, `resolve_subject`, `resolve_model_meta`, `disk_preflight`, `run_lock`, and the runner, and never calls `check_hook_dims`.

**Consequence.** The gate documented at interp/harvest.py:171-176 as "the Gemma 4 gate before committing GPU-days" sits in an exclusive CLI branch that returns at harvest.py:63 before run_harvest at :65. A repo-wide grep shows check_hook_dims has exactly one production caller, this branch: run_harvest (interp/harvest.py:294-428) never invokes it, so a 300M-token overnight harvest takes the hook name straight from the registry template with no dimension assertion. docs/interpretability/investigations/p4_d3_sae_training_investigation.md:120 asserts the gate "already proved" the captured vectors are the residual stream - a claim no path inside run_harvest can produce.  Propagated to: rust-harness/runtime/src/model/cpu.rs:308. One defect: the PLE hook-dimension gate runs only under the opt-in --check-hooks flag and run_harvest never calls it. commands/harvest.py:56 defines the flag gating (ROOT of the opt-in); harvest.py:308 is the call site where the invocation is missing. One fix closes both.

- upheld: Verified at the cited site: interp/commands/harvest.py:52-65 puts check_hook_dims inside `if args.check_hooks:` and returns 0 at line 63, before `run_harvest(recipe)` at line 65. I read run_harvest in full (interp/harvest.py:294-428): it calls validate_harvest, resolve_subject, corpus_identity, resolve_model_meta, _load_progress, disk_preflight, run_lock, Watchdog and the runner — no forward pass and no tensor-width assertion. Searched for handling elsewhere and found none: validate_harvest (:59-89) checks only config fields; resolve_model_meta (:141-165) reads hidden_size from the config and cannot prove what the hooked module emits; interp/families.py:44-47 itself states "the harvest hook-dimension gate is the runtime backstop for layout drift", i.e. the registry template is unverified data; repo-wide grep shows exactly one production caller of check_hook_dims (commands/harvest.py:58), the rest being the definition, docs, and tests (tests/hardware/test_gpu.py:237 pairs it with run_harvest only in test code); Makefile:303-311 defines `harvest` and `harvest-check-hooks` as independent targets and `harvest` runs only scripts/interp/preflight.sh, which checks GPU-service concurrency alone; downstream check_capture/require_match and interp/sae/train.py:184 take d_in from the manifest's own config-derived d_model, so a wrong-width capture stays self-consistent. Only mitigation is the manual runbook ordering at docs/verification/interp.md:344-348, which is procedural, not enforced.

### GC-100 System text is folded into a user turn and the generation prompt omits the empty thought channel; both unsettled by the spec

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `needs-host` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/harness/src/template.rs:37` |
| Variants | E2B, E4B, all |
| Claims | C10-12, C10-13 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:472 "The tokenizer defines asymmetric paired markers where the bar denotes the open or close side"; :482 "A reasoning marker `<\|think\|>` and an escape token `<\|\"\|>` are also defined."

**Implementation.** template.rs:31-40 `Role::System \| Role::User \| Role::Tool \| Role::Compaction => "user",` with the comment "Gemma has no system role; system text leads the first user turn by convention"; template.rs:18 `MODEL_TURN_OPEN: &str = "<start_of_turn>model\n"`

**Consequence.** template.rs:31-40 maps Role::System, Tool, and Compaction all to "user" on the premise stated at template.rs:32 that "Gemma has no system role", and MODEL_TURN_OPEN (template.rs:18) ends the prompt with a bare model turn. The authority settles the marker strings (gemma-4.md:472-483) but not the role vocabulary, the existence of a native system turn, or the generation-prompt tail, so this cannot be graded from the spec alone. Two in-tree readings of the official template disagree with the code: training/models/families.py:76-77 records "native system turn" and :87 assistant_role "model", and docs/training/investigations/f1_sft_data_investigation.md:211-213 records that with thinking disabled the official generation prompt ends "<\|turn>model\n<\|channel>thought\n<channel\|>". If both hold, the harness mislabels the system prompt (HARNESS.md content arrives as a user turn) and omits the empty thought prefix the model was tuned to continue from - plausible wrong output, no error. Settle by reading chat_template.jinja at the pinned revision before rewriting template.rs.

- upheld: Reproduced the cited site exactly. /work/rust-harness/harness/src/template.rs:37 reads `Role::System \| Role::User \| Role::Tool \| Role::Compaction => "user",` under the comment at :32 "Gemma has no system role; system text leads the first user turn by convention", and :18 defines `MODEL_TURN_OPEN: &str = "<start_of_turn>model\n"`, appended unconditionally by render (:49) and by every real caller (agent.rs:145, chat.rs:95, compact.rs:78). Searched for the compensating handling my lens asks about and found none: (1) Role::System is constructed on the main path — main.rs:379 reads root.join("HARNESS.md") and passes it as `memory` to both entry points (:403, :423), which agent.rs:75-82 and chat.rs:34-40 wrap as Entry{role: Role::System} and hand to render_entry, so the system prompt provably renders as `<start_of_turn>user\n...`; (2) transcript.rs:113-134 `effective` folds Compaction records into Role::System entries, which then hit the same `=> "user"` arm, so the fold does not rescue the mapping; (3) no thought/channel handling exists anywhere — grep -rniE "thought\|thinking\|channel" over harness/src, harness/tests, runtime/src returns only mpsc/tokio channels and one unrelated cpu.rs comment, and generator.rs:62 PROMPT_SPECIALS lists only TURN_OPEN and TURN_CLOSE while complete() (generator.rs:65-76) only prepends BOS, rewriting no tail; (4) existing tests pin the behavior rather than guard it — agent_e2e.rs:290 asserts prompts[0].starts_with("<start_of_turn>user\nprefer small diffs<end_of_turn>\n") and chat.rs:229 the same for the memory turn. I also checked the finding's own premise about the authority: /work/docs/models/gemma-4.md:472-483 is only a marker table plus padding_side/model_max_length, and grep -Ei "system turn\|role" over that file returns just the table header at :474 and an unrelated layer_scalar line at :405, so the spec genuinely does not settle the role vocabulary or generation-prompt tail and the [needs-host] "read chat_template.jinja at the pinned revision" disposition stands.

### GC-101 gguf-info's "tool and loader cannot disagree" guarantee is false for the dual-head-dim geometry

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `guard-ineffective` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/bin/gguf-info.rs:11` |
| Variants | 12B, 31B, all |
| Claims | C1-01, C1-02, C3-07, C3-18 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** spec:180-182 "The two attention types use **different head dimensions**. Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512)."; spec:206-208 "The shapes follow `num_attention_heads * head_dim` for sliding layers and `num_attention_heads * global_head_dim` for full layers"

**Implementation.** gguf-info.rs:11-13 "The verdict calls `cpu::check_supported`, the function the loader calls, so this tool and the loader cannot disagree."; gguf-info.rs:164-166 "match Config::from_gguf(file).and_then(\|c\| cpu::check_supported(file, &c)) { Ok(()) => println!(\"[OK] CPU backend compatibility: supported\"),"

**Consequence.** check_supported (cpu.rs:206-330) inspects tensor names, ffn_up widths, rope keys and softcapping keys and never compares attention projection dims to the config; its backstop collapses blk.N. to blk.*. (cpu.rs:184-194), so 35 tensors at [2048,2560] and 7 at [4096,2560] (spec:381-388) are one consumed family. The per-tensor length checks that do catch the geometry live in CpuModel::from_gguf (cpu.rs:658-693), which gguf-info never calls. A text-only 12B or 31B GGUF therefore prints "[OK] CPU backend compatibility: supported" and then fails to load, and docs/verification/rust-harness.md:123-126 tells the operator to act on that verdict by proceeding to RH.2. On E2B/E4B the per-layer-embedding check masks it, so the reachable case is the two out-of-scope variants; medium, not high.

- upheld: Reproduced every cited step in code. /work/rust-harness/runtime/src/bin/gguf-info.rs contains the quoted guarantee at lines 11-13 and computes its verdict at line 164 as Config::from_gguf(file).and_then(\|c\| cpu::check_supported(file, &c)); the binary (read in full, 169 lines) never constructs a CpuModel. check_supported (/work/rust-harness/runtime/src/model/cpu.rs:206-330) performs exactly five checks - per_layer tensor names, attn_k/attn_v presence, ffn_up last-dim vs config.ffn_dim, a {arch}.rope.* key allowlist, and softcapping/attention.scale/logit_scale metadata keys - then a purely name-based unconsumed-tensor backstop (309-313) whose reporting collapses blk.N. to blk.*. via tensor_family (184-194). No attention projection dim is ever compared to head_dim/n_heads. The element-count checks that would catch the 256/512 dual-head-dim geometry are in CpuModel::from_gguf (646-694, q_dim = n_heads*head_dim from a single head_dim). Searched for handling elsewhere and found none: grep shows check_supported is called only at cpu.rs:650 and gguf-info.rs:164 plus tests; gguf.rs/tensor.rs/kv.rs carry no config-vs-dim validation. The repo's own test corroborates the split - a_consumed_tensor_of_the_wrong_shape_is_refused_naming_the_tensor (cpu.rs:1310-1343) routes every wrong-shape case, including blk.0.attn_q.weight at 128 vs 256 elements, through CpuModel::from_gguf, while all check_supported tests (1192-1308) mutate only names, ffn width, or metadata. The operator-facing repeat of the false guarantee is at /work/docs/verification/rust-harness.md:124-129. Caveat that does not overturn the finding: the concrete "12B/31B prints [OK]" step depends on converter metadata the spec says is uncollected for those models (gemma-4.md:60-62), and the all-variant invariants final_logit_softcapping 30.0 and dual RoPE theta (gemma-4.md:130-146) could trip earlier guards if a converter emits them; the guarantee text is nonetheless unconditional and the tool/loader gap applies to any shape mismatch. Static verification only - no cargo in the sandbox.

### GC-104 KvGeometry carries one head_dim for every layer, so the KV budget prices full-attention layers at 256 instead of global_head_dim 512

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `unrepresentable` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/kv.rs:18` |
| Variants | 12B, 31B, E2B, E4B, all |
| Claims | C1-01, C1-02, C1-11, C1-15, C10-02, C13-06, C13-08, C2-09, C3-07, C3-19, C3-20, C6-05, C6-09, C8-11, C8-12 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** L180-182: "The two attention types use **different head dimensions**. Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512). This is visible directly in the weight shapes and is not an inference from the configuration."

**Implementation.** L18-24: "/// Per-layer cache geometry, shared by every layer of a model." `pub struct KvGeometry { pub n_kv_heads: usize, pub head_dim: usize, pub bytes_per_element: usize }`; L29: `2 * self.n_kv_heads * self.head_dim * self.bytes_per_element`

**Consequence.** Spec 180-182 gives sliding layers head_dim 256 and full-attention layers global_head_dim 512, confirmed in the weight shapes (spec 189-190: k_proj/v_proj [512, 2560] against [1024, 2560] on E4B). KvGeometry has one head_dim field and one bytes_per_token(), and budget_bytes (kv.rs:34-43) applies it to both LayerKind arms; the only caller fills it from the single scalar Config::head_dim, which Config::from_gguf reads from `{arch}.attention.key_length` (cpu.rs:89), at harness/src/main.rs:253-256. The understated term is exactly the one that grows with context - sliding layers cap at the window. Arithmetic, transcribed and checked: E2B at its declared 131072 context (spec:153), n_kv_heads 1, f32 gives 7*131072*2048 + 28*512*2048 = 1,908,408,320 bytes, below the `if bytes > 2 << 30` threshold at main.rs:261, so no [WARN] prints; with 512 on the 7 full layers the true figure is 3,787,456,512 bytes and the warning should fire. Graded medium, not high: the chain through main.rs:253-261 is independently followable and the wrong number is silent rather than loud, but no Gemma 4 checkpoint reaches it today - check_supported refuses per-layer embeddings first (cpu.rs:211-217) - so the blast radius is bounded to a future backend inheriting this contract. Nothing in the crate mentions global_head_dim; this struct is where the absence belongs. The merged S26-M3-12 and S30-M4-abs-03 also carry the KV-sharing half, which is triaged separately at kv.rs:34.  Propagated to: rust-harness/harness/src/main.rs:255. One struct, one defect: KvGeometry carries a single head_dim, so full-attention layers are priced at 256 instead of 512 (2x undercount). kv.rs:18 is the definition ROOT; main.rs:255 is the caller reporting the same miscount.

- upheld: Read kv.rs directly: L18-24 defines KvGeometry with a single scalar head_dim ("shared by every layer of a model"), L28-30 has one bytes_per_token(), and L34-43 budget_bytes computes per_token once and applies it identically to both LayerKind::Global and LayerKind::SlidingWindow arms. Spec /work/docs/models/gemma-4.md L110-111 (invariants: head_dim 256, global_head_dim 512, all four variants), L180-182 and L568-569 confirm full-attention layers use 512; weight shapes at L189-190/L200-205 corroborate. Caller chain verified line-for-line: main.rs:253-256 fills head_dim from the single Config::head_dim, which cpu.rs:89 reads from {arch}.attention.key_length; main.rs:258 calls budget_bytes and main.rs:261 tests `bytes > 2 << 30`. Arithmetic reproduced independently for E2B (35 layers, period 5 -> 7 global/28 sliding, n_kv_heads 1, window 512, context 131072, f32): 7*131072*2048 + 28*512*2048 = 1,908,408,320 < 2,147,483,648 so no [WARN]; at 512 for full layers it is 3,787,456,512 > threshold and the warning should fire. Both figures match the finding. Searched for missed handling and found none: grep for global_head_dim across /work/rust-harness returns zero hits; no per-layer geometry exists anywhere (cpu.rs uses config.head_dim uniformly at L494, L672-674, L684-685, L774, L960, L993, L1036). The only near-miss guard is load-time refusal (check_supported per-layer embeddings / shared-KV, plus the tensor element-count check), which is a side effect of other unimplemented features rather than handling of the head-dim split, and the finding already discloses it as the reason for the medium grade. synth.rs:241-248 emits general.architecture "gemma4" with a single key_length, so gemma4-labeled models do reach this budget today.

### GC-106 p-RoPE guard's `if let Some(MetaValue::U32)` silently no-ops on any other type, and never runs when the key is absent

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `guard-ineffective` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/model/cpu.rs:266` |
| Variants | 12B, 31B, E2B, E4B, all |
| Claims | C13-05, C4-03, C4-05 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:233-236 "First, full-attention layers rotate only a **quarter of each head's dimensions**. With `global_head_dim` of 512, `partial_rotary_factor` of 0.25 means 128 dimensions receive rotary treatment and 384 pass through unrotated. Applying rotation to all 512 dimensions is a silent numerical error."

**Implementation.** cpu.rs:265-274 `"dimension_count" => { if let Some(MetaValue::U32(v)) = file.metadata.get(key) { ensure!( *v as usize == config.head_dim, "{key} is {v} but head_dim is {}; partial rotary embedding (p-RoPE) is not implemented", config.head_dim ); } }`

**Consequence.** The arm at cpu.rs:265-274 has no else branch, so a `{arch}.rope.dimension_count` decoded as I32 or U64 (both real MetaValue variants, gguf.rs:20-32, produced at gguf.rs:141-146) evaluates to () and the key is accepted as inert; and because the loop iterates over keys that exist, an omitted key means the ensure never runs at all. Either hole leaves `rope()` (622-631) rotating every dimension of every head, the silent numerical error spec 235-236 names. Contrast Config::from_gguf's `u()` helper (65-71), which bails with "expected u32, got {other:?}" — the rope arm is the one integer read in this module without that fallback, and the tree already records a converter type surprise (A1-R07, sliding_window_pattern arriving as a boolean array). Latent rather than live today: the head-dimension length check and the dual-theta arm refuse a real E-series file first, and the test at 1237-1251 only ever inserts U32.

- upheld: Read cpu.rs:259-280: the "dimension_count" arm is exactly as quoted, an `if let Some(MetaValue::U32(v))` with no else, inside `for key in file.metadata.keys()`. gguf.rs:19-32 defines I32 and U64 as real variants and gguf.rs:140-149 produces them for GGUF value types 5 and 10 with no per-key type policy, so a non-U32 dimension_count skips the ensure and is accepted inert; an absent key is never visited by the loop. I searched for handling elsewhere and found none: grep shows `dimension_count` appears only at cpu.rs:265 and in the test at 1237-1251; Config::from_gguf (cpu.rs:56-140) reads only rope.freq_base, so its strict `u()`/`f()` helpers never see this key; check_supported has exactly two callers (cpu.rs:650 loader, bin/gguf-info.rs:164), and gguf-info adds no validation — it prints rope keys then reports "[OK] CPU backend compatibility: supported" from the same call. The tensor-name backstop (302-328) and the softcapping/scale backstop (288-300) cannot match this metadata key. Consequence verified: rope() at 620-631 uses half = head.len()/2 and the call sites at 812-818 pass full head_dim chunks for all Q and K heads of every layer with one global freq table, i.e. full rotation, the error docs/models/gemma-4.md:233-236 names (quote verified verbatim). The test at 1236-1251 inserts only MetaValue::U32 in both legs, exercising neither hole. Caveat: no cargo in the sandbox, so this is verified by reading, not by executing a repro; the finding's own "latent not live" qualifier is accurate since a real E-series file hits the `_ => bail!` dual-theta arm first.

### GC-107 p-RoPE detection compares one dimension_count to one head_dim, a pair that describes at most one of the two regimes

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `guard-ineffective` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/model/cpu.rs:268` |
| Variants | 31B, E2B, E4B |
| Claims | C1-02, C13-05, C13-06, C4-03, C4-04, C4-05 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:234-235 "With `global_head_dim` of 512, `partial_rotary_factor` of 0.25 means 128 dimensions receive rotary treatment and 384 pass through unrotated." and docs/models/gemma-4.md:568-569 "**`head_dim` is not the head dimension of every layer.** Full-attention layers use `global_head_dim`."

**Implementation.** cpu.rs:267-272 `ensure!( *v as usize == config.head_dim, "{key} is {v} but head_dim is {}; partial rotary embedding (p-RoPE) is not implemented", config.head_dim );` where `head_dim: u("attention.key_length")?` (cpu.rs:89) is one scalar

**Consequence.** The accept condition is a single equality between two scalars (cpu.rs:267-272), while the architecture has two rotation regimes: 256 of 256 on sliding layers and 128 of 512 on full layers (spec 180-182, 233-235). A converter writing attention.key_length 256 and rope.dimension_count 256 describes the sliding regime truthfully, satisfies the equality, and leaves the partial rotation on the 7 full-attention layers undiagnosed; the opposite convention (128) makes the guard fire loudly. Which value a real GGUF carries is a host question. Blast radius is bounded on E2B/E4B, where the per-layer-embedding refusal and the head-width length checks fire first; the exposed shape is a uniform-head-width checkpoint with partial rotation.  Propagated to: rust-harness/runtime/src/bin/gguf-info.rs:164. One defect: the p-RoPE check compares a single rope.dimension_count against a single head_dim, a pair that can describe at most one of the two rotary regimes. cpu.rs:268 is the loader ROOT; gguf-info.rs:164 is the mirrored copy (README:106: 'gguf-info prints the same verdict'). The type-match no-op at cpu.rs:266 (S16-M2-C4-01) is a distinct defect and stays separate.

- upheld: Reproduced at /work/rust-harness/runtime/src/model/cpu.rs:263-274: the p-RoPE accept condition is `ensure!(*v as usize == config.head_dim, ...)`, a single equality against one scalar `head_dim` read from `attention.key_length` at cpu.rs:89. `Config` (cpu.rs:40-53) has no second head dimension and `layer_kind` (cpu.rs:143-149) distinguishes layer types only for windowing, so the pair can describe at most one of the two regimes the spec states (gemma-4.md:180-182 sliding 256 / full 512; :233-235 128 of 512 rotated; :568-569). The in-tree test `partial_rotary_dimension_count_is_refused` (cpu.rs:1237-1251) confirms `dimension_count == head_dim` is accepted as "a full rotation". Searched for handling elsewhere: a tree-wide grep for global_head_dim/partial_rotary/key_length finds no second head dim; check_supported's other guards key on tensor names or ffn_up width only, and the generic unconsumed-tensor backstop (cpu.rs:304-328) matches attn_* by name regardless of dims. The only mitigation is the tensor-length check in CpuModel::from_gguf (cpu.rs:658-694, q_norm/k_norm sized to the single head_dim), which the finding already discloses, and which does not apply to the propagated site /work/rust-harness/runtime/src/bin/gguf-info.rs:164 — that path runs only Config::from_gguf + check_supported and would print "[OK] CPU backend compatibility: supported". README:106 ("gguf-info prints the same verdict") is present verbatim.

### GC-108 Dual-RoPE refusal only sees keys under {arch}.rope.; a theta spelled elsewhere, or folded away, escapes it

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `needs-host` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/model/cpu.rs:258` |
| Variants | 12B, 31B, E2B, E4B, all |
| Claims | C1-13, C1-14, C13-04, C13-05, C13-15, C4-01, C4-05, C4-06 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:238-240 "Second, the two regimes differ in base frequency by two orders of magnitude (1000000.0 against 10000.0). A single global `rope_theta` is incorrect for this architecture regardless of which value is chosen."

**Implementation.** cpu.rs:258-262 `let rope_prefix = format!("{arch}.rope."); for key in file.metadata.keys() { let Some(suffix) = key.strip_prefix(&rope_prefix) else { continue; };` - and the only other metadata guard, cpu.rs:293, fires solely on `suffix.contains("softcapping") \|\| matches!(suffix, "attention.scale" \| "logit_scale")`

**Consequence.** Extends A1-R04, refuted in docs/verification/rust-harness-qa.md:454-466: four checkpoints built with the converter's exact key set showed the second base arrives as {arch}.rope.freq_base_swa, which the catch-all at cpu.rs:275-278 does refuse — so the wrong-numerics path is unreachable on the evidence in this tree, and the severity is capped accordingly. The open residual is that detection is entirely key-name dependent: `strip_prefix("{arch}.rope.")` (258-262) cannot see a base spelled gemma4.rope_theta (the spelling the parser's own test writes, gguf.rs:383) or gemma4.attention.rope_freq_base_swa, and it cannot see a theta the converter folds into a hardcoded rule with no key at all. Meanwhile cpu.rs:16-21 and rust-harness/README.md:103-106 state the refusal unconditionally, which is the imprecise-guarantee residual A1-R04 left open. Only the pinned converter or a real gemma4 GGUF settles the spelling.  Propagated to: rust-harness/README.md:105. One defect: the dual-RoPE refusal only inspects keys under {arch}.rope.*, so a single-theta or differently-spelled checkpoint passes silently. cpu.rs:258 is the code ROOT; README:105 advertises the same allowlist-scoped guarantee.

- upheld: cpu.rs:258-262 is verbatim as cited: `strip_prefix("{arch}.rope.")` with `continue` on non-match, so the catch-all bail at 275-278 can only fire on keys already inside that namespace. The only other metadata guard (cpu.rs:289-293) fires solely on softcapping/attention.scale/logit_scale, and the generic backstop (304-328) iterates tensors, not metadata; gguf.rs parse applies no key allowlist, and every load path in the workspace (harness/src/main.rs:188, bin/activation-capture.rs:204, gguf-info.rs:164) funnels through the same Config::from_gguf + check_supported pair, so there is no second guard. Traced escape: with gemma4.rope.freq_base present plus gemma4.attention.rope_freq_base_swa, both metadata loops pass and no tensor exists, so the file loads and rope_frequencies(head_dim, config.rope_freq_base) (cpu.rs:614, 703) applies one theta to every layer. A theta folded into converter rules with no key is likewise undetectable, and INERT_TENSORS (cpu.rs:180) whitelists rope_freqs.weight by design. The unconditional guarantee text exists as cited at cpu.rs:16-21 and README.md:103-106, and the existing test (cpu.rs:1253-1262) only exercises the in-prefix key gemma4.rope.local_freq_base. One correction that does not close the gap: Config::from_gguf:94 hard-requires {arch}.rope.freq_base, so a checkpoint whose only theta is spelled gemma4.rope_theta is refused at config load ("missing gemma4.rope.freq_base") rather than passing silently, making that particular illustration weaker than stated. Caveat: no cargo in the sandbox, so this is a read-only trace, not an executed counterexample.

### GC-109 The per-layer-embedding refusal matches 'per_layer', an HF name the GGUF converter need not preserve

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `needs-host` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/model/cpu.rs:211` |
| Variants | E2B, E4B |
| Claims | C5-01, C5-03, C5-04, C5-05, C5-06, C5-07, C5-08, C5-09, C5-15 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** "\| `model.language_model.embed_tokens_per_layer.weight` \| `[262144, 8960]` \| `[262144, 10752]` \| 1 \|" and "\| `model.language_model.layers.{L}.per_layer_input_gate.weight` \| `[256, 1536]` \| `[256, 2560]` \| per layer \|" (/work/docs/models/gemma-4.md:292, 295)

**Implementation.** "if let Some(hit) = names.iter().find(\|n\| n.contains(\"per_layer\")) { bail!(\"checkpoint uses per-layer embeddings (tensor {hit}) ...\") }" (rust-harness/runtime/src/model/cpu.rs:211-217)

**Consequence.** Spec 291-297 gives safetensors names, six of which contain per_layer; the loader matches GGUF names, which the converter rewrites (token_embd, blk.N.attn_q, ffn_up). The guard's test pins the llama.cpp Gemma-3n spelling per_layer_token_embd.weight (cpu.rs:1201-1206), not any observed Gemma 4 name. If the converted names drop the substring, the named diagnosis is skipped and the generic backstop at 304-328 still refuses the load, so this is message quality and RH.1 answer quality rather than correctness — which is why it is medium and not critical. The same question applies to gguf-info's metadata filter, which greps for per_layer and matformer keys that may not exist.  Propagated to: rust-harness/runtime/src/bin/gguf-info.rs:164, rust-harness/README.md:104, rust-harness/fixtures/reference_logits.py:176. One guard: the per-layer-embedding refusal is a bare 'per_layer' substring match on an HF name the GGUF converter need not preserve. cpu.rs:211 is the ROOT; gguf-info.rs:164 mirrors it, README:104 advertises it, and reference_logits.py:176 records that no fixture proves it fires on converter-produced names. All four resolve with the same host check.

- upheld: Read /work/rust-harness/runtime/src/model/cpu.rs:206-217: the guard is exactly `names.iter().find(\|n\| n.contains(\"per_layer\"))` applied only to GGUF tensor names, and its test (cpu.rs:1200-1207) pushes the hand-written name `per_layer_token_embd.weight` onto the synthetic `build_tiny_model` fixture, so no converter-produced name is ever exercised. I searched for missed handling and found none that restores the named diagnosis: `Config::from_gguf` (cpu.rs:56-140) reads a fixed key set and validates dimensions only; the metadata loop at 288-300 refuses only softcapping/attention.scale/logit_scale; the RoPE loop at 257-280 is prefix-scoped to `{arch}.rope.`. The generic unconsumed-tensor backstop at 302-328, reached via `from_gguf` at cpu.rs:650, does preserve correctness by refusing the load — but the auditor cites it and grades the finding on it, so it is not missed handling. Spec citations verified verbatim at /work/docs/models/gemma-4.md:292 and :295, and /work/docs/inference/rust-harness-runtime-design.md:314-319 plus the PENDING RH.1 entry (/work/docs/verification/rust-harness.md:88-140) confirm nothing in-repo pins the converter's tensor naming. Only imprecision: the substring is shared by the llama.cpp Gemma-3n GGUF spelling as well as the HF names, which the finding itself discloses.

### GC-120 SamplingParams cannot express top_k 64 and defaults temperature to 0.8 against the published 1.0

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `unrepresentable` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/sampler.rs:17` |
| Variants | E2B, E4B |
| Claims | C10-16, C10-17 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** L495: `  "top_k": 64,` (E4B `generation_config.json`, with L494 `"temperature": 1.0,` and L496 `"top_p": 0.95`)

**Implementation.** L16-21: `pub struct SamplingParams { pub temperature: f32, pub top_p: f32, pub seed: u64 }`; L23-31 `Default` yields `temperature: 0.8, top_p: 0.95, seed: 0`

**Consequence.** SamplingParams holds only temperature/top_p/seed (sampler.rs:16-21) and `top_k` appears nowhere in rust-harness (grep: zero hits, CLI flags at harness/src/main.rs:487-488 are --temperature/--top-p/--seed/--max-tokens). The pipeline is mask -> softmax at temperature -> top-p, so the checkpoint's own published recipe (spec:494-496 temperature 1.0, top_k 64, top_p 0.95) cannot be reproduced and every token outside the top 64 stays reachable whenever those 64 do not cover 0.95 of the mass. top_p 0.95 matches; the 0.8 default is undeclared. Overridable and non-silent, hence medium rather than high.  Propagated to: rust-harness/harness/src/main.rs:566. Same two values: no top_k anywhere (E4B ships top_k 64) and temperature defaulting to 0.8 against the published 1.0. sampler.rs:17 SamplingParams is the ROOT; main.rs:566 is the CLI default reporting the same pair.

- upheld: Read /work/rust-harness/runtime/src/sampler.rs:16-31 directly: SamplingParams holds exactly {temperature, top_p, seed} and Default yields temperature 0.8, top_p 0.95, seed 0 — verbatim as the finding states. Spec /work/docs/models/gemma-4.md lines 494-496 are verbatim "temperature": 1.0, "top_k": 64, "top_p": 0.95. Refutation attempts all failed: (1) grep -rn 'top_k\|top-k\|topk\|TopK' /work/rust-harness/ returns zero hits (repo-wide top_k hits are only training/ distillation cache width, unrelated); (2) sample() at sampler.rs:70-155 truncates solely via cumulative top_p (candidates.truncate(cut), line 143) with no rank bound, so tokens past rank 64 stay reachable when the top 64 miss 0.95 mass; (3) the only mask is the grammar legality mask (generator.rs:22), not a rank mask, and no caller builds one; (4) SamplingParams is never populated from checkpoint metadata — nothing in rust-harness reads generation_config.json (only training/export/merge.py:187 copies the filename), and all construction sites are test literals plus main.rs:566 ::default(); (5) 0.8 is undeclared — README:116-117 names the flags without values, and the only documented 0.8 is the unrelated --compact-at threshold. main.rs:566/USAGE 487-488 confirm the propagation site with no --top-k flag.

### GC-121 Single rope_freq_base throughout: the dual-RoPE refusal fires only if the converter names a second rope key

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `needs-host` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/synth.rs:258` |
| Variants | 12B, 31B, E2B, E4B |
| Claims | C1-13, C1-14, C13-04, C4-01, C4-06 |
| Verification | 3 verdicts recorded, 1 refutations |

**Reference.** L238-240: "the two regimes differ in base frequency by two orders of magnitude (1000000.0 against 10000.0). A single global `rope_theta` is incorrect for this architecture regardless of which value is chosen."

**Implementation.** L31/L48: `pub rope_freq_base: f32,` defaulting to `rope_freq_base: 10000.0,`; L258: `w.kv_f32("gemma4.rope.freq_base", spec.rope_freq_base);` — one base key for the whole model

**Consequence.** TinySpec (synth.rs:30, default 10000.0 at :48) and cpu::Config each hold one `rope_freq_base`, written as the single key `gemma4.rope.freq_base` at synth.rs:258. The refusal at cpu.rs:275-278 is key-name driven: it bails on any other `{arch}.rope.*` key. If the converter publishes a second base key the load is honestly refused; if it hardcodes 10000.0 for sliding layers in the graph and publishes only one base, nothing fires and every layer rotates at one base, which spec:222-224 says produces plausible logits rather than a crash. Unreachable for E2B/E4B today because the per-layer-embedding refusal (cpu.rs:211) fires first; load-bearing for 31B and once per-layer embeddings land.  Propagated to: rust-harness/runtime/src/synth.rs:258. Two hunters on the same line: synth.rs:258 writes one rope_freq_base, so no fixture, golden or determinism test can see a regime split. Same site, same defect, different framings (guard dependence vs test blindness). The numpy reference's single rope_theta (S3-M1-tests-05) is a separate implementation and stays its own finding.

- upheld: Verified against /work/docs/models/gemma-4.md directly. L238-240 reads verbatim: "Second, the two regimes differ in base frequency by two orders of magnitude (1000000.0 against 10000.0). A single global `rope_theta` is incorrect for this architecture regardless of which value is chosen." — the finding's spec_says is an exact transcription. L222-224 reads verbatim: "Gemma 4 uses **two distinct RoPE regimes within a single model**, selected by layer type. This is the highest-risk area for a from-scratch implementation, because an incorrect implementation produces plausible logits rather than a crash." — matching the finding's spec:222-224 citation in context. Scope is not over-read: the per-layer-type RoPE rows appear at L126-127 inside the section "Invariants across all four variants" (L103-107, "identical in E2B, E4B, 12B, and 31B"), and the attention schedule at L171-176 shows all four variants carry both sliding and full layers, so the [E2B/E4B/12B/31B] tag is supported; trap 4 at L565 restates it. The only RoPE entry in Open questions (L587-590) concerns the undefined scaling formula of `rope_type: proportional`, not the dual base values, so the finding does not cite an open question as settled and the spec is not weaker or hedged on the proposition asserted. Code reachability claims (cpu.rs:275-278 key-name refusal, cpu.rs:211 precedence, synth.rs:258) are outside this lens.
- upheld: Reproduced the cited code exactly. /work/rust-harness/runtime/src/synth.rs:30 declares one `pub rope_freq_base: f32`, :48 defaults it to 10000.0, and :258 writes the single key `w.kv_f32("gemma4.rope.freq_base", spec.rope_freq_base)` — even though the same builder writes `gemma4.attention.sliding_window` (:250) and `sliding_window_pattern` (:254), so the fixture has both layer kinds but cannot express two bases. /work/rust-harness/runtime/src/model/cpu.rs:51/:94 mirror the single scalar; :703 builds one `rope_freqs` table; :814 and :818 rotate Q and K with `&w.rope_freqs` inside the layer loop with no reference to `c.layer_kind(i)`, which is consulted only for the attention window start at :826-829. The sole dual-RoPE defense is the name-driven metadata guard at :254-280 (accepts `freq_base`, range-checks `dimension_count`, bails on any other `{arch}.rope.*`), exactly as the finding describes. Searched for handling elsewhere and found none: the other check_supported guards (per_layer :211, shared KV :221, MatFormer :234, softcapping/scale :288, unconsumed-tensor backstop :304) never touch the base frequency; gguf-info.rs:148-167 only prints rope keys and delegates to the same check_supported; the tests at cpu.rs:1237 and :1254 exercise the metadata guard by key insertion, not numerics; and fixtures/reference_logits.py:160 reads one `rope.freq_base` and passes it to both rope calls (:184, :192), so the golden is blind too. Spec quotes verified verbatim at /work/docs/models/gemma-4.md:222-228 and 238-240. Only counter-evidence is /work/docs/verification/rust-harness-qa.md:455-467, a prior pass that refuted the gemma3-framed equivalent because llama.cpp's converter emits `freq_base_swa`; that is unverifiable upstream/host evidence about gemma3, not gemma4, and GC-018 is explicitly scoped to the residual it leaves.
- REFUTED: Cited lines are accurate (synth.rs:258 writes one gemma4.rope.freq_base; cpu.rs:257-280 refuses only by key name), but the stated consequence is unreachable. All loads funnel through CpuModel::from_gguf -> Config::from_gguf -> check_supported -> per-tensor length checks (harness/src/main.rs:188, activation-capture.rs:204, gguf-info.rs:164). E2B/E4B are refused at cpu.rs:211 (per_layer tensors) and cpu.rs:221-231 (shared KV), which the finding itself concedes. 12B/31B are refused too: docs/models/gemma-4.md's invariant table gives head_dim 256 and global_head_dim 512 for all four variants ("The two attention types use different head dimensions... visible directly in the weight shapes"), while Config carries one head_dim from attention.key_length and from_gguf asserts values.len() == want for every attn_q/k/v tensor, so one layer family always mismatches; final_logit_softcapping 30.0 (an all-variant invariant) is additionally refused at cpu.rs:283-296. Therefore no checkpoint of any variant can load and rotate every layer at one base; the worst real outcome is a refusal with a misdiagnosed reason. The test-blindness framing describes documented intended scope (cpu.rs banner declares dual RoPE unimplemented and refused; the tiny fixture cannot express two head dims and is checked against the numpy reference, not a real Gemma 4).

### GC-122 The whole attention schedule reduces to one sliding_window_pattern integer; E2B's period of 5 is unverifiable in-tree

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `needs-host` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/synth.rs:253` |
| Variants | E2B, E4B |
| Claims | C3-01, C3-02, C3-25 |
| Verification | 3 verdicts recorded, 1 refutations |

**Reference.** L167-169: "Gemma 4 interleaves two attention types on a fixed schedule given explicitly by the `layer_types` array. Read the array. Do not compute the schedule from a stride constant: E2B uses a different period from the other three variants."

**Implementation.** L253-256: `w.kv_u32("gemma4.attention.sliding_window_pattern", spec.sliding_window_pattern as u32,);` — the fixture supplies the key that cpu.rs:93 requires (`sliding_window_pattern: u("attention.sliding_window_pattern")?`, a hard bail when missing)

**Consequence.** GGUF carries no `layer_types` array, so cpu.rs:93 derives the schedule from the single required key `attention.sliding_window_pattern` (hard bail when missing) and cpu.rs:143-148 makes every pattern-th layer global. Spec:167-169 says to read the array and not compute from a stride because E2B's period differs. Three outcomes, none decidable here: key absent (loud failure on first real load), key correct, or a converter reusing the Gemma-3 constant 6 for E2B, giving full layers 5/11/17/23/29 instead of 4/9/14/19/24/29/34 - twelve layers masked wrongly and layer 34 sliding, contradicting spec:178, with no error.

- upheld: Spec-fidelity lens only. /work/docs/models/gemma-4.md lines 167-169 (under the '## Attention' heading at 165) contain the finding's spec_says quote character-for-character: 'Gemma 4 interleaves two attention types on a fixed schedule given explicitly by the `layer_types` array. Read the array. Do not compute the schedule from a stride constant: E2B uses a different period from the other three variants.' The E2B specifics are also spec text, not auditor inference: line 173 gives 'E2B \| 35 \| 28 \| 7 \| 4, 9, 14, 19, 24, 29, 34 \| every 5th (4:1)' while lines 174-176 give every 6th for E4B/12B/31B, and line 178 states 'In every variant the final layer is a full-attention layer' (E2B index 34), which the finding cites correctly. The point is reinforced, not hedged: line 271-272, E-series trap 7, says 'E2B places a full-attention layer every fifth layer; every other variant uses every sixth. A hardcoded stride works for three of four models.' Critically, the '## Open questions' section (582-605) lists seven unresolved items (proportional rope, layer_scalar, per-layer embedding data path, E4B shared-KV boundary, 12B/31B tensor data, use_bidirectional_attention, attention_k_eq_v) and the attention schedule is NOT among them; grep for 'layer_types\|period' across all 646 lines returns only 168, 171, 173-176, 271 with no uncertainty marker. None of the three over-read patterns are present: nothing upgraded from unverified to confirmed, no single-variant claim generalized to the family, no Open Question cited as settled. One derived arithmetic slip exists but is not a spec misrepresentation: the failure_mode says a stride of 6 mis-masks 'twelve layers', but the symmetric difference between the spec's {4,9,14,19,24,29,34} and pattern-6's {5,11,17,23,29} is ten layers (29 is in both). That does not change direction or severity. Out of my lens but flagged: cpu.rs:93 reads the stride from GGUF metadata rather than hardcoding it, and (i+1)%5==0 reproduces the spec's E2B indices exactly when the value is 5 - whether that satisfies 'read the array' is a code-lens question for another verifier.
- upheld: All cited lines read as claimed. /work/rust-harness/runtime/src/synth.rs:253-256 writes gemma4.attention.sliding_window_pattern as a single u32 and writes no per-layer type array; /work/rust-harness/runtime/src/model/cpu.rs:93 obtains it through the `u` closure whose None arm is `bail!("missing {arch}.{key}")` (hard bail); cpu.rs:143-149 derives every layer's kind from that one integer via `(i + 1).is_multiple_of(self.sliding_window_pattern)`, consumed at cpu.rs:826-828 to set the attention start. I searched for the handling my lens asks about and found none: `grep -rn layer_types` hits only docs/models/gemma-4.md:168 and an agent-memory note; the sole validation is `ensure!(sliding_window_pattern > 0)` at cpu.rs:124-127; check_supported (cpu.rs:206-329) never inspects the schedule; and kv.rs:47-58, harness/src/main.rs:252 and fixtures/reference_logits.py:159 all recompute the same stride, so the numpy reference cannot catch a wrong period either. No test pins a real variant's schedule. Severity lowered to medium: the stride reproduces all four documented schedules given the right integer, and a real E2B/E4B checkpoint is refused before the schedule matters (per_layer bail at cpu.rs:211-217; element-count check at cpu.rs:658-671 versus the universal head_dim 256 / global_head_dim 512 of spec:110-111), so the "twelve layers masked wrongly with no error" outcome is contingent rather than reachable today.
- REFUTED: The stated silent consequence is unreachable for E2B/E4B. Every consumer of gemma4.attention.sliding_window_pattern (cpu.rs:143 layer_kind, main.rs:252 interleaved_layers) sits behind gguf::parse -> Config::from_gguf -> check_supported (cpu.rs:646-650; main.rs:187-188; gguf-info.rs:164). check_supported hard-bails on any tensor containing "per_layer" (cpu.rs:211-217, tested at cpu.rs:1201), on the shared-KV signature (cpu.rs:221-230), on per-layer FFN widths i.e. E2B's double-wide MLP (cpu.rs:234-252), on any second rope key i.e. the E-series dual RoPE (cpu.rs:275-279), and on any unconsumed tensor family such as layer_scalar (cpu.rs:304-328) - so no faithful E2B/E4B checkpoint ever reaches the mask. Independently, gguf.rs:117 rejects bool arrays and the repo's own verification record (docs/verification/rust-harness-qa.md, A1-R07/A1-R04) states the real converter writes this key as a boolean array, so a real gemma4 file fails in parsing before Config::from_gguf. The premise is also overstated: a per-file stride of 5 reproduces the spec's E2B schedule exactly (i+1 multiple of 5 -> 4,9,14,19,24,29,34, final layer global per spec:178) and 6 reproduces E4B/12B/31B; nothing in-tree hardcodes a stride, and TinySpec accepts any usize so a 35-layer/pattern-5 case is expressible in-tree. Only the loud outcomes (parse error, missing-key bail, check_supported refusal) are reachable; residual is a documentation note about the converter's array encoding.

### GC-129 The numpy reference reads one head dimension for every layer, so no golden can see a sliding/full head-dim split

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `unrepresentable` |
| Tier | `tests` |
| Site | `/work/rust-harness/fixtures/reference_logits.py:156` |
| Variants | all |
| Claims | C1-01, C1-02, C13-06, C3-07, C3-12, C3-17, C3-18 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:180-182 — "The two attention types use **different head dimensions**. Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512). This is visible directly in the weight shapes and is not an inference from the configuration."

**Implementation.** reference_logits.py:156 `    hd = m("attention.key_length")`, then :180 `            q = q.reshape(n_heads, hd)` and :181 `            k = k.reshape(n_kv, hd)` — one scalar `hd` used for sliding and global layers alike (the `is_global` branch at :200 selects only the mask window, never a head dimension).

**Consequence.** reference_logits.py:156 binds a single `hd = m("attention.key_length")` used at :180 and :181 for both regimes; the `is_global` branch at :200 selects only the mask window. The committed fixture pins one key (synth.rs:43, head_dim 8; written as gemma4.attention.key_length at synth.rs:248), so no fixture can express a per-layer head dimension. On real E4B layer 5's q_proj is [4096, 2560] against layer 4's [2048, 2560] (spec:381-382). Regraded high -> medium: the stated workstation outcome is a loud panic in matvec_into, not silent wrong output, and check_supported would refuse an E-series checkpoint earlier on the per_layer tensors, so the blast radius is bounded to verification coverage. Suspected root: docs/inference/rust-harness-runtime-design.md:281/295, the Gemma-3-shaped reconciliation table that never lists global_head_dim.

- upheld: Verified at the cited site: /work/rust-harness/fixtures/reference_logits.py:156 binds a single `hd = m(\"attention.key_length\")` outside the layer loop (loop starts :171), and it is used at :180 (`q.reshape(n_heads, hd)`), :181 (`k.reshape(n_kv, hd)`), :199, :202 and :211 for every layer; the `is_global` branch at :200-201 only chooses the mask start index (`first = 0 if is_global else max(0, position + 1 - window)`) and never a head dimension. Spec citation is accurate (/work/docs/models/gemma-4.md:180-182 carries the quoted sentence; :381-382 the [2048,2560] vs [4096,2560] E4B q_proj split). Fixture citation is accurate: /work/rust-harness/runtime/src/synth.rs:43 `head_dim: 8,` and :248 `w.kv_u32(\"gemma4.attention.key_length\", spec.head_dim as u32);` — one scalar key. Refutation attempts failed: repo-wide grep shows `global_head_dim` only in docs and an agent-memory note, never in .rs or .py; `attention.key_length` appears in exactly one Python file (the cited one); /work/rust-harness/fixtures/ holds no second reference or fixture; and the Rust side is equally single-scalar (cpu.rs:46 `pub head_dim: usize`, :89, :672-704 derive one q_dim/kv_dim and size q_norm/k_norm uniformly), so no golden could express a split even in principle. The only nearby guards — check_supported's per_layer/shared-KV refusals (cpu.rs:206-229) and the `ensure!(values.len() == want, ...)` tensor-size check (cpu.rs:664-668) — bound blast radius but add no coverage; the finding already accounts for them at medium.

### GC-130 The numpy reference applies one rope_theta to both attention regimes, so the cross-language check shares the backend's assumption

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `unrepresentable` |
| Tier | `tests` |
| Site | `/work/rust-harness/fixtures/reference_logits.py:160` |
| Variants | E2B, E4B, all |
| Claims | C1-02, C1-06, C1-13, C1-14, C13-04, C4-01, C4-03, C4-04, C4-06 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:238-240 — "Second, the two regimes differ in base frequency by two orders of magnitude (1000000.0 against 10000.0). A single global `rope_theta` is incorrect for this architecture regardless of which value is chosen."

**Implementation.** reference_logits.py:160 `    freq_base = m("rope.freq_base")`, passed unchanged to both the query and key rotations at :185 and :193 (`rms_norm(qh, t("attn_q_norm.weight"), eps), position, freq_base`) for every layer index `i`, sliding and global alike. The fixture carries one key, `gemma4.rope.freq_base = 10000.0`.

**Consequence.** reference_logits.py:160 reads a single `rope.freq_base` and passes it unchanged at :185 and :193 for every layer, against spec:238-240 ("A single global rope_theta is incorrect for this architecture regardless of which value is chosen"). The merged twin generalizes the same point: the golden also carries one head width and no logit cap, so C1-02, C1-06, C1-13 and C1-14 are all invisible to runtime/tests/cpu_reference.rs. Blast radius bounded because cpu.rs:259-280 refuses any {arch}.rope.* key beyond freq_base/dimension_count and docs/verification/rust-harness-qa.md:454 reports the converter emits freq_base_swa, so the real path refuses loudly. The defect is that the second implementation is not independent on the spec's own highest-risk axis (spec:222-224). Suspected root: the reconciliation table at docs/inference/rust-harness-runtime-design.md:301.

- upheld: Verified directly in /work/rust-harness/fixtures/reference_logits.py: line 160 reads a single scalar `freq_base = m(\"rope.freq_base\")` outside the layer loop (loop starts :171) and passes it unchanged to rope() for queries at :185 and keys at :193 for every layer index. The regime split at :200 (`is_global = (i + 1) % pattern == 0`) is computed after the rotation and only selects the mask start at :201, never the RoPE base; rope() at :133-143 also rotates all head dimensions, so there is no partial-rotary path either. I looked for handling elsewhere and found none that rescues the reference: parse_gguf (:57-115) raises only on unknown GGUF type codes and forward() silently ignores any extra gemma4.rope.* key, so the Python has no load-time validation; the only refusal is on the Rust side at runtime/src/model/cpu.rs:256-280 (test a_second_rope_frequency_base_is_refused at cpu.rs:1253-1262), which guards the backend loader, not the cross-language reference, and is already credited in the finding as bounding blast radius. The "caller never passes the bad input" defense also fails: the fixture comes from TinySpec::default() (bin/tiny-model.rs:19) and synth.rs:258 writes exactly one key, gemma4.rope.freq_base = 10000.0 (synth.rs:48). Notably the default spec has n_layers 2 and sliding_window_pattern 2, so the golden contains one sliding and one global layer and gives both the same theta - exactly what docs/models/gemma-4.md:238-240 declares incorrect. Merged-twin clauses (head width, logit cap, C1-02/06/13/14) were out of my lens and not assessed.

### GC-131 The only model fixture sets num_key_value_heads to 1, so no GQA head-to-KV mapping is ever exercised (extends A1-15)

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `absent-declared` |
| Tier | `tests` |
| Site | `/work/rust-harness/fixtures/reference_logits.py:161` |
| Variants | 12B, 31B, E4B |
| Claims | C13-13, C2-06, C2-07, C3-19, C3-26 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:146-147 — "\| `num_key_value_heads` \| 1 \| 2 \| 8 \| 16 \|" and "\| GQA ratio \| 8:1 (MQA) \| 4:1 \| 2:1 \| 2:1 \|"

**Implementation.** reference_logits.py:161 `    heads_per_kv = n_heads // n_kv` and :204 `                kvh = qh // heads_per_kv` — a blocked mapping. Against the fixture's `head_count = 2`, `head_count_kv = 1`, `heads_per_kv` is 2 and `kvh` is 0 for both query heads, so blocked and interleaved (`qh % n_kv`) mappings are indistinguishable.

**Consequence.** reference_logits.py:161/204 encode a blocked mapping (kvh = qh // heads_per_kv); against synth.rs:41-42 (n_heads 2, n_kv_heads 1) heads_per_kv is 2 and kvh is 0 for both heads, so blocked and interleaved mappings are indistinguishable. E4B is 8 heads over 2 KV heads (spec:146-147), where the two mappings disagree on six of eight heads and produce plausible rather than obviously broken logits. Declared honestly at docs/verification/rust-harness-qa.md:467 (A1-15), hence absent-declared, but no second fixture followed.

- upheld: Reproduced the cited site: /work/rust-harness/fixtures/reference_logits.py:161 is `heads_per_kv = n_heads // n_kv` and :204 is `kvh = qh // heads_per_kv`, with the cache sliced by that single kvh at 205-210. I parsed the committed /work/rust-harness/fixtures/tiny-gemma4.gguf metadata directly: head_count=2, head_count_kv=1, matching synth.rs:41-42, so heads_per_kv=2 and kvh=0 for both query heads, identical to an interleaved qh % n_kv. reference_logits.py hardcodes that one fixture path (line 238) and cpu_reference.rs:243 pins the fixture to TinySpec::default(), so the numpy cross-check never sees n_kv>1. Searched for missed handling: cpu.rs:129-132 validates only divisibility and n_kv_heads>0 (shape, not mapping order); cpu.rs:775/833 uses the same blocked kvh = qh / heads_per_kv, so implementation and reference share the convention; the one GQA geometry built anywhere (cpu.rs:1346-1366, n_heads 4 / n_kv_heads 2) asserts only logits length and is_finite(), which both mappings satisfy; no other model .gguf fixture exists and no gqa/repeat_kv/interleave test exists in rust-harness. Declaration A1-15 in docs/verification/rust-harness-qa.md states the same blindness verbatim. Only imprecision: the title's "never exercised" overlooks that cpu.rs:1346 does execute a 4:2 geometry, but it verifies nothing about the mapping, so the substance holds. Severity stays medium: a declared coverage gap, not an observed wrong result, since blocked matches the llama.cpp/HF convention.

### GC-133 Tier 3 tokenizer fixture recipe uses lowercase google/gemma-4-e4b with no revision pin

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `mismatch` |
| Tier | `tests` |
| Site | `/work/rust-harness/runtime/tests/eval_acceptance.rs:24` |
| Variants | E2B, E4B |
| Claims | C1-07, C1-17, C11-01, C11-04, C11-13, C8-10 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:47 — "\| E4B \| `google/gemma-4-E4B` \| `411aa17b749aa952df1359d2dcea73917a544d9a` \|"; and :98-101 — "**All four `config.json` files target unreleased development versions of `transformers`.** Treat the configuration schema as unstable, and pin the `transformers` revision that actually implements `Gemma4ForConditionalGeneration` rather than assuming a released version supports it."

**Implementation.** eval_acceptance.rs:25 `///   t = AutoTokenizer.from_pretrained('google/gemma-4-e4b'); \` — lowercase `e4b`, no `revision=` argument, and no `transformers` version named anywhere in the recipe (:22-27).

**Consequence.** One committed line carries two defects and both merged findings describe it. (a) Casing: Hugging Face repository ids are case sensitive and the spec pins google/gemma-4-E4B (spec:47); every other reference in the tree uses that casing, so this is a single-site typo, and an operator following the recipe gets a repository-not-found error at step 2 of a two-step fixture build. (b) No revision= argument and no transformers version: the recipe silently resolves whatever main points to, so the token ids baked into fixtures/wikitext2_sample.json need not be the pinned 411aa17b tokenizer's, and the perplexity comparison against the llama.cpp reference is then computed over different tokens with no signal that anything moved. The missing pin is a direct violation of the pin-then-fix-then-update rule in /AGENTS.md. Kept at medium rather than promoted: the enclosing test is #[ignore] and unimplemented!(), so the wrong value is acted on by a human, not by code.

- upheld: Read /work/rust-harness/runtime/tests/eval_acceptance.rs:22-27 directly: line 24 is `///   t = AutoTokenizer.from_pretrained('google/gemma-4-e4b'); \` — lowercase `e4b`, no `revision=` argument, and no `transformers` version named anywhere in the file. The spec at /work/docs/models/gemma-4.md:47 pins `google/gemma-4-E4B` at revision `411aa17b749aa952df1359d2dcea73917a544d9a`, and :98-101 requires pinning the transformers revision. Searched for mitigating handling and found none: `grep -rn 411aa17b /work` hits only docs/models/gemma-4.md (never rust-harness); `grep -rni revision rust-harness/` yields only an unrelated `"model_revision": "gguf"` in activation-capture.rs:303; /work/rust-harness/fixtures/ contains no wikitext2_sample.json or reference_ppl.json, so the recipe is the operator's only instruction and nothing pre-generated bypasses it; the sibling Tier 3 file cuda_acceptance.rs:29 only says "transformers: load Gemma 4 E4B" with no pin; and no README, Makefile target, or script restates the recipe correctly. Every other hub-id reference in the tree uses `google/gemma-4-E4B`/`-E2B` casing (docs/training/distillation.md:35, docs/verification/training-phase1.md:78, training/recipes/interp/harvest-smoke.yaml:17); the only lowercase occurrences are local filesystem paths. Both tests are #[ignore] with unimplemented!() bodies, so no code consumes the fixture — consistent with the finding's own medium rating.

### GC-141 Param-count drift check never runs on the E2B continual recipe and never reaches calibrate's exit code (extends KB-03)

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `guard-ineffective` |
| Tier | `training` |
| Site | `/work/training/calculator/calibrate.py:106` |
| Variants | E2B, E4B |
| Claims | C2-09, C9-05, C9-08, C9-10 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:438-439 — "Continual pretraining plans that size against the effective parameter count underestimate resident weight memory by roughly a factor of two."

**Implementation.** calibrate.py:75 declares the invariant — "model_param_count: int  # torch-reported, must match the analytic count" — but calibrate.py:105-110 refuses the recipe outright: "if recipe.memory.offload == \"zero_offload\": raise CalibrationUnsupported(...)"; the only comparison, training/cli.py:186-192, emits "out.warn(f\"Parameter count drift: model ... vs analytic ...\")" and the exit code at cli.py:212-215 is computed from `within` alone

**Consequence.** calibrate.py:75 declares 'model_param_count: int  # torch-reported, must match the analytic count' and the module docstring (calibrate.py:23-24) promises drift 'is caught immediately', but two links break that promise. (1) training/recipes/reference/continual-gemma4-e2b.yaml sets 'optimizer.name: cpu_adam' (:66) and 'offload: zero_offload' (:85), so _check_supported (calibrate.py:106-111) raises CalibrationUnsupported before a model is built; cli.py:173-175 prints the error and returns 1. The repository's only E-series continual pretraining recipe therefore never exercises the detector, so any analytic undercount on that exact recipe has no reachable check. (2) Where calibrate does run (the QLoRA recipes), the comparison at cli.py:186-193 is only out.warn, and the exit code at cli.py:239 is 'return 0 if within and not record.reserved_underpredicts_nvml else 1' - the drift booleans are absent from it, so a several-hundred-million-parameter gap prints one yellow line and the acceptance command still exits 0. Note the earlier finding cited cli.py:212-215 for the exit code; 212-215 only computes 'within'. The refusal or hard failure would belong at calibrate.py:169-174, where report.total and analytic_total are both in hand before the record is returned. Suspected root: the KB-03 fix (known-bugs.md:17) introduced the zero_offload refusal for measurement fidelity and, as a side effect, removed the only recipe class where the per-layer-embedding undercount described at gemma-4.md:435-439 would be observable. Blast radius is bounded and visible (the operator sees either a loud CalibrationUnsupported or a printed warn line), so this is a detection gap rather than silent wrong output - the wrong numbers themselves live in the components/solve findings, not here.

- upheld: Reproduced every cited line. calibrate.py:104-111 raises CalibrationUnsupported on memory.offload=="zero_offload", and calibrate() calls _check_supported at line 143 before torch init and before _build_calibration_model (line 157), so no model is built; cli.py:171-175 catches it and returns 1. training/recipes/reference/continual-gemma4-e2b.yaml sets optimizer name cpu_adam (:66) and offload: zero_offload (:85) on google/gemma-4-E2B (:27), and it is the only continual recipe on an E-series model (the other is continual-qwen3.yaml), so the shipped E-series continual recipe never reaches the detector. Where calibrate does run, cli.py:186-206 is warn-only and the exit expression at cli.py:239 is "return 0 if within and not record.reserved_underpredicts_nvml else 1", with within computed at 212-215 from allocated error alone -- neither drift boolean participates. The invariant text exists as quoted at calibrate.py:75 and the docstring at 23-24, echoed at docs/training/memory_fit.md:94-95 ("caught immediately"). Searched for handling elsewhere and found none: cli.py:186 is the only comparison of a torch-reported total against the analytic total in the whole tree; preflight.py and smoke_test.py have no param/count logic; validate.py:101 only enforces cpu_adam->zero_offload (reinforcing the refusal); Makefile:170-171 is a bare passthrough with no post-hoc gate; test_calculator.py param assertions are analytic-vs-analytic constants. The repo's own test tests/training/test_cli_handlers_memfit.py:251 is named test_param_count_drift_warns_but_still_passes and asserts rc == 0 with a drifted model_param_count, codifying the gap. Only an operator override pair (memory.offload=none plus a different optimizer) could route the continual recipe into the detector, which is a different configuration, not a repository guard.

### GC-142 ple_param_count omits per_layer_model_projection, per_layer_projection_norm and post_per_layer_input_norm

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `absent-undeclared` |
| Tier | `training` |
| Site | `/work/training/calculator/components.py:141` |
| Variants | 12B, 31B, E2B, E4B |
| Claims | C1-01, C1-02, C1-11, C1-15, C10-02, C10-03, C10-04, C10-05, C10-06, C10-07, C10-08, C10-09, C10-10, C13-08, C3-01, C3-07, C4-01, C4-03, C4-04, C4-06, C5-01, C5-04, C5-05, C5-06, C5-09, C5-12, C5-13, C5-15, C7-06, C8-11, C8-12, C8-13, C9-03 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** line 293-297: "\| `model.language_model.per_layer_model_projection.weight` \| `[8960, 1536]` \| `[10752, 2560]` \| 1 \|"  "\| `model.language_model.per_layer_projection_norm.weight` \| `[256]` \| `[256]` \| 1 \|"  "\| `model.language_model.layers.{L}.post_per_layer_input_norm.weight` \| `[1536]` \| `[2560]` \| per layer \|"  line 419: "\| Per-layer model projection \| 8960 x 1536 = 13,762,560 \| 10752 x 2560 = 27,525,120 \|"

**Implementation.** line 139-141: "table = self.ple_vocab * self.layers * self.ple_dim" / "per_layer_linears = self.layers * 2 * self.hidden * self.ple_dim" / "return table + per_layer_linears"

**Consequence.** `return table + per_layer_linears` (:141) covers only two of the tensor families the spec inventories at :291-297. Missing: per_layer_model_projection (8960*1536 = 13,762,560 on E2B, 10752*2560 = 27,525,120 on E4B, itemized at spec:419), per_layer_projection_norm ([256]), and the per-layer post_per_layer_input_norm ([hidden] per layer). Undeclared: the docstring at :132-136 enumerates the model as complete. Bounded blast radius - about 27.6 MB (E2B) / 55.3 MB (E4B) of bf16 weights, and training/runtime/offload.py:50 freezes all of them under transformer_stack, so the effect is resident frozen-weight memory rather than trainable state. Kept at medium rather than the low several auditors gave it: it is a silent undercount on a predictor whose acceptance bar is 10 percent, and it also means the analytic count can never reconcile tensor-for-tensor with a calibration run.  Propagated to: docs/training/memory_fit.md:73, rust-harness/fixtures/reference_logits.py:361. One wrong PLE parameter model (omits per_layer_model_projection and the per-layer projection norms) at code ROOT components.py:141, restated in memory_fit.md:73 and pinned by test_calculator.py:361, which asserts the omission and so must be edited with the fix. The separate two-norms-per-decoder-layer undercount (S5-M1-cal-08) stays its own finding.

- upheld: Read components.py:132-141 directly: ple_param_count() returns table (ple_vocab*layers*ple_dim = spec:292 embed_tokens_per_layer) plus per_layer_linears with factor 2 (spec:295-296 per_layer_input_gate and per_layer_projection, exactly as the docstring at :134 and the field comment at :113-114 enumerate). No term corresponds to spec:293 per_layer_model_projection, spec:294 per_layer_projection_norm, or spec:297 post_per_layer_input_norm. I checked the near-miss trap that factor 2 might cover gate + model projection (layers*hidden*ple_dim numerically equals 8960*1536 = 13,762,560 on E2B) — the docstring rules that reading out; a correct count needs factor 3 plus the norms. Handling elsewhere: grep for the three tensor names across .py/.md/.rs/.yaml hits only /work/docs/models/gemma-4.md. ple_param_count() has one consumer (param_count() at :157), which feeds weight_bytes/base_bytes (:308,:315,:318,:441,:443), solve.py:132 and calibrate.py:172; no compensating term. No caller-side escape: spec_from_hf_config (:244-256) populates ple_vocab/ple_dim from the HF fields and the only fail-loud refusal on the Gemma 4 path is enable_moe_block (:238-243); live E-series recipes exist (continual-gemma4-e2b.yaml, sft-qlora-e2b.yaml, sft-qlora-gemma-e4b.yaml, distill-e2b.yaml). calibrate.py:172/195-196 records analytic vs measured counts but only reports them, confirming the reconciliation can never close. No declared limitation: only pre-fix investigation notes mention it, and f5_distillation_investigation.md:258 asserts PLE is already modeled. Arithmetic verified: E2B missing 13,762,560+256+35*1536 = 13,816,576 params (~27.6 MB bf16), E4B 27,525,120+256+42*2560 = 27,632,896 (~55.3 MB bf16), matching the finding. Propagation verified at test_calculator.py:357-371 (pins the omission) and memory_fit.md:73.

### GC-143 activation_phase_bytes prices one uniform layer; E2B's widest layer is full-attention plus double-wide MLP

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `unrepresentable` |
| Tier | `training` |
| Site | `/work/training/calculator/components.py:408` |
| Variants | 12B, 31B, E2B, E4B |
| Claims | C1-01, C1-02, C11-06, C11-07, C11-08, C11-09, C11-11, C13-07, C3-07, C3-13, C3-16, C7-02, C7-04, C8-11 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:344 "\| 15 through 34 \| 20 \| `[12288, 1536]` \| `[1536, 12288]` \|", :199 "\| `q_proj.weight` \| `[2048, 1536]` \| `[4096, 1536]` \|", :401-404 "Each layer carries **four normalization tensors** for the transformer block (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`), the Gemma sandwich-normalization pattern, plus a fifth (`post_per_layer_input_norm`) for the per-layer embedding path."

**Implementation.** components.py:408-412 `    attn = tokens * d * (3 * spec.hidden + 2 * spec.kv_dim)\n    mlp = tokens * d * (2 * spec.hidden + 3 * spec.ffn)\n    norms = tokens * d * 2 * spec.hidden\n    flash_lse = b * spec.heads * s * 4  # fp32 softmax statistics\n    per_layer = attn + mlp + norms + flash_lse`

**Consequence.** The per-layer activation terms at :408-412 use the single `spec.ffn`, `spec.kv_dim` and `2 * spec.hidden` norms, so a stack whose layers differ in MLP width and head dimension cannot be expressed. On E2B, indices 19/24/29/34 are simultaneously full-attention (spec:173) and double-wide (spec:344). Every Gemma 4 reference recipe sets gradient_checkpointing: true, so :414-415 makes this exactly one layer's workspace and the understatement is bounded at roughly 155-250 MB per the merged estimates (larger with checkpointing off, where :418 multiplies by 35). Downgraded from high to medium: bounded and visible blast radius, and it biases solve()'s micro-batch search rather than producing an unbounded error. Same data-model root as the ffn and head_dim findings; fixing ArchitectureSpec fixes this consumer.

- upheld: Reproduced the cited site verbatim: /work/training/calculator/components.py:408-412 computes per_layer from scalar spec.ffn, spec.kv_dim and 2*spec.hidden, and :414-418 either takes it as one layer's workspace (checkpointing) or multiplies by spec.layers. ArchitectureSpec (:98-130) holds scalar layers/hidden/heads/kv_heads/ffn plus a single explicit_head_dim with kv_dim = kv_heads*head_dim, so per-layer variation is unrepresentable. spec_from_hf_config (:216-260) sets ffn=config.intermediate_size (6144 for E2B) and explicit_head_dim=config.head_dim (256), i.e. the narrow/sliding values, while gemma-4.md:344 gives layers 15-34 a 12288 MLP and :173/:199 give the seven full-attention layers (4,9,14,19,24,29,34) head_dim 512 - indices 19/24/29/34 are both. Searched for the handling the lens asks about and found none: the only fail-loud guard is the MoE refusal at :238-243, gemma4/gemma4_text are on the supported list (:85-89), and a repo-wide grep of *.py/*.yaml/*.json/*.rs for double_wide, num_kv_shared, global_head_dim, layer_types returns zero hits (concepts exist only in docs). The consumer is live: solve.py:101-122 puts per_layer directly into late_backward and thus allocated_total/fits/solve()'s search; OVERHEAD_BYTES and SAFETY_MARGIN_BYTES (solve.py:31-38) are calibrated for allocator slack and CUDA context, not for this. Independently recomputed the shortfall: 3*(12288-6144)+2*(512-256)=18944 bytes per token per dtype-byte, i.e. ~310 MB for sft-qlora-e2b (4x2048 bf16) and ~155 MB for continual-gemma4-e2b (1x4096), so the finding's 155-250 MB is conservative; all Gemma 4 reference recipes checked set gradient_checkpointing: true, matching the bounding claim.

### GC-144 head_term books no tensor for final_logit_softcapping 30.0, a vocabulary-wide activation on every step

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `absent-undeclared` |
| Tier | `training` |
| Site | `/work/training/calculator/components.py:420` |
| Variants | 12B, 31B, E2B, E4B |
| Claims | C1-06 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:115 "\| `final_logit_softcapping` \| 30.0 \|" (listed at :105-106 among values "identical in E2B, E4B, 12B, and 31B")

**Implementation.** components.py:420 `    head_term = tokens * d * spec.hidden + tokens * spec.vocab * (d + 4)`, documented at :396-398 as "(final hidden states, logits in model dtype, and the fp32 upcast the HF loss path materializes for cross-entropy)". `softcap` / `final_logit_softcapping` appear nowhere in training/calculator/ or docs/training/memory_fit.md.

**Consequence.** `head_term = tokens * d * spec.hidden + tokens * spec.vocab * (d + 4)` (:420) books exactly two vocabulary-wide tensors, documented at :395-398 as the logits and the fp32 cross-entropy upcast. The spec fixes final_logit_softcapping at 30.0 for all four variants (gemma-4.md:115), and tanh's backward retains its own output, so at least one further [tokens, vocab] tensor is live. At continual-gemma4-e2b.yaml's shape (4096 tokens, vocab 262144) that is 2.00 GB bf16 or 4.00 GB fp32 against a 1 GB SAFETY_MARGIN, and solve.py makes head_backward a peak candidate under checkpointing. Downgraded from high to medium: whether transformers applies the cap inside the retained loss path, and in which dtype, was not established from any pinned library source in this tree - the multiplier is reasoned, not reproduced. Needs a calibration measurement before it is graded higher.

- upheld: Site claims reproduce exactly. /work/training/calculator/components.py:420 is verbatim `head_term = tokens * d * spec.hidden + tokens * spec.vocab * (d + 4)` — two vocabulary-wide tensors, documented at :392-398 as the model-dtype logits plus the fp32 cross-entropy upcast, and pinned by /work/tests/training/test_calculator.py:153-171. `grep -rn softcap /work/training/calculator/ /work/docs/training/memory_fit.md` returns nothing, so the omission is undeclared. No handling elsewhere: components.py:80-90 lists `gemma4`/`gemma4_text` in HF_SUPPORTED_MODEL_TYPES and spec_from_hf_config deliberately descends text_config (only MoE is refused), so /work/training/recipes/reference/continual-gemma4-e2b.yaml (gradient_checkpointing true, micro_batch 1, seq_len 4096; vocab 262144 per docs/models/gemma-4.md:112) does reach the formula; solve.py:111-122 makes head_backward a peak candidate against SAFETY_MARGIN_BYTES = 1 GiB (:38); OVERHEAD_BYTES 2.5 GiB is documented at solve.py:31-35 as measured allocator slack plus CUDA context, not activation headroom; known-bugs.md has calculator entries KB-03/53/55/56 but none for this. Contrary-direction evidence (training/tune/distill.py:157-167 check_softcap_parity, rust-harness/runtime/src/model/cpu.rs:283-293 refusing softcapping metadata) treats the cap as a real forward-path transform. Only residual gap is the unpinned transformers source, which the finding already discloses and which justifies medium rather than high.

### GC-152 Bare LoRA target names may suffix-match audio/vision tower projections when the composite E-series model is wrapped

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `needs-host` |
| Tier | `training` |
| Site | `/work/training/models/families.py:33` |
| Variants | E2B, E4B |
| Claims | C12-10, C12-11, C12-23 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** spec 522-527: "...self-attention with a `relative_k_proj` and a `per_dim_scale` tensor of shape `[128]`. Many audio and vision linear layers carry `input_min`, `input_max`, `output_min`, and `output_max` scalar tensors alongside `linear.weight`, consistent with `use_clipped_linears: true`."

**Implementation.** training/models/families.py:33-41: "_LLAMA_LINEAGE_TARGET_MODULES = (\n    \"q_proj\",\n    \"k_proj\",\n    \"v_proj\",\n    \"o_proj\",\n    \"gate_proj\",\n    \"up_proj\",\n    \"down_proj\",\n)"

**Consequence.** PEFT resolves target_modules by dotted-suffix match over every module of the wrapped model. Every Gemma 4 QLoRA recipe uses target_modules: auto and loads through AutoModelForCausalLM against a Gemma4ForConditionalGeneration config, so any tower attention submodule literally named q_proj/k_proj/v_proj/o_proj receives an adapter alongside the text decoder's. adapter_param_count multiplies per-module cost by the text layer count only, so the F2 within-10-percent prediction would be low and the exported adapter would carry tower deltas trained on text. relative_k_proj is genuinely safe (it ends '_k_proj', not '.k_proj'), which is what the families.py:125 note covers; the plain-named tower projections are the unexamined case. The spec does not enumerate tower submodule names, so only a safetensors header settles it.

- upheld: The cited site is verbatim accurate: /work/training/models/families.py:33-41 is a bare-name tuple used by the gemma4, gemma4_text, and qwen3 specs, and nothing scopes it. /work/training/models/peft_utils.py:38-51 returns the list unchanged and :90-98 feeds it straight into the repo's only LoraConfig/get_peft_model call with no exclude_modules, layers_pattern, or tower filter; a repo-wide grep for named_modules/get_peft_model/LoraConfig finds no other injection point. Callers do supply the composite: /work/training/tune/sft.py:249-270 and /work/training/calculator/calibrate.py:124-135 load via AutoModelForCausalLM and resolve on config.model_type ("gemma4" per docs/models/gemma-4.md:85, which is registered), and every Gemma 4 QLoRA recipe pins google/gemma-4-E2B/E4B with target_modules: auto. /work/training/recipes/validate.py:235-277 validates only field domains. The estimator side matches the predicted failure: components.py:273-287 multiplies per-module cost by spec.layers taken from text_config only (components.py:232-233). The only related mechanism, cli.py:197-205, is a post-hoc [WARN] on trainable-count drift during calibrate, which detects rather than prevents. The unresolved part is solely whether the towers use plain q_proj/k_proj names, which the spec does not enumerate - exactly the [needs-host] caveat the finding already states.

### GC-154 build_scratch_model emits a LlamaConfig: silu, single rope_theta, no softcap, where the Gemma 4 block is mandated

| Field | Value |
|-------|-------|
| Severity | medium |
| Verdict | `mismatch` |
| Tier | `training` |
| Site | `/work/training/models/model.py:66` |
| Variants | E2B, E4B |
| Claims | C1-04, C1-05, C1-06, C1-08, C1-10, C5-14, C9-09, C9-10 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:113 "\| `hidden_activation` \| `gelu_pytorch_tanh` \|"; :115 "\| `final_logit_softcapping` \| 30.0 \|"

**Implementation.** training/models/model.py:66-77 "    config = LlamaConfig(  # type: ignore[call-arg]\n        num_hidden_layers=spec.layers,\n        hidden_size=spec.hidden,\n        num_attention_heads=spec.heads,\n        num_key_value_heads=spec.kv_heads,\n        intermediate_size=spec.ffn,\n        vocab_size=spec.vocab,\n        tie_word_embeddings=spec.tie_embeddings,\n        rope_theta=scratch.rope_theta if scratch is not None else 10000.0,\n        max_position_embeddings=recipe.data.seq_len,\n        attn_implementation=attention_implementation(),\n    )"

**Consequence.** docs/training/training-scale-reference.md:136-140 makes the Gemma 4 block at reduced scale a design constraint (attention layout, norm placement, activation, local/global pattern), and E2B is the stated basis for from-scratch work. The only builder constructs LlamaConfig, whose hidden_act is silu rather than gelu_pytorch_tanh (spec:113), which has no final_logit_softcapping field at all (spec:115), one rope_theta rather than the two regimes differing by 100x (spec:228-229, 238-240), no partial_rotary_factor 0.25 for full-attention layers, no sliding/full interleave, and no layer_scalar. Of the invariants LlamaConfig can express, initializer_range, rms_norm_eps, attention_bias and attention_dropout match by upstream default, so the divergence is exactly activation, softcap and RoPE. Held at medium, not promoted: the silent-wrong-output path (a Llama-block model exported as GGUF arch 'gemma' and read back by Config::from_gguf, which gates only on the name prefix) runs through a phase-4 exporter that does not exist yet, so that half of the chain is not independently followable today.

- upheld: The cited site is accurate. /work/training/models/model.py:66-77 constructs LlamaConfig with exactly the quoted fields and line 78 immediately calls AutoModelForCausalLM.from_config, so nothing mutates the config afterward (callers at training/runtime/loop.py:252,258 and training/calculator/calibrate.py:127 only move/wrap the model). LlamaConfig supplies silu and has no final_logit_softcapping field, and only one scalar rope_theta is passed, versus docs/models/gemma-4.md:113,115 (gelu_pytorch_tanh, softcap 30.0) and :228-229/:238-240 (two RoPE regimes, partial_rotary_factor 0.25). I searched for handling elsewhere and found none: build_scratch_model is the only scratch builder (calibrate.py:50 merely re-exports it); gelu_pytorch_tanh/partial_rotary/sliding_window appear in no Python file outside rust-harness/fixtures/reference_logits.py; ScratchModelConfig (training/recipes/schema.py:72-84) has no architecture-family field so a Gemma block cannot be requested; training/recipes/validate.py:70-81 checks only hf-xor-scratch and head divisibility; the three scratch recipes are explicitly Llama-shaped rather than a mitigation. docs/training/training-scale-reference.md:136-140 does state the Gemma-4-block-at-reduced-scale design constraint in bold. Supporting the finding's own choice not to promote: training/export/convert.py:79-95 gates on config.json architectures[0], which would be LlamaForCausalLM, so the GGUF would carry arch 'llama' and be rejected by the gemma-prefix gate rather than silently misread; convert.py:5-8 concedes the gap without resolving it. transformers is not installed in the sandbox, so the LlamaConfig defaults (silu, no softcap) rest on known upstream defaults rather than a live probe.

### GC-166 Reconciliation table has no row for the dual head dimension (256 sliding / 512 full), the largest attention divergence

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `absent-undeclared` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-runtime-design.md:295` |
| Variants | E2B, E4B, all |
| Claims | C1-01, C1-02, C13-06, C3-07, C3-08, C3-11, C3-12, C3-13, C4-05 |
| Verification | 3 verdicts recorded, 1 refutations |

**Reference.** docs/models/gemma-4.md:180-182: "The two attention types use **different head dimensions**. Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512). This is visible directly in the weight shapes and is not an inference from the configuration." Spec 188: "\| `q_proj.weight` \| `[2048, 2560]` \| `[4096, 2560]` \|"

**Implementation.** docs/inference/rust-harness-runtime-design.md:280-286: "The CPU reference backend ... was written against a Gemma-3-shaped specification: ... grouped-query attention with per-head QK-norm and NEOX RoPE, interleaved sliding-window and global layers ... Published descriptions of Gemma 4 E4B describe a superset of that." The table at :295-304 lists eight features and none of them is the head dimension.

**Consequence.** Spec 180-182: sliding layers use head_dim 256, full-attention layers use global_head_dim 512, visible directly in the weight shapes (spec 188: q_proj [2048,2560] against [4096,2560]). The eight-row table at :295-304 has no such row, and :285-286 calls Gemma 4 E4B "a superset" of the single-head-dim Gemma 3 shape it is not. The string global_head_dim appears nowhere in the repository outside the spec. Consequence: cpu.rs:46 holds one scalar head_dim read from attention.key_length (cpu.rs:89) and derives q_dim once (cpu.rs:673), so the 7 full-attention layers of E4B abort at the element-count check (cpu.rs:667, "tensor {name} has {} elements, the configuration implies {want}") - a loud failure the reconciliation table does not predict and check_supported cannot diagnose by name. The same scalar makes the spec's 128-of-512 partial rotation (spec 233-236) inexpressible in the p-RoPE check at cpu.rs:265-273.

- upheld: Spec-fidelity lens only. /work/docs/models/gemma-4.md:180-182 matches the finding's quote verbatim ("The two attention types use **different head dimensions**. Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512). This is visible directly in the weight shapes and is not an inference from the configuration."), and line 188 matches the quoted q_proj row `[2048, 2560]` vs `[4096, 2560]` inside the E4B table headed "with counts from the safetensors header". The claim is not hedged and is not an Open Question: it is reinforced at 206-210, 212-213, and appears as Traps item 6 (568-569, "`head_dim` is not the head dimension of every layer"), while the Open questions section (582-605) lists seven other items and not this one. Scope is not inflated: the Invariants table (110-111) declares head_dim 256 and global_head_dim 512 identical across all four variants, and tensor-level evidence exists for E2B/E4B (provenance note 57-59), matching the finding's [all/E2B/E4B] tag. The p-RoPE sub-claim is likewise settled in the spec (233-236 and Traps item 5, 128 of 512 rotated), with only the `proportional` scaling formula — which the finding does not assert — left open. grep confirms `global_head_dim` appears nowhere in the repository outside this spec. No opinion offered on the design-doc table or cpu.rs consequence chain; those are other verifiers' lenses.
- upheld: Reproduced the doc gap directly. docs/inference/rust-harness-runtime-design.md:295-304 lists exactly eight rows (AltUp, PLE, MatFormer, shared KV, dual RoPE, value norm, LAuReL, sliding window); none is head dimension, and grep for head/dimension/512/256 across the whole design doc finds no declaration of a per-layer-type head width. Lines 285-286 do frame Gemma 4 E4B as a "superset" of the enumerated single-head-dim Gemma-3 shape. The spec at docs/models/gemma-4.md:180-193 states the 256/512 split and shows q_proj [2048,2560] vs [4096,2560], q_norm [256] vs [512]. I searched for handling elsewhere and found none: cpu.rs:40-53/89 holds one scalar head_dim from attention.key_length with only positivity/divisibility/evenness validation at 102-138; check_supported (cpu.rs:206-329) diagnoses only PLE, shared KV, MatFormer ffn width, and RoPE metadata keys, plus a name-keyed unconsumed-tensor backstop with no shape check on attn_q/attn_q_norm; the length check at cpu.rs:665-669 is generic ("tensor {name} has {} elements, the configuration implies {want}") with q_dim derived once at 673; the p-RoPE guard at 265-273 compares against the single head_dim. global_head_dim appears nowhere in the repo outside the spec, and RH.1 (docs/verification/rust-harness.md:88-113) repeats the "superset" framing without declaring the split. Only imprecision: on a real E4B GGUF the PLE bail (cpu.rs:211-217) fires before the element-count abort, so the cited failure is second in queue - which does not change the undeclared divergence.
- REFUTED: The doc omission is real (rust-harness-runtime-design.md contains no occurrence of head_dim/head dimension/key_length/global_head_dim, and the :295-304 table has eight rows, none about head dimension), but the stated consequence is unreachable. In CpuModel::from_gguf the order is Config::from_gguf (cpu.rs:647), check_supported (cpu.rs:650), then the name-directed tensor loads whose element-count ensure! is cpu.rs:665-667. check_supported's FIRST test (cpu.rs:209-216) bails on any tensor name containing "per_layer", and gemma-4.md:292-297/366-380 shows every E2B and E4B checkpoint carries embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm plus three per_layer_* tensors per layer. So an E2B/E4B load is refused, by name, for per-layer embeddings before a single q_proj is length-checked - and that refusal IS predicted by the table ("Per-layer embeddings (PLE) \| Retained \| No \| High"). Even if a converter renamed PLE tensors, the generic unconsumed-family backstop later in check_supported still fires before line 667. All real entry points (harness/src/main.rs:188, runtime/src/bin/activation-capture.rs:204, runtime/src/bin/gguf-info.rs:164) go through the same guard, so no E2B/E4B path reaches the cited abort; only a hand-built GGUF with exactly the 13 layer tensors and doubled q/k/v/o widths could. The p-RoPE claim is also milder: cpu.rs:265-273 ensure!s dimension_count == head_dim, so 128-of-512 produces a loud named p-RoPE refusal (wrong reference dim in the message, not a silent pass), and "Dual RoPE" already has a table row. Gap is latent, observable only after PLE/shared-KV/dual-RoPE guards are lifted.

### GC-174 P4-D2 assigns E2B sliding_window_pattern 6 and a 5:1 cycle; E2B's period is 5 (4:1) per the layer_types array

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/interpretability/investigations/p4_d2_activation_harvesting_investigation.md:119` |
| Variants | E2B, E4B |
| Claims | C1-01, C2-02, C2-03, C2-05, C2-20, C3-01, C3-02, C3-03, C3-25 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** gemma-4.md:167-169 `Gemma 4 interleaves two attention types on a fixed schedule given explicitly by the `layer_types` array. Read the array. Do not compute the schedule from a stride constant: E2B uses a different period from the other three variants.`; gemma-4.md:173 `\| E2B \| 35 \| 28 \| 7 \| 4, 9, 14, 19, 24, 29, 34 \| every 5th (4:1) \|`

**Implementation.** p4_d2_activation_harvesting_investigation.md:119-121 `- Attention interleaves 5 sliding-window layers per 1 global layer (`sliding_window_pattern=6`); irrelevant to residual capture but relevant to choosing representative depths.`

**Consequence.** Three divergences in one bullet, in a section headed "Gemma 4 architecture facts relevant to capture" whose subject is E2B. (1) The ratio is E4B's, not E2B's (spec:173: full attention at 4, 9, 14, 19, 24, 29, 34; period 6 puts globals at 5, 11, 17, 23, 29, 35, of which only 29 coincides). (2) sliding_window_pattern is not a field the spec records anywhere; the schedule is the layer_types array. (3) spec:167-169 and :271-273 explicitly forbid deriving the schedule from a stride constant because "a hardcoded stride works for three of four models". Graded medium rather than low because the bullet offers itself for "choosing representative depths", and the harvest recipes then chose one. Suspected root is the E4B row of the secondhand table at docs/inference/rust-harness-runtime-design.md:304 applied to E2B; note that table is designated suspect and is not corroboration.  Propagated to: docs/testing/test_coverage_investigation.md:390, docs/training/investigations/f8_recipe_cards_investigation.md:137, docs/models/gemma-4.md:131. One misstated interleave period for E2B (given as a 5:1 cycle / sliding_window_pattern 6; E2B is 4:1, period 5). P4-D2:119 is the ROOT, taking the value from the same transformers class-default read as the 2304 family, and three documents restate it. Caveat recorded: p4_d1:131 phrases it as 6:1 and also miscounts E4B, so if a skeptic finds it is an independent slip rather than a restatement it should be split back out.

- upheld: Verified at the cited site: /work/docs/interpretability/investigations/p4_d2_activation_harvesting_investigation.md:119-121 reads "Attention interleaves 5 sliding-window layers per 1 global layer (`sliding_window_pattern=6`)" inside the bullet list headed at :107 "Gemma 4 architecture facts relevant to capture", whose subject is unambiguously E2B (:80 fixes google/gemma-4-E2B as the subject; :109-112 gives hidden_size=2304; the sizing table at :138 says "Gemma 4 E2B"). The authoritative spec /work/docs/models/gemma-4.md:173 gives E2B as 35 layers with full attention at 4, 9, 14, 19, 24, 29, 34 — "every 5th (4:1)" — and :167-169 plus :272-273 explicitly forbid computing the schedule from a stride constant because E2B differs from the other three variants. Grepping gemma-4.md for sliding_window_pattern returns nothing, so the third sub-claim (the field is not a spec-recorded field; the schedule is layer_types) also holds. Searched for the handling the lens asks about and found none: grep for sliding/global/attention/layer_types/interleav across the whole P4-D2 file returns only lines 119-120, so there is no later correction or qualification; the one hedge at :110-112 is scoped to hidden_size/num_hidden_layers/vocab_size "before sizing buffers", not the attention schedule; and no downstream guard can exist because the deliverable is greenfield (:46 "No reference to CacheActivationsRunner exists anywhere in the repository yet"; /work/scripts/interp/ contains no harvest.py). One part of the finding's rationale does fail: the medium grade rests on "the harvest recipes then chose one", but /work/training/recipes/interp/harvest-smoke.yaml:18 and harvest-false-fact.yaml:22 pick hook_layers [15] from "50 percent depth of a 30-layer stack" — a layer-count issue, not the attention cycle — and layer 15 is a sliding layer under both period 5 and period 6, so the misstated period propagated into no value. The divergence is real and unmitigated but doc-only.

### GC-185 12B sits in the training ladder, but gemma4_unified_text is outside the memory-fit calculator's supported model types

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `absent-undeclared` |
| Tier | `docs` |
| Site | `/work/docs/training/training-scale-reference.md:103` |
| Variants | 12B |
| Claims | C11-08, C11-10, C11-11, C13-07 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** spec 86: "\| Unified \| 12B \| `Gemma4UnifiedForConditionalGeneration` \| `gemma4_unified_text` \|"  spec 89-92: "The 12B model is a separate implementation, not a size variant. ... Any dispatch that matches on `model_type == \"gemma4_text\"` silently excludes 12B."

**Implementation.** docs/training/training-scale-reference.md:103: "\| 12B \| offload, slow \| offload, slow \| offload or PEFT \| days-scale campaigns; PEFT for fast iteration \|", while docs/training/memory_fit.md:72 lists only "(`model_type` `gemma4`/`gemma4_text`, the Phase 1 primary family)" as supported and :71 states "Unknown architectures fail loudly rather than guess."

**Consequence.** docs/training/training-scale-reference.md:103 plans offload and PEFT campaigns for 12B, but spec:86 gives 12B the text model_type gemma4_unified_text (top level gemma4_unified) and spec:89-92 warns that "any dispatch that matches on model_type == \"gemma4_text\" silently excludes 12B". HF_SUPPORTED_MODEL_TYPES (training/calculator/components.py:80-90) lists gemma4 and gemma4_text and no unified variant, so spec_from_hf_config raises UnknownArchitectureError at components.py:226-231. The refusal is loud and correct - nothing is silently mispriced - but no document in the training tier records that a variant the ladder plans campaigns for cannot be sized by the preflight the entry points run first; docs/training/memory_fit.md:71-76 names only gemma4/gemma4_text as supported and never mentions the unified class.  Propagated to: docs/interpretability/interp-scale-reference.md:40. One defect: a scale-reference ladder plans work on 12B, whose gemma4_unified_text model_type both Python tiers refuse by name. training-scale-reference.md:103 is the ROOT ladder; the interp ladder mirrors its rungs (see the paired lineage-04 findings) and inherits the same unreachable entry.

- upheld: All three legs reproduce. /work/docs/training/training-scale-reference.md:103 is verbatim the quoted 12B row, and :93 scopes the table to the Gemma 4 dense variants while :87-89 explicitly feeds the ladder into memory_fit.md. /work/docs/models/gemma-4.md:86 and :89-92 give 12B `Gemma4UnifiedForConditionalGeneration` / top-level `gemma4_unified` / text `gemma4_unified_text` and warn that `gemma4_text` dispatch excludes it (trap 7 at :570); 31B by contrast is `gemma4_text`, so 12B is the only rung missed. /work/training/calculator/components.py:80-90 omits every unified variant and :226-231 raises UnknownArchitectureError; the text_config descent at :232-234 is gated on the literal "gemma4", so neither `gemma4_unified` nor `gemma4_unified_text` can reach it. I searched for the handling that would refute this: `grep -rn gemma4_unified` over the whole tree hits only docs/models/gemma-4.md (86, 90, 546) - no guard, alias, or normalization anywhere; /work/training/models/families.py:120-152 registers only gemma4/gemma4_text/qwen3 and fails the same way; no training-tier doc records the exclusion (memory_fit.md:71-76 names only gemma4/gemma4_text/qwen3). The calculator is also not behind an unreachable caller: /work/training/tune/sft.py:163-181 check_memory_fit calls estimate -> spec_from_recipe -> spec_from_hf_config, invoked at /work/training/cli.py:306 and :374 and /work/training/tune/distill.py:518, and --force only downgrades a fit failure, not the exception. Severity downgraded to low: the document's own preamble (:5-9) says "Nothing in this document is scheduled work", the ladder arithmetic is parameter-count based and model_type-independent, the refusal is loud with nothing mispriced, and no 12B recipe exists under training/recipes.

### GC-187 No verification pass asserts the cross-variant invariants; RH.1's four questions cover none of them

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `absent-undeclared` |
| Tier | `docs` |
| Site | `/work/docs/verification/rust-harness.md:106` |
| Variants | 12B, 31B, E2B, E4B |
| Claims | C1-02, C1-03, C1-06, C1-16 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** docs/models/gemma-4.md:105-106 "These values are identical in E2B, E4B, 12B, and 31B. They are safe to treat as constants, and a verification pass should assert them rather than read them per model."

**Implementation.** docs/verification/rust-harness.md:106-113 "Four questions decide the follow-up work:\n\n1. Are there per-layer embedding tensors?\n2. Do any layers lack `attn_k`/`attn_v` (the shared-KV signature)?\n3. Are the feed-forward widths uniform across layers, or per-layer\n   (MatFormer slicing)?\n4. Is RoPE a single frequency base at full head dimension, or dual and\n   partial?"

**Consequence.** spec:105-106 asks a verification pass to assert the invariant table (head_dim 256, global_head_dim 512, vocab 262144, rms_norm_eps 1e-06, final_logit_softcapping 30.0, tie_word_embeddings true, the two RoPE regimes, bos 2 / eos 1 / pad 0). No test, preflight, or runbook step in the tree compares a loaded config or GGUF against any of them, and RH.1 - the only host-side entry that touches a real artifact - asks four tensor-level questions instead. Distinct from S11-M1-verif-07, which is about the per-layer head dimension specifically; this is the whole invariant table.  Propagated to: rust-harness/runtime/src/model/cpu.rs:102, rust-harness/runtime/src/bin/gguf-info.rs:74, rust-harness/runtime/tests/cpu_reference.rs:22. One missing artifact reported from four angles: nothing in the rust tier asserts the spec's cross-variant invariants (head_dim, vocab_size, rms_norm_eps read per model instead). The runbook entry carries the group because that is where the pass would be declared and RH.1's four questions cover none of them. tests/training/test_calculator.py:317 (S9-M1-tests-11) is left out: it is the Python tier and also carries a fixture-field claim.

- upheld: Site text at /work/docs/verification/rust-harness.md:104-113 is quoted verbatim, and I found no handling elsewhere for most of the table: Config::from_gguf (/work/rust-harness/runtime/src/model/cpu.rs:79-137) reads vocab_size, head_dim (attention.key_length) and rms_eps per model with only positivity/divisibility/evenness guards; tokenizer.rs:129-130 reads bos/eos from metadata (the synth.rs:264-265 fixture even writes them reversed, bos 1 / eos 2, unchallenged); gguf-info.rs only dumps metadata and delegates to check_supported; rust-harness-qa.md Phase Q2 "engine invariants" is about prefill/budget/sampler/grammar, not config constants; training/commands/preflight.py and training/export/preflight.py mention none of the fields. Correction: the finding's clause "RH.1's four questions cover none of them" is wrong for the RoPE rows - question 4 asks exactly whether RoPE is dual and partial, and cpu.rs:254-300 (the check_supported verdict RH.1 prints) hard-errors on rope.dimension_count != head_dim, on any other rope.* key, and on any *softcapping* key, which covers the invariant table's two RoPE rows and final_logit_softcapping. The genuinely unasserted residue is head_dim 256, global_head_dim 512, vocab_size 262144, rms_norm_eps 1e-06, tie_word_embeddings, bos 2 / eos 1 / pad 0, and the spec sentence is advisory, so the impact is lower than medium.

### GC-191 interp hook registry pins gemma4 to model.layers.{n}; E-series decoders live at model.language_model.layers.{n}

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `mismatch` |
| Tier | `interp` |
| Site | `/work/interp/families.py:47` |
| Variants | E2B, E4B |
| Claims | C11-07, C11-12, C2-09, C3-07, C5-03, C5-04, C5-05, C5-07, C5-08, C5-09, C5-15, C6-05, C6-09, C8-01, C8-02, C8-03, C8-09 |
| Verification | 3 verdicts recorded, 1 refutations |

**Reference.** docs/models/gemma-4.md:370 `\| 42 \| `model.language_model.layers.{L}.input_layernorm.weight` \| `[2560]` \|`; docs/models/gemma-4.md:85 `\| E-series \| E2B, E4B \| `Gemma4ForConditionalGeneration` \| `gemma4_text` \|`

**Implementation.** interp/families.py:47 `_FLAT_DECODER_TEMPLATE = "model.layers.{layer}"` (docstring line 13: "Hook paths are HF named-module paths (model.layers.{n})"), registered for both `gemma4` and `gemma4_text` at lines 52-53

**Consequence.** Safetensors keys mirror module paths, and every E-series language-model tensor in spec:365-392 carries the model.language_model. prefix under Gemma4ForConditionalGeneration (spec:85). _FLAT_DECODER_TEMPLATE is registered for both gemma4 and gemma4_text, so resolve_submodule (families.py:100-109) walks getattr(root,'model') then getattr(node,'layers') and raises. Three consumers act on the dead path: interp/harvest.py:200 (AttributeError converted to a named HarvestError at :205-209 - the one declared backstop), interp/harvest.py:372/407 which hand the same string to SAELens and persist it in the manifest, and interp/probes/capture.py:102, which has no catch and lets a raw AttributeError escape an nnsight trace. interp/sae/evaluate.py:138 resolves the same way. The registry comment at families.py:44-46 asserts the flat layout for every registered family, which is what makes the mismatch invisible on inspection; families.py:78-79 already knows the config is composite but unwraps only the config, never the module path. High, not critical: loud on a reachable input. Residual host question - whether AutoModelForCausalLM returns the composite or a text-only wrapper - decides the failure shape, not whether the registry disagrees with the spec.  Propagated to: training/recipes/schema.py:406, rust-harness/runtime/tests/cpu_reference.rs:35, training/recipes/interp/sae-false-fact.yaml:4. One wrong module path (model.layers.{n}; E-series decoders are model.language_model.layers.{L}). interp/families.py:47 _FLAT_DECODER_TEMPLATE is the ROOT of record - harvest.py:11 and schema.py:406 both cite the registry layout - with copies in schema.py:406, tests/interp/test_families.py:35 and sae-false-fact.yaml:4.

- REFUTED: Both quotes are verbatim accurate (gemma-4.md:370 tensor name with the model.language_model. prefix; :85 E-series / Gemma4ForConditionalGeneration / gemma4_text), but the spec is narrower than the finding's use of it. The section at :358 is a "Tensor inventory ... from the safetensors header", and the reproduction recipe at :630-643 derives it by parsing the safetensors JSON header - these are checkpoint keys. The spec contains no occurrence of "hook", "named-module", "module path", "state_dict", "AutoModel", or "from_pretrained", and never states which class is instantiated, so it says nothing about the runtime module tree that resolve_submodule walks. "Safetensors keys mirror module paths" is the finding's own premise, not spec text. Compounding this, the cited column is headed "Text `model_type`" (:83) and :138 states values come from the text_config block, while interp/families.py:5-6 keys the registry "by HF config model_type after the text-config unwrap" and documents the template as "the flat text-model decoder layout" - a model built from a gemma4_text config exposes model.layers.{n} with no prefix, the prefix existing only because the text model is nested in the composite. The finding concedes this residual load-time question and then dismisses it, which is precisely the over-read. Nothing in Traps (:553-581) or Open questions (:582-605) closes the gap.
- upheld: Verified verbatim at /work/interp/families.py:47 (`_FLAT_DECODER_TEMPLATE = "model.layers.{layer}"`), registered for both `gemma4` and `gemma4_text` at lines 52-53, with the flat-layout assertion in the docstring (line 13) and registry comment (44-46), and `resolve_submodule` (100-109) doing a plain getattr walk that raises on a composite root. Spec /work/docs/models/gemma-4.md:85 gives E2B/E4B `Gemma4ForConditionalGeneration` and lines 291-297/365-392 put every language-model tensor under `model.language_model.`. I searched for handling the auditor missed and found none — instead the guard is weaker than credited: `check_hook_dims` (harvest.py:168-224, AttributeError->HarvestError at 201-209) is invoked only under `--check-hooks` in /work/interp/commands/harvest.py; `run_harvest` (harvest.py:354-407) never calls it and feeds the hook string straight to SAELens and the manifest. /work/interp/probes/capture.py:100-103 and /work/interp/sae/evaluate.py:138 have no catch at all. No hook-path override exists in the recipe schema (HarvestConfig, schema.py:417-428 exposes hook_layers only), and the problematic input is reachable from a shipped recipe (/work/training/recipes/harvest-smoke.yaml:17 `model_id: google/gemma-4-E2B`). The only open item is the host-side question of whether AutoModelForCausalLM/nnsight returns the composite or a text-only wrapper (transformers is not installed in the sandbox), which the finding itself scoped as affecting failure shape, not the registry/spec disagreement.
- upheld: Reachable from real E2B entry points, and the consequence holds (if anything it is worse than stated). Chain reproduced: /work/training/recipes/interp/harvest-smoke.yaml:17 pins model_id google/gemma-4-E2B with hook_layers [15]; /work/docs/verification/interp.md:346-347 prescribes `make harvest-check-hooks` and `make harvest` on that recipe; Makefile:303-311 -> interp/cli.py -> interp/commands/harvest.py:58/64. resolve_model_meta (harvest.py:141-165) unwraps the composite config to model_type gemma4_text, which IS registered (families.py:53), so it emits "model.layers.15". I reproduced resolve_submodule against a stand-in tree built from docs/models/gemma-4.md:365-392 (model.language_model.layers.{L}): AttributeError: object has no attribute 'layers'. torch get_submodule raises the same class, so harvest.py:201 does convert to HarvestError. Two corrections, both away from "milder": (1) run_harvest (harvest.py:294-428) never calls check_hook_dims - the declared backstop only runs under the opt-in --check-hooks flag - so the default `make harvest` path hands the dead hook to SAELens at :372 and the resulting exception is not in commands/harvest.py's catch tuple (HarvestError, ManifestError, TrainLockError, RecipeLoadError, OSError) -> raw traceback; (2) probe-capture is documented with --model google/gemma-4-E2B at docs/verification/interp.md:138 and catches only (CaptureError, OSError), so capture.py:102 escapes as a raw AttributeError. Overstated sub-claims, not severity-bearing: the manifest write at harvest.py:407 and interp/sae/evaluate.py:138 both require a completed harvest, which cannot complete with a dead hook, so the bad string is not actually persisted or re-consumed for E2B. No in-repo test refutes it - all fixtures use a flat stand-in (tests/interp/test_families.py:87, tests/interp/test_harvest_runner_branches.py:247) and "language_model" appears nowhere in /work outside docs/models/gemma-4.md. Residual host question (which class AutoModelForCausalLM returns) changes the error text, not reachability; host command to settle: python -c "from transformers import AutoModelForCausalLM as A; m=A.from_pretrained('google/gemma-4-E2B', revision='d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f', torch_dtype='auto'); print(type(m)); print([n for n,_ in m.named_modules()][:15]); print(m.get_submodule('model.layers.15'))".

**Resolution — upheld and fixed.** The residual host question is settled. Enumerating
`named_modules()` of a meta-device model built from the E2B configuration gives
`model.vision_tower` and `model.language_model` as siblings under `model`, with the decoder at
`model.language_model.layers.{n}` and no `model.layers` attribute. `AutoModelForCausalLM`
returns the composite, so the flat template raised `AttributeError` on every E-series
checkpoint and the tier could not harvest from these weights at all.

The REFUTED verdict was right on its own terms and did not change the outcome: it objected that
the tensor inventory is a state-dict namespace and therefore cannot establish a module path.
Correct — and the module path had to be observed rather than inferred. It agrees.

Fixed by making the hook path follow the loaded config rather than the family:
`HookFamilySpec` gained `composite_hook_template`, and `family_from_config`
(`interp/families.py:99-119`) applies it when the config carried a `text_config`. A family with
no composite template now raises `UnknownHookFamilyError` on a composite config instead of
falling back to the flat path. Regression tests at `tests/interp/test_families.py:67-93`,
demonstrated failing against the prior code with `- model.language_model.layers.0 /
+ model.layers.0`. The three propagated prose copies are corrected (`interp/harvest.py:10-13`,
`training/recipes/schema.py:405-408`, `training/recipes/interp/sae-false-fact.yaml:4-5`);
`rust-harness/runtime/tests/cpu_reference.rs:35` carries no hook path and was a false positive.

Not fixed here, and still open: `check_hook_dims` remains opt-in and `run_harvest` never calls
it, so the backstop this finding credited is still weaker than it reads. See
`/docs/interpretability/gemma-4-interp.md`.

### GC-205 Nothing verifies that PROMPT_SPECIALS resolve to real vocabulary pieces, so an unresolvable turn marker silently byte-encodes

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `absent-undeclared` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/harness/src/generator.rs:62` |
| Variants | E2B, E4B |
| Claims | C1-03, C10-11, C10-12, C10-13 |
| Verification | 3 verdicts recorded, 1 refutations |

**Reference.** docs/models/gemma-4.md:472-480 — "The tokenizer defines asymmetric paired markers where the bar denotes the open or close side: \| Turn \| `<\\|turn>` \| `<turn\\|>` \|"

**Implementation.** generator.rs:62 — "const PROMPT_SPECIALS: &[&str] = &[crate::template::TURN_OPEN, crate::template::TURN_CLOSE];" with template.rs:15-18 — "pub const TURN_OPEN: &str = \"<start_of_turn>\"; pub const TURN_CLOSE: &str = \"<end_of_turn>\"; pub const MODEL_TURN_OPEN: &str = \"<start_of_turn>model\\n\";"

**Consequence.** generator.rs:62 declares PROMPT_SPECIALS from the two template constants and generator.rs:67 passes them to encode_prompt. special_at (tokenizer.rs:322-338) returns None for any literal absent from the vocabulary and encode_inner (tokenizer.rs:296-316) then byte-encodes it through encode_span, with no error at load, at encode, or at generate. The natural site for the missing check is the SessionGenerator constructor (generator.rs:35-38), where the vocabulary view is already materialized and each PROMPT_SPECIALS entry could be required to resolve to a control or user-defined piece. Independent of whether the constants are right - any converter that renames or omits a marker degrades silently - which is why it is a separate finding from S2-M1-harness-01; graded high rather than critical because realizing the wrong output requires the constants or the export to be wrong, and that case is already carried at critical.

- upheld: Spec-fidelity lens only. /work/docs/models/gemma-4.md:472 and :476 read verbatim as the finding quotes them (verified byte-for-byte with cat -A), including the markdown pipe escapes: "The tokenizer defines asymmetric paired markers where the bar denotes the open or close side" with the Turn row "\| Turn \| `<\\|turn>` \| `<turn\\|>` \|". The statement is not hedged: the Open questions section (582-605) lists rope `proportional`, `layer_scalar`, the per-layer embedding path, the E4B shared-KV boundary, 12B/31B tensor data, `use_bidirectional_attention`, and `attention_k_eq_v` — the tokenizer marker table is in none of them, and the only tokenizer item in Traps (553-581) is #8 on eos/bos ordering. The author uses "(not stated)" at lines 451-452 in the same section where uncertainty was intended, so the unhedged phrasing at 472 is deliberate. Scope is not over-read: the marker table carries no variant attribution, unlike the generation-config block at 486 which is explicitly "from E4B generation_config.json", so applying it to E2B/E4B narrows rather than broadens. The finding also does not attribute a validation mandate to the spec — its spec_says field quotes only the marker table, and the "nothing verifies" claim is argued at code level (generator.rs:62/67, tokenizer.rs:296-338). Corroborating: grep for "turn" across the spec yields only line 476 as a marker reference, and `<start_of_turn>` (the literal behind template.rs:15 TURN_OPEN) appears nowhere in the spec; lines 466-470 further warn that trimming the vocabulary "breaks ... tokenizer round-tripping". Only nit: the finding reproduces markdown escapes rather than the token literals `<\|turn>`/`<turn\|>`, which is a faithful quote but could mislead a skimming reader; cosmetic, not refuting.
- upheld: Reproduced the full chain and found no guard anywhere. generator.rs:62 declares PROMPT_SPECIALS from template.rs TURN_OPEN/TURN_CLOSE exactly as quoted, and passes it to tokenizer.encode_prompt at generator.rs:67 (complete) and :83 (count_tokens). encode_prompt -> encode_inner (tokenizer.rs:296-316) -> special_at (322-338): special_at returns None whenever the literal is not a key of self.specials, and encode_inner then merely advances pos and hands the span to encode_span; there is no Err and no warning. Searched for the handling the finding says is missing: SessionGenerator::new (generator.rs:29-46) materializes the whole piece list for Vocab but checks nothing; Vocab::new (grammar.rs:263-266) asserts only eos < tokens.len(); Tokenizer::from_gguf (tokenizer.rs:98-137) validates array-length agreement and bos/eos range only; Tokenizer::new (159-214) builds the specials map without requiring any member; the sole production caller main.rs:302-303 loads the tokenizer and constructs the generator with no preflight. A repo-wide grep for PROMPT_SPECIALS/encode_prompt/special_at returns only these sites plus tests. The only error encode_span can raise is 'no tokenization covers byte N of the input span' (tokenizer.rs:381-383), which is about missing byte-fallback coverage, not allowlist resolution. The tests codify the silent path rather than catching it: tokenizer.rs:642 a_vocabulary_without_special_pieces_encodes_exactly_as_before asserts Ok for the absent-piece case, and generator.rs:126-145 (the test literally named the_templates_own_delimiters_still_encode_as_single_ids) admits in its comment 'The fixture has no chat-delimiter piece, so both paths agree here' and then asserts only listed.len() <= unlisted.len() — trivially true with zero specials resolved, so the test passes in exactly the failure state described. Only quibble, not a refutation: with a real Gemma vocabulary an unresolved marker would more likely segment into normal sub-word pieces than into <0xNN> byte tokens, so 'byte-encodes' is loose wording; the substantive claim (no error at load, encode, or generate) holds.
- REFUTED: The mechanism is real but unreachable for the scoped variants. Verified: generator.rs:62 passes PROMPT_SPECIALS to encode_prompt (67, 83); tokenizer.rs:322-338 special_at returns None for an absent literal and encode_inner (296-316) byte/piece-encodes it with no error; SessionGenerator::new (29-46) validates nothing, and grep shows no other validation site. However, the only production construction is main.rs:303, preceded at main.rs:188 by CpuModel::from_gguf -> check_supported (cpu.rs:206-217), which bails on any tensor name containing "per_layer". docs/models/gemma-4.md:150,252-254 makes per-layer input embeddings definitional for E2B and E4B (hidden_size_per_layer_input 256, 0 on 12B/31B), and two further independent refusals fire on the same checkpoints: the shared-KV signature (num_kv_shared_layers 20/18, doc:149; cpu.rs:221-230) and dual RoPE (doc:226-240; cpu.rs:257-279, pinned by the test at cpu.rs:1253-1262). The tokenizer is only constructed at main.rs:302, after that bail, and main.rs has no mock/replay fallback (main.rs:160; --record still wraps a loaded CpuModel). So an E2B/E4B run fails loudly at load with "checkpoint uses per-layer embeddings ... does not implement" and never reaches encode_prompt with a Gemma 4 vocabulary. All in-tree exercises (cpu_smoke.rs, generator_e2e.rs, the generator unit tests) use synthetic delimiter-free vocabularies with synthetic weights. The stated consequence -- silent byte-encoding with no error anywhere -- is therefore not what a reader would observe today; it is a latent gap that goes live only when a backend accepting a real Gemma 4 GGUF lands (the CUDA phase A seam, main.rs:6-7).

### GC-206 Prompt-injection sanitizer replaces only the Gemma 3 literals, so hostile tool output can fabricate Gemma 4 turns

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `guard-ineffective` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/harness/src/template.rs:25` |
| Variants | E2B, E4B, all |
| Claims | C10-12 |
| Verification | 3 verdicts recorded, 1 refutations |

**Reference.** docs/models/gemma-4.md:476-480 "\| Turn \| `<\\|turn>` \| `<turn\\|>` \| ... \| Tool call \| `<\\|tool_call>` \| `<tool_call\\|>` \|"

**Implementation.** template.rs:23-27 `fn sanitize(content: &str) -> String { content.replace(TURN_OPEN, "[start_of_turn]").replace(TURN_CLOSE, "[end_of_turn]") }`

**Consequence.** sanitize (template.rs:23-27) rewrites TURN_OPEN and TURN_CLOSE only, and template.rs:20-22 states its purpose is to stop file or command output from fabricating turns "including injected tool calls the agent would then execute". On Gemma 4 the markers are <\|turn>/<turn\|> and <\|tool_call>/<tool_call\|> (gemma-4.md:476-478), none of which the two replace calls touch, so a tool result carrying "harmless<turn\|>\n<\|turn>model\n{...}" reaches the prompt verbatim and whatever the model then emits is fed to ToolCall::parse (agent.rs:151). Graded high rather than critical because escalation to a trained control id is blocked independently: generator.rs:62 allowlists only the two constants and special_at resolves nothing outside the allowlist, so the exposure is text-level persuasion, not id-level turn forgery. The regression test that claims the property, turn_delimiters_in_content_cannot_fabricate_turns (template.rs:112-127), injects the Gemma 3 markers and therefore passes forever. Same root as S2-M1-harness-01.

- upheld: Spec-fidelity lens only. Read /work/docs/models/gemma-4.md:441-484 (Tokenizer and special tokens) plus 553-605 (Traps, Open questions). Line 472 states unqualifiedly "The tokenizer defines asymmetric paired markers where the bar denotes the open or close side:", and the table at 474-480 lists Turn `<\|turn>`/`<turn\|>` (476) and Tool call `<\|tool_call>`/`<tool_call\|>` (478) — a verbatim match to the finding's quote, at the correct cited line range 476-480. Checked for the standard over-reading failures and found none: (1) no hedging — grep for unverified/uncertain/assumed/open question hits only lines 243, 333, 406, 561 and the Open questions block, none of which touches the turn or tool-call markers, so this is settled text and not an Open Question; (2) no variant over-generalization — the marker table carries no per-variant qualifier while the very next paragraph (486) explicitly narrows to "from E4B generation_config.json", showing the author distinguishes scope deliberately, so the finding's already-hedged all/E2B/E4B tag does not over-claim; (3) the spec affirmatively contradicts the Gemma 3 literals the code uses — grep -i "start_of_turn" over the whole file returns zero hits, and lines 9-10 warn that Gemma 1-3 knowledge "does not transfer reliably". The only residual weakness is that the spec defines the marker strings and open/close polarity without laying out a chat-template grammar, but the finding's spec claim is exactly that narrow point and does not lean on the spec for template structure. Other lenses (code behavior of sanitize, the test, severity grading) were out of scope and not assessed.
- upheld: Read /work/rust-harness/harness/src/template.rs: lines 15-16 hardcode TURN_OPEN/TURN_CLOSE as the Gemma 3 literals <start_of_turn>/<end_of_turn>, and sanitize (lines 23-27) is exactly two .replace calls over those constants - no marker table, no regex, no third replacement. render_entry (line 40) is the sole sanitization funnel and is reached by every untrusted path (agent.rs:76,108 including Role::Tool, chat.rs:35,42,87,100, compact.rs:71,73), so no caller pre-filters. Searched for handling elsewhere: grep for the literal '\|>' across the whole rust-harness tree returns zero hits, while training/models/families.py:76-84 and tests/training/test_sft.py do use <\|turn>/<turn\|>, so the harness has no Gemma 4 marker knowledge anywhere; the constants are compile-time const with no load-time derivation (no chat_template read in runtime/src). The only real compensating control is runtime/src/tokenizer.rs:292-340 encode_prompt/special_at with generator.rs:62 PROMPT_SPECIALS, which keeps a non-allowlisted special as literal text - that blocks id-level forgery, which the finding already excluded when grading high rather than critical. permissions.rs defaults to Ask but its own header documents that a bash prefix-allow is an allow-anything rule, so it is a downstream mitigation, not the missing guard. The named regression test template.rs:112-127 builds its injection from the same TURN_OPEN/TURN_CLOSE constants and asserts contains("[start_of_turn]model"), so it is tautological and cannot fail on the Gemma 4 markers. Caveat that tempers but does not refute: the template emits Gemma 3 delimiters end to end, so on a Gemma 4 model neither marker set is a trained boundary in the prompt and the residual exposure is textual persuasion - the exact reasoning the finding used for its high grade.
- REFUTED: The text-level observation is correct (template.rs:23-27 strips only the two Gemma 3 literals, and tool output does reach the prompt via agent.rs:185 -> prompt() -> render_entry), but the stated consequence does not occur on any reachable path. (1) No E2B/E4B model can be behind that prompt: main.rs:188 -> CpuModel::from_gguf -> check_supported (cpu.rs:206-252) bails on any tensor containing "per_layer", which gemma-4.md:252,289-301 says both E-series variants carry, and E2B also trips the per-layer FFN width (MatFormer) check; the only other backend is refused at main.rs:180-184 ("the cuda backend (roadmap phase A) has not landed"). (2) Even hypothetically served, <\|turn> cannot become a trained delimiter: special_at (tokenizer.rs:322-339) resolves only allowlisted literals and Kind::Control/UserDefined pieces are excluded from the Viterbi lookup table (tokenizer.rs:171-193), so a control id can never be produced from text; on a Gemma 4 vocab the harness's own <start_of_turn> is equally literal text, so the injected marker has no more authority than any other attacker prose - baseline untrusted text, not an incremental guard failure. (3) The "agent would then execute" tail is gated: permissions.evaluate defaults to Decision::Ask (permissions.rs:83) and main.rs:414-418 installs an empty rule set when permissions.yaml is absent, so an injected call hits the interactive y/N prompt. (4) The guard's real invariant holds: the sanitize set equals PROMPT_SPECIALS, both being the same two template.rs constants (generator.rs:62), and the cited regression test uses those constants, so it fails if sanitize is neutered. The residual is a latent coupling hazard in an as-yet-unported template (self-declared TODO at template.rs:1-11), not an ineffective guard reachable by E2B or E4B.

### GC-211 One head_dim scalar cannot express sliding 256 vs full 512; no named guard, so refusal is a bare length error

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `unrepresentable` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/model/cpu.rs:46` |
| Variants | 12B, 31B, E2B, E4B, all |
| Claims | C1-01, C1-02, C11-07, C12-10, C13-06, C3-07, C3-08, C3-09, C3-10, C3-11, C3-12, C3-13, C3-14, C3-15, C3-16, C3-17, C3-18, C8-07, C8-08 |
| Verification | 3 verdicts recorded, 1 refutations |

**Reference.** spec 180-182: "The two attention types use **different head dimensions**. Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512). This is visible directly in the weight shapes and is not an inference from the configuration."

**Implementation.** cpu.rs:46 "pub head_dim: usize,"; cpu.rs:89 "head_dim: u(\"attention.key_length\")?,"; cpu.rs:673-674 "let q_dim = config.n_heads * head_dim; let kv_dim = config.n_kv_heads * head_dim;"; cpu.rs:684 "q_norm: t(\"attn_q_norm.weight\", head_dim)?,"

**Consequence.** Spec 180-182 and 188/199 give two head widths per model. Config holds one `head_dim` (cpu.rs:46, from attention.key_length at :89), which feeds q_dim/kv_dim (673-674), the q_norm/k_norm lengths (684-685), the RoPE table (703) and the attention scale (830). An E4B full-attention layer supplies 4096*2560 = 10,485,760 elements against the implied 2560*2048 = 5,242,880 (E2B: 6,291,456 against 3,145,728), so the load aborts at the element check (665-669) with "tensor blk.5.attn_q.weight has 10485760 elements, the configuration implies 5242880". Loud, but undiagnosed and undeclared: check_supported has no branch for it and the header's superset list (16-18) omits it, so the operator gets a shape complaint rather than "full-attention layers use global_head_dim". Second consequence, and the reason this is graded high rather than medium: gguf-info calls only Config::from_gguf + check_supported and loads no tensor, so for a variant with no per-layer tensors (31B-shaped) it prints "supported" for a file the loader then refuses, contradicting the docstring at cpu.rs:204-205 that the tool and the loader cannot disagree. The string global_head_dim appears nowhere in the tree outside the spec; kv.rs::KvGeometry mirrors the single head_dim, so the KV budget prices global layers at half their true width.

- upheld: Spec fidelity holds. /work/docs/models/gemma-4.md:180-182 contains the quoted sentence verbatim ("The two attention types use **different head dimensions**. Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512). This is visible directly in the weight shapes and is not an inference from the configuration."), unhedged and outside the Open Questions section. The family-wide scope tag is the spec's own: lines 105-111 place head_dim 256 and global_head_dim 512 under "Invariants across all four variants ... safe to treat as constants", and trap 6 at 568-569 repeats the rule unqualified. The finding's arithmetic derives from the cited rows: line 188 (E4B q_proj sliding [2048,2560] vs full [4096,2560]) and line 199 (E2B [2048,1536] vs [4096,1536]). The q_norm/k_norm consequence is independently supported at 192-193 and 212-213, and 206-210 restates the rule as a formula. The only narrowing I found is the Open Question "Tensor-level data for 12B and 31B. Not collected", which limits the weight-shape evidence to E2B/E4B; the finding cites only E2B/E4B tables for shapes and rests the all-variant scope on the invariants table, so this is not an over-read. Spec contains no GGUF mapping (no hits for key_length/gguf), so the attention.key_length claim is code-side and outside this lens; line 536's head_dim 64/72 is the vision tower, not a conflicting text-path value.
- upheld: Reproduced every cited line in /work/rust-harness/runtime/src/model/cpu.rs: :46 `pub head_dim: usize` is a single scalar, :89 sources it solely from `attention.key_length` (model-level, no per-layer variant), :673-674 compute `q_dim`/`kv_dim` once outside the per-layer loop, and :684-685/:703/:830 reuse the same width for q_norm/k_norm, the RoPE table and the attention scale. I searched for the handling the finding claims is missing and it is genuinely absent: `check_supported` (~:207-330) validates per-layer widths only for `ffn_up` (MatFormer, named diagnostic); its sole `attn_q` reference is the name-presence shared-KV test; the rope.dimension_count check compares against the same single head_dim; the unconsumed-tensor backstop is name-based and the names are identical. The only thing that catches divergence is the generic `ensure!` at :665-669 whose format string matches the quoted message verbatim ("tensor {name} has {} elements, the configuration implies {want}"). The header superset list at :16-18 omits dual head dims, `grep -rni head_dim /work/docs/inference/*.md` returns nothing, and `global_head_dim` appears only in /work/docs/models/gemma-4.md (spec lines 180-182 confirmed) plus a triage memory file. Secondary claim also verified structurally: /work/rust-harness/runtime/src/bin/gguf-info.rs:164 calls only `Config::from_gguf` + `check_supported` and never `load_tensor`, printing "[OK] CPU backend compatibility: supported", contradicting the docstring at cpu.rs:204-205; spec lines 148-150 confirm 31B has hidden_size_per_layer_input 0 and num_kv_shared_layers 0 so the named branches do not fire there. kv.rs:22/29 with harness/src/main.rs:255 confirms KvGeometry mirrors the one head_dim, and synth.rs:235-248 writes uniform widths so no test exercises the case. Only caveat: spec line 58 says tensor-level data for 12B/31B was not collected, so the 31B example is inference — this weakens the variant example, not the mechanism.
- REFUTED: Traced harness/src/main.rs:188 -> CpuModel::from_gguf (cpu.rs:645-648): Config::from_gguf then check_supported run before the tensor closure that contains the element check at cpu.rs:665-669. check_supported's first guard (cpu.rs:311-320) bails on any tensor name containing "per_layer", and the spec (gemma-4.md:150, 252, 292-297, 366-380) says E2B and E4B both carry per-layer embedding tensors, so an E-series checkpoint is refused with a named, diagnosed message, never producing the quoted "tensor blk.5.attn_q.weight has 10485760 elements..." error; E2B additionally trips the MatFormer ffn-width guard (use_double_wide_mlp true) and the E-series cross-layer KV sharing trips the shared-KV guard, and even under renaming the generic unconsumed-tensor backstop (cpu.rs:302-329) fires inside check_supported. The high-grade rationale also fails: gguf-info.rs:164-167 calls the same check_supported, so for E2B/E4B it prints "[ERROR] ... per-layer embeddings", not "supported" — the tool/loader disagreement needs a file with no per_layer tensors (12B/31B-shaped, recorded-only in this campaign), and even there is contingent on attention_k_eq_v (spec:215) and the dual-RoPE key guard not firing first. The KV-budget consequence is likewise unreachable for E2B/E4B: main.rs builds KvGeometry at :253, after the load at :188 has already failed. The underlying single-head_dim unrepresentability is real, but the stated reachability and consequence are not.

### GC-212 No final logit softcap in the forward pass; the only defense is a metadata key spelling no artifact confirms

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `needs-host` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/model/cpu.rs:887` |
| Variants | 12B, 31B, E2B, E4B, all |
| Claims | C1-06, C12-06, C13-01, C2-13 |
| Verification | 3 verdicts recorded, 1 refutations |

**Reference.** spec 115: "\| `final_logit_softcapping` \| 30.0 \|" (listed among values that are "identical in E2B, E4B, 12B, and 31B", spec 105)

**Implementation.** cpu.rs:885-892: "rms_norm_into(x, &w.output_norm, c.rms_eps, normed); let hidden: &[f32] = normed; let logits = (0..c.vocab_size).map(\|t\| { let row = &w.embed[t * c.embed_dim..(t + 1) * c.embed_dim]; row.iter().zip(hidden).map(\|(a, b)\| a * b).sum() }).collect();" - no tanh cap between the tied projection and the returned Logits

**Consequence.** spec 115 lists final_logit_softcapping 30.0 among the values identical in all four variants (spec 105). cpu.rs:885-892 goes from the final norm straight to the tied dot product with no tanh cap. Softcapping carries no tensor, so the unconsumed-tensor backstop cannot see it; the whole defense is the substring test at cpu.rs:293, which fires only on a key under the {arch}. prefix whose suffix contains "softcapping" (or is attention.scale / logit_scale). That is the llama.cpp Gemma-2 spelling, and Gemma 4's own cap vocabulary is not uniform — spec 517 gives the audio tower's attention_logit_cap 50.0 — so the spelling cannot be assumed. If the key is absent or spelled otherwise, the checkpoint loads and every logit is uncapped: greedy argmax is unchanged because 30*tanh(z/30) is monotone, so the wrongness hides while perplexity (eval), temperature/top-p sampling and every captured logprob are wrong. The test at cpu.rs:1284-1301 proves only that the guard fires on hand-inserted keys. 31B is the variant whose load reaches this check with nothing earlier to stop it (E2B/E4B are refused at cpu.rs:211). Note the header list at cpu.rs:16-18 does not name softcapping either, so nothing in the file declares the gap.  Propagated to: rust-harness/runtime/src/eval.rs:55. One defect: final_logit_softcapping 30.0 is never applied and the only defense is a metadata-key spelling no artifact confirms. cpu.rs:887 (forward pass) is the ROOT; eval.rs:55 reports the same absence downstream in eval and sampling.

- upheld: Spec-lens check only. All three citations are verbatim accurate in /work/docs/models/gemma-4.md. Line 115 is exactly "\| `final_logit_softcapping` \| 30.0 \|", inside the table under H2 "## Invariants across all four variants" (line 103). Lines 105-106 read "These values are identical in E2B, E4B, 12B, and 31B. They are safe to treat as constants, and a verification pass should assert them rather than read them per model." - the spec is stronger than the finding claims, since it explicitly directs a verification pass to assert these constants; the finding quotes only the first half. Line 517 is exactly "\| `attention_logit_cap` \| 50.0 \|" in the E-series audio tower table under "## Modality towers" (line 500), and the finding correctly labels it as the audio tower's. Grep for softcap\|logit_cap\|tanh\|capping returns only lines 113 (gelu_pytorch_tanh), 115, and 517, confirming the finding's point that the family document's cap vocabulary is non-uniform. Searched for weakening qualifiers: the seven "## Open questions" entries (lines 582-606: proportional rope, layer_scalar, per-layer embedding data path, E4B shared-KV boundary, tensor-level data for 12B/31B, use_bidirectional_attention, attention_k_eq_v) and the 14 "## Traps and discrepancies" items (lines 553-581) contain nothing about softcapping, and nothing in the spec suggests the field is inert or unverified. The "tensor-level data for 12B and 31B not collected; all statements derive from config.json" open question does not undercut the 31B emphasis, because final_logit_softcapping is a config.json scalar. The spec does not state the 30*tanh(z/30) formula, but the finding does not attribute that formula to the spec. Only mild stretch: line 517 sits in a section the spec introduces as "scoped out of text-only work", but the quoted value and its labeling are correct. Code claims (cpu.rs:885-892, the cpu.rs:293 guard, cpu.rs:1284-1301, eval.rs:55) are outside this lens.
- upheld: Read /work/rust-harness/runtime/src/model/cpu.rs:882-894 directly: after rms_norm_into(x, &w.output_norm, ...) the code maps over vocab_size doing row.iter().zip(hidden).map(\|(a,b)\| a*b).sum() and returns Logits — no tanh/30.0 cap, exactly as the finding states. A repo-wide grep for softcap\|soft_cap\|tanh\|logit_scale\|logit_cap over all rust-harness .rs files finds no cap applied anywhere (the only tanh is inside gelu at cpu.rs:601); eval.rs:35-63 consumes the raw logits. The sole defense is the load-time bail at cpu.rs:288-300, which strips the {arch}. prefix and fires only when the suffix contains "softcapping" or equals attention.scale/logit_scale; it is live (called from from_gguf:650 and gguf-info.rs:164) but is purely spelling-dependent — Config::from_gguf (56-140) and check_supported impose deny-by-default only on the rope.* prefix, so any other unrecognized {arch}.* key loads silently. The test at cpu.rs:1284-1308 only inserts those four keys by hand. Spec /work/docs/models/gemma-4.md:105 and :115 do list final_logit_softcapping 30.0 as identical across all four variants, with no statement that it is unused; cpu.rs:16-21 does not name softcapping among the refused features. training/tune/distill.py:149-165 only compares the HF config field between teacher and student and applies no cap. No guard, load-time validation, or caller elsewhere handles this.
- REFUTED: The code fact is true (cpu.rs:885-893 goes final RMSNorm -> tied dot product -> Logits with no tanh cap, and cpu.rs:288-300 is a substring guard on the {arch}. prefix), but the stated failure mode is not reachable and the consequence does not occur. Only real entry point for a real checkpoint is harness/src/main.rs:188 -> CpuModel::from_gguf (cpu.rs:646), which runs check_supported (650) then an exact per-tensor element-count check (665-669). E2B/E4B are refused at cpu.rs:211 (per-layer embeddings, spec 150) and cpu.rs:223 (shared KV, spec 149) - the finding concedes this yet tags itself [all/E2B/E4B/12B/31B]. The finding's claim that 31B reaches the softcap guard "with nothing earlier to stop it" is false: spec 180-183 states full-attention layers use global_head_dim 512 while sliding layers use head_dim 256, "visible directly in the weight shapes", and 12B/31B have 8 and 10 full-attention layers (spec 173-174); Config holds one head_dim from attention.key_length (cpu.rs:89), so attn_q/k/v/output/q_norm/k_norm for one layer class always mismatches and the loader bails before any forward pass. Independently, the RoPE guard at cpu.rs:257-280 precedes the softcapping guard and bails on partial rotary 0.25 / a second theta (spec 126-127, 228-232), and spec 213-215's attention_k_eq_v=true on 12B/31B is exactly the shared-KV signature refused at cpu.rs:223. The only checkpoints that currently reach cpu.rs:887 are the synthetic tiny-model fixtures, which have no 30.0 cap to honor; README.md:194 still lists loading E4B as unlanded roadmap phase A. The gap is latent in a forward pass that cannot yet load any real Gemma 4 variant, not a live wrong-logprob path.

### GC-219 The committed gemma4 GGUF fixture writes bos=1 / eos=2, the transposition spec trap 8 warns is not a typo

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `mismatch` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/synth.rs:264` |
| Variants | 12B, 31B, E2B, E4B |
| Claims | C1-11, C1-15, C10-02, C10-12, C10-13, C10-15, C13-08, C5-07, C5-08, C5-09, C8-11, C8-12 |
| Verification | 1 verdicts recorded, 0 refutations |

**Reference.** L129-130: "The token identifier ordering is not a transcription error. `eos_token_id` is 1 and `bos_token_id` is 2 in every variant. Preserve that ordering."

**Implementation.** L264-265: `w.kv_u32("tokenizer.ggml.bos_token_id", 1); w.kv_u32("tokenizer.ggml.eos_token_id", 2);` with L64: `let mut pieces = vec!["<pad>".to_string(), "<s>".to_string(), "</s>".to_string()];`

**Consequence.** synth.rs:264-265 writes bos 1 and eos 2 over a vocabulary whose first three pieces are `<pad>`, `<s>`, `</s>` (synth.rs:64), i.e. the Llama assignment, while the file banner (synth.rs:5-7) claims the metadata follows the conventions the real E4B checkpoint uses and synth.rs:241 declares `general.architecture = gemma4`. Gemma 4 is pad 0, eos 1, bos 2 (spec:120-122, 448-450). No runtime path is wrong today because Tokenizer::from_gguf reads both ids from metadata; the cost is that synth.rs:346-347 pins the transposed pair as an assertion, so this is the only Gemma-shaped checkpoint in the tree and it teaches and validates the inverse convention. Root: one assumption copied, since tokenizer.rs and harness/tests/generator_e2e.rs repeat it.  Propagated to: rust-harness/fixtures/tiny-gemma4.gguf:1, rust-harness/runtime/src/tokenizer.rs:409. One transposition (bos=1 / eos=2 where Gemma 4 is eos 1 / bos 2) written once and copied. synth.rs:264 is the ROOT generator; fixtures/tiny-gemma4.gguf is its output, and tokenizer.rs:409's in-file test vocab repeats the same pair. The Gemma-3 delimiter half of the tokenizer finding is covered by the turn-marker group.

- upheld: Read the cited code directly: /work/rust-harness/runtime/src/synth.rs:264-265 emits `w.kv_u32("tokenizer.ggml.bos_token_id", 1)` and `w.kv_u32("tokenizer.ggml.eos_token_id", 2)` under `general.architecture = "gemma4"` (line 241), over the vocab `["<pad>", "<s>", "</s>"]` at line 64, with the banner (lines 4-7) claiming the metadata follows the real E4B checkpoint conventions. Spec citations are exact: /work/docs/models/gemma-4.md:120-122 (bos 2, eos 1, pad 0), 129-130 (the "not a transcription error / Preserve that ordering" paragraph), 448-450, and trap 8 at line 560. Propagation verified from actual bytes, not inference: parsing /work/rust-harness/fixtures/tiny-gemma4.gguf yields bos_token_id = UINT32 1 and eos_token_id = UINT32 2. I searched for the handling that would refute the finding and it does not exist: Tokenizer::from_gguf (tokenizer.rs:122-136) reads both ids through a generic id_key closure with only a bounds check at 131-135 ("bos/eos id outside vocabulary"), no gemma4 branch, no piece-text cross-check; from_parts (141-157) takes them as unvalidated parameters; a tree-wide grep for these keys returns only synth.rs:264-265, tokenizer.rs:129-130, and test fixtures; and callers do use them (generator.rs:66 and activation-capture.rs:228 prepend tokenizer.bos(), session.rs:353 stops on eos). synth.rs:346-347 does pin the pair as assertions. The finding's own concession that no runtime path is wrong holds — the ids are self-consistent inside the synthetic 288-token vocab — so the cost is a misleading fixture plus a test that validates the inverse convention, which is below medium.

## Confirmed findings (unverified)

The 189 findings below come from completeness-critic gap-fill rounds. They
were produced by the same method as the verified set and cite the same kind of evidence,
but they carry zero recorded verdicts and were never deduplicated against the verified
set. Read them as leads, not as conclusions, and expect repetition.

| ID | Severity | Verdict | Site | Summary |
|----|----------|---------|------|---------|
| GC-001 | critical | `guard-ineffective` | `docs/verification/rust-harness.md:583` | RH.11's expected signal cannot distinguish a correct stop from the real failure, so the entry closes VERIFIED either way |
| GC-003 | critical | `mismatch` | `rust-harness/harness/src/template.rs:15` | rust harness emits Gemma-3 <start_of_turn>/<end_of_turn> turn delimiters; Gemma 4's markers are <\|turn>/<turn\|> |
| GC-004 | critical | `guard-ineffective` | `rust-harness/runtime/src/bin/gguf-info.rs:140` | Shared-KV detection keys on layers lacking attn_k/attn_v, a signature the spec says never occurs, so the doc's prescribed reconciliation reports a false negative |
| GC-006 | high | `stale-doc` | `docs/inference/rust-harness-runtime-design.md:318` | The authority document is cited by no code, runbook, investigation, recipe or roadmap; the suspect table still names a pending runbook as the authority |
| GC-007 | high | `mismatch` | `docs/inference/rust-harness-runtime-design.md:299` | Reconciliation table attributes E4B's feed-forward widths to MatFormer slicing; the spec says E4B widths are uniform at 10240 across all 42 layers |
| GC-008 | high | `mismatch` | `docs/inference/rust-harness-runtime-design.md:300` | Reconciliation table states E4B's last 18 layers reuse K/V from the last non-shared layer; the spec says all 42 layers carry their own k_proj and v_proj and the 18-layer boundary is unverified |
| GC-009 | high | `mismatch` | `docs/interpretability/rust-harness-interp.md:240` | The interpretability contract asserts a specific per-layer-embedding data path that the spec records as an undetermined open question |
| GC-010 | high | `stale-doc` | `docs/models/gemma-4.md:610` | Spec says a contained sandbox cannot reach huggingface.co, but the sandbox proxy allowlists it (and .cdn.hf.co) for exactly this purpose |
| GC-012 | high | `mismatch` | `docs/training/training-scale-reference.md:93` | The training ladder names the E-series by its effective parameter count and sizes GPU fit against it, the exact underestimate the spec warns of |
| GC-013 | high | `mismatch` | `docs/training/training-scale-reference.md:93` | The Gemma 4 ladder names the E-series "2B" and "4B" and sizes VRAM against the effective count, understating resident weights ~2x |
| GC-015 | high | `guard-ineffective` | `docs/verification/rust-harness.md:109` | RH.1's diagnostic question 2 detects shared KV by a tensor-absence signature the spec says never occurs, so the runbook records a false negative |
| GC-016 | high | `stale-doc` | `docs/verification/rust-harness.md:100` | RH.1, the entry whose purpose is architecture reconciliation, routes to the secondhand table and never cites the in-tree authority |
| GC-017 | high | `stale-doc` | `docs/verification/rust-harness.md:566` | RH.11's stop-behavior narrative names Gemma 3's `<end_of_turn>` where the spec gives Gemma 4's `<turn\|>` |
| GC-018 | high | `mismatch` | `interp/families.py:47` | Hook registry hardcodes the flat decoder path model.layers.{layer} for gemma4/gemma4_text, which the E-series composite layout does not expose |
| GC-019 | high | `mismatch` | `interp/families.py:47` | The interp hook registry encodes model.layers.{n}, but every E-series decoder tensor is under model.language_model.layers.{L} |
| GC-020 | high | `needs-host` | `interp/harvest.py:181` | Whether the gate can run at all on E2B is unsettled: it assumes AutoModelForCausalLM accepts Gemma4ForConditionalGeneration and that model.layers.{n} resolves |
| GC-022 | high | `mismatch` | `rust-harness/harness/src/template.rs:15` | Harness chat template renders Gemma 3's `<start_of_turn>`/`<end_of_turn>`, not Gemma 4's `<\|turn>`/`<turn\|>` |
| GC-023 | high | `guard-ineffective` | `rust-harness/harness/src/template.rs:23` | Prompt-injection sanitizer neutralizes only the Gemma-3 markers, so real Gemma 4 turn and tool-call markers pass through untouched |
| GC-024 | high | `needs-host` | `rust-harness/harness/src/template.rs:32` | Two tiers assert contradictory structural facts about the Gemma 4 system role and nothing in the tree compares them |
| GC-025 | high | `mismatch` | `rust-harness/runtime/src/bin/gguf-info.rs:140` | gguf-info will print an affirmative "Shared KV cache: none" on E2B/E4B, contradicting num_kv_shared_layers 20/18, in the tool designated for the RH.1 reconciliation |
| GC-026 | high | `mismatch` | `rust-harness/runtime/src/bin/gguf-info.rs:140` | gguf-info prints "Shared KV cache: none" on a checkpoint the spec says has 18 (E4B) / 20 (E2B) shared-cache layers |
| GC-027 | high | `guard-ineffective` | `rust-harness/runtime/src/model/cpu.rs:223` | The shared-KV refusal keys on absent attn_k/attn_v, a signature the spec says never occurs, so it cannot fire on E2B or E4B |
| GC-028 | high | `needs-host` | `rust-harness/runtime/src/model/cpu.rs:211` | Whether the GGUF converter's per-layer tensor names contain the substring the loader refusal matches can only be settled on a real file |
| GC-029 | high | `guard-ineffective` | `rust-harness/runtime/src/model/cpu.rs:223` | The shared-KV refusal fires only on layers missing attn_k/attn_v, a signature the spec says never occurs (extends the C6-01/C6-02 sweep) |
| GC-030 | high | `needs-host` | `rust-harness/runtime/src/model/cpu.rs:210` | The per-layer-embedding refusal keys on the substring "per_layer" in GGUF tensor names; only a real GGUF settles whether the converter preserves it |
| GC-031 | high | `guard-ineffective` | `rust-harness/runtime/src/model/cpu.rs:223` | The rust shared-KV refusal keys on layers missing attn_k/attn_v, a signature the spec says never occurs in Gemma 4 weights |
| GC-032 | high | `needs-host` | `rust-harness/runtime/src/session.rs:353` | Generation stops on a single GGUF eos id; a Gemma 4 chat turn closes with <turn\|>, which the harness has no stop path for |
| GC-033 | high | `mismatch` | `rust-harness/runtime/src/tokenizer.rs:549` | The only test of encode_prompt's special resolution pins Gemma 3 delimiters, a marker set the spec does not contain |
| GC-034 | high | `needs-host` | `tests/interp/test_sae.py:49` | SAE test fixtures pin `model.layers.{L}` as the Gemma 4 E2B hook path, but the spec names decoder layers `model.language_model.layers.{L}` |
| GC-036 | high | `mismatch` | `training/commands/download.py:109` | download prints a prepare-only destination path as the recipe model id; training and export see that tree at /app/models |
| GC-037 | high | `needs-host` | `training/data/prepare.py:42` | load_tokenizer's EOS fallback list omits Gemma 4's `<eos>`, and a local Gemma tokenizer directory never reaches AutoTokenizer |
| GC-038 | high | `mismatch` | `training/data/prepare.py:42` | load_tokenizer's EOS fallback list omits Gemma 4's `<eos>`, so a local Gemma tokenizer directory cannot resolve its EOS id |
| GC-039 | high | `guard-ineffective` | `training/recipes/reference/continual-gemma4-e2b.yaml:8` | The only Gemma-4-E2B continual recipe declares D4 calibration coverage, but calibrate() refuses every zero_offload recipe, so its parameter model is never checked |
| GC-041 | medium | `stale-doc` | `.claude/agent-memory/triage-agent/project_gemma4_2304_propagation.md:13` | The 2304-propagation ledger's "Known copies" list omits two interp test modules that bind model_type gemma4_text to hidden_size 2304 |
| GC-042 | medium | `needs-host` | `.claude/agent-memory/triage-agent/project_gemma4_synth_fixture_root.md:23` | Triage memory states as settled fact that check_supported refuses real E-series checkpoints, but the guard keys on a GGUF substring no artifact in the tree confirms |
| GC-043 | medium | `stale-doc` | `.claude/agent-memory/triage-agent/project_gemma4_synth_fixture_root.md:28` | Synth-fixture memory tells triage to file the bos/eos transposition as "extends A1-15", but A1-15's recorded scope and followup cover only QK scale and GQA mapping |
| GC-044 | medium | `mismatch` | `.claude/skills/verify-host/SKILL.md:15` | Spec says a contained sandbox cannot reach huggingface.co; the repo's proxy allowlist and sandbox doc say it deliberately can |
| GC-045 | medium | `absent-undeclared` | `Makefile:143` | The documented staging entry point asks for REVISION=<sha> with no in-tree source for the four Gemma 4 pins |
| GC-047 | medium | `stale-doc` | `docs/inference/rust-harness-observability.md:135` | Observability design doc still frames E4B per-layer embeddings as an open risk and names E4B as the phase-3 target; the spec settles both against it |
| GC-048 | medium | `stale-doc` | `docs/inference/rust-harness-observability.md:133` | The doc's architecture risk register names only MatFormer and per-layer embeddings, omitting the dual RoPE regimes, partial rotary, split head dims, and logit softcapping |
| GC-057 | medium | `mismatch` | `docs/inference/rust-harness-runtime-design.md:302` | Reconciliation table lists "Value normalization" and "LAuReL" as retained Gemma 4 E4B features; neither appears in the spec's exact E4B tensor inventory |
| GC-058 | medium | `stale-doc` | `docs/inference/rust-harness-runtime-design.md:269` | The runtime design doc calls E2B "the same code path as E4B", which the spec contradicts on period, KV heads, and MLP width |
| GC-059 | medium | `stale-doc` | `docs/inference/rust-harness-runtime-design.md:301` | The reconciliation table describes the full-attention RoPE regime as p-RoPE plus theta 1M, dropping the undetermined `proportional` scaling entirely |
| GC-060 | medium | `stale-doc` | `docs/inference/rust-harness-runtime-design.md:300` | The reconciliation table states the E4B shared-KV boundary and mechanism as fact where the spec records both as unverified |
| GC-061 | medium | `stale-doc` | `docs/inference/rust-harness-runtime-design.md:191` | Design doc states the harness renders Gemma 4's turn format including the function-calling convention; it renders neither |
| GC-062 | medium | `stale-doc` | `docs/inference/rust-harness-runtime-design.md:295` | The suspect reconciliation table has no dual-head-dimension row and asserts a value-normalization feature the spec's complete E4B inventory lacks |
| GC-063 | medium | `stale-doc` | `docs/interpretability/investigations/p4_d2_activation_harvesting_investigation.md:127` | P4-D2 investigation sizes the E2B stack at 30 layers and hidden 2304, against the spec's 35 layers and 1536, and is the origin of the layer-15 hook |
| GC-064 | medium | `mismatch` | `docs/interpretability/investigations/p4_d2_activation_harvesting_investigation.md:119` | P4-D2 states E2B interleaves 5 sliding layers per global layer with sliding_window_pattern=6; the spec gives E2B every 5th layer, a 4:1 ratio |
| GC-065 | medium | `stale-doc` | `docs/interpretability/investigations/p4_d2_activation_harvesting_investigation.md:109` | P4-D2 records confirming E2B's real config.json as an unmet precondition, although the pinned confirmation exists in-tree |
| GC-066 | medium | `stale-doc` | `docs/interpretability/investigations/p4_d2_activation_harvesting_investigation.md:116` | The gate's stated expected value is a 2304-dim residual; E2B's hidden_size is 1536 (extends the known 2304 propagation root) |
| GC-067 | medium | `stale-doc` | `docs/interpretability/investigations/p4_d2_activation_harvesting_investigation.md:119` | The hook-depth guidance beside the gate states E2B interleaves 5 sliding per 1 global with sliding_window_pattern 6; E2B is every 5th layer (4:1) |
| GC-070 | medium | `stale-doc` | `docs/interpretability/rust-harness-interp.md:240` | The interpretability design asserts a specific per-layer-embedding data path that the spec records as undetermined |
| GC-071 | medium | `stale-doc` | `docs/interpretability/rust-harness-interp.md:240` | The interp contract doc states where the per-layer embedding joins the residual stream and that it is additive; the spec records both as undetermined |
| GC-075 | medium | `needs-host` | `docs/training/investigations/p2_d3_warmup_stable_decay_investigation.md:37` | The 200-step stability canary that p2_d3 defers peak_lr to cannot run here and its memfit preflight is the unchecked path from GX2-08-01 |
| GC-077 | medium | `stale-doc` | `docs/training/investigations/p2_d5_optimizer_offload_investigation.md:125` | P2-D5 investigation attributes 30 layers and hidden 2304 to Gemma-4-E2B as architecture confirmed from configuration_gemma4.py |
| GC-082 | medium | `stale-doc` | `docs/verification/rust-harness.md:109` | RH.1 asks for the shared-KV feature by the tensor-absence signature, so the runbook cannot detect KV sharing on the real weights |
| GC-083 | medium | `needs-host` | `docs/verification/rust-harness.md:135` | No runbook entry resolves the GGUF's EOS piece string or searches the vocabulary for the turn markers |
| GC-084 | medium | `absent-undeclared` | `docs/verification/rust-harness.md:19` | The campaign's scope note declares the template golden test unimplemented but not that the template's markers are the wrong family's |
| GC-085 | medium | `absent-undeclared` | `docs/verification/rust-harness.md:170` | RH.2 states coherence as an exclusive oracle for forward-pass numerics, with the chat-template cause unacknowledged |
| GC-086 | medium | `needs-host` | `docs/verification/step-zero-environment.md:26` | The step-zero environment gate pins transformers 5.13.1 and never checks that it implements Gemma4ForConditionalGeneration, which the spec forbids assuming |
| GC-087 | medium | `guard-ineffective` | `docs/verification/training-phase2.md:172` | P2.6's host-RAM acceptance names a test that never measures host RAM, so a calculator under-prediction on the E2B continual recipe cannot fail it |
| GC-088 | medium | `absent-undeclared` | `docs/verification/training-phase2.md:131` | The Phase 2 runbook runs the Gemma 4 E2B continual recipe three times with no staging step and no pinned revision |
| GC-091 | medium | `needs-host` | `docker/interp/Dockerfile.interp:65` | Interp image pins released transformers 5.13.1 and nothing in the build verifies it implements Gemma4ForConditionalGeneration |
| GC-092 | medium | `stale-doc` | `docs/interpretability/investigations/p4_d1_model_organisms_investigation.md:257` | The declared "SAELens hook-point smoke test on Gemma 4" gate does not exist in the smoke test; a differently-scoped harvest-only gate exists instead |
| GC-094 | medium | `guard-ineffective` | `interp/harvest.py:294` | The "gate before committing GPU-days" is opt-in: run_harvest never calls check_hook_dims, so `make harvest` spends the budget ungated |
| GC-095 | medium | `absent-undeclared` | `interp/organisms/seal.py:86` | Every sealed organism manifest records base_revision null: the pre-registration names google/gemma-4-E2B without the pinned commit |
| GC-096 | medium | `mismatch` | `docker/agents/proxy/squid.conf:38` | Spec says a contained sandbox cannot reach huggingface.co; the shipped egress allowlist carries .huggingface.co and .cdn.hf.co |
| GC-097 | medium | `mismatch` | `opencode.json:25` | opencode.json grants webfetch to huggingface.co and squid.conf allowlists it, contradicting the spec's "sandbox does not reach huggingface.co" (extends GX1-G2-hf-egress-03 to the OpenCode config) |
| GC-098 | medium | `needs-host` | `pyproject.toml:19` | transformers is pinned to a released version (5.13.1) with no recorded evidence that it implements Gemma4ForConditionalGeneration |
| GC-099 | medium | `mismatch` | `rust-harness/harness/src/chat.rs:26` | chat.rs trim_reply strips <end_of_turn>, so a Gemma 4 turn-close marker survives into the transcript and the next prompt |
| GC-102 | medium | `stale-doc` | `rust-harness/runtime/src/gguf.rs:53` | is_vision_tensor has no caller, so the documented load-time skip of tower tensors never happens and the loader refuses instead |
| GC-103 | medium | `absent-undeclared` | `rust-harness/runtime/src/gguf.rs:53` | is_vision_tensor has no caller and no audio branch, so the documented "vision components are skipped at load time" does not happen and the E-series audio tower is unhandled |
| GC-105 | medium | `unrepresentable` | `rust-harness/runtime/src/kv.rs:20` | KvGeometry is one per-model geometry, so the KV budget prices the 512-wide global layers at 256 |
| GC-110 | medium | `stale-doc` | `rust-harness/runtime/src/model/cpu.rs:245` | "MatFormer slicing" and "value normalization" are named as Gemma 4 features in code, an error message, a test assertion, and the README; the spec contains neither word |
| GC-111 | medium | `stale-doc` | `rust-harness/runtime/src/model/cpu.rs:22` | The only code-level Gemma 4 architecture citation in the tree points at the known-bad table and enumerates two features the spec does not support |
| GC-112 | medium | `unrepresentable` | `rust-harness/runtime/src/model/cpu.rs:46` | CpuModel Config carries one head_dim, so the spec's 512-wide full-attention layers are unrepresentable and undiagnosed |
| GC-113 | medium | `needs-host` | `rust-harness/runtime/src/model/cpu.rs:93` | Whether a Gemma 4 GGUF carries attention.sliding_window_pattern, and whether it is 5 on E2B, can only be settled on a real file |
| GC-114 | medium | `unrepresentable` | `rust-harness/runtime/src/model/cpu.rs:830` | One Config.head_dim scalar makes the attention softmax scale single-valued, but the spec gives full-attention layers a 512-wide head |
| GC-115 | medium | `absent-undeclared` | `rust-harness/runtime/src/model/cpu.rs:16` | The backend's declared-subset list omits the dual head dimension, so the one Gemma 4 feature behind the hardcoded scale is an undeclared gap |
| GC-116 | medium | `needs-host` | `rust-harness/runtime/src/model/cpu.rs:293` | The scale the code hardcodes has no source: the spec states none, no test checks it externally, and cpu.rs:293 refuses to be told one |
| GC-117 | medium | `absent-undeclared` | `rust-harness/runtime/src/model/mod.rs:20` | The trait sanctions a `trimmed slice` for Logits with no invariant tying its width to vocab_size, so a trimmed row silently renumbers tokens |
| GC-118 | medium | `unrepresentable` | `rust-harness/runtime/src/model/record.rs:198` | Replay caps a recording at 8191 absolute positions at vocab 262144, 16x below the E-series 131072 context |
| GC-119 | medium | `absent-undeclared` | `rust-harness/runtime/src/model/record.rs:29` | Recording header's only field is vocab_size, which the spec makes invariant across all four variants, so replay cannot detect a cross-model recording |
| GC-123 | medium | `needs-host` | `rust-harness/runtime/src/tensor.rs:6` | No bfloat16 tensor codec exists in the Rust tree and no dependency supplies one, while the spec records dtype bfloat16 for E2B and E4B |
| GC-124 | medium | `needs-host` | `rust-harness/runtime/src/tokenizer.rs:107` | Runtime tokenizer hard-requires SentencePiece scores, but Gemma 4 ships a tokenizers-backend tokenizer.json and no SentencePiece model file |
| GC-125 | medium | `needs-host` | `rust-harness/runtime/src/tokenizer.rs:630` | Whether Gemma 4 turn markers decode to nothing or leak as literal text depends on their GGUF token_type, which no artifact here can settle |
| GC-126 | medium | `stale-doc` | `training/recipes/reference/continual-gemma4-e2b.yaml:31` | The E2B continual-pretraining recipe carries revision: null and defers the pin to a future host pull, while the pinned revision is already recorded in-tree |
| GC-127 | medium | `needs-host` | `scripts/benchmark/server-bench.py:173` | server-bench.py's derived default LLAMA_ARG_CTX_SIZE is 137,568, above the E-series max_position_embeddings of 131072 |
| GC-128 | medium | `absent-undeclared` | `docs/verification/rust-harness.md:88` | No verification pass anywhere in the tree asserts the spec's invariant table, and no document declares the gap |
| GC-132 | medium | `absent-undeclared` | `rust-harness/harness/src/template.rs:138` | The only test that could catch a wrong turn delimiter is #[ignore]d with an unimplemented body, and the fixture vocabulary has no chat-delimiter piece |
| GC-134 | medium | `mismatch` | `rust-harness/runtime/tests/eval_acceptance.rs:24` | Tier-3 fixture recipe pins no hub revision: AutoTokenizer.from_pretrained('google/gemma-4-e4b') resolves main, not the pinned 411aa17b (same class as KB-40/KB-51, new site) |
| GC-135 | medium | `guard-ineffective` | `tests/hardware/test_gpu.py:197` | The only test that could catch an unsupported gemma4 architecture under the pinned transformers converts that failure into a skip labelled "weights/config absent" |
| GC-136 | medium | `mismatch` | `tests/interp/test_report_envelope.py:47` | Report-envelope fixture pins hook "model.layers.15" for google/gemma-4-E2B, a module path the E-series checkpoint does not expose |
| GC-137 | medium | `absent-undeclared` | `tests/interp/test_sae.py:47` | No Gemma 4 capture or SAE run pins a hub revision: the fixture default is None and the only revision test uses a synthetic value (extends KB-51) |
| GC-138 | medium | `mismatch` | `tests/training/test_calculator.py:322` | The only Gemma 4 fixture in the Python tier sets num_key_value_heads=4, a 2:1 GQA ratio no Gemma 4 variant with 8 heads has |
| GC-139 | medium | `mismatch` | `tests/training/test_calculator.py:348` | assert spec.kv_dim == 4 * 256 pins a KV width no Gemma 4 variant has, so any correction to E2B's 1 or E4B's 2 breaks a green test |
| GC-140 | medium | `needs-host` | `tests/training/test_cli_handlers_train.py:291` | The distillation teacher asserted verbatim by the assigned tests, google/gemma-4-E4B-it, is a repository the spec never establishes |
| GC-145 | medium | `mismatch` | `training/calculator/components.py:153` | ArchitectureSpec prices two RMSNorm tensors per layer; Gemma 4 E-series carries five hidden-wide norms plus q_norm and k_norm |
| GC-146 | medium | `unrepresentable` | `training/calculator/components.py:111` | ArchitectureSpec's single explicit_head_dim prices E-series full-attention layers at 256 instead of 512 (extends the calculator scalar root) |
| GC-147 | medium | `absent-undeclared` | `training/calculator/components.py:139` | ple_param_count omits per_layer_model_projection and the PLE norms, a fourth face of the scalar root the memo does not enumerate |
| GC-148 | medium | `mismatch` | `training/commands/download.py:43` | download --revision help directs the operator to resolve the hub's floating default branch instead of the recorded pinned revision |
| GC-149 | medium | `guard-ineffective` | `training/commands/download.py:40` | The required-revision guard enforces presence, not pinning: --revision main passes and is then printed back as a recipe pin |
| GC-150 | medium | `guard-ineffective` | `training/eval/scorecard.py:56` | Scorecard thinking_mode confound is skipped whenever either side is unknown, unlike quant_type which refuses on None |
| GC-151 | medium | `guard-ineffective` | `training/export/convert.py:88` | GGUF export guard keys on architectures[0], a string the spec says E2B/E4B and 31B share, so it cannot separate the three implementation groups |
| GC-153 | medium | `needs-host` | `training/models/families.py:131` | Gemma 4 thinking-trace guard keys on `<\|channel>thought`, a channel label the spec never defines; only `<\|channel>`/`<channel\|>` are established |
| GC-155 | medium | `mismatch` | `training/recipes/interp/harvest-smoke.yaml:18` | Harvest recipes hook E2B layer 15 as "50 percent depth of a 30-layer stack"; on the real 35-layer E2B that is the first layer of the shared-KV / double-wide tail |
| GC-156 | medium | `needs-host` | `training/recipes/organisms/taboo.yaml:52` | Nothing binds the organism recipes' tokenizer artifact to Gemma 4 E2B's tokenizer; vocab 262144 and pad/eos/bos 0/1/2 are unverified |
| GC-157 | medium | `absent-undeclared` | `training/recipes/reference/continual-gemma4-e2b.yaml:31` | The spec's pinned revisions and config blob hashes reach no recipe; both E-series recipes still carry an unpinned revision |
| GC-158 | low | `stale-doc` | `.claude/agent-memory/triage-agent/MEMORY.md:3` | MEMORY.md index says the 2304 root has six copies; its detail file enumerates eight sites and the tree holds two more files |
| GC-159 | low | `stale-doc` | `.claude/agent-memory/triage-agent/project_gemma4_2304_propagation.md:27` | The 2304-propagation memory's closing pointer [[gemma4-spec-authority]] resolves to a superseded stub, not to the authority rule |
| GC-160 | low | `stale-doc` | `.claude/agent-memory/triage-agent/project_gemma4_conformance_audit.md:23` | Conformance-audit memory cites spec:138 as warning against reading transformers class defaults; that line warns against the top-level config block |
| GC-161 | low | `stale-doc` | `.claude/agent-memory/triage-agent/project_gemma4_conformance_campaign.md:22` | Memo's absence claim is string-scoped: two of the six concepts do have tree implementations under other names |
| GC-162 | low | `stale-doc` | `.claude/agent-memory/triage-agent/project_gemma4_spec_authority.md:9` | Extends GX1-M11-memory-08: the stub's "Safe to delete" is contradicted by the inbound link it is the only resolution for |
| GC-163 | low | `needs-host` | `docker/agents/proxy/squid.conf:36` | The proxy allowlist cites the authority as its reason to reach huggingface.co, while the authority states a contained sandbox does not reach huggingface.co |
| GC-164 | low | `stale-doc` | `docs/agents/tooling-parity.md:278` | tooling-parity.md still describes the OpenCode webfetch allowlist as two domains; opencode.json carries fourteen, including huggingface.co |
| GC-165 | low | `stale-doc` | `docs/inference/rust-harness-observability.md:146` | The doc records "one storage decision" as deferred and omits cross-layer KV cache sharing, the second and larger one on the E-series |
| GC-167 | low | `stale-doc` | `docs/inference/rust-harness-runtime-design.md:198` | rust-harness runtime design repeats the same stale sandbox-cannot-reach-HF premise for the reference tokenizer |
| GC-168 | low | `mismatch` | `docs/inference/rust-harness-runtime-design.md:298` | Reconciliation table gives E4B a total of 8B parameters; the spec derives approximately 7.52 billion including the per-layer table |
| GC-169 | low | `stale-doc` | `docs/inference/rust-harness-runtime-design.md:298` | The reconciliation table prices E4B's per-layer table against an 8B total where the spec derives 7.52B |
| GC-170 | low | `absent-undeclared` | `docs/inference/rust-harness-runtime-design.md:75` | The E-series audio tower is absent from the tree and from every scope statement, which names vision only |
| GC-171 | low | `absent-declared` | `docs/inference/rust-harness-runtime-design.md:75` | No vision-tower value is encoded anywhere, which is consistent with the declared no-multimodality scope |
| GC-172 | low | `stale-doc` | `docs/interpretability/investigations/p4_d1_model_organisms_investigation.md:131` | The P4-D1 architecture note gives E2B and E4B a "6:1" sliding-to-full interleave, which is neither variant's period or ratio |
| GC-173 | low | `stale-doc` | `docs/interpretability/investigations/p4_d1_model_organisms_investigation.md:131` | The organism investigation states a 6:1 sliding-to-full ratio for both E2B and E4B; the spec gives 4:1 for E2B and 5:1 for E4B |
| GC-175 | low | `needs-host` | `docs/models/gemma-4.md:616` | The regeneration recipe carries no Hugging Face token, while the repository elsewhere records Gemma 4 weights as gated |
| GC-176 | low | `stale-doc` | `docs/models/gemma-4.md:610` | The spec says a contained sandbox cannot reach huggingface.co; the repository's own proxy allowlists it for this purpose |
| GC-177 | low | `stale-doc` | `docs/testing/test_coverage_investigation.md:423` | Coverage investigation frames the composite hook-path risk as conditional and cites a capture.py hardcode that no longer exists |
| GC-178 | low | `stale-doc` | `docs/testing/test_coverage_investigation.md:389` | The readiness paragraph states one transformers floor (>= 5.5.0) and one model_type family for all Gemma 4 sizes, contradicting the 12B split and the E2B period |
| GC-179 | low | `stale-doc` | `docs/testing/test_coverage_investigation.md:390` | The test-coverage audit records a uniform 5:1 sliding-to-full cycle for every Gemma 4 size, which is wrong for E2B |
| GC-180 | low | `stale-doc` | `docs/testing/test_coverage_investigation.md:390` | test_coverage_investigation.md states one 5:1 sliding:full cycle for the whole Gemma 4 family; the spec gives E2B as 4:1, every 5th layer |
| GC-181 | low | `stale-doc` | `docs/testing/test_coverage_investigation.md:154` | The coverage investigation still records the Gemma 4 gate as raising raw AttributeError and as untested; both were since fixed |
| GC-182 | low | `needs-host` | `docs/training/investigations/f3_sft_entry_point_investigation.md:234` | A risks section asserts "transformers 5.13.1 ships the gemma4 architecture" as settled fact and narrows the live risk to TRL/PEFT |
| GC-183 | low | `stale-doc` | `docs/training/investigations/f4_teacher_synthetic_data_investigation.md:164` | F4's work breakdown lists a thinking-mode flag the shipped harness never gained, while F5 already cites that flag as the control point for distillation captures |
| GC-184 | low | `stale-doc` | `docs/training/investigations/f5_distillation_investigation.md:268` | F5 investigation states F4's sampling config controls thinking mode at capture time; F4 has no such control |
| GC-186 | low | `stale-doc` | `docs/training/training-scale-reference.md:142` | training-scale-reference claims a scaled Gemma 4 block loads with zero runtime changes and calls it verified; the spec's Gemma 4 block is refused by the loader |
| GC-188 | low | `absent-declared` | `docs/verification/training-phase0.md:320` | Phase 0's D4 "fully observed" verdict rests on two Llama-shaped scratch recipes, and the Gemma-4 calculator path is declared unverified in Phase 1 rather than silently assumed |
| GC-189 | low | `stale-doc` | `docker-compose.interp.yaml:34` | Interp compose scopes HF_TOKEN to the gated Gemma-2-2B download although the container's Gemma 4 E-series subjects are also gated |
| GC-190 | low | `absent-declared` | `docker/interp/Dockerfile.interp:108` | Interp build-time gate exercises Gemma-2-2B only; the Gemma 4 E-series path is deferred to a runtime check (declared) |
| GC-192 | low | `absent-declared` | `interp/harvest.py:253` | The transformer-lens gap is declared honestly and structurally bypassed: the pinned SAELens runner loads the HF checkpoint, not a HookedTransformer |
| GC-193 | low | `guard-ineffective` | `interp/harvest.py:218` | The gate is width-only, so it cannot prove "not a PLE tensor": per_layer_projection returns the PLE path to exactly hidden_size |
| GC-194 | low | `absent-undeclared` | `interp/organisms/generate.py:232` | The organism teacher draw uses its own sampling defaults; the shipped generation_config top_k 64 is applied nowhere in the repository |
| GC-195 | low | `absent-undeclared` | `interp/probes/capture.py:103` | The nnsight probe-capture path has no dimension gate at all, though families.py advertises the harvest gate as the runtime backstop |
| GC-196 | low | `mismatch` | `.claude/settings.json:54` | cdn.hf.co is in the proxy allowlist but in neither tool config, breaking the documented squid/WebFetch agreement invariant |
| GC-197 | low | `absent-declared` | `.opencode/agents/bug-hunter.md:28` | The bug-hunter exclusion-list step is inert in OpenCode, so the recorded "no Gemma 4 weights on this workstation" rule reaches only Claude sessions |
| GC-198 | low | `absent-declared` | `.opencode/commands/pins.md:5` | The shared /pins audit covers image tags and git clones only; the spec's model-revision and transformers-revision pins are outside its grep set |
| GC-199 | low | `stale-doc` | `docker/agents/proxy/squid.conf:36` | squid.conf comment tells the reader the Xet CDN domain is not yet listed, six lines before listing it |
| GC-200 | low | `absent-declared` | `docs/inference/rust-harness-runtime-design.md:75` | The entire modality-tower configuration surface is absent from the tree, consistent with the declared text-only scope |
| GC-201 | low | `mismatch` | `opencode.json:25` | opencode.json's webfetch allowlist carries huggingface.co but not cdn.hf.co, the second domain the spec's safetensors range-request method crosses |
| GC-202 | low | `needs-host` | `pyproject.toml:19` | The one dependency pin in the repository that bears on the spec is Python-side (transformers==5.13.1) and is unverified against Gemma4ForConditionalGeneration |
| GC-203 | low | `needs-host` | `docker-compose.rust-harness.yaml:7` | Container run of a real E4B GGUF (RH.12) has no capability-verdict step and the image ships no gguf-info binary |
| GC-204 | low | `absent-declared` | `docker-compose.rust-harness.yaml:26` | Harness container is structurally CPU-subset only and can never run the real E-series backend; declared at RH.1 |
| GC-207 | low | `mismatch` | `rust-harness/runtime/src/bin/gguf-info.rs:116` | Non-uniform feed-forward width is diagnosed as "MatFormer per-layer slicing"; on E2B the spec gives it a deterministic name and rule |
| GC-208 | low | `absent-undeclared` | `rust-harness/runtime/src/bin/gguf-info.rs:150` | gguf-info's architecture summary derives per-layer feed-forward widths but has no per-layer head-dimension line, and its key filter hides scale keys |
| GC-209 | low | `absent-declared` | `rust-harness/runtime/src/gguf.rs:4` | Safetensors is unhandled and the gap is declared: the runtime consumes GGUF v3 only, with no crate that could parse a safetensors header |
| GC-210 | low | `unrepresentable` | `rust-harness/runtime/src/kv.rs:29` | KvGeometry carries one head_dim "shared by every layer", so the KV budget understates every full-attention layer by exactly 2x |
| GC-213 | low | `guard-ineffective` | `rust-harness/runtime/src/model/cpu.rs:223` | The shared-KV refusal in check_supported keys on absent attn_k/attn_v, a signature the spec says never occurs in any Gemma 4 checkpoint |
| GC-214 | low | `mismatch` | `rust-harness/runtime/src/model/cpu.rs:245` | The per-layer feed-forward-width refusal is correct for E4B and fires for E2B, but names the cause MatFormer slicing where the spec names use_double_wide_mlp |
| GC-215 | low | `stale-doc` | `rust-harness/runtime/src/model/mod.rs:6` | mod.rs banner states a real Gemma 4 E4B backend implements the Model trait behind `cuda`; no such backend exists anywhere in the tree |
| GC-216 | low | `unrepresentable` | `rust-harness/runtime/src/model/mod.rs:31` | The Model trait exposes no cache geometry, so KV budgeting is computed outside the seam from a single head_dim that cannot express 256 sliding / 512 full |
| GC-217 | low | `stale-doc` | `rust-harness/runtime/src/model/mod.rs:37` | `prefill` contract claims cost is never proportional to cached_len, but every variant keeps 7 to 10 full-attention layers that attend the whole cache |
| GC-218 | low | `absent-declared` | `rust-harness/runtime/src/model/mod.rs:22` | The tree's only definition of Logits states no softcapping convention, though final_logit_softcapping 30.0 holds in all four variants |
| GC-220 | low | `mismatch` | `rust-harness/runtime/src/tokenizer.rs:443` | Tokenizer test fixture assigns <bos>=1 and <eos>=2, the exact transposition spec trap 8 warns against |
| GC-221 | low | `absent-undeclared` | `rust-harness/runtime/src/tokenizer.rs:584` | The tested prefix-overlap shape is fabricated and has no counterpart in Gemma 4; the shape that does occur is untested |
| GC-222 | low | `absent-undeclared` | `scripts/benchmark/test_perplexity.sh:254` | The perplexity gate's default -c 512 equals the E-series sliding_window exactly, so no default run ever exercises the sliding attention mask |
| GC-223 | low | `stale-doc` | `training/recipes/interp/harvest-smoke.yaml:13` | Shipped harvest recipes state d_model 2304 and a 30-layer stack for google/gemma-4-E2B |
| GC-224 | low | `absent-undeclared` | `tests/hardware/test_gpu.py:25` | FAMILY_PINS calls its Gemma 4 entries "pinned checkpoints" but pins no hub revision, and AutoConfig is called without one |
| GC-225 | low | `absent-undeclared` | `tests/interp/test_capture_guards.py:1` | No test in the audited slice asserts any Gemma 4 invariant, so the spec's "assert, do not read per model" direction has no enforcement here |
| GC-226 | low | `absent-declared` | `tests/interp/test_organisms.py:504` | Recipe-validation tests accept Gemma 4 recipes with no pinned hub revision; the schema declares the unset case, two of the six recipes do not |
| GC-227 | low | `mismatch` | `tests/interp/test_sae.py:154` | Layer-15 capture siting asserted six times is derived from a 30-layer stack; E2B has 35 layers and 15 is the first layer of the shared-KV / double-wide tail |
| GC-228 | low | `absent-undeclared` | `tests/interp/test_sae.py:320` | TestEvaluate injects activations and sae_fn on every path, so the hook string is never dereferenced against any module tree in this file |
| GC-229 | low | `absent-undeclared` | `tests/training/test_cli_handlers_prepare.py:378` | prepare-sft's --model-type is an unverified operator string: the assigned tests render gemma4 and qwen3 framing from the same tokenizer with no cross-check |
| GC-230 | low | `stale-doc` | `tests/training/test_data_tail_branches.py:529` | The fixture comment labels three non-Gemma-4 special tokens as "gemma4 framing" |
| GC-231 | low | `mismatch` | `tests/training/test_eval_runner.py:61` | Family-resolution tests fix on architecture string `Gemma4ForCausalLM`, which is not any Gemma 4 variant's architectures value |
| GC-232 | low | `mismatch` | `tests/training/test_offload.py:96` | The freeze-policy test fixture uses parameter names without the language_model segment the real E-series checkpoint carries |
| GC-233 | low | `absent-undeclared` | `tests/training/test_prepare.py:22` | No test in the tree exercises EOS resolution for a Gemma-4-style tokenizer, so M7-tests-01 is invisible to the suite |
| GC-234 | low | `absent-declared` | `tests/training/test_tokenizer.py:58` | From-scratch tokenizer fixes bos=0/eos=1/pad=2 against Gemma 4's pad=0/eos=1/bos=2, declared as a Phase 3 from-scratch path only |
| GC-235 | low | `absent-undeclared` | `training/calculator/components.py:89` | The calculator accepts 31B through gemma4_text while modelling none of attention_k_eq_v, num_global_key_value_heads, or use_bidirectional_attention |
| GC-236 | low | `unrepresentable` | `training/calculator/components.py:139` | ple_param_count books the packed table and the two per-layer linears but omits per_layer_model_projection, per_layer_projection_norm, and post_per_layer_input_norm |
| GC-237 | low | `absent-undeclared` | `training/calculator/components.py:249` | num_global_key_value_heads is read nowhere in the tree; harmless on the E-series where it is null, mispriced on 31B where it is 4 |
| GC-238 | low | `absent-declared` | `training/calculator/components.py:78` | 12B's separate class is excluded by every dispatch table with a loud error, and no code asserts tensor-level facts about 12B or 31B |
| GC-239 | low | `absent-undeclared` | `training/commands/download.py:104` | Nothing in the staging path verifies the staged tree against the spec's recorded config.json blob hashes |
| GC-240 | low | `absent-declared` | `training/data/teacher.py:68` | F4 teacher harness has no thinking-mode field and its manifest records none, with the gap declared in the investigation's risk 4 |
| GC-241 | low | `absent-declared` | `training/data/tokenizer.py:40` | From-scratch tokenizer fixes bos 0 / eos 1 / pad 2, the exact mirror of Gemma 4's pad 0 / eos 1 / bos 2, and no file names the transposition |
| GC-242 | low | `mismatch` | `training/eval/runner.py:51` | thinking_mode_for's substring family match silently resolves 12B's unified architecture strings onto the E-series entry |
| GC-243 | low | `absent-undeclared` | `training/models/families.py:120` | The seven multimodal and control token identifiers appear nowhere in the tree |
| GC-244 | low | `absent-undeclared` | `training/models/model.py:66` | attention_dropout and initializer_range appear nowhere in the tree; the from-scratch path inherits LlamaConfig defaults unasserted |
| GC-245 | low | `absent-undeclared` | `training/recipes/interp/harvest-smoke.yaml:18` | E2B's shared-KV / double-wide tail at layers 15 through 34 is recorded nowhere, and the interp hook sits on its first layer |
| GC-246 | low | `absent-declared` | `training/recipes/reference/continual-gemma4-e2b.yaml:31` | Three Gemma reference recipes leave model.hf.revision unpinned pending a hub pull, though the spec records the exact revisions offline |
| GC-247 | low | `stale-doc` | `training/recipes/reference/continual-gemma4-e2b.yaml:31` | The spec's pinned revisions and config blob hashes are not propagated to the recipes that pull these weights, which still record the revision as TBD |
| GC-248 | low | `stale-doc` | `training/runtime/offload.py:41` | The freeze-policy comment attributes an untied lm_head to Gemma 4, which the spec says the family does not have |

## Findings recorded without verification

The 83 findings below are recorded without adversarial verification. Low-severity
findings take this path by policy. Findings marked `unverified-over-budget` are not
low-value; they exceeded the verification budget and were demoted rather than dropped.
Zero recorded verdicts is unverified, never confirmed.

| ID | Severity | Status | Verdict | Site | Summary |
|----|----------|--------|---------|------|---------|
| GU-018 | high | unverified-over-budget | `unrepresentable` | `rust-harness/runtime/tests/cpu_reference.rs:346` | No fixture carries global_head_dim; the scalar q_dim/kv_dim model cannot express the 512-wide full-attention layers |
| GU-019 | high | unverified-over-budget | `absent-undeclared` | `rust-harness/fixtures/reference_logits.py:324` | No fixture models use_double_wide_mlp, so the calculator's 33% parameter undercount on E2B is undetectable |
| GU-056 | high | unverified-over-budget | `mismatch` | `training/recipes/interp/sae-false-fact.yaml:111` | Online distillation loss materializes four fp32 [B,S,262144] tensors, about 17 GB at the reference recipe geometry |
| GU-057 | high | unverified-over-budget | `mismatch` | `training/recipes/interp/harvest-smoke.yaml:66` | Logit-cache pre-pass batches 8 examples; at vocab 262144 that is a 17 GB logits tensor beside a 15 GB teacher |
| GU-080 | high | unverified-over-budget | `mismatch` | `training/data/prepare.py:42` | FALLBACK_EOS_TOKENS omits Gemma 4's `<eos>`, so prepare-sft against a local Gemma 4 tokenizer directory cannot resolve EOS and has no override flag |
| GU-015 | medium | unverified-over-budget | `mismatch` | `rust-harness/fixtures/tiny-gemma4.gguf:204` | Family-load test shrinks a Gemma 4 config to 2 layers, leaving num_kv_shared_layers, layer_types and the PLE fields full size |
| GU-016 | medium | unverified-over-budget | `needs-host` | `rust-harness/runtime/tests/cpu_reference.rs:111` | Real-config RAM test errors rather than skips when weights are absent, and its 75% bound hides a 33% undercount |
| GU-017 | medium | unverified-over-budget | `absent-undeclared` | `rust-harness/fixtures/reference_logits.py:91` | Every layer-stack fixture is featureless and 2-4 layers deep, so no attention schedule can be expressed or checked |
| GU-020 | medium | unverified-over-budget | `absent-undeclared` | `rust-harness/fixtures/reference_logits.py:322` | No fixture anywhere sets num_key_value_heads=1, so E2B's true multi-query path is never exercised |
| GU-024 | medium | unverified-over-budget | `absent-undeclared` | `training/calculator/components.py:232` | param_count models the language model only; the audio and vision towers in the same checkpoint are unpriced |
| GU-078 | medium | unverified-over-budget | `needs-host` | `training/export/convert.py:81` | The pinned llama.cpp converter is recorded as not mapping layer_scalar, and no export guard checks tensor coverage |
| GU-002 | low | unverified-by-policy | `absent-declared` | `docs/verification/rust-harness-qa.md:820` | Extends A1-15: the fixture's head_dim = embed_dim/n_heads matches no Gemma 4 variant; the deferral is declared honestly |
| GU-004 | low | unverified-by-policy | `stale-doc` | `rust-harness/Makefile:45` | The committed fixture the Makefile drives declares general.architecture "gemma4" while carrying a shape no Gemma 4 variant has |
| GU-038 | low | unverified-by-policy | `stale-doc` | `rust-harness/docs/observation-frames.md:61` | observation-frames calls phase-4 exports "scaled-Gemma" without saying which Gemma 4 features the backend refuses |
| GU-039 | low | unverified-by-policy | `mismatch` | `docs/inference/rust-harness-runtime-design.md:298` | PLE row's 2.82B figure transcribes the spec exactly, but its "of 8B total parameters" denominator overstates E4B's 7.52B |
| GU-040 | low | unverified-by-policy | `stale-doc` | `docs/inference/rust-harness-runtime-design.md:302` | "Value normalization" row asserts an attention step absent from the spec, and cpu.rs claims a refusal for it that does not exist |
| GU-041 | low | unverified-by-policy | `stale-doc` | `docs/inference/rust-harness-runtime-design.md:390` | VRAM-fit estimate is sized against the effective parameter count, which the spec shows excludes the 2.82B per-layer table |
| GU-042 | low | unverified-by-policy | `stale-doc` | `docs/inference/rust-harness-runtime-design.md:387` | Risk bullet still lists per-layer-embedding inheritance as unresolved, contradicting the same document two sections earlier |
| GU-043 | low | unverified-by-policy | `absent-undeclared` | `docs/inference/rust-harness-runtime-design.md:301` | rope_type `proportional` and its undetermined scaling formula appear nowhere outside the spec; the doc labels the regime settled |
| GU-045 | low | unverified-by-policy | `stale-doc` | `docs/testing/test_coverage_investigation.md:389` | Recorded Gemma 4 transformers floor of >= 5.5.0 covers 12B, which declares 5.10.0.dev0 |
| GU-046 | low | unverified-by-policy | `stale-doc` | `docs/training/investigations/f1_sft_data_investigation.md:116` | F1 places relative_k_proj in the Gemma 4 text decoder; the spec puts it in the audio-tower conformer |
| GU-047 | low | unverified-by-policy | `needs-host` | `docs/training/investigations/f3_sft_entry_point_investigation.md:234` | F3 asserts released transformers 5.13.1 ships gemma4; unverified in-tree, and the spec warns against assuming it |
| GU-049 | low | unverified-by-policy | `needs-host` | `docs/training/investigations/f8_recipe_cards_investigation.md:140` | F8's 'critical' KV-sharing training-bug claim is uncited, already promoted into a recipe preflight, and half-applied |
| GU-061 | low | unverified-by-policy | `needs-host` | `training/recipes/reference/continual-gemma4-e2b.yaml:10` | Recipes state the Gemma 4 weights are gated while the spec's reproduce curls are unauthenticated |
| GU-077 | low | unverified-by-policy | `guard-ineffective` | `docs/training/investigations/f5_distillation_investigation.md:261` | Extends KB-17: the documented softcap-parity preflight can never fire, because final_logit_softcapping is 30.0 in every variant |
| GU-003 | low | unverified-by-policy | `absent-declared` | `interp/families.py:49` | No gemma4_unified_text registry entry: 12B is refused by name with the supported set, the loud exclusion the spec asks for |
| GU-036 | low | unverified-by-policy | `needs-host` | `interp/harvest.py:182` | Interp loaders pass the transformers v4 torch_dtype kwarg while parts of the training tier use the v5 dtype name |
| GU-001 | low | unverified-by-policy | `stale-doc` | `docker/agents/proxy/squid.conf:38` | Spec's reproduce preamble says a sandbox cannot reach huggingface.co; squid.conf allowlists it and cites this spec as the reason |
| GU-044 | low | unverified-by-policy | `absent-undeclared` | `docs/models/gemma-4.md:293` | Six spec tensor families (per_layer_model_projection, per_dim_scale, embed_audio/embed_vision, ...) appear nowhere in the tree |
| GU-081 | low | unverified-by-policy | `absent-undeclared` | `training/commands/make_replay_synthetic.py:93` | The seven multimodal control identifiers appear nowhere in the repository and no code or doc records that they occupy in-vocabulary positions |
| GU-005 | low | unverified-by-policy | `absent-undeclared` | `rust-harness/runtime/src/gguf.rs:53` | The only modality-aware predicate in the runtime names vision alone; the E-series audio tower has no representation at all |
| GU-006 | low | unverified-by-policy | `mismatch` | `rust-harness/runtime/src/gguf.rs:292` | GGUF parser fixture builds a gemma4 image with token_embd [256000, 2048]; Gemma 4's vocabulary is 262144 |
| GU-007 | low | unverified-by-policy | `needs-host` | `rust-harness/runtime/src/gguf.rs:113` | Unsettled: if a converter writes feed_forward_length as a per-layer array, E2B is refused before the double-wide diagnostic runs |
| GU-008 | low | unverified-by-policy | `absent-declared` | `rust-harness/runtime/src/kv.rs:34` | budget_bytes gives every layer independent storage and never models num_kv_shared_layers; the module banner still calls the budget exact |
| GU-009 | low | unverified-by-policy | `unrepresentable` | `rust-harness/runtime/src/kv.rs:45` | The schedule is derived from a period rather than read from layer_types, so a non-periodic schedule is inexpressible |
| GU-010 | low | unverified-by-policy | `absent-undeclared` | `rust-harness/runtime/src/kv.rs:20` | num_global_key_value_heads, attention_k_eq_v and use_bidirectional_attention appear nowhere outside the spec document |
| GU-011 | low | unverified-by-policy | `absent-undeclared` | `rust-harness/runtime/src/kv.rs:34` | No weight-memory accounting exists in rust-harness; the per-layer table that dominates E-series residency is unrepresented |
| GU-012 | low | unverified-by-policy | `mismatch` | `rust-harness/runtime/src/bin/gguf-info.rs:150` | gguf-info's interesting-key filter omits softcapping/logit_scale and includes matformer |
| GU-013 | low | unverified-by-policy | `absent-declared` | `rust-harness/runtime/src/bin/gguf-info.rs:90` | bfloat16 is not a dequantizable storage type, so a bf16-preserving conversion is refused at load |
| GU-028 | low | unverified-by-policy | `absent-undeclared` | `rust-harness/runtime/src/model/cpu.rs:249` | num_global_key_value_heads is read nowhere, so 31B full-attention layers get 16 KV heads instead of 4 |
| GU-031 | low | unverified-by-policy | `unrepresentable` | `rust-harness/runtime/src/synth.rs:24` | TinySpec holds one head_dim and one ffn_dim, so no fixture carries layer-type-dependent shapes (extends A1-15) |
| GU-032 | low | unverified-by-policy | `absent-declared` | `rust-harness/runtime/src/tensor.rs:56` | Dequantizer implements F32/F16/Q8_0 only; bfloat16 storage is refused per tensor and declared by gguf-info |
| GU-033 | low | unverified-by-policy | `absent-undeclared` | `rust-harness/runtime/src/tokenizer.rs:129` | The special-token surface is bos and eos only: pad, unk, mask and the seven multimodal ids are unrepresented |
| GU-035 | low | unverified-by-policy | `absent-undeclared` | `rust-harness/runtime/src/gguf.rs:69` | validate_harvest bounds hook_layers only from below; num_hidden_layers is never read to bound them from above |
| GU-037 | low | unverified-by-policy | `absent-undeclared` | `rust-harness/runtime/src/tensor.rs:409` | Capture manifests record d_model but neither num_hidden_layers nor the layer's attention type, so a capture cannot be placed in the schedule |
| GU-060 | low | unverified-by-policy | `absent-declared` | `rust-harness/runtime/src/model/cpu.rs:31` | No Gemma 4 recipe pins model.hf.revision although the spec publishes all four commit SHAs |
| GU-065 | low | unverified-by-policy | `stale-doc` | `rust-harness/runtime/src/model/cpu.rs:16` | The header's unimplemented-feature list names 'value normalization', absent from the spec, and omits three real ones |
| GU-066 | low | unverified-by-policy | `absent-declared` | `rust-harness/runtime/src/model/cpu.rs:51` | One rope_freq_base and one frequency table serve both regimes, declared unimplemented in the header |
| GU-067 | low | unverified-by-policy | `needs-host` | `rust-harness/runtime/src/model/cpu.rs:833` | Rust and the numpy golden share one blocked GQA mapping; the n_kv_heads=1 fixture cannot falsify it — extends A1-15 |
| GU-068 | low | unverified-by-policy | `needs-host` | `rust-harness/runtime/src/model/cpu.rs:79` | vocab_size is the token-array length, so a short array would refuse a conforming 262144-row checkpoint |
| GU-069 | low | unverified-by-policy | `absent-declared` | `rust-harness/runtime/src/model/cpu.rs:367` | Every layer stores the full sequence, so E2B needs about 9.4GB and E4B 22.55GB of f32 cache at declared context |
| GU-070 | low | unverified-by-policy | `absent-undeclared` | `rust-harness/runtime/src/model/cpu.rs:62` | The same prefix test admits gemma2 and gemma3 images: the three-code-path split has no representation |
| GU-071 | low | unverified-by-policy | `mismatch` | `rust-harness/runtime/src/model/cpu.rs:463` | The 'roughly 1088 at E4B geometry' comment uses 35 layers; E4B has 42, which gives 1305 |
| GU-072 | low | unverified-by-policy | `needs-host` | `rust-harness/runtime/src/model/cpu.rs:156` | The 13 GGUF layer tensor names the loader consumes are a converter convention no artifact here confirms |
| GU-073 | low | unverified-by-policy | `needs-host` | `rust-harness/runtime/src/model/cpu.rs:175` | A converter that materializes the tied head as output.weight would be refused by the unconsumed-tensor backstop |
| GU-074 | low | unverified-by-policy | `stale-doc` | `rust-harness/runtime/src/model/cpu.rs:232` | The ffn-width refusal, gguf-info, README and runbook name MatFormer; the spec's rule is use_double_wide_mlp |
| GU-075 | low | unverified-by-policy | `absent-undeclared` | `rust-harness/runtime/src/model/cpu.rs:156` | layer_scalar, a [1] tensor on every E-series layer, is named nowhere in the repository outside the spec |
| GU-014 | low | unverified-by-policy | `absent-undeclared` | `rust-harness/runtime/tests/cuda_acceptance.rs:29` | Phase A logits fixture recipe elides the whole command and names no transformers version or model revision |
| GU-021 | low | unverified-by-policy | `absent-undeclared` | `rust-harness/fixtures/reference_logits.py:317` | No fixture carries the E-series discriminator fields, and no test asserts the spec's cross-variant invariants |
| GU-022 | low | unverified-by-policy | `absent-declared` | `rust-harness/harness/tests/agent_e2e.rs:41` | Both family registries pin gemma4/gemma4_text only, so 12B's gemma4_unified_text is refused but never tested |
| GU-034 | low | unverified-by-policy | `absent-undeclared` | `rust-harness/runtime/tests/cpu_reference.rs:243` | No fixture carries non-uniform feed-forward widths, so E2B's double-wide refusal is asserted by inspection only |
| GU-048 | low | unverified-by-policy | `stale-doc` | `rust-harness/runtime/src/tokenizer.rs:261` | F5 calls final_logit_softcapping an optional field; the spec makes it an invariant 30.0 (extends KB-17) |
| GU-055 | low | unverified-by-policy | `stale-doc` | `training/runtime/offload.py:46` | offload.py claims its freeze patterns are verified against the pinned revision; the test fixture uses invented names |
| GU-023 | low | unverified-by-policy | `mismatch` | `training/calculator/components.py:153` | param_count books 2 norms per layer; E-series layers carry 5 plus q_norm, k_norm and layer_scalar |
| GU-025 | low | unverified-by-policy | `needs-host` | `training/calculator/components.py:232` | text_config descent keys on the literal "gemma4"; the spec never states the E-series top-level model_type |
| GU-026 | low | unverified-by-policy | `absent-declared` | `training/calculator/components.py:80` | 12B (gemma4_unified_text) is absent from HF_SUPPORTED_MODEL_TYPES and refused loudly, not silently excluded |
| GU-027 | low | unverified-by-policy | `stale-doc` | `training/calculator/components.py:5` | Module docstring and activation comment call the MLP SwiGLU/SiLU; Gemma 4 is gelu_pytorch_tanh (GeGLU) |
| GU-029 | low | unverified-by-policy | `absent-undeclared` | `training/calculator/components.py:245` | max_position_embeddings and sliding_window are never read, so seq_len is unchecked against the declared 131072 |
| GU-030 | low | unverified-by-policy | `absent-declared` | `training/calculator/components.py:245` | num_kv_shared_layers and attention_k_eq_v have no field on ArchitectureSpec; benign while use_cache is off |
| GU-050 | low | unverified-by-policy | `unrepresentable` | `training/recipes/schema.py:83` | ScratchModelConfig carries one scalar rope_theta; Gemma 4 needs two RoPE regimes, so any single value is wrong |
| GU-051 | low | unverified-by-policy | `unrepresentable` | `training/recipes/schema.py:81` | ScratchModelConfig has no head_dim, layer_types, PLE or per-layer FFN fields, so no Gemma 4 variant is expressible |
| GU-052 | low | unverified-by-policy | `absent-undeclared` | `training/recipes/validate.py:456` | No validation bounds data.seq_len or the harvest context length against max_position_embeddings |
| GU-053 | low | unverified-by-policy | `stale-doc` | `training/runtime/offload.py:42` | offload.py and schema.py both describe an untied lm_head; the E-series ties embeddings and ships no lm_head tensor |
| GU-054 | low | unverified-by-policy | `stale-doc` | `training/runtime/offload.py:44` | The per_layer freeze pattern also freezes post_per_layer_input_norm, which the comment beside it does not enumerate |
| GU-058 | low | unverified-by-policy | `guard-ineffective` | `training/models/families.py:157` | extends KB-17: softcap parity cannot fire on any pair clearing the tokenizer gate, since 30.0 is a family invariant |
| GU-059 | low | unverified-by-policy | `unrepresentable` | `training/models/families.py:134` | model_type gemma4_text cannot separate E2B/E4B from 31B, so the registry's E-series-only scope is unexpressible |
| GU-062 | low | unverified-by-policy | `needs-host` | `training/recipes/reference/distill-logit-gemma-e2b.yaml:43` | Two F8 distill recipes name teacher google/gemma-4-E4B-it, a repository the spec's provenance table omits |
| GU-063 | low | unverified-by-policy | `absent-declared` | `training/recipes/reference/sft-qlora-gemma-e4b.yaml:18` | E4B SFT card names the E-model KV-sharing risk under use_cache=False as a host preflight rather than assuming it |
| GU-064 | low | unverified-by-policy | `stale-doc` | `training/recipes/reference/sft-qlora-gemma-e4b.yaml:19` | The E4B SFT card's KV-sharing preflight cites docs/training/f8_recipe_cards_investigation.md, which does not exist |
| GU-076 | low | unverified-by-policy | `absent-undeclared` | `training/calculator/solve.py:47` | solve("seq") searches to MAX_SEQ_LEN 1048576 and never consults max_position_embeddings (131072 on the E-series) |
| GU-079 | low | unverified-by-policy | `absent-undeclared` | `training/export/merge.py:184` | Export staging copies only weights, config.json and generation_config.json, dropping the processor_config.json the E-series repositories ship |
| GU-082 | low | unverified-by-policy | `absent-undeclared` | `training/export/merge.py:22` | Export staging copies every tokenizer artifact but not processor_config.json, which the Gemma 4 repository ships for Gemma4Processor |
| GU-083 | low | unverified-by-policy | `absent-declared` | `training/data/tokenizer.py:40` | The from-scratch tokenizer fixes <\|bos\|>=0, <\|eos\|>=1, <\|pad\|>=2, a different assignment from Gemma 4's pad=0, eos=1, bos=2 |

## Refuted findings

Refuted findings stay in the record with their refutation. A campaign that deletes what it
disproved teaches the next campaign nothing, and the same claim returns.

### GR-009 Shared-KV row reads cache sharing as weight omission, and the guard built from it keys on a signature that never occurs

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `guard-ineffective` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-runtime-design.md:300` |
| Variants | E2B, E4B |
| Claims | C12-01, C12-10, C12-11, C12-23, C13-02, C6-01, C6-02, C6-03, C6-04, C6-05, C8-07 |
| Verification | 3 verdicts recorded, 2 refutations |

**Reference.** docs/models/gemma-4.md:314-316: "`num_kv_shared_layers` does **not** indicate shared weights. The safetensors headers show that every layer carries its own `k_proj` and `v_proj`: 42 of each on E4B, 35 of each on E2B, with no layer omitted. The sharing applies to the key-value cache at inference time, not to the projection parameters."

**Implementation.** docs/inference/rust-harness-runtime-design.md:300: "\| Shared KV cache \| Last 18 of 42 layers reuse K/V from the last non-shared layer \| No \| Medium \|", and :322-323: "diagnoses per-layer embeddings, shared KV, per-layer feed-forward widths, and non-single-theta RoPE by name and fails the load"

**Consequence.** Row :300 ("Last 18 of 42 layers reuse K/V from the last non-shared layer") encodes weight-level reuse. Spec 314-316: every layer carries its own k_proj/v_proj, 42 on E4B and 35 on E2B, with no layer omitted; the sharing is cache-time only. The refusal written from this row (cpu.rs:221-229) tests has("attn_q") && !(has("attn_k") && has("attn_v")), so on any real Gemma 4 checkpoint it cannot fire and the doc's assurance at :321-323 that the backend refuses rather than guesses is false for this feature. gguf-info.rs:140 goes further and prints the affirmative "Shared KV cache: none, all N layers carry attn_k/attn_v". Realised consequence is spec failure mode 2 (spec 323-326): every layer gets independent cache storage, the model loads and runs, memory over-allocates silently. This row is the propagation root: cpu.rs:219-220 repeats its wording verbatim, and it reaches rust-harness/README.md:104, docs/verification/rust-harness.md:98/109/134, and the training tier.  Propagated to: rust-harness/runtime/src/model/cpu.rs:223, rust-harness/runtime/src/bin/gguf-info.rs:140, rust-harness/runtime/tests/cpu_reference.rs:25, rust-harness/runtime/src/model/cpu.rs:1214, rust-harness/README.md:104, docs/verification/rust-harness.md:109. One defect: cache sharing read as weight omission, so every shared-KV check keys on absent attn_k/attn_v - a signature the spec says never occurs. The runtime-design reconciliation row is the documentary ROOT (the finding itself says the guard was built from it); cpu.rs:223 is the guard, gguf-info.rs:140 prints the same verdict (README:106 states the tool mirrors the loader), cpu_reference.rs:25 and cpu.rs:1214 test the impossible signature, and README:104 plus RH.1 question 2 advertise it. Note S18-C6-docs-04 at the same doc line is NOT folded in: it is the separate defect of stating the unverified 18-layer boundary as settled.

- upheld: Spec-fidelity lens only. /work/docs/models/gemma-4.md:314-317 reads verbatim as quoted: "`num_kv_shared_layers` does **not** indicate shared weights. The safetensors headers show that every layer carries its own `k_proj` and `v_proj`: 42 of each on E4B, 35 of each on E2B, with no layer omitted. The sharing applies to the key-value cache at inference time, not to the projection parameters." (finding cites 314-316; sentence ends at 317 - immaterial). The claim is unhedged, not an Open Question: sourced to the safetensors headers (:57, :287), corroborated by the tensor inventory at :383-384 (35x k_proj [512,2560] + 7x k_proj [1024,2560] = 42 on E4B), and restated as item 2 of the "obvious reading is wrong" checklist at :562-563. Both failure modes the finding relies on are present: :321-322 phrases the loader case generically ("A loader that expects 18 of E4B's 42 layers to omit key and value projections fails to load the weights, because those tensors are present"), which is exactly the omission-signature premise; :323-326 is failure mode 2, load-and-run with silent cache over-allocation, as cited. Variant scope matches the [E4B/E2B] label (both counts given; :255-258 lists 20/18/0/0). The genuinely uncertain neighbouring statement - the E4B tail boundary, called "unverified" at :330-333 and listed as an Open Question at :597-599 - is explicitly excluded by the finding's closing note, so no Open Question is treated as settled. Only gap noted, not refuting: the spec never mentions GGUF (grep for gguf on the spec is empty) while the guard keys on GGUF names attn_k/attn_v, but :321-322's loader-generic phrasing covers that step. Whether the runtime-design row at :300 actually encodes weight-level rather than cache-level reuse is a doc/code reading question outside this lens.
- REFUTED: Site check fails on two independent grounds. (1) The cited line, /work/docs/inference/rust-harness-runtime-design.md:300, reads "\| Shared KV cache \| Last 18 of 42 layers reuse K/V from the last non-shared layer \| No \| Medium \|". It names the feature as a cache and describes cache reuse; it never states or implies that k_proj/v_proj tensors are absent. The spec uses materially the same phrasing (gemma-4.md:347-348, "The tail layers share key-value cache", and gemma-4.md:562, "shares cache, not weights"), so the row is consistent with spec 314-316. The weight-level inference exists only in rust-harness/runtime/src/model/cpu.rs:219-220 ("...so they carry no attn_k/attn_v of their own"); the trailing clause appears nowhere in the design doc (grep of every shared/KV occurrence: lines 300, 322, 332, 334). So the finding's "cpu.rs repeats its wording verbatim" and its designation of :300 as the documentary root are inaccurate. (2) Handling the auditor missed: check_supported bails at cpu.rs:211-217 on any tensor name containing "per_layer", and per the spec's own config table (gemma-4.md:149, :252-258) num_kv_shared_layers is nonzero only on E2B (20) and E4B (18) - exactly the models carrying per-layer input embeddings (0 on 12B/31B). Every Gemma 4 checkpoint that actually has KV sharing therefore fails the load at cpu.rs:650 before the shared-KV loop and before any cache is sized, with the generic unconsumed-tensor backstop at cpu.rs:304-328 as a second net. The claimed realized consequence (model loads and runs, cache silently over-allocated, spec failure mode 2) cannot occur in this backend. Residual real defects - the dead clause at cpu.rs:223, the false comment at :220, and the misleading "Shared KV cache: none" line at gguf-info.rs:140 - are code-side and at the propagated sites, not at the cited root, and are low severity given the earlier refusal.
- REFUTED: Traced every loader entry point (harness/src/main.rs:188, activation-capture.rs:204, gguf-info.rs:164) into CpuModel::from_gguf, which calls check_supported at cpu.rs:650 before reading any weight, with no bypass path. Inside check_supported the per-layer-embedding check (cpu.rs:211-217, `names.iter().find(\|n\| n.contains("per_layer"))`) runs BEFORE the shared-KV loop at cpu.rs:221-229. Per the authoritative spec, every E2B/E4B checkpoint carries per-layer embedding tensors (gemma-4.md:289-301, 366-380; hidden_size_per_layer_input=256 at :150), so the load is refused at the first check; even absent the substring match, the generic unconsumed-tensor backstop (cpu.rs:304-328) refuses them, E2B also trips the MatFormer width check (gemma-4.md:341-344), and dual RoPE trips cpu.rs:275-278. KvCache (cpu.rs:367+) is therefore never constructed for E2B/E4B, so the finding's stated realised consequence -- spec failure mode 2, "model loads and runs, memory over-allocates silently" -- is unreachable in this repo. 12B/31B cannot reach it either (num_kv_shared_layers = 0, gemma-4.md:149). No Python/training consumer exists (grep for shared_kv/num_kv_shared returns only the two docs, a triage memory file, and cpu.rs), and the cited cpu_reference.rs:25 is an unrelated golden-logits load of the synthetic fixture, not a test of the shared-KV signature. What survives is real but milder: cpu.rs:221-229 is dead code keyed on a signature the spec says never occurs, docs/inference/rust-harness-runtime-design.md:321-323 overstates that shared KV is refused by name, and gguf-info.rs:139-140 prints the misleading "Shared KV cache: none, all 42 layers carry attn_k/attn_v" during the RH.1 reconciliation -- mitigated because gguf-info.rs:148-158 prints every metadata key containing "shared" in the same dump and the literal tensor claim matches gemma-4.md:314-316. Documentation/diagnostics accuracy defect with no runtime effect, not a critical guard failure.

### GR-010 KV budget claimed "computable exactly at startup" from a formula assuming one head dimension for all layers and no cache sharing

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `unrepresentable` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-runtime-design.md:163` |
| Variants | E2B, E4B |
| Claims | C1-02, C2-12, C3-07, C6-05, C6-09 |
| Verification | 3 verdicts recorded, 2 refutations |

**Reference.** docs/models/gemma-4.md:180-181: "Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512)." And spec 149: "\| `num_kv_shared_layers` \| 20 \| 18 \| 0 \| 0 \|", spec 323-326: "A cache allocator that gives all 42 layers independent key-value storage loads and runs correctly but consumes substantially more memory than the architecture requires."

**Implementation.** docs/inference/rust-harness-runtime-design.md:162-164: "Sliding-window layers cap their cache at window size regardless of context length, so the total budget is computable exactly at startup: fixed cost for windowed layers plus a per-token cost for global layers only."

**Consequence.** Lines 161-164 state the budget as exact. kv.rs:18-31 implements it literally: one KvGeometry { n_kv_heads, head_dim, bytes_per_element } "shared by every layer of a model", with bytes_per_token = 2 * n_kv_heads * head_dim * bytes_per_element, summed in budget_bytes (kv.rs:34-43). Spec 180-181 gives full-attention layers head_dim 512 against the sliding 256, so a budget computed at 256 under-allocates exactly the layers whose cache grows with context; and ignoring num_kv_shared_layers (spec 149: 18 on E4B, 20 on E2B) over-allocates the tail. The two errors do not cancel and neither is visible at startup, which is the one property the sentence claims.

- upheld: Every spec citation in GC-005 is verbatim and settled, not over-read. /work/docs/models/gemma-4.md:180-182 reads "The two attention types use **different head dimensions**. Sliding layers use `head_dim` (256); full-attention layers use `global_head_dim` (512). This is visible directly in the weight shapes and is not an inference from the configuration." — reinforced by the invariants table (110-111) and checklist item 6 (568-569), and proven at tensor level for both audited variants by the key/value projection shapes: E4B k_proj/v_proj [512,2560] sliding vs [1024,2560] full (189-190), E2B [256,1536] vs [512,1536] (200-201). So per-token KV bytes provably differ 2x between layer types on E2B and E4B. Line 149 is verbatim ("\| `num_kv_shared_layers` \| 20 \| 18 \| 0 \| 0 \|"), restated at 255-258 as forfeiting "the memory saving that makes these models fit in a single GPU". Lines 323-326 are verbatim on the over-allocating cache allocator. The only nearby uncertainty is the Open Question at 597-599 (and 331-333): the E4B shared-KV *boundary* (which 18 layers) is assumed, not verified. The finding's word "tail" brushes that, but its substance — that 18 layers' worth of independent cache is redundant on E4B — rests on the settled count and the settled failure mode, not on the open boundary. No statement anywhere in the spec supports a single uniform head dimension or walks back cache sharing.
- REFUTED: Traced the only non-test caller of kv.rs: harness/src/main.rs:249-267 calls interleaved_layers/KvGeometry/budget_bytes solely to print "[INFO] KV budget: ... estimated X GiB f32" and a [WARN] above 2 GiB. Nothing allocates from it — CpuModel's cache is a Vec grown per token via cache.k.extend_from_slice (cpu.rs:820-822), never sized from budget_bytes — so the claimed "under-allocates the global layers / over-allocates the shared tail" consequence describes an allocator that does not exist. Moreover the line is unreachable with E2B/E4B: main.rs:188 loads the model first, and CpuModel::from_gguf runs check_supported (cpu.rs:210-327) before any weight, which hard-bails on any tensor containing "per_layer" (every E-series checkpoint has them per gemma-4.md:287-297), on any rope key beyond a single freq_base or a dimension_count below head_dim (dual RoPE, gemma-4.md:226-240), and on any unconsumed tensor family; strict length checks on attn_q/attn_q_norm (cpu.rs:665-685) would also reject dual head dims. Only single-head-dim, no-KV-sharing synthetic fixtures reach budget_bytes, for which the formula is correct. main.rs:232-237 additionally already documents in-tree that the estimate does not match what the backend stores. The doc sentence at rust-harness-runtime-design.md:161-164 is inaccurate versus the spec, but that is a prose defect with no reachable memory consequence.
- REFUTED: The finding's causal chain breaks at every step after the quoted text. (a) No allocation: kv::budget_bytes has one non-test caller, harness/src/main.rs:249-267, which only prints "[INFO] KV budget: ... estimated ... GiB f32" and a >2 GiB warning; the actual cache is struct LayerKv { k: Vec<f32>, v: Vec<f32> } (cpu.rs:371-374) grown with extend_from_slice per token (cpu.rs:821-822), so there is nothing to "under-allocate". docs/verification/rust-harness-qa.md:565 independently notes budget_bytes is never tied to the real allocation. (b) Guard in another function the auditor missed: CpuModel::from_gguf calls check_supported before loading any weight (cpu.rs:646-650), and main.rs:188 runs that before the budget block at main.rs:249. check_supported (cpu.rs:206-329) bails on any tensor name containing "per_layer" (present in every E2B/E4B checkpoint per spec 289-297), on the shared-KV signature, on any RoPE metadata key beyond a single freq_base (E4B dual-theta/p-RoPE), and on any unconsumed tensor family; additionally every tensor is length-checked against the single head_dim (cpu.rs:658-693, e.g. attn_q_norm wants head_dim elements, wk wants embed_dim*n_kv_heads*head_dim), so a checkpoint with 512-wide full-attention layers is rejected. An E4B/E2B model therefore never reaches budget_bytes. (c) The same document scopes the claim: lines 280-286 state the CPU backend was written against a Gemma-3-shaped specification, line 300 marks Shared KV cache "Implemented: No", and lines 321-327 state check_supported refuses per-layer embeddings, shared KV, MatFormer widths, and non-single-theta RoPE. (d) The cited sentence contains no head-dimension term; "computable exactly at startup" is a claim about the structural decomposition, and spec 149/180-181 change only the constants — layer_types, head_dim, global_head_dim, and num_kv_shared_layers are all static config, so the true E4B budget remains exactly computable at startup. A genuine but different inexactness exists (main.rs:232-237 admits the CPU backend caches f32 K/V for the full sequence in every layer, making the printed number a lower bound), but that is not the chain GC-005 asserts.

### GR-011 Shared-KV row states the E4B 18-layer boundary as settled fact where the spec explicitly declares it unverified

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/inference/rust-harness-runtime-design.md:300` |
| Variants | 12B, 31B, E2B, E4B |
| Claims | C1-11, C10-02, C10-12, C10-13, C13-18, C6-01, C6-03, C6-05, C6-08, C8-01, C9-07 |
| Verification | 1 verdicts recorded, 1 refutations |

**Reference.** docs/models/gemma-4.md:330-333: "The corresponding E4B region of 18 layers is indices 24 through 41 if the same tail convention holds; this has not been independently confirmed from the weights, since E4B's MLP widths are uniform and provide no corroborating signal. Treat the E4B boundary as unverified." and 323-326: "A cache allocator that gives all 42 layers independent key-value storage loads and runs correctly but consumes substantially more memory than the architecture requires."

**Implementation.** docs/inference/rust-harness-runtime-design.md:300: "\| Shared KV cache \| Last 18 of 42 layers reuse K/V from the last non-shared layer \| No \| Medium \|"

**Consequence.** Distinct from the weight-versus-cache misreading at the same row: this is the confidence claim. Spec 330-333 says the E4B region of 18 layers is indices 24 through 41 "if the same tail convention holds; this has not been independently confirmed from the weights, since E4B's MLP widths are uniform and provide no corroborating signal. Treat the E4B boundary as unverified", and open question 4 (spec 597-599) repeats it. The row states "Last 18 of 42" flatly, at Medium confidence sourced from the secondhand list at :329-333. A cache implementation written from this row hardcodes a boundary and a donor-layer rule the source of truth declines to certify; the 18-of-42 count itself is correct (spec 149).

- REFUTED: Opened /work/docs/inference/rust-harness-runtime-design.md:295-337. Line 300 is not a bare factual assertion: it is a table row in a table whose fourth column is "Confidence", and the row is graded "Medium" with Implemented="No". The paragraph immediately below, which the finding did not weigh, supplies the hedging it says is absent: :314-316 states "Two caveats bound these findings. They are secondhand... the confidence column reflects how many independent sources agreed" (so the Medium marker IS this document's unverified flag); :317-319 strips the whole table of authority ("what governs the code is not the published architecture but what the GGUF file on the workstation stores... Runbook entry RH.1 remains the authority"); :321-327 precludes the posited failure mode ("the backend refuses rather than guesses. cpu::check_supported diagnoses... shared KV... by name and fails the load"). That refusal is real and verified in code: /work/rust-harness/runtime/src/model/cpu.rs:206 pub fn check_supported, called from CpuModel::from_gguf (cpu.rs:650) and gguf-info (bin/gguf-info.rs:164), bails at cpu.rs:221-230 with "this backend gives every layer its own cache and does not implement KV sharing" — so no cache allocator can be written from row :300 without first deleting a load-time refusal. The finding also concedes the Medium grade and secondhand sourcing in its own text, meaning it is arguing hedge wording, not missed hedging; it concedes the 18-of-42 count is correct (gemma-4.md:149); the spec itself asserts the E4B tail flatly at gemma-4.md:355 ("despite also having a shared-KV tail"); and grep finds no donor-layer statement anywhere in gemma-4.md, so the source of truth does not "decline to certify" the donor clause — it is silent on it. The only residual is that the row omits a literal cross-reference to gemma-4.md:332-333/597-599, a wording nit, not a medium-severity stale-doc defect.

### GR-001 Phase 5 DPO doubling is applied to an already halved base, so "2B and 4B remain pure-GPU" compounds the ladder error

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/training/training-scale-reference.md:186` |
| Variants | E2B, E4B |
| Claims | C5-14, C9-07, C9-10 |
| Verification | 1 verdicts recorded, 1 refutations |

**Reference.** spec 438-439: "Continual pretraining plans that size against the effective parameter count underestimate resident weight memory by roughly a factor of two."

**Implementation.** docs/training/training-scale-reference.md:180-187: "Preference methods in the DPO family need a frozen reference model alongside the policy, roughly doubling resident weight memory. This pushes each variant one rung down the ladder above. ... With those, 2B and 4B remain pure-GPU".

**Consequence.** docs/training/training-scale-reference.md:179-186 states the DPO reference model roughly doubles resident weight memory and "pushes each variant one rung down the ladder above", concluding "With those, 2B and 4B remain pure-GPU". The doubling is applied to the effective-count rows the ladder already understates (see S32-M4-lineage-04-root). Against spec:428, a frozen bf16 reference copy of E2B is 4.65e9 x 2 = 9.3 GB, not the ~4 GB a 2B label implies, and E4B is 15.0 GB, not ~9 GB - two rungs down, not one. The stated mitigations (precomputed reference log-probabilities, adapter-based reference) still stand; only the unmitigated pure-GPU claim for E2B and E4B fails. Separate site from the ladder table and needs its own edit even after the ladder is corrected.

- REFUTED: Read /work/docs/training/training-scale-reference.md:179-188 in full. The quotation is accurate but the mechanism is not. (1) The pure-GPU conclusion at 185-186 opens with "With those", scoping it to the two mitigations named one sentence earlier - precomputed reference log-probabilities (no reference model resident) and adapter-based reference (the single base copy is the reference). Both eliminate the second copy, so the doubling the finding says is "applied to an already halved base" is not in force at the cited sentence. (2) The finding's own arithmetic supports the sentence: 4.65e9 x 2 = 9.3 GB (E2B) and 7.52e9 x 2 = 15.0 GB (E4B) both fit 32 GB, so on true un-halved counts 2B and 4B do remain pure-GPU under the adapter mitigation. (3) The doc makes no unmitigated pure-GPU claim for 4B - "one rung down the ladder above" moves 4B to the 12B row, which the ladder at lines 99-105 marks "offload or PEFT". (4) "Two rungs down, not one" conflates weight bytes with the ladder's controlling per-parameter state budget (lines 59-64: ~13 and ~7 bytes/param); a frozen bf16 reference adds 2 bytes/param, i.e. +15-29%, not a doubling of the rung. (5) Handling elsewhere: lines 87-89 route these budgets into the memory-fit calculator, and /work/training/calculator/components.py:140-157 sums ple_param_count() (per-layer embedding table plus per-layer linears, populated from vocab_size_per_layer_input / hidden_size_per_layer_input at 254-255) into total_params, so the operational sizing path does not inherit the halved base; /work/docs/training/memory_fit.md:72-76 states this. The ladder prose itself carries no per-layer-table caveat (grep for per-layer/effective param/4.65/7.52/E2B/E4B returns nothing in that file), but that is the separate root ladder finding; this site adds no independent compounding error.

### GR-002 Extends A1-R04: the refutation settles the second base only, not partial_rotary_factor 0.25 or rope_type proportional

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `needs-host` |
| Tier | `docs` |
| Site | `/work/docs/verification/rust-harness-qa.md:454` |
| Variants | all |
| Claims | C13-15, C4-05, C4-06 |
| Verification | 1 verdicts recorded, 1 refutations |

**Reference.** docs/models/gemma-4.md:233-236 "First, full-attention layers rotate only a **quarter of each head's dimensions**. With `global_head_dim` of 512, `partial_rotary_factor` of 0.25 means 128 dimensions receive rotary treatment and 384 pass through unrotated. Applying rotation to all 512 dimensions is a silent numerical error."

**Implementation.** docs/verification/rust-harness-qa.md:454-466 "**A1-R04 (refuted).** ... The decisive question was whether the converter writes `{arch}.rope.freq_base_swa`, and it does: `Gemma3Model` calls the base class first, which harvests `rope_local_base_freq` and emits the key, and gemma3n and gemma4 subclass it. ... The wrong behavior is unreachable - every path refuses first."

**Consequence.** docs/verification/rust-harness-qa.md:454-466 refutes the single-rope_freq_base claim by establishing that the converter emits {arch}.rope.freq_base_swa, and that is sound for the second-base question. Its scope is narrower than "every path refuses first" reads: spec:233-236 states that full-attention layers rotate 128 of 512 dimensions and that "applying rotation to all 512 dimensions is a silent numerical error", and that fact travels on a different key (rope.dimension_count), through a guard that compares against the single Config::head_dim (cpu.rs:265-273) - so a file declaring dimension_count == head_dim passes while the global-layer rotation is still wrong. The refutation also says nothing about rope_type proportional, whose formula the spec records as an open question. Its own escape clause ("the gemma4 encoding fails earlier still in parsing") means no Gemma 4 encoding was actually driven through check_supported, so the evidence is about gemma3 encodings. Not a reopening of A1-R04: a scope annotation. No Gemma 4 weights exist on this workstation, so this becomes a runbook entry rather than a resolved fact.

- REFUTED: Read qa.md:454-466, cpu.rs:254-296/645-700/1236-1262, synth.rs:241, gemma-4.md:110-236,568-569. (1) The doc entry is self-scoped: its topic sentence names "the claim that one rope_freq_base is silently applied to sliding-window and global layers alike", so "the wrong behavior is unreachable" refers to that claim; it never asserts partial rotation is settled. (2) rope_type proportional IS handled: cpu.rs:275-278 is a catch-all `_ => bail!` over every {arch}.rope.* key other than freq_base and dimension_count ("dual or scaled RoPE is not implemented"), which is exactly where llama.cpp writes rope scaling type/factor; tested at cpu.rs:1253. (3) The finding's constructed file ("dimension_count == head_dim passes while global-layer rotation is still wrong") cannot be a Gemma 4 checkpoint: the spec pins head_dim 256 vs global_head_dim 512 as an invariant of all four variants (gemma-4.md:110-111, 180-182, trap 6 at 568-569), and CpuModel::from_gguf (cpu.rs:658-695) length-checks every per-layer tensor (attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_output) against the single config.head_dim, erroring by tensor name on the first full-attention layer. Rotation width and projection width are the same head_dim here, so the rotation cannot be wrong while the shapes agree. Additionally the second rope base and final_logit_softcapping 30.0 (an all-variant invariant, gemma-4.md:115) each independently refuse any faithful Gemma 4 file. (4) "No Gemma 4 encoding was driven through check_supported" is false for the repo: the fixture declares general.architecture = gemma4 (synth.rs:241) and the rope tests push gemma4.rope.* keys through check_supported, including the accepted dimension_count == head_dim case.

### GR-003 P2.6's expected signal has the operator confirm a frozen LM head; E2B and E4B carry no lm_head tensor

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `stale-doc` |
| Tier | `docs` |
| Site | `/work/docs/verification/training-phase2.md:167` |
| Variants | E2B |
| Claims | C1-07, C1-17, C8-10 |
| Verification | 1 verdicts recorded, 1 refutations |

**Reference.** spec 132-134: "`tie_word_embeddings` is confirmed in the weights, not only in the configuration: neither E2B nor E4B contains an `lm_head` tensor. The output projection reuses `model.language_model.embed_tokens.weight`."  spec 400: "There is **no `lm_head`**. Output logits reuse the tied input embedding."

**Implementation.** docs/verification/training-phase2.md:166-168: "Expected signal: exit 0; DeepSpeed initializes ZeRO stage 2 with CPU optimizer offload; frozen parameters (embeddings, LM head, per-layer tables) never enter the optimizer groups".

**Consequence.** spec:132-134 and 400 record that neither E2B nor E4B contains an lm_head tensor and that output logits reuse the tied embedding, so the lm_head substring in FROZEN_PATTERNS (training/runtime/offload.py:50) matches nothing on the E2B recipe this step runs. The behavior is correct - freezing embed_tokens freezes the tied projection - but the runbook's expected signal names three frozen groups where only two exist, leaving the operator to decide whether the third is a failure. Verdict changed mismatch->stale-doc: the site is runbook prose, not a value the code encodes. The same wrong assertion survives in the offload.py:41-43 comment ("an untied output head lm_head"), which is outside this chunk's file set.

- REFUTED: The cited line (/work/docs/verification/training-phase2.md:166-168) says "frozen parameters (embeddings, LM head, per-layer tables) never enter the optimizer groups". Under the spec (/work/docs/models/gemma-4.md:132-134, :400) the LM head is the tied input embedding, so it is in fact frozen and never enters the optimizer groups — the sentence is true and never asserts a separate lm_head tensor exists (unlike /work/training/runtime/offload.py:42 and /work/training/recipes/schema.py:93, which say "untied lm_head" and are outside this site). The recipe P2.6 actually runs, /work/training/recipes/reference/continual-gemma4-e2b.yaml:6-8, enumerates the same three items and explicitly writes "the tied LM head", so the runbook prose restates the repo's own governing artifact. The claimed harm also has no mechanism: initialize_engine (/work/training/runtime/offload.py:230-264) prints nothing about frozen names or per-pattern match counts, and apply_trainable_filter (:69-97) raises only when the filter freezes nothing or everything — on E2B, embed_tokens and per_layer both match, so no output ever surfaces an unmatched lm_head for an operator to misread as a failure.

### GR-006 The independent reference applies no final_logit_softcapping, so the golden certifies an uncapped forward pass

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `absent-undeclared` |
| Tier | `tests` |
| Site | `/work/rust-harness/fixtures/reference_logits.py:230` |
| Variants | all |
| Claims | C1-06 |
| Verification | 1 verdicts recorded, 1 refutations |

**Reference.** docs/models/gemma-4.md:115 — "\| `final_logit_softcapping` \| 30.0 \|", listed under "These values are identical in E2B, E4B, 12B, and 31B" (:105).

**Implementation.** reference_logits.py:229-230 `        x = rms_norm(x, tensors["output_norm.weight"], eps)` / `        all_logits.append((embed @ x).astype(np.float32).tolist())` — the tied projection is the last operation; there is no `tanh(logits/30)*30` step anywhere in the file, and `final_logit_softcapping` appears nowhere in fixtures/ or in either tests directory.

**Consequence.** reference_logits.py:229-230 ends at output_norm and the tied projection; `final_logit_softcapping` (spec:115, an all-variant invariant) appears nowhere in fixtures/ or either tests directory. Softcapping carries no tensor, so the unconsumed-tensor backstop cannot see it; the only defence is the metadata-key bail at runtime/src/model/cpu.rs:289-299, whose effectiveness depends on whether the gemma4 converter emits a softcapping key at all -- unknowable here, and the same needs-host question as tests-03. Graded medium rather than high: the tiny fixture declares no softcapping, so the reference is self-consistent and today's defect is a verification gap rather than a wrong number.

- REFUTED: The site reading is literally correct (reference_logits.py:229-230 ends at output_norm plus the tied projection, no tanh cap anywhere in the file), but the finding's residual claim - that the only defence is an untested bail of unknowable effectiveness - fails. /work/rust-harness/runtime/src/model/cpu.rs:646-650 calls check_supported inside the production loader CpuModel::from_gguf before any weight is read; cpu.rs:288-300 refuses every {arch}.-prefixed key whose suffix contains "softcapping" (a substring match, not a named allowlist, so key-naming variants cannot slip past); and cpu.rs:1283-1308 (numeric_metadata_with_no_tensor_to_notice_is_refused) inserts the exact spec key/value ("gemma4.final_logit_softcapping", 30.0) and asserts the load must fail naming the key - the finding's "appears nowhere in either tests directory" is true only because this is an inline #[cfg(test)] module under src/. The omission is also declared, contradicting the absent-undeclared class: cpu.rs:16-22 states the backend is "a strict subset of the published Gemma 4 architecture" and that extra structure "is refused by check_supported rather than loaded ... silently dropped", and the bail message repeats it. The converter hedge is likewise answerable repo-locally: /work/training/export/convert.py:43-44 delegates to llama.cpp convert_hf_to_gguf.py, whose gguf-py keys are {arch}.attn_logit_softcapping / {arch}.final_logit_softcapping, both caught by the substring test and both pinned by name in the cpu.rs test. A capped Gemma 4 checkpoint therefore cannot reach the golden's math; it is rejected at load with a positioned error. The only residual is forward-looking (implementing softcapping later must update the numpy reference in the same change), not a present conformance defect at the cited site.

### GR-007 The reference rotates every head dimension; partial rotary (128 of 512) has no representation in any fixture

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `absent-undeclared` |
| Tier | `tests` |
| Site | `/work/rust-harness/fixtures/reference_logits.py:134` |
| Variants | all |
| Claims | C13-05, C13-15, C4-02, C4-05, C4-07 |
| Verification | 1 verdicts recorded, 1 refutations |

**Reference.** docs/models/gemma-4.md:233-236 — "First, full-attention layers rotate only a **quarter of each head's dimensions**. With `global_head_dim` of 512, `partial_rotary_factor` of 0.25 means 128 dimensions receive rotary treatment and 384 pass through unrotated. Applying rotation to all 512 dimensions is a silent numerical error."

**Implementation.** reference_logits.py:133-134 `def rope(head, position, freq_base):` / `    half = head.shape[0] // 2` — the loop at :136 runs `for j in range(half)`, so all `head.shape[0]` dimensions are rotated. Neither `partial_rotary_factor` nor `rope.dimension_count` appears anywhere in fixtures/ or in either tests directory.

**Consequence.** reference_logits.py:133-137 rotates head.shape[0]//2 pairs, i.e. all dimensions, against spec:233-236 (128 rotated, 384 passed through on full-attention layers). Neither partial_rotary_factor nor rope.dimension_count appears in fixtures/ or either tests directory, and the fixture's head_dim of 8 offers no case where the rotated count differs from the head width, so the declared guard at cpu.rs:265-273 is never exercised. Spec:587-590 records the `proportional` scaling formula as undetermined, so a fixture written today would assert an unsupported assumption -- the correct outcome is a refusal, and proving the refusal is what is missing.

- REFUTED: The literal code claim holds (reference_logits.py:133-143 rotates all head dims via half=shape[0]//2 over pairs (j, j+half)), but the finding's operative claim — that the p-RoPE refusal is unproven and the guard at cpu.rs:265-273 is never exercised — is false. /work/rust-harness/runtime/src/model/cpu.rs:254-280 refuses any gemma4.rope.dimension_count != head_dim with a "p-RoPE is not implemented" error and bails on every other rope.* key (catching a second freq base or proportional scaling); check_supported is called on the real load path from CpuModel::from_gguf at cpu.rs:650. The refusal is directly tested at cpu.rs:1236-1251 (partial_rotary_dimension_count_is_refused), which injects head_dim/2 and asserts the error, then asserts head_dim is accepted. The auditor's search scoped to fixtures/ and tests/ directories missed the inline #[cfg(test)] module, a placement /work/.styles/RUST_STYLE.md explicitly sanctions. The reference script also never receives the problematic input: it runs only against fixtures/tiny-gemma4.gguf, which carries no rope.dimension_count, and any checkpoint that declared partial rotary is rejected before a forward pass, so the spec's "silent numerical error" cannot occur here. The unimplemented status is additionally declared in docs/inference/rust-harness-runtime-design.md:301.

### GR-008 The fixture makes n_heads * head_dim equal embed_dim, so o_proj is square and a width conflation is undetectable (extends A1-15)

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `absent-declared` |
| Tier | `tests` |
| Site | `/work/rust-harness/fixtures/reference_logits.py:215` |
| Variants | 12B, 31B, E2B, E4B |
| Claims | C3-08, C3-09, C3-10, C3-11, C3-13, C3-14, C3-15, C3-16, C3-18 |
| Verification | 1 verdicts recorded, 1 refutations |

**Reference.** docs/models/gemma-4.md:206-208 — "The shapes follow `num_attention_heads * head_dim` for sliding layers and `num_attention_heads * global_head_dim` for full layers, with the key and value projections scaled by `num_key_value_heads` instead." E2B sliding `o_proj.weight` is `[1536, 2048]` (:202).

**Implementation.** reference_logits.py:215 `            projected = (t("attn_output.weight") @ attn.reshape(-1)).astype(np.float32)` — against the committed fixture's `blk.0.attn_output.weight dims=[16, 16]`, since `embedding_length` 16 = `head_count` 2 x `key_length` 8. Every attention matmul in the golden is square.

**Consequence.** reference_logits.py:215 projects attn.reshape(-1) through attn_output.weight against a [16, 16] fixture tensor, because synth.rs:39-43 sets embed_dim 16 = head_count 2 x head_dim 8. On every real variant the two widths differ (E2B 8 x 256 = 2048 against hidden_size 1536, spec:202), so o_proj is rectangular. A backend sizing the attention scratch or the o_proj input at embed_dim passes cpu_reference.rs, activation_capture.rs, probe_identity.rs and cpu_smoke.rs and fails only on the workstation. docs/verification/rust-harness-qa.md:467 (A1-15) declares the coincidence but names only its QK-scale consequence, not the rectangular-projection one, and no second fixture shape was added.

- REFUTED: The literal citation checks out: reference_logits.py:215 does project attn.reshape(-1) through attn_output.weight, and parsing the committed tiny-gemma4.gguf myself shows embedding_length=16, head_count=2, key_length=8, blk.*.attn_output.weight dims=[16,16] (square), matching synth.rs:39-43. But the declared failure mode ("undetectable", "fails only on the workstation", "no second fixture shape was added") is refuted by handling the auditor missed: (1) synth.rs:236,266 writes attn_output.weight as [q_out=n_heads*head_dim, d] parameterized on the two widths independently, so a second shape needs no new committed file; (2) runtime/src/model/cpu.rs:1346-1367, the_shape_check_accepts_a_geometry_the_committed_fixture_cannot_express, builds TinySpec{embed_dim:16, n_heads:4, n_kv_heads:2, head_dim:6} — q_dim 24 != embed_dim 16, o_proj [24,16] = 384 elements — loads it and runs a full prefill asserting finite full-vocab logits; it is a plain #[test] with no #[ignore] and no cfg(feature) in the file, so it runs under cargo test --workspace, and its comment names the very coincidence the finding says is undeclared; (3) two hard non-debug guards fire under the hypothesized bug: the loader's wo: t("attn_output.weight", q_dim * embed_dim) at cpu.rs:673-687 with q_dim derived from n_heads*head_dim, and matvec_into's assert_eq!(w.len(), in_dim*out_dim) at cpu.rs:541-548 whose doc comment states it is a real assertion specifically to catch "a Scratch buffer sized against the wrong dimension" (384 != 16*16 panics), plus Scratch::new/carve/assert!(rest.is_empty()) at cpu.rs:494,781-795. The only surviving true statement is the much narrower one that the Python cross-language golden never numerically cross-checks a rectangular projection.

### GR-004 No handling for the channel/tool-call marker family: a thought channel in model output reaches ToolCall::parse as a malformed call

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `absent-undeclared` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/harness/src/chat.rs:98` |
| Variants | all |
| Claims | C10-12, C10-13 |
| Verification | 1 verdicts recorded, 1 refutations |

**Reference.** docs/models/gemma-4.md:477-480 "\| Channel \| `<\\|channel>` \| `<channel\\|>` \| \| Tool call \| `<\\|tool_call>` \| `<tool_call\\|>` \| ..."; :482 "A reasoning marker `<\|think\|>` and an escape token `<\|\"\|>` are also defined."

**Implementation.** chat.rs:98 `let reply = trim_reply(&output?).to_string();` - trim_reply (chat.rs:23-29) knows only TURN_CLOSE; nothing anywhere in harness/src matches `<\|channel>`, `<channel\|>`, `<\|tool_call>`, `<tool_call\|>`, `<\|think\|>`, or `<\|\"\|>`

**Consequence.** Gemma 4 defines five asymmetric marker pairs plus <\|think\|> and <\|"\|> (gemma-4.md:476-482); the harness models exactly one pair and names none of the rest. chat.rs:98 persists the raw reply through trim_reply, which knows only TURN_CLOSE. Two reachable consequences: (1) a reply opening with <\|channel>thought ... <channel\|> - plausible per docs/training/investigations/f1_sft_data_investigation.md:211-213 - is persisted verbatim and, in agent mode, fails ToolCall::parse so agent.rs:189-192 appends "tool error: ..." and burns a turn on a well-formed reply; (2) the harness's tool protocol is a bare JSON object in model text while the model was tuned to emit <\|tool_call>-delimited calls, so a model following its own training issues a call the harness never recognizes. Nothing in harness/src acknowledges either gap.

- REFUTED: The claimed causal chain fails at its first step. /work/rust-harness/harness/src/tools.rs:111-115 begins `ToolCall::parse` with `let trimmed = text.trim(); if !trimmed.starts_with('{') { return Ok(None); }`, so a reply opening with `<\|channel>thought ... <channel\|>` returns Ok(None) before any JSON parse. In /work/rust-harness/harness/src/agent.rs:151-152 that hits `Ok(None) => return Ok(output)` — returned as the final answer. The Err arm at agent.rs:189-192 requires trimmed to start with `{` plus a JSON/schema error, so no "tool error: ..." is appended and no turn is burned for the described input; the finding's title ("reaches ToolCall::parse as a malformed call") is contradicted by the code it cites. The site is also mis-attributed: chat.rs is the REPL without the agent loop (chat.rs:5-7, "No tools, permissions, or grammar are involved") and never calls ToolCall::parse, so nothing can flow from chat.rs:98 to ToolCall::parse. The "absent-undeclared" label is further undercut by /work/rust-harness/harness/src/template.rs:1-11, which explicitly declares the turn and function-calling format provisional pending the Gemma 4 model card. What survives is only the milder, different fact that markers are not stripped, so a channel-wrapped reply is persisted verbatim and silently returned as the final answer.

### GR-005 Architecture summary tests per-layer uniformity for ffn_up only, so the 35/7 attention split stays invisible

| Field | Value |
|-------|-------|
| Severity | low |
| Verdict | `absent-undeclared` |
| Tier | `rust-harness` |
| Site | `/work/rust-harness/runtime/src/bin/gguf-info.rs:99` |
| Variants | all |
| Claims | C3-07, C3-08, C3-09, C3-10, C3-11, C3-12, C3-18 |
| Verification | 1 verdicts recorded, 1 refutations |

**Reference.** spec:181-182 "This is visible directly in the weight shapes and is not an inference from the configuration."; spec:381-388 "\| 35 \| `model.language_model.layers.{L}.self_attn.q_proj.weight` \| `[2048, 2560]` \| / \| 7 \| ... \| `[4096, 2560]` \|"

**Implementation.** gguf-info.rs:99-101 "    // Feed-forward widths: one line when uniform, a table when MatFormer\n    // slicing varies them per layer.\n    let widths: Vec<(String, u64)> = file"; gguf-info.rs:81-84 "    println!(\"Tensor families:\");\n    for (family, count) in &families {\n        println!(\"  {family}  x{count}\");"

**Consequence.** The summary applies the right test (are per-layer shapes uniform?) to the one family that is uniform on E4B and to none of the six attention families that are not. Families print as blk.*.attn_q.weight x42 with no dims (gguf-info.rs:81-84), and the four RH.1 questions (docs/verification/rust-harness.md:108-113) never ask about head dimensions, so the entry designed to settle the architecture cannot surface what spec:181-182 calls directly visible in the weight shapes and lists as trap 6 at :568-569.

- REFUTED: The finding's descriptive half is accurate (gguf-info.rs:99-121 tests per-layer uniformity for ffn_up only; lines 81-84 print family counts without dims; the four RH.1 questions at docs/verification/rust-harness.md:108-113 never ask about head dimension), but its stated failure -- that the entry "cannot surface" the directly-visible shape split -- is contradicted by the same binary. gguf-info.rs:59-64 prints "{name}  dims={:?}  type=..  offset=.." for every tensor, and RH.1's command is an unfiltered run whose output the operator reports verbatim, so a 35/7 split appears as blk.0.attn_q.weight dims=[2560, 2048] against blk.5.attn_q.weight dims=[2560, 4096], plus the [256] vs [512] q_norm/k_norm contrast, in the section immediately above the summary. Second missed guard: CpuModel::from_gguf (rust-harness/runtime/src/model/cpu.rs:658-694) length-checks each attention tensor against config-derived sizes -- wq against embed_dim*q_dim, q_norm against head_dim -- with ensure!(values.len() == want, "tensor {name} has {} elements, the configuration implies {want}"), so a split checkpoint errors at load naming the tensor rather than loading with wrong strides. Residual, not carrying the finding: cpu::check_supported (cpu.rs:206-320) does not name the head-dim split among the features it diagnoses, so a hypothetical split-only checkpoint would print "[OK] CPU backend compatibility: supported" from gguf-info before from_gguf refused it; on the real E4B this is moot because check_supported bails first on per_layer tensors (cpu.rs:211-217).

## Conformant points checked

The 104 points below were checked and found to agree with the reference. This
section exists so that a reader can distinguish what was checked and found correct from
what was never checked.

### docs

| ID | Site | Claims | Agrees on |
|----|------|--------|-----------|
| GM-002 | `docs/inference/future-optimizations.md:55` | C1-03 | Distillation memory analysis uses the exact vocabulary size 262144, matching the spec invariant |
| GM-060 | `docs/inference/rust-harness-runtime-design.md:297` | C8-01, C8-11 | AltUp row (Dropped, Not applicable) agrees with the spec's exact E4B tensor inventory |
| GM-061 | `docs/inference/rust-harness-runtime-design.md:301` | C1-13, C1-14, C13-04, C13-05, C4-01, C4-03, C4-04, C4-05, C4-06 | Dual RoPE row transcribes both regimes correctly: p = 0.25 with theta 1M on global, plain theta 10k on local |
| GM-062 | `docs/inference/rust-harness-runtime-design.md:304` | C2-02, C2-09, C2-12, C3-01, C3-03, C3-06 | Sliding-window row matches E4B exactly (512 tokens, 5:1 over 42 layers), as does the shared-KV row's 18-of-42 count |
| GM-063 | `docs/inference/rust-harness-runtime-design.md:267` | C5-01, C5-14, C9-02, C9-08 | E4B target stated as ~4.5B effective parameters, and the PLE row's 2.82B, both match the spec's figures |
| GM-064 | `docs/inference/rust-harness-runtime-design.md:172` | C1-03, C10-11 | Vocabulary described as ~256K entries, matching the 262144 all-variant invariant |
| GM-065 | `docs/inference/rust-harness-runtime-design.md:284` | C1-07, C1-17, C8-10 | Tied LM head assumption matches the spec's weight-level confirmation that no lm_head tensor exists |
| GM-004 | `docs/interpretability/investigations/p4_d1_model_organisms_investigation.md:151` | C9-05, C9-06, C9-07, C9-08, C9-10 | P4-D1's approximately 2.3B effective parameters for E2B tracks the spec's 2.28B body-plus-embedding row |
| GM-005 | `docs/interpretability/investigations/p4_d2_activation_harvesting_investigation.md:141` | C5-14, C9-02, C9-07, C9-09 | MATCH: P4-D2 sizes E2B at approximately 10GB bf16 including the PLE table, not the effective count |
| GM-066 | `docs/models/gemma-4.md:558` | C13-01 | No document overclaims 256K context for the E-series; the only Gemma 4 context constant in the tree is 131072 |
| GM-009 | `docs/training/investigations/p2_d5_optimizer_offload_investigation.md:219` | C1-07, C1-17, C2-10, C2-11, C5-10, C8-10 | Match: p2_d5 states the tied-embedding consequence and the packed PLE table formula exactly as the spec does |
| GM-010 | `docs/training/memory_fit.md:74` | C1-03, C1-12, C10-11, C13-09, C2-10, C2-11, C2-19 | Match: vocabulary 262144, the enable_moe_block refusal, and the PLE gate on hidden_size_per_layer_input are all recorded correctly |
| GM-003 | `docs/verification/rust-harness-qa.md:456` | C13-04, C4-05, C4-06 | Match: the dual-RoPE refusal fires on the converter's freq_base_swa key, so the A1-R04 refutation holds as recorded |
| GM-006 | `docs/verification/rust-harness.md:112` | C4-01, C4-03, C4-04, C4-05 | RH.1 question 4 asks the dual-and-partial RoPE question the spec requires; entry is PENDING with no artifact |
| GM-007 | `docs/verification/rust-harness.md:108` | C5-01, C5-15 | RH.1 question 1 asks for per-layer embedding tensors, the spec's top-impact E-series feature |
| GM-019 | `rust-harness/README.md:105` | C7-02, C7-04, C7-07, C7-08 | Match: per-layer feed-forward-width refusal fires on E2B's 20 double-wide layers, silent on uniform E4B |
| GM-020 | `rust-harness/README.md:101` | C3-01, C3-02, C3-03, C3-06, C3-25 | Match: interleaved sliding/global schedule reproduces spec full-attention indices for E2B (5) and E4B (6) |
| GM-021 | `rust-harness/README.md:101` | C1-07, C1-17, C8-10 | Match: README's tied LM head agrees with the spec's confirmed absence of any lm_head tensor |
| GM-022 | `rust-harness/README.md:100` | C13-13, C2-06, C3-19, C3-26 | Match: loader accepts n_kv_heads = 1, so E2B's multi-query attention is not excluded |
| GM-052 | `rust-harness/docs/live-events.md:70` | C13-01, C2-13 | Match: harness docs size against a 128k context, the E-series max_position_embeddings, never the 256K release figure |
| GM-053 | `rust-harness/docs/optimization-candidates.md:52` | C5-14, C9-05, C9-07, C9-09 | Match: the near-4 GB Q4 E4B resident estimate is consistent with the spec's 7.52B total including the per-layer table |

### interp

| ID | Site | Claims | Agrees on |
|----|------|--------|-----------|
| GM-008 | `interp/families.py:81` | C11-07, C11-13, C12-12, C12-13, C12-22, C2-01, C2-03 | unwrap_text_config and every consumer read hidden_size and model_type from text_config, not the vision tower |
| GM-046 | `interp/harvest.py:218` | C12-22, C2-10, C2-19, C5-01, C5-12, C5-15 | Match: check_hook_dims asserts the observed hook width equals text_config.hidden_size, refusing a 256-wide PLE capture |
| GM-047 | `interp/probes/capture.py:102` | C3-07, C3-08, C3-13, C3-18, C3-19, C3-26, C7-02, C7-04 | Match: the capture and SAE stack reads only decoder-block output width, so double-wide MLP and dual head dim cannot leak into d_in |
| GM-048 | `interp/probes/capture.py:108` | C10-14 | Match: batched capture gathers the last real token first-one-from-the-right, correct for the spec's left padding_side |
| GM-049 | `interp/sae/train.py:184` | C2-01, C2-03 | Match: SAE training takes d_in from the capture manifest's measured d_model, never a hardcoded hidden_size |
| GM-050 | `interp/steering/export.py:46` | C3-01, C3-02, C3-06 | Match: the tier's 1-based (llama.cpp) and 0-based layer conventions are each asserted at their artifact boundary |

### rust-harness

| ID | Site | Claims | Agrees on |
|----|------|--------|-----------|
| GM-011 | `rust-harness/harness/src/generator.rs:35` | C1-03, C10-11 | Vocabulary is taken whole from the checkpoint and never trimmed or assumed to be a fixed size |
| GM-012 | `rust-harness/harness/src/generator.rs:66` | C1-15, C10-02, C13-08 | BOS and EOS ids come from checkpoint metadata, so the counter-intuitive eos=1 / bos=2 ordering cannot be transposed |
| GM-013 | `rust-harness/harness/src/main.rs:238` | C10-15, C13-01, C2-13, C2-20 | Context budget is read from the checkpoint and capped there; no 256K assumption and no model_max_length sentinel anywhere |
| GM-014 | `rust-harness/harness/src/main.rs:254` | C2-06, C2-07, C3-19, C3-26 | E2B's single key-value head is safe here: the KV budget multiplies by n_kv_heads and derives no group size |
| GM-015 | `rust-harness/harness/src/main.rs:252` | C2-12 | Sliding-window size is read from checkpoint metadata, not hardcoded to a variant's value |
| GM-023 | `rust-harness/runtime/src/bin/gguf-info.rs:164` | C11-07, C11-08, C11-11, C13-07 | Match: architecture dispatch is a gemma prefix match, so it does not exclude the 12B unified path |
| GM-024 | `rust-harness/runtime/src/bin/gguf-info.rs:107` | C7-02, C7-03, C7-04 | Match: gguf-info prints per-layer feed-forward widths, surfacing E2B's 15/20 split on the workstation |
| GM-016 | `rust-harness/runtime/src/gguf.rs:157` | C12-11, C13-12 | Zero-dimensional tensors parse and count as one element, so the towers' clipped-linear companions do not break the reader |
| GM-017 | `rust-harness/runtime/src/gguf.rs:291` | C13-01, C2-13, C2-20 | The only Gemma 4 context constant in the tree is the E-series 131072, and nothing hardcodes a context length elsewhere |
| GM-018 | `rust-harness/runtime/src/kv.rs:47` | C2-02, C2-20, C3-02, C3-03, C3-04, C3-05, C3-06 | interleaved_layers reproduces all four published attention schedules exactly, including the final-layer-is-global property |
| GM-079 | `rust-harness/runtime/src/model/cpu.rs:275` | C10-12, C10-13, C13-04, C13-15, C4-01, C4-02, C4-03, C4-04, C4-06, C4-07 | Any second or scaled RoPE key is refused by an allowlist over {arch}.rope.*, including the proportional type |
| GM-080 | `rust-harness/runtime/src/model/cpu.rs:144` | C13-06, C2-12, C3-01, C3-02, C3-03, C3-04, C3-05, C3-06, C3-25, C5-05, C5-06, C5-09, C5-13, C9-03 | The stride formula reproduces every published full-attention index, including the final-layer rule |
| GM-081 | `rust-harness/runtime/src/model/cpu.rs:775` | C13-13, C2-05, C2-06, C2-07, C3-18, C3-19, C3-26 | E2B's single key-value head is handled: the divisibility guard admits 1 and the head mapping degenerates correctly |
| GM-082 | `rust-harness/runtime/src/model/cpu.rs:211` | C13-09, C2-10, C2-19, C5-01, C5-03, C5-04, C5-05, C5-06, C5-07, C5-08, C5-09, C5-11, C5-12, C5-13, C5-15 | Per-layer input embeddings are refused by tensor-name substring before any weight is read |
| GM-083 | `rust-harness/runtime/src/model/cpu.rs:304` | C1-08, C1-09, C1-12, C12-11, C12-23, C13-16, C13-17, C5-07, C5-08, C5-09, C5-15, C6-06, C6-07, C6-08, C8-04, C8-11, C8-12, C8-13 | The unconsumed-tensor backstop refuses layer_scalar, the fifth norm, biases, experts and tower tensors by family |
| GM-084 | `rust-harness/runtime/src/model/cpu.rs:175` | C1-03, C1-07, C1-17, C10-11, C8-02, C8-10 | The LM head is tied to token_embd and no output.weight is expected, matching the absence of lm_head |
| GM-085 | `rust-harness/runtime/src/model/cpu.rs:79` | C1-03, C10-11 | Vocabulary width comes from the token array and is cross-checked against the embedding row count |
| GM-086 | `rust-harness/runtime/src/model/cpu.rs:808` | C1-05, C3-20, C8-02, C8-03, C8-11 | Four sandwich norms per layer in the published order, plus per-head q_norm/k_norm sized to the head dimension |
| GM-087 | `rust-harness/runtime/src/model/cpu.rs:95` | C1-05 | rms_norm_eps is read from checkpoint metadata rather than hardcoded, and every norm call uses it |
| GM-088 | `rust-harness/runtime/src/model/cpu.rs:599` | C1-04 | The activation is tanh-approximated GELU on the gate branch, matching gelu_pytorch_tanh with a GeGLU MLP |
| GM-089 | `rust-harness/runtime/src/model/cpu.rs:83` | C13-01, C2-02, C2-03, C2-04, C2-05, C2-12, C2-13, C2-20, C8-05, C8-06 | The scalar per-variant fields map one-to-one onto GGUF metadata reads with no hardcoded geometry |
| GM-090 | `rust-harness/runtime/src/model/cpu.rs:837` | C2-14, C3-21, C3-23 | Attention reads separate K and V caches, the correct path for attention_k_eq_v false on the E-series |
| GM-091 | `rust-harness/runtime/src/model/cpu.rs:88` | C2-08, C3-24 | No num_global_key_value_heads is required or defaulted, which is right for the E-series where the field is null |
| GM-092 | `rust-harness/runtime/src/model/cpu.rs:61` | C11-06, C11-07, C11-08, C11-11, C13-07 | Architecture dispatch is a gemma prefix test, so it does not fall into the gemma4_text exact-match trap |
| GM-093 | `rust-harness/runtime/src/model/cpu.rs:242` | C13-03, C2-15, C7-01, C7-02, C7-03, C7-04, C7-06, C7-07, C7-10 | The feed-forward width guard fires on E2B's 20 double-wide layers, naming the tensor and both widths |
| GM-094 | `rust-harness/runtime/src/model/cpu.rs:673` | C3-13, C3-14, C3-15, C3-16, C3-18, C9-01 | Sliding-layer projection shapes agree exactly because head_dim is read, not derived from hidden_size / n_heads |
| GM-040 | `rust-harness/runtime/src/synth.rs:301` | C1-07, C1-17, C8-10 | The fixture emits no output.weight and ties the LM head to token_embd, matching the confirmed weight tying |
| GM-041 | `rust-harness/runtime/src/synth.rs:49` | C1-05 | TinySpec's rms_eps is the invariant 1e-06 |
| GM-042 | `rust-harness/runtime/src/synth.rs:42` | C13-13, C2-06, C2-07, C3-26 | Single-KV-head MQA is the fixture default, and no code derives a group size that breaks at n_kv_heads 1 |
| GM-043 | `rust-harness/runtime/src/synth.rs:281` | C3-20 | Q/K norms are emitted and applied per head at the head width, matching Gemma 4's per-head QK normalization |
| GM-044 | `rust-harness/runtime/src/tensor.rs:24` | C12-11, C13-12 | The GGUF reader treats a zero-dimensional tensor as a scalar, tolerating the tower clipping companions |
| GM-045 | `rust-harness/runtime/src/tokenizer.rs:129` | C1-11, C1-15, C10-02, C13-08 | The tokenizer reads bos and eos ids from metadata and bounds-checks them, assuming no ordering |

### schemas

| ID | Site | Claims | Agrees on |
|----|------|--------|-----------|
| GM-051 | `pyproject.toml:19` | C11-13, C2-18 | Match: transformers==5.13.1 exceeds both dev versions the Gemma 4 configs target (5.5.0.dev0 / 5.10.0.dev0) |
| GM-025 | `rust-harness/runtime/src/bin/activation-capture.rs:207` | C2-02, C2-03 | Match: capture writers bound layer indices and row widths against the loaded model, not a constant |
| GM-026 | `schemas/gguf_artifacts.json:25` | C1-01, C2-02, C2-03, C5-11 | All three cross-language artifact contracts are dimension-agnostic: no layer count, width, or head geometry bound |

### tests

| ID | Site | Claims | Agrees on |
|----|------|--------|-----------|
| GM-028 | `rust-harness/fixtures/reference_logits.py:339` | C1-07, C1-17, C8-10 | The MoE refusal keys on enable_moe_block, false in all four dense variants, so it cannot misfire on E2B/E4B |
| GM-029 | `rust-harness/fixtures/reference_logits.py:412` | C13-13, C2-06, C2-07, C3-19, C3-26 | The distillation softcap parity test pins 30.0 and reads it from text_config, matching the invariant |
| GM-030 | `rust-harness/fixtures/reference_logits.py:96` | C3-01, C3-02, C3-03, C3-04, C3-05, C3-06, C3-25 | The freeze-policy fixture names the real PLE tensors and treats KV sharing as cache-only, matching the spec |
| GM-054 | `rust-harness/fixtures/reference_logits.py:200` | C3-01, C3-02, C3-03, C3-04, C3-05, C3-06, C3-25 | Match: the reference reads the layer-type period from metadata per model rather than hardcoding a stride |
| GM-055 | `rust-harness/fixtures/reference_logits.py:230` | C1-07, C1-17, C8-10 | Match: the reference projects logits from the tied input embedding with no lm_head, and the fixture carries no output.weight |
| GM-056 | `rust-harness/fixtures/reference_logits.py:128` | C1-04 | Match: the reference activation is the tanh-approximated GELU the spec names (gelu_pytorch_tanh) |
| GM-057 | `rust-harness/fixtures/reference_logits.py:185` | C3-20 | Match: the reference applies per-head RMS normalization to queries and keys before RoPE |
| GM-058 | `rust-harness/fixtures/reference_logits.py:179` | C13-21, C3-21, C3-23 | Match: key and value stay separate tensors and separate caches, correct for the E-series where attention_k_eq_v is false |
| GM-059 | `rust-harness/fixtures/tiny-gemma4.gguf:1` | C1-05 | Match: the fixture's RMSNorm epsilon is the spec's 1e-06 to f32 precision |
| GM-027 | `rust-harness/harness/tests/cpu_smoke.rs:346` | C10-16, C10-17 | head_dim decoupling, the 256-wide PLE gate and the PLE table width formula are all pinned correctly |
| GM-031 | `rust-harness/runtime/tests/cuda_acceptance.rs:75` | C10-02, C10-12, C11-01, C11-03, C11-13 | SFT goldens use the asymmetric turn markers, a single leading <bos>, and a chat-template version guard |
| GM-001 | `rust-harness/runtime/tests/eval_acceptance.rs:65` | C11-01, C11-04, C11-13 | Shard writer test pins uint32 for the 262144-entry vocab and round-trips id 262143 |

### training

| ID | Site | Claims | Agrees on |
|----|------|--------|-----------|
| GM-032 | `training/calculator/components.py:238` | C1-12 | enable_moe_block refusal reads the descended text_config and does not misfire on the four dense variants |
| GM-033 | `training/calculator/components.py:137` | C11-08, C11-10, C11-11, C13-07, C13-09, C2-10, C2-11, C2-19, C5-10, C5-14, C5-15 | Per-layer embeddings gated on hidden_size_per_layer_input, not the always-populated vocabulary field |
| GM-034 | `training/calculator/components.py:139` | C5-01, C5-02, C5-03, C5-04, C5-07, C5-08, C5-10, C5-11, C5-14, C9-01, C9-02, C9-07, C9-09, C9-10 | Packed PLE table priced as vocab x layers x 256, reproducing the spec's E2B and E4B counts exactly |
| GM-035 | `training/calculator/components.py:129` | C13-13, C2-06, C2-07, C2-08, C3-19, C3-24, C3-26 | E2B's single key-value head is handled by multiplication only, with no division or minimum-of-two assumption |
| GM-036 | `training/calculator/components.py:151` | C2-14, C3-21, C3-23, C6-01, C6-02, C6-03, C6-04 | Every layer is priced with its own k_proj and v_proj, matching num_kv_shared_layers sharing cache not weights |
| GM-037 | `training/calculator/components.py:155` | C1-03, C1-07, C1-17, C10-11, C8-10, C9-01 | Tied embeddings counted once with no lm_head term, and the full 262144 vocabulary is used unmodified |
| GM-038 | `training/calculator/components.py:151` | C1-08, C1-09 | No attention bias or dropout term in the parameter formulas, matching attention_bias false and dropout 0.0 |
| GM-039 | `training/calculator/components.py:253` | C1-01 | head_dim is read explicitly rather than derived as hidden // heads, and a unit test pins the decoupling |
| GM-099 | `training/commands/download.py:42` | C11-01 | The single documented hub-staging path refuses to run without an exact revision |
| GM-095 | `training/data/distill.py:44` | C1-03 | match: the logit cache stores token ids as int32, which holds the full 262144-entry vocabulary |
| GM-101 | `training/data/prepare.py:125` | C10-03 | The --eos-token unk guard reads unk_token_id defensively, so a tokenizer that does not state an unk id does not break it |
| GM-096 | `training/data/shards.py:38` | C1-03 | Match: the 262144 vocabulary is carried correctly through the shard dtype selection and the replay generator |
| GM-103 | `training/data/shards.py:40` | C1-03, C11-05, C13-10, C10-12, C10-13, C2-02 | Vocabulary dtype, absent shard index, chat-turn markers and the layer-count read all agree with the spec |
| GM-098 | `training/export/convert.py:88` | C11-07, C11-08 | GGUF export dispatches on config.json architectures[0], the field the spec's group table keys on |
| GM-097 | `training/export/merge.py:185` | C13-10, C11-05 | No loader in the tree requires model.safetensors.index.json; the shard index is copied when present and never demanded |
| GM-100 | `training/export/merge.py:26` | C11-04 | Export tokenizer staging tolerates the exact E-series file list (no tokenizer.model, no special_tokens_map.json) |
| GM-102 | `training/export/merge.py:187` | C10-17 | Export staging preserves generation_config.json on both the full-model and adapter-merge paths |
| GM-104 | `training/export/merge.py:185` | C11-05, C13-10, C11-04 | Match: no loader in the training or export path requires model.safetensors.index.json |
| GM-074 | `training/models/families.py:258` | C12-10, C12-11, C12-23 | Nothing in the training tier trims the vocabulary or resizes the tied embedding |
| GM-075 | `training/models/families.py:120` | C11-07, C11-08, C11-11, C13-07 | Family registry omits 12B and raises UnknownModelFamilyError rather than pattern-matching gemma4_text |
| GM-076 | `training/models/families.py:82` | C1-11, C10-02, C10-12, C10-13, C5-04, C5-07, C5-08, C5-10, C5-14, C9-02 | Gemma 4 chat template encodes the spec's <\|turn>/<turn\|> markers, <bos>, and the <\|think\|>/<\|channel>thought pair |
| GM-077 | `training/models/families.py:33` | C3-18, C8-05, C8-07 | LoRA target-module list names only projections present in the spec's E4B tensor inventory |
| GM-072 | `training/recipes/reference/continual-gemma4-e2b.yaml:149` | C1-06, C12-06, C13-01, C2-12, C2-13, C3-01, C3-02, C3-03, C3-06 | Softcap parity reads final_logit_softcapping through the text_config descent, as the spec requires |
| GM-078 | `training/recipes/reference/continual-gemma4-e2b.yaml:56` | C13-01, C2-12, C2-13 | No Gemma 4 recipe sizes a sequence buffer above 4096, well under the E-series 131072 max_position_embeddings |
| GM-067 | `training/recipes/schema.py:421` | C10-15 | No consumer reads model_max_length; every context bound comes from explicit configuration |
| GM-073 | `training/recipes/schema.py:123` | C10-02, C13-04, C13-05, C4-03, C4-04, C4-05, C4-06 | The SFT collator pads with id 0, matching pad_token_id, and masks those positions out of loss and attention |
| GM-068 | `training/recipes/validate.py:78` | C13-13, C2-06, C2-07, C3-19, C3-26 | The scratch GQA divisibility check accepts kv_heads=1, so E2B's true MQA shape passes validation |
| GM-069 | `training/runtime/checkpoint.py:326` | C11-05 | No loader requires model.safetensors.index.json; shards are globbed and the index is optional |
| GM-070 | `training/runtime/checkpoint.py:326` | C12-11 | Checkpoint restore loads safetensors generically, tolerating the zero-dimensional clipped-linear companions |
| GM-071 | `training/runtime/offload.py:50` | C1-07, C11-01, C13-09, C2-19, C5-03, C5-04, C5-05, C5-06, C5-07, C5-08, C5-09, C5-13, C5-15 | FROZEN_PATTERNS transformer_stack matches all seven PLE tensor names, and the froze-nothing guard fires correctly |

## Host verification runbook

No Gemma 4 weights exist on this workstation, so every entry below is PENDING and blocked
on acquiring a checkpoint. The agent sandbox cannot reach `huggingface.co`; the human runs
these on the workstation and pastes the command and its output into the Observed block, per
the recording protocol in [README.md](README.md).

### GH.1 Eval serves with --no-jinja unconditionally, so serve-time turn framing is llama.cpp's built-in guess, not the <\|turn> framing SFT trained

Implementation: `/work/docker/llama-gpu/entrypoint.sh:132`  
Finding: GC-021 (high, `needs-host`)

entrypoint.sh:37 defaults JINJA_ENABLED=false and :132-133 appends --no-jinja whenever it is not "true"; a grep of the whole tree finds JINJA_ENABLED set nowhere except the two entrypoints and the two overview docs, so false is the operative value. common/serving.py:61-62 builds chat_url as base_url + "/v1/chat/completions" and :81-88 pins llama-gpu to port 8002, so llama-server, not the repo, applies the chat template during eval. training/models/families.py:82-86 supervised turn_prefix "<\|turn>{role}\n" with eot "<turn\|>", matching spec:476. With --no-jinja the server ignores the GGUF's embedded template and uses legacy built-in selection; if that resolves to the Gemma 1-3 built-in (<start_of_turn>/<end_of_turn>) or to chatml, every eval prompt is framed differently from training and the scorecard silently reports train/serve skew as model quality. The chain is followable end to end inside the repo; only the last link - what the pinned llama.cpp template table picks, and what the converter writes into tokenizer.chat_template - needs a real GGUF and a running service. High, not critical: the silent-wrong-scorecard outcome is contingent on that one unresolved link. Distinct from KB-24 (thinking_mode_for), which concerns manifest architecture detection, not serve-time templating.

Command:

```bash
$ ~/ai-experiments/rust-harness/target/release/gguf-info /mnt/ai-data/models/gguf/<e4b>.gguf | grep -i 'chat_template\|tokenizer.ggml.model'; docker compose logs llama-gpu | grep -i 'chat template'
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:476 "\| Turn \| `<\\|turn>` \| `<turn\\|>` \|"

Observed:

```
```

Status: PENDING

### GH.2 RH.1's expected signal contradicts the QA record that gguf-info exits 1 parsing a real checkpoint

Implementation: `/work/docs/verification/rust-harness.md:123`  
Finding: GC-014 (high, `needs-host`)

rust-harness.md:123-129 promises a compatibility verdict from cpu::check_supported, while rust-harness-qa.md:736-740 records A1-R07 confirmed: the converter always writes sliding_window_pattern as a boolean array that gguf.rs cannot parse, so gguf-info exits 1 before printing any verdict, and rust-harness-qa.md:463 adds that the gemma4 encoding fails earlier still in parsing. If A1-R07 holds, the campaign's highest-risk entry answers none of its four questions. Separately, a scalar sliding_window_pattern can express both schedules of spec:167-176 (5 for E2B, 6 for E4B), so the stride representation is adequate once it parses. Only a real GGUF settles which output appears.

Command:

```bash
$ cd ~/ai-experiments/rust-harness && cargo run -p runtime --bin gguf-info -- /mnt/ai-data/models/gguf/<e4b>.gguf; echo "exit: $?"
```

Expected signal: agreement with the reference - spec 167-169: "Gemma 4 interleaves two attention types on a fixed schedule given explicitly by the `layer_types` array. Read the array. Do not compute the schedule from a stride constant: E2B uses a different period from the other three variants."

Observed:

```
```

Status: PENDING

### GH.3 Whether the gate can run at all on E2B is unsettled: it assumes AutoModelForCausalLM accepts Gemma4ForConditionalGeneration and that model.layers.{n} resolves

Implementation: `/work/interp/harvest.py:181`  
Finding: GC-020 (high, `needs-host`)

The safetensors prefix at spec:370 pins the module tree to model.language_model.layers.{L}, so `get_submodule("model.layers.15")` resolves only if transformers 5.13.1 (pyproject.toml:19) exposes a compatibility alias. Two distinct outcomes, both unverifiable offline: (a) the path is missing, get_submodule raises AttributeError, the gate refuses loudly with "the model does not expose the flat text-model layout this hook path assumes" (harvest.py:206-209) — the gate working as designed; (b) AutoModelForCausalLM refuses the composite `Gemma4ForConditionalGeneration` config outright, which raises ValueError, is not caught at harvest.py:201, and is absent from the CLI except tuple at interp/commands/harvest.py:66-72 — the gate cannot reach its own refusal and the operator gets a raw traceback. Also bears on spec:98-101, which directs pinning the transformers revision that implements Gemma4ForConditionalGeneration rather than assuming a released version supports it; the repo pins released 5.13.1 against configs targeting 5.5.0.dev0 (spec:158). Extends the prior hook-path findings (GC-012, GX1-G5-envelope-01/02) by pinning the question to the gate's own load path rather than the capture path. No Gemma 4 weights exist on this workstation yet, so this becomes a runbook entry.

Command:

```bash
$ make harvest-check-hooks RECIPE=training/recipes/interp/harvest-smoke.yaml; docker compose -f docker-compose.interp.yaml run --rm interp python -c "from transformers import AutoConfig; from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as M; c=AutoConfig.from_pretrained('google/gemma-4-E2B'); print(c.model_type, c.architectures, M.get(c.model_type))"
```

Expected signal: agreement with the reference - spec:85: "\| E-series \| E2B, E4B \| `Gemma4ForConditionalGeneration` \| `gemma4_text` \|"; spec:370: "\| 42 \| `model.language_model.layers.{L}.input_layernorm.weight` \| `[2560]` \|"; spec:394: "E2B holds the same tensor names with 35 layers"

Observed:

```
```

Status: PENDING

### GH.4 Two tiers assert contradictory structural facts about the Gemma 4 system role and nothing in the tree compares them

Implementation: `/work/rust-harness/harness/src/template.rs:32`  
Finding: GC-024 (high, `needs-host`)

If the Python tier is right (it cites a READ source: families.py:47-48 and docs/testing/test_coverage_investigation.md:393 "format READ"), then every system instruction the rust harness carries is mis-framed. The agent's memory/system prompt is rendered as a user turn — agent.rs constructs Entry{role: Role::System} for memory and chat.rs:34-40 does the same, and both are frozen that way by tests: harness/tests/agent_e2e.rs:290 asserts the prompt starts with "<start_of_turn>user\nprefer small diffs<end_of_turn>\n" and harness/src/chat.rs:229 asserts "<start_of_turn>user\nproject notes<end_of_turn>\n". Tool results are collapsed the same way, where the spec defines a dedicated `<\|tool_response>` pair (spec:480). Nothing in the repository compares the two tiers: grep for `families`, `ChatTemplateSpec`, or `chat_template` across rust-harness/ returns no hit outside unrelated tensor-family code, and the Python tree never references harness/src/template.rs. The spec's own file list (docs/models/gemma-4.md:63-72) shows no chat_template.jinja in the model repository, so the authoritative role vocabulary must be read out of tokenizer_config.json at the pinned revision.

Command:

```bash
$ curl -sL https://huggingface.co/google/gemma-4-E4B/raw/411aa17b749aa952df1359d2dcea73917a544d9a/tokenizer_config.json | python3 -c "import json,sys; print(json.load(sys.stdin).get('chat_template'))"
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:472-476 — "The tokenizer defines asymmetric paired markers where the bar denotes the open or close side:" / "\| Turn \| `<\\|turn>` \| `<turn\\|>` \|". The spec fixes the turn markers but never enumerates the role names admitted in the turn header, so it does not by itself settle whether a system role exists.

Observed:

```
```

Status: PENDING

### GH.5 Whether the GGUF converter's per-layer tensor names contain the substring the loader refusal matches can only be settled on a real file

Implementation: `/work/rust-harness/runtime/src/model/cpu.rs:211`  
Finding: GC-028 (high, `needs-host`)

The spec's names are Hugging Face safetensors names; the runtime consumes GGUF, where the converter renames tensors. If the gemma4 converter emits per-layer tables under names without the literal substring "per_layer" (llama.cpp's gemma3n lineage uses short forms such as per_layer_token_embd and blk.N.inp_gate), this refusal never fires and the checkpoint falls through to the generic unconsumed-family backstop -- or, if the names happen to collide with a consumed family, loads with 2.35B (E2B) / 2.82B (E4B) parameters ignored and returns fluent wrong logits. Every other guard in check_supported is ordered behind this one, so its effectiveness decides whether the refusal chain works at all.

Command:

```bash
$ cd /work/rust-harness && cargo run -p runtime --bin gguf-info -- /mnt/ai-data/models/gguf/<e4b>.gguf | grep -iE 'per_layer|inp_gate|token_embd'
```

Expected signal: agreement with the reference - "\| `model.language_model.embed_tokens_per_layer.weight` \| `[262144, 8960]` \| `[262144, 10752]` \| 1 \|" and "\| `model.language_model.layers.{L}.per_layer_input_gate.weight` \| `[256, 1536]` \| `[256, 2560]` \| per layer \|" (docs/models/gemma-4.md:292, 295)

Observed:

```
```

Status: PENDING

### GH.6 The per-layer-embedding refusal keys on the substring "per_layer" in GGUF tensor names; only a real GGUF settles whether the converter preserves it

Implementation: `/work/rust-harness/runtime/src/model/cpu.rs:210`  
Finding: GC-030 (high, `needs-host`)

The spec's names are Hugging Face safetensors names; the runtime consumes GGUF, where the converter renames tensors. If the GGUF names for the per-layer table and its per-layer gate/projection do not contain the literal substring "per_layer", the refusal at cpu.rs:210 cannot fire and the E-series per-layer path — 2.35 billion parameters on E2B, 2.82 billion on E4B — would fall through to the generic unconsumed-tensor report or be silently ignored, which is the wrong-logits outcome the bail message itself describes. No Gemma 4 weights exist on this workstation, so this is a runbook entry rather than a resolved fact; do not settle it from llama.cpp naming recollection. May overlap a round-1 cpu.rs finding.

Command:

```bash
$ cd /work/rust-harness && cargo run -q --release --bin gguf-info -- /mnt/ai-data/models/gemma-4-E4B/gemma-4-E4B-Q8_0.gguf | sed -n '/=== Tensors/,/^$/p' | grep -ciE 'per_layer|per-layer|plu|altup'
```

Expected signal: agreement with the reference - "The E-series carries a second embedding table, indexed by token and by layer, in addition to / the standard input embedding." (docs/models/gemma-4.md:281-282); tensor names given as `model.language_model.embed_tokens_per_layer.weight`, `model.language_model.layers.{L}.per_layer_input_gate.weight`, `model.language_model.layers.{L}.per_layer_projection.weight` (docs/models/gemma-4.md:292-296)

Observed:

```
```

Status: PENDING

### GH.7 Generation stops on a single GGUF eos id; a Gemma 4 chat turn closes with <turn\|>, which the harness has no stop path for

Implementation: `/work/rust-harness/runtime/src/session.rs:353`  
Finding: GC-032 (high, `needs-host`)

The runtime carries exactly one stop id, taken from tokenizer.ggml.eos_token_id. The spec pins eos_token_id to 1 (`<eos>`), but an instruction-tuned turn ends with the turn-close marker `<turn\|>`, not `<eos>`. If the GGUF converter writes eos_token_id = 1 and records the turn-close id only under a separate key (llama.cpp's eot/eog convention), every chat and agent turn generates to max_tokens instead of stopping, and the reply carries the trailing marker. Nothing in the sandbox can settle which id the converter writes for Gemma 4; the answer is a property of a real GGUF.

Command:

```bash
$ python3 -c "from gguf import GGUFReader; r=GGUFReader('/mnt/ai-data/models/gguf/<e4b>.gguf'); f=r.fields['tokenizer.ggml.tokens']; v=[bytes(f.parts[i]).decode('utf-8','replace') for i in f.data]; [print(k, int(r.fields[k].parts[r.fields[k].data[0]]), v[int(r.fields[k].parts[r.fields[k].data[0]])]) for k in r.fields if k.endswith('token_id')]"
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:449 — "\| End of sequence \| `<eos>` \| 1 \|"; docs/models/gemma-4.md:476 — "\| Turn \| `<\\|turn>` \| `<turn\\|>` \|"

Observed:

```
```

Status: PENDING

### GH.8 SAE test fixtures pin `model.layers.{L}` as the Gemma 4 E2B hook path, but the spec names decoder layers `model.language_model.layers.{L}`

Implementation: `/work/tests/interp/test_sae.py:49`  
Finding: GC-034 (high, `needs-host`)

Every safetensors key in the spec's E4B inventory (spec:365-392) and the E-series PLE table (spec:291-297) is rooted at `model.language_model.`, so the state_dict/module path of decoder layer 15 on the pinned E2B checkpoint is `model.language_model.layers.15`, not `model.layers.15`. The string `language_model` appears nowhere in the code tree (grep over *.py/*.yaml/*.rs outside docs/ returns nothing); the single source of the hook path is interp/families.py:47 `_FLAT_DECODER_TEMPLATE = "model.layers.{layer}"`, registered for both `gemma4` and `gemma4_text` (families.py:52-53). These fixtures are the only tier-wide statement of what a Gemma 4 capture manifest looks like, and they assert the flat form in six places, so the convention is now load-bearing on tests as well as code: correcting families.py would fail TestCaptureGate, TestTraining and TestRegistry. Two live consumers dereference the string against a real module tree - interp/harvest.py:200 `model.get_submodule(hook_name(layer, spec))` and interp/sae/evaluate.py:138 `module = model.get_submodule(hook)` (delta-LM-loss splice, whose docstring at :129 says "Uses the same HF named-module path as the harvest"). If `AutoModelForCausalLM` returns the composite `Gemma4ForConditionalGeneration`, both miss: harvest turns it into a styled HarvestError (harvest.py:205-209, which names the risk honestly - "the model does not expose the flat text-model layout this hook path assumes"), while sae-eval raises a raw AttributeError after loading the model. What cannot be settled offline is whether transformers at the pinned dev revision re-roots the text model under `AutoModelForCausalLM`; that is exactly why the repo's own audit already says to run the host gate first (docs/testing/test_coverage_investigation.md:423-427). Sibling site, already documented there, not re-reported: interp/probes/capture.py hardcodes `model.model.layers[layer]`.

Command:

```bash
$ make harvest-check-hooks RECIPE=training/recipes/interp/harvest-smoke.yaml
```

Expected signal: agreement with the reference - line 370: "\| 42 \| `model.language_model.layers.{L}.input_layernorm.weight` \| `[2560]` \|"; line 394: "E2B holds the same tensor names as E4B with 35 layers, `hidden_size` 1536, the split MLP widths"; line 85: "\| E-series \| E2B, E4B \| `Gemma4ForConditionalGeneration` \| `gemma4_text` \|"

Observed:

```
```

Status: PENDING

### GH.9 load_tokenizer's EOS fallback list omits Gemma 4's `<eos>`, and a local Gemma tokenizer directory never reaches AutoTokenizer

Implementation: `/work/training/data/prepare.py:42`  
Finding: GC-037 (high, `needs-host`)

A local Gemma 4 tokenizer directory takes the bare-tokenizers branch, so `tokenizer_config.json`/`eos_token_id` is never consulted and only the three fallback strings are tried; `<eos>` is not among them. Reachable two ways: training/recipes/reference/sft-qlora-e2b.yaml:44 `path: /mnt/ai-data/training/tokenizers/gemma-4-e2b` reaches load_tokenizer through training/recipes/validate.py:560 and :750, and docs/training/data_pipeline.md:85-86 instructs "pass any AutoTokenizer-loadable tokenizer (hub id or local directory)" for `prepare-data --tokenizer`, which for a directory is exactly the branch that bypasses AutoTokenizer. Which of two outcomes occurs is decided by the real 262144-entry vocabulary: if none of `<\|eos\|>`, `<\|endoftext\|>`, `</s>` is present, load_tokenizer raises ValueError (prepare.py:90) and a correct recipe is rejected as unloadable; if `</s>` or `<\|endoftext\|>` exists as an ordinary or reserved piece, eos_id resolves to something other than 1 and is silently appended after every document (prepare.py:286 `writer.write_tokens(list(ids) + [tokenizer.eos_id])`) and stamped into the shard manifest (:308 `"eos_id": tokenizer.eos_id`). Suspected root: the list was written for the D3 from-scratch tokenizer (`<\|eos\|>` first; docs/training/investigations/d3_tokenizer_training_investigation.md:95-97 "`<\|eos\|>` must resolve as the EOS token") and never extended when Gemma 4 became the primary family (training/models/families.py:7-11).

Command:

```bash
$ curl -sL https://huggingface.co/google/gemma-4-E2B/resolve/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f/tokenizer.json -o /tmp/g4e2b-tok.json && python3 -c "import json;d=json.load(open('/tmp/g4e2b-tok.json'));v=dict(d['model']['vocab']);v.update({a['content']:a['id'] for a in d.get('added_tokens',[])});print({t:v.get(t) for t in ('<eos>','<bos>','<pad>','</s>','<|endoftext|>','<|eos|>')})"
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:449 `\| End of sequence \| `<eos>` \| 1 \|`; :443-444 "The tokenizer class is `GemmaTokenizer` with a `tokenizers` backend, wrapped by `Gemma4Processor`."; :61-72 the E2B/E4B repository file list, which contains `tokenizer.json` and `tokenizer_config.json` and no SentencePiece model file

Observed:

```
```

Status: PENDING

### GH.10 Triage memory states as settled fact that check_supported refuses real E-series checkpoints, but the guard keys on a GGUF substring no artifact in the tree confirms

Implementation: `/work/.claude/agent-memory/triage-agent/project_gemma4_synth_fixture_root.md:23`  
Finding: GC-042 (medium, `needs-host`)

The memory's severity floor for the whole rust-harness fixture family rests on one unverified premise: that a real Gemma 4 E-series GGUF carries at least one tensor whose name contains the literal substring `per_layer`. The spec's names are HF safetensors names (`model.language_model.layers.{L}.per_layer_input_gate.weight`); the converter renames tensors, and the only evidence in-tree is a synthetic test tensor the repo invented for itself (cpu.rs:1204, `per_layer_token_embd.weight`). This is precisely the HF-versus-GGUF class the campaign's own sibling memory (project_gemma4_conformance_audit.md:24-26) says cannot be settled offline. If the converter emits e.g. `altup_*`/`inp_gate`/`laurel_*` style names with no `per_layer` token, the first guard misses, the second guard (cpu.rs:223) also cannot fire because spec:314-316 says every layer keeps its own k/v, and an E-series checkpoint loads with the ~2.35B-parameter per-layer table silently ignored. The memory should carry this as a premise, not a fact.

Command:

```bash
$ python3 -c "import json,sys; d=json.load(open(sys.argv[1])); [print(k) for k in d if 'per_layer' in k or 'inp_gate' in k or 'altup' in k]" e4.json   # e4.json from the safetensors-header range request; then, once a GGUF exists: ./rust-harness/target/release/gguf-info /mnt/ai-data/models/gemma-4-E4B.gguf | grep -iE 'per_layer|inp_gate'
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:295-297 — "\| `model.language_model.layers.{L}.per_layer_input_gate.weight` \| `[256, 1536]` \| `[256, 2560]` \| per layer \|" (HF safetensors names); docs/models/gemma-4.md:57-59 — "Tensor names and shapes for E2B and E4B come from the safetensors headers at those revisions."

Observed:

```
```

Status: PENDING

### GH.11 Interp image pins released transformers 5.13.1 and nothing in the build verifies it implements Gemma4ForConditionalGeneration

Implementation: `/work/docker/interp/Dockerfile.interp:65`  
Finding: GC-091 (medium, `needs-host`)

The interp container is the execution environment for every Gemma 4 E-series interp job (training/recipes/interp/harvest-smoke.yaml:17 `model_id: google/gemma-4-E2B`, sae-smoke.yaml `expect_model_id: google/gemma-4-E2B`, and the organism recipes on google/gemma-4-E2B). Its transformers comes from a released pin, which is precisely the assumption spec:98-101 forbids. If 5.13.1 does not carry `gemma4_text`/`Gemma4ForConditionalGeneration` with per-layer-embedding support (spec:252-254 "Without this mechanism the E-series does not run at all"), `AutoConfig.from_pretrained('google/gemma-4-E2B')` in interp/harvest.py:154 fails at the first harvest, after image build and after the GPU is claimed. Cannot be settled offline: no network, no Gemma 4 weights, and the version's model registry is not vendored in this tree. The doc assertion at f3:234 is the only corroboration and is itself unpinned to a revision.

Command:

```bash
$ cd ~/ai-experiments && docker compose -f docker-compose.interp.yaml run --rm interp python -c "import transformers; from transformers import AutoConfig; print(transformers.__version__); c=AutoConfig.from_pretrained('google/gemma-4-E2B', revision='d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f'); t=getattr(c,'text_config',c); print(type(c).__name__, t.model_type, t.num_hidden_layers, t.hidden_size, t.hidden_size_per_layer_input)"   # requires HF_TOKEN in .env.interp; expect gemma4_text 35 1536 256
```

Expected signal: agreement with the reference - spec:98-101 "**All four `config.json` files target unreleased development versions of `transformers`.** Treat the configuration schema as unstable, and pin the `transformers` revision that actually implements `Gemma4ForConditionalGeneration` rather than assuming a released version supports it." and spec:158 "\| `transformers_version` \| 5.5.0.dev0 \| 5.5.0.dev0 \| 5.10.0.dev0 \| 5.5.0.dev0 \|"

Observed:

```
```

Status: PENDING

### GH.12 Tokenizer module specified as SentencePiece unigram; the spec names a tokenizers-backed GemmaTokenizer with no SentencePiece model file

Implementation: `/work/docs/inference/rust-harness-runtime-design.md:117`  
Finding: GC-054 (medium, `needs-host`)

Spec 443-444 gives the class as GemmaTokenizer with a tokenizers backend wrapped by Gemma4Processor, and the pinned repository file list (spec 63-72) contains tokenizer.json but no tokenizer.model. That settles the Hugging Face side but not what the converter writes into tokenizer.ggml.model, which is the only thing the runtime reads. If it is not unigram, a unigram decoder segments every prompt differently from the reference tokenizer, logits stay finite, and the only test that would notice is the #[ignore]d golden test at :194-198. Records as a runbook entry: no Gemma 4 GGUF exists on this workstation.  Propagated to: rust-harness/README.md:83, rust-harness/runtime/src/tokenizer.rs:104. One assumption (SentencePiece unigram with tokenizer.ggml.scores; Gemma 4 ships a tokenizers backend and only tokenizer.json). The design doc at runtime-design.md:117 is the ROOT that specified the module, README:84 restates it verbatim ('SentencePiece unigram tokenizer'), and tokenizer.rs:104 implements it as a hard requirement.

Command:

```bash
$ cd /work/rust-harness && cargo run --release -p runtime --bin gguf-info -- /mnt/ai-data/models/gemma-4-E4B-Q8_0.gguf | grep -E 'tokenizer\.ggml\.(model|pre|tokens|scores|token_type)'
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:443-444: "The tokenizer class is `GemmaTokenizer` with a `tokenizers` backend, wrapped by `Gemma4Processor`." Spec 63-72 lists the complete repository file list: ".gitattributes / README.md / config.json / generation_config.json / model.safetensors / processor_config.json / tokenizer.json / tokenizer_config.json" — no `tokenizer.model`.

Observed:

```
```

Status: PENDING

### GH.13 The 200-step stability canary that p2_d3 defers peak_lr to cannot run here and its memfit preflight is the unchecked path from GX2-08-01

Implementation: `/work/docs/training/investigations/p2_d3_warmup_stable_decay_investigation.md:37`  
Finding: GC-075 (medium, `needs-host`)

No Gemma 4 weights exist on this workstation (docs/verification/training-phase1.md:50-52 records the first attempt stopping at an unauthenticated hub request), so the canary and its memfit preflight are unexecuted. When it runs, memfit will report a model-parameter line derived from components.py; the spec-derived total is 4,647,449,891 and the calculator's is 4,017,728,000, so a printed "Model parameters" figure near 4.02 billion confirms the shortfall reaches the reference run's sizing, and a figure near 4.65 billion would retire it. This is the runbook entry that turns GX2-08-01 from analysis into evidence.

Command:

```bash
$ make memfit CONFIG=training/recipes/reference/continual-gemma4-e2b.yaml 2>&1 | grep -E 'Model parameters|weights|Total allocated|reserved'
```

Expected signal: agreement with the reference - gemma-4.md:435-439 — "For memory planning on a 32GB RTX 5090, the practical consequence is that the per-layer embedding table dominates. At bfloat16, E2B's per-layer table alone occupies approximately 4.7GB ... Continual pretraining plans that size against the effective parameter count underestimate resident weight memory by roughly a factor of two."

Observed:

```
```

Status: PENDING

### GH.14 No runbook entry resolves the GGUF's EOS piece string or searches the vocabulary for the turn markers

Implementation: `/work/docs/verification/rust-harness.md:135`  
Finding: GC-083 (medium, `needs-host`)

RH.1 is the entry that captures the checkpoint's metadata, and its expected signal (lines 131-137) enumerates four architecture questions plus "every RoPE, window, and per-layer metadata key" — the tokenizer is never named. The dump it produces cannot settle the terminator question in any case: gguf-info.rs:46 prints `MetaValue::ArrStr(v) => println!("{key} = [str; {}]", v.len())`, so `tokenizer.ggml.tokens` appears only as a length, never as pieces. The scalar `tokenizer.ggml.eos_token_id` does print, but nothing in the campaign maps that id back to its piece string. The three facts RH.11 actually needs — which piece the GGUF's eos id names, whether `<turn\|>` is present and whether it is Control or UserDefined (tokenizer.rs:250-251 decides visibility), and whether `<start_of_turn>`/`<end_of_turn>` exist in the Gemma 4 vocabulary at all (if absent, generator.rs:62 PROMPT_SPECIALS resolves nothing and tokenizer.rs:305 falls through to Viterbi/byte-fallback silently, with no error) — are settleable only against a real GGUF, and no Gemma 4 weights exist on this workstation yet. This becomes a runbook entry rather than a resolved fact.

Command:

```bash
$ cd ~/ai-experiments && uv run python -c "import gguf; r=gguf.GGUFReader('/mnt/ai-data/models/gguf/<e4b>.gguf'); t=r.fields['tokenizer.ggml.tokens'].contents(); e=r.fields['tokenizer.ggml.eos_token_id'].contents(); print('eos', e, repr(t[e])); [print(s, t.index(s) if s in t else 'ABSENT') for s in ('<turn|>','<|turn>','<end_of_turn>','<start_of_turn>','<eos>')]"
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:448-449 — "\| Beginning of sequence \| `<bos>` \| 2 \|" / "\| End of sequence \| `<eos>` \| 1 \|"; docs/models/gemma-4.md:476 — "\| Turn \| `<\\|turn>` \| `<turn\\|>` \|"

Observed:

```
```

Status: PENDING

### GH.15 The step-zero environment gate pins transformers 5.13.1 and never checks that it implements Gemma4ForConditionalGeneration, which the spec forbids assuming

Implementation: `/work/docs/verification/step-zero-environment.md:26`  
Finding: GC-086 (medium, `needs-host`)

Step zero "blocks all other phases" (step-zero-environment.md:3) and marks 0.1 through 0.8 VERIFIED, yet every recorded Observed block reads "Run on the workstation 2026-08-01; output not recorded." None of the eight entries instantiates a Gemma 4 config: 0.6's train-smoke checks CUDA/flash-attn/bitsandbytes/CPUAdam, and 0.7's gated Gemma step is Gemma-2-2B plus a Gemma Scope SAE (interp/commands/smoke_test.py:199, "Step 6 (gated): Gemma-2-2B at bf16 plus one Gemma Scope SAE via SAELens"), not Gemma 4. So the first Gemma 4 execution is deferred to training-phase1.md P1.0, still PENDING. If transformers 5.13.1 does not carry the gemma4 modelling code, every Gemma-4 recipe, the memfit HF path, and the family registry fail at first use; if it does, its `text_config` field names must still be confirmed against the spec's per-variant table before spec_from_hf_config's mapping can be trusted. Unresolvable in the sandbox: transformers is not importable here (ModuleNotFoundError) and there is no network.

Command:

```bash
$ docker compose -f docker-compose.training.yaml run --rm training python -c "import transformers; print(transformers.__version__); from transformers import AutoConfig; c=AutoConfig.from_pretrained('google/gemma-4-E2B'); t=c.text_config; print(c.model_type, t.model_type, t.num_hidden_layers, t.hidden_size, t.intermediate_size, t.num_key_value_heads, t.head_dim, getattr(t,'global_head_dim',None), t.hidden_size_per_layer_input, getattr(t,'use_double_wide_mlp',None), getattr(t,'num_kv_shared_layers',None))"
```

Expected signal: agreement with the reference - gemma-4.md:98-101 — "**All four `config.json` files target unreleased development versions of `transformers`.** Treat the configuration schema as unstable, and pin the `transformers` revision that actually implements `Gemma4ForConditionalGeneration` rather than assuming a released version supports it."

Observed:

```
```

Status: PENDING

### GH.16 transformers is pinned to a released version (5.13.1) with no recorded evidence that it implements Gemma4ForConditionalGeneration

Implementation: `/work/pyproject.toml:19`  
Finding: GC-098 (medium, `needs-host`)

Every Gemma 4 path in the tree resolves the architecture through this pin: training/tune/sft.py:247-252 (AutoConfig/AutoModelForCausalLM), training/calculator/components.py:202-205, interp/harvest.py:152-154 and interp/probes/capture.py:81 (nnsight LanguageModel over the same HF stack). If release 5.13.1 does not carry the gemma4 / gemma4_text registration (E2B, E4B, 31B) or gemma4_unified / gemma4_unified_text (12B, spec:86), every one of those calls raises at AutoConfig on google/gemma-4-E2B. Nothing in the tree records that check: transformers is absent from the sandbox venv (`python3 -c "import transformers"` -> ModuleNotFoundError), and the workstation campaign paused before any Gemma 4 load (docs/verification/training-phase1.md:70-78 "Campaign paused 2026-08-01 ... P0.4 onward not yet executed"). The repo applies exactly the discipline spec:98-101 asks for to the other converter it depends on - training/export/llama_cpp_pin.txt pins commit e3546c7948e3af463d0b401e6421d5a4c2faf565 and training/export/convert.py:81-98 `check_architecture` asserts the pinned converter registers the architecture - but no equivalent exists for transformers. Second copy of the same pin, kept in sync by hand and able to drift: docker/training/Dockerfile.export:66 `transformers==5.13.1` (declared at Dockerfile.export:58-62 as "the one deliberate deviation from lockfile installs"); docker/comfyui/Dockerfile.comfyui:50 carries a third.

Command:

```bash
$ uv run --no-sync python -c "import transformers as t; print(t.__version__); from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M; print(sorted(k for k in M if k.startswith('gemma4'))); print(hasattr(t,'Gemma4ForConditionalGeneration'), hasattr(t,'Gemma4UnifiedForConditionalGeneration'))"
```

Expected signal: agreement with the reference - spec:98-101: "**All four `config.json` files target unreleased development versions of `transformers`.** Treat the configuration schema as unstable, and pin the `transformers` revision that actually implements `Gemma4ForConditionalGeneration` rather than assuming a released version supports it." and spec:158: "\| `transformers_version` \| 5.5.0.dev0 \| 5.5.0.dev0 \| 5.10.0.dev0 \| 5.5.0.dev0 \|"

Observed:

```
```

Status: PENDING

### GH.17 System text is folded into a user turn and the generation prompt omits the empty thought channel; both unsettled by the spec

Implementation: `/work/rust-harness/harness/src/template.rs:37`  
Finding: GC-100 (medium, `needs-host`)

template.rs:31-40 maps Role::System, Tool, and Compaction all to "user" on the premise stated at template.rs:32 that "Gemma has no system role", and MODEL_TURN_OPEN (template.rs:18) ends the prompt with a bare model turn. The authority settles the marker strings (gemma-4.md:472-483) but not the role vocabulary, the existence of a native system turn, or the generation-prompt tail, so this cannot be graded from the spec alone. Two in-tree readings of the official template disagree with the code: training/models/families.py:76-77 records "native system turn" and :87 assistant_role "model", and docs/training/investigations/f1_sft_data_investigation.md:211-213 records that with thinking disabled the official generation prompt ends "<\|turn>model\n<\|channel>thought\n<channel\|>". If both hold, the harness mislabels the system prompt (HARNESS.md content arrives as a user turn) and omits the empty thought prefix the model was tuned to continue from - plausible wrong output, no error. Settle by reading chat_template.jinja at the pinned revision before rewriting template.rs.

Command:

```bash
$ curl -sL https://huggingface.co/google/gemma-4-E4B/resolve/411aa17b749aa952df1359d2dcea73917a544d9a/tokenizer_config.json | python3 -c "import json,sys;print(json.load(sys.stdin)['chat_template'])" > /tmp/gemma4_chat_template.jinja; cat /tmp/gemma4_chat_template.jinja
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:472 "The tokenizer defines asymmetric paired markers where the bar denotes the open or close side"; :482 "A reasoning marker `<\|think\|>` and an escape token `<\|\"\|>` are also defined."

Observed:

```
```

Status: PENDING

### GH.18 Dual-RoPE refusal only sees keys under {arch}.rope.; a theta spelled elsewhere, or folded away, escapes it

Implementation: `/work/rust-harness/runtime/src/model/cpu.rs:258`  
Finding: GC-108 (medium, `needs-host`)

Extends A1-R04, refuted in docs/verification/rust-harness-qa.md:454-466: four checkpoints built with the converter's exact key set showed the second base arrives as {arch}.rope.freq_base_swa, which the catch-all at cpu.rs:275-278 does refuse — so the wrong-numerics path is unreachable on the evidence in this tree, and the severity is capped accordingly. The open residual is that detection is entirely key-name dependent: `strip_prefix("{arch}.rope.")` (258-262) cannot see a base spelled gemma4.rope_theta (the spelling the parser's own test writes, gguf.rs:383) or gemma4.attention.rope_freq_base_swa, and it cannot see a theta the converter folds into a hardcoded rule with no key at all. Meanwhile cpu.rs:16-21 and rust-harness/README.md:103-106 state the refusal unconditionally, which is the imprecise-guarantee residual A1-R04 left open. Only the pinned converter or a real gemma4 GGUF settles the spelling.  Propagated to: rust-harness/README.md:105. One defect: the dual-RoPE refusal only inspects keys under {arch}.rope.*, so a single-theta or differently-spelled checkpoint passes silently. cpu.rs:258 is the code ROOT; README:105 advertises the same allowlist-scoped guarantee.

Command:

```bash
$ docker compose -f docker-compose.training.yaml run --rm --entrypoint sh export -c "grep -n -i 'rope' /opt/llama.cpp/gguf-py/gguf/constants.py; ls /opt/llama.cpp/conversion 2>/dev/null | grep -i gemma"
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:238-240 "Second, the two regimes differ in base frequency by two orders of magnitude (1000000.0 against 10000.0). A single global `rope_theta` is incorrect for this architecture regardless of which value is chosen."

Observed:

```
```

Status: PENDING

### GH.19 The per-layer-embedding refusal matches 'per_layer', an HF name the GGUF converter need not preserve

Implementation: `/work/rust-harness/runtime/src/model/cpu.rs:211`  
Finding: GC-109 (medium, `needs-host`)

Spec 291-297 gives safetensors names, six of which contain per_layer; the loader matches GGUF names, which the converter rewrites (token_embd, blk.N.attn_q, ffn_up). The guard's test pins the llama.cpp Gemma-3n spelling per_layer_token_embd.weight (cpu.rs:1201-1206), not any observed Gemma 4 name. If the converted names drop the substring, the named diagnosis is skipped and the generic backstop at 304-328 still refuses the load, so this is message quality and RH.1 answer quality rather than correctness — which is why it is medium and not critical. The same question applies to gguf-info's metadata filter, which greps for per_layer and matformer keys that may not exist.  Propagated to: rust-harness/runtime/src/bin/gguf-info.rs:164, rust-harness/README.md:104, rust-harness/fixtures/reference_logits.py:176. One guard: the per-layer-embedding refusal is a bare 'per_layer' substring match on an HF name the GGUF converter need not preserve. cpu.rs:211 is the ROOT; gguf-info.rs:164 mirrors it, README:104 advertises it, and reference_logits.py:176 records that no fixture proves it fires on converter-produced names. All four resolve with the same host check.

Command:

```bash
$ cd /work/rust-harness && cargo run -p runtime --bin gguf-info -- /mnt/ai-data/models/gguf/<e2b>.gguf | grep -iE 'per_layer|inp_gate|proj_norm|altup'
```

Expected signal: agreement with the reference - "\| `model.language_model.embed_tokens_per_layer.weight` \| `[262144, 8960]` \| `[262144, 10752]` \| 1 \|" and "\| `model.language_model.layers.{L}.per_layer_input_gate.weight` \| `[256, 1536]` \| `[256, 2560]` \| per layer \|" (/work/docs/models/gemma-4.md:292, 295)

Observed:

```
```

Status: PENDING

### GH.20 Whether a Gemma 4 GGUF carries attention.sliding_window_pattern, and whether it is 5 on E2B, can only be settled on a real file

Implementation: `/work/rust-harness/runtime/src/model/cpu.rs:93`  
Finding: GC-113 (medium, `needs-host`)

The whole schedule hangs on one scalar the converter may not write: the prior campaign already observed that "a stock gemma3 file is refused for a missing scalar `sliding_window_pattern`" (docs/verification/rust-harness-qa.md:461-462). If a gemma4 export omits it, every load fails at Config::from_gguf; if it writes a scalar that does not equal 5 for E2B or 6 for E4B, layer_kind silently produces the wrong sliding/global assignment and the model returns plausible wrong logits. No Gemma 4 weights exist on this workstation, so this becomes a runbook entry rather than a resolved fact.

Command:

```bash
$ cd /work/rust-harness && cargo run -p runtime --bin gguf-info -- /mnt/ai-data/models/gguf/<e4b>.gguf | grep -i 'sliding_window'
```

Expected signal: agreement with the reference - "Gemma 4 interleaves two attention types on a fixed schedule given explicitly by the `layer_types` array. Read the array." (docs/models/gemma-4.md:167-168)

Observed:

```
```

Status: PENDING

### GH.21 The scale the code hardcodes has no source: the spec states none, no test checks it externally, and cpu.rs:293 refuses to be told one

Implementation: `/work/rust-harness/runtime/src/model/cpu.rs:293`  
Finding: GC-116 (medium, `needs-host`)

The forward pass does implement an attention scale, 1/sqrt(attention.key_length), while the guard refuses any checkpoint that declares one - the message claims the scale is a feature "this forward pass does not implement" (cpu.rs:296-297). Which way that cuts depends on a real file. If a Gemma 4 GGUF carries `gemma4.attention.scale`, the guard fires and the refusal is loud and correct (the unit test at cpu.rs:1291 exercises exactly that key at 0.0625 = 1/sqrt(256)); if the converter omits it, the unsourced 1/sqrt(key_length) governs silently and nothing in the tree can notice, because the only cross-check is fixtures/reference_logits.py:211 "scores = (keys @ q[qh]) / np.float32(math.sqrt(hd))", which reads the same `attention.key_length` (reference_logits.py:156) and encodes the same assumption. The same question decides C1-06: the softcapping substring test at cpu.rs:293 only protects the logits if the converter emits a `*softcapping*` key for the spec's `final_logit_softcapping` 30.0 (spec:115); if it does not, every logit is silently uncapped. One dump settles both, plus the per-layer q_proj widths of M7-scale-01. No Gemma 4 weights exist on this workstation yet, so this is a runbook entry (RH.1) rather than a resolved fact.

Command:

```bash
$ cargo run --manifest-path rust-harness/Cargo.toml -p runtime --bin gguf-info -- /mnt/ai-data/models/gemma-4-E4B.gguf | grep -Ei 'attention\.scale|logit_scale|softcapping|key_length|value_length|query_pre_attn'
```

Expected signal: agreement with the reference - spec:105-106 "These values are identical in E2B, E4B, 12B, and 31B. They are safe to treat as constants,\nand a verification pass should assert them rather than read them per model." - and the invariant table it introduces (spec:108-127) lists no attention scale and no `query_pre_attn_scalar`

Observed:

```
```

Status: PENDING

### GH.22 Single rope_freq_base throughout: the dual-RoPE refusal fires only if the converter names a second rope key

Implementation: `/work/rust-harness/runtime/src/synth.rs:258`  
Finding: GC-121 (medium, `needs-host`)

TinySpec (synth.rs:30, default 10000.0 at :48) and cpu::Config each hold one `rope_freq_base`, written as the single key `gemma4.rope.freq_base` at synth.rs:258. The refusal at cpu.rs:275-278 is key-name driven: it bails on any other `{arch}.rope.*` key. If the converter publishes a second base key the load is honestly refused; if it hardcodes 10000.0 for sliding layers in the graph and publishes only one base, nothing fires and every layer rotates at one base, which spec:222-224 says produces plausible logits rather than a crash. Unreachable for E2B/E4B today because the per-layer-embedding refusal (cpu.rs:211) fires first; load-bearing for 31B and once per-layer embeddings land.  Propagated to: rust-harness/runtime/src/synth.rs:258. Two hunters on the same line: synth.rs:258 writes one rope_freq_base, so no fixture, golden or determinism test can see a regime split. Same site, same defect, different framings (guard dependence vs test blindness). The numpy reference's single rope_theta (S3-M1-tests-05) is a separate implementation and stays its own finding.

Command:

```bash
$ cd /work/rust-harness && cargo run -p runtime --bin gguf-info -- /mnt/ai-data/models/gguf/<e4b>.gguf | grep -E '^gemma4\.rope\.'
```

Expected signal: agreement with the reference - L238-240: "the two regimes differ in base frequency by two orders of magnitude (1000000.0 against 10000.0). A single global `rope_theta` is incorrect for this architecture regardless of which value is chosen."

Observed:

```
```

Status: PENDING

### GH.23 The whole attention schedule reduces to one sliding_window_pattern integer; E2B's period of 5 is unverifiable in-tree

Implementation: `/work/rust-harness/runtime/src/synth.rs:253`  
Finding: GC-122 (medium, `needs-host`)

GGUF carries no `layer_types` array, so cpu.rs:93 derives the schedule from the single required key `attention.sliding_window_pattern` (hard bail when missing) and cpu.rs:143-148 makes every pattern-th layer global. Spec:167-169 says to read the array and not compute from a stride because E2B's period differs. Three outcomes, none decidable here: key absent (loud failure on first real load), key correct, or a converter reusing the Gemma-3 constant 6 for E2B, giving full layers 5/11/17/23/29 instead of 4/9/14/19/24/29/34 - twelve layers masked wrongly and layer 34 sliding, contradicting spec:178, with no error.

Command:

```bash
$ cd /work/rust-harness && for m in <e2b> <e4b>; do cargo run -q -p runtime --bin gguf-info -- /mnt/ai-data/models/gguf/$m.gguf | grep -E 'block_count|sliding_window'; done
```

Expected signal: agreement with the reference - L167-169: "Gemma 4 interleaves two attention types on a fixed schedule given explicitly by the `layer_types` array. Read the array. Do not compute the schedule from a stride constant: E2B uses a different period from the other three variants."

Observed:

```
```

Status: PENDING

### GH.24 No bfloat16 tensor codec exists in the Rust tree and no dependency supplies one, while the spec records dtype bfloat16 for E2B and E4B

Implementation: `/work/rust-harness/runtime/src/tensor.rs:6`  
Finding: GC-123 (medium, `needs-host`)

The comment asserts as fact which storage types "the target checkpoints actually use", but the only published dtype for E2B and E4B is bfloat16 (spec:157). Converting a bf16 checkpoint losslessly yields GGML type 30 (BF16), which byte_size and dequantize both reject with `unsupported GGML tensor type 30` — a loud, declared refusal, so this is not silent corruption, but it means the lossless conversion of the exact variants the E-series work targets cannot be loaded, and the source comment states the opposite. Only a real converted GGUF settles which ggml types the workstation's E2B/E4B image carries; the sandbox has no Gemma 4 weights, so this becomes a runbook entry rather than a resolved fact.

Command:

```bash
$ cargo run -p runtime --bin gguf-info -- /mnt/ai-data/models/gemma-4-E4B.gguf | sed -n '/Quantization types/,/Feed-forward/p'
```

Expected signal: agreement with the reference - spec:157 "\| `dtype` \| `bfloat16` \| `bfloat16` \| (unset) \| `bfloat16` \|"

Observed:

```
```

Status: PENDING

### GH.25 Runtime tokenizer hard-requires SentencePiece scores, but Gemma 4 ships a tokenizers-backend tokenizer.json and no SentencePiece model file

Implementation: `/work/rust-harness/runtime/src/tokenizer.rs:107`  
Finding: GC-124 (medium, `needs-host`)

The spec's E-series repositories carry no SentencePiece model file, so a GGUF conversion must come from the tokenizers-format tokenizer.json. If that conversion writes a BPE-style vocab (tokens plus merges, no scores array), Tokenizer::from_gguf bails "missing tokenizer.ggml.scores" and no Gemma 4 GGUF loads at all. If it writes a scores array, unigram Viterbi over pieces trained as BPE merges is a different segmentation than the reference tokenizer produces, and the U+2581 space marker (tokenizer.rs:23-24 `const SPACE_PIECE: char = '\u{2581}';`) is the wrong marker for a byte-level export — different token ids for the same prompt, with no error. The synthetic fixture cannot expose either branch because runtime/src/synth.rs writes tokens, scores, and token_type by construction.

Command:

```bash
$ cargo run -p runtime --bin gguf-info -- /mnt/ai-data/models/gemma-4-E2B.gguf | grep -i 'tokenizer.ggml' | head -20
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:443-444 "The tokenizer class is `GemmaTokenizer` with a `tokenizers` backend, wrapped by `Gemma4Processor`."; :61-72 repository file list: `.gitattributes` / `README.md` / `config.json` / `generation_config.json` / `model.safetensors` / `processor_config.json` / `tokenizer.json` / `tokenizer_config.json`

Observed:

```
```

Status: PENDING

### GH.26 Whether Gemma 4 turn markers decode to nothing or leak as literal text depends on their GGUF token_type, which no artifact here can settle

Implementation: `/work/rust-harness/runtime/src/tokenizer.rs:630`  
Finding: GC-125 (medium, `needs-host`)

If the converter writes Gemma 4's `<turn\|>` as CONTROL (3) it vanishes from decoded text and harness/src/chat.rs:23-28 (`.strip_suffix(TURN_CLOSE)`) is dead code; if it writes it as USER_DEFINED (4) the marker survives into the reply string, and chat.rs strips the wrong literal (`<end_of_turn>`, per template.rs:16), so the delimiter is displayed to the user and appended verbatim into the persisted transcript, which then re-renders it as content. The test module pins both behaviors but only on fabricated pieces, so neither branch is grounded in a real export. No Gemma 4 weights exist on this workstation, so this becomes a runbook entry rather than a resolved fact.

Command:

```bash
$ python3 -c "import gguf; r=gguf.GGUFReader('/mnt/ai-data/models/gemma-4-E2B.gguf'); t=r.fields['tokenizer.ggml.tokens']; k=r.fields['tokenizer.ggml.token_type']; [print(int(k.parts[k.data[i]][0]), bytes(t.parts[t.data[i]]).decode()) for i in range(len(t.data)) if b'turn' in bytes(t.parts[t.data[i]]) or b'think' in bytes(t.parts[t.data[i]])]"
```

Expected signal: agreement with the reference - gemma-4.md:472-480 — "The tokenizer defines asymmetric paired markers where the bar denotes the open or close side:" \| Turn \| `<\|turn>` \| `<turn\|>` \|. The spec states the marker strings but assigns them no SentencePiece/llama.cpp token role.

Observed:

```
```

Status: PENDING

### GH.27 Real-config RAM test errors rather than skips when weights are absent, and its 75% bound hides a 33% undercount

Implementation: `/work/rust-harness/runtime/tests/cpu_reference.rs:111`  
Finding: GU-016 (medium, `needs-host`)

spec_from_recipe on continual-gemma4-e2b.yaml reaches AutoConfig.from_pretrained('google/gemma-4-E2B', local_files_only=True), so with no Gemma 4 weights on the workstation the call raises and the test errors instead of skipping - contrary to the tier's own policy in tests/hardware/conftest.py:10-12, which tests/hardware/test_gpu.py:194-197 honours. Once the config is cached the assertion still cannot discriminate: 75 percent of 128 GB admits both the 19.8 GB the calculator predicts and the 29.7 GB the spec's shapes imply (see S9-M1-tests-03). This is the only place in tests/ where a real Gemma 4 config meets the memory model, and it is a headroom bound, not a conformance check. Settle the skip-versus-error behaviour on the host before regrading.

Expected signal: agreement with the reference - docs/models/gemma-4.md:105-106 — "These values are identical in E2B, E4B, 12B, and 31B. They are safe to treat as constants, and a verification pass should assert them rather than read them per model."

Observed:

```
```

Status: PENDING

### GH.28 server-bench.py's derived default LLAMA_ARG_CTX_SIZE is 137,568, above the E-series max_position_embeddings of 131072

Implementation: `/work/scripts/benchmark/server-bench.py:173`  
Finding: GC-127 (medium, `needs-host`)

Bare `python3 scripts/benchmark/server-bench.py` takes stock defaults: --path_server "llama-server" (a binary, so external_server is false at line 150), --prompt_source rng-1024-2048, --n_prompts 100, --n_predict 2048, --seed_offset 0, LLAMA_ARG_N_PARALLEL defaulted to 32. I replayed get_prompt_lengths_rng with those exact seeds: max prompt length 2047, so context_per_slot = int(1.05 * (2048 + 2047)) = 4299 and context_total = 4299 * 32 = 137568, which is set as LLAMA_ARG_CTX_SIZE. 137568 exceeds the E2B/E4B declared 131072 by 6496. Whether llama-server refuses, warns, or divides by n_parallel (4299 per sequence, well inside 131072) and accepts silently cannot be settled here and must not be guessed from half-remembered llama.cpp conventions; no Gemma 4 GGUF exists on this workstation. Bounding the blast radius: the two documented entry points never reach this branch (benchmark_gpu.sh:377 and benchmark_cpu.sh:483 both pass `--path_server "http://127.0.0.1:${PORT}"`, making external_server true, and both set N_PARALLEL=1), so only direct invocation is affected. Provenance: this is not a value authored in this repo. pyproject.toml:300 records `# Vendored from llama.cpp; keep upstream style to ease future diffs` with server-bench.py excluded from ruff, so the 1.05 heuristic and the 32-slot default are upstream llama.cpp's and were never sized against any model in this tree — the suspected root is the vendoring, not a local design document.

Command:

```bash
$ docker exec llama-gpu /app/llama-server -m /mnt/ai-data/models/gemma-4-E2B-Q8_0.gguf -c 137568 -np 32 --host 127.0.0.1 --port 8080 2>&1 | grep -iE 'n_ctx_train|n_ctx_per_seq|n_ctx +=|exceed|warn' | head -20
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:153 `\| `max_position_embeddings` \| 131072 \| 131072 \| 262144 \| 262144 \|`; docs/models/gemma-4.md:558-561 "Release material cites up to 256K tokens. E2B and E4B declare `max_position_embeddings: 131072`. Only 12B and 31B declare 262144. Sizing E-series buffers against 256K overshoots by a factor of two, and assuming 256K works at inference is unverified."

Observed:

```
```

Status: PENDING

### GH.29 The distillation teacher asserted verbatim by the assigned tests, google/gemma-4-E4B-it, is a repository the spec never establishes

Implementation: `/work/tests/training/test_cli_handlers_train.py:291`  
Finding: GC-140 (medium, `needs-host`)

The literal is pinned in the test suite, so the tests lock in a repository identity that has no pinned revision, no config.json blob hash, and no entry in the spec's provenance table. If `google/gemma-4-E4B-it` does not exist (or is named differently), `make train-distill CONFIG=training/recipes/reference/distill-logit-gemma-e2b.yaml` fails at teacher load with a hub 404 after the student has already been staged, and `make generate-teacher --teacher google/gemma-4-E4B-it` fails at the first step of the F8 distillation runbook. A second, quieter dependency rides on the same unverified premise: the logit-tier preflight (training/tune/distill.py:171 check_tokenizer_parity) compares tokenizer content hashes, so the recipe's stated assumption that the -it variant "shares the base tokenizer" is what makes E2B-student/E4B-it-teacher logit distillation legal at all. Suspected root: docs/training/investigations/f8_recipe_cards_investigation.md:235 ("**Teacher is `google/gemma-4-E4B-it` for both stages.** ... the `-it` variant shares the base tokenizer, so the logit-tier identity preflight holds (verified by the preflight at run time)"), which propagated into training/recipes/reference/distill-logit-gemma-e2b.yaml (`teacher: google/gemma-4-E4B-it`) and distill-response-gemma-e2b.yaml, and from there into the assertions in these tests, docs/training/teacher_synthetic_data.md:32, tests/training/test_distill.py, test_eval_runner.py, and test_teacher.py. Fixing the root fixes all seven sites.

Command:

```bash
$ curl -sI https://huggingface.co/google/gemma-4-E4B-it/resolve/main/config.json | head -1; curl -sI https://huggingface.co/google/gemma-4-E4B-it/resolve/main/config.json | grep -i '^x-repo-commit'; for m in gemma-4-E4B gemma-4-E4B-it; do printf '%s ' "$m"; curl -sL "https://huggingface.co/google/$m/raw/main/tokenizer.json" | sha256sum; done
```

Expected signal: agreement with the reference - spec 44-49 (provenance table) lists exactly four repositories: "\| E2B \| `google/gemma-4-E2B` \| ... \| E4B \| `google/gemma-4-E4B` \| ... \| 12B \| `google/gemma-4-12B` \| ... \| 31B \| `google/gemma-4-31B` \|"; spec 40-42: "Every fact in this document derives from the files at these exact repository commits. Record a new revision here before updating any value"

Observed:

```
```

Status: PENDING

### GH.30 The pinned llama.cpp converter is recorded as not mapping layer_scalar, and no export guard checks tensor coverage

Implementation: `/work/training/export/convert.py:81`  
Finding: GU-078 (medium, `needs-host`)

42 (E4B) / 35 (E2B) learned per-layer scalars exist whose position in the residual or attention path is an open question (spec 591-592). If the pinned converter drops them, every GGUF-served or GGUF-evaluated Gemma 4 checkpoint runs without them and nothing in this tree would notice: the rust backstop (cpu.rs:304-320) only sees tensors that reached the file, and check_architecture passes on the architecture name alone. If instead the converter raises, the export fails after the full staging step. Which of the two happens cannot be settled here - it is a property of the pinned llama.cpp checkout e3546c7948e3af463d0b401e6421d5a4c2faf565.

Command:

```bash
$ grep -rn "layer_scalar" /opt/llama.cpp/convert_hf_to_gguf.py /opt/llama.cpp/gguf-py/gguf/tensor_mapping.py /opt/llama.cpp/conversion/ 2>/dev/null; git -C /opt/llama.cpp rev-parse HEAD
```

Expected signal: agreement with the reference - line 405-406: "Each layer carries a scalar tensor `layer_scalar` of shape `[1]`. Its role is not documented in the configuration."; line 375: "\| 42 \| `model.language_model.layers.{L}.layer_scalar` \| `[1]` \|"

Observed:

```
```

Status: PENDING

### GH.31 Bare LoRA target names may suffix-match audio/vision tower projections when the composite E-series model is wrapped

Implementation: `/work/training/models/families.py:33`  
Finding: GC-152 (medium, `needs-host`)

PEFT resolves target_modules by dotted-suffix match over every module of the wrapped model. Every Gemma 4 QLoRA recipe uses target_modules: auto and loads through AutoModelForCausalLM against a Gemma4ForConditionalGeneration config, so any tower attention submodule literally named q_proj/k_proj/v_proj/o_proj receives an adapter alongside the text decoder's. adapter_param_count multiplies per-module cost by the text layer count only, so the F2 within-10-percent prediction would be low and the exported adapter would carry tower deltas trained on text. relative_k_proj is genuinely safe (it ends '_k_proj', not '.k_proj'), which is what the families.py:125 note covers; the plain-named tower projections are the unexamined case. The spec does not enumerate tower submodule names, so only a safetensors header settles it.

Command:

```bash
$ docker compose -f docker-compose.training.yaml run --rm training python -c "from transformers import AutoModelForCausalLM; m=AutoModelForCausalLM.from_pretrained('/mnt/ai-data/models/hf/gemma-4-E2B', local_files_only=True); print([n for n,_ in m.named_modules() if n.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'} and 'language_model' not in n][:40])"
```

Expected signal: agreement with the reference - spec 522-527: "...self-attention with a `relative_k_proj` and a `per_dim_scale` tensor of shape `[128]`. Many audio and vision linear layers carry `input_min`, `input_max`, `output_min`, and `output_max` scalar tensors alongside `linear.weight`, consistent with `use_clipped_linears: true`."

Observed:

```
```

Status: PENDING

### GH.32 Gemma 4 thinking-trace guard keys on `<\|channel>thought`, a channel label the spec never defines; only `<\|channel>`/`<channel\|>` are established

Implementation: `/work/training/models/families.py:131`  
Finding: GC-153 (medium, `needs-host`)

check_thinking_markers is a substring guard. It fires on a reasoning section only if the served checkpoint labels its thought channel with the exact literal `thought` immediately after `<\|channel>`. The spec establishes only the paired tokens `<\|channel>` / `<channel\|>`; the label is not spec-supported. If a real Gemma 4 teacher emits, say, `<\|channel>analysis ... <channel\|>`, or omits the label, the guard does not fire, SFT prep accepts the record, and raw reasoning traces are supervised as assistant content - exactly the silent contamination the guard exists to prevent. The closing token `<channel\|>`, which the spec does define, is not in thinking_markers, so there is no fallback catch. The test that would catch this pins the same assumed string (tests/training/test_sft.py:187 `            "<\|channel>thought planning <channel\|> done",  # Gemma 4 trace`), so a green suite proves nothing about the real label. Provenance: the literal originates in docs/training/investigations/f1_sft_data_investigation.md:121 ("`<\|think\|>` in the system turn and `<\|channel>thought\n...<channel\|>`") and :212, sourced there from the "official chat_template.jinja (mirrored in llama.cpp models/templates) and convert_gemma4_weights.py" - neither artifact is present in this tree. From that single root it propagated to families.py:78/131/138, tests/training/test_sft.py:187, docs/training/investigations/f3_sft_entry_point_investigation.md:155, and docs/testing/test_coverage_investigation.md:395/418. Fixing the root fixes six sites. The assigned file states the precondition that was skipped: f4_teacher_synthetic_data_investigation.md:230-232 "Thinking-mode control for Gemma 4 teachers depends on how the serving stack exposes it (chat-template kwargs versus system prompt); needs a probe against the real service before the config field is fixed." The marker literal was fixed without that probe.

Command:

```bash
$ curl -sL "https://huggingface.co/google/gemma-4-E4B/resolve/411aa17b749aa952df1359d2dcea73917a544d9a/tokenizer_config.json" -o /tmp/g4tok.json && grep -o 'channel[^"\\]\{0,24\}' /tmp/g4tok.json | sort -u | head -40
```

Expected signal: agreement with the reference - gemma-4.md:477 `\| Channel \| `<\\|channel>` \| `<channel\\|>` \|` (introduced at 472: "The tokenizer defines asymmetric paired markers where the bar denotes the open or close side"), and 482 "A reasoning marker `<\|think\|>` and an escape token `<\|"\|>` are also defined." No channel *name* ("thought" or otherwise) appears anywhere in the spec.

Observed:

```
```

Status: PENDING

### GH.33 Nothing binds the organism recipes' tokenizer artifact to Gemma 4 E2B's tokenizer; vocab 262144 and pad/eos/bos 0/1/2 are unverified

Implementation: `/work/training/recipes/organisms/taboo.yaml:52`  
Finding: GC-156 (medium, `needs-host`)

The corpus generate.py writes is tokenized with whatever sits at that path. Validation binds the tokenizer only to the data shards, never to the base model: training/recipes/validate.py:727 `_check_tokenizer_matches_shards` compares `load_tokenizer(recipe.tokenizer.path).content_hash` against the shard manifest hash, and training/commands/preflight.py:192 only checks that the path exists. `--tokenizer` accepts a directory, a tokenizer.json, or a hub id (training/commands/prepare_sft.py:58), so the same local directory can hold either a saved copy of Gemma 4's 262144-entry tokenizer or an artifact from the from-scratch trainer, whose special tokens are a different scheme entirely (training/data/tokenizer.py:9 `tokens <\|bos\|>, <\|eos\|>, <\|pad\|> plus a 16-id reserved block occupy fixed ids` at 0-18, against the spec's pad 0 / eos 1 / bos 2). A from-scratch artifact used consistently for prepare-sft and train-sft passes every in-repo check and silently trains E2B on wrong token ids — the exact failure training/data/tokenizer.py:4 warns about. Only the artifact on the workstation settles which it is; no Gemma 4 tokenizer exists in-tree to check against.

Command:

```bash
$ python3 -c "from tokenizers import Tokenizer; t=Tokenizer.from_file('/mnt/ai-data/training/tokenizers/gemma-4-e2b/tokenizer.json'); print('vocab', t.get_vocab_size()); print('pad/eos/bos', [t.token_to_id(s) for s in ('<pad>','<eos>','<bos>')])"   # expect: vocab 262144, pad/eos/bos [0, 1, 2]
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:112 `\| `vocab_size` \| 262144 \|`; docs/models/gemma-4.md:120-122 `\| `bos_token_id` \| 2 \|`, `\| `eos_token_id` \| 1 \|`, `\| `pad_token_id` \| 0 \|`; docs/models/gemma-4.md:466 `These identifiers occupy positions inside the 262144-entry vocabulary. **Do not trim the`

Observed:

```
```

Status: PENDING

### GH.34 Container run of a real E4B GGUF (RH.12) has no capability-verdict step and the image ships no gguf-info binary

Implementation: `/work/docker-compose.rust-harness.yaml:7`  
Finding: GC-203 (low, `needs-host`)

docs/verification/rust-harness.md:607 runs `make harness-run ARGS='--model /app/models/gguf/<e4b>.gguf --seed 7 ...'` as RH.12's acceptance. The container binary is the CPU subset (runtime/Cargo.toml:13 `cuda = []` is empty, harness/Cargo.toml declares no cuda passthrough, docker/rust-harness/Dockerfile.rust-harness:28 builds `cargo build --release -p harness --locked` with no features, and the compose file reserves no GPU), so on a real E-series GGUF the run ends in cpu::check_supported's refusal rather than a reply. Whether it refuses on per-layer tensor names, the dual-RoPE metadata key, or the softcapping key depends on GGUF conventions that cannot be read offline. The container also cannot produce RH.1's verdict itself, because `gguf-info` is not copied into the image; the four RH.1 questions are answerable only from a host build.

Command:

```bash
$ cd ~/ai-experiments && make harness-build && docker compose -f docker-compose.rust-harness.yaml run --rm harness --model /app/models/gguf/<e4b>.gguf --seed 7 "Reply with the single word: ready"; echo "exit: $?"   # then compare against the host verdict: cd rust-harness && cargo run -p runtime --bin gguf-info -- /mnt/ai-data/models/gguf/<e4b>.gguf
```

Expected signal: agreement with the reference - spec:252-254 "1. **Per-layer input embeddings.** `hidden_size_per_layer_input` is 256 on the E-series and 0 on 12B and 31B. Without this mechanism the E-series does not run at all."; spec:314-316 "The safetensors headers show that every layer carries its own `k_proj` and `v_proj`: 42 of each on E4B, 35 of each on E2B, with no layer omitted."

Observed:

```
```

Status: PENDING

### GH.35 The proxy allowlist cites the authority as its reason to reach huggingface.co, while the authority states a contained sandbox does not reach huggingface.co

Implementation: `/work/docker/agents/proxy/squid.conf:36`  
Finding: GC-163 (low, `needs-host`)

The two non-memory citations of the authority document disagree with it, and with each other about the same fact. squid.conf:38-45 allowlists `.huggingface.co` and `.cdn.hf.co` and carries a first-hand empirical note ("Verified: without it the redirect hop returns squid ERR_ACCESS_DENIED while the first hop succeeds"), which is stronger evidence than the spec's unqualified sentence and suggests the spec's Reproducing section predates the allowlist. If the allowlist works, the spec's regeneration instructions needlessly serialize a cheap check onto the human, and the whole 'Reproducing this document' section is stale. If it does not, the proxy comment advertises an egress path that does not exist. I did not test, per the standing instruction not to fetch huggingface.co from this session.

Command:

```bash
$ make agent-sandbox   # then, inside the sandbox: curl -sI https://huggingface.co/google/gemma-4-E2B/raw/main/config.json | head -1   # a 200 settles it against docs/models/gemma-4.md:610-611
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:609-611 — "Every value above regenerates from the pinned revisions with no GPU and no model download beyond a few hundred kilobytes. Run these on the workstation; a contained agent sandbox does not reach `huggingface.co`."

Observed:

```
```

Status: PENDING

### GH.36 The regeneration recipe carries no Hugging Face token, while the repository elsewhere records Gemma 4 weights as gated

Implementation: `/work/docs/models/gemma-4.md:616`  
Finding: GC-175 (low, `needs-host`)

If `google/gemma-4-*` is gated, an unauthenticated fetch of even `config.json` returns 401 for every command in the Reproducing section (spec:616, :624, :633-635), so the recipe fails on the workstation too and C11-14's reproduction claim is incomplete rather than merely misplaced. None of the four commands passes an Authorization header, and no sandbox carries an HF token in any case (docs/sandbox/agent-sandbox.md:180-183: a Claude session gets only CLAUDE_CODE_OAUTH_TOKEN, an OpenCode session only RUNPOD_API_KEY). This also partially rehabilitates spec:610-611 - a reachable domain is not an authorized fetch. Only a live request settles it; no Gemma 4 artifact exists on this workstation.

Command:

```bash
$ curl -sS -o /dev/null -w "unauth=%{http_code}\n" https://huggingface.co/google/gemma-4-E4B/raw/main/config.json; curl -sS -H "Authorization: Bearer $HF_TOKEN" -o /dev/null -w "auth=%{http_code}\n" https://huggingface.co/google/gemma-4-E4B/raw/main/config.json
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:609-616 - "Every value above regenerates from the pinned revisions with no GPU and no model download beyond a few hundred kilobytes." ... "curl -sL https://huggingface.co/google/gemma-4-E4B/raw/main/config.json"

Observed:

```
```

Status: PENDING

### GH.37 A risks section asserts "transformers 5.13.1 ships the gemma4 architecture" as settled fact and narrows the live risk to TRL/PEFT

Implementation: `/work/docs/training/investigations/f3_sft_entry_point_investigation.md:234`  
Finding: GC-182 (low, `needs-host`)

The sentence retires the exact risk the spec says is open, and it does so inside the document that sets the F3 verification agenda: an operator reading it verifies TRL/PEFT interaction and never verifies that the pinned release resolves `gemma4` at all. No observation backs the assertion - the workstation has never loaded a Gemma 4 checkpoint (docs/verification/training-phase1.md:70-78) and no test asserts it (see M9-pins-02). Suspected root of the whole "transformers already supports Gemma 4" family of claims in this tree: reading transformers' own source at `main` as if it described the pinned release, the same substitution that produced the 2304/30-layer geometry error at docs/training/investigations/p2_d5_optimizer_offload_investigation.md:124-126 ("Public sources (transformers `configuration_gemma4.py`) confirm the architecture") and the readiness paragraph in M9-pins-04.

Command:

```bash
$ uv run --no-sync python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('google/gemma-4-E2B', revision='d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f'); print(type(c).__name__, c.model_type, c.text_config.model_type, c.architectures)"
```

Expected signal: agreement with the reference - spec:98-101: "**All four `config.json` files target unreleased development versions of `transformers`.** Treat the configuration schema as unstable, and pin the `transformers` revision that actually implements `Gemma4ForConditionalGeneration` rather than assuming a released version supports it."

Observed:

```
```

Status: PENDING

### GH.38 F3 asserts released transformers 5.13.1 ships gemma4; unverified in-tree, and the spec warns against assuming it

Implementation: `/work/docs/training/investigations/f3_sft_entry_point_investigation.md:234`  
Finding: GU-047 (low, `needs-host`)

Verdict sharpened from stale-doc to needs-host: nothing in the tree contradicts the claim at f3_sft_entry_point_investigation.md:234-235 ('transformers 5.13.1 ships the gemma4 architecture'), and nothing in the tree confirms it either - only importing the pinned package settles it. The spec's instruction is explicit (spec:98-101: pin the revision that actually implements Gemma4ForConditionalGeneration 'rather than assuming a released version supports it'), and two dev schemas are in play (5.5.0.dev0 for E2B/E4B/31B, 5.10.0.dev0 for 12B, spec:158). The pin itself is exact in pyproject.toml, so reproducibility holds; what is missing is the evidence. docs/verification/training-phase1.md still lists gemma4 support as unproven, so no site in the tree turns the assertion into evidence. Low: a wrong assumption here fails loudly at import.

Command:

```bash
$ docker compose -f docker-compose.training.yaml run --rm training python -c "import transformers; print(transformers.__version__); from transformers import AutoConfig; print(AutoConfig.for_model('gemma4_text').__class__.__name__)"
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:98-101 "**All four `config.json` files target unreleased development versions of `transformers`.** Treat the configuration schema as unstable, and pin the `transformers` revision that actually implements `Gemma4ForConditionalGeneration` rather than assuming a released version supports it."; docs/models/gemma-4.md:158 "\| `transformers_version` \| 5.5.0.dev0 \| 5.5.0.dev0 \| 5.10.0.dev0 \| 5.5.0.dev0 \|"

Observed:

```
```

Status: PENDING

### GH.39 F8's 'critical' KV-sharing training-bug claim is uncited, already promoted into a recipe preflight, and half-applied

Implementation: `/work/docs/training/investigations/f8_recipe_cards_investigation.md:140`  
Finding: GU-049 (low, `needs-host`)

Verdict changed from stale-doc to needs-host and severity from medium to low. f8_recipe_cards_investigation.md:140-144 records a 'Known training bug (critical)': E2B/E4B layers that recompute K/V under use_cache=False, 'corrupting logits and diverging loss'. The finding's premise - that the spec's cache-only reading makes this mechanically impossible - does not survive scrutiny: spec:314-326 describes what the weights contain and how a cache allocator should behave, and says nothing about what the transformers forward pass does when caching is disabled, so the spec neither supports nor refutes the report. Only reading the pinned transformers implementation on the workstation settles it. What is verifiable in-tree, and is the part worth keeping: the claim is uncited and marked 'reported', it has already been promoted into a release gate at training/recipes/reference/sft-qlora-gemma-e4b.yaml:18-21, and the instruction at :143-144 to note it 'in both cards' was half-applied - distill-response-gemma-e2b.yaml and distill-logit-gemma-e2b.yaml carry no such note (grep for use_cache across training/recipes/reference returns only the E4B card). Acting on the note means blocking a run pending a fix that may not exist.

Expected signal: agreement with the reference - docs/models/gemma-4.md:323-326: "A cache allocator that gives all 42 layers independent key-value storage loads and runs correctly but consumes substantially more memory than the architecture requires." and 316-317: "The sharing applies to the key-value cache at inference time, not to the projection parameters."

Observed:

```
```

Status: PENDING

### GH.40 Interp loaders pass the transformers v4 torch_dtype kwarg while parts of the training tier use the v5 dtype name

Implementation: `/work/interp/harvest.py:182`  
Finding: GU-036 (low, `needs-host`)

The spec's field table names the config key `dtype` (spec:157), the v5 rename of `torch_dtype`, and the repo pins transformers 5.13.1. Both spellings coexist: torch_dtype at interp/harvest.py:182, interp/probes/capture.py:89, interp/sae/evaluate.py:135, interp/steering/extract.py:92 and training/eval/holdout.py:49; dtype= at training/export/merge.py:167, training/tune/distill.py:224 and interp/commands/smoke_test.py:215. The split is not tier-clean, so the original framing (interp versus export) is sharpened to: the tree calls from_pretrained both ways and only the pinned image can say whether the alias still resolves. Unsettleable in the sandbox: torch/transformers are absent from the agent image.

Command:

```bash
$ docker compose -f docker-compose.interp.yaml run --rm interp python -c "import inspect, transformers; print(transformers.__version__); print('torch_dtype' in inspect.signature(transformers.AutoModelForCausalLM.from_pretrained).parameters)"
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:157 "\| `dtype` \| `bfloat16` \| `bfloat16` \| (unset) \| `bfloat16` \|"; docs/models/gemma-4.md:98-99 "**All four `config.json` files target unreleased development versions of `transformers`.** Treat the configuration schema as unstable"

Observed:

```
```

Status: PENDING

### GH.41 The one dependency pin in the repository that bears on the spec is Python-side (transformers==5.13.1) and is unverified against Gemma4ForConditionalGeneration

Implementation: `/work/pyproject.toml:19`  
Finding: GC-202 (low, `needs-host`)

5.13.1 is an exact pin and postdates both dev markers (5.5.0.dev0, 5.10.0.dev0), so it plausibly carries the class — but that is exactly the assumption spec:98-101 forbids making without checking, and no artifact in the tree records the check. The consequence is bounded and loud (an ImportError or an unrecognized model_type at fixture-generation time on the workstation), not silent. Noting it here closes the gap-fill question directly: the Rust manifests contribute nothing to the spec's provenance chain, and pyproject.toml:19 is the single dependency pin in the repository that does.

Command:

```bash
$ python3 -c "import transformers; print(transformers.__version__); from transformers import Gemma4ForConditionalGeneration; print('class present')"
```

Expected signal: agreement with the reference - spec:98-101 "**All four `config.json` files target unreleased development versions of `transformers`.** Treat the configuration schema as unstable, and pin the `transformers` revision that actually implements `Gemma4ForConditionalGeneration` rather than assuming a released version supports it." and spec:158 "\| `transformers_version` \| 5.5.0.dev0 \| 5.5.0.dev0 \| 5.10.0.dev0 \| 5.5.0.dev0 \|"

Observed:

```
```

Status: PENDING

### GH.42 Unsettled: if a converter writes feed_forward_length as a per-layer array, E2B is refused before the double-wide diagnostic runs

Implementation: `/work/rust-harness/runtime/src/gguf.rs:113`  
Finding: GU-007 (low, `needs-host`)

Spec 350-352 makes E2B's final num_kv_shared_layers layers 2 * intermediate_size, so a converter must encode two feed-forward widths somehow. If the pinned llama.cpp converter (training/export/convert.py:43-44, convert_hf_to_gguf.py under DEFAULT_LLAMA_CPP_DIR) writes `gemma4.feed_forward_length` as an array, the GGUF is refused one stage earlier than the MatFormer guard at cpu.rs:234-252: either at gguf.rs:117 with "unsupported GGUF array element type" (only types 5/6/8 decode) or at cpu.rs:67-68 with "expected u32, got ArrI32([...])". Both are loud and neither names the feature, and gguf-info's width table is only reached if parse succeeds. Cannot be settled from this tree and must not be settled from remembered llama.cpp conventions; no Gemma 4 weights exist on the workstation, so this becomes a runbook entry.

Command:

```bash
$ python3 -c "import sys,json;d=open('/mnt/ai-data/models/gguf/<e2b>.gguf','rb').read(4096)" ; cd rust-harness && cargo run -p runtime --bin gguf-info -- /mnt/ai-data/models/gguf/<e2b>.gguf | grep -E 'feed_forward_length|Feed-forward widths' -A40
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:350-352 "The operative rule: when `use_double_wide_mlp` is true, the final `num_kv_shared_layers` layers use `2 * intermediate_size`, and the remainder use `intermediate_size`." The specification is derived from Hugging Face config.json and safetensors headers (gemma-4.md:57), not from GGUF.

Observed:

```
```

Status: PENDING

### GH.43 No final logit softcap in the forward pass; the only defense is a metadata key spelling no artifact confirms

Implementation: `/work/rust-harness/runtime/src/model/cpu.rs:887`  
Finding: GC-212 (low, `needs-host`)

spec 115 lists final_logit_softcapping 30.0 among the values identical in all four variants (spec 105). cpu.rs:885-892 goes from the final norm straight to the tied dot product with no tanh cap. Softcapping carries no tensor, so the unconsumed-tensor backstop cannot see it; the whole defense is the substring test at cpu.rs:293, which fires only on a key under the {arch}. prefix whose suffix contains "softcapping" (or is attention.scale / logit_scale). That is the llama.cpp Gemma-2 spelling, and Gemma 4's own cap vocabulary is not uniform — spec 517 gives the audio tower's attention_logit_cap 50.0 — so the spelling cannot be assumed. If the key is absent or spelled otherwise, the checkpoint loads and every logit is uncapped: greedy argmax is unchanged because 30*tanh(z/30) is monotone, so the wrongness hides while perplexity (eval), temperature/top-p sampling and every captured logprob are wrong. The test at cpu.rs:1284-1301 proves only that the guard fires on hand-inserted keys. 31B is the variant whose load reaches this check with nothing earlier to stop it (E2B/E4B are refused at cpu.rs:211). Note the header list at cpu.rs:16-18 does not name softcapping either, so nothing in the file declares the gap.  Propagated to: rust-harness/runtime/src/eval.rs:55. One defect: final_logit_softcapping 30.0 is never applied and the only defense is a metadata-key spelling no artifact confirms. cpu.rs:887 (forward pass) is the ROOT; eval.rs:55 reports the same absence downstream in eval and sampling.

Command:

```bash
$ python3 -c "from gguf import GGUFReader; r=GGUFReader('/mnt/ai-data/models/gemma-4-E4B-f16.gguf'); print([k for k in r.fields if 'softcap' in k or 'logit' in k or 'scale' in k])"
```

Expected signal: agreement with the reference - spec 115: "\| `final_logit_softcapping` \| 30.0 \|" (listed among values that are "identical in E2B, E4B, 12B, and 31B", spec 105)

Observed:

```
```

Status: PENDING

### GH.44 Rust and the numpy golden share one blocked GQA mapping; the n_kv_heads=1 fixture cannot falsify it — extends A1-15

Implementation: `/work/rust-harness/runtime/src/model/cpu.rs:833`  
Finding: GU-067 (low, `needs-host`)

cpu.rs:833 and fixtures/reference_logits.py:204 encode the identical blocked mapping (kvh = qh / heads_per_kv), so the cross-language golden cannot catch a shared misunderstanding of the grouping order, and the committed TinySpec sets n_kv_heads = 1 (synth.rs:42), where blocked and interleaved coincide. E2B is immune for the same reason; E4B's 8 query heads over 2 KV heads is the first geometry where the two mappings differ, and a wrong mapping there produces fluent, wrong text with no error. The spec does not state the grouping order, so only real weights or the reference implementation settle it. Already carried as deferred finding A1-15 (docs/verification/rust-harness-qa.md:467-470); recorded here rather than reopened.

Command:

```bash
$ cd ~/ai-experiments/rust-harness && cargo run -q -p runtime --bin tiny-model -- /tmp/g4geom.gguf # then regenerate goldens at a spec (n_heads 8, n_kv_heads 2) fixture and diff against transformers' Gemma4 attention on the same weights
```

Expected signal: agreement with the reference - spec:206-208 "The shapes follow `num_attention_heads * head_dim` for sliding layers ... with the key and value projections scaled by `num_key_value_heads` instead."; spec:147 "\| GQA ratio \| 8:1 (MQA) \| 4:1 \| 2:1 \| 2:1 \|"

Observed:

```
```

Status: PENDING

### GH.45 vocab_size is the token-array length, so a short array would refuse a conforming 262144-row checkpoint

Implementation: `/work/rust-harness/runtime/src/model/cpu.rs:79`  
Finding: GU-068 (low, `needs-host`)

vocab_size is derived from tokenizer.ggml.tokens (cpu.rs:79-82) and used to length-check token_embd.weight (:675) and to size the tied logits loop (:887); a disagreeing {arch}.vocab_size key is explicitly accepted as descriptive (test at 1304-1307). Spec 466-469 forbids trimming the 262144-entry vocabulary, and the multimodal control ids run to 258884, so if a converted file's token array is shorter than the embedding row count the loader refuses a correct checkpoint with a length error. Downgraded to low: the premise is speculative, there is no in-tree evidence of a trimmed array, and the failure would be loud.

Command:

```bash
$ cd ~/ai-experiments/rust-harness && cargo run -p runtime --bin gguf-info -- /mnt/ai-data/models/gguf/<e4b>.gguf | grep -iE 'tokenizer.ggml.tokens|vocab_size|token_embd'
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:112 "\| `vocab_size` \| 262144 \|"; :466-468 "These identifiers occupy positions inside the 262144-entry vocabulary. **Do not trim the vocabulary for text-only work.** The embedding table is sized to the full vocabulary and the weights are tied to it"

Observed:

```
```

Status: PENDING

### GH.46 The 13 GGUF layer tensor names the loader consumes are a converter convention no artifact here confirms

Implementation: `/work/rust-harness/runtime/src/model/cpu.rs:156`  
Finding: GU-072 (low, `needs-host`)

None of LAYER_TENSORS (cpu.rs:156-170), TOP_LEVEL_TENSORS (:175), INERT_TENSORS (:180) or the test-only per_layer_token_embd name appears in the spec, which is derived from safetensors headers. The intended correspondence (input_layernorm -> attn_norm, post_attention_layernorm -> post_attention_norm, pre_feedforward_layernorm -> ffn_norm, post_feedforward_layernorm -> post_ffw_norm, {q,k,v,o}_proj -> attn_{q,k,v,output}, {gate,up,down}_proj -> ffn_*) is plausible but is exactly the half-remembered convention the audit forbids asserting. Every named guard rides on it. Graded low because a wrong guess produces a loud missing-tensor refusal, not wrong numbers, and one gguf-info run settles the whole set at once.

Command:

```bash
$ cd /work/rust-harness && cargo run -p runtime --bin gguf-info -- /mnt/ai-data/models/gguf/<e4b>.gguf | sed -n '/Tensor families/,/Quantization/p'
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:370-392 — "\| 42 \| `model.language_model.layers.{L}.input_layernorm.weight` \| `[2560]` \| ... \| 35 \| `model.language_model.layers.{L}.self_attn.q_norm.weight` \| `[256]` \| / \| 7 \| `model.language_model.layers.{L}.self_attn.q_norm.weight` \| `[512]` \|"

Observed:

```
```

Status: PENDING

### GH.47 A converter that materializes the tied head as output.weight would be refused by the unconsumed-tensor backstop

Implementation: `/work/rust-harness/runtime/src/model/cpu.rs:175`  
Finding: GU-073 (low, `needs-host`)

The safetensors side is settled (spec 132-134, 400: no lm_head). The GGUF side is not: converters sometimes materialize a tied head as output.weight, which is in neither TOP_LEVEL_TENSORS nor INERT_TENSORS, so the backstop at 321-328 would refuse a conforming checkpoint naming a tensor family that duplicates one already loaded. Cannot be settled from llama.cpp conventions; only the file settles it.

Command:

```bash
$ python3 -c "import gguf;r=gguf.GGUFReader('/mnt/ai-data/models/gguf/<e4b>.gguf');print([t.name for t in r.tensors if t.name in ('output.weight','token_embd.weight')])"
```

Expected signal: agreement with the reference - line 400: "- There is **no `lm_head`**. Output logits reuse the tied input embedding."; line 132-134: "`tie_word_embeddings` is confirmed in the weights, not only in the configuration: neither E2B nor E4B contains an `lm_head` tensor."

Observed:

```
```

Status: PENDING

### GH.48 text_config descent keys on the literal "gemma4"; the spec never states the E-series top-level model_type

Implementation: `/work/training/calculator/components.py:232`  
Finding: GU-025 (low, `needs-host`)

Both HF_SUPPORTED_MODEL_TYPES (:80-90) and the descent condition (:232) assume the E-series composite config reports model_type exactly "gemma4". The spec derives its E-series rows from `architectures` and `text_config.model_type` (gemma-4.md:85) and states a top-level string only for 12B (`gemma4_unified`, :89-90). Unsettleable offline; no Gemma 4 weights or configs exist on this workstation, so this becomes a runbook entry for the first host-side pull. Low because the failure is loud either way - a mismatch raises UnknownArchitectureError before any number is produced.

Command:

```bash
$ curl -sL https://huggingface.co/google/gemma-4-E2B/resolve/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f/config.json | python3 -c "import json,sys;c=json.load(sys.stdin);print(repr(c.get('model_type')), repr(c['text_config'].get('model_type')), c.get('architectures'))"
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:89-91 "The 12B model is a separate implementation, not a size variant. Its top-level `model_type` is `gemma4_unified`" — the corresponding top-level string for the E-series and 31B is stated nowhere; :85 gives only "\| E-series \| E2B, E4B \| `Gemma4ForConditionalGeneration` \| `gemma4_text` \|" for architectures and text model_type.

Observed:

```
```

Status: PENDING

### GH.49 Recipes state the Gemma 4 weights are gated while the spec's reproduce curls are unauthenticated

Implementation: `/work/training/recipes/reference/continual-gemma4-e2b.yaml:10`  
Finding: GU-061 (low, `needs-host`)

A gated Hugging Face repository returns 401 for /raw/ and /resolve/ without an Authorization header, config.json included. Either the repositories are not gated - in which case the recipes' HF_TOKEN requirement and the phase-1 blocker at docs/verification/training-phase1.md:51 are carrying a Gemma 2/3 assumption forward - or the spec's reproduce block (spec:609-611, :616, :622-625, :633-635) is missing the bearer header on all three commands and every future regeneration fails at the first curl. One unauthenticated request settles it; no weights are needed.

Command:

```bash
$ curl -s -o /dev/null -w '%{http_code}\n' https://huggingface.co/google/gemma-4-E2B/raw/main/config.json
```

Expected signal: agreement with the reference - docs/models/gemma-4.md:609-611 "Every value above regenerates from the pinned revisions with no GPU and no model download beyond a few hundred kilobytes."; docs/models/gemma-4.md:616 "curl -sL https://huggingface.co/google/gemma-4-E4B/raw/main/config.json"

Observed:

```
```

Status: PENDING

### GH.50 Two F8 distill recipes name teacher google/gemma-4-E4B-it, a repository the spec's provenance table omits

Implementation: `/work/training/recipes/reference/distill-logit-gemma-e2b.yaml:43`  
Finding: GU-062 (low, `needs-host`)

spec:44-49 lists exactly four repositories and spec:61 enumerates the base repo file list; no instruction-tuned variant appears anywhere in the spec, so the existence, revision, tokenizer identity and chat template of google/gemma-4-E4B-it are all unattested. If the id is wrong, make train-distill fails at teacher load after the student is already resident; if it exists but its tokenizer differs from google/gemma-4-E2B (the recipe's tokenizer.path), the logit-tier hash-identity preflight refuses the run. Neither can be settled in the sandbox. Same repository named at distill-response-gemma-e2b.yaml:5, :21 and :40.

Command:

```bash
$ curl -sI https://huggingface.co/google/gemma-4-E4B-it/resolve/main/config.json | head -1; curl -sI https://huggingface.co/google/gemma-4-E4B-it/resolve/main/config.json | grep -i '^x-repo-commit'
```

Expected signal: agreement with the reference - spec:44-49 lists exactly four repositories — `google/gemma-4-E2B`, `google/gemma-4-E4B`, `google/gemma-4-12B`, `google/gemma-4-31B` — and spec:61 "Repository file list (E2B, and E4B matches):" enumerates the base repo contents

Observed:

```
```

Status: PENDING

