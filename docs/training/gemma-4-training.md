# Gemma 4 training reference

What the training tier requires to size, load, fine-tune, and export a Gemma 4 checkpoint.
Every value below has a named consumer in `/training` or `/common`; facts with no consumer are
deliberately absent, and the general family reference at `/docs/models/gemma-4.md` covers them.

Scope: E2B and E4B, the two variants named in the recipes. The 12B and 31B variants appear in
no code path here.

## Table of contents

- [Provenance and the revisions to pin](#provenance-and-the-revisions-to-pin)
- [Architecture ingestion](#architecture-ingestion)
- [Parameter accounting](#parameter-accounting)
- [What the parameter count counts](#what-the-parameter-count-counts)
- [Memory](#memory)
- [Tensor names the freeze patterns depend on](#tensor-names-the-freeze-patterns-depend-on)
- [Dispatch, identity, and export](#dispatch-identity-and-export)
- [Tokenizer and chat template](#tokenizer-and-chat-template)
- [Context and attention](#context-and-attention)
- [Open questions](#open-questions)
- [Reproducing this document](#reproducing-this-document)

## Provenance and the revisions to pin

| Model | Repository | Revision (`x-repo-commit`) | `config.json` blob |
|-------|------------|----------------------------|--------------------|
| E2B | `google/gemma-4-E2B` | `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` | `5f0662eb398ede0ab1515c4760e13bcd6bf13f2f` |
| E4B | `google/gemma-4-E4B` | `411aa17b749aa952df1359d2dcea73917a544d9a` | `080fd8d51bb6e846df168157c9d57b928a2a2687` |

**Every Gemma 4 recipe whose `model.hf.id` or `harvest.model_id` is a hub id pins the
block's `revision` to the commit above**, and `/tests/training/test_recipe_pins.py` holds each
pin equal to `/schemas/gemma4_facts.json` across the reference, organism, and interp recipes. The field has two live consumers —
`/training/models/architecture.py` (`spec_from_recipe`) passes it to `AutoConfig.from_pretrained`, and
`/training/export/merge.py:148-160` passes it to `AutoModelForCausalLM.from_pretrained` — so
pinning costs nothing and is required by the pin-then-fix-then-update rule in `/AGENTS.md`.
A recipe whose id is a local export directory has no hub revision to pin.

The blob values above are **git blob hashes**, as published by Hugging Face. They are not
comparable to the `config_sha256` that `/training/export/preflight.py:49-98` records, which is a
SHA-256 over the raw `config.json` bytes. Do not expect the two to match.

### Repository layout

```
config.json              generation_config.json    model.safetensors
processor_config.json    tokenizer.json            tokenizer_config.json
README.md
```

Three consequences:

- **There is no `model.safetensors.index.json`.** The weights ship as a single unsharded file
  (10,246,621,918 bytes for E2B; 15,992,595,884 for E4B). `/training/export/merge.py:185-191`
  copies an index only when present, so it is already correct; a loader that requires one is not.
- **There is no `chat_template.jinja`**, and `tokenizer_config.json` carries no `chat_template`
  key either. `/training/export/merge.py:22-31,100` copies `chat_template.jinja` when it exists, so
  a merged export inherits no chat template from the base repository. The template must come
  from `/training/models/families.py`; see
  [Tokenizer and chat template](#tokenizer-and-chat-template).
- **`.cache/` and `.gitattributes` are download artifacts**, not upstream content. They vary
  between local copies and carry no meaning.

## Architecture ingestion

All architecture reads happen in one function, `spec_from_hf_config`
(`/training/models/architecture.py`). It descends into `text_config` only when the
**top-level** `model_type` is `gemma4`, then builds an `ArchitectureSpec`.

| `ArchitectureSpec` field | `config.json` source | E2B | E4B |
|--------------------------|----------------------|-----|-----|
| `layers` | `text_config.num_hidden_layers` | 35 | 42 |
| `hidden` | `text_config.hidden_size` | 1536 | 2560 |
| `heads` | `text_config.num_attention_heads` | 8 | 8 |
| `kv_heads` | `text_config.num_key_value_heads` | 1 | 2 |
| `ffn` | `text_config.intermediate_size` | 6144 | 10240 |
| `vocab` | `text_config.vocab_size` | 262144 | 262144 |
| `tie_embeddings` | `text_config.tie_word_embeddings` | `true` | `true` |
| `explicit_head_dim` | `text_config.head_dim` | 256 | 256 |
| `ple_vocab` | `text_config.vocab_size_per_layer_input` | 262144 | 262144 |
| `ple_dim` | `text_config.hidden_size_per_layer_input` | 256 | 256 |

E2B has a single key-value head, making it a true multi-query model. Code that derives a group
size by division without checking for 1 breaks on it.

Two fields are read but are not what their names suggest:

- **`head_dim` is not the head dimension of every layer.** Sliding layers use 256; the seven
  full-attention layers use `global_head_dim`, which is 512 in both variants. The spec reads the
  `layer_types` array and prices each layer at its own width. See
  [What the parameter count counts](#what-the-parameter-count-counts).
- **`vocab_size_per_layer_input` is populated even where the feature is disabled.** Gate on
  `hidden_size_per_layer_input`, which is the field that determines whether per-layer embeddings
  exist. Both are populated on E2B and E4B, so this matters only if a variant without the
  feature is added later.

`enable_moe_block` is `false` and `num_experts` is `null` on both variants, so the guard at
`components.py:373-378` does not fire.

## Parameter accounting

Exact counts, summed from the safetensors tensor shapes. `ArchitectureSpec.param_count()`
reproduces the language-model row to the parameter (minus the `layer_scalar` buffers, which
are not parameters), pinned by `tests/training/test_architecture.py`; the tower row is read from
the checkpoint header. `python -m training step` (`/training/commands/train_step.py`) compares
the sum against the torch-reported count of the built model.

| Quantity | E2B | E4B |
|----------|-----|-----|
| Main embedding | 402,653,184 | 671,088,640 |
| Per-layer embedding table | 2,348,810,240 | 2,818,572,288 |
| Per-layer gates, projections, norms | 41,341,696 | 82,683,136 |
| Attention | 297,292,800 | 642,277,888 |
| Feed-forward | 1,557,135,360 | 3,303,014,400 |
| Transformer body (no embeddings, no per-layer table) | 1,895,986,467 | 4,028,408,106 |
| Body plus main embedding | 2,298,639,651 | 4,699,496,746 |
| Language model total | 4,647,449,891 | 7,518,069,034 |
| Vision and audio towers, with bridges | 475,729,088 | 478,088,384 |
| **Total resident weights** | **5,123,178,979** | **7,996,157,418** |

The `E` prefix in the model names denotes effective parameters. The "body plus main embedding"
row — 2.30 billion and 4.70 billion — is the row that tracks the `2` and `4` in `E2B` and
`E4B`, which suggests the effective count excludes the per-layer embedding table. That reading
is an inference from arithmetic and the naming, not a definition stated by any file here.

The one size figure the artifacts do carry is `general.size_label` in the converted GGUF files:
`4.6B` for E2B and `7.5B` for E4B. Those match the language-model total row
(4,647,449,891 and 7,518,069,034), so `size_label` counts the full language model including the
per-layer table and excluding the towers. Do not read it as the effective count.

There is **no `lm_head` tensor** in either checkpoint. Output logits reuse the tied input
embedding, which is why `tie_embeddings` is load-bearing rather than cosmetic.

### The double-wide feed-forward rule

E2B sets `use_double_wide_mlp: true`, and the flag does not apply uniformly. Of its 35 layers,
15 use the configured `intermediate_size` of 6144 and **20 use 12288**.

| Layer range | Count | `gate_proj` / `up_proj` | `down_proj` |
|-------------|-------|-------------------------|-------------|
| 0 through 14 | 15 | `[6144, 1536]` | `[1536, 6144]` |
| 15 through 34 | 20 | `[12288, 1536]` | `[1536, 12288]` |

The operative rule: when `use_double_wide_mlp` is true, the final `num_kv_shared_layers` layers
use `2 * intermediate_size` and the remainder use `intermediate_size`. On E2B
`num_kv_shared_layers` is 20, and the double-wide region is exactly the tail 20 layers.

E4B sets the flag false and has uniform widths of 10240 across all 42 layers, despite also
having a shared-KV tail. The compensation is not implied by KV sharing; it requires the flag.

## What the parameter count counts

Until 2026-09-19 `ArchitectureSpec.param_count()` assumed one feed-forward width, one head
dimension and two normalizations per layer, and undercounted E2B by 13.5 percent (629,721,891
parameters) and E4B by 1.6 percent. The double-wide feed-forward rule was fixed first; the
rest landed with the memory-fit calculator's 2026-09-19 reduction (the calculator retired on 2026-09-20; the count stayed, in `training/models/architecture.py`), and the count now
matches the language-model row above exactly for both variants
(`tests/training/test_architecture.py`,
`test_language_model_count_matches_the_checkpoint_exactly`). What each former cause maps to
in `spec_from_hf_config` (`/training/models/architecture.py`):

| Cause | Now |
|-------|-----|
| Feed-forward: the 20 double-wide layers | `double_wide_layers` from `use_double_wide_mlp` and `num_kv_shared_layers` |
| Attention projections: seven 512-wide full-attention layers | `full_attention_layers` counted from `config.layer_types`, priced at `global_head_dim` |
| Query and key norms on every layer | `qk_norms`, at each layer's own head width |
| Per-layer embeddings: `per_layer_model_projection` and the norms | `ple_param_count()` carries the model-level projection and its norm, and the per-layer norm |
| Block normalizations: four, not two | `block_norms` is 4 for the Gemma sandwich |
| `layer_scalar`: one scalar per layer | a buffer, not a parameter; deliberately uncounted |

Two deltas remain by design, and FT.calibrate-qlora's `Parameter count` line is where they show:

- **Shared-KV residue.** The count includes `k_proj`, `v_proj` and `k_norm` on all 35 layers,
  as the checkpoint stores them. The pinned `transformers` guards those modules on the 20
  shared layers (`docs/models/gemma-4.md`, the shared-KV weight conflict), so the built model
  reports fewer parameters: 18,880,512 at each layer's own width (16 sliding layers at
  2 x 256 x 1536 + 256, and the 4 full-attention layers among the tail, 19, 24, 29 and 34,
  at 2 x 512 x 1536 + 512). An earlier figure here, 15,733,760, priced all 20 at the sliding
  width. The built model reported 18,881,440 on 2026-09-20 (FT.calibrate-qlora's seventh
  attempt): the residue plus 928 that nothing yet explains, pinned as measured by
  `tests/hardware/test_step_qlora.py`. A drift warning of that size is the modeling code's
  reading, not a counting defect. The adapter count is the other way round: it prices what
  the built model constructs, so a KV-shared layer contributes no `k_proj` or `v_proj`
  adapter (KB-90).
- **Towers.** `from_pretrained` loads the vision and audio towers with the language model, so
  they are resident and must be in the weights term. They are not modeled: `spec_from_recipe`
  reads the `model.safetensors` header of a local base directory and sums every tensor
  outside `model.language_model.` into `extra_params` (475,729,088 on E2B). A hub id resolved
  through a cache prices them at zero and says nothing.

Nothing in the tree prices activations or a key-value cache: what a recipe holds is
measured by stepping it (see [train_step.md](train_step.md)), so `num_kv_shared_layers`,
`layer_types` and the sliding window have no consumer beyond the parameter count.

## Memory

At bfloat16, two bytes per parameter:

| Quantity | E2B | E4B |
|----------|-----|-----|
| Per-layer embedding table alone | 4.38 GiB | 5.25 GiB |
| Language model total | 8.66 GiB | 14.00 GiB |
| Towers and bridges | 0.89 GiB | 0.89 GiB |
| **Total resident weights** | **9.54 GiB** | **14.89 GiB** |

The per-layer embedding table dominates: on E2B it exceeds the entire transformer body. A plan
sized against the advertised effective parameter count underestimates resident weight memory by
roughly a factor of two, before optimizer state.

The tower mass is close to a gigabyte on both variants and is loaded by
`AutoModelForCausalLM.from_pretrained` against these repositories, so a 32 GiB budget must carry
it even for text-only work.

## Tensor names the freeze patterns depend on

`FROZEN_PATTERNS["transformer_stack"]` (`/training/runtime/offload.py:48-50`) is the substring
tuple `("embed_tokens", "lm_head", "per_layer")`. Against these checkpoints:

| Pattern | Matches | Parameters frozen (E2B) |
|---------|---------|-------------------------|
| `embed_tokens` | `model.language_model.embed_tokens.weight`, and also `model.language_model.embed_tokens_per_layer.weight` | 2,751,463,424 |
| `per_layer` | `embed_tokens_per_layer`, `per_layer_model_projection`, `per_layer_projection_norm`, and per-layer `per_layer_input_gate`, `per_layer_projection`, `post_per_layer_input_norm` | overlaps the above |
| `lm_head` | nothing — the head is tied and no such tensor exists | 0 |

The two effective patterns overlap on `embed_tokens_per_layer`, which is harmless. The union
freezes the main embedding and every per-layer-embedding term, which is the documented intent.
`lm_head` matching nothing is expected, not a bug.

Note that `layer_scalar` matches no pattern, but that is not the same as staying trainable. It
is an `nn.Buffer`, so it never appears in `named_parameters()` — which is what the freeze
filters iterate (`training/runtime/offload.py`) — and no freeze pattern can expose a buffer to
an optimizer. It is untrainable by construction rather than by filter.

This is worth stating explicitly because the stored values are **not** 1.0: they span 0.0186 to
0.879 on E2B, so a scalar that did reach the optimizer would be a live tuning knob rather than
an inert one. It does not. See
[`layer_scalar` in docs/models/gemma-4.md](../models/gemma-4.md#layer_scalar).

## Dispatch, identity, and export

| Fact | Value | Consumer |
|------|-------|----------|
| Top-level `model_type` | `gemma4` | `components.py:361,368`; `/interp/families.py:49-57` |
| Text `model_type` | `gemma4_text` | same |
| `architectures[0]` | `Gemma4ForConditionalGeneration` | `/rust-harness/runtime/src/checkpoint.rs` (declines the towers under `--text-only`) |
| `final_logit_softcapping` | 30.0 | `training/models/head_loss.py` applies it per chunk inside the training loss (KB-87); the rust harness applies it |
| `dtype` | `bfloat16` | — |
| `transformers_version` | `5.5.0.dev0` | — |

`architectures[0]` is `Gemma4ForConditionalGeneration`, not `Gemma4ForCausalLM`: the
checkpoint carries vision and audio towers beside the text stack. The exporter keeps them in
the merged file, and the harness loads that file under `--text-only`, declining the towers
by tensor prefix rather than by this string.

The towers matter to the PEFT wrap as well. PEFT matches a `target_modules` list by suffix
against every module key of the composite model, and the vision and audio towers name their
projections `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj`
just as the decoder does, behind `Gemma4ClippableLinear` wrappers (a clamp around an
`nn.Linear`) that PEFT cannot adapt: the first FT.calibrate-qlora attempt died on
`vision_tower.encoder.layers.0.self_attn.q_proj` (KB-83). `build_peft_model`
(`/training/models/peft_utils.py`) therefore passes an `exclude_modules` pattern that keeps the
adapter on `model.language_model.` and off everything else, so a text-only adapter touches the
35 decoder layers alone, which is also what `adapter_param_count` prices.

**The configuration targets an unreleased `transformers` development version.** Treat the
configuration schema as unstable and pin the `transformers` revision that actually implements
`Gemma4ForConditionalGeneration` rather than assuming a released version supports it.

## Tokenizer and chat template

The tokenizer class is `GemmaTokenizer`, wrapped by `Gemma4Processor`. `padding_side` is
`left`. `model_max_length` is the sentinel 1000000000000000019884624838656 and carries no
information; use `max_position_embeddings` instead.

| Token | String | Identifier |
|-------|--------|------------|
| Beginning of sequence | `<bos>` | 2 |
| End of sequence | `<eos>` | 1 |
| Padding | `<pad>` | 0 |

The ordering of the first two looks transposed and is not. Preserve it.

`/training/data/prepare.py:42,65-139` resolves the end-of-sequence token by string, with fallbacks
`("<|eos|>", "<|endoftext|>", "</s>")` — **none of which is `<eos>`**. Pass `--eos-token <eos>`
explicitly, or rely on the tokenizer exposing `eos_token_id`.

`tokenizer_config.json` reports zero `added_tokens_decoder` entries and no `chat_template`, so
the template must come from the family registry. `/training/models/families.py:82-89` carries
it, and the values match these weights:

| Field | Value |
|-------|-------|
| `bos` | `<bos>` |
| `turn_prefix` | `<\|turn>{role}\n` |
| `eot` | `<turn\|>` |
| `turn_suffix` | `\n` |
| `assistant_role` | `model` |

The markers are asymmetric: the bar denotes the open or close side. The same convention governs
`<|channel>`/`<channel|>`, `<|tool_call>`/`<tool_call|>`, `<|tool>`/`<tool|>`, and
`<|tool_response>`/`<tool_response|>`. Thinking markers are `<|think|>` and
`<|channel>thought`, with `thinking_default` off.

Multimodal identifiers occupy positions inside the 262144-entry vocabulary: `boi_token_id`
255999, `boa_token_id` 256000, `image_token_id` 258880, `audio_token_id` 258881, `eoi_token_id`
258882, `eoa_token_id` 258883, `video_token_id` 258884. **Do not trim the vocabulary for
text-only work.** The embedding table is sized to the full vocabulary and the output weights are
tied to it, so removing entries breaks both the output projection and tokenizer round-tripping.

Default generation parameters, from `generation_config.json`:

```json
{
  "bos_token_id": 2,
  "do_sample": true,
  "eos_token_id": 1,
  "pad_token_id": 0,
  "temperature": 1.0,
  "top_k": 64,
  "top_p": 0.95,
  "transformers_version": "5.5.0.dev0"
}
```

## Context and attention

`max_position_embeddings` is **131072** on both variants, not the 256K cited in release
material. Nothing in the training tier reads it: sequence length comes entirely from
`recipe.data.seq_len`, which ranges from 1024 to 4096 across the reference recipes. No check
compares a requested length against the model limit.

`sliding_window` is 512, and the attention type alternates on a fixed schedule:

| Model | Layers | Sliding | Full | Full-attention indices |
|-------|--------|---------|------|------------------------|
| E2B | 35 | 28 | 7 | 4, 9, 14, 19, 24, 29, 34 |
| E4B | 42 | 35 | 7 | 5, 11, 17, 23, 29, 35, 41 |

The attention implementation selected by `/training/models/model.py:25-40` is
`flash_attention_2` or `sdpa`, neither of which is given sliding-window parameters here. The
varlen packing at `/training/tune/sft.py:80-127` treats a position reset as a sequence boundary,
which is a global-attention assumption. At the sequence lengths in use (up to 4096) this is a
question about correctness of the mask, not about memory.

## Resolved from upstream source

Three questions previously recorded here are settled against the pinned `transformers`
sources. [docs/models/gemma-4.md](../models/gemma-4.md) holds the full derivations with line
citations; `docs/models/gemma4-upstream.txt` is the committed transcript they cite, and
`tests/models/test_gemma4_facts_contract.py` asserts those citations still resolve.

- **`layer_scalar` is a buffer applied last.** `modeling_gemma4.py` declares it as
  `nn.Buffer(torch.ones(1))` and multiplies the whole layer output by it as the final
  operation, after all three residual updates. It is initialised to ones and has no
  `config.json` field. Because it is a buffer rather than a parameter, it is not trainable
  under the `transformer_stack` filter and the count is right not to include it. The
  released checkpoints **do** store values other than 1.0 — 0.0186 to 0.879 on E2B — so the
  multiplication is not a no-op, though it remains outside the optimizer's reach.
- **The per-layer embedding gate is multiplicative, and joins after the feed-forward block.**
  The per-layer input is gated in 256-dimensional space, projected back to `hidden_size`, and
  added as a third residual update following attention and the feed-forward block. The layer
  order is therefore attention, feed-forward, per-layer embedding, then `layer_scalar`.
- **The shared-KV boundary is `num_hidden_layers - num_kv_shared_layers`**, giving index 15 on
  E2B and 24 on E4B. E4B's boundary is no longer an assumption. The same computation also
  gates the double-wide MLP, which is why the two regions coincide on E2B.

  Upstream also **revises** what this document previously implied: there are **two** donor
  layers, not one, keyed by layer type. A shared layer reads the donor matching its own
  sliding or full attention type. An implementation that designates a single donor for every
  shared layer is wrong for one of the two regimes.

## Resolved: the auto-class accepts Gemma 4

The merge path in `training/export/merge.py` loads through `AutoModelForCausalLM` while the
checkpoint declares `Gemma4ForConditionalGeneration`, and the committed transcript carries no
auto-class mapping, so acceptance was unverified. It is now verified against a real load on
the workstation:

```
transformers 5.13.1
gemma4 in causal-lm map: ['gemma4', 'gemma4_assistant', 'gemma4_text',
                          'gemma4_unified', 'gemma4_unified_assistant',
                          'gemma4_unified_text']
has Gemma4ForConditionalGeneration: True
config type: Gemma4Config | model_type: gemma4
AutoModelForCausalLM OK: Gemma4ForConditionalGeneration
```

`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` registers six `gemma4*` keys, and
`AutoModelForCausalLM.from_config` returns `Gemma4ForConditionalGeneration` rather than
refusing. Both the composite `gemma4` and the text-only `gemma4_text` config resolve, so the
merge path and any caller that hardcodes the causal-LM auto-class work unmodified. This also
clears the interp harvest, which reaches the model through SAELens's hardcoded
`model_class_name="AutoModelForCausalLM"`.

## Open questions

Recorded rather than guessed.

None outstanding for this document. The last one — the stored value of `layer_scalar` — was
resolved by reading it out of the released checkpoints: the values are not 1.0, and the
consequences for this tier are covered under the freeze filter above.

## Reproducing this document

The safetensors transcript regenerates on the workstation with no GPU and no network:

```bash
./scripts/models/gemma4_header_transcript.sh
```

The upstream sources behind [Resolved from upstream source](#resolved-from-upstream-source)
are fetched at their pinned SHAs by a second script, which needs no workstation:

```bash
./scripts/models/upstream_source_transcript.sh --out docs/models/gemma4-upstream.txt
```

Parameter totals recompute from the tensor inventory in that transcript by summing the product
of each shape times its count.
