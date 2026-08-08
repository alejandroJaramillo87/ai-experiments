# Gemma 4 model family reference

Architecture reference for the Gemma 4 dense models, written so that a session working on
`rust-harness`, `training`, or `interp` does not have to infer model internals. Every
structural fact below is traced to a pinned Hugging Face revision and, where possible, to the
tensor shapes in the published weights rather than to the configuration file alone.

Google DeepMind released Gemma 4 on 2 April 2026 under the Apache 2.0 license. The release
postdates the training cutoff of several models in common use, so prior knowledge of Gemma 1
through Gemma 3 does not transfer reliably. Several fields in Gemma 4 carry names that
suggest behavior they do not have; those are called out explicitly in
[Traps and discrepancies](#traps-and-discrepancies).

Scope: the dense text models E2B, E4B, 12B, and 31B. The 26B mixture-of-experts variant is out
of scope. The vision and audio towers are documented structurally in
[Modality towers](#modality-towers) but are not the subject of this document.

## Table of contents

- [Provenance and pinned revisions](#provenance-and-pinned-revisions)
- [Three code paths, not one family](#three-code-paths-not-one-family)
- [Invariants across all four variants](#invariants-across-all-four-variants)
- [Per-variant text configuration](#per-variant-text-configuration)
- [Attention](#attention)
- [Rotary position embeddings](#rotary-position-embeddings)
- [E-series exclusive features](#e-series-exclusive-features)
- [Per-layer input embeddings](#per-layer-input-embeddings)
- [Cross-layer KV cache sharing](#cross-layer-kv-cache-sharing)
- [The double-wide MLP](#the-double-wide-mlp)
- [Tensor inventory](#tensor-inventory)
- [Parameter accounting and memory](#parameter-accounting-and-memory)
- [Tokenizer and special tokens](#tokenizer-and-special-tokens)
- [Modality towers](#modality-towers)
- [Traps and discrepancies](#traps-and-discrepancies)
- [Resolved from upstream source](#resolved-from-upstream-source)
- [Per-tier support](#per-tier-support)
- [Open questions](#open-questions)
- [Reproducing this document](#reproducing-this-document)

## Provenance and pinned revisions

Every fact in this document derives from the files at these exact repository commits. Record a
new revision here before updating any value, in keeping with the pin-then-fix-then-update rule
in `/AGENTS.md`.

| Model | Repository | Revision (`x-repo-commit`) |
|-------|------------|----------------------------|
| E2B | `google/gemma-4-E2B` | `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` |
| E4B | `google/gemma-4-E4B` | `411aa17b749aa952df1359d2dcea73917a544d9a` |
| 12B | `google/gemma-4-12B` | `023679ed352de9bb66cc873c9009ce3482585c08` |
| 31B | `google/gemma-4-31B` | `5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89` |

The `config.json` blob hash for E4B at that revision is
`080fd8d51bb6e846df168157c9d57b928a2a2687`; for E2B,
`5f0662eb398ede0ab1515c4760e13bcd6bf13f2f`; for 12B,
`6d3994fd0b98fd5eaaaadbf918bd826d778ce5ea`; for 31B,
`b469f1aca87e14dab2fcf32d071d3c58043357b7`.

Tensor names and shapes for E2B and E4B come from the safetensors headers at those revisions.
Tensor-level data for 12B and 31B has not been collected; statements about those two models
derive from `config.json` only, and are marked as such where the distinction matters.

Repository file list (E2B, and E4B matches):

```
.gitattributes
README.md
config.json
generation_config.json
model.safetensors
processor_config.json
tokenizer.json
tokenizer_config.json
```

The weights ship as a single unsharded `model.safetensors`. There is no
`model.safetensors.index.json`; requesting one returns `Entry not found`. Code that locates
weights by reading a shard index fails on these repositories.

## Three code paths, not one family

The four dense models do not share a single implementation. They split into three groups, and
the split does not follow parameter count.

| Group | Models | `architectures` | Text `model_type` |
|-------|--------|-----------------|-------------------|
| E-series | E2B, E4B | `Gemma4ForConditionalGeneration` | `gemma4_text` |
| Unified | 12B | `Gemma4UnifiedForConditionalGeneration` | `gemma4_unified_text` |
| Standard dense | 31B | `Gemma4ForConditionalGeneration` | `gemma4_text` |

The 12B model is a separate implementation, not a size variant. Its top-level `model_type` is
`gemma4_unified`, its vision and audio configurations are restructured, and its
`transformers_version` is `5.10.0.dev0` against `5.5.0.dev0` for the other three. Any dispatch
that matches on `model_type == "gemma4_text"` silently excludes 12B.

The E-series shares a class with 31B but differs from it structurally more than 12B does: the
E-series carries per-layer input embeddings and cross-layer KV sharing, and 31B carries
neither.

**All four `config.json` files target unreleased development versions of `transformers`.**
Treat the configuration schema as unstable, and pin the `transformers` revision that actually
implements `Gemma4ForConditionalGeneration` rather than assuming a released version supports
it.

## Invariants across all four variants

These values are identical in E2B, E4B, 12B, and 31B. They are safe to treat as constants,
and a verification pass should assert them rather than read them per model.

| Field | Value |
|-------|-------|
| `head_dim` | 256 |
| `global_head_dim` | 512 |
| `vocab_size` | 262144 |
| `hidden_activation` | `gelu_pytorch_tanh` |
| `rms_norm_eps` | 1e-06 |
| `final_logit_softcapping` | 30.0 |
| `tie_word_embeddings` | `true` |
| `attention_bias` | `false` |
| `attention_dropout` | 0.0 |
| `initializer_range` | 0.02 |
| `bos_token_id` | 2 |
| `eos_token_id` | 1 |
| `pad_token_id` | 0 |
| `enable_moe_block` | `false` |
| `num_experts` | `null` |
| `top_k_experts` | `null` |
| RoPE (full attention) | `rope_theta` 1000000.0, `rope_type` `proportional`, `partial_rotary_factor` 0.25 |
| RoPE (sliding attention) | `rope_theta` 10000.0, `rope_type` `default` |

The token identifier ordering is not a transcription error. `eos_token_id` is 1 and
`bos_token_id` is 2 in every variant. Preserve that ordering.

`tie_word_embeddings` is confirmed in the weights, not only in the configuration: neither E2B
nor E4B contains an `lm_head` tensor. The output projection reuses
`model.language_model.embed_tokens.weight`.

## Per-variant text configuration

All values come from the `text_config` block of each `config.json`.

| Field | E2B | E4B | 12B | 31B |
|-------|-----|-----|-----|-----|
| `num_hidden_layers` | 35 | 42 | 48 | 60 |
| `hidden_size` | 1536 | 2560 | 3840 | 5376 |
| `intermediate_size` | 6144 | 10240 | 15360 | 21504 |
| `num_attention_heads` | 8 | 8 | 16 | 32 |
| `num_key_value_heads` | 1 | 2 | 8 | 16 |
| GQA ratio | 8:1 (MQA) | 4:1 | 2:1 | 2:1 |
| `num_global_key_value_heads` | `null` | `null` | 1 | 4 |
| `num_kv_shared_layers` | 20 | 18 | 0 | 0 |
| `hidden_size_per_layer_input` | 256 | 256 | 0 | 0 |
| `vocab_size_per_layer_input` | 262144 | 262144 | 262144 | 262144 |
| `sliding_window` | 512 | 512 | 1024 | 1024 |
| `max_position_embeddings` | 131072 | 131072 | 262144 | 262144 |
| `attention_k_eq_v` | `false` | `false` | `true` | `true` |
| `use_double_wide_mlp` | `true` | `false` | `false` | `false` |
| `use_bidirectional_attention` | `null` | `null` | `"vision"` | `"vision"` |
| `dtype` | `bfloat16` | `bfloat16` | (unset) | `bfloat16` |
| `transformers_version` | 5.5.0.dev0 | 5.5.0.dev0 | 5.10.0.dev0 | 5.5.0.dev0 |

Note that `vocab_size_per_layer_input` is 262144 in all four variants, including 12B and 31B
where `hidden_size_per_layer_input` is 0. The per-layer vocabulary field is populated even
when the feature is disabled; `hidden_size_per_layer_input` is the field that determines
whether per-layer embeddings exist. Read that field, not the vocabulary field.

## Attention

Gemma 4 interleaves two attention types on a fixed schedule given explicitly by the
`layer_types` array. Read the array. Do not compute the schedule from a stride constant: E2B
uses a different period from the other three variants.

| Model | Layers | Sliding | Full | Full-attention indices | Period |
|-------|--------|---------|------|------------------------|--------|
| E2B | 35 | 28 | 7 | 4, 9, 14, 19, 24, 29, 34 | every 5th (4:1) |
| E4B | 42 | 35 | 7 | 5, 11, 17, 23, 29, 35, 41 | every 6th (5:1) |
| 12B | 48 | 40 | 8 | 5, 11, 17, 23, 29, 35, 41, 47 | every 6th (5:1) |
| 31B | 60 | 50 | 10 | 5, 11, 17, 23, 29, 35, 41, 47, 53, 59 | every 6th (5:1) |

In every variant the final layer is a full-attention layer.

The two attention types use **different head dimensions**. Sliding layers use `head_dim`
(256); full-attention layers use `global_head_dim` (512). This is visible directly in the
weight shapes and is not an inference from the configuration.

E4B projection shapes, with counts from the safetensors header:

| Tensor | Sliding (35 layers) | Full (7 layers) |
|--------|---------------------|-----------------|
| `q_proj.weight` | `[2048, 2560]` | `[4096, 2560]` |
| `k_proj.weight` | `[512, 2560]` | `[1024, 2560]` |
| `v_proj.weight` | `[512, 2560]` | `[1024, 2560]` |
| `o_proj.weight` | `[2560, 2048]` | `[2560, 4096]` |
| `q_norm.weight` | `[256]` | `[512]` |
| `k_norm.weight` | `[256]` | `[512]` |

E2B projection shapes, with counts:

| Tensor | Sliding (28 layers) | Full (7 layers) |
|--------|---------------------|-----------------|
| `q_proj.weight` | `[2048, 1536]` | `[4096, 1536]` |
| `k_proj.weight` | `[256, 1536]` | `[512, 1536]` |
| `v_proj.weight` | `[256, 1536]` | `[512, 1536]` |
| `o_proj.weight` | `[1536, 2048]` | `[1536, 4096]` |
| `q_norm.weight` | `[256]` | `[512]` |
| `k_norm.weight` | `[256]` | `[512]` |

The shapes follow `num_attention_heads * head_dim` for sliding layers and
`num_attention_heads * global_head_dim` for full layers, with the key and value projections
scaled by `num_key_value_heads` instead. E2B has a single key-value head, making it a true
multi-query attention model; its `k_proj` output dimension of 256 equals one head at
`head_dim`.

Gemma 4 applies RMS normalization to the query and key projections per head
(`q_norm`, `k_norm`), sized to the head dimension in use for that layer type.

`attention_k_eq_v` is `true` on 12B and 31B and `false` on the E-series. Where it is true, the
key and value share a tensor, roughly halving key-value cache footprint. A cache implementation
validated against 31B and then applied to E4B computes the wrong size and, worse, may read the
wrong values without erroring.

## Rotary position embeddings

Gemma 4 uses **two distinct RoPE regimes within a single model**, selected by layer type. This
is the highest-risk area for a from-scratch implementation, because an incorrect
implementation produces plausible logits rather than a crash.

| Layer type | `rope_type` | `rope_theta` | `partial_rotary_factor` |
|------------|-------------|--------------|-------------------------|
| `full_attention` | `proportional` | 1000000.0 | 0.25 |
| `sliding_attention` | `default` | 10000.0 | (absent) |

Two consequences follow.

First, full-attention layers rotate only a **quarter of each head's dimensions**. With
`global_head_dim` of 512, `partial_rotary_factor` of 0.25 means 128 dimensions receive rotary
treatment and 384 pass through unrotated. Applying rotation to all 512 dimensions is a silent
numerical error.

Second, the two regimes differ in base frequency by two orders of magnitude (1000000.0 against
10000.0). A single global `rope_theta` is incorrect for this architecture regardless of which
value is chosen.

### How `proportional` implements the partial rotation

`_compute_proportional_rope_parameters` in `transformers/src/transformers/modeling_rope_utils.py`
(revision `9451e3ab0b21cc51638dce25c101ab5f2233e91d`, lines 193-266) registers under the key
`"proportional"` in `ROPE_INIT_FUNCTIONS` (line 674). It differs from every other partial-rotary
rope type in one respect, stated in its own docstring at lines 216-219: it "will always return an
encoding that is the size of `head_dim`". The mechanism is a zero-padded frequency vector:

```python
rope_angles = int(rope_proportion * head_dim // 2)
inv_freq_rotated = 1.0 / (
    base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(...) / head_dim)
)
nope_angles = head_dim // 2 - rope_angles
if nope_angles > 0:
    inv_freq = torch.cat((inv_freq_rotated, torch.zeros(nope_angles, ...)), dim=0)
```

For `global_head_dim` of 512 and `partial_rotary_factor` of 0.25, `rope_angles` is 64 and
`nope_angles` is 192. The returned `inv_freq` has 256 entries: 64 live frequencies followed by
192 zeros. A zero frequency yields an angle of 0 radians, so `cos` is 1.0 and `sin` is 0.0, and
the corresponding dimensions pass through unchanged. The exponent divisor stays `head_dim` (512),
not `2 * rope_angles`, so the 64 live frequencies are the first 64 of the full 512-dimension
schedule rather than a rescaled 128-dimension schedule.

The layout is **split-half**, not interleaved. `Gemma4TextRotaryEmbedding.forward`
(`modeling_gemma4.py`, revision `ff886afacfd81e09dcf70d29d5bbf76babf672a5`, lines 1144-1163)
builds `emb = torch.cat((freqs, freqs), dim=-1)`, and `rotate_half` (lines 777-781) splits the
head dimension at its midpoint. Pairing is therefore between dimension `i` and dimension
`i + 256`. Dimensions 0-63 and 256-319 rotate; dimensions 64-255 and 320-511 pass through. That
is 128 rotating dimensions of 512, as `partial_rotary_factor` states, but they are not
contiguous. `apply_rotary_pos_emb` (lines 784-803) applies the full-width expression
`(x * cos) + (rotate_half(x) * sin)` to all 512 dimensions; the pass-through is a property of
the frequencies, not of a narrowed slice.

Sliding-attention layers use `rope_type: default` with `head_dim` of 256 and no
`partial_rotary_factor`, so all 256 dimensions rotate.

## E-series exclusive features

This section is the practical core of the document. Each item below is present in E2B and E4B
and absent from 12B and 31B. An implementation that handles 12B and 31B correctly can still
fail completely on the E-series, and these are the reasons why. They are listed in descending
order of impact.

1. **Per-layer input embeddings.** `hidden_size_per_layer_input` is 256 on the E-series and 0
   on 12B and 31B. Without this mechanism the E-series does not run at all. See
   [Per-layer input embeddings](#per-layer-input-embeddings).
2. **Cross-layer KV cache sharing.** `num_kv_shared_layers` is 20 on E2B and 18 on E4B, and 0
   on 12B and 31B. Ignoring it does not produce wrong output, but it forfeits the memory saving
   that makes these models fit in a single GPU. See
   [Cross-layer KV cache sharing](#cross-layer-kv-cache-sharing).
3. **Separate key and value tensors.** `attention_k_eq_v` is `false` on the E-series and `true`
   on 12B and 31B. An implementation that assumes the shared-tensor path produces incorrect
   attention on the E-series without raising an error.
4. **Absent `num_global_key_value_heads`.** The field is `null` on the E-series, meaning
   full-attention layers use the standard `num_key_value_heads` count. On 12B it is 1 and on
   31B it is 4. Code that requires the field, or that defaults it to 1, is wrong for the
   E-series.
5. **Double-wide MLP on E2B only.** `use_double_wide_mlp` is `true` on E2B alone, and it applies
   to a subset of layers. See [The double-wide MLP](#the-double-wide-mlp).
6. **Halved context and sliding window.** The E-series has `max_position_embeddings` of 131072
   and `sliding_window` of 512, against 262144 and 1024 for 12B and 31B. Both position buffer
   sizing and the sliding attention mask differ.
7. **Distinct layer-type period on E2B.** E2B places a full-attention layer every fifth layer;
   every other variant uses every sixth. A hardcoded stride works for three of four models.
8. **Multi-query attention on E2B.** `num_key_value_heads` is 1, making E2B genuinely MQA
   rather than GQA. Implementations that assume at least two key-value heads, or that derive a
   group size by division without checking for 1, break here.
9. **Full audio tower on the E-series.** E2B and E4B carry a complete `gemma4_audio`
   configuration and audio tower weights. 31B has `audio_config: null`. Audio support is
   per-variant, not a family property.

## Per-layer input embeddings

The E-series carries a second embedding table, indexed by token and by layer, in addition to
the standard input embedding. This is the defining architectural feature of the E-series and
the origin of the `E` prefix, which denotes effective parameters: the per-layer table is
excluded from the advertised parameter count.

Tensors, confirmed from the safetensors headers:

| Tensor | E2B shape | E4B shape | Count |
|--------|-----------|-----------|-------|
| `model.language_model.embed_tokens.weight` | `[262144, 1536]` | `[262144, 2560]` | 1 |
| `model.language_model.embed_tokens_per_layer.weight` | `[262144, 8960]` | `[262144, 10752]` | 1 |
| `model.language_model.per_layer_model_projection.weight` | `[8960, 1536]` | `[10752, 2560]` | 1 |
| `model.language_model.per_layer_projection_norm.weight` | `[256]` | `[256]` | 1 |
| `model.language_model.layers.{L}.per_layer_input_gate.weight` | `[256, 1536]` | `[256, 2560]` | per layer |
| `model.language_model.layers.{L}.per_layer_projection.weight` | `[1536, 256]` | `[2560, 256]` | per layer |
| `model.language_model.layers.{L}.post_per_layer_input_norm.weight` | `[1536]` | `[2560]` | per layer |

The width of `embed_tokens_per_layer` is `num_hidden_layers * hidden_size_per_layer_input`:
35 × 256 = 8960 for E2B, and 42 × 256 = 10752 for E4B. The table is a single tensor that must
be split per layer, not a stack of per-layer tensors.

Each layer holds a gate projecting the model hidden state down to 256 dimensions, a projection
returning 256 dimensions back to the model width, and a normalization applied afterwards. A
model-level projection (`per_layer_model_projection`) and a 256-wide normalization
(`per_layer_projection_norm`) sit outside the layer stack.

The per-layer table is large. On E2B it holds 262144 × 8960 = 2,348,810,240 parameters,
exceeding the entire transformer body. Any memory plan for the E-series that omits it is wrong
by more than a factor of two.

## Cross-layer KV cache sharing

`num_kv_shared_layers` does **not** indicate shared weights. The safetensors headers show that
every layer carries its own `k_proj` and `v_proj`: 42 of each on E4B, 35 of each on E2B, with
no layer omitted. The sharing applies to the key-value cache at inference time, not to the
projection parameters.

Two failure modes follow from misreading the field:

- A loader that expects 18 of E4B's 42 layers to omit key and value projections fails to load
  the weights, because those tensors are present.
- A cache allocator that gives all 42 layers independent key-value storage loads and runs
  correctly but consumes substantially more memory than the architecture requires. On models
  chosen specifically to fit training entirely in GPU memory, this is the difference between
  fitting and not fitting.

The shared region is the **tail** of the layer stack. On E2B the 20 shared layers are indices
15 through 34, which is confirmed by their correlation with the double-wide MLP region
described below. The corresponding E4B region of 18 layers is indices 24 through 41 if the
same tail convention holds; this has not been independently confirmed from the weights, since
E4B's MLP widths are uniform and provide no corroborating signal. Treat the E4B boundary as
unverified.

## The double-wide MLP

E2B sets `use_double_wide_mlp: true`. The flag does not apply uniformly. Of E2B's 35 layers,
15 use the configured `intermediate_size` of 6144 and **20 use 12288**, exactly twice the
configured value.

| Layer range | Count | `gate_proj` / `up_proj` | `down_proj` |
|-------------|-------|-------------------------|-------------|
| 0 through 14 | 15 | `[6144, 1536]` | `[1536, 6144]` |
| 15 through 34 | 20 | `[12288, 1536]` | `[1536, 12288]` |

The 20 double-wide layers are indices 15 through 34, which is precisely E2B's
`num_kv_shared_layers` count of 20 occupying the tail of the stack. The tail layers share
key-value cache and receive a doubled feed-forward width in compensation.

The operative rule: when `use_double_wide_mlp` is true, the final `num_kv_shared_layers`
layers use `2 * intermediate_size`, and the remainder use `intermediate_size`. A loader that
trusts `intermediate_size` for every layer mis-shapes 20 of E2B's 35 layers.

E4B has `use_double_wide_mlp: false` and uniform MLP widths of 10240 across all 42 layers,
despite also having a shared-KV tail. The double-wide compensation is therefore not implied by
KV sharing; it requires the flag.

## Tensor inventory

Complete language-model tensor inventory for E4B, from the safetensors header at revision
`411aa17b`. Counts are exact. Layer indices are collapsed to `{L}`.

| Count | Tensor | Shape |
|-------|--------|-------|
| 1 | `model.language_model.embed_tokens.weight` | `[262144, 2560]` |
| 1 | `model.language_model.embed_tokens_per_layer.weight` | `[262144, 10752]` |
| 1 | `model.language_model.per_layer_model_projection.weight` | `[10752, 2560]` |
| 1 | `model.language_model.per_layer_projection_norm.weight` | `[256]` |
| 1 | `model.language_model.norm.weight` | `[2560]` |
| 42 | `model.language_model.layers.{L}.input_layernorm.weight` | `[2560]` |
| 42 | `model.language_model.layers.{L}.post_attention_layernorm.weight` | `[2560]` |
| 42 | `model.language_model.layers.{L}.pre_feedforward_layernorm.weight` | `[2560]` |
| 42 | `model.language_model.layers.{L}.post_feedforward_layernorm.weight` | `[2560]` |
| 42 | `model.language_model.layers.{L}.post_per_layer_input_norm.weight` | `[2560]` |
| 42 | `model.language_model.layers.{L}.layer_scalar` | `[1]` |
| 42 | `model.language_model.layers.{L}.mlp.gate_proj.weight` | `[10240, 2560]` |
| 42 | `model.language_model.layers.{L}.mlp.up_proj.weight` | `[10240, 2560]` |
| 42 | `model.language_model.layers.{L}.mlp.down_proj.weight` | `[2560, 10240]` |
| 42 | `model.language_model.layers.{L}.per_layer_input_gate.weight` | `[256, 2560]` |
| 42 | `model.language_model.layers.{L}.per_layer_projection.weight` | `[2560, 256]` |
| 35 | `model.language_model.layers.{L}.self_attn.q_proj.weight` | `[2048, 2560]` |
| 7 | `model.language_model.layers.{L}.self_attn.q_proj.weight` | `[4096, 2560]` |
| 35 | `model.language_model.layers.{L}.self_attn.k_proj.weight` | `[512, 2560]` |
| 7 | `model.language_model.layers.{L}.self_attn.k_proj.weight` | `[1024, 2560]` |
| 35 | `model.language_model.layers.{L}.self_attn.v_proj.weight` | `[512, 2560]` |
| 7 | `model.language_model.layers.{L}.self_attn.v_proj.weight` | `[1024, 2560]` |
| 35 | `model.language_model.layers.{L}.self_attn.o_proj.weight` | `[2560, 2048]` |
| 7 | `model.language_model.layers.{L}.self_attn.o_proj.weight` | `[2560, 4096]` |
| 35 | `model.language_model.layers.{L}.self_attn.q_norm.weight` | `[256]` |
| 7 | `model.language_model.layers.{L}.self_attn.q_norm.weight` | `[512]` |
| 35 | `model.language_model.layers.{L}.self_attn.k_norm.weight` | `[256]` |
| 7 | `model.language_model.layers.{L}.self_attn.k_norm.weight` | `[512]` |

E2B holds the same tensor names with 35 layers, `hidden_size` 1536, the split MLP widths
described above, single-head key and value projections, and 28 sliding against 7 full
attention layers.

Structural observations:

- There is **no `lm_head`**. Output logits reuse the tied input embedding.
- Each layer carries **four normalization tensors** for the transformer block
  (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`,
  `post_feedforward_layernorm`), the Gemma sandwich-normalization pattern, plus a fifth
  (`post_per_layer_input_norm`) for the per-layer embedding path.
- Each layer carries a scalar tensor `layer_scalar` of shape `[1]`, which scales the whole
  layer output as its final operation. It has no `config.json` field, and its stored values are
  not 1.0 — E2B spans 0.0186 to 0.879. See [`layer_scalar`](#layer_scalar).
- The multimodal projections `model.embed_audio.embedding_projection.weight`
  (`[2560, 1536]` on E4B) and `model.embed_vision.embedding_projection.weight`
  (`[2560, 768]` on E4B) bridge the towers into the language model width.

## Parameter accounting and memory

Exact counts for the embedding tables, computed from the tensor shapes:

| Quantity | E2B | E4B |
|----------|-----|-----|
| Main embedding | 262144 × 1536 = 402,653,184 | 262144 × 2560 = 671,088,640 |
| Per-layer embedding | 262144 × 8960 = 2,348,810,240 | 262144 × 10752 = 2,818,572,288 |
| Per-layer model projection | 8960 × 1536 = 13,762,560 | 10752 × 2560 = 27,525,120 |

Transformer body totals below are **derived** by summing the tensor shapes in the inventory,
not read from a published figure. They are approximate to within rounding.

| Quantity | E2B | E4B |
|----------|-----|-----|
| Transformer body | approximately 1.88 billion | approximately 4.00 billion |
| Body plus main embedding | approximately 2.28 billion | approximately 4.67 billion |
| Total including per-layer table | approximately 4.65 billion | approximately 7.52 billion |

The middle row corresponds closely to the advertised effective parameter counts of
approximately 2 billion for E2B and 4.5 billion for E4B. The effective count therefore appears
to exclude the per-layer embedding table. This is consistent across both variants but is an
inference from arithmetic, not a documented definition.

For memory planning on a 32GB RTX 5090, the practical consequence is that the per-layer
embedding table dominates. At bfloat16, E2B's per-layer table alone occupies approximately
4.7GB and E4B's approximately 5.6GB, before optimizer state. Continual pretraining plans that
size against the effective parameter count underestimate resident weight memory by roughly a
factor of two.

## Tokenizer and special tokens

The tokenizer class is `GemmaTokenizer` with a `tokenizers` backend, wrapped by
`Gemma4Processor`.

| Token | String | Identifier |
|-------|--------|------------|
| Beginning of sequence | `<bos>` | 2 |
| End of sequence | `<eos>` | 1 |
| Padding | `<pad>` | 0 |
| Unknown | `<unk>` | (not stated in `tokenizer_config.json`) |
| Mask | `<mask>` | (not stated) |

Multimodal and control identifiers, from the top level of `config.json`:

| Token | Identifier |
|-------|------------|
| `boi_token_id` (begin image) | 255999 |
| `boa_token_id` (begin audio) | 256000 |
| `image_token_id` | 258880 |
| `audio_token_id` | 258881 |
| `eoi_token_id` (end image) | 258882 |
| `eoa_token_id` (end audio) | 258883 |
| `video_token_id` | 258884 |

These identifiers occupy positions inside the 262144-entry vocabulary. **Do not trim the
vocabulary for text-only work.** The embedding table is sized to the full vocabulary and the
weights are tied to it; removing multimodal entries breaks both the output projection and
tokenizer round-tripping, and converts later multimodal support from an addition into a
rewrite.

The tokenizer defines asymmetric paired markers where the bar denotes the open or close side:

| Role | Open | Close |
|------|------|-------|
| Turn | `<\|turn>` | `<turn\|>` |
| Channel | `<\|channel>` | `<channel\|>` |
| Tool call | `<\|tool_call>` | `<tool_call\|>` |
| Tool | `<\|tool>` | `<tool\|>` |
| Tool response | `<\|tool_response>` | `<tool_response\|>` |

A reasoning marker `<|think|>` and an escape token `<|"|>` are also defined. `padding_side` is
`left`. `model_max_length` is set to the sentinel value 1000000000000000019884624838656 and
carries no information; use `max_position_embeddings` instead.

Default generation parameters, from E4B `generation_config.json`:

```json
{
  "bos_token_id": 2,
  "do_sample": true,
  "eos_token_id": 1,
  "pad_token_id": 0,
  "temperature": 1.0,
  "top_k": 64,
  "top_p": 0.95
}
```

## Modality towers

Documented for completeness and scoped out of text-only work. Recording the shapes here means
that adding a modality later is a matter of reading this section rather than re-deriving it.

Audio tower, E-series (`model_type: gemma4_audio`), absent on 31B and restructured on 12B:

| Field | Value |
|-------|-------|
| `hidden_size` | 1024 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 8 |
| `hidden_act` | `silu` |
| `conv_kernel_size` | 5 |
| `attention_chunk_size` | 12 |
| `attention_context_left` | 13 |
| `attention_context_right` | 0 |
| `attention_logit_cap` | 50.0 |
| `output_proj_dims` | 1536 |
| `subsampling_conv_channels` | `[128, 32]` |
| `residual_weight` | 0.5 |

The audio tower is a conformer: each layer holds two feed-forward blocks
(`feed_forward1`, `feed_forward2`), a depthwise separable convolution block (`lconv1d`), and
self-attention with a `relative_k_proj` and a `per_dim_scale` tensor of shape `[128]`. Many
audio and vision linear layers carry `input_min`, `input_max`, `output_min`, and `output_max`
scalar tensors alongside `linear.weight`, consistent with `use_clipped_linears: true`. Any
generic safetensors loader must tolerate these zero-dimensional companions.

Vision tower:

| Field | E-series | 31B |
|-------|----------|-----|
| `hidden_size` | 768 | 1152 |
| `num_hidden_layers` | 16 | 27 |
| `num_attention_heads` | 12 | 16 |
| `head_dim` | 64 | 72 |
| `intermediate_size` | 3072 | 4304 |
| `patch_size` | 16 | 16 |
| `pooling_kernel_size` | 3 | 3 |
| `position_embedding_size` | 10240 | 10240 |
| `rope_theta` | 100.0 | 100.0 |
| `standardize` | `false` | `true` |
| `use_clipped_linears` | `true` | `false` |

`vision_soft_tokens_per_image` is 280 in every variant that defines it. The 12B vision
configuration uses the different `gemma4_unified_vision` type with `mm_embed_dim` 3840,
`mm_posemb_size` 1120, and `model_patch_size` 48.

Note that the vision tower's `num_hidden_layers` of 16 and `hidden_size` of 768 are easily
mistaken for language model values when reading `config.json` casually or through a search
result. They are not.

## Traps and discrepancies

A checklist of points where the obvious reading is wrong. Each is independently verifiable
against the pinned revisions.

1. **Advertised context does not match configuration on the E-series.** Release material cites
   up to 256K tokens. E2B and E4B declare `max_position_embeddings: 131072`. Only 12B and 31B
   declare 262144. Sizing E-series buffers against 256K overshoots by a factor of two, and
   assuming 256K works at inference is unverified.
2. **`num_kv_shared_layers` shares cache, not weights.** All layers carry their own key and
   value projections. See [Cross-layer KV cache sharing](#cross-layer-kv-cache-sharing).
3. **`intermediate_size` is not per-layer truth on E2B.** Twenty of 35 layers are double-wide.
4. **Two RoPE regimes coexist in one model**, differing in base frequency by 100 times.
5. **`partial_rotary_factor: 0.25` applies to full-attention layers only**, rotating 128 of 512
   dimensions.
6. **`head_dim` is not the head dimension of every layer.** Full-attention layers use
   `global_head_dim`.
7. **12B is a different class.** Matching on `model_type == "gemma4_text"` excludes it.
8. **`eos_token_id` is 1 and `bos_token_id` is 2.** The ordering looks transposed and is not.
9. **`vocab_size_per_layer_input` is populated even where per-layer embeddings are disabled.**
   Gate on `hidden_size_per_layer_input` instead.
10. **There is no shard index file.** Loaders that require `model.safetensors.index.json` fail.
11. **All configurations target unreleased `transformers` development versions**, and the two
    versions in play differ (5.5.0.dev0 and 5.10.0.dev0).
12. **Zero-dimensional clipping tensors accompany many tower linear layers.** A loader that
    assumes every tensor has a non-empty shape fails on them.
13. **E2B is multi-query, not grouped-query.** `num_key_value_heads` is 1.
14. **Audio support is per-variant.** 31B sets `audio_config: null`.

## Resolved from upstream source

The items below were open and are now settled against upstream source. Every claim cites a file,
a commit SHA, and a line range or function name. The revisions read are:

| Source | Path | Commit |
|--------|------|--------|
| `huggingface/transformers` | `src/transformers/models/gemma4/modeling_gemma4.py` | `ff886afacfd81e09dcf70d29d5bbf76babf672a5` |
| `huggingface/transformers` | `src/transformers/modeling_rope_utils.py` | `9451e3ab0b21cc51638dce25c101ab5f2233e91d` |
| `huggingface/transformers` | `src/transformers/models/gemma4/configuration_gemma4.py` | `3e9d3e50e71442a3173bdf01cd45ba5833533efe` |
| `ggml-org/llama.cpp` | `conversion/gemma.py` | `f4043fec0103872bf4339f6fa18d8b17824d5b6a` |
| `ggml-org/llama.cpp` | `gguf-py/gguf/constants.py`, `gguf-py/gguf/tensor_mapping.py` | `07132750825a4f2d27a547cd9cdde1c6f6001885` |

### The `proportional` rope type

Resolved. See [How `proportional` implements the partial rotation](#how-proportional-implements-the-partial-rotation).
The formula lives in `_compute_proportional_rope_parameters`
(`modeling_rope_utils.py`, lines 193-266): the inverse-frequency vector is computed at full
`head_dim` width and its tail is zero-filled, so the unrotated dimensions receive an identity
rotation rather than being excluded from the operation.

### `layer_scalar`

Resolved. `Gemma4TextDecoderLayer` declares `self.layer_scalar = nn.Buffer(torch.ones(1))`
(`modeling_gemma4.py`, line 1371) and applies it as the **last operation of the decoder layer**,
after the attention residual, after the feed-forward residual, and after the per-layer-embedding
residual:

```python
hidden_states *= self.layer_scalar
return hidden_states
```

That is `modeling_gemma4.py`, line 1444. It multiplies the entire layer output, including the
residual carried through the layer, and is not confined to the attention or feed-forward branch.
Initialization is `init.ones_(module.layer_scalar)` (line 1523), so an *untrained* value is 1.0.
It is a buffer rather than a parameter and has no `config.json` field, which is why it appears
in the weights and not in the configuration.

#### The stored values are not 1.0

Read from the released checkpoints and recorded in the `layer_scalar stored values` sections of
`docs/models/gemma4-safetensors.txt`. This is a measurement, not an inference from the source:

| | E2B (35 layers) | E4B (42 layers) |
|---|---|---|
| Minimum | 0.0185546875 (layer 0) | 0.431640625 and below |
| Maximum | 0.87890625 (layer 30) | 0.703125 and above |
| Mean | 0.5366 | — |

Every layer attenuates. The profile is structured rather than noisy: layer 0 is scaled by
0.0186, roughly a 54x reduction; layers 13 and 14 sit near 0.05-0.07; layers 26 through 32
approach pass-through at 0.83-0.88; and the final layer returns to 0.161.

The initialization is therefore misleading as a guide to the released weights. `nn.Buffer` is
not an `nn.Parameter`, so no optimizer updates it and reading line 1523 alone suggests the
multiply is inert. It is not: the values were set during Google's training and saved into the
checkpoint. **Any reimplementation that omits this multiplication diverges from the reference
model**, and by a factor of about 54 in the first layer.

The reason this was worth measuring rather than assuming, in three parts, because each looked
like a hazard and each resolves differently:

- **Activation capture (interp) is unaffected.** The harvest hook is a forward hook on the
  decoder layer module, and the multiply is that module's last operation, so the captured
  tensor is the post-scale output — the values the next layer actually receives. See
  [docs/interp/gemma-4-interp.md](../interp/gemma-4-interp.md).
- **GGUF serving is unaffected.** The llama.cpp converter does not drop the tensor; it renames
  it, and the pinned GGUF carries `blk.{n}.layer_output_scale.weight [1] F32` for every layer
  (`docs/models/gemma4-gguf.txt`).
- **Fine-tuning is unaffected.** A buffer is absent from `named_parameters()`, which is what
  the freeze filters iterate, so no freeze pattern can expose it to an optimizer. See
  [docs/training/gemma-4-training.md](../training/gemma-4-training.md).

What remains is the rust harness, which does not read the tensor under either name — a gap
that matters only when Gemma 4 support is implemented there, and which is recorded in
[docs/rust-harness/gemma-4-ingestion.md](../rust-harness/gemma-4-ingestion.md).

### The per-layer embedding data path

Resolved. The path has two stages.

The first stage runs once per forward, before the decoder stack, in `Gemma4TextModel.forward`
(`modeling_gemma4.py`, lines 1661-1664):

1. `get_per_layer_inputs` (lines 1724-1766) looks `input_ids` up in `embed_tokens_per_layer`, a
   scaled embedding whose `embed_scale` is `hidden_size_per_layer_input ** 0.5` (line 1608), and
   reshapes the packed result from `[batch, seq, num_hidden_layers * 256]` to
   `[batch, seq, num_hidden_layers, 256]`.
2. `project_per_layer_inputs` (lines 1768-1801) applies `per_layer_model_projection`, multiplies
   by `per_layer_model_projection_scale`, which is `hidden_size ** -0.5` (line 1616), reshapes to
   the same four-dimensional form, and normalizes with `per_layer_projection_norm`, an RMSNorm
   over 256 dimensions (line 1617). It then returns
   `(per_layer_projection + per_layer_inputs) * self.per_layer_input_scale`, where
   `per_layer_input_scale` is `2.0 ** -0.5` (line 1610). When `per_layer_inputs` is `None`, as for
   multimodal inputs without `input_ids`, it returns the projection alone (lines 1798-1799).

Layer `i` receives the slice `per_layer_inputs[:, :, i, :]` (line 1703).

The second stage runs inside each decoder layer. It joins the residual stream **after the
feed-forward block**, as a third residual update, not before or inside attention
(`modeling_gemma4.py`, lines 1435-1442):

```python
if self.hidden_size_per_layer_input:
    residual = hidden_states
    hidden_states = self.per_layer_input_gate(hidden_states)
    hidden_states = self.act_fn(hidden_states)
    hidden_states = hidden_states * per_layer_input
    hidden_states = self.per_layer_projection(hidden_states)
    hidden_states = self.post_per_layer_input_norm(hidden_states)
    hidden_states = residual + hidden_states
```

The gate is **multiplicative**, and it gates in the 256-dimensional per-layer space rather than
in the 1536-dimensional model space. `per_layer_input_gate` projects `hidden_size` down to
`hidden_size_per_layer_input`, the activation `config.hidden_activation` is applied to that
projection, and the per-layer embedding multiplies the activated gate elementwise.
`per_layer_projection` maps back up to `hidden_size`, `post_per_layer_input_norm` normalizes, and
the result is added to the residual. The full layer order is therefore: attention with residual,
feed-forward with residual, per-layer-embedding block with residual, then `layer_scalar`.

### The E4B shared-KV layer boundary

Resolved as to the boundary formula, and revised as to the donor.

`Gemma4TextAttention.__init__` (`modeling_gemma4.py`, lines 1186-1191) computes:

```python
first_kv_shared_layer_idx = self.config.num_hidden_layers - getattr(self.config, "num_kv_shared_layers", 0)
self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
prev_layers = config.layer_types[:first_kv_shared_layer_idx]
self.store_full_length_kv = not self.is_kv_shared_layer and layer_idx == len(prev_layers) - 1 - prev_layers[
    ::-1
].index(config.layer_types[layer_idx])
```

The boundary is `num_hidden_layers - num_kv_shared_layers`, confirming the tail convention: 15 on
E2B (35 layers, 20 shared) and 24 on E4B (42 layers, 18 shared). `Gemma4TextMLP` uses the same
expression for the double-wide feed-forward region (lines 1068-1070), which is why the E2B
feed-forward widths corroborate the boundary; the correlation is structural, not coincidental,
and it holds on E4B for the same reason. `conversion/gemma.py` line 682 repeats the formula as
`first_kv_shared_layer_idx = self.block_count - num_kv_shared_layers`.

There are **two donor layers, not one**. `store_full_length_kv` selects, for each layer type
independently, the last non-shared layer of that type, and the cache is keyed by layer type:
`shared_kv_states[self.layer_type] = key_states, value_states` (line 1259), read back as
`key_states, value_states = shared_kv_states[self.layer_type]` (line 1241). A shared
`full_attention` layer reuses the last non-shared `full_attention` layer's keys and values; a
shared `sliding_attention` layer reuses the last non-shared `sliding_attention` layer's. Any
implementation that designates a single donor layer for all shared layers is wrong for one of the
two regimes.

The comment at lines 1237-1239 records why the donor state is used rather than the cache: a
sliding layer's own cache does not retain the full sequence once the window is passed, so
`shared_kv_states` is consulted even when `past_key_values` is present.

### `attention_k_eq_v` semantics

Resolved, and narrower than the name suggests. `Gemma4TextAttention.__init__` sets
`self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding`
(`modeling_gemma4.py`, line 1179), so the sharing applies to **full-attention layers only**. Where
it applies, `v_proj` is not constructed at all (lines 1206-1212) and the forward pass sets
`value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states`
(line 1247). The value states are the key states before `k_norm` and before rotary embedding;
`v_norm` is then applied to them separately (line 1253), and `v_norm` is constructed with
`with_scale=False` (line 1201). Key and value therefore share the projection output, not the
post-norm post-rotary tensor.

### The shared-KV weight conflict

The modeling code and the released weights disagree about whether shared-KV layers carry key
and value projections at all. Both sides are quoted below because the disagreement is real and
neither source is obviously the stale one.

**What the code says.** `Gemma4TextAttention.__init__` guards four modules behind
`if not self.is_kv_shared_layer:` — `k_norm`, `v_norm`, `k_proj`, and `v_proj`
(`docs/models/gemma4-upstream.txt` lines 1213-1227, upstream 1198-1212), under the comment
"Layers sharing kv states don't need any weight matrices". For E2B that means 15 of 35 layers
construct them.

**What the weights say.** All 35 E2B layers carry `k_proj`, `v_proj`, and `k_norm`. The
safetensors headers record 28 at head width 256 and 7 at 512
(`docs/models/gemma4-safetensors.txt` lines 414 and 529); the GGUF inventory lists
`attn_k.weight`, `attn_v.weight`, and `attn_k_norm.weight` for the same 35 blocks
(`docs/models/gemma4-gguf.txt` lines 613 and 625). Note `k_norm` is inside the same guard, so
the surplus is four tensor families and not two.

**The converter is not the cause.** `conversion/gemma.py` computes `first_kv_shared_layer_idx`
and uses it for one purpose only — widening the feed-forward layers when `use_double_wide_mlp`
is set (`docs/models/gemma4-upstream.txt` lines 4851-4854, upstream 682-686). It never filters
key or value tensors. The GGUF therefore mirrors the safetensors, and the safetensors are where
the surplus originates.

**The file contradicts itself, which is the sharpest evidence available.** E2B's published FFN
widths are not uniform: 20 layers at 12288 and 15 at 6144
(`docs/models/gemma4-gguf.txt` lines 617-622). That split is `num_kv_shared_layers = 20` applied
by the converter. So the same boundary value is simultaneously honored — the trailing 20 layers
are double-wide — and contradicted, because those same 20 layers also carry the key and value
projections the code says a shared layer does not have. Whatever produced these files acted on
`num_kv_shared_layers` for the MLP and not for attention.

Two readings remain, and they are not equally cheap to be wrong about:

| Reading | Implication | Cost if wrong |
|---------|-------------|---------------|
| The code is authoritative; the tensors are residue | Shared layers read a donor's K/V; the surplus tensors are dead | 18 of E4B's 42 layers attend over the wrong keys — fluent, wrong output, no error |
| The weights are authoritative; the code is ahead of them | Every layer computes its own K/V, and `num_kv_shared_layers` describes cache reuse only | Wasted memory, correct output |

The `rust-harness` CPU backend implements the first reading. The choice is marked in the loader
and pinned by `a_shared_layers_own_key_and_value_weights_are_deliberately_unread` in
`rust-harness/runtime/src/model/cpu.rs`, which is a tripwire rather than evidence: it asserts
the tensors stay unread, so it passes by construction and fails only if someone changes the
loader without reading this section.

**Resolving it requires the host.** Nothing in this repository distinguishes the two readings —
both load, both produce fluent text, and the transcripts record the conflict rather than
settling it. The procedure is RH.1a in
[docs/verification/rust-harness.md](../verification/rust-harness.md), which runs the cheap
weight-statistics test before the definitive logit comparison.

## Per-tier support

What each tier does with a real checkpoint today. The target is E2B and E4B; 12B and 31B are
out of scope until the E-series pair is settled.

The distinction that matters here is between reading the geometry and running the model. All
three tiers read it generically — no layer count, width, or stride is hardcoded in any of them,
and the schedule comes from the `layer_types` array rather than a stride constant. Loading a
checkpoint and producing logits is a separate question, and for the CPU backend every
architectural feature the two E-series variants declare is now implemented.

| Tier | E2B | E4B | What decides it |
|------|-----|-----|-----------------|
| `rust-harness` CPU backend | Yes (unverified against real weights) | Yes (unverified against real weights) | Every declared feature implemented; correctness checked against synthetic fixtures only |
| `rust-harness` GGUF metadata reader | Yes | Yes | Every value read by key; schedule decoded from the bool array, E2B stride 5 and E4B 6 |
| `training` | Yes | Yes | `spec_from_hf_config` maps any config carrying the field names; reference recipes exist for both |
| `interp` | Yes | Yes (no recipe) | No geometry constants at all; hook path and width resolve from the loaded config |

Per-layer embeddings, per-layer feed-forward widths, per-layer head dimensions, dual RoPE, the
per-layer output scale, final logit softcapping, and cross-layer KV sharing were all refusals
and are now implemented. Sharing was the last of them, and the order mattered: the per-layer
embedding refusal was the binding one on a real file, so implementing it first left the
shared-KV metadata key as the only thing standing between a released checkpoint and a load
that would have given all 42 layers independent caches. That key was briefly a refusal in its
own right for exactly that reason, and is now read.

Two caveats on the "yes" in the table above, both material:

- **No real weights have been run through it.** Every feature is verified against synthetic
  fixtures and, where the arithmetic allows, an independent recomputation. Two implementations
  written from the same documents agree on a shared misreading; only Google's actual
  checkpoint breaks that tie, and that run is workstation-only.
- **The shared-KV weight layout follows the modeling code, which the checkpoints contradict.**
  Shared layers' own `attn_k` and `attn_v` are left unread — see the first open question
  below. If that reading is wrong, the result is fluent and wrong rather than a load error,
  which makes it the highest-risk assumption in the backend.

The tensor signature for sharing — a layer with `attn_q` but no `attn_k`/`attn_v` — is no
longer a refusal, because it is now the legitimate shape of a shared layer. It survives
inverted: a layer that *owns* its cache and lacks those tensors has nothing to attend over, and
that is still refused.

`training` and `interp` need no such work. Their E-series gap is coverage rather than
capability: every interp recipe under `training/recipes/interp/` names an E2B subject, so E4B
is supported but unexercised there.

## Open questions

Recorded rather than guessed. Resolve these against the `transformers` implementation at the
pinned revision before relying on any assumption about them.

- **Whether the released E-series checkpoints match the shared-KV weight layout of the modeling
  code.** See [The shared-KV weight conflict](#the-shared-kv-weight-conflict) below, which states
  the evidence on both sides and the host procedure that decides it. This is the only open
  question the CPU backend has already had to answer in order to run, so it is the one whose
  resolution changes code rather than prose.
- **Tensor-level data for 12B and 31B.** Not collected. All statements about those two models
  derive from `config.json`.
- **`use_bidirectional_attention: "vision"`.** Present on 12B and 31B, `null` on the E-series.
  `modeling_gemma4.py` line 1183 uses only the comparison
  `self.is_causal = config.use_bidirectional_attention != "all"`, so the `"vision"` value takes
  the same branch as `null` for that flag. Whether it changes the text path through the mask
  construction was not traced and remains undetermined.
The numeric value of `layer_scalar` was an open question here and is no longer: the transcript
records the stored values, and they are not 1.0 — E2B spans 0.0186 to 0.879. The derivation
moved to [`layer_scalar`](#layer_scalar).

## Reproducing this document

Every value above regenerates from the pinned revisions with no GPU and no model download
beyond a few hundred kilobytes. These run in a contained agent sandbox as well as on the
workstation: `huggingface.co` and `raw.githubusercontent.com` are both on the sandbox proxy
allowlist. Only the header transcripts need the workstation, because they read `/mnt/ai-data`.

The upstream sources behind [Resolved from upstream source](#resolved-from-upstream-source)
are fetched at their pinned SHAs by one script, and the result is committed:

```bash
$ ./scripts/models/upstream_source_transcript.sh --out docs/models/gemma4-upstream.txt
```

`tests/models/test_gemma4_facts_contract.py` asserts the line citations in that section
against the committed transcript, so a revision bump that moves a line fails a test rather
than leaving a stale citation for a reader to trip over.

Configuration for one variant:

```bash
curl -sL https://huggingface.co/google/gemma-4-E4B/raw/main/config.json
```

Pinned revision for each variant:

```bash
for m in gemma-4-E2B gemma-4-E4B gemma-4-12B gemma-4-31B; do
  printf '%s ' "$m"
  curl -sI "https://huggingface.co/google/$m/resolve/main/config.json" | grep -i '^x-repo-commit'
done
```

Tensor names and shapes, without downloading the weights. The safetensors format stores an
eight-byte little-endian header length followed by a JSON header, so a range request retrieves
the full tensor map from a file of several gigabytes:

```bash
url=https://huggingface.co/google/gemma-4-E4B/resolve/main/model.safetensors
n=$(curl -sL -r 0-7 "$url" | python3 -c "import sys;print(int.from_bytes(sys.stdin.buffer.read(8),'little'))")
curl -sL -r 8-$((n+7)) "$url" > e4.json
```

Tensor inventory with layer indices collapsed and counts:

```bash
python3 -c "import json;d=json.load(open('e4.json'));[print(k,d[k].get('shape')) for k in d]" > e4.txt
grep language_model e4.txt | sed 's/\.[0-9]*\./.{L}./' | sort | uniq -c
```

Keep each command short. A terminal that hard-wraps long lines corrupts multi-line Python
passed with `-c`.
