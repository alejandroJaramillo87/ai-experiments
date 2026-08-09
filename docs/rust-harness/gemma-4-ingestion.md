# Gemma 4 ingestion reference

What the `rust-harness` runtime requires to ingest a Gemma 4 checkpoint, what the published
checkpoints actually contain, and which architecture features the backend checks by name.

This document exists because the runtime reads GGUF and the model is published as safetensors
with a Hugging Face `config.json`. The two use different key names, different value types, and
in one case different semantics for the same key. A reader who knows the Hugging Face
configuration still cannot predict what the loader will do.

Scope: E2B and E4B, the two variants present on this workstation. The 12B and 31B variants have
no code path here. The general family reference is `/docs/models/gemma-4.md`.

## Table of contents

- [Provenance](#provenance)
- [The GGUF contract](#the-gguf-contract)
- [Hugging Face to GGUF translation](#hugging-face-to-gguf-translation)
- [What the backend checks](#what-the-backend-checks)
- [The layer schedule is expressible](#the-layer-schedule-is-expressible)
- [Chat markers](#chat-markers)
- [Status of the local GGUF files](#status-of-the-local-gguf-files)
- [Open questions](#open-questions)
- [Reproducing this document](#reproducing-this-document)

## Provenance

| Model | Repository | Revision (`x-repo-commit`) | `config.json` blob |
|-------|------------|----------------------------|--------------------|
| E2B | `google/gemma-4-E2B` | `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` | `5f0662eb398ede0ab1515c4760e13bcd6bf13f2f` |
| E4B | `google/gemma-4-E4B` | `411aa17b749aa952df1359d2dcea73917a544d9a` | `080fd8d51bb6e846df168157c9d57b928a2a2687` |

The blob values are git blob hashes as published by Hugging Face. They are not comparable to a
SHA-256 taken over the file bytes.

Configuration and tensor facts derive from the safetensors headers of the local copies under
`/mnt/ai-data/models/hf/`. GGUF facts derive from the six converted files under
`/mnt/ai-data/models/gguf/`. Both were transcribed by the scripts named in
[Reproducing this document](#reproducing-this-document).

## The GGUF contract

`ModelSpec::from_gguf` (`/rust-harness/runtime/src/model/cpu.rs:358`) requires eleven
architecture keys, all prefixed with the value of `general.architecture`, which must begin with
`gemma`. Every one is mandatory: a missing key or a wrong value type fails the load.

| Key (after the `{arch}.` prefix) | Required type | Becomes |
|----------------------------------|---------------|---------|
| `embedding_length` | uint32 | `Config::embed_dim` |
| `block_count` | uint32 | `Config::n_layers` |
| `attention.head_count` | uint32 | `Config::n_heads` |
| `attention.head_count_kv` | uint32 | `Config::n_kv_heads` |
| `context_length` | uint32 | `Config::context_length` |
| `attention.sliding_window` | uint32 | `Config::sliding_window` |
| `attention.sliding_window_pattern` | uint32 or bool array | `Config::sliding_window_pattern` |
| `attention.layer_norm_rms_epsilon` | float32 | `Config::rms_eps` |
| `attention.key_length` | uint32 | `LayerSpec::head_dim`, global regime |
| `rope.freq_base` | float32 | `LayerSpec::rope_freq_base`, global regime |
| `feed_forward_length` | uint32 or int32 array | `LayerSpec::ffn_dim` |

Six further architecture keys are optional. Absent means the checkpoint does not have the
feature, which is a fact about the file rather than a defaulted guess; a present key of the
wrong type is still an error.

| Key (after the `{arch}.` prefix) | Required type | Absent means |
|----------------------------------|---------------|--------------|
| `attention.key_length_swa` | uint32 | one head width for every layer |
| `rope.freq_base_swa` | float32 | one RoPE regime |
| `attention.shared_kv_layers` | uint32 | every layer owns its cache |
| `final_logit_softcapping` | float32 | the uncapped forward pass |
| `embedding_length_per_layer_input` | uint32 | no per-layer input embeddings |

Two more keys under the `rope.` prefix, `dimension_count` and `dimension_count_swa`, are read
but supply no value. Each is checked against the head width of the regime it governs, and a
narrower value fails the load: a rotation narrower than the head is partial rotary embedding
applied by slicing, which this backend does not implement. Gemma 4 expresses its partial
rotation through `rope_freqs.weight` at full width instead. Any other key under that prefix is
refused, because it configures a RoPE scaling scheme the forward pass would ignore.

Two further keys are read outside `Config`: `general.architecture`, and
`tokenizer.ggml.tokens`, whose length supplies `vocab_size` (there is no `vocab_size` key).

`Config` still derives `Copy` and still holds only scalars, but it no longer holds everything.
It was once the whole configuration, and because a scalar field cannot say "512 on this layer
and 256 on that one", every axis Gemma 4 varies per layer had to be either flattened to a
uniform value or refused. That constraint shaped most of this document, and it is gone:
`ModelSpec` pairs the `Config` scalars with one `LayerSpec` per layer, carrying head width,
feed-forward width, RoPE frequency base, and KV donor. `Config` keeps the quantities that are
genuine model-wide invariants; anything per-layer has to name the layer it means.

`attention.value_length` is never read. The forward pass computes
`kv_dim = n_kv_heads * head_dim` and uses it for both keys and values (`cpu.rs:819`), which
assumes the key and value head dimensions are equal. That assumption holds for Gemma 4 — the
published files set `value_length` equal to `key_length` — but the code does not check it.

### Parser limits

The reader accepts GGUF version 3 only. Beyond the config layer, two limits reject files
outright:

| Limit | Accepted | Site |
|-------|----------|------|
| Tensor storage types | F32 (0), F16 (1), Q8_0 (8), BF16 (30) | `/rust-harness/runtime/src/tensor.rs:20-23` |
| Metadata array element types | int32 (5), float32 (6), bool (7), string (8) | `/rust-harness/runtime/src/gguf.rs` (`Reader::array`) |
| Metadata scalar types | uint32, int32, uint64, float32, bool, string | `/rust-harness/runtime/src/gguf.rs` (`parse`) |

BF16 is the storage type the published Gemma 4 files use and the type the training tier
exports, so it is the reference format rather than a tolerated one. Widening it to f32 is a
shift, not a rounding: the checkpoint's own values reach the forward pass unchanged.

Note that uint32 arrays are still refused even though uint32 scalars are accepted.

### Tensor names the forward pass consumes

Thirteen per-layer tensors (`LAYER_TENSORS`, `cpu.rs:201`), two top-level
(`TOP_LEVEL_TENSORS`, `:220`), and one tolerated as inert (`INERT_TENSORS`, `:225`):

```
blk.{L}.attn_norm.weight        blk.{L}.ffn_norm.weight
blk.{L}.attn_q.weight           blk.{L}.ffn_gate.weight
blk.{L}.attn_k.weight           blk.{L}.ffn_up.weight
blk.{L}.attn_v.weight           blk.{L}.ffn_down.weight
blk.{L}.attn_q_norm.weight      blk.{L}.post_attention_norm.weight
blk.{L}.attn_k_norm.weight      blk.{L}.post_ffw_norm.weight
blk.{L}.attn_output.weight

token_embd.weight               output_norm.weight        (top level)
rope_freqs.weight                                         (inert, tolerated)
```

There is no `output.weight`. The language-model head is tied to `token_embd.weight`, following
the llama.cpp Gemma convention, which matches the published weights: neither E2B nor E4B
contains an `lm_head` tensor.

## Hugging Face to GGUF translation

The same quantity carries a different name in each format. This table is the bridge.

| Hugging Face `text_config` field | GGUF key | E2B | E4B |
|----------------------------------|----------|-----|-----|
| `hidden_size` | `gemma4.embedding_length` | 1536 | 2560 |
| `num_hidden_layers` | `gemma4.block_count` | 35 | 42 |
| `num_attention_heads` | `gemma4.attention.head_count` | 8 | 8 |
| `num_key_value_heads` | `gemma4.attention.head_count_kv` | 1 | 2 |
| `global_head_dim` | `gemma4.attention.key_length` | 512 | 512 |
| `head_dim` | `gemma4.attention.key_length_swa` | 256 | 256 |
| `intermediate_size` | `gemma4.feed_forward_length` | per-layer array | 10240 |
| `max_position_embeddings` | `gemma4.context_length` | 131072 | 131072 |
| `sliding_window` | `gemma4.attention.sliding_window` | 512 | 512 |
| `layer_types` | `gemma4.attention.sliding_window_pattern` | bool array, n=35 | bool array, n=42 |
| `num_kv_shared_layers` | `gemma4.attention.shared_kv_layers` | 20 | 18 |
| `hidden_size_per_layer_input` | `gemma4.embedding_length_per_layer_input` | 256 | 256 |
| `rms_norm_eps` | `gemma4.attention.layer_norm_rms_epsilon` | 1e-06 | 1e-06 |
| `final_logit_softcapping` | `gemma4.final_logit_softcapping` | 30.0 | 30.0 |
| `rope_parameters.full_attention.rope_theta` | `gemma4.rope.freq_base` | 1000000.0 | 1000000.0 |
| `rope_parameters.sliding_attention.rope_theta` | `gemma4.rope.freq_base_swa` | 10000.0 | 10000.0 |

Three properties of this mapping are worth stating plainly, because each one breaks a
reasonable assumption.

**The converter expresses the two attention regimes with an `_swa` key suffix.** Where the
Hugging Face configuration nests two objects under `rope_parameters`, GGUF publishes paired
flat keys: `key_length` and `key_length_swa`, `value_length` and `value_length_swa`,
`rope.dimension_count` and `rope.dimension_count_swa`, `rope.freq_base` and
`rope.freq_base_swa`. The unsuffixed key is the full-attention value; the `_swa` key is the
sliding-window value.

**`attention.key_length` is the global head dimension, not `head_dim`.** The runtime reads that
key into `Config::head_dim`, so on these files `Config::head_dim` is 512, and the 256-wide
sliding layers have no representation at all.

**`feed_forward_length` changes type between variants.** On E4B it is a uint32 scalar of 10240.
On E2B it is an **int32 array of 35 entries**, carrying 6144 for layers 0 through 14 and 12288
for layers 15 through 34. A loader that reads this key as a scalar works on E4B and fails on
E2B.

**`attention.sliding_window_pattern` is a period, re-encoded.** The runtime interprets the key
as a stride (`Config::layer_kind`, `cpu.rs:188`), while the converter writes a bool array of
length `block_count`, with `true` marking a sliding layer and `false` a full-attention layer.
The encodings differ; the schedules do not. Every shipped array is periodic, and its period is
exactly the stride the runtime wants:

| File | Array | Equivalent stride |
|------|-------|-------------------|
| E2B | `[t,t,t,t,f]` x 7, n=35 | 5 |
| E4B | `[t,t,t,t,t,f]` x 7, n=42 | 6 |
| mtp | `[t,t,t,f]`, n=4 | 4 |

`Config::from_gguf` therefore derives the stride from the array and verifies that the derived
period reproduces every entry, refusing a non-periodic array rather than reinterpreting it. An
earlier revision of this document asserted the key "is not a period"; that was wrong.

### Tensor name translation

| Safetensors | GGUF |
|-------------|------|
| `model.language_model.embed_tokens.weight` | `token_embd.weight` |
| `model.language_model.norm.weight` | `output_norm.weight` |
| `model.language_model.layers.{L}.input_layernorm.weight` | `blk.{L}.attn_norm.weight` |
| `model.language_model.layers.{L}.post_attention_layernorm.weight` | `blk.{L}.post_attention_norm.weight` |
| `model.language_model.layers.{L}.pre_feedforward_layernorm.weight` | `blk.{L}.ffn_norm.weight` |
| `model.language_model.layers.{L}.post_feedforward_layernorm.weight` | `blk.{L}.post_ffw_norm.weight` |
| `model.language_model.layers.{L}.post_per_layer_input_norm.weight` | `blk.{L}.post_norm.weight` |
| `model.language_model.layers.{L}.self_attn.{q,k,v}_proj.weight` | `blk.{L}.attn_{q,k,v}.weight` |
| `model.language_model.layers.{L}.self_attn.o_proj.weight` | `blk.{L}.attn_output.weight` |
| `model.language_model.layers.{L}.self_attn.{q,k}_norm.weight` | `blk.{L}.attn_{q,k}_norm.weight` |
| `model.language_model.layers.{L}.mlp.{gate,up,down}_proj.weight` | `blk.{L}.ffn_{gate,up,down}.weight` |
| `model.language_model.layers.{L}.layer_scalar` | `blk.{L}.layer_output_scale.weight` |
| `model.language_model.layers.{L}.per_layer_input_gate.weight` | `blk.{L}.inp_gate.weight` |
| `model.language_model.layers.{L}.per_layer_projection.weight` | `blk.{L}.proj.weight` |
| `model.language_model.embed_tokens_per_layer.weight` | `per_layer_token_embd.weight` |
| `model.language_model.per_layer_model_projection.weight` | `per_layer_model_proj.weight` |
| `model.language_model.per_layer_projection_norm.weight` | `per_layer_proj_norm.weight` |

The GGUF form also transposes: safetensors `q_proj.weight` is `[2048, 1536]` where GGUF
`attn_q.weight` is `[1536, 2048]`.

Three per-block tensors of the per-layer-embedding path lose the `per_layer` substring in
translation: `per_layer_input_gate` becomes `inp_gate`, `per_layer_projection` becomes `proj`,
and `post_per_layer_input_norm` becomes `post_norm`. This matters because the runtime detects
per-layer embeddings by substring (see below), and those three names do not contain it. The
three file-level tensors keep the substring, which is what makes the guard fire at all. See
[The `per_layer` substring guard](#the-per_layer-substring-guard).

## What the backend checks

`check_supported` (`cpu.rs:616`) diagnoses by name before falling back to a generic report.
The design is deliberate: loading is name-directed, so an unimplemented feature would otherwise
present as ignored tensors and plausible, wrong logits.

Seven architecture features were once refused here and are now implemented. Each refusal has
become a coherence check on the same signal: the load still fails if the metadata and the
tensors describe different models, but no longer merely because the feature is present.

| Feature | What replaced the refusal | Site |
|---------|---------------------------|------|
| Per-layer input embeddings | All six tensor families are required together, and the declared width must be present so the packed table can be split | `cpu.rs:622-668` |
| Shared KV cache | Only a layer that owns its cache must carry `attn_k`/`attn_v`; a shared layer legitimately carries neither | `cpu.rs:670-698` |
| Per-layer feed-forward width | Each `ffn_up` last dimension must equal the width `feed_forward_length` declares for that layer | `cpu.rs:700-734` |
| Per-layer head dimension | `rope.dimension_count` and `dimension_count_swa` are each checked against the head width of the regime they govern | `cpu.rs:747-787` |
| Dual RoPE | `rope.freq_base` and `freq_base_swa` are read; any other `{arch}.rope.*` key is still refused | `cpu.rs:761` |
| Logit softcapping | `final_logit_softcapping` is read; `attn_logit_softcapping`, `attention.scale`, and `logit_scale` are still refused | `cpu.rs:813-831` |
| `layer_output_scale` | Listed in `OPTIONAL_LAYER_TENSORS`, so it is consumed rather than reported unconsumed | `cpu.rs:563` |

Four things still refuse a checkpoint outright, and all four remain correct behavior:

- **Partial rotary applied by slicing.** A `rope.dimension_count` narrower than its regime's
  head width. Gemma 4 expresses its partial rotation through `rope_freqs.weight` at full width
  instead, so a narrower declaration describes a different model.
- **Attention softcapping or logit rescaling.** `attn_logit_softcapping`, `attention.scale`, and
  `logit_scale` carry no tensor, so the unconsumed backstop cannot see them and either would
  change every logit silently. Gemma 2 publishes both softcapping keys, which is why the test
  names `final_logit_softcapping` explicitly rather than relaxing to a substring.
- **Any other `{arch}.rope.*` key.** The allowlist stays an allowlist: it is what stands between
  a converter that adds a third RoPE regime and a silent wrong rotation.
- **Anything else unconsumed** (`cpu.rs:834-866`). Tensor families outside the consumed lists,
  including features published after this code. Value normalization and key-equals-value
  attention, the two published Gemma 4 features this backend does not implement, arrive here.

### The axis with no named diagnostic

One remains. **Per-layer head dimension** is now carried by `LayerSpec::head_dim` and read from
`attention.key_length` and `attention.key_length_swa`, but a checkpoint whose weight shapes
disagree with those keys is still caught only by the tensor loader, which reports a shape rather
than the feature. The `rope.dimension_count` cross-check narrows the gap without closing it: it
fires only when the checkpoint declares a rotation width at all.

### `layer_output_scale`

Every layer carries a scalar of shape `[1]` (`blk.{L}.layer_output_scale.weight` in GGUF) with
no corresponding configuration field. It scales the entire layer output as the layer's last
operation, and the released checkpoints store values from **0.0186 to 0.879**, not 1.0. A
forward pass that ignored it would diverge from the reference model by roughly 54x in the first
layer, which is why it is loaded rather than treated as an inert family. The values are measured
in the `layer_scalar stored values` sections of `docs/models/gemma4-safetensors.txt`; see
[`layer_scalar` in docs/models/gemma-4.md](../models/gemma-4.md#layer_scalar).

Because the tensor is optional rather than required, a checkpoint that omits it loads and
computes as though every scale were 1.0. That is the correct reading of an absent tensor, and it
is why the absent case is covered by its own committed fixture rather than by assertion.

## The layer schedule is expressible

The runtime models the attention schedule as a stride,
`(i + 1) % sliding_window_pattern == 0 => Global` (`Config::layer_kind`, `cpu.rs:188`;
`kv::interleaved_layers`, `/rust-harness/runtime/src/kv.rs:47`). Checked against the published
`layer_types` arrays:

| Model | Layers | Full-attention indices | Period | Rule reproduces the array |
|-------|--------|------------------------|--------|---------------------------|
| E2B | 35 | 4, 9, 14, 19, 24, 29, 34 | 5 | Yes |
| E4B | 42 | 5, 11, 17, 23, 29, 35, 41 | 6 | Yes |

Both variants place a full-attention layer last. No per-layer array is needed to express either
schedule; a converter that wrote `sliding_window_pattern` as the uint32 period 5 or 6 would
satisfy the runtime exactly. The published files write a bool array instead, and the loader
decodes it: `Config::from_gguf` reads `attention.sliding_window_pattern` as either a uint32
stride or a bool array marking each layer sliding (`cpu.rs:221-241`). Given an array it takes
the first global layer as the period and then verifies every remaining entry against that
stride, so a schedule that is not periodic fails the load naming the key rather than being
approximated. The three shipped schedules and the non-periodic refusal are covered at
`cpu.rs:3155` and `cpu.rs:3180`.

## Chat markers

Gemma 4 does not use the Gemma 3 turn markers. It uses asymmetric paired markers where the bar
denotes the open or close side:

| Role | Open | Close |
|------|------|-------|
| Turn | `<\|turn>` | `<turn\|>` |
| Channel | `<\|channel>` | `<channel\|>` |
| Tool call | `<\|tool_call>` | `<tool_call\|>` |
| Tool | `<\|tool>` | `<tool\|>` |
| Tool response | `<\|tool_response>` | `<tool_response\|>` |

A reasoning marker `<|think|>` is also defined. The assistant role string is `model`.

`/rust-harness/harness/src/template.rs` emits these markers. It previously carried Gemma 3's
`<start_of_turn>` and `<end_of_turn>`, which are absent from the Gemma 4 vocabulary and so
tokenized as byte runs rather than failing; that defect is fixed. The reference it was ported
against is the Python tier at `/training/models/families.py:75-89`, and
`/schemas/chat_template.json` now pins the two tiers together, asserted from both sides by
`/rust-harness/harness/tests/template_contract.rs` and
`/tests/training/test_chat_template_contract.py`.

The renderer emits turn markers only. The channel, tool-call, tool, and think markers above are
neutralized in untrusted content but never written: the canonical template strips thought
channels from history, and the harness carries a tool call as JSON inside a model turn.

The GGUF files embed the canonical template in `tokenizer.chat_template` (18,567 characters on
E2B), so a token-level golden can be checked without a download. That golden remains a
workstation task: it needs the real Gemma 4 vocabulary, which no committed fixture carries.

Special token identifiers are `<bos>` 2, `<eos>` 1, `<pad>` 0. The ordering of the first two
looks transposed and is not.

## Status of the local GGUF files

Six files exist under `/mnt/ai-data/models/gguf/`. **The two format-level blockers are
cleared; what remains is architectural.**

| File | Architecture | Tensors | BF16 tensors | Bool array |
|------|--------------|---------|--------------|------------|
| `gemma-4-E2B-it/gemma-4-E2B-it-BF16.gguf` | `gemma4` | 601 | 318 | Yes, n=35 |
| `gemma-4-E2B-it/mmproj-…-BF16.gguf` | `clip` | — | 247 | No |
| `gemma-4-E2B-it/mtp-…-BF16.gguf` | `gemma4-assistant` | — | 23 | Yes, n=4 |
| `gemma-4-E4B-it/gemma-4-E4B-it-BF16.gguf` | `gemma4` | — | 381 | Yes, n=42 |
| `gemma-4-E4B-it/mmproj-…-BF16.gguf` | `clip` | — | 247 | No |
| `gemma-4-E4B-it/mtp-…-BF16.gguf` | `gemma4-assistant` | — | 23 | Yes, n=4 |

**Storage type — resolved.** All six carry BF16 tensors, GGML type 30: 1,239 in total. The
reader now decodes them. Requantizing to Q8_0 was the obvious alternative and would have been
the wrong one — it is lossy, and the interp tier fits probes and sparse autoencoders on
activations of these exact weights, so a quantized harness would silently be studying a
different model.

**Metadata type — resolved.** Four of six carry `sliding_window_pattern` as a bool array, which
the parser refused before any tensor was read. Bool arrays now decode. Note this was the
blocker requantizing would *not* have cleared, since it is a metadata property the converter
emits regardless of weight type.

**Architecture — resolved.** The features that once refused these files are implemented. E2B
publishes `feed_forward_length` as an int32 array, both variants publish `rope.freq_base_swa`,
`final_logit_softcapping`, `attention.shared_kv_layers`, and the per-layer embedding tensors,
and every one of those is now read. Nothing in the published `gemma4` architecture is known to
refuse.

That is a claim about the synthetic fixtures, not about these files. Both the uniform and the
E4B-shaped fixture match an independently written numpy reference within `2e-4`, but two
implementations written from the same documents agree on a shared misreading. Loading a real
checkpoint is runbook entry RH.2 and remains outstanding.

**The schedule is consumed, not merely parsed.** `Config::from_gguf` decodes the bool array to
the stride `Config::layer_kind` uses, and refuses a schedule that is not periodic rather than
approximating it (see
[The layer schedule is expressible](#the-layer-schedule-is-expressible)). Both published
variants are periodic.

The `mtp-*.gguf` files carry a separate `gemma4-assistant` architecture with `block_count` 4.
They are not a text model for this runtime.

Independent corroboration worth recording: the `sliding_window_pattern` arrays are n=35 and
n=42, matching the layer counts read from the safetensors headers. Two file formats and two
readers agree.

Recorded negative result: **no MTP, altup, or laurel tensors exist in either safetensors
checkpoint**, despite the separate `mtp-*.gguf` files.

The per-finding record of these blockers lives in
`/docs/verification/investigations/gemma-4-conformance_investigation.md` under the `GC-` and `GH.` identifiers.

## Resolved from upstream source

Revisions read:

| Source | Path | Commit |
|--------|------|--------|
| `ggml-org/llama.cpp` | `conversion/gemma.py` | `f4043fec0103872bf4339f6fa18d8b17824d5b6a` |
| `ggml-org/llama.cpp` | `gguf-py/gguf/constants.py`, `gguf-py/gguf/tensor_mapping.py` | `07132750825a4f2d27a547cd9cdde1c6f6001885` |
| `huggingface/transformers` | `src/transformers/models/gemma4/modeling_gemma4.py` | `ff886afacfd81e09dcf70d29d5bbf76babf672a5` |
| `huggingface/transformers` | `src/transformers/modeling_rope_utils.py` | `9451e3ab0b21cc51638dce25c101ab5f2233e91d` |

### `rope.dimension_count` against `partial_rotary_factor`

**The two representations do not contradict each other.** Both describe a rotation performed at
the full head width in which the trailing frequencies are inert. The partial factor is carried by
the frequency vector, not by the width.

On the Hugging Face side, `_compute_proportional_rope_parameters`
(`modeling_rope_utils.py`, lines 193-266) states in its docstring, lines 216-219, that
"proportional RoPE will always return an encoding that is the size of `head_dim`". It computes
`rope_angles = int(rope_proportion * head_dim // 2)`, which is 64 for `head_dim` 512 and
`partial_rotary_factor` 0.25, then appends `nope_angles = head_dim // 2 - rope_angles`, that is
192, zeros. `apply_rotary_pos_emb` (`modeling_gemma4.py`, lines 784-803) runs over all 512
dimensions; the 192 zero frequencies produce `cos` of 1.0 and `sin` of 0.0, an identity.

On the llama.cpp side, `Gemma4Model.set_gguf_parameters` (`conversion/gemma.py`, lines 695-700)
writes the full width deliberately, with the reason in the comment:

```python
n_rot_full = int(head_dim_full) # "proportional" is used, see generate_extra_tensors
n_rot_swa = int(head_dim_swa * partial_rotary_factor_swa)
self.gguf_writer.add_rope_dimension_count(n_rot_full)
self.gguf_writer.add_rope_dimension_count_swa(n_rot_swa)
```

`Gemma4Model.generate_extra_tensors` (lines 702-717) then emits the partial factor as a
`rope_freqs` tensor of frequency divisors:

```python
n_rot_full = int(head_dim_full * partial_rotary_factor_full / 2)
n_unrot_full = int(head_dim_full / 2) - n_rot_full
values = [1.0] * n_rot_full + [1e30] * n_unrot_full
```

That is 64 entries of 1.0 followed by 192 entries of 1e30, a 256-element vector. Dividing an
inverse frequency by 1e30 drives the angle to approximately zero, reproducing the Hugging Face
identity. The GGUF inventory confirms the tensor and its length: `rope_freqs.weight [256] F32`,
one per file (`docs/models/gemma4-gguf.txt` lines 630 and 639). The converter's comment at lines
703-706 records that the tensor is a workaround, because the Hugging Face split-half ordering is
`cc000000ss000000` while ggml's neox rotation supports only `ccss000000000000`, and the head
cannot be rearranged without breaking `use_alternative_attention`.

Two consequences for ingestion:

- `gemma4.rope.dimension_count = 512` is the **width of the rotation operation**, not a claim
  that 512 dimensions receive a non-trivial rotation. A reader that treats it as the count of
  rotating dimensions and skips `rope_freqs.weight` produces wrong logits without erroring.
- `gemma4.rope.dimension_count_swa = 256` is `head_dim_swa * partial_rotary_factor_swa`
  (line 698). With `head_dim` 256 and no sliding-layer partial factor, the default of 1.0 applies
  and the sliding regime genuinely rotates its full width. `rope_freqs.weight` applies to the
  full-attention layers only, as stated at line 708.

### The `proportional` rope type

Resolved. The formula is quoted above and in
`/docs/models/gemma-4.md`. The GGUF form carries no `rope_type` key because the type is expressed
structurally, through `rope_freqs.weight` plus the full-width `dimension_count`.

### `layer_scalar`

Resolved. `Gemma4TextDecoderLayer` declares it as `nn.Buffer(torch.ones(1))`
(`modeling_gemma4.py`, line 1371) and applies `hidden_states *= self.layer_scalar` at line 1444,
the final statement of the decoder layer, after the attention residual, the feed-forward residual,
and the per-layer-embedding residual. It scales the whole layer output including the residual.
The GGUF name is `blk.{bid}.layer_output_scale` (`constants.py`, line 1320), mapped from
`model.layers.{bid}.layer_scalar` (`tensor_mapping.py`, lines 735-738). The converter appends the
`.weight` suffix in `Gemma4Model.filter_tensors` (`conversion/gemma.py`, lines 747-748), which is
why the transcript shows `blk.{L}.layer_output_scale.weight [1] F32`.

Note that `filter_tensors` **renames** rather than drops it: the `name.endswith("layer_scalar")`
branch appends a suffix and falls through to `super().filter_tensors`. The tensor survives
conversion, and the GGUF carries one per layer. The stored values are not 1.0, so a backend that
reads the GGUF and ignores this tensor is not merely omitting a no-op — see
[`layer_output_scale`](#layer_output_scale).

### The per-layer embedding data path

Resolved. Full derivation in `/docs/models/gemma-4.md`. In summary, mapped onto GGUF names:

1. Once per forward, `per_layer_token_embd` is looked up and reshaped to
   `[batch, seq, n_layers, 256]` (`modeling_gemma4.py`, lines 1724-1766).
2. `per_layer_model_proj` projects the token embeddings, the result is scaled by
   `hidden_size ** -0.5`, reshaped identically, and normalized by `per_layer_proj_norm`; the two
   components are then summed and scaled by `2.0 ** -0.5` (lines 1768-1801).
3. Inside layer `i`, after the feed-forward residual and before `layer_output_scale`, the block at
   lines 1435-1442 runs: `inp_gate` projects the hidden state from 1536 to 256, the activation is
   applied, the layer's 256-dimensional per-layer embedding **multiplies** it elementwise, `proj`
   maps back to 1536, `post_norm` normalizes, and the result is added to the residual.

The gate is multiplicative and operates in the 256-dimensional per-layer space. The join point is
a third residual update after the feed-forward block, not before or inside attention.

### The E4B shared-KV boundary

Resolved. `conversion/gemma.py` line 682 and `modeling_gemma4.py` line 1186 both compute the
boundary as `num_hidden_layers - num_kv_shared_layers`: index 15 on E2B (35 layers, 20 shared)
and index 24 on E4B (42 layers, 18 shared). `Gemma4TextMLP` derives the double-wide feed-forward
region from the same expression (`modeling_gemma4.py`, lines 1068-1070), and `conversion/gemma.py`
lines 683-686 writes the per-layer feed-forward array from it, so the E2B feed-forward correlation
is structural rather than coincidental and holds on E4B for the same reason.

Correction to the donor: there are **two donors, not one**. `store_full_length_kv`
(`modeling_gemma4.py`, lines 1189-1191) selects the last non-shared layer *of each layer type*,
and the shared state is keyed by layer type at lines 1241 and 1259. A shared `full_attention`
layer reuses the last non-shared `full_attention` layer's keys and values; a shared
`sliding_attention` layer reuses the last non-shared `sliding_attention` layer's. The comment at
`cpu.rs:264-265`, "the trailing layers reuse K/V from the last non-shared layer", is therefore
correct in direction and incomplete in detail.

### The `per_layer` substring guard

Confirmed in part, and the operational conclusion is the opposite of what the claim suggests.

The GGUF names are fixed in `constants.py` (revision
`07132750825a4f2d27a547cd9cdde1c6f6001885`, lines 1321-1328):

| Hugging Face name (`tensor_mapping.py`, lines 740-770) | GGUF name (`constants.py`) | Contains `per_layer` |
|---|---|---|
| `model.embed_tokens_per_layer` | `per_layer_token_embd` (line 1321) | yes |
| `model.per_layer_model_projection` | `per_layer_model_proj` (line 1322) | yes |
| `model.per_layer_projection_norm` | `per_layer_proj_norm` (line 1323) | yes |
| `model.layers.{bid}.per_layer_input_gate` | `blk.{bid}.inp_gate` (line 1326) | no |
| `model.layers.{bid}.per_layer_projection` | `blk.{bid}.proj` (line 1327) | no |
| `model.layers.{bid}.post_per_layer_input_norm` | `blk.{bid}.post_norm` (line 1328) | no |

The claim at lines 197-200 of this document holds and understates the case: **three** per-block
families lose the substring, not two. `post_per_layer_input_norm` becomes `blk.{bid}.post_norm`,
which additionally collides in appearance with the ordinary Gemma post-norm naming and is
distinguishable from `blk.{bid}.post_attention_norm` and `blk.{bid}.post_ffw_norm` only by
knowing the architecture.

The substring was once how the backend detected the feature, and it worked only because the three
file-level tensors retain it and are present in every per-layer-embedding checkpoint
(`docs/models/gemma4-gguf.txt` lines 632-635 and 641). It was sound for refusal and would have
been unsound for enumeration: a whitelist keyed on the same substring silently omits `inp_gate`,
`proj`, and `post_norm`, which is three of the six families and every per-block one.

The implementation does not key on it. `PLE_TOP_LEVEL_TENSORS` and `PLE_LAYER_TENSORS`
(`cpu.rs:568` and `cpu.rs:584`) list all six by name, and `check_supported` requires them
together: a checkpoint carrying five loads nothing, rather than skipping the residual join
wherever the sixth is missing.

## Open questions

Recorded rather than guessed.

- **Whether shared-KV layers in the released checkpoints carry key and value weights.**
  `modeling_gemma4.py` line 1199 builds `k_proj` and `v_proj` only for non-shared layers, which
  for E2B means 15 of 35. Both transcripts disagree with that: `docs/models/gemma4-safetensors.txt`
  lines 414 and 529 record `k_proj.weight` on 28 plus 7 language-model layers, and
  `docs/models/gemma4-gguf.txt` lines 613 and 625 record `attn_k.weight` on 28 plus 7 blocks, 35 in total.
  Nothing in `conversion/gemma.py` drops shared-layer key or value tensors, so the converter is
  passing through what the checkpoint holds. Whether these are unused residue or the code and the
  weights are out of step was not settled from the sources read.

  This was directly load-bearing for the missing-`attn_k` signature, which on these two files is
  never true, so the shared-KV refusal never fired and the `per_layer` guard was what actually
  stopped the load. Both refusals are now gone: the backend implements sharing, keyed on
  `attention.shared_kv_layers`, and the tensor signature survives only inverted — a layer that
  owns its cache and lacks `attn_k`/`attn_v` has nothing to attend over.

  The backend follows the modeling code and leaves the shared tail's key and value tensors
  unread. That began as a decision made under the uncertainty recorded here. It is now a
  resolution: RH.1a check 1, run on 2026-08-09, showed the shared tail's `k_proj` and `v_proj`
  retain their truncated-normal initialization — a `|mean| / std` ratio of 0.8217 with a spread
  of 0.0010 and negative excess kurtosis on all 20 E2B layers — while `q_proj` and `o_proj` in
  the same layers carry the trained signature. The surplus tensors never entered the gradient
  path. The measurement is in
  [docs/models/gemma-4.md](../models/gemma-4.md#the-shared-kv-weight-conflict) and the procedure
  is RH.1a in [docs/verification/rust-harness.md](../verification/rust-harness.md).
- ~~**The stored value of `layer_output_scale`.** The transcripts record the shape `[1]` but not
  the value, so whether the multiplication is a no-op in the released weights is unverified.~~

  Resolved, and stale when written: the `layer_scalar stored values` sections of
  `docs/models/gemma4-safetensors.txt` dump every layer, and the values run 0.0186 to 0.879, a
  spread of roughly 54x. The multiplication is not a no-op on any layer. The forward pass
  applies it as the layer's last operation.
- **The `1e30` divisor as an exact identity.** `conversion/gemma.py` line 715 uses 1e30 rather
  than infinity, so the resulting angle is very small rather than exactly zero. Whether the
  residual differs from the Hugging Face exact-zero frequency at F32 precision was not measured.

## Reproducing this document

Both transcripts regenerate on the workstation with no GPU and no network:

```bash
$ ./scripts/models/gemma4_header_transcript.sh    # safetensors headers and config
$ ./scripts/models/gguf_header_transcript.sh      # GGUF metadata and tensor inventory
```

To inspect a single GGUF file against the runtime's own limits, use the inspection binary,
which calls the same `check_supported` the loader does, so the tool and the loader cannot
disagree:

```bash
$ cargo run --bin gguf-info -- /mnt/ai-data/models/gguf/gemma-4-E2B-it/gemma-4-E2B-it-BF16.gguf
```
