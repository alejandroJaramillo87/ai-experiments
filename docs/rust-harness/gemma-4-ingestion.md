# Gemma 4 ingestion reference

What the `rust-harness` runtime requires to ingest a Gemma 4 checkpoint, what the published
checkpoints actually contain, and which architecture features the backend refuses by name.

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
- [What the backend refuses](#what-the-backend-refuses)
- [Refusals that do not fire](#refusals-that-do-not-fire)
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

`Config::from_gguf` (`/rust-harness/runtime/src/model/cpu.rs:83-96`) requires eleven
architecture keys, all prefixed with the value of `general.architecture`, which must begin with
`gemma`. Every one is mandatory: a missing key or a wrong value type fails the load.

| Key (after the `{arch}.` prefix) | Required type | Becomes |
|----------------------------------|---------------|---------|
| `embedding_length` | uint32 | `embed_dim` |
| `block_count` | uint32 | `n_layers` |
| `attention.head_count` | uint32 | `n_heads` |
| `attention.head_count_kv` | uint32 | `n_kv_heads` |
| `attention.key_length` | uint32 | `head_dim` |
| `feed_forward_length` | uint32 | `ffn_dim` |
| `context_length` | uint32 | `context_length` |
| `attention.sliding_window` | uint32 | `sliding_window` |
| `attention.sliding_window_pattern` | uint32 | `sliding_window_pattern` |
| `rope.freq_base` | float32 | `rope_freq_base` |
| `attention.layer_norm_rms_epsilon` | float32 | `rms_eps` |

Two further keys are read outside `Config`: `general.architecture`, and
`tokenizer.ggml.tokens`, whose length supplies `vocab_size` (there is no `vocab_size` key).

`Config` derives `Copy` (`cpu.rs:39-53`). Every field is a scalar, so the type is structurally
incapable of holding a per-layer vector. This is the single constraint that most shapes the
rest of this document: every place Gemma 4 varies a quantity per layer, the runtime must either
express it as a uniform value or refuse the checkpoint.

`attention.value_length` is never read. The forward pass computes
`kv_dim = n_kv_heads * head_dim` and uses it for both keys and values (`cpu.rs:682-683`), which
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

Thirteen per-layer tensors (`cpu.rs:156-169`), two top-level (`:172-175`), and one tolerated as
inert (`:177-180`):

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
as a stride (`Config::layer_kind`, `cpu.rs:142-149`), while the converter writes a bool array of
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

Two per-layer tensors of the per-layer-embedding path lose the `per_layer` substring in
translation: `per_layer_input_gate` becomes `inp_gate` and `per_layer_projection` becomes
`proj`. This matters because the runtime detects per-layer embeddings by substring
(see below), and those two names do not contain it.

## What the backend refuses

`check_supported` (`cpu.rs:206-330`) diagnoses unimplemented features by name before falling
back to a generic report. The design is deliberate: loading is name-directed, so an
unimplemented feature would otherwise present as ignored tensors and plausible, wrong logits.
Each refusal below is correct behavior, not a defect.

| Feature | Trigger on a Gemma 4 file | Site |
|---------|---------------------------|------|
| Per-layer input embeddings | Any tensor name containing `per_layer` — matches `per_layer_token_embd`, `per_layer_model_proj`, `per_layer_proj_norm` | `cpu.rs:209-217` |
| Shared KV cache | A layer with `attn_q` but no `attn_k`/`attn_v` | `cpu.rs:219-230` |
| Per-layer feed-forward width | Any `ffn_up` last dimension differing from `feed_forward_length` — fires on E2B's 12288 layers | `cpu.rs:232-252` |
| Dual or scaled RoPE | Any `{arch}.rope.*` key other than `freq_base` and `dimension_count` — fires on `rope.freq_base_swa` and `rope.dimension_count_swa` | `cpu.rs:254-280` |
| Partial rotary | `rope.dimension_count` unequal to `head_dim` | `cpu.rs:265-273` |
| Logit or attention softcapping | Any `{arch}.*softcapping*` key — fires on `final_logit_softcapping` | `cpu.rs:282-300` |
| Anything else unconsumed | Tensor families outside the consumed lists | `cpu.rs:302-328` |

The generic backstop catches two Gemma 4 tensor families that no named check covers:
`blk.{L}.post_norm.weight` (the fifth normalization, on the per-layer embedding path) and
`blk.{L}.layer_output_scale.weight`. Both report as unconsumed families rather than by feature
name.

### The two axes with no named diagnostic

**Per-layer head dimension.** Sliding layers use 256 and full-attention layers use 512, visible
directly in the weight shapes. `Config::head_dim` is one scalar. There is no check for this,
and no error message names it: a mismatch surfaces only when the tensor loader finds a shape it
did not expect, and the message reports a shape rather than the feature. Of everything Gemma 4
varies per layer, this is the only axis the runtime cannot name.

**`layer_scalar`.** Every layer carries a scalar of shape `[1]`
(`blk.{L}.layer_output_scale.weight` in GGUF) with no corresponding configuration field. It
reaches the generic unconsumed-tensor backstop. Its role in the residual or attention path is
undetermined; see [Open questions](#open-questions).

## Refusals that do not fire

Two Gemma 4 features are declared in the GGUF metadata, are not implemented by the forward
pass, and are **not** refused. Neither currently causes a wrong result, because an earlier
refusal stops the load first. Both would matter if the earlier refusals were lifted.

**`gemma4.attention.shared_kv_layers`** is present and set (20 on E2B, 18 on E4B). The named
shared-KV check looks for *missing* `attn_k`/`attn_v` tensors, and the published files carry
every projection for every layer, so it does not fire. The metadata refusal list names only
softcapping, `attention.scale`, and `logit_scale`, so this key passes through silently. A cache
allocator would give all 35 or 42 layers independent storage — correct output, more memory than
the architecture requires. The allowlist-versus-denylist tradeoff is recorded at
`cpu.rs:285-287`: naming refused keys avoids refusing every descriptive key a converter adds
later, at the cost of gaps like this one.

**`gemma4.embedding_length_per_layer_input`** is present and set to 256. It is likewise not in
the refused set. In practice the per-layer tensor check fires first.

## The layer schedule is expressible

A positive result, and the one Gemma 4 feature the existing `Config` shape handles without
change. The runtime derives the attention schedule arithmetically as
`(i + 1) % sliding_window_pattern == 0 => Global` (`cpu.rs:142-149`,
`/rust-harness/runtime/src/kv.rs:46-58`). Checked against the published `layer_types` arrays:

| Model | Layers | Full-attention indices | Period | Rule reproduces the array |
|-------|--------|------------------------|--------|---------------------------|
| E2B | 35 | 4, 9, 14, 19, 24, 29, 34 | 5 | Yes |
| E4B | 42 | 5, 11, 17, 23, 29, 35, 41 | 6 | Yes |

Both variants place a full-attention layer last. No per-layer array is needed to express either
schedule; a converter that wrote `sliding_window_pattern` as the uint32 period 5 or 6 would
satisfy the runtime exactly. The published files write a bool array instead.

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

`/rust-harness/harness/src/template.rs:2-9,15-16,40` currently emits Gemma 3's `<start_of_turn>` and
`<end_of_turn>`. That is a defect against these weights. The Python tier already carries the
correct form at `/training/models/families.py:82-89`, which is the reference to match. The
GGUF files embed the canonical template in `tokenizer.chat_template` (18,567 characters on
E2B), so it can be checked without a download.

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

**What still refuses these files, correctly.** E2B publishes `feed_forward_length` as an int32
array, and both variants publish `rope.freq_base_swa` and `final_logit_softcapping`. The config
layer or `check_supported` refuses each in turn, and the per-layer embedding tensors trip the
named check first. The failure now lands on the architecture, with a message naming the
feature, rather than on a storage type — which is where it belongs.

**Parsing the schedule is not consuming it.** `sliding_window_pattern` now decodes as a bool
array, but `Config::layer_kind` still expects a uint32 stride and ignores the array. A file
whose schedule disagrees with the derived `(i + 1) % period` rule would therefore be built as
though the rule held. Both published variants happen to match the rule (see
[The layer schedule is expressible](#the-layer-schedule-is-expressible)), and the architectural
refusals above stop these files well before it matters, but the gap is real and open.

The `mtp-*.gguf` files carry a separate `gemma4-assistant` architecture with `block_count` 4.
They are not a text model for this runtime.

Independent corroboration worth recording: the `sliding_window_pattern` arrays are n=35 and
n=42, matching the layer counts read from the safetensors headers. Two file formats and two
readers agree.

Recorded negative result: **no MTP, altup, or laurel tensors exist in either safetensors
checkpoint**, despite the separate `mtp-*.gguf` files.

The per-finding record of these blockers lives in
`/docs/verification/gemma-4-conformance.md` under the `GC-` and `GH.` identifiers.

## Open questions

Recorded rather than guessed.

- **`rope.dimension_count` against `partial_rotary_factor`.** The GGUF files set
  `rope.dimension_count` to 512, equal to `attention.key_length`, which reads as full rotation.
  The Hugging Face configuration sets `partial_rotary_factor: 0.25` on full-attention layers,
  which implies 128 of 512 dimensions rotate. The two representations disagree. Resolving this
  requires reading the converter, and it determines whether the partial-rotary check at
  `cpu.rs:265-273` is measuring what its message claims.
- **The `proportional` rope type.** Its scaling formula is not defined in `config.json` and has
  no counterpart in earlier Gemma releases. The GGUF form carries no equivalent key at all.
- **`layer_scalar`.** Present on every layer with no configuration field and no documented
  position in the residual or attention path.
- **The per-layer embedding data path.** The tensor names establish which projections exist.
  The order of operations, the join point with the residual stream, and whether
  `per_layer_input_gate` gates additively or multiplicatively are not determined by shapes.
- **The E4B shared-KV boundary.** E2B's 20 shared layers are indices 15 through 34, corroborated
  by the double-wide feed-forward region. E4B's 18-layer boundary has no corroborating signal in
  either file format and is assumed, not verified.

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
