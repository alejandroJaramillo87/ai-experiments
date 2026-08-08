# Gemma 4 interpretability reference

What the interp tier requires to harvest activations, train probes and sparse autoencoders, and
export steering vectors against a Gemma 4 checkpoint.

This tier consumes very little of the architecture, so this document is short by design. It
records the handful of facts `/interp` actually reads and the two hazards specific to these
weights: the per-layer embedding tensors a hook must never return, and the composite module
layout. The general family reference is `/docs/models/gemma-4.md`.

Scope: E2B and E4B. The 12B and 31B variants have no code path here, and
`gemma4_unified_text` is deliberately unregistered.

## Table of contents

- [Provenance](#provenance)
- [Registry facts](#registry-facts)
- [Residual width and layer counts](#residual-width-and-layer-counts)
- [Uniform post-block width](#uniform-post-block-width)
- [The per-layer embedding hazard](#the-per-layer-embedding-hazard)
- [The module path is composite](#the-module-path-is-composite)
- [Export safety](#export-safety)
- [Ecosystem gaps](#ecosystem-gaps)
- [Reproducing this document](#reproducing-this-document)

## Provenance

| Model | Repository | Revision (`x-repo-commit`) |
|-------|------------|----------------------------|
| E2B | `google/gemma-4-E2B` | `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` |
| E4B | `google/gemma-4-E4B` | `411aa17b749aa952df1359d2dcea73917a544d9a` |

Facts derive from the `config.json` and safetensors headers of the local copies under
`/mnt/ai-data/models/hf/`.

## Registry facts

`/interp/families.py:62-72` keys its registry on the configuration `model_type`. Gemma 4 needs
both spellings, and both are present:

| Key | Where it appears | Purpose |
|-----|------------------|---------|
| `gemma4` | top level of `config.json` | the composite multimodal configuration |
| `gemma4_text` | `config.json` under `text_config` | the language model |

`family_from_config` calls `unwrap_text_config` first (`families.py:90-96`), so a loaded
composite configuration resolves through `gemma4_text`. Registering both means a caller that
passes either the wrapper or the unwrapped text configuration succeeds.

`architectures[0]` is `Gemma4ForConditionalGeneration` on both variants, while the harvest
path loads through `AutoModelForCausalLM` (`interp/harvest.py`, in `check_hook_dims` and
`_saelens_runner`). That mismatch is resolved and benign: transformers 5.13.1 registers six
`gemma4*` keys in `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`, and `AutoModelForCausalLM.from_config`
returns `Gemma4ForConditionalGeneration` rather than refusing. Verified on the workstation;
the output is recorded in
[gemma-4-training.md](../training/gemma-4-training.md). This matters beyond our own code
because SAELens hardcodes `model_class_name="AutoModelForCausalLM"` with no override hook.
See also [The module path is composite](#the-module-path-is-composite).

## Residual width and layer counts

| Fact | `config.json` field | E2B | E4B |
|------|---------------------|-----|-----|
| Residual stream width (`d_model`) | `text_config.hidden_size` | 1536 | 2560 |
| Decoder layers | `text_config.num_hidden_layers` | 35 | 42 |
| Per-layer embedding width | `text_config.hidden_size_per_layer_input` | 256 | 256 |

`residual_width` (`families.py:110-112`) reads `hidden_size` after unwrapping, which is correct.
The value flows to `d_model` in the harvest manifest (`harvest.py:141-165`) and from there to
`d_in` for every sparse autoencoder (`/interp/sae/train.py:184`).

**Do not read `hidden_size_per_layer_input` as the residual width.** It is 256 on both variants
— a sixth of E2B's residual stream — and it describes an auxiliary input path, not the stream
itself.

The layer counts matter because `validate_harvest` sees only the recipe, and depth is a property
of the model. It checks that a hook index is non-negative and can go no further. The upper bound
lives in `check_hook_layer_bounds` (`harvest.py`), which reads `num_hidden_layers` from the same
single config read that resolves the width and family, and refuses an index the subject has no
layer for. It runs in the preflight before the weights load and again in `run_harvest`.

That bounds the range, which is all a depth can decide. It does **not** catch an in-range index
chosen for a wrong reason: a recipe naming layer 15 as "50 percent of a 30-layer stack" is a
legal index on a 35-layer model, merely the wrong one. Recipes stating a width or depth that
contradicts `schemas/gemma4_facts.json` are caught separately, by
`tests/models/test_gemma4_facts_propagation.py`.

Note also that layer 15 is the worst available choice on E2B: it is the
`num_hidden_layers - num_kv_shared_layers` boundary (35 − 20), so it is simultaneously the first
shared-KV layer and the first double-wide-MLP layer. The reference recipes hook layer 17, the
midpoint of indices 0–34 and a plain sliding-attention block.

## Uniform post-block width

A positive result, and the reason the interp tier can stay as simple as it is.

Gemma 4 varies two quantities per layer. The feed-forward width on E2B is 6144 for layers 0
through 14 and 12288 for layers 15 through 34. The attention head dimension is 256 on sliding
layers and 512 on the seven full-attention layers. **Neither is visible at the block boundary.**
Every decoder layer's output is `hidden_size` wide, in both variants.

That makes the uniform-width assumption throughout the tier sound:

- `/interp/probes/train.py:72-102` sizes `mu`, `sigma`, and weights from the captured array's
  second dimension, and compares across layers.
- `/interp/sae/evaluate.py:88-89` infers `d_model` from the cached rows.
- `/interp/steering/export.py:41-43` **hard-refuses** a direction set whose per-layer shapes
  differ: `directions must share one 1-D width`.
- `/interp/probes/export.py:31-35` applies the same refusal to probe weights.

Those refusals would fire if per-layer widths ever reached the residual stream. They do not
here. Recording this explicitly matters because the assumption is load-bearing and was
previously undocumented — a reader who saw the 256/512 head-dim split in the family reference
might reasonably conclude the tier was unsound.

## The per-layer embedding hazard

The E-series injects a 256-wide per-token vector into each decoder layer through a gate and a
projection. The tensors are:

| Tensor | E2B shape | E4B shape |
|--------|-----------|-----------|
| `model.language_model.embed_tokens_per_layer.weight` | `[262144, 8960]` | `[262144, 10752]` |
| `model.language_model.per_layer_model_projection.weight` | `[8960, 1536]` | `[10752, 2560]` |
| `model.language_model.per_layer_projection_norm.weight` | `[256]` | `[256]` |
| `model.language_model.layers.{L}.per_layer_input_gate.weight` | `[256, 1536]` | `[256, 2560]` |
| `model.language_model.layers.{L}.per_layer_projection.weight` | `[1536, 256]` | `[2560, 256]` |
| `model.language_model.layers.{L}.post_per_layer_input_norm.weight` | `[1536]` | `[2560]` |

The upstream sources settle how these join the stream, which the shapes alone do not say. The
per-layer input is gated **multiplicatively** in 256-dimensional space, projected back to
`hidden_size`, and added as a **third residual update, after the feed-forward block**. The
layer order is attention, feed-forward, per-layer embedding, then `layer_scalar`. So a hook on
the decoder layer's output captures the residual with the per-layer contribution already
folded in — it is not a separable side channel. Derivations and line citations are in
[docs/models/gemma-4.md](../models/gemma-4.md).

A hook must return the post-block residual, never one of these. `check_hook_dims`
(`harvest.py:168-224`) exists for exactly this, and its docstring names Gemma 4: the check
"proves the hook output is the hidden_size residual and not a PLE tensor."

Two limits on that guarantee, both worth knowing before trusting it:

**The check is width-only, and cannot exclude every per-layer tensor.** Note the shape of
`per_layer_projection`: it maps 256 dimensions back to `hidden_size`. Its output is 1536 wide on
E2B — identical to the residual stream. A width assertion cannot distinguish the two. The check
does exclude the 256-wide gate output and the packed table, which are the likelier mistakes.

**The check is opt-in, and mutually exclusive with harvesting.** It runs only under the
`--check-hooks` flag, and that branch reports the widths and returns immediately rather than
proceeding to harvest (`/interp/commands/harvest.py:56-63`). `run_harvest` never calls it. So
the check and the harvest are two separate invocations, and nothing enforces that the check ran
first, or that it ran against the same recipe. `/interp/probes/capture.py:66-118` performs no
width check at all.

## `layer_scalar` scales what the hook captures

Every decoder layer carries `layer_scalar`, a shape-`[1]` buffer declared as
`nn.Buffer(torch.ones(1))` and applied as the **last operation of the layer**, after all three
residual updates. It multiplies the entire layer output, which is precisely the tensor a
residual-stream hook records.

Two consequences for this tier:

- Captured activations are scaled by it. It is a per-layer scalar, so it cannot change the
  direction of any activation vector or the geometry of a probe or SAE trained on one layer.
  It does rescale magnitudes, so a comparison of raw magnitudes *across* layers is comparing
  differently-scaled quantities.
- It has no `config.json` field, so nothing in the config or the manifest records it. Reading
  it requires the checkpoint.

Resolved: the released checkpoints store values well away from 1.0. E2B spans 0.0186 at layer 0
to 0.879 at layer 30, with layer 17 — the hook layer of the reference captures — at 0.641. The
values are in the `layer_scalar stored values` sections of `docs/models/gemma4-safetensors.txt`;
[docs/models/gemma-4.md](../models/gemma-4.md#layer_scalar) owns the derivation.

What that means here, stated carefully because the answer is reassuring and the reasoning is
the only reason to believe it:

- **Captures are correct as taken.** The hook is a forward hook on the decoder layer module
  (`interp/families.py`, `_COMPOSITE_DECODER_TEMPLATE`), and the multiply is that module's last
  operation, so the tensor the hook receives is already scaled. It is the same tensor the next
  layer consumes. No correction factor is needed, and none should be applied.
- **Magnitude measurements inherit the scaling.** The fp16 headroom figures in
  [harvest-precision-and-resume.md](harvest-precision-and-resume.md) were measured through the
  same hook, so they describe post-scale values. Since every scalar is below 1, the scaling can
  only reduce magnitudes, which widens headroom rather than narrowing it. The fp16 decision is
  unaffected.
- **Cross-layer magnitude comparisons are not provisional, they are wrong** unless the scalars
  are divided out. A 47x spread in per-layer scale sits underneath any such comparison. This is
  no longer a caveat pending measurement; it is a known factor with known values.

Within-layer work — which is where this tier operates — is unaffected throughout.

## Key and value share a projection on full-attention layers

`attention_k_eq_v` is narrower than its name suggests. It applies only where the layer is
**not** sliding, so on the seven full-attention layers and nowhere else. On those layers
`v_proj` is not constructed at all: the value states are the key projection's output taken
before `k_norm` and before the rotary embedding, with `v_norm` applied to them separately.

This explains the `self_attn.v_norm` warning below. The module exists in a config-built tree
because the code constructs the norm; no checkpoint tensor backs it because the projection it
would normalise was never built.

## The module path is composite

The hook path is a **PyTorch named-module path**, resolved through `model.get_submodule`
(`harvest.py:200`) and by `resolve_submodule` (`families.py:120-129`). The safetensors
inventory keys its tensors as `model.language_model.layers.{L}.*`, which is a state-dict key
namespace — a distinct thing that is not required to agree.

For these checkpoints the two **do** agree, and the module path is composite:

```
model
model.vision_tower
model.vision_tower.encoder.layers.0
model.language_model
model.language_model.layers.0
model.language_model.layers.0.self_attn
```

`Gemma4ForConditionalGeneration` places the vision tower and the language model as siblings
under `model`. There is no `model.layers` attribute, so the flat template that
`/interp/families.py` registered until this was verified raised `AttributeError` at
hook-attachment time on every E-series checkpoint. The tier could not harvest from these
weights at all; it failed loudly rather than reading the wrong tensor.

`family_from_config` (`families.py:99-119`) now selects the path on the presence of
`text_config` rather than on the family alone, because the layout follows the config the caller
loaded, not the model family:

| Config passed | Hook path |
|---------------|-----------|
| composite (`model_type: gemma4`, has `text_config`) | `model.language_model.layers.{n}` |
| bare text (`model_type: gemma4_text`) | `model.layers.{n}` |

A family with no composite template registered raises `UnknownHookFamilyError` when handed a
composite config, rather than falling back to the flat path — the fallback is the defect.

Reproduce the module tree on the workstation. This builds the model on the meta device from
the configuration alone, so it reads no weights and needs no GPU:

```bash
$ uv run python -c "
import torch
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained('/mnt/ai-data/models/hf/gemma-4-E2B')
with torch.device('meta'):
    m = AutoModelForCausalLM.from_config(cfg)
print([n for n, _ in m.named_modules()][:40])
"
```

Use `uv run`, not a bare `python3`: `transformers` is a core-group dependency and is absent
from a system interpreter.

**Warning!** That module tree includes `self_attn.v_norm`, which **no checkpoint tensor backs**
— only `q_norm` and `k_norm` are present in the safetensors inventory. A module in a
config-built tree is not evidence of a weight. Do not derive parameter counts from
`named_modules()`.

This settles the contested finding tracked in `/docs/verification/investigations/gemma-4-conformance_investigation.md`.

## Export safety

Both GGUF writers in this tier stay inside the limits of the `rust-harness` reader, which
accepts F32, F16, BF16, and Q8_0 tensors and int32, float32, bool, and string metadata arrays:

| Writer | Architecture | Metadata | Tensors |
|--------|--------------|----------|---------|
| `/interp/steering/export.py:33-57` | `controlvector` | one string, one uint32 | `direction.{layer}`, float32 |
| `/interp/probes/export.py:23-53` | `probe` | two strings, one uint32, two float32 | `probe.weights`, `probe.mu`, `probe.sigma`, float32 |

Neither writes an array-valued metadata key, and both write float32. The Rust reader at
`/rust-harness/runtime/src/model/probe.rs:41-58` demands exactly these types, and probe and
steering artifacts stay float32 regardless of the model's storage type — they are small, and
exactness in the direction vectors is worth more than their size.

The **model** GGUF path is a separate contract: those files are BF16, which the reader now
decodes losslessly, and what still refuses them is architectural rather than format-level. See
`/docs/rust-harness/gemma-4-ingestion.md`.

## Ecosystem gaps

- **No Gemma-Scope or Neuronpedia assets exist for Gemma 4.** `gemma2` is registered in the
  family table specifically as the anchor for that ecosystem (`families.py:8-11`); Gemma 4
  sparse autoencoders must be trained in-tree.
- **TransformerLens has no Gemma 4 support in its architecture registry**, so
  `HookedTransformer` aliases are not used (`families.py:14-15`). Hooks go through nnsight or
  plain torch. The reason is the registry rather than the library: `HookedTransformer`
  converts a fixed set of families, and Gemma 4 is not among them at our 3.5.1 pin. TL 3.x
  does ship `TransformerBridge`, which wraps arbitrary HF models including multimodal ones,
  so a future revisit would go through the bridge rather than waiting for an alias table
  entry. Current upstream is 3.7.0; the gap between that and our pin has not been surveyed.
- **`gemma4_unified_text` is not registered.** That is the 12B text `model_type`, deliberately
  refused rather than mapped to the flat template.

## Reproducing this document

The configuration and tensor facts regenerate on the workstation with no GPU and no network:

```bash
$ ./scripts/models/gemma4_header_transcript.sh
```

The module path requires loading the model; see
[The module path is composite](#the-module-path-is-composite).
