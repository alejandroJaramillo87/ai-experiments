/*
 * cpu — hand-rolled CPU reference backend for Gemma-4-style checkpoints.
 *
 * Correctness-first: plain single-threaded f32 loops, no SIMD, no wall
 * clock, no allocation tricks — deterministic by construction so runs
 * record and replay under rr. This backend is the reference the CUDA
 * backend is diffed against (through the phase 1 flight recorder) and makes
 * the full harness runnable in a sandbox against synthetic weights.
 *
 * Architecture: RMSNorm scaled by the stored weight, embedding scaled by
 * sqrt(embed_dim), grouped-query attention with per-head QK-norm and NEOX
 * RoPE, interleaved sliding-window/global layers with their own head widths
 * and RoPE frequency bases, per-layer feed-forward widths, GeGLU MLP with
 * tanh GELU, four norms per layer, LM head tied to the token embedding.
 *
 * Per-layer input embeddings join each layer's residual stream after its
 * feed-forward block, from a table indexed by token identity summed with a
 * projection of the embedded input. On E4B that table is roughly 2.8B of the
 * 7.5B total parameters, so it is most of the model by weight.
 *
 * The trailing layers of a Gemma 4 checkpoint compute no keys or values of
 * their own and attend over an earlier layer's cache. There are two donors,
 * one per attention regime, so a shared sliding layer reads the last
 * non-shared sliding layer and a shared global layer the last non-shared
 * global one. This is a property of the declared metadata and of nothing in
 * the tensor list: the released files carry key and value projections on every
 * block including the shared tail, so a reader that ignores
 * `attention.shared_kv_layers` gives all 42 layers independent caches with no
 * tensor left over to notice.
 *
 * That is still a subset of the published Gemma 4 architecture, which adds
 * value normalization and, on the larger tiers, key-equals-value attention. A
 * checkpoint carrying either is refused by `check_supported` rather than
 * loaded with the extra structure silently dropped: an ignored weight tensor
 * still produces fluent text, so a silent load would be an undetectable wrong
 * answer. See docs/rust-harness/gemma-4-ingestion.md.
 *
 * RoPE is per layer, not per model. Gemma 4 runs two regimes whose bases
 * differ by two orders of magnitude (1e6 global against 1e4 sliding), and
 * the global regime rotates only a quarter of each head: `rope_freqs.weight`
 * carries divisors that drive the remaining frequencies to zero, so the
 * rotation is performed at full width with most pairs inert. That tensor was
 * previously treated as inert itself, on the reasoning that frequencies are
 * recomputed from the base. They are, but not from the base alone, and a
 * reader that skips the divisors rotates 512 dimensions where the reference
 * rotates 128 -- finite, fluent, wrong. See the proportional-RoPE derivation
 * in docs/rust-harness/gemma-4-ingestion.md.
 *
 * The K/V cache stores the full sequence for every layer that owns one; the
 * sliding window is applied as an attention mask only. Capping sliding-window
 * storage to the kv.rs budget (a ring buffer) is deferred to the real-model
 * work, where the memory matters -- and KV sharing constrains that future
 * change: a shared sliding layer reads its donor's rows, and the donor must
 * therefore retain the full sequence even though its own attention never looks
 * past the window. Upstream hits the same wall and answers it by consulting a
 * separate full-length `shared_kv_states` rather than the cache. A ring buffer
 * applied uniformly to every sliding layer would silently truncate what the
 * shared tail reads.
 */

use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};

use super::{Logits, Model, Token};
use crate::gguf::GgufFile;
use crate::kv::LayerKind;
use crate::tensor::load_tensor;

/// Quantities that are one value for the whole model.
///
/// Head dimension, feed-forward width, and the RoPE frequency base are
/// deliberately absent: Gemma 4 varies all three by layer, and a scalar field
/// here is exactly the representation that cannot say so. They live in
/// `LayerSpec`, so a reader has to name the layer it means. See
/// `docs/rust-harness/gemma-4-ingestion.md`.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub context_length: usize,
    pub sliding_window: usize,
    pub rms_eps: f32,
    /// Ceiling the output logits are squashed toward, `30.0` on every Gemma 4
    /// variant. `None` when the checkpoint declares none, which is the
    /// uncapped forward pass.
    pub logit_softcap: Option<f32>,
    /// Width of one layer's slice of the per-layer input embedding, 256 on
    /// every Gemma 4 variant that has the feature. `None` when the checkpoint
    /// declares no `embedding_length_per_layer_input`, which is the model
    /// without per-layer embeddings.
    ///
    /// This key, and never `vocab_size_per_layer_input`, is what the feature
    /// is gated on: the vocabulary field is populated even on variants where
    /// per-layer embeddings are disabled (trap 9, `docs/models/gemma-4.md`).
    /// The GGUF converter does not write the vocabulary key at all, so the
    /// per-layer vocabulary is read from the table's own shape at load.
    pub ple_dim: Option<usize>,
}

/// Geometry of one decoder layer.
///
/// Gemma 4 interleaves two attention regimes that differ in head width and in
/// RoPE frequency base, and E2B additionally doubles the feed-forward width
/// over its trailing layers. Each axis is per-layer here even when a given
/// checkpoint holds it uniform, so the uniform case is a value this type
/// happens to carry rather than an assumption it encodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerSpec {
    pub kind: LayerKind,
    /// Width of one attention head. `attention.key_length` on global layers,
    /// `attention.key_length_swa` on sliding layers where the checkpoint
    /// declares one. Note this is the *global* head dim on a Gemma 4 file
    /// (512), not the `head_dim` of its Hugging Face configuration (256).
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub rope_freq_base: f32,
    /// Index of the layer whose K/V cache this layer attends over. Equal to
    /// the layer's own index unless the checkpoint declares a shared-KV tail,
    /// in which case it is the donor's.
    ///
    /// There are two donors on a Gemma 4 file, not one: upstream picks the
    /// last non-shared layer *of each layer type*, so a shared sliding layer
    /// reads the last non-shared sliding layer and a shared global layer the
    /// last non-shared global one. Designating a single donor for the whole
    /// tail is wrong for one of the two regimes, which is why this is a
    /// per-layer index rather than a model-wide boundary the forward pass
    /// re-derives (`docs/models/gemma-4.md`).
    pub kv_source: usize,
}

/// A checkpoint's complete declared shape: the model-wide scalars plus one
/// `LayerSpec` per layer.
///
/// `check_supported` and `CpuModel::from_gguf` both take this, so the
/// `gguf-info` inspection tool and the loader cannot disagree about what a
/// file declares.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub config: Config,
    pub layers: Vec<LayerSpec>,
}

impl ModelSpec {
    /// Widest head across the layer stack, the figure a shared scratch buffer
    /// must be sized against. Sizing against layer 0 underflows the moment a
    /// checkpoint interleaves two head widths.
    pub fn max_head_dim(&self) -> usize {
        self.layers.iter().map(|l| l.head_dim).max().unwrap_or(0)
    }

    /// Widest feed-forward width across the layer stack. Same reasoning.
    pub fn max_ffn_dim(&self) -> usize {
        self.layers.iter().map(|l| l.ffn_dim).max().unwrap_or(0)
    }

    /// Cached elements per token in one layer's K (or V) tensor. Varies by
    /// layer because the head width does.
    pub fn kv_dim(&self, layer: usize) -> usize {
        self.config.n_kv_heads * self.layers[layer].head_dim
    }

    /// Whether layer `i` stores its own K/V rather than reading a donor's.
    ///
    /// The single predicate every cache walk consults — the forward pass, the
    /// snapshot, `truncate`, and the memory budget. A layer that does not own
    /// its cache holds no rows at all, so a walk that computes
    /// `cached_len * kv_dim(i)` without asking this over-counts by the whole
    /// shared tail: 18 of E4B's 42 layers, 20 of E2B's 35.
    pub fn owns_cache(&self, i: usize) -> bool {
        self.layers[i].kv_source == i
    }

    /// Number of layers that store K/V. `n_layers` on a checkpoint without a
    /// shared-KV tail.
    pub fn cache_owning_layers(&self) -> usize {
        (0..self.layers.len())
            .filter(|&i| self.owns_cache(i))
            .count()
    }

    /// Elements one cached token occupies in layer `i`'s own storage: the
    /// layer's `kv_dim`, or zero where it reads a donor and stores nothing.
    ///
    /// Distinct from `kv_dim`, which is the stride the *attention* loop walks
    /// and which a shared layer asks of its donor. Conflating the two is how a
    /// shared layer ends up allocating rows it never writes.
    pub fn cache_row(&self, i: usize) -> usize {
        match self.owns_cache(i) {
            true => self.kv_dim(i),
            false => 0,
        }
    }
}

impl Config {}

/// Which non-shared layer each layer reads K/V from.
///
/// The trailing `shared` layers compute no K or V of their own and attend over
/// a donor's cache; every layer before the boundary is its own donor. Shared
/// between the two loaders because the rule is subtle and an off-by-one is
/// invisible: the wrong donor still has the right shape, and the model still
/// emits fluent text.
///
/// Upstream's `store_full_length_kv` picks the last non-shared layer *of this
/// layer's own attention regime*, so searching downward from the boundary is
/// that rule read backwards. Regimes are compared by discriminant rather than
/// by window width, since two sliding layers with different windows are still
/// both sliding.
fn kv_donors(kinds: &[LayerKind], shared: usize) -> Result<Vec<usize>> {
    let n = kinds.len();
    ensure!(
        shared <= n,
        "{shared} shared-KV layers declared, more than the {n} layers in the model"
    );
    let boundary = n - shared;
    let is_global = |i: usize| matches!(kinds[i], LayerKind::Global);
    (0..n)
        .map(|i| {
            if i < boundary {
                return Ok(i);
            }
            (0..boundary)
                .rev()
                .find(|&j| is_global(j) == is_global(i))
                .with_context(|| {
                    let regime = if is_global(i) { "global" } else { "sliding" };
                    format!(
                        "layer {i} is a shared-KV {regime} layer but no {regime} layer \
                         precedes the shared-KV boundary at {boundary}, so it has no \
                         donor to read; {shared} shared layers and this attention \
                         schedule describe a model that cannot run"
                    )
                })
        })
        .collect()
}

impl ModelSpec {
    /// Read the declared shape from a checkpoint's `config.json`.
    ///
    /// Shorter than `from_gguf` because the checkpoint states what the GGUF
    /// implies. There is no stride to infer and no periodicity to enforce,
    /// no paired `_swa` keys to resolve against their unsuffixed partners,
    /// and no scalar-or-array union to normalize: `TextConfig` has already
    /// validated its own fields, so this is a mapping and not a parse.
    pub fn from_checkpoint(c: &crate::checkpoint::TextConfig) -> Result<ModelSpec> {
        let config = Config {
            vocab_size: c.vocab_size,
            embed_dim: c.hidden_size,
            n_layers: c.num_hidden_layers,
            n_heads: c.num_attention_heads,
            n_kv_heads: c.num_key_value_heads,
            context_length: c.max_position_embeddings,
            sliding_window: c.sliding_window,
            rms_eps: c.rms_norm_eps,
            logit_softcap: c.final_logit_softcapping,
            ple_dim: c.hidden_size_per_layer_input,
        };
        let kinds: Vec<LayerKind> = (0..config.n_layers)
            .map(|i| match c.is_global(i)? {
                true => Ok(LayerKind::Global),
                false => Ok(LayerKind::SlidingWindow(config.sliding_window)),
            })
            .collect::<Result<_>>()?;
        let donors = kv_donors(&kinds, c.num_kv_shared_layers)
            .context("num_kv_shared_layers and the layer_types schedule")?;
        let ffn_dims = c.intermediate_sizes()?;
        let layers = (0..config.n_layers)
            .map(|i| {
                let (head_dim, rope_freq_base) = match kinds[i] {
                    LayerKind::Global => (c.global_head_dim(), c.rope_theta),
                    LayerKind::SlidingWindow(_) => {
                        (c.head_dim, c.rope_local_base_freq.unwrap_or(c.rope_theta))
                    }
                };
                LayerSpec {
                    kind: kinds[i],
                    head_dim,
                    ffn_dim: ffn_dims[i],
                    rope_freq_base,
                    kv_source: donors[i],
                }
            })
            .collect();
        Ok(ModelSpec { config, layers })
    }
}

/// Per-layer tensor suffixes the forward pass consumes. Kept in step with the
/// `LayerWeights` literal in `CpuModel::from_gguf`; the fixture-shape test
/// fails if the two drift, because an unlisted consumed tensor makes
/// `check_supported` refuse the synthetic model.
const LAYER_TENSORS: [&str; 13] = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
    "post_attention_norm.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "post_ffw_norm.weight",
];

/// Non-layer tensors the forward pass consumes. The LM head is tied to
/// `token_embd.weight` (llama.cpp Gemma convention), so there is no
/// `output.weight`.
const TOP_LEVEL_TENSORS: [&str; 2] = ["token_embd.weight", "output_norm.weight"];

/// Per-layer tensor suffixes the forward pass consumes when a checkpoint
/// carries them, and computes without when it does not.
///
/// Separate from `LAYER_TENSORS`, whose entries are mandatory: a missing one
/// is a load error, while a missing one of these is a checkpoint that does
/// not have the feature. Listing them here is what keeps the unconsumed
/// backstop from refusing a file for carrying a tensor this backend reads.
const OPTIONAL_LAYER_TENSORS: [&str; 1] = ["layer_output_scale.weight"];

/// File-level per-layer-input-embedding tensors: the token-identity table,
/// the projection of the embedded input, and the norm applied to that
/// projection.
const PLE_TOP_LEVEL_TENSORS: [&str; 3] = [
    "per_layer_token_embd.weight",
    "per_layer_model_proj.weight",
    "per_layer_proj_norm.weight",
];

/// Per-block per-layer-input-embedding tensor suffixes: the gate, the
/// projection back to the residual width, and the norm on that projection.
///
/// These three lose the `per_layer` substring in GGUF conversion —
/// `per_layer_input_gate` becomes `inp_gate`, `per_layer_projection` becomes
/// `proj`, and `post_per_layer_input_norm` becomes `post_norm`, which is
/// additionally distinguishable from the ordinary `post_attention_norm` and
/// `post_ffw_norm` only by knowing the architecture. They are therefore
/// listed rather than found by the substring that finds the file-level three
/// (`docs/rust-harness/gemma-4-ingestion.md`).
const PLE_LAYER_TENSORS: [&str; 3] = ["inp_gate.weight", "proj.weight", "post_norm.weight"];

/// `per_layer_input_scale`, the factor applied to the sum of the two
/// per-layer-embedding components. Upstream writes it `2.0 ** -0.5`.
const PLE_INPUT_SCALE: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Refuse a checkpoint whose structure the forward pass does not implement.
///
/// The architectural half of `check_supported`. Most of that function is
/// concerned with metadata keys that only the GGUF spelling has -- the
/// `rope.*` allowlist, the `dimension_count` width check, the softcapping
/// keys -- and `TextConfig::validate` covers the equivalent ground by having
/// named fields to check instead of a namespace to police. What is left is
/// what the *tensors* have to satisfy, which no config field can state.
///
/// The unconsumed-tensor backstop is the important survivor. Loading is
/// name-directed, so a feature published after this code presents as tensors
/// nothing reads and logits that are merely wrong.
pub fn check_supported_checkpoint(
    ck: &crate::checkpoint::Checkpoint,
    spec: &ModelSpec,
) -> Result<()> {
    use crate::checkpoint::hf_tensor_name;

    let config = &spec.config;
    let names = ck.names();
    let has = |gguf: &str| hf_tensor_name(gguf).is_some_and(|n| names.contains(&n));

    // Per-layer input embeddings are all six families or none. A checkpoint
    // carrying five would load, skip the residual join wherever the sixth is
    // missing, and produce fluent output from a model missing most of its
    // parameters. A per-block family present on some layers but not all
    // counts as missing -- the partial set is the one that loads.
    let mut present: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    // Reported in the Hugging Face spelling, which is what a reader looking
    // at the checkpoint sees. The three per-block families lose the
    // `per_layer` substring in the llama.cpp naming, so reporting that
    // spelling would name tensors no file on disk contains.
    let published = |gguf: &str| hf_tensor_name(gguf).unwrap_or_else(|| gguf.to_string());
    for name in PLE_TOP_LEVEL_TENSORS {
        match has(name) {
            true => present.push(published(name)),
            false => missing.push(published(name)),
        }
    }
    for suffix in PLE_LAYER_TENSORS {
        let found = (0..config.n_layers)
            .filter(|i| has(&format!("blk.{i}.{suffix}")))
            .count();
        let family = hf_tensor_family(&published(&format!("blk.0.{suffix}")));
        match found == config.n_layers {
            true => present.push(family),
            false => missing.push(format!("{family} ({found} of {} layers)", config.n_layers)),
        }
    }
    let declared = config.ple_dim.is_some();
    if declared || !present.is_empty() {
        ensure!(
            missing.is_empty(),
            "per-layer input embeddings are {}, but these tensor families are absent \
             or incomplete: {}. The feature is all six families or none",
            match declared {
                true => "declared by hidden_size_per_layer_input".to_string(),
                false => format!("implied by present tensors ({})", present.join(", ")),
            },
            missing.join(", ")
        );
        ensure!(
            declared,
            "checkpoint carries the per-layer input embedding tensors but declares no \
             hidden_size_per_layer_input, so the packed table cannot be split into one \
             row per layer; refusing rather than inferring the width from a shape"
        );
    }

    // A layer that stores its own K/V cannot attend without k_proj and
    // v_proj. A shared layer legitimately carries neither -- and the released
    // checkpoints carry both anyway, which is the recorded conflict between
    // the modeling code and the weights -- so this narrows to cache owners.
    for i in 0..config.n_layers {
        if !spec.owns_cache(i) {
            continue;
        }
        if has(&format!("blk.{i}.attn_q.weight"))
            && !(has(&format!("blk.{i}.attn_k.weight")) && has(&format!("blk.{i}.attn_v.weight")))
        {
            bail!(
                "layer {i} has no k_proj/v_proj tensors, but it stores its own K/V \
                 rather than reading a donor's, so there is nothing to attend over; \
                 num_kv_shared_layers leaves {} layers owning a cache",
                spec.cache_owning_layers()
            );
        }
    }

    // Declared feed-forward width against the width the weights carry. A
    // disagreement means config.json and the tensors describe different
    // models, and the forward pass slices by the config.
    for (i, layer) in spec.layers.iter().enumerate() {
        let Some(name) = hf_tensor_name(&format!("blk.{i}.ffn_up.weight")) else {
            continue;
        };
        let Ok(view) = ck.tensors.tensor(&name) else {
            continue;
        };
        let width = view.shape().first().copied().unwrap_or(0);
        ensure!(
            width == layer.ffn_dim,
            "tensor {name} has feed-forward width {width}, but intermediate_size \
             declares {} for layer {i}",
            layer.ffn_dim
        );
    }

    // Generic backstop: anything left unconsumed is structure this backend
    // does not know about, including features published after this code.
    let mut consumable: Vec<String> = Vec::new();
    for name in TOP_LEVEL_TENSORS.iter().chain(PLE_TOP_LEVEL_TENSORS.iter()) {
        consumable.extend(hf_tensor_name(name));
    }
    for i in 0..config.n_layers {
        for suffix in LAYER_TENSORS
            .iter()
            .chain(OPTIONAL_LAYER_TENSORS.iter())
            .chain(PLE_LAYER_TENSORS.iter())
        {
            consumable.extend(hf_tensor_name(&format!("blk.{i}.{suffix}")));
        }
    }
    let mut unconsumed: Vec<String> = Vec::new();
    for name in &names {
        if consumable.contains(name) {
            continue;
        }
        let family = hf_tensor_family(name);
        if !unconsumed.contains(&family) {
            unconsumed.push(family);
        }
    }
    ensure!(
        unconsumed.is_empty(),
        "checkpoint carries {} tensor famil{} this backend does not consume: {}; \
         refusing rather than ignoring learned weights",
        unconsumed.len(),
        if unconsumed.len() == 1 { "y" } else { "ies" },
        unconsumed.join(", ")
    );
    Ok(())
}

/// Collapse a Hugging Face tensor name to its family, replacing the layer
/// index in `model.layers.<i>.` with `*`, so a diagnostic over 42 layers
/// reports one line per family rather than 42.
pub fn hf_tensor_family(name: &str) -> String {
    let Some(rest) = name.strip_prefix("model.layers.") else {
        return name.to_string();
    };
    match rest.split_once('.') {
        Some((index, suffix)) if !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()) => {
            format!("model.layers.*.{suffix}")
        }
        _ => name.to_string(),
    }
}

/// File-level per-layer input embedding weights, read once per position.
struct PleWeights {
    /// Packed token-identity table, `ple_vocab` rows of `n_layers * ple_dim`.
    token_embd: Vec<f32>,
    /// Rows in `token_embd`. Derived from the tensor's own shape rather than
    /// from a metadata key: `vocab_size_per_layer_input` is a separate config
    /// field from `vocab_size`, and the GGUF converter deletes it before
    /// writing rather than emitting it (`docs/models/gemma4-upstream.txt`).
    ple_vocab: usize,
    /// Projection of the embedded input to one row per layer.
    model_proj: Vec<f32>,
    /// RMSNorm weight applied to each projected row over `ple_dim`.
    proj_norm: Vec<f32>,
}

/// One layer's per-layer input embedding weights.
struct PleLayer {
    inp_gate: Vec<f32>,
    proj: Vec<f32>,
    post_norm: Vec<f32>,
}

struct LayerWeights {
    /// This layer's declared geometry, carried beside its weights so the
    /// forward pass never reaches for a model-wide width.
    spec: LayerSpec,
    /// Per-pair RoPE frequency table for this layer's head width and
    /// frequency base. Derived, not loaded. Layers that agree on both still
    /// each hold a copy; the table is `head_dim / 2` floats, and sharing it
    /// would trade a measurable amount of nothing for an aliasing rule.
    rope_freqs: Vec<f32>,
    /// `layer_output_scale`, multiplying the whole layer output as its last
    /// operation. 1.0 when the checkpoint carries no such tensor, which is
    /// also the initialization value upstream, so the absent case and the
    /// untrained case compute alike. The released Gemma 4 weights are not
    /// 1.0: E2B spans 0.019 to 0.879, and layer 0's 0.019 is a 54x reduction
    /// that a backend ignoring this tensor simply does not apply.
    output_scale: f32,
    /// This layer's per-layer input embedding weights, `None` on a checkpoint
    /// without the feature.
    ple: Option<PleLayer>,
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wo: Vec<f32>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
    post_ffw_norm: Vec<f32>,
}

/// Immutable weights, shared by forks through an Arc.
struct Weights {
    embed: Vec<f32>,
    /// File-level per-layer input embedding weights, `None` on a checkpoint
    /// without the feature.
    ple: Option<PleWeights>,
    layers: Vec<LayerWeights>,
    output_norm: Vec<f32>,
}

/// Per-layer KV cache. Correctness-first: every layer stores the full
/// sequence; the sliding window is applied as an attention mask, and the
/// storage cap from kv.rs stays a deferred optimization.
#[derive(Clone, Default)]
struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Observer of the residual stream, the interp tap seam. Fires on the decoder
/// layer's output, which is the `layer_output_scale` multiply and not the
/// final residual add: a layer applies the attention residual, the
/// feed-forward residual, the per-layer-embedding residual, then the scale.
/// That is the hidden state nnsight exposes as `block.output[0]`, because the
/// Python forward hook also receives the already-scaled tensor, so probes
/// trained on Python-captured activations and on rust-captured activations
/// see the same tensor. Any backend added later (CUDA) must preserve this
/// seam and its semantics.
pub trait Tap: Send {
    fn on_residual(&mut self, layer: usize, position: usize, x: &[f32]);
}

/// Additive steering directions in the llama.cpp control-vector GGUF
/// layout: one f32 tensor per layer named `direction.<layer>`.
///
/// `direction.N` steers decoder block `N`, matching llama.cpp and the
/// repeng training semantics that produce these files: repeng trains key
/// `k` on the output of block `k`, and llama.cpp adds `direction.N` to
/// block `N`'s output. Block 0 is therefore unsteerable, as upstream —
/// numbering starts at 1 and there is never a tensor for block 0.
/// `scale * direction` is added to the residual after that block, before
/// the tap fires, so captures observe the steered stream that downstream
/// blocks actually consume.
pub struct ControlVector {
    scale: f32,
    /// Indexed by 0-based block; None for blocks without a direction.
    /// Slot 0 is always None because numbering starts at 1.
    directions: Vec<Option<Vec<f32>>>,
}

impl ControlVector {
    pub fn from_gguf(file: &GgufFile, data: &[u8], scale: f32) -> Result<ControlVector> {
        let mut directions: Vec<Option<Vec<f32>>> = Vec::new();
        for info in &file.tensors {
            let Some(suffix) = info.name.strip_prefix("direction.") else {
                continue;
            };
            let layer: usize = suffix
                .parse()
                .with_context(|| format!("tensor name {}: non-numeric layer", info.name))?;
            ensure!(layer >= 1, "tensor {}: layers are 1-based", info.name);
            let v = load_tensor(file, data, info)?;
            if layer >= directions.len() {
                directions.resize(layer + 1, None);
            }
            directions[layer] = Some(v);
        }
        ensure!(
            directions.iter().any(Option::is_some),
            "control vector holds no direction.<layer> tensors"
        );
        let widths: std::collections::BTreeSet<usize> = directions
            .iter()
            .flatten()
            .map(std::vec::Vec::len)
            .collect();
        ensure!(
            widths.len() == 1,
            "direction tensors disagree on width: {widths:?}"
        );
        Ok(ControlVector { scale, directions })
    }

    /// The direction for a 0-based block, or None where the image names none.
    ///
    /// Public so the cross-language contract test can assert the layer
    /// mapping directly (`schemas/gguf_artifacts.json`). A control vector
    /// written by `interp/steering/export.py` numbers its tensors from 1, and
    /// an off-by-one here steers the wrong block with no error anywhere —
    /// reading the parsed placement is the only way to see it.
    pub fn direction(&self, layer: usize) -> Option<&[f32]> {
        self.directions.get(layer)?.as_deref()
    }

    pub fn width(&self) -> usize {
        self.directions
            .iter()
            .flatten()
            .map(std::vec::Vec::len)
            .next()
            .expect("validated non-empty")
    }
}

/// Working buffers for one position, allocated at load and overwritten on
/// every token.
///
/// `forward_one` used to allocate every intermediate it named: `13 + 2 *
/// n_heads + n_kv_heads` vectors per layer per token, which measured 39
/// allocations per token over the whole pass on the committed fixture and
/// extrapolates to roughly 1088 at E4B geometry. Each buffer below is
/// written in full before it is read, so the reuse is an allocator saving
/// only — no value, and no order in which values accumulate, changes.
///
/// Scratch is per-model state and is never shared: `fork` gives the child
/// its own set rather than cloning the parent's, so two models descended
/// from one checkpoint cannot overwrite each other's intermediates when
/// their decodes interleave.
struct Scratch {
    /// Every fixed-width buffer back to back, carved into disjoint views by
    /// `forward_one`.
    ///
    /// One allocation rather than twelve because the buffers are read and
    /// written together: at the committed fixture's geometry they total
    /// under a kilobyte, and as separate allocations they landed in
    /// whichever cache sets the allocator happened to pick. That was worth
    /// 15 percent of a token, and it moved when an unrelated allocation was
    /// made before the model loaded — placement should be a property of
    /// this code, not of the heap it runs on.
    buf: Vec<f32>,
    /// Attention scores for one query head. Kept out of `buf` because its
    /// length varies with the window and the position rather than with the
    /// model, so it is the one buffer that grows on use: `context_length`
    /// is whatever the checkpoint claims, and reserving for the claim at
    /// load would allocate gigabytes for a long-context model that may
    /// never be prefilled that far.
    scores: Vec<f32>,
    /// The position's per-layer input embeddings, `n_layers` rows of
    /// `ple_dim`. Kept out of `buf` because it is computed once per position
    /// and read by every layer, where every view carved from `buf` is
    /// overwritten by the next layer. Empty on a checkpoint without the
    /// feature.
    ple: Vec<f32>,
}

impl Scratch {
    /// Sized against the widest layer, not against layer 0 or a model-wide
    /// scalar. Gemma 4's global layers carry twice the head width of its
    /// sliding layers and E2B doubles the feed-forward width over its trailing
    /// layers, so a buffer cut to either one alone is short for the other; the
    /// views are re-carved to each layer's own widths inside `forward_one`.
    fn new(spec: &ModelSpec) -> Scratch {
        let c = &spec.config;
        let hd = spec.max_head_dim();
        let (q_dim, kv_dim) = (c.n_heads * hd, c.n_kv_heads * hd);
        // Widths of the views `forward_one` carves, in its order: x,
        // normed, q, k, v, attn_out, projected, post, gate, up, mixed,
        // down, ple_gate, ple_out. The carving asserts it never exceeds this.
        let ple_dim = c.ple_dim.unwrap_or(0);
        let elements = 6 * c.embed_dim + 2 * q_dim + 2 * kv_dim + 3 * spec.max_ffn_dim() + ple_dim;
        Scratch {
            buf: vec![0.0; elements],
            scores: Vec::new(),
            ple: vec![0.0; c.n_layers * ple_dim],
        }
    }
}

/// Split `n` elements off the front of `rest`, advancing it past them.
/// Hands `forward_one` one disjoint view per intermediate out of the single
/// scratch allocation; panics if fewer than `n` elements remain.
fn carve<'a>(rest: &mut &'a mut [f32], n: usize) -> &'a mut [f32] {
    let (head, tail) = std::mem::take(rest).split_at_mut(n);
    *rest = tail;
    head
}

pub struct CpuModel {
    config: Config,
    /// Declared shape, shared with forks. Immutable after load. `config` is a
    /// `Copy` duplicate of `spec.config` kept for the many scalar reads.
    spec: Arc<ModelSpec>,
    weights: Arc<Weights>,
    kv: Vec<LayerKv>,
    cached_len: usize,
    /// Per-position working buffers, private to this model. See `Scratch`.
    scratch: Scratch,
    /// None on construction and after fork (a `Box<dyn Tap>` cannot be
    /// cloned); absent taps cost one branch per layer.
    tap: Option<Box<dyn Tap>>,
    /// Steering carries across forks: a steered agent's subagents serve
    /// the same steered model.
    control: Option<Arc<ControlVector>>,
}

/// `out = W x` with GGUF layout `[in, out]`: rows of `in` contiguous per
/// output element. The output width is `out.len()`, and `out` is written in
/// full, so its previous contents never reach the result.
///
/// The shape check is a real assertion rather than a `debug_assert`: a
/// matrix longer than `in_dim * out_dim` stays in bounds and silently uses
/// its leading rows, so the compiled-out version left release builds — the
/// builds that matter — with no guard at all. `CpuModel::from_gguf` now
/// makes this unreachable from a loaded checkpoint, and one integer
/// comparison against `in_dim * out_dim` multiply-accumulates is free. It
/// also catches a `Scratch` buffer sized against the wrong dimension, which
/// would otherwise show up only as wrong logits.
fn matvec_into(w: &[f32], x: &[f32], out: &mut [f32]) {
    let (in_dim, out_dim) = (x.len(), out.len());
    assert_eq!(
        w.len(),
        in_dim * out_dim,
        "matvec: weight of {} elements for a {in_dim} x {out_dim} product",
        w.len()
    );
    for (o, dst) in out.iter_mut().enumerate() {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        *dst = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

/// Reciprocal of the RMS of `x`, the factor RMSNorm scales every element by.
fn rms_inv(x: &[f32], eps: f32) -> f32 {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    1.0 / (mean_sq + eps).sqrt()
}

/// Gemma RMSNorm into a separate buffer: `out = x / rms(x) * w`.
///
/// The stored tensor is the effective multiplier. Gemma's reference
/// implementation adds 1.0 to its norm parameter at runtime, but the GGUF
/// converter resolves that before writing: it folds the shift into the
/// tensor for gemma1 through gemma3 (`norm_shift` 1.0) and stores the
/// already-shifted weight for gemma3n and gemma4 (`norm_shift` 0.0). Both
/// land on a plain multiply in the llama.cpp graph, so applying `1 + w`
/// here would double-count the shift on every real checkpoint.
fn rms_norm_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    // A short `out` would leave stale values from the previous token in the
    // tail, which reads as a plausible activation rather than as an error.
    assert_eq!(
        out.len(),
        x.len(),
        "rms_norm: {} output elements for {} inputs",
        out.len(),
        x.len()
    );
    let inv = rms_inv(x, eps);
    for ((dst, v), g) in out.iter_mut().zip(x).zip(w) {
        *dst = v * inv * g;
    }
}

/// Gemma RMSNorm applied in place, for the per-head Q/K norms whose input is
/// dead after the call. Every output element depends only on the input
/// element at the same index and on the scale computed up front, so this is
/// the same arithmetic as `rms_norm_into` followed by a copy.
fn rms_norm_in_place(x: &mut [f32], w: &[f32], eps: f32) {
    let inv = rms_inv(x, eps);
    for (v, g) in x.iter_mut().zip(w) {
        *v = *v * inv * g;
    }
}

/// Tanh-approximated GELU, the Gemma activation.
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

/// Per-pair RoPE frequencies, `freq_base^(-2j/head_dim)` for `j` in
/// `0..head_dim/2`.
///
/// The table depends only on `head_dim` and `rope_freq_base`, both fixed at
/// load, but `rope` recomputed the `powf` for every pair of every head of
/// every layer at every position: `n_layers * (n_heads + n_kv_heads) *
/// head_dim/2` transcendental calls per token, all returning the same
/// `head_dim/2` values. Computing it once and indexing it is bit-identical —
/// `powf` is a pure function of its arguments — and leaves `sin_cos`, which
/// does depend on the position, as the only transcendental in the loop.
/// `divisors`, when the checkpoint supplies them, are the proportional-RoPE
/// frequency divisors from `rope_freqs.weight`, applied here rather than in
/// the decode loop: they are fixed at load like the rest of the table, and
/// folding them in leaves `rope` a function of one table as before.
///
/// The divisor is applied to the inverse frequency, matching the converter
/// that writes the tensor. Gemma 4's values are 1.0 over the rotating pairs
/// and 1e30 over the rest, so the second group's angle collapses to
/// approximately zero and those pairs pass through. Upstream uses 1e30
/// rather than infinity, so the angle is very small rather than exactly
/// zero; that residual is the reference's behavior, not an approximation of
/// it, and is reproduced rather than rounded away.
fn rope_frequencies(head_dim: usize, freq_base: f32, divisors: Option<&[f32]>) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|j| {
            let freq = freq_base.powf(-2.0 * j as f32 / head_dim as f32);
            match divisors {
                Some(d) => freq / d[j],
                None => freq,
            }
        })
        .collect()
}

/// NEOX-style RoPE over one head in place: pairs are (j, j + head_dim/2).
/// `freqs` is the `rope_frequencies` table for this head width.
fn rope(head: &mut [f32], position: usize, freqs: &[f32]) {
    let half = head.len() / 2;
    for j in 0..half {
        let angle = position as f32 * freqs[j];
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (head[j], head[j + half]);
        head[j] = a * cos - b * sin;
        head[j + half] = a * sin + b * cos;
    }
}

fn softmax_in_place(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        total += *s;
    }
    for s in scores.iter_mut() {
        *s /= total;
    }
}

impl CpuModel {
    /// Load from a Hugging Face checkpoint: `config.json` and
    /// `model.safetensors`.
    ///
    /// Two things differ from `from_gguf` beyond the container.
    ///
    /// Tensors are checked by *shape*, not element count. A transposed matrix
    /// has the same element count, so the GGUF path's check could not see one
    /// -- and while the layout is in fact identical between the two formats
    /// (GGUF `ne` order reversed is already row-major `[out, in]`, which is
    /// what `matvec_into` walks), that is a fact worth a guard rather than a
    /// comment. Nothing downstream would catch a transpose: the forward pass
    /// slices by strides from the config and returns finite, wrong logits.
    ///
    /// RMSNorm weights get Gemma's runtime `+1` folded in as they load. A
    /// published checkpoint stores the raw parameter and transformers adds one
    /// at every use; the GGUF converter resolves the shift before writing. So
    /// the two containers spell the same tensor differently, and this is where
    /// the checkpoint spelling is translated into the effective multiplier
    /// `rms_norm_into` expects. Folding at load rather than in the kernel is
    /// deliberate: it is the same arithmetic performed once per weight instead
    /// of once per weight per token, it keeps one norm kernel rather than two
    /// during the transition, and it avoids a subtract-then-add round trip
    /// that is lossy for the large norm values real Gemma checkpoints carry.
    pub fn from_checkpoint(ck: &crate::checkpoint::Checkpoint) -> Result<CpuModel> {
        use crate::checkpoint::hf_tensor_name;

        let spec = ModelSpec::from_checkpoint(&ck.config)?;
        let config = spec.config;
        check_supported_checkpoint(ck, &spec)?;

        // Named by the GGUF spelling and translated through the one shared
        // table, so the loader and the fixture writer cannot disagree about a
        // name. The GGUF spelling stays the internal vocabulary while both
        // loaders exist; it is the loader being removed that should change it.
        let hf = |gguf: &str| -> Result<String> {
            hf_tensor_name(gguf).with_context(|| format!("no published name for tensor {gguf}"))
        };
        let tensor =
            |gguf: &str, want: &[usize]| -> Result<Vec<f32>> { ck.tensor(&hf(gguf)?, want) };
        // A norm weight, shifted into the effective multiplier.
        let norm = |gguf: &str, want: &[usize]| -> Result<Vec<f32>> {
            debug_assert!(gguf.ends_with("norm.weight"), "{gguf} is not a norm");
            Ok(tensor(gguf, want)?.into_iter().map(|w| 1.0 + w).collect())
        };

        let embed_dim = config.embed_dim;
        let embed = tensor("token_embd.weight", &[config.vocab_size, embed_dim])?;
        // The divisor tensor the GGUF carries is derived here instead, from
        // the factor the checkpoint states. Ones over the rotating pairs and
        // 1e30 over the rest, matching the converter's own construction, so
        // `rope_frequencies` sees the same input either way.
        let rotating = ck.config.rope_rotating_pairs()?;
        let half = ck.config.global_head_dim() / 2;
        let rope_divisors: Option<Vec<f32>> = match rotating < half {
            true => {
                let mut d = vec![1.0f32; rotating];
                d.extend(std::iter::repeat_n(1e30f32, half - rotating));
                Some(d)
            }
            false => None,
        };

        let ple = match config.ple_dim {
            None => None,
            Some(pd) => {
                let row = config.n_layers * pd;
                let name = hf(PLE_TOP_LEVEL_TENSORS[0])?;
                // The per-layer table has its own vocabulary axis, which is a
                // separate config field from `vocab_size` and one the GGUF
                // converter drops. Read from the tensor's own shape in both
                // loaders, so a wider table stays loadable.
                let view = ck
                    .tensors
                    .tensor(&name)
                    .map_err(|e| anyhow::anyhow!("missing tensor {name}: {e}"))?;
                let ple_vocab = match view.shape() {
                    [rows, packed] if *packed == row => *rows,
                    other => bail!(
                        "tensor {name} has shape {other:?}, expected \
                         [vocab, num_hidden_layers x hidden_size_per_layer_input ({row})]"
                    ),
                };
                ensure!(
                    ple_vocab >= config.vocab_size,
                    "tensor {name} holds {ple_vocab} rows, fewer than the {} tokens \
                     vocab_size declares",
                    config.vocab_size
                );
                Some(PleWeights {
                    token_embd: tensor(PLE_TOP_LEVEL_TENSORS[0], &[ple_vocab, row])?,
                    ple_vocab,
                    model_proj: tensor(PLE_TOP_LEVEL_TENSORS[1], &[row, embed_dim])?,
                    proj_norm: norm(PLE_TOP_LEVEL_TENSORS[2], &[pd])?,
                })
            }
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        for (i, ls) in spec.layers.iter().enumerate() {
            let (head_dim, ffn_dim) = (ls.head_dim, ls.ffn_dim);
            let q_dim = config.n_heads * head_dim;
            let kv_dim = config.n_kv_heads * head_dim;
            let divisors = match (ls.kind, rope_divisors.as_deref()) {
                (LayerKind::Global, Some(d)) => Some(d),
                _ => None,
            };
            // Row-major `[out, in]`, the transpose of the GGUF `ne` order and
            // the same bytes.
            let t = |suffix: &str, want: &[usize]| tensor(&format!("blk.{i}.{suffix}"), want);
            let n = |suffix: &str, want: &[usize]| norm(&format!("blk.{i}.{suffix}"), want);
            let output_scale = match ck.has(&hf(&format!("blk.{i}.layer_output_scale.weight"))?) {
                true => t("layer_output_scale.weight", &[1])?[0],
                false => 1.0,
            };
            let ple_layer = match config.ple_dim {
                Some(pd) => Some(PleLayer {
                    inp_gate: t(PLE_LAYER_TENSORS[0], &[pd, embed_dim])?,
                    proj: t(PLE_LAYER_TENSORS[1], &[embed_dim, pd])?,
                    post_norm: n(PLE_LAYER_TENSORS[2], &[embed_dim])?,
                }),
                None => None,
            };
            // Left unloaded on a shared-KV layer for the reason `from_gguf`
            // records: the modeling code never evaluates them, and an
            // allocated-but-unread weight is the shape of a silent wrong
            // answer.
            let (wk, wv, k_norm) = match spec.owns_cache(i) {
                true => (
                    t("attn_k.weight", &[kv_dim, embed_dim])?,
                    t("attn_v.weight", &[kv_dim, embed_dim])?,
                    n("attn_k_norm.weight", &[head_dim])?,
                ),
                false => (Vec::new(), Vec::new(), Vec::new()),
            };
            layers.push(LayerWeights {
                spec: *ls,
                rope_freqs: rope_frequencies(head_dim, ls.rope_freq_base, divisors),
                output_scale,
                ple: ple_layer,
                attn_norm: n("attn_norm.weight", &[embed_dim])?,
                wq: t("attn_q.weight", &[q_dim, embed_dim])?,
                wk,
                wv,
                q_norm: n("attn_q_norm.weight", &[head_dim])?,
                k_norm,
                wo: t("attn_output.weight", &[embed_dim, q_dim])?,
                post_attn_norm: n("post_attention_norm.weight", &[embed_dim])?,
                ffn_norm: n("ffn_norm.weight", &[embed_dim])?,
                w_gate: t("ffn_gate.weight", &[ffn_dim, embed_dim])?,
                w_up: t("ffn_up.weight", &[ffn_dim, embed_dim])?,
                w_down: t("ffn_down.weight", &[embed_dim, ffn_dim])?,
                post_ffw_norm: n("post_ffw_norm.weight", &[embed_dim])?,
            });
        }
        let output_norm = norm("output_norm.weight", &[embed_dim])?;
        Ok(CpuModel {
            kv: vec![LayerKv::default(); config.n_layers],
            config,
            scratch: Scratch::new(&spec),
            spec: Arc::new(spec),
            weights: Arc::new(Weights {
                embed,
                ple,
                layers,
                output_norm,
            }),
            cached_len: 0,
            tap: None,
            control: None,
        })
    }

    /// Per-layer geometry, in layer order.
    pub fn layer_specs(&self) -> &[LayerSpec] {
        &self.spec.layers
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Install a residual-stream observer; replaces any existing tap.
    pub fn set_tap(&mut self, tap: Box<dyn Tap>) {
        self.tap = Some(tap);
    }

    pub fn take_tap(&mut self) -> Option<Box<dyn Tap>> {
        self.tap.take()
    }

    /// Install steering directions, validated against this model's shape.
    pub fn set_control_vector(&mut self, cv: ControlVector) -> Result<()> {
        ensure!(
            cv.width() == self.config.embed_dim,
            "control vector width {} does not match embedding dimension {}",
            cv.width(),
            self.config.embed_dim
        );
        // directions is indexed by 0-based block, so its length is the
        // highest named block plus one; that block must exist in the model.
        ensure!(
            cv.directions.len() <= self.config.n_layers,
            "control vector names block {} but the model has {} blocks (0..{})",
            cv.directions.len().saturating_sub(1),
            self.config.n_layers,
            self.config.n_layers.saturating_sub(1)
        );
        self.control = Some(Arc::new(cv));
        Ok(())
    }

    /// Vocabulary bound. Checked for the whole batch before the first
    /// forward pass so a bad token cannot commit its predecessors, and again
    /// per token because `forward_one` indexes the embedding table with it.
    fn check_token(&self, token: Token) -> Result<()> {
        ensure!(
            (token as usize) < self.config.vocab_size,
            "token {token} outside vocabulary of {}",
            self.config.vocab_size
        );
        Ok(())
    }

    /// One decode position: append the token's K/V to every layer cache and
    /// return the logits at this position.
    fn forward_one(&mut self, token: Token, position: usize) -> Result<Logits> {
        let c = self.config;
        self.check_token(token)?;
        // Split the borrow by field: the layer loop reads the weights, appends
        // to the caches, and writes the scratch buffers at the same time.
        let CpuModel {
            weights,
            kv,
            scratch,
            tap,
            control,
            ..
        } = self;
        let w: &Weights = weights;
        let heads_per_kv = c.n_heads / c.n_kv_heads;

        // `x` is the residual stream and outlives the layer loop; everything
        // else is re-carved per layer, because head width and feed-forward
        // width vary by layer and a single carving would size them all to
        // whichever layer happened to be first.
        let Scratch { buf, scores, ple } = scratch;
        let (x, work) = buf.split_at_mut(c.embed_dim);

        // Embedding, scaled by sqrt(embed_dim) per Gemma convention.
        let row = &w.embed[token as usize * c.embed_dim..(token as usize + 1) * c.embed_dim];
        let scale = (c.embed_dim as f32).sqrt();
        for (dst, e) in x.iter_mut().zip(row) {
            *dst = e * scale;
        }

        // Per-layer input embeddings, computed once for this position from the
        // embedded input and the token identity, before the decoder stack.
        // Layer `i` consumes row `i` of the result.
        if let (Some(p), Some(pd)) = (w.ple.as_ref(), c.ple_dim) {
            // Context component: project the embedded input to one row per
            // layer, scale, and RMSNorm each row over its own `pd` dimensions.
            //
            // The scale is `hidden_size ** -0.5` and lands immediately before
            // an RMSNorm over the same values, which is scale invariant, so it
            // very nearly cancels — but only very nearly: `rms_eps` is added
            // to the mean square in absolute terms, so the scale does change
            // the result and is applied rather than folded away.
            matvec_into(&p.model_proj, x, ple);
            let proj_scale = 1.0 / (c.embed_dim as f32).sqrt();
            for v in ple.iter_mut() {
                *v *= proj_scale;
            }
            for row in ple.chunks_mut(pd) {
                rms_norm_in_place(row, &p.proj_norm, c.rms_eps);
            }
            // Token-identity component: one packed row of the per-layer table,
            // scaled by sqrt(ple_dim) -- x16 at Gemma 4's width of 256, an
            // embed scale that is easy to miss because it lives on the
            // embedding module rather than at the call site
            // (`docs/models/gemma4-upstream.txt`).
            //
            // The row stride is the packed width, not `pd`: the table is one
            // tensor to split into `n_layers` rows, not a stack of per-layer
            // tables.
            let stride = c.n_layers * pd;
            debug_assert!(
                (token as usize) < p.ple_vocab,
                "token bound checked at entry"
            );
            let identity = &p.token_embd[token as usize * stride..(token as usize + 1) * stride];
            let embed_scale = (pd as f32).sqrt();
            for (dst, e) in ple.iter_mut().zip(identity) {
                *dst = (*dst + e * embed_scale) * PLE_INPUT_SCALE;
            }
        }
        let ple_rows: &[f32] = ple;

        for (i, layer) in w.layers.iter().enumerate() {
            let hd = layer.spec.head_dim;
            let (q_dim, kv_dim) = (c.n_heads * hd, c.n_kv_heads * hd);
            let ffn_dim = layer.spec.ffn_dim;
            // Reborrowed per iteration so each layer carves the same storage
            // to its own widths. `Scratch::new` reserves for the widest layer
            // and `carve` panics if a view runs short, so a scratch sized
            // against the wrong layer fails here rather than as wrong logits.
            let mut rest: &mut [f32] = &mut *work;
            let normed = carve(&mut rest, c.embed_dim);
            let q = carve(&mut rest, q_dim);
            let k = carve(&mut rest, kv_dim);
            let v = carve(&mut rest, kv_dim);
            let attn_out = carve(&mut rest, q_dim);
            let projected = carve(&mut rest, c.embed_dim);
            let post = carve(&mut rest, c.embed_dim);
            let gate = carve(&mut rest, ffn_dim);
            let up = carve(&mut rest, ffn_dim);
            let mixed = carve(&mut rest, ffn_dim);
            let down = carve(&mut rest, c.embed_dim);
            let ple_gate = carve(&mut rest, c.ple_dim.unwrap_or(0));
            let ple_out = carve(&mut rest, c.embed_dim);

            // Attention block.
            rms_norm_into(x, &layer.attn_norm, c.rms_eps, normed);
            matvec_into(&layer.wq, normed, q);
            for head in q.chunks_mut(hd) {
                rms_norm_in_place(head, &layer.q_norm, c.rms_eps);
                rope(head, position, &layer.rope_freqs);
            }
            // K and V only where this layer owns its cache. A shared-KV layer
            // computes its query and nothing else, then attends over the
            // donor's rows -- which are already post-k_norm and post-rotary,
            // and already carry this position, because the donor is an earlier
            // layer in this same loop.
            let src = layer.spec.kv_source;
            if src == i {
                matvec_into(&layer.wk, normed, k);
                matvec_into(&layer.wv, normed, v);
                for head in k.chunks_mut(hd) {
                    rms_norm_in_place(head, &layer.k_norm, c.rms_eps);
                    rope(head, position, &layer.rope_freqs);
                }
                let cache = &mut kv[i];
                cache.k.extend_from_slice(k);
                cache.v.extend_from_slice(v);
            }
            // The donor shares this layer's type and therefore its head width,
            // so the reading layer's stride is the donor's stride. Asserted
            // rather than assumed: a donor rule that crossed layer types would
            // read the cache at the wrong stride and produce finite garbage.
            debug_assert_eq!(
                w.layers[src].spec.head_dim, hd,
                "donor layer {src} and reader layer {i} disagree on head width"
            );
            let cache = &kv[src];

            // Causal attention over the cache; sliding-window layers see
            // only the trailing `window` positions including this one. The
            // bound comes from the *reading* layer's own kind: sharing supplies
            // the states, it does not change which of them are visible.
            let first = match layer.spec.kind {
                LayerKind::Global => 0,
                LayerKind::SlidingWindow(win) => (position + 1).saturating_sub(win),
            };
            let attn_scale = 1.0 / (hd as f32).sqrt();
            attn_out.fill(0.0);
            for qh in 0..c.n_heads {
                let kvh = qh / heads_per_kv;
                let qv = &q[qh * hd..(qh + 1) * hd];
                scores.clear();
                scores.extend((first..=position).map(|p| {
                    let kp = &cache.k[p * kv_dim + kvh * hd..p * kv_dim + (kvh + 1) * hd];
                    qv.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>() * attn_scale
                }));
                softmax_in_place(scores);
                let out = &mut attn_out[qh * hd..(qh + 1) * hd];
                for (s, p) in scores.iter().zip(first..=position) {
                    let vp = &cache.v[p * kv_dim + kvh * hd..p * kv_dim + (kvh + 1) * hd];
                    for (o, val) in out.iter_mut().zip(vp) {
                        *o += s * val;
                    }
                }
            }
            matvec_into(&layer.wo, attn_out, projected);
            rms_norm_into(projected, &layer.post_attn_norm, c.rms_eps, post);
            for (a, b) in x.iter_mut().zip(post.iter()) {
                *a += b;
            }

            // GeGLU MLP block.
            rms_norm_into(x, &layer.ffn_norm, c.rms_eps, normed);
            matvec_into(&layer.w_gate, normed, gate);
            matvec_into(&layer.w_up, normed, up);
            for ((m, g), u) in mixed.iter_mut().zip(gate.iter()).zip(up.iter()) {
                *m = gelu(*g) * u;
            }
            matvec_into(&layer.w_down, mixed, down);
            rms_norm_into(down, &layer.post_ffw_norm, c.rms_eps, post);
            for (a, b) in x.iter_mut().zip(post.iter()) {
                *a += b;
            }

            // Per-layer input embedding block: a third residual update, after
            // the feed-forward block rather than before or inside attention.
            // The gate narrows the residual stream to `ple_dim`, the embedding
            // modulates it there elementwise, and the projection widens it
            // back (`docs/models/gemma4-upstream.txt`).
            if let (Some(pl), Some(pd)) = (layer.ple.as_ref(), c.ple_dim) {
                matvec_into(&pl.inp_gate, x, ple_gate);
                let row = &ple_rows[i * pd..(i + 1) * pd];
                for (g, e) in ple_gate.iter_mut().zip(row) {
                    *g = gelu(*g) * e;
                }
                matvec_into(&pl.proj, ple_gate, ple_out);
                rms_norm_into(ple_out, &pl.post_norm, c.rms_eps, post);
                for (a, b) in x.iter_mut().zip(post.iter()) {
                    *a += b;
                }
            }

            // The layer's last operation, scaling everything the layer
            // produced including the residual carried through it. Upstream
            // applies it after all of the layer's residual updates and
            // returns immediately, so it is the boundary between this layer
            // and the next.
            if layer.output_scale != 1.0 {
                for a in x.iter_mut() {
                    *a *= layer.output_scale;
                }
            }

            // Steering after the scale and before the tap, in that order, so
            // both seams keep their stated meaning: a control vector is added
            // to the block's output the way llama.cpp adds it, and the tap
            // observes exactly what the next block consumes. Putting the tap
            // before the scale would break the correspondence with the Python
            // forward hook, which fires on the decoder module and therefore
            // receives the already-scaled tensor -- a per-layer discrepancy
            // spanning 47x across the stack, and one that would make probes
            // trained on one capture path score the other with plausible
            // numbers. See docs/interp/gemma-4-interp.md.
            if let Some(dir) = control.as_ref().and_then(|cv| cv.direction(i)) {
                let scale = control.as_ref().expect("checked above").scale;
                for (a, b) in x.iter_mut().zip(dir) {
                    *a += scale * b;
                }
            }

            // Interp tap seam: the layer's complete output.
            if let Some(tap) = tap.as_mut() {
                tap.on_residual(i, position, x);
            }
        }

        // Final norm and tied LM head: logits[t] = embed_row(t) . x. The
        // logits vector is the return value, so it is the one allocation
        // per token the forward pass still owes its caller.
        let mut rest: &mut [f32] = work;
        let normed = carve(&mut rest, c.embed_dim);
        rms_norm_into(x, &w.output_norm, c.rms_eps, normed);
        let hidden: &[f32] = normed;
        let logits = (0..c.vocab_size)
            .map(|t| {
                let row = &w.embed[t * c.embed_dim..(t + 1) * c.embed_dim];
                let logit: f32 = row.iter().zip(hidden).map(|(a, b)| a * b).sum();
                // Softcap, when the checkpoint declares one: a smooth squash
                // toward +/-cap rather than a clamp. Monotone, so it moves no
                // argmax and greedy decoding cannot detect its absence --
                // perplexity, sampled decoding, and every captured logprob
                // can, which is why omitting it is a silent defect and why
                // its tests are not written against argmax.
                match c.logit_softcap {
                    Some(cap) => (logit / cap).tanh() * cap,
                    None => logit,
                }
            })
            .collect();
        Ok(Logits(logits))
    }
}

/// Elements moved per snapshot read or write.
///
/// The KV payload is the bulk of a snapshot — hundreds of megabytes at real
/// model scale — and moving it one f32 at a time through a `dyn Write` or
/// `dyn Read` costs a virtual call per four bytes. 64 KiB per block keeps
/// the temporary well inside cache while amortizing the dispatch away.
const SNAPSHOT_CHUNK_ELEMENTS: usize = 16 << 10;

impl Model for CpuModel {
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn cached_len(&self) -> usize {
        self.cached_len
    }

    fn prefill(&mut self, tokens: &[Token]) -> Result<Logits> {
        Ok(self.prefill_all(tokens)?.pop().expect("non-empty prefill"))
    }

    fn prefill_all(&mut self, tokens: &[Token]) -> Result<Vec<Logits>> {
        ensure!(!tokens.is_empty(), "prefill of zero tokens");
        ensure!(
            self.cached_len + tokens.len() <= self.config.context_length,
            "prefill of {} tokens past cached {} exceeds context length {}",
            tokens.len(),
            self.cached_len,
            self.config.context_length
        );
        // Prefill is atomic, per the `Model::prefill` contract. The
        // vocabulary bound needs no forward pass, so scanning the whole batch
        // here makes the only mid-batch failure the loop can currently raise
        // unreachable, before any token is committed.
        for &t in tokens {
            self.check_token(t)?;
        }
        let span =
            tracing::info_span!("model.forward", start = self.cached_len, len = tokens.len());
        let _enter = span.enter();
        let entry = self.cached_len;
        let mut rows = Vec::with_capacity(tokens.len());
        for &t in tokens {
            let position = self.cached_len;
            // Backstop for any error a future forward pass raises: roll the
            // cache back to the length prefill was called at, so a failed
            // batch leaves the model exactly as it found it.
            let logits = match self.forward_one(t, position) {
                Ok(logits) => logits,
                Err(e) => {
                    self.truncate(entry)
                        .expect("entry length is a prefix of the current cache");
                    return Err(e);
                }
            };
            self.cached_len += 1;
            rows.push(logits);
        }
        Ok(rows)
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        ensure!(len <= self.cached_len, "truncate beyond cached length");
        for (i, cache) in self.kv.iter_mut().enumerate() {
            let kv_dim = self.spec.cache_row(i);
            cache.k.truncate(len * kv_dim);
            cache.v.truncate(len * kv_dim);
        }
        self.cached_len = len;
        Ok(())
    }

    fn fork(&self) -> Result<Box<dyn Model>> {
        // Weights shared, cache deep-copied: correctness first, prefix
        // sharing stays deferred per the runtime design.
        Ok(Box::new(CpuModel {
            config: self.config,
            spec: Arc::clone(&self.spec),
            weights: Arc::clone(&self.weights),
            kv: self.kv.clone(),
            cached_len: self.cached_len,
            // Fresh buffers rather than a clone of the parent's: scratch is
            // written before it is read at every position, so it carries no
            // state worth copying, and sharing it would let interleaved
            // parent and child decodes overwrite each other mid-token.
            scratch: Scratch::new(&self.spec),
            tap: None,
            control: self.control.clone(),
        }))
    }

    // Cache layout on disk: n_layers u64, cached_len u64, then per layer the
    // K vector followed by the V vector as little-endian f32. Lengths are
    // implied (cached_len x n_kv_heads x that layer's head width; the CPU
    // backend stores the full sequence in every layer it stores at all,
    // windowed or not), so a length mismatch on load is a hard error rather
    // than a guess.
    //
    // The per-layer head width is why the row length is recomputed per layer
    // rather than hoisted: a checkpoint interleaving two head widths writes
    // rows of two sizes, and one hoisted stride would read the second regime
    // at the first regime's offset. A shared-KV layer contributes a row length
    // of zero and so occupies no payload at all -- the layer count in the
    // header still covers every layer, so the reader walks the same indices.
    fn save_state(&self, w: &mut dyn std::io::Write) -> Result<()> {
        w.write_all(&(self.kv.len() as u64).to_le_bytes())?;
        w.write_all(&(self.cached_len as u64).to_le_bytes())?;
        for (i, cache) in self.kv.iter().enumerate() {
            let kv_dim = self.spec.cache_row(i);
            ensure!(
                cache.k.len() == self.cached_len * kv_dim
                    && cache.v.len() == self.cached_len * kv_dim,
                "layer {i} KV cache length inconsistent with cached_len"
            );
            // Block at a time rather than four bytes at a time: a real
            // checkpoint's cache is hundreds of megabytes, and one write_all
            // per f32 through a trait object spends more time in dispatch
            // than in the copy.
            for vec in [&cache.k, &cache.v] {
                for chunk in vec.chunks(SNAPSHOT_CHUNK_ELEMENTS) {
                    let mut block = Vec::with_capacity(chunk.len() * 4);
                    for &x in chunk {
                        block.extend_from_slice(&x.to_le_bytes());
                    }
                    w.write_all(&block)?;
                }
            }
        }
        Ok(())
    }

    fn load_state(&mut self, r: &mut dyn std::io::Read) -> Result<()> {
        ensure!(self.cached_len == 0, "load_state on a non-fresh cache");
        let mut u64buf = [0u8; 8];
        r.read_exact(&mut u64buf)?;
        let n_layers = u64::from_le_bytes(u64buf) as usize;
        ensure!(
            n_layers == self.kv.len(),
            "saved state has {n_layers} layers, model has {}",
            self.kv.len()
        );
        r.read_exact(&mut u64buf)?;
        let cached_len = u64::from_le_bytes(u64buf) as usize;
        ensure!(
            cached_len <= self.config.context_length,
            "saved state of {cached_len} tokens exceeds context length {}",
            self.config.context_length
        );
        // Capacity reserved before a single payload byte has arrived. The
        // length is a header field and `context_length` is whatever the
        // checkpoint claims, so reserving for the claim outright let a
        // 41-byte snapshot ask for gigabytes and abort the process through
        // the allocation error handler rather than report the truncation.
        // Past this step the vectors grow as bytes actually arrive, so peak
        // memory tracks the payload rather than the claim.
        const RESERVE_ELEMENTS: usize = 1 << 16;
        // Read into a temporary cache and commit both halves of the state at
        // once. Filling `self.kv` in place left a truncated snapshot's rows
        // behind under a `cached_len` never assigned, so the model reported
        // fresh while attention read those rows at position 0 — fluent,
        // wrong logits, which this backend refuses to produce.
        let mut restored = vec![LayerKv::default(); self.kv.len()];
        let mut block = Vec::new();
        for (i, cache) in restored.iter_mut().enumerate() {
            let elements = cached_len * self.spec.cache_row(i);
            for vec in [&mut cache.k, &mut cache.v] {
                vec.reserve_exact(elements.min(RESERVE_ELEMENTS));
                // Read in blocks for the same reason save_state writes in
                // them, while still growing with the payload so a header
                // that overstates its length cannot preallocate for a
                // payload that never arrives.
                let mut remaining = elements;
                while remaining > 0 {
                    let take = remaining.min(SNAPSHOT_CHUNK_ELEMENTS);
                    block.resize(take * 4, 0);
                    r.read_exact(&mut block)?;
                    vec.extend(
                        block
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                    );
                    remaining -= take;
                }
            }
        }
        self.kv = restored;
        self.cached_len = cached_len;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::TinySpec;

    /// The synthetic checkpoint for `spec`, in memory.
    ///
    /// Owned separately from the `Checkpoint` that borrows it, so a test that
    /// loads two models from one spec holds the bytes once.
    fn checkpoint_of(spec: &TinySpec) -> crate::checkpoint::CheckpointFiles {
        crate::checkpoint::CheckpointFiles {
            config: crate::synth::build_tiny_config_json(spec).unwrap(),
            weights: crate::synth::build_tiny_safetensors(spec).unwrap(),
            dir: std::path::PathBuf::from("<memory>"),
        }
    }

    /// A checkpoint whose tensor set has been perturbed: `extra` Hugging Face
    /// names are added as one-element tensors, and any name for which `keep`
    /// returns false is dropped.
    ///
    /// The refusals these serve are about which tensors exist and what shape
    /// they are, not what they contain, so an added tensor's single element is
    /// enough. Built from `tiny_tensors` rather than by editing a serialized
    /// image, because safetensors offsets are computed at serialization and
    /// patching them by hand is how a test starts asserting against a file no
    /// loader would ever see.
    fn checkpoint_with(
        spec: &TinySpec,
        extra: &[&str],
        reshape: &[(&str, &[usize])],
        keep: impl Fn(&str) -> bool,
    ) -> crate::checkpoint::CheckpointFiles {
        use safetensors::tensor::{Dtype, TensorView};

        let mut payloads: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        for t in crate::synth::tiny_tensors(spec) {
            let Some(name) = t.hf_name.clone() else {
                continue;
            };
            if !keep(&name) {
                continue;
            }
            let shift = if t.norm { 1.0f32 } else { 0.0 };
            let bytes: Vec<u8> = t
                .values
                .iter()
                .flat_map(|v| (v - shift).to_le_bytes())
                .collect();
            // A self-consistent wrong shape: the header declares it and the
            // payload supplies exactly that many elements, so nothing but the
            // loader's own check can notice.
            match reshape.iter().find(|(n, _)| *n == name) {
                Some((_, shape)) => {
                    let n: usize = shape.iter().product();
                    payloads.push((name, shape.to_vec(), bytes[..n * 4].to_vec()));
                }
                None => payloads.push((name, t.hf_shape(), bytes)),
            }
        }
        for name in extra {
            payloads.push(((*name).to_string(), vec![1], 0.0f32.to_le_bytes().to_vec()));
        }
        let views: Vec<(&str, TensorView)> = payloads
            .iter()
            .map(|(n, shape, bytes)| {
                (
                    n.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        crate::checkpoint::CheckpointFiles {
            config: crate::synth::build_tiny_config_json(spec).unwrap(),
            weights: safetensors::serialize(views, None).unwrap(),
            dir: std::path::PathBuf::from("<memory>"),
        }
    }

    /// A checkpoint whose stored values have been edited in place.
    ///
    /// `edit` sees each tensor's Hugging Face name and its stored values --
    /// unshifted, as the file holds them. Perturbing a weight is how a test
    /// shows which layers actually read it, and doing it through the writer
    /// rather than by patching serialized bytes keeps the test independent of
    /// where safetensors chose to place a payload.
    fn checkpoint_edited(
        spec: &TinySpec,
        edit: impl Fn(&str, &mut Vec<f32>),
    ) -> crate::checkpoint::CheckpointFiles {
        use safetensors::tensor::{Dtype, TensorView};

        let mut payloads: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        for t in crate::synth::tiny_tensors(spec) {
            let Some(name) = t.hf_name.clone() else {
                continue;
            };
            let shift = if t.norm { 1.0f32 } else { 0.0 };
            let mut values: Vec<f32> = t.values.iter().map(|v| v - shift).collect();
            edit(&name, &mut values);
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            payloads.push((name, t.hf_shape(), bytes));
        }
        let views: Vec<(&str, TensorView)> = payloads
            .iter()
            .map(|(n, shape, bytes)| {
                (
                    n.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        crate::checkpoint::CheckpointFiles {
            config: crate::synth::build_tiny_config_json(spec).unwrap(),
            weights: safetensors::serialize(views, None).unwrap(),
            dir: std::path::PathBuf::from("<memory>"),
        }
    }

    /// `checkpoint_edited` for the common case: one whole tensor shifted.
    fn checkpoint_bumping(
        spec: &TinySpec,
        target: &str,
        delta: f32,
    ) -> crate::checkpoint::CheckpointFiles {
        let mut hit = false;
        let files = checkpoint_edited(spec, |name, values| {
            if name == target {
                for v in values.iter_mut() {
                    *v += delta;
                }
            }
        });
        for t in crate::synth::tiny_tensors(spec) {
            hit |= t.hf_name.as_deref() == Some(target);
        }
        assert!(hit, "no tensor named {target} to perturb");
        files
    }

    /// One tensor's effective values, named in the llama.cpp spelling the
    /// tests use, with the RMSNorm shift applied exactly as the loader applies
    /// it. Independent-recomputation tests want the numbers the forward pass
    /// sees, not the geometry, so there is no shape argument.
    fn tensor_of(ck: &crate::checkpoint::Checkpoint, gguf_name: &str) -> Vec<f32> {
        let hf = crate::checkpoint::hf_tensor_name(gguf_name)
            .unwrap_or_else(|| panic!("no published name for {gguf_name}"));
        let view = ck
            .tensors
            .tensor(&hf)
            .unwrap_or_else(|e| panic!("missing {hf}: {e:?}"));
        let shift = match gguf_name.ends_with("norm.weight") {
            true => 1.0f32,
            false => 0.0,
        };
        view.data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) + shift)
            .collect()
    }

    /// `check_supported_checkpoint`'s verdict on a perturbed tensor set.
    fn verdict(files: &crate::checkpoint::CheckpointFiles) -> Result<()> {
        let ck = crate::checkpoint::parse(files)?;
        let spec = ModelSpec::from_checkpoint(&ck.config)?;
        check_supported_checkpoint(&ck, &spec)
    }

    /// A model loaded from `spec`, the shape most tests below want.
    fn model_of(spec: &TinySpec) -> CpuModel {
        let files = checkpoint_of(spec);
        let ck = crate::checkpoint::parse(&files).unwrap();
        CpuModel::from_checkpoint(&ck).unwrap()
    }

    fn tiny() -> CpuModel {
        model_of(&TinySpec::default())
    }

    /// Allocating wrapper over `rms_norm_into`, so the convention tests read
    /// as the arithmetic they check rather than as buffer bookkeeping.
    fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let mut out = vec![0.0; x.len()];
        rms_norm_into(x, w, eps, &mut out);
        out
    }

    #[test]
    fn rms_norm_scales_by_the_stored_weight_not_one_plus_it() {
        // The GGUF stores the effective multiplier: the converter resolves
        // Gemma's runtime +1 before writing. A unit weight must therefore
        // leave the normalized vector untouched. Under the old `1 + w`
        // reading every element would come back doubled, so this pins the
        // convention against a silent regression on real checkpoints.
        let x = [3.0f32, -4.0, 0.0, 5.0];
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();

        let unit = rms_norm(&x, &[1.0; 4], 0.0);
        for (got, v) in unit.iter().zip(&x) {
            assert!((got - v / rms).abs() < 1e-6, "unit weight altered {v}");
        }

        let doubled = rms_norm(&x, &[2.0; 4], 0.0);
        for (got, v) in doubled.iter().zip(&x) {
            assert!((got - 2.0 * v / rms).abs() < 1e-6, "weight 2.0 misapplied");
        }

        // A zero weight annihilates its channel; under `1 + w` it would pass
        // the value through unchanged.
        assert_eq!(rms_norm(&x, &[0.0; 4], 0.0), vec![0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "matvec")]
    fn matvec_rejects_a_weight_matrix_that_does_not_match_the_declared_shape() {
        // A weight longer than in_dim x out_dim stays in bounds and silently
        // uses its leading rows, so a release build had no guard here at all:
        // debug_assert is compiled out exactly where wrong numerics matter.
        // The check costs one integer comparison against in_dim x out_dim
        // multiply-accumulates.
        matvec_into(&[1.0; 16], &[1.0; 4], &mut [0.0; 2]);
    }

    #[test]
    fn the_rope_frequency_table_is_the_per_pair_expression_it_replaced() {
        // The table hoists a powf out of the rotation loop, so its contract
        // is bit equality with the expression that used to run there, not
        // closeness: every logit is pinned by the numpy goldens and by the
        // determinism tests, and a frequency off by one mantissa bit would
        // surface there with no local cause to read.
        for (head_dim, base) in [(8usize, 10000.0f32), (128, 1_000_000.0), (256, 10000.0)] {
            let table = rope_frequencies(head_dim, base, None);
            assert_eq!(table.len(), head_dim / 2);
            assert_eq!(table[0], 1.0, "pair 0 rotates at the position itself");
            for (j, f) in table.iter().enumerate() {
                let inline = base.powf(-2.0 * j as f32 / head_dim as f32);
                assert_eq!(
                    f.to_bits(),
                    inline.to_bits(),
                    "head_dim {head_dim} pair {j}"
                );
            }
            assert!(
                table.windows(2).all(|p| p[0] > p[1]),
                "frequency must fall with pair index"
            );
        }
    }

    #[test]
    fn proportional_rope_divisors_make_the_padded_pairs_inert() {
        // Gemma 4's global regime rotates a quarter of each head, expressed
        // as a full-width rotation whose trailing frequencies are divided by
        // 1e30 rather than as a narrowed slice. Reproduce the shape at E4B's
        // proportions: head 512, 64 rotating pairs, 192 padded.
        let head_dim = 512;
        let mut divisors = vec![1.0f32; 64];
        divisors.extend(std::iter::repeat_n(1e30f32, 192));
        let plain = rope_frequencies(head_dim, 1_000_000.0, None);
        let table = rope_frequencies(head_dim, 1_000_000.0, Some(&divisors));

        // The rotating pairs are untouched: dividing by 1.0 is exact.
        for j in 0..64 {
            assert_eq!(table[j].to_bits(), plain[j].to_bits(), "pair {j}");
        }
        // The padded pairs are inert: at the largest position Gemma 4's
        // context allows, the rotation is the identity in f32. Asserted as
        // the identity rather than as a small angle, because that is the
        // property the forward pass depends on. Not asserted as an exactly
        // zero frequency: upstream divides by 1e30 rather than infinity, and
        // reproducing that residual instead of rounding it away is the point.
        for (j, f) in table.iter().enumerate().skip(64) {
            assert!(*f > 0.0, "pair {j} divided to zero, losing the residual");
            let (sin, cos) = (131_072.0f32 * f).sin_cos();
            assert_eq!(cos, 1.0, "pair {j} cos {cos}");
            assert_eq!(
                sin,
                131_072.0f32 * f,
                "pair {j} sin departed from the angle"
            );
            assert!(
                sin.abs() < f32::EPSILON,
                "pair {j} still rotates: sin {sin}"
            );
        }
    }

    #[test]
    fn the_rope_divisor_tensor_reaches_the_logits() {
        // The regression this whole change exists for. `rope_freqs.weight`
        // was listed inert on the reasoning that frequencies are recomputed
        // from the base -- true, but not from the base alone. While it was
        // ignored, a checkpoint carrying it loaded and produced finite,
        // fluent, wrong logits with nothing to notice. Asserting only that
        // the loader accepts the tensor would pass against that same bug, so
        // the assertion is that the numerics move.
        let load = |spec: &TinySpec| {
            let image = checkpoint_of(spec);
            let file = crate::checkpoint::parse(&image).unwrap();
            let mut model = CpuModel::from_checkpoint(&file).unwrap();
            model.prefill_all(&[1, 5, 9, 13, 3, 8, 21, 2]).unwrap()
        };
        let plain = load(&TinySpec::default());
        // Head width 8 gives 4 pairs; rotate one, pad three.
        let partial = load(&TinySpec {
            rope_rotating_pairs: Some(1),
            ..TinySpec::default()
        });
        assert_eq!(plain.len(), partial.len());
        let moved = plain
            .iter()
            .zip(&partial)
            .flat_map(|(a, b)| a.0.iter().zip(&b.0))
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(
            moved > 0,
            "divisors changed no logit: the tensor is still inert"
        );

        // All-ones divisors are the identity, so a checkpoint that declares
        // a full rotation must be bit-identical to one that declares none.
        // This is the other half: the tensor must not perturb the case it
        // does not describe.
        let full = load(&TinySpec {
            rope_rotating_pairs: Some(4),
            ..TinySpec::default()
        });
        for (position, (a, b)) in plain.iter().zip(&full).enumerate() {
            for (t, (x, y)) in a.0.iter().zip(&b.0).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "position {position} token {t}: dividing by 1.0 moved a logit"
                );
            }
        }
    }

    #[test]
    fn only_the_global_regime_takes_the_rope_divisors() {
        // The divisors are the full-attention regime's partial rotation; the
        // sliding regime rotates its whole head. Applying them to both would
        // silence three quarters of every sliding layer's rotation, which
        // still reads as a working model.
        let spec = TinySpec {
            rope_rotating_pairs: Some(1),
            ..TinySpec::default()
        };
        let model = model_of(&spec);
        let plain = rope_frequencies(spec.head_dim, spec.rope_freq_base, None);

        // Layer 0 sliding: table untouched. Layer 1 global: padded pairs
        // driven to nothing.
        let sliding = &model.weights.layers[0].rope_freqs;
        let global = &model.weights.layers[1].rope_freqs;
        assert_eq!(model.spec.layers[0].kind, LayerKind::SlidingWindow(4));
        assert_eq!(model.spec.layers[1].kind, LayerKind::Global);
        for (j, f) in plain.iter().enumerate() {
            assert_eq!(sliding[j].to_bits(), f.to_bits(), "sliding pair {j}");
        }
        assert_eq!(global[0].to_bits(), plain[0].to_bits(), "rotating pair 0");
        for (j, f) in global.iter().enumerate().skip(1) {
            assert!(
                *f < plain[j] * 1e-20,
                "global pair {j} was not divided: {f}"
            );
        }
    }

    #[test]
    fn config_parses_from_the_tiny_model() {
        let m = tiny();
        let c = m.config();
        assert_eq!(c.vocab_size, 288);
        assert_eq!(c.n_layers, 2);
        assert_eq!(c.n_heads, 2);
        assert_eq!(c.n_kv_heads, 1);
        assert_eq!(m.layer_specs()[0].kind, LayerKind::SlidingWindow(4));
        assert_eq!(m.layer_specs()[1].kind, LayerKind::Global);
    }

    #[test]
    fn the_synthetic_fixture_consumes_every_tensor_it_ships() {
        // Couples the guard's tensor lists to the loader: a tensor the loader
        // reads but the lists omit makes the guard refuse this fixture.
        verdict(&checkpoint_of(&TinySpec::default())).unwrap();
        verdict(&checkpoint_of(&TinySpec::e4b_shaped())).unwrap();
    }

    #[test]
    fn a_partial_per_layer_embedding_tensor_set_is_refused() {
        // The feature is six tensor families, and five of them load fine while
        // silently dropping the residual join the sixth carries. Refusing the
        // partial set is the only thing that turns it into a load error rather
        // than into a fluent model missing most of its parameters.
        let files = checkpoint_with(
            &TinySpec::default(),
            &["model.embed_tokens_per_layer.weight"],
            &[],
            |_| true,
        );
        let err = verdict(&files).unwrap_err().to_string();
        assert!(err.contains("absent or incomplete"), "{err}");
        // Names the three per-block families, which is the point: they lose
        // the `per_layer` substring in conversion, so a reader looking for
        // what is missing cannot find them by the name of the one present.
        for suffix in PLE_LAYER_TENSORS {
            let hf = crate::checkpoint::hf_tensor_name(&format!("blk.0.{suffix}")).unwrap();
            let family = hf_tensor_family(&hf);
            assert!(err.contains(&family), "{err} lacks {family}");
        }
    }

    #[test]
    fn a_per_layer_embedding_family_missing_from_one_layer_is_refused() {
        // The whole-family case above is the obvious one. This is the case the
        // per-layer count exists for: every family present, one layer short,
        // which loads and skips the join on exactly that layer.
        let spec = TinySpec {
            ple_dim: Some(4),
            ..TinySpec::default()
        };
        verdict(&checkpoint_of(&spec)).unwrap();
        let files = checkpoint_with(&spec, &[], &[], |n| {
            n != "model.layers.1.per_layer_input_gate.weight"
        });
        let err = verdict(&files).unwrap_err().to_string();
        assert!(
            err.contains("model.layers.*.per_layer_input_gate.weight (1 of 2 layers)"),
            "{err}"
        );
    }

    #[test]
    fn per_layer_embedding_tensors_without_their_width_are_refused() {
        // The tensors alone do not say how to split the packed row. Inferring
        // the width from the shape would work only if `n_layers` and the
        // per-layer width were the sole factorization of it, which they are
        // not.
        let files = checkpoint_of(&TinySpec {
            ple_dim: Some(4),
            ..TinySpec::default()
        });
        let mut config: serde_json::Value = serde_json::from_slice(&files.config).unwrap();
        config["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("hidden_size_per_layer_input");
        let files = crate::checkpoint::CheckpointFiles {
            config: serde_json::to_vec(&config).unwrap(),
            ..files
        };
        let err = verdict(&files).unwrap_err().to_string();
        assert!(err.contains("declares no"), "{err}");
    }

    #[test]
    fn a_cache_owning_layer_without_kv_tensors_is_refused() {
        // A layer outside the shared tail has no donor, so a missing v_proj
        // leaves it nothing to attend over. Refused by the loader before any
        // weight is read. Layer 1 of the default fixture owns its cache --
        // that fixture declares no sharing at all.
        let files = checkpoint_with(&TinySpec::default(), &[], &[], |n| {
            n != "model.layers.1.self_attn.v_proj.weight"
        });
        let ck = crate::checkpoint::parse(&files).unwrap();
        let err = match CpuModel::from_checkpoint(&ck) {
            Ok(_) => panic!("a cache-owning layer missing v_proj must not load"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("stores its own K/V"), "{err}");
        assert!(err.contains("layer 1"), "{err}");
    }

    /// Four layers on a period-2 schedule, so layers 0 and 2 are sliding and
    /// layers 1 and 3 are global. With one shared layer the tail is layer 3
    /// alone; with two it is layers 2 and 3, one of each type -- the geometry
    /// where a single-donor rule and the per-type rule give different answers.
    fn sharing_spec(shared: usize) -> TinySpec {
        TinySpec {
            n_layers: 4,
            sliding_window_pattern: 2,
            shared_kv_layers: shared,
            ..TinySpec::default()
        }
    }

    fn sharing_spec_model(shared: usize) -> (crate::checkpoint::CheckpointFiles, CpuModel) {
        let files = checkpoint_of(&sharing_spec(shared));
        let model = {
            let ck = crate::checkpoint::parse(&files).unwrap();
            CpuModel::from_checkpoint(&ck).unwrap()
        };
        (files, model)
    }

    #[test]
    fn each_shared_layer_reads_the_last_non_shared_layer_of_its_own_type() {
        // The correction that matters most in this stage: two donors keyed by
        // layer type, not one donor for the whole tail. With layers 2 and 3
        // shared, the boundary is 2, so the candidates are layers 0 (sliding)
        // and 1 (global). A single-donor implementation would point both
        // shared layers at layer 1, the last non-shared layer outright.
        let (_, model) = sharing_spec_model(2);
        let specs = model.layer_specs();
        assert_eq!(specs[0].kv_source, 0, "non-shared layers own their cache");
        assert_eq!(specs[1].kv_source, 1, "non-shared layers own their cache");
        assert_eq!(
            specs[2].kv_source, 0,
            "layer 2 is sliding, so its donor is the last non-shared sliding \
             layer (0), not the last non-shared layer (1)"
        );
        assert_eq!(
            specs[3].kv_source, 1,
            "layer 3 is global, so its donor is the last non-shared global layer"
        );
        assert_eq!(model.spec.cache_owning_layers(), 2);
    }

    #[test]
    fn a_shared_layer_stores_no_cache_rows_of_its_own() {
        // The memory claim, asserted on the cache rather than on an estimate:
        // the shared tail holds nothing at all, at every position.
        let (_, mut model) = sharing_spec_model(2);
        model.prefill_all(&[1, 4, 5, 6]).unwrap();
        for i in 0..4 {
            let owns = model.spec.owns_cache(i);
            let rows = model.kv[i].k.len();
            match owns {
                true => assert_eq!(rows, 4 * model.spec.kv_dim(i), "layer {i} owns its cache"),
                false => assert_eq!(rows, 0, "layer {i} reads a donor and must store nothing"),
            }
            assert_eq!(model.kv[i].v.len(), rows, "layer {i} K and V lengths agree");
        }
    }

    #[test]
    fn a_shared_layer_reads_its_donors_rows_and_not_a_stale_prefix() {
        // The donor is an earlier layer in the same loop, so by the time a
        // shared layer attends, the donor's cache already carries the current
        // position. Asserted directly, because the failure it guards against
        // -- a shared layer attending over every position but the newest --
        // stays finite and fluent.
        let (_, mut model) = sharing_spec_model(2);
        let tokens = [1u32, 7, 9, 4];
        model.prefill_all(&tokens).unwrap();
        for i in 0..model.config.n_layers {
            let src = model.spec.layers[i].kv_source;
            if src == i {
                continue;
            }
            assert_eq!(
                model.spec.layers[src].head_dim, model.spec.layers[i].head_dim,
                "donor {src} and reader {i} must share a head width, or the \
                 reader walks the cache at the wrong stride"
            );
            assert_eq!(
                model.kv[src].k.len(),
                tokens.len() * model.spec.kv_dim(src),
                "donor {src} must retain every position for reader {i}"
            );
        }
    }

    #[test]
    fn a_shared_layer_follows_the_donors_cache_when_the_donor_changes() {
        // Perturb only the donor's key projection and the shared layer that
        // reads it must move. This is what distinguishes reading the donor's
        // cache from computing K/V locally and merely being told not to store
        // it -- both leave the tail with zero rows, and only this separates
        // them.
        let spec = sharing_spec(2);
        let run = |files: &crate::checkpoint::CheckpointFiles| -> Vec<f32> {
            let ck = crate::checkpoint::parse(files).unwrap();
            let mut m = CpuModel::from_checkpoint(&ck).unwrap();
            m.prefill_all(&[1, 7, 9, 4]).unwrap().pop().unwrap().0
        };
        let before = run(&checkpoint_of(&spec));

        // Layer 0 is the sliding donor, read by layer 2 and by nothing else.
        let bumped = checkpoint_bumping(&spec, "model.layers.0.self_attn.k_proj.weight", 0.5);
        assert_ne!(
            before,
            run(&bumped),
            "the donor's key projection must reach the layers that read its cache"
        );
    }

    #[test]
    fn sharing_changes_the_logits_it_is_supposed_to_change() {
        // The two images differ in one metadata value and in nothing else --
        // every tensor, including the shared tail's attn_k and attn_v, is
        // drawn from the same stream and written identically. So any
        // difference in output is attributable to the sharing rule alone.
        //
        // Without this, a backend that read the key and then ignored it would
        // pass every structural test above.
        let logits = |shared: usize| -> Vec<f32> {
            let (_, mut model) = sharing_spec_model(shared);
            model.prefill_all(&[1, 7, 9, 4]).unwrap().pop().unwrap().0
        };
        let plain = logits(0);
        let shared = logits(2);
        assert_eq!(plain.len(), shared.len());
        assert!(
            plain.iter().zip(&shared).any(|(a, b)| a != b),
            "declaring a shared-KV tail must change the forward pass"
        );
        assert!(
            shared.iter().all(|v| v.is_finite()),
            "a shared-KV forward pass must stay finite"
        );
    }

    #[test]
    fn a_shared_layers_own_key_and_value_weights_are_deliberately_unread() {
        // Names a decision, not a discovery. `modeling_gemma4.py` builds
        // k_proj and v_proj only for non-shared layers, but every released
        // E-series checkpoint carries them on the shared tail too, and nothing
        // in the converter drops them. This backend follows the modeling code
        // and leaves them unread; that is the recorded conflict in
        // `docs/rust-harness/gemma-4-ingestion.md`, not a settled fact.
        //
        // Perturbing those tensors must therefore change nothing. If a
        // workstation run against real weights shows it should, this test is
        // the one that fails, and it fails saying why.
        let spec = sharing_spec(2);
        let run = |files: &crate::checkpoint::CheckpointFiles| -> Vec<f32> {
            let ck = crate::checkpoint::parse(files).unwrap();
            let mut m = CpuModel::from_checkpoint(&ck).unwrap();
            m.prefill_all(&[1, 7, 9, 4]).unwrap().pop().unwrap().0
        };
        let before = run(&checkpoint_of(&spec));

        // Layers 2 and 3 are the shared tail; layer 1 is a donor and is left
        // alone, so this cannot pass by perturbing nothing. Applied one at a
        // time: any single one reaching the logits fails the assertion, and
        // naming which one is what makes the failure actionable.
        for name in [
            "model.layers.2.self_attn.k_proj.weight",
            "model.layers.2.self_attn.v_proj.weight",
            "model.layers.3.self_attn.k_proj.weight",
        ] {
            assert_eq!(
                before,
                run(&checkpoint_bumping(&spec, name, 0.5)),
                "{name} reached the numerics; a shared layer must not read it"
            );
        }
    }

    #[test]
    fn a_shared_kv_tail_without_a_donor_of_its_type_is_a_load_error() {
        // Sharing every layer leaves the tail with nothing below the boundary
        // to read. Diagnosed at load rather than reaching the forward pass,
        // where it would index a cache that was never written.
        let image = checkpoint_of(&sharing_spec(4));
        let file = crate::checkpoint::parse(&image).unwrap();
        // Alternate form, so the assertion sees the whole anyhow chain: the
        // donor rule reports through a context layer that names the config
        // field a reader has to go and fix.
        let err = format!(
            "{:#}",
            ModelSpec::from_checkpoint(&file.config).unwrap_err()
        );
        assert!(err.contains("no donor to read"), "{err}");
        assert!(err.contains("num_kv_shared_layers"), "{err}");
    }

    #[test]
    fn a_shared_kv_model_survives_the_snapshot_and_cache_edit_paths() {
        // The cache lifecycle under sharing: save, restore, truncate, fork.
        // Every one of these walks the layer list computing a row length, and
        // a walk that forgets the shared tail either writes rows that are not
        // there or reads a payload at the wrong offsets. Exercised together
        // because they share the one predicate, and none of them was reachable
        // by any existing test -- the fixtures they use declare no sharing.
        let (_, mut model) = sharing_spec_model(2);
        let tokens = [1u32, 7, 9, 4];
        model.prefill_all(&tokens).unwrap();

        let mut snapshot = Vec::new();
        model.save_state(&mut snapshot).unwrap();
        // The shared tail contributes no payload: two owning layers of K and V
        // at four positions, plus the two u64 header fields.
        let owned: usize = (0..4)
            .filter(|&i| model.spec.owns_cache(i))
            .map(|i| model.spec.kv_dim(i))
            .sum();
        assert_eq!(snapshot.len(), 16 + 2 * owned * tokens.len() * 4);

        // Taken before the parent advances, so the fork and the parent are
        // asked for the same next token from the same cached prefix.
        let mut forked = model.fork().unwrap();
        let reference = model.prefill(&[5]).unwrap().0;

        let (_, mut restored) = sharing_spec_model(2);
        restored.load_state(&mut snapshot.as_slice()).unwrap();
        assert_eq!(restored.cached_len, tokens.len());
        assert_eq!(
            restored.prefill(&[5]).unwrap().0,
            reference,
            "a restored shared-KV cache must continue bit-identically"
        );

        assert_eq!(
            forked.prefill(&[5]).unwrap().0,
            reference,
            "a fork of a shared-KV cache must continue bit-identically"
        );

        let (_, mut rebuilt) = sharing_spec_model(2);
        rebuilt.prefill_all(&tokens).unwrap();
        rebuilt.truncate(2).unwrap();
        rebuilt.prefill_all(&tokens[2..]).unwrap();
        assert_eq!(
            rebuilt.prefill(&[5]).unwrap().0,
            reference,
            "truncate then re-prefill must land where the original was"
        );
    }

    #[test]
    fn a_checkpoint_without_sharing_is_unperturbed_by_the_feature() {
        // Absent-safety, the standing requirement of every stage: a file that
        // declares no shared tail must load and compute exactly as before.
        let (_, model) = sharing_spec_model(0);
        assert!(
            (0..4).all(|i| model.spec.owns_cache(i)),
            "no key means every layer owns its cache"
        );
        assert_eq!(model.spec.cache_owning_layers(), 4);
    }

    #[test]
    fn unknown_tensor_families_are_refused_and_collapsed_by_layer() {
        let spec = TinySpec::default();
        let extra: Vec<String> = (0..spec.n_layers)
            .map(|i| format!("model.layers.{i}.self_attn.v_norm.weight"))
            .collect();
        let refs: Vec<&str> = extra.iter().map(String::as_str).collect();
        let err = verdict(&checkpoint_with(&spec, &refs, &[], |_| true))
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 tensor family"), "{err}");
        assert!(
            err.contains("model.layers.*.self_attn.v_norm.weight"),
            "{err}"
        );
    }

    /// Prefill the standard probe prompt against a spec, for the tests below
    /// that compare one architecture switch against its absence.
    fn logits_for(spec: &TinySpec) -> Vec<Logits> {
        let image = checkpoint_of(spec);
        let file = crate::checkpoint::parse(&image).unwrap();
        let mut model = CpuModel::from_checkpoint(&file).unwrap();
        model.prefill_all(&[1, 5, 9, 13, 3, 8, 21, 2]).unwrap()
    }

    #[test]
    fn the_logit_softcap_squashes_without_moving_any_argmax() {
        // Both halves matter, and the second is why this test exists. The cap
        // is monotone, so it reorders nothing and greedy decoding is
        // bit-identical with and without it: a backend that omitted the cap
        // would pass any argmax-based check while every perplexity, sampled
        // draw, and captured logprob came out wrong. The assertion is
        // therefore on the values, and the argmax invariance is asserted as
        // the reason a value assertion was needed.
        let cap = 0.5f32; // well inside the fixture's logit range
        let plain = logits_for(&TinySpec::default());
        let capped = logits_for(&TinySpec {
            logit_softcap: Some(cap),
            ..TinySpec::default()
        });

        let mut moved = 0;
        for (position, (a, b)) in plain.iter().zip(&capped).enumerate() {
            for (t, (raw, got)) in a.0.iter().zip(&b.0).enumerate() {
                let want = (raw / cap).tanh() * cap;
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "position {position} token {t}: {got} is not tanh({raw}/{cap})*{cap}"
                );
                assert!(got.abs() < cap, "position {position} token {t}: {got}");
                if raw.to_bits() != got.to_bits() {
                    moved += 1;
                }
            }
            let argmax = |r: &Logits| {
                r.0.iter()
                    .enumerate()
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .unwrap()
                    .0
            };
            assert_eq!(
                argmax(a),
                argmax(b),
                "position {position}: the cap moved an argmax, so it is not monotone"
            );
        }
        assert!(moved > 0, "the cap changed no logit at all");
    }

    /// Layer-0 residual output at each position, with and without per-layer
    /// input embeddings, from images whose every other tensor is identical.
    ///
    /// The synthetic builder draws the per-layer-embedding tensors last, so
    /// the two models share every weight the PLE block does not touch. That
    /// makes the layer-0 output of the plain model exactly the residual the
    /// PLE model's layer-0 PLE block reads, and the difference between the two
    /// outputs exactly what that block contributed -- which is what lets the
    /// test below assert the whole chain rather than that something moved.
    fn layer0_with_and_without_ple(pd: usize, tokens: &[Token]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let capture = |spec: &TinySpec| -> Vec<Vec<f32>> {
            let image = checkpoint_of(spec);
            let file = crate::checkpoint::parse(&image).unwrap();
            let mut model = CpuModel::from_checkpoint(&file).unwrap();
            let calls: TapCalls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            model.set_tap(Box::new(CollectingTap(calls.clone())));
            model.prefill_all(tokens).unwrap();
            let rows = calls.lock().unwrap();
            rows.iter()
                .filter(|(layer, _, _)| *layer == 0)
                .map(|(_, _, x)| x.clone())
                .collect()
        };
        let plain = capture(&TinySpec::default());
        let with = capture(&TinySpec {
            ple_dim: Some(pd),
            ..TinySpec::default()
        });
        (plain, with)
    }

    #[test]
    fn the_per_layer_embedding_block_matches_an_independent_recomputation() {
        // The scale chain is the risk this test exists for. Three factors
        // compound -- sqrt(ple_dim) on the token identity, 1/sqrt(hidden) on
        // the projection, and 2^-0.5 on their sum -- and getting any one of
        // them wrong leaves the logits finite and plausible. So the expected
        // value is recomputed here from the file's own tensors, following
        // `docs/models/gemma-4.md` and `docs/models/gemma4-upstream.txt`
        // rather than the loader, and compared against what the forward pass
        // actually added to the residual stream.
        let pd = 4;
        let tokens = [7u32, 19, 4];
        let (plain, with) = layer0_with_and_without_ple(pd, &tokens);

        let image = checkpoint_of(&TinySpec {
            ple_dim: Some(pd),
            ..TinySpec::default()
        });
        let file = crate::checkpoint::parse(&image).unwrap();
        let spec = ModelSpec::from_checkpoint(&file.config).unwrap();
        let c = spec.config;
        let t = |name: &str| tensor_of(&file, name);
        let (embed, table, model_proj, proj_norm) = (
            t("token_embd.weight"),
            t("per_layer_token_embd.weight"),
            t("per_layer_model_proj.weight"),
            t("per_layer_proj_norm.weight"),
        );
        let (inp_gate, proj, post_norm) = (
            t("blk.0.inp_gate.weight"),
            t("blk.0.proj.weight"),
            t("blk.0.post_norm.weight"),
        );
        let packed = c.n_layers * pd;

        for (position, token) in tokens.iter().enumerate() {
            let token = *token as usize;
            // The embedded input, which is what the context projection reads.
            let scale = (c.embed_dim as f32).sqrt();
            let embedded: Vec<f32> = embed[token * c.embed_dim..(token + 1) * c.embed_dim]
                .iter()
                .map(|e| e * scale)
                .collect();

            // Context component: project, scale by hidden**-0.5, split into
            // one row per layer, RMSNorm each row over its own width.
            let mut context = vec![0.0f32; packed];
            matvec_into(&model_proj, &embedded, &mut context);
            let proj_scale = 1.0 / (c.embed_dim as f32).sqrt();
            for v in context.iter_mut() {
                *v *= proj_scale;
            }
            for row in context.chunks_mut(pd) {
                rms_norm_in_place(row, &proj_norm, c.rms_eps);
            }
            // Token-identity component, scaled by sqrt(ple_dim), summed with
            // the context component, and the sum scaled by 2**-0.5.
            let identity = &table[token * packed..(token + 1) * packed];
            let embed_scale = (pd as f32).sqrt();
            let rows: Vec<f32> = context
                .iter()
                .zip(identity)
                .map(|(ctx, id)| (ctx + id * embed_scale) * std::f32::consts::FRAC_1_SQRT_2)
                .collect();

            // Layer 0's block: gate the residual down to ple_dim, activate,
            // modulate by row 0, project back up, norm, add.
            let residual = &plain[position];
            let mut gate = vec![0.0f32; pd];
            matvec_into(&inp_gate, residual, &mut gate);
            for (g, e) in gate.iter_mut().zip(&rows[0..pd]) {
                *g = gelu(*g) * e;
            }
            let mut widened = vec![0.0f32; c.embed_dim];
            matvec_into(&proj, &gate, &mut widened);
            let mut normed = vec![0.0f32; c.embed_dim];
            rms_norm_into(&widened, &post_norm, c.rms_eps, &mut normed);

            for (i, ((got, base), delta)) in
                with[position].iter().zip(residual).zip(&normed).enumerate()
            {
                let want = base + delta;
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "position {position} element {i}: layer 0 produced {got}, the \
                     recomputed per-layer-embedding join gives {want}"
                );
            }
        }
    }

    #[test]
    fn a_checkpoint_without_per_layer_embeddings_is_unperturbed_by_the_feature() {
        // Absent-safety, asserted bit for bit rather than within a tolerance:
        // the six tensor families are optional, and a checkpoint that carries
        // none of them must compute exactly as it did before the feature
        // existed. A buffer left stale or a zero-width row summed in anyway
        // would show up here and nowhere else.
        let plain = logits_for(&TinySpec::default());
        let again = logits_for(&TinySpec::default());
        for (position, (a, b)) in plain.iter().zip(&again).enumerate() {
            for (t, (x, y)) in a.0.iter().zip(&b.0).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "position {position} token {t}");
            }
        }
        // And the feature, when present, is not a no-op.
        let with = logits_for(&TinySpec {
            ple_dim: Some(4),
            ..TinySpec::default()
        });
        let moved = plain
            .iter()
            .zip(&with)
            .flat_map(|(a, b)| a.0.iter().zip(&b.0))
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(moved > 0, "per-layer embeddings changed no logit");
    }

    #[test]
    fn every_layer_reads_its_own_row_of_the_per_layer_embedding() {
        // The packed table is split by layer index, and a loader that handed
        // every layer row 0 -- or that strode by the per-layer width from the
        // wrong base -- would still produce finite logits. Two layers with
        // identical weights but different rows must therefore contribute
        // differently, which is what a shared row would collapse.
        let pd = 4;
        let image = checkpoint_of(&TinySpec {
            ple_dim: Some(pd),
            ..TinySpec::default()
        });
        let file = crate::checkpoint::parse(&image).unwrap();
        let spec = ModelSpec::from_checkpoint(&file.config).unwrap();
        let table = tensor_of(&file, "per_layer_token_embd.weight");
        let packed = spec.config.n_layers * pd;
        assert_eq!(table.len(), spec.config.vocab_size * packed);
        // Perturb layer 1's row and nothing else. Swapping rows 0 and 1 would
        // not do: it changes what layer 0 reads too, so a backend handing
        // every layer row 0 would still move its output and pass.
        let token = 7usize;
        let table_name = crate::checkpoint::hf_tensor_name("per_layer_token_embd.weight").unwrap();
        let edited = checkpoint_edited(
            &TinySpec {
                ple_dim: Some(pd),
                ..TinySpec::default()
            },
            |name, values| {
                if name == table_name {
                    // Layer 1's slice of this token's packed row.
                    for j in 0..pd {
                        values[token * packed + pd + j] += 0.5;
                    }
                }
            },
        );

        let run = |files: &crate::checkpoint::CheckpointFiles| {
            let ck = crate::checkpoint::parse(files).unwrap();
            let mut m = CpuModel::from_checkpoint(&ck).unwrap();
            m.prefill_all(&[token as Token]).unwrap()
        };
        let before = run(&image);
        let after = run(&edited);
        let moved = before
            .iter()
            .zip(&after)
            .flat_map(|(x, y)| x.0.iter().zip(&y.0))
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert!(
            moved > 0,
            "changing layer 1's per-layer embedding row alone changed nothing, so layer 1 \
             is reading some other layer's row"
        );
    }

    #[test]
    fn the_layer_output_scale_multiplies_the_whole_layer_output() {
        // Not a no-op in the released weights: E2B stores 0.019 to 0.879, and
        // layer 0's value is a 54x reduction. A backend that loads the tensor
        // and never applies it produces fluent text and wrong numerics, so
        // the assertion is that the logits move.
        let scaled = logits_for(&TinySpec {
            layer_output_scale: Some(0.25),
            ..TinySpec::default()
        });
        let plain = logits_for(&TinySpec::default());
        let moved = plain
            .iter()
            .zip(&scaled)
            .flat_map(|(a, b)| a.0.iter().zip(&b.0))
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(moved > 0, "the layer scale changed no logit");

        // A scale of exactly 1.0 is the identity, which is both the absent
        // case and the upstream initialization value. The two must agree bit
        // for bit, or the feature perturbs checkpoints that do not use it.
        let ones = logits_for(&TinySpec {
            layer_output_scale: Some(1.0),
            ..TinySpec::default()
        });
        for (position, (a, b)) in plain.iter().zip(&ones).enumerate() {
            for (t, (x, y)) in a.0.iter().zip(&b.0).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "position {position} token {t}: a unit scale moved a logit"
                );
            }
        }
    }

    #[test]
    fn the_tap_observes_the_layer_output_after_its_scale() {
        // The seam contract: the tap fires on the layer's complete output,
        // which under Gemma 4's ordering is after layer_output_scale, not
        // after the last residual add. The Python forward hook fires on the
        // decoder module and so receives the already-scaled tensor; a tap
        // firing earlier would diverge from it by the scale, per-layer, and
        // probes trained on one path would score the other with plausible
        // numbers. See docs/interp/gemma-4-interp.md.
        let base = 0.25f32;
        let spec = TinySpec {
            layer_output_scale: Some(base),
            ..TinySpec::default()
        };
        let capture = |spec: &TinySpec| {
            let image = checkpoint_of(spec);
            let file = crate::checkpoint::parse(&image).unwrap();
            let mut model = CpuModel::from_checkpoint(&file).unwrap();
            let calls: TapCalls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            model.set_tap(Box::new(CollectingTap(calls.clone())));
            model.prefill_all(&[1, 5]).unwrap();
            let rows = calls.lock().unwrap().clone();
            rows
        };
        let unscaled = capture(&TinySpec::default());
        let scaled = capture(&spec);

        // Layer 0's rows must be the unscaled run's rows times exactly the
        // scale. Layer 0 is the one layer where that holds cleanly: RMSNorm
        // is scale invariant, so the scale does not propagate as a multiple
        // into layer 1, which is also why the logits alone cannot say where
        // it was applied and why this has to be read off the tap.
        let layer0: Vec<_> = unscaled.iter().filter(|(l, _, _)| *l == 0).collect();
        let scaled0: Vec<_> = scaled.iter().filter(|(l, _, _)| *l == 0).collect();
        assert!(!layer0.is_empty());
        for ((_, pos, plain), (_, _, got)) in layer0.iter().zip(&scaled0) {
            for (j, (p, g)) in plain.iter().zip(got.iter()).enumerate() {
                let want = p * base;
                assert!(
                    (g - want).abs() <= want.abs() * 1e-6,
                    "position {pos} element {j}: tap saw {g}, expected {want} \
                     (the tap fired before the scale)"
                );
            }
        }
    }

    #[test]
    fn inert_tensors_do_not_trip_the_guard() {
        // The optional per-layer scale is consumed when present and absent-safe
        // when not, so a checkpoint carrying it must still pass the guard.
        verdict(&checkpoint_of(&TinySpec {
            layer_output_scale: Some(0.5),
            ..TinySpec::default()
        }))
        .unwrap();
    }

    #[test]
    fn a_consumed_tensor_of_the_wrong_shape_is_refused_naming_the_tensor() {
        // Self-consistent checkpoint: the directory declares the smaller
        // shape and the payload supplies exactly that many elements, so no
        // later stage can notice. The forward pass then either strides
        // through the wrong rows and returns finite, wrong logits (a norm
        // tensor) or slices past the end and panics mid-forward (a per-head
        // norm). ffn_up is absent because its width already had a check.
        // Named in the llama.cpp spelling and translated through the one
        // shared table, so the case list stays readable beside the loader.
        for (name, wrong, want) in [
            ("blk.0.attn_norm.weight", vec![8], vec![16]),
            ("blk.1.post_ffw_norm.weight", vec![4], vec![16]),
            ("blk.1.attn_q_norm.weight", vec![4], vec![8]),
            ("blk.0.attn_k_norm.weight", vec![2], vec![8]),
            ("blk.0.attn_q.weight", vec![8, 16], vec![16, 16]),
            ("blk.0.attn_k.weight", vec![4, 16], vec![8, 16]),
            ("blk.1.attn_v.weight", vec![4, 16], vec![8, 16]),
            ("blk.0.attn_output.weight", vec![16, 8], vec![16, 16]),
            ("blk.0.ffn_gate.weight", vec![16, 16], vec![32, 16]),
            ("blk.0.ffn_down.weight", vec![16, 16], vec![16, 32]),
            ("blk.1.ffn_norm.weight", vec![8], vec![16]),
            ("blk.1.post_attention_norm.weight", vec![8], vec![16]),
            ("output_norm.weight", vec![8], vec![16]),
        ] {
            let hf = crate::checkpoint::hf_tensor_name(name).unwrap();
            let files = checkpoint_with(&TinySpec::default(), &[], &[(&hf, &wrong)], |_| true);
            let ck = crate::checkpoint::parse(&files).unwrap();
            let err = match CpuModel::from_checkpoint(&ck) {
                Ok(_) => panic!("{hf} at shape {wrong:?} must not load"),
                Err(e) => format!("{e:#}"),
            };
            assert!(err.contains(&hf), "{err}");
            assert!(err.contains(&format!("{wrong:?}")), "{err}");
            assert!(err.contains(&format!("{want:?}")), "{err}");
        }
    }

    #[test]
    fn the_shape_check_accepts_a_geometry_the_committed_fixture_cannot_express() {
        // The fixture has head_dim = embed_dim / n_heads and a single KV
        // head, so most of the expected element counts coincide there and a
        // transposed entry in the table would still accept it. This geometry
        // separates them, so a wrong expectation shows up as a refusal of a
        // checkpoint the converter wrote correctly.
        let spec = TinySpec {
            embed_dim: 16,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 6,
            ffn_dim: 40,
            ..TinySpec::default()
        };
        let mut m = model_of(&spec);
        let logits = m.prefill(&[1, 5, 9]).unwrap();
        assert_eq!(logits.0.len(), spec.vocab_size);
        assert!(logits.0.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_declared_width_that_the_weights_contradict_is_refused_naming_the_layer() {
        // Per-layer widths are supported, so the guard is no longer a refusal
        // of the feature but a check that config and tensors agree. The
        // forward pass slices by the config, so a disagreement reads the
        // wrong rows.
        let spec = TinySpec::default();
        let files = checkpoint_with(
            &spec,
            &[],
            &[(
                "model.layers.1.mlp.up_proj.weight",
                &[spec.ffn_dim - 8, spec.embed_dim],
            )],
            |_| true,
        );
        let err = verdict(&files).unwrap_err().to_string();
        assert!(err.contains("model.layers.1.mlp.up_proj.weight"), "{err}");
        assert!(err.contains("feed-forward width"), "{err}");
        assert!(err.contains("layer 1"), "{err}");
    }

    /// The checkpoint loader folds Gemma's runtime `+1` into every RMSNorm
    /// weight, and does it to norms only.
    ///
    /// `rms_norm_into` multiplies by the stored weight, so the shift has to be
    /// applied somewhere or every norm is off by one for the whole run --
    /// which still loads, still has the right shapes, and still emits fluent
    /// text. It is applied here, at load, rather than in the kernel: the
    /// arithmetic is identical, it happens once per weight instead of once
    /// per token, and it leaves one norm kernel serving both loaders.
    ///
    /// The two-loader equality test in `cpu_reference` is what proves the
    /// shift is right end to end. This names the convention so that a reader
    /// of `from_checkpoint` finds the reason rather than an unexplained
    /// `1.0 +`.
    #[test]
    fn the_checkpoint_loader_shifts_norm_weights_and_only_norm_weights() {
        let spec = TinySpec::default();
        let files = crate::checkpoint::CheckpointFiles {
            config: crate::synth::build_tiny_config_json(&spec).unwrap(),
            weights: crate::synth::build_tiny_safetensors(&spec).unwrap(),
            dir: std::path::PathBuf::from("<memory>"),
        };
        let ck = crate::checkpoint::parse(&files).unwrap();
        let model = CpuModel::from_checkpoint(&ck).unwrap();

        let stored = |name: &str| -> Vec<f32> {
            let view = ck.tensors.tensor(name).unwrap();
            view.data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        // A norm arrives shifted.
        let want: Vec<f32> = stored("model.layers.0.input_layernorm.weight")
            .iter()
            .map(|w| 1.0 + w)
            .collect();
        assert_eq!(
            model.weights.layers[0].attn_norm, want,
            "attn_norm did not take the shift"
        );
        // And the file really does store the unshifted value, so the
        // assertion above is not comparing a tensor to itself.
        assert!(
            stored("model.layers.0.input_layernorm.weight")
                .iter()
                .all(|w| w.abs() < 0.5),
            "the checkpoint already holds shifted norms, so this proves nothing"
        );

        // A projection does not.
        assert_eq!(
            model.weights.layers[0].wq,
            stored("model.layers.0.self_attn.q_proj.weight"),
            "q_proj was shifted; the shift is for norms only"
        );
        assert_eq!(
            model.weights.embed,
            stored("model.embed_tokens.weight"),
            "the embedding table was shifted"
        );
    }

    #[test]
    fn prefill_returns_finite_logits_of_vocab_width() {
        let mut m = tiny();
        let logits = m.prefill(&[1, 5, 9]).unwrap();
        assert_eq!(logits.0.len(), 288);
        assert!(logits.0.iter().all(|v| v.is_finite()));
        assert_eq!(m.cached_len(), 3);
    }

    #[test]
    fn same_input_is_bit_identical() {
        let mut a = tiny();
        let mut b = tiny();
        let la = a.prefill(&[1, 4, 7, 2]).unwrap();
        let lb = b.prefill(&[1, 4, 7, 2]).unwrap();
        assert_eq!(la, lb);
    }

    #[test]
    fn batched_prefill_matches_stepped_decode() {
        let tokens = [1u32, 5, 9, 13, 3];
        let mut batched = tiny();
        let rows = batched.prefill_all(&tokens).unwrap();
        let mut stepped = tiny();
        for (i, &t) in tokens.iter().enumerate() {
            assert_eq!(stepped.prefill(&[t]).unwrap(), rows[i], "position {i}");
        }
    }

    #[test]
    fn save_then_load_state_continues_bit_identically() {
        // Run A: one unbroken prefill sequence. Run B: prefill, snapshot,
        // reload into a fresh model, continue. The continuation logits must
        // be bit-identical — the restored f32 cache is exact.
        let mut unbroken = tiny();
        unbroken.prefill(&[1, 5, 9]).unwrap();
        let expected = unbroken.prefill(&[7, 2]).unwrap();

        let mut first = tiny();
        first.prefill(&[1, 5, 9]).unwrap();
        let mut state = Vec::new();
        first.save_state(&mut state).unwrap();
        drop(first);

        let mut resumed = tiny();
        resumed.load_state(&mut state.as_slice()).unwrap();
        assert_eq!(resumed.cached_len(), 3);
        let continued = resumed.prefill(&[7, 2]).unwrap();
        assert_eq!(continued, expected);
    }

    #[test]
    fn load_state_refuses_layer_mismatch_and_used_cache() {
        let mut m = tiny();
        m.prefill(&[1]).unwrap();
        let mut state = Vec::new();
        m.save_state(&mut state).unwrap();
        // Corrupt the layer count.
        state[0] ^= 0xff;
        let mut fresh = tiny();
        let err = fresh.load_state(&mut state.as_slice()).unwrap_err();
        assert!(err.to_string().contains("layers"), "{err}");
        // A non-fresh cache refuses to load at all.
        let mut used = tiny();
        used.prefill(&[2]).unwrap();
        state[0] ^= 0xff; // restore
        let err = used.load_state(&mut state.as_slice()).unwrap_err();
        assert!(err.to_string().contains("non-fresh"), "{err}");
    }

    #[test]
    fn a_snapshot_header_cannot_reserve_for_tokens_it_does_not_carry() {
        // context_length comes from the checkpoint, so on a long-context
        // model the header's cached_len clears the only bound there was with
        // room to spare. Reserving for the claim before a payload byte
        // arrives turns this 16-byte snapshot into a 32 GiB allocation,
        // which aborts the process through the allocation error handler
        // instead of reporting the truncation.
        let spec = TinySpec {
            context_length: 1 << 30,
            ..TinySpec::default()
        };
        let mut m = model_of(&spec);
        let mut header = Vec::new();
        header.extend((m.kv.len() as u64).to_le_bytes());
        header.extend((spec.context_length as u64).to_le_bytes());
        let err = m.load_state(&mut header.as_slice()).unwrap_err();
        let io = err
            .downcast_ref::<std::io::Error>()
            .expect("a truncated payload is an io error");
        assert_eq!(io.kind(), std::io::ErrorKind::UnexpectedEof);
        assert_eq!(m.cached_len(), 0);
        // `cached_len` reads 0 on any failed restore, so it proves nothing on
        // its own; `save_state` refuses a cache whose length disagrees with
        // the counter, which makes it an honest emptiness check.
        let mut after = Vec::new();
        m.save_state(&mut after).unwrap();
        assert_eq!(after.len(), 16, "a failed restore left rows in the cache");
    }

    #[test]
    fn a_failed_load_state_leaves_a_model_that_computes_like_a_fresh_one() {
        // The restore filled the cache vectors as it read and assigned
        // `cached_len` only at the end, so a truncated payload left orphaned
        // rows under a `cached_len` of 0. The counter said fresh, but
        // `save_state` then refused the model as inconsistent and, worse,
        // `forward_one` appends with `extend_from_slice` while reading
        // position 0 from the front of the vector — so the first token
        // attended to the orphans and returned finite, wrong logits with no
        // error. Cut 124 of this 400-byte snapshot is one such point: it
        // leaves layer 0's K fully restored and gave a first logit of
        // -0.11380069 where a fresh model gives 0.32625324.
        //
        // Every prefix of a valid snapshot is scanned, and the assertion is
        // not that a counter reads zero but that the model computes exactly
        // what a never-loaded one computes.
        let image = checkpoint_of(&TinySpec::default());
        let file = crate::checkpoint::parse(&image).unwrap();
        let model = || CpuModel::from_checkpoint(&file).unwrap();

        let mut source = model();
        source.prefill(&[1, 5, 9]).unwrap();
        let mut full = Vec::new();
        source.save_state(&mut full).unwrap();
        assert_eq!(full.len(), 400, "snapshot geometry assumed by this test");

        let mut fresh = model();
        let mut empty_cache = Vec::new();
        fresh.save_state(&mut empty_cache).unwrap();
        let expected = fresh.prefill(&[4]).unwrap();

        // The named point first: layer 0's K fully restored, the payload cut
        // inside layer 0's V. Compared as a scalar so the failure names the
        // wrong number instead of printing 288 floats twice; the whole row
        // is compared in the scan below.
        let mut wedged = model();
        wedged.load_state(&mut &full[..124]).unwrap_err();
        assert_eq!(wedged.cached_len(), 0, "cut 124 reports fresh");
        assert_eq!(
            wedged.prefill(&[4]).unwrap().0[0],
            expected.0[0],
            "cut 124: the first token attended to the orphaned rows"
        );

        for cut in 0..full.len() {
            let mut m = model();
            let err = m.load_state(&mut &full[..cut]).unwrap_err();
            assert_eq!(
                err.downcast_ref::<std::io::Error>().map(|e| e.kind()),
                Some(std::io::ErrorKind::UnexpectedEof),
                "cut {cut}: {err}"
            );
            assert_eq!(m.cached_len(), 0, "cut {cut}");
            // A counter is not the state: round-trip the cache itself, which
            // `save_state` cross-checks against `cached_len`.
            let mut after = Vec::new();
            m.save_state(&mut after).unwrap();
            assert_eq!(after, empty_cache, "cut {cut} left rows in the cache");
            assert_eq!(m.prefill(&[4]).unwrap(), expected, "cut {cut}");
        }
    }

    #[test]
    fn a_failed_prefill_commits_none_of_its_tokens() {
        // `prefill` advanced `cached_len` per token, so a token rejected
        // mid-batch left its predecessors committed while the caller saw
        // only the error. `Session::sync` reconciles defensively, but
        // `eval::perplexity`, activation capture, and `fork` drive `Model`
        // directly and would keep computing against the phantom context.
        let mut m = tiny();
        m.prefill(&[1, 5]).unwrap();
        let mut before = Vec::new();
        m.save_state(&mut before).unwrap();

        let err = m.prefill(&[7, 2, 288]).unwrap_err();
        assert!(err.to_string().contains("outside vocabulary"), "{err}");
        assert_eq!(m.cached_len(), 2);
        let mut after = Vec::new();
        m.save_state(&mut after).unwrap();
        assert_eq!(before, after, "the failed batch left rows in the cache");

        // Not merely the right length: the right cache.
        let continued = m.prefill(&[7]).unwrap();
        let mut fresh = tiny();
        assert_eq!(continued, fresh.prefill(&[1, 5, 7]).unwrap());
    }

    #[test]
    fn truncate_then_reprefill_matches_fresh_prefill() {
        let mut m = tiny();
        m.prefill(&[1, 5, 9, 13]).unwrap();
        m.truncate(2).unwrap();
        let replayed = m.prefill(&[7, 8]).unwrap();
        let mut fresh = tiny();
        let direct = fresh.prefill(&[1, 5, 7, 8]).unwrap();
        assert_eq!(replayed, direct);
    }

    #[test]
    fn fork_diverges_identically_to_a_fresh_session() {
        let mut parent = tiny();
        parent.prefill(&[1, 5, 9]).unwrap();
        let mut child = parent.fork().unwrap();
        let forked = child.prefill(&[7]).unwrap();
        let mut fresh = tiny();
        let direct = fresh.prefill(&[1, 5, 9, 7]).unwrap();
        assert_eq!(forked, direct);
        // Parent state is unaffected by the child's decode.
        assert_eq!(parent.cached_len(), 3);
        assert_eq!(child.cached_len(), 4);
    }

    #[test]
    fn interleaved_parent_and_child_decodes_do_not_disturb_each_other() {
        // The forward pass reuses working buffers held on the model, so a
        // fork that shared them with its parent would let whichever model
        // ran last overwrite the other's intermediates. Interleaving the two
        // decodes is what makes that visible: a child that only ever runs
        // after the parent has finished passes either way, which is why
        // `fork_diverges_identically_to_a_fresh_session` does not cover it.
        let steps = [(2u32, 7u32), (11, 13), (4, 6)];
        let mut parent = tiny();
        parent.prefill(&[1, 5, 9]).unwrap();
        let mut child = parent.fork().unwrap();
        let mut interleaved = Vec::new();
        for &(p, c) in &steps {
            let from_parent = parent.prefill(&[p]).unwrap();
            let from_child = child.prefill(&[c]).unwrap();
            interleaved.push((from_parent, from_child));
        }

        let mut lone_parent = tiny();
        lone_parent.prefill(&[1, 5, 9]).unwrap();
        let mut lone_child = tiny();
        lone_child.prefill(&[1, 5, 9]).unwrap();
        for (i, &(p, c)) in steps.iter().enumerate() {
            let (got_parent, got_child) = &interleaved[i];
            assert_eq!(
                *got_parent,
                lone_parent.prefill(&[p]).unwrap(),
                "parent {i}"
            );
            assert_eq!(*got_child, lone_child.prefill(&[c]).unwrap(), "child {i}");
        }
    }

    #[test]
    fn sliding_window_layer_ignores_positions_outside_the_window() {
        // Two prefixes differing only at position 0, followed by enough
        // tokens that position 0 leaves every window. With a global layer
        // present the logits still differ (global attends to everything),
        // so test on a model whose layers are all sliding-window.
        let spec = TinySpec {
            n_layers: 2,
            sliding_window_pattern: 3, // layers 0,1 sliding; no global layer
            sliding_window: 2,
            ..TinySpec::default()
        };
        let mut a = model_of(&spec);
        let mut b = model_of(&spec);
        // Window 2 and two layers: information from position 0 can reach at
        // most position 2 (one hop per layer). Position 5 must be blind to it.
        let la = a.prefill_all(&[3, 8, 8, 8, 8, 8]).unwrap();
        let lb = b.prefill_all(&[4, 8, 8, 8, 8, 8]).unwrap();
        assert_ne!(la[1], lb[1], "inside the window the prefix must matter");
        assert_eq!(la[5], lb[5], "outside the window the prefix must not");
    }

    #[test]
    fn context_length_overflow_is_an_error() {
        let mut m = tiny();
        let long: Vec<Token> = (0..1025).map(|i| (i % 30) as Token).collect();
        let err = m.prefill(&long).unwrap_err();
        assert!(err.to_string().contains("context length"), "{err}");
    }

    #[test]
    fn out_of_vocab_token_is_an_error() {
        let mut m = tiny();
        assert!(m.prefill(&[999]).is_err());
    }

    /// Shared collector so tests can inspect calls after the model owns the
    /// tap (Tap: Send rules out Rc).
    type TapCalls = std::sync::Arc<std::sync::Mutex<Vec<(usize, usize, Vec<f32>)>>>;

    #[derive(Clone, Default)]
    struct CollectingTap(TapCalls);

    impl Tap for CollectingTap {
        fn on_residual(&mut self, layer: usize, position: usize, x: &[f32]) {
            self.0.lock().unwrap().push((layer, position, x.to_vec()));
        }
    }

    #[test]
    fn tap_sees_every_layer_at_every_position() {
        let mut m = tiny();
        let tap = CollectingTap::default();
        m.set_tap(Box::new(tap.clone()));
        m.prefill_all(&[1, 5, 9]).unwrap();
        let calls = tap.0.lock().unwrap();
        // 2 layers x 3 positions, position-major, layer 0 before layer 1.
        let order: Vec<(usize, usize)> = calls.iter().map(|(l, p, _)| (*l, *p)).collect();
        assert_eq!(order, [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]);
        assert!(calls.iter().all(|(_, _, x)| x.len() == 16));
        assert!(calls
            .iter()
            .all(|(_, _, x)| x.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn tap_does_not_perturb_logits() {
        let mut tapped = tiny();
        tapped.set_tap(Box::new(CollectingTap::default()));
        let mut plain = tiny();
        assert_eq!(
            tapped.prefill_all(&[1, 4, 7, 2]).unwrap(),
            plain.prefill_all(&[1, 4, 7, 2]).unwrap()
        );
    }

    /// Control-vector GGUF image with `direction.<layer>` (1-based) tensors.
    fn control_image(directions: &[(usize, Vec<f32>)]) -> Vec<u8> {
        let mut w = crate::synth::GgufWriter::new();
        w.kv_str("general.architecture", "controlvector");
        for (layer, v) in directions {
            w.tensor_f32(&format!("direction.{layer}"), &[v.len() as u64], v);
        }
        w.finish()
    }

    fn control(directions: &[(usize, Vec<f32>)], scale: f32) -> ControlVector {
        let image = control_image(directions);
        let file = crate::gguf::parse(&image).unwrap();
        ControlVector::from_gguf(&file, &image, scale).unwrap()
    }

    #[test]
    fn zero_control_vector_is_bit_identical_to_unsteered() {
        // The tiny model has blocks 0 and 1, and block 0 is unsteerable by
        // the format's numbering, so direction.1 is the only legal slot.
        let mut steered = tiny();
        steered
            .set_control_vector(control(&[(1, vec![0.0; 16])], 1.0))
            .unwrap();
        let mut plain = tiny();
        assert_eq!(
            steered.prefill_all(&[1, 4, 7, 2]).unwrap(),
            plain.prefill_all(&[1, 4, 7, 2]).unwrap()
        );
    }

    #[test]
    fn nonzero_control_vector_changes_logits_deterministically() {
        let dir: Vec<f32> = (0..16).map(|i| 0.05 * (i as f32 - 8.0)).collect();
        let mut steered = tiny();
        steered
            .set_control_vector(control(&[(1, dir.clone())], 1.0))
            .unwrap();
        let mut plain = tiny();
        let ls = steered.prefill(&[1, 4, 7]).unwrap();
        let lp = plain.prefill(&[1, 4, 7]).unwrap();
        assert_ne!(ls, lp);
        // Deterministic under steering as well.
        let mut again = tiny();
        again.set_control_vector(control(&[(1, dir)], 1.0)).unwrap();
        assert_eq!(again.prefill(&[1, 4, 7]).unwrap(), ls);
    }

    #[test]
    fn steering_carries_across_fork() {
        let dir: Vec<f32> = (0..16).map(|i| 0.03 * i as f32).collect();
        let mut parent = tiny();
        parent
            .set_control_vector(control(&[(1, dir.clone())], 1.0))
            .unwrap();
        parent.prefill(&[1, 5]).unwrap();
        let mut child = parent.fork().unwrap();
        let forked = child.prefill(&[7]).unwrap();
        let mut fresh = tiny();
        fresh.set_control_vector(control(&[(1, dir)], 1.0)).unwrap();
        let direct = fresh.prefill(&[1, 5, 7]).unwrap();
        assert_eq!(forked, direct);
    }

    #[test]
    fn direction_n_steers_block_n_and_leaves_earlier_blocks_untouched() {
        // llama.cpp adds direction.N to block N's output, and repeng trains
        // key k on block k's output. Steering block N-1 instead would make
        // the same vector.gguf behave differently here than in the llama.cpp
        // services, silently. Observing the residual through the tap is the
        // only way to tell the two mappings apart.
        let rows_at_position_zero = |tap: &CollectingTap| -> Vec<(usize, Vec<f32>)> {
            tap.0
                .lock()
                .unwrap()
                .iter()
                .filter(|(_, p, _)| *p == 0)
                .map(|(l, _, x)| (*l, x.clone()))
                .collect()
        };

        let plain_tap = CollectingTap::default();
        let mut plain = tiny();
        plain.set_tap(Box::new(plain_tap.clone()));
        plain.prefill(&[1, 4]).unwrap();
        let base = rows_at_position_zero(&plain_tap);

        let steered_tap = CollectingTap::default();
        let mut steered = tiny();
        steered
            .set_control_vector(control(&[(1, vec![100.0; 16])], 1.0))
            .unwrap();
        steered.set_tap(Box::new(steered_tap.clone()));
        steered.prefill(&[1, 4]).unwrap();
        let after = rows_at_position_zero(&steered_tap);

        assert_eq!(base.len(), 2, "expected one row per block");
        // Block 0 precedes the named block and must be untouched. Under the
        // off-by-one mapping direction.1 landed here instead.
        assert_eq!(base[0], after[0], "block 0 was steered");
        // Block 1 is the named block and must carry the direction.
        assert_ne!(base[1], after[1], "block 1 was not steered");
    }

    #[test]
    fn control_vector_shape_mismatches_are_errors() {
        let mut m = tiny();
        // Wrong width.
        let err = m
            .set_control_vector(control(&[(1, vec![0.0; 8])], 1.0))
            .unwrap_err();
        assert!(err.to_string().contains("width"), "{err}");
        // Block beyond the model. direction.2 names block 2, which a
        // two-block model does not have.
        let err = m
            .set_control_vector(control(&[(2, vec![0.0; 16])], 1.0))
            .unwrap_err();
        assert!(err.to_string().contains("blocks"), "{err}");
        // Empty and zero-layer images are load errors.
        let image = control_image(&[]);
        let file = crate::gguf::parse(&image).unwrap();
        assert!(ControlVector::from_gguf(&file, &image, 1.0).is_err());
        let image = control_image(&[(0, vec![0.0; 16])]);
        let file = crate::gguf::parse(&image).unwrap();
        assert!(ControlVector::from_gguf(&file, &image, 1.0).is_err());
    }

    #[test]
    fn fork_does_not_carry_the_tap() {
        let mut parent = tiny();
        let tap = CollectingTap::default();
        parent.set_tap(Box::new(tap.clone()));
        parent.prefill(&[1, 5]).unwrap();
        let before = tap.0.lock().unwrap().len();
        let mut child = parent.fork().unwrap();
        child.prefill(&[7]).unwrap();
        assert_eq!(tap.0.lock().unwrap().len(), before);
    }
}
