/*
 * cpu — hand-rolled CPU reference backend for Gemma-4-style checkpoints.
 *
 * Correctness-first: plain f32 arithmetic, no wall clock, no allocation
 * tricks — deterministic by construction so runs record and replay under rr.
 * This backend is the reference the CUDA backend is diffed against (through
 * the phase 1 flight recorder) and makes the full harness runnable in a
 * sandbox against synthetic weights.
 *
 * Two concessions to speed, and neither is an exception to the rule above
 * because neither reorders any arithmetic. `par_rows` splits matvec and the
 * tied LM head across cores by output row. `Weight::row_dot_block::<R>`
 * evaluates R output rows in one pass, which the compiler places in the R
 * lanes of one vector register — so the loop is SIMD, while each row's fold
 * stays sequential over in_dim in its original order. Both carry their
 * arguments at their definitions. Everything else runs sequentially.
 *
 * A reordered reduction does exist, in `Weight::row_dot_lanes`, and it is not
 * the default. It is 1% faster than the block fold and produces different
 * logits; it stays reachable through `DotKind::Lanes16` because a CUDA kernel
 * will face the same question. See the optimization candidates document for
 * the measurement that rejected it.
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
 * checkpoint carrying either is refused by `check_supported_checkpoint`
 * rather than
 * loaded with the extra structure silently dropped: an ignored weight tensor
 * still produces fluent text, so a silent load would be an undetectable wrong
 * answer. See docs/models/gemma-4.md.
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
 * in docs/models/gemma-4.md.
 *
 * The K/V cache caps sliding layers at `kv::window_capacity` and keeps the
 * full sequence only in global ones. `LayerKv` is a linear compacting buffer,
 * not a ring, for a reason recorded there. Two things about it are worth
 * knowing before touching this file.
 *
 * The cap is `window - 1 + group`, not `window`. A group's first query row
 * attends over the rows before it while the group's own appends have already
 * moved the tail forward, so the retention point is the group's *first*
 * position. Evicting against its last is the trap mistral.rs documents, and
 * `runtime/tests/long_context.rs` exists because it bites at a batch group
 * boundary and nowhere a short prompt reaches.
 *
 * KV sharing turned out not to constrain it. This header used to warn that a
 * donor must retain the full sequence for its shared readers, and that
 * upstream answers the same problem with a separate full-length
 * `shared_kv_states`. The warning was wrong for this backend, and settled by
 * construction rather than by measurement: `kv_donors` pairs a reader with a
 * donor of its own attention regime, and `config.sliding_window` is
 * model-wide, so a sliding reader and its donor always carry the same window
 * and the reader never asks for a row the donor has dropped. What survives of
 * the warning is `ModelSpec::retention_window`, which sizes a donor for the
 * widest of its readers -- because if that ever ceases to hold the symptom
 * changes from a harmless over-read to an out-of-bounds one.
 */

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use anyhow::{bail, ensure, Context, Result};

use super::{Logits, Model, Token};
use crate::align;
use crate::checkpoint::Image;
use crate::kv::LayerKind;
use crate::tensor::{widened_row_dot_block, Weight};

/// Quantities that are one value for the whole model.
///
/// Head dimension, feed-forward width, and the RoPE frequency base are
/// deliberately absent: Gemma 4 varies all three by layer, and a scalar field
/// here is exactly the representation that cannot say so. They live in
/// `LayerSpec`, so a reader has to name the layer it means. See
/// `docs/models/gemma-4.md`.
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
/// `check_supported_checkpoint` and `CpuModel::from_checkpoint` both take
/// this, so the `checkpoint-info` inspection tool and the loader cannot
/// disagree about what a file declares.
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

    /// The window layer `i`'s cache must retain, or `None` if it retains
    /// everything.
    ///
    /// The widest window among layer `i` and every shared layer reading it,
    /// not layer `i`'s own. Today those are always equal -- `kv_donors` pairs
    /// a reader with a donor of its own attention regime, and
    /// `config.sliding_window` is model-wide, so a sliding reader's donor
    /// always carries the same window. The max is here because the
    /// consequence of that ceasing to hold changed with this package. Under
    /// full storage a reader with a wider window than its donor simply read
    /// rows that happened to be present; against a capped cache the donor has
    /// evicted them and the read is out of bounds.
    ///
    /// `None` for a global layer and for any layer a global layer reads: a
    /// global reader's `first` is zero, so its donor can never drop a row.
    pub fn retention_window(&self, i: usize) -> Option<usize> {
        if !self.owns_cache(i) {
            return None;
        }
        let mut widest = None;
        for layer in self.layers.iter() {
            if layer.kv_source != i {
                continue;
            }
            match layer.kind {
                LayerKind::Global => return None,
                LayerKind::SlidingWindow(w) => {
                    widest = Some(widest.map_or(w, |seen: usize| seen.max(w)))
                }
            }
        }
        widest
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
    /// A mapping, not a parse. The checkpoint states what a converted
    /// container only implies: there is no stride to infer and no periodicity
    /// to enforce, no paired `_swa` keys to resolve against their unsuffixed
    /// partners, and no scalar-or-array union to normalize, because
    /// `TextConfig` has already validated its own fields.
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
        // Read once, outside the per-layer closure: the accessor resolves the
        // flat and nested config spellings and is fallible, which a closure
        // returning a plain `LayerSpec` cannot propagate.
        let rope_theta = c.rope_theta()?;
        let rope_local = c.rope_local_base_freq();
        let layers = (0..config.n_layers)
            .map(|i| {
                let (head_dim, rope_freq_base) = match kinds[i] {
                    LayerKind::Global => (c.global_head_dim(), rope_theta),
                    LayerKind::SlidingWindow(_) => (c.head_dim, rope_local.unwrap_or(rope_theta)),
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
/// `LayerWeights` literal in `CpuModel::from_checkpoint`; the fixture-shape
/// test fails if the two drift, because an unlisted consumed tensor makes
/// `check_supported_checkpoint` refuse the synthetic model.
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
/// (`docs/models/gemma-4.md`).
const PLE_LAYER_TENSORS: [&str; 3] = ["inp_gate.weight", "proj.weight", "post_norm.weight"];

/// `per_layer_input_scale`, the factor applied to the sum of the two
/// per-layer-embedding components. Upstream writes it `2.0 ** -0.5`.
const PLE_INPUT_SCALE: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Refuse a checkpoint whose structure the forward pass does not implement.
///
/// The architectural half of the load-time refusal. `TextConfig::validate`
/// covers what the config can state, having named fields to check instead of
/// a metadata namespace to police. What is left here is what the *tensors*
/// have to satisfy, which no config field can state.
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
        // Through `shape` rather than the raw map: the accessor resolves the
        // checkpoint's own spelling, and a direct lookup of the canonical name
        // silently missed on every flat-named checkpoint -- skipping the check
        // on exactly the files it was written to guard.
        let Ok(shape) = ck.shape(&name) else {
            continue;
        };
        let width = shape.first().copied().unwrap_or(0);
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
    // Split what is left in two. A non-text tower is a known part of a
    // multimodal release and the caller can decline it; anything else is
    // structure this backend has never heard of, and no flag accepts it.
    //
    // The two are summarized differently on purpose. An unknown family is a
    // surprise, so every one is named. A tower is a known, whole subsystem: E4B
    // carries 114 non-text families, and listing them turns the refusal into a
    // wall of text that buries the flag the reader needs. The tower prefixes
    // and a family count say the same thing in one line, and
    // `checkpoint-info` prints the full list in its own section for anyone who
    // wants it.
    let mut unconsumed: Vec<String> = Vec::new();
    let mut towers: Vec<&str> = Vec::new();
    let mut tower_families: Vec<String> = Vec::new();
    for name in &names {
        if consumable.contains(name) {
            continue;
        }
        let family = hf_tensor_family(name);
        match NON_TEXT_PREFIXES.iter().find(|p| name.starts_with(**p)) {
            Some(prefix) => {
                if !towers.contains(prefix) {
                    towers.push(prefix);
                }
                if !tower_families.contains(&family) {
                    tower_families.push(family);
                }
            }
            None => {
                if !unconsumed.contains(&family) {
                    unconsumed.push(family);
                }
            }
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
    ensure!(
        towers.is_empty() || ck.text_only,
        "checkpoint is multimodal: it carries {} tensor famil{} in {}, and this backend \
         serves text only. Pass --text-only to load the text stack and decline the rest",
        tower_families.len(),
        if tower_families.len() == 1 {
            "y"
        } else {
            "ies"
        },
        towers.join(", ")
    );
    Ok(())
}

/// Tensor prefixes belonging to a tower this backend does not serve.
///
/// Named rather than pattern-matched on "not the text prefix", so a fourth
/// tower in a future release lands in the unconsumed list and is refused,
/// instead of being silently swept up by `--text-only`. The flag declines
/// what is listed here and nothing else.
const NON_TEXT_PREFIXES: [&str; 4] = [
    "model.audio_tower.",
    "model.vision_tower.",
    "model.embed_audio.",
    "model.embed_vision.",
];

/// Non-text tensor families a checkpoint carries, collapsed and sorted.
///
/// `checkpoint-info` reports these whether or not `--text-only` was passed:
/// inspecting a file should not require agreeing to load it.
pub fn non_text_families(ck: &crate::checkpoint::Checkpoint) -> Vec<String> {
    let mut families: Vec<String> = Vec::new();
    for name in ck.names() {
        if !NON_TEXT_PREFIXES.iter().any(|p| name.starts_with(p)) {
            continue;
        }
        let family = hf_tensor_family(&name);
        if !families.contains(&family) {
            families.push(family);
        }
    }
    families.sort();
    families
}

/// Collapse a Hugging Face tensor name to its family, replacing the layer
/// index in `layers.<i>.` with `*`, so a diagnostic over 42 layers reports one
/// line per family rather than 42.
///
/// Matched on the `layers.` segment rather than a fixed leading prefix,
/// because the towers of a multimodal checkpoint are each a stack of their own
/// -- `model.language_model.layers.<i>.`, `model.audio_tower.layers.<i>.` --
/// and a diagnostic naming the declined towers wants them collapsed too.
pub fn hf_tensor_family(name: &str) -> String {
    const SEGMENT: &str = "layers.";
    let Some(at) = name.find(SEGMENT) else {
        return name.to_string();
    };
    let (head, rest) = name.split_at(at + SEGMENT.len());
    match rest.split_once('.') {
        Some((index, suffix)) if !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()) => {
            format!("{head}*.{suffix}")
        }
        _ => name.to_string(),
    }
}

/// File-level per-layer input embedding weights, read once per position.
struct PleWeights {
    /// Packed token-identity table, `ple_vocab` rows of `n_layers * ple_dim`.
    token_embd: Weight,
    /// Rows in `token_embd`. Derived from the tensor's own shape rather than
    /// from a metadata key: `vocab_size_per_layer_input` is a separate config
    /// field from `vocab_size`, and the GGUF converter deletes it before
    /// writing rather than emitting it (`docs/models/gemma4-upstream.txt`).
    ple_vocab: usize,
    /// Projection of the embedded input to one row per layer.
    model_proj: Weight,
    /// RMSNorm weight applied to each projected row over `ple_dim`.
    proj_norm: Vec<f32>,
}

/// One layer's per-layer input embedding weights.
struct PleLayer {
    inp_gate: Weight,
    proj: Weight,
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
    /// The norms stay f32. Every norm in E4B together is a few megabytes, so
    /// widening them once at load costs nothing worth measuring and leaves
    /// `rms_norm` reading a plain slice. Only the matrices below — the ones
    /// whose bytes decode actually reads per token — are held as stored.
    attn_norm: Vec<f32>,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wo: Weight,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    w_gate: Weight,
    w_up: Weight,
    w_down: Weight,
    post_ffw_norm: Vec<f32>,
}

/// Immutable weights, shared by forks through an Arc.
struct Weights {
    embed: Weight,
    /// File-level per-layer input embedding weights, `None` on a checkpoint
    /// without the feature.
    ple: Option<PleWeights>,
    layers: Vec<LayerWeights>,
    output_norm: Vec<f32>,
}

/// Per-layer KV cache: a linear compacting buffer holding the rows
/// `base .. base + rows` of the sequence.
///
/// Global layers keep `base` at zero and grow with the context. A sliding
/// layer is capped at `kv::window_capacity`, and when it overflows the
/// retained tail is relocated to index 0 and `base` advances -- a memmove, not
/// a wrapped write. That choice is load-bearing rather than incidental: a
/// modular ring would make attention's read order depend on the write cursor,
/// and this backend's whole standing rests on producing the same bits whatever
/// order a prefill was grouped in. Here the stored rows are always contiguous
/// and in natural token order, so the mask arithmetic is unchanged and only
/// the index translation is new.
///
/// The retention point is the *first* query position of the group being
/// appended, never its last. A group's earliest row legitimately attends over
/// rows its latest row has already pushed past, so evicting against the tail
/// drops rows the same forward pass is about to read -- the trap
/// `runtime/tests/long_context.rs` was written to catch, and the reason
/// `kv::window_residency` carries a batch term.
#[derive(Clone, Default)]
struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Absolute sequence position of the row stored at index 0.
    base: usize,
}

impl LayerKv {
    /// Append one group's K/V, first evicting whatever the group's earliest
    /// query row can no longer reach.
    ///
    /// `retain` is the earliest absolute position this layer must still be
    /// able to answer for, and `capacity` the rows it may hold. Compaction
    /// keeps the newest `capacity` rows rather than the bare minimum: the
    /// surplus is the slack that makes the relocation amortized, and it is
    /// also the rewind margin, so cutting to the minimum here would leave a
    /// cache that cannot step back a single position.
    fn append(
        &mut self,
        k: &[f32],
        v: &[f32],
        row: usize,
        end: usize,
        retain: usize,
        capacity: usize,
    ) {
        if capacity > 0 && end.saturating_sub(self.base) > capacity {
            let want = end.saturating_sub(capacity);
            // Never past what the group still needs. Guaranteed by
            // `capacity >= window_residency`, and asserted because a capacity
            // computed from the wrong window would otherwise show up as
            // finite, plausible, wrong logits rather than as a panic.
            debug_assert!(
                want <= retain,
                "capacity {capacity} would evict position {want}, still needed from {retain}"
            );
            let drop = want.min(retain).saturating_sub(self.base);
            if drop > 0 {
                self.k.drain(..drop * row);
                self.v.drain(..drop * row);
                self.base += drop;
            }
        }
        self.k.extend_from_slice(k);
        self.v.extend_from_slice(v);
    }

    /// Drop every row from absolute position `len` onward. `base` is
    /// unchanged: truncation removes the tail, and the caller has already
    /// checked `len` against the rewind floor.
    fn truncate_to(&mut self, len: usize, row: usize) {
        let keep = len.saturating_sub(self.base) * row;
        self.k.truncate(keep);
        self.v.truncate(keep);
    }

    fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.base = 0;
    }
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
/// The residual stream as layer 0 receives it: the embedding row after the
/// `sqrt(embed_dim)` scale.
pub const STAGE_EMBED: &str = "embed";

/// The token-identity half of the per-layer inputs: one packed row of
/// `per_layer_token_embd`, scaled by `sqrt(ple_dim)`.
pub const STAGE_PLE_IDENTITY: &str = "ple_identity";

/// The context half: the embedded input projected to one row per layer,
/// scaled by `1/sqrt(embed_dim)` and normed over each row.
pub const STAGE_PLE_CONTEXT: &str = "ple_context";

/// The two halves summed and scaled by `1/sqrt(2)` -- what the decoder
/// stack consumes.
pub const STAGE_PLE: &str = "ple";

/// Every stage a prelude capture reports, in the order they are computed.
///
/// Ordered because the reader wants the first divergence and the stages are
/// a chain: `ple` is wrong whenever either half is, so reporting it first
/// would name the sum instead of the term.
pub const PRELUDE_STAGES: [&str; 4] = [
    STAGE_EMBED,
    STAGE_PLE_IDENTITY,
    STAGE_PLE_CONTEXT,
    STAGE_PLE,
];

/// Which point within a decoder block a tapped row came from.
///
/// One variant today, and the widening is the whole of it: a tap keyed on the
/// layer alone records "layer 7" for a row whose meaning is fixed by prose
/// somewhere else, and a stored row carries that ambiguity into the artifact.
/// Naming the site costs nothing while there is one of them and cannot be
/// added cheaply once several implementations and a stored format have grown
/// around the narrower key.
///
/// `#[non_exhaustive]`, so a second point is an additive change for a tap in
/// another crate. Within this crate it is not: a `match` here must name every
/// variant, which is the compiler telling each in-crate tap to decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Site {
    /// The block's complete output -- after the post-MLP residual add and the
    /// layer output scale, the point this seam has always fired at.
    LayerOutput,
}

/// The site's name in a stored artifact. Parsed back by `FromStr`, so the two
/// stay one decision.
impl fmt::Display for Site {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Site::LayerOutput => "layer_output",
        })
    }
}

impl std::str::FromStr for Site {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Site> {
        match s {
            "layer_output" => Ok(Site::LayerOutput),
            other => bail!("unknown tap site {other:?}"),
        }
    }
}

pub trait Tap: Send {
    /// A residual row, at `site` of decoder block `layer`, for `position`.
    ///
    /// A tap that wants only one site says so by comparing rather than by
    /// matching: `if site != Site::LayerOutput { return; }`. Within this crate
    /// a `match` with a `_` arm is an unreachable pattern while the enum has
    /// one variant, and the gate runs `-D warnings`.
    fn on_residual(&mut self, layer: usize, site: Site, position: usize, x: &[f32]);

    /// Whether this tap wants the named stages below.
    ///
    /// Asked before the work is done, not after: `PLE_IDENTITY` has no
    /// buffer of its own in the forward pass and materializing it costs an
    /// allocation per position, which a probe capture should not pay for a
    /// diagnostic it never reads.
    fn wants_stages(&self) -> bool {
        false
    }

    /// A named intermediate from before the decoder stack.
    ///
    /// Named rather than one method per point, because localizing a bug is
    /// bisection and bisection does not stop at a depth chosen in advance.
    /// "Layer 0 diverges" turned out to mean "the per-layer embeddings
    /// diverge"; "the per-layer embeddings diverge" means one of four
    /// components, and finding out which needs stages the trait did not have
    /// yesterday. It will need more.
    fn on_stage(&mut self, _stage: &str, _position: usize, _x: &[f32]) {}

    /// Whether this tap wants post-softmax attention weights.
    ///
    /// Asked once per forward pass rather than per head, and defaulting to
    /// false, because this is the first site whose volume is quadratic in the
    /// context: a residual tap records `embed_dim` floats per layer per
    /// position, and an attention tap records up to `n_heads * position`. Opt-in
    /// is therefore per site and not per tap -- a probe monitor pays nothing
    /// for a diagnostic it never reads.
    fn wants_attention(&self) -> bool {
        false
    }

    /// One query head's attention distribution over `first ..= position`.
    ///
    /// The weights are post-softmax, so they are non-negative and sum to one;
    /// the pre-softmax logits are a different site and are not offered here.
    ///
    /// `w` is indexed by *absolute* key position offset from `first`, and
    /// `first` is not recoverable from the vector: a global layer sees
    /// `0 ..= position` while a sliding layer sees only the trailing window, so
    /// two vectors of equal length can name different spans. A consumer that
    /// drops `first` has recorded a histogram it cannot place against the text.
    ///
    /// `head` is a *query* head. Its key-value head is
    /// `head / (n_heads / n_kv_heads)`, the blocked GQA mapping the forward
    /// pass uses; the weights themselves are per query head, so the per-layer
    /// head width -- 256 on a sliding Gemma 4 layer against 512 on a global one
    /// -- does not enter. It is a key count that varies here, not a head width.
    fn on_attention(
        &mut self,
        _layer: usize,
        _position: usize,
        _head: usize,
        _first: usize,
        _w: &[f32],
    ) {
    }
}

/// Tap keeping only the most recent row per requested layer, so after a
/// prefill the surviving rows are the last position's.
///
/// Shared by `activation-capture`, which writes those rows as probe training
/// data, and by `layer-diff`, which compares them against a transformers
/// capture. Both want the same thing for the same reason: the Python forward
/// hook fires once per layer per forward and yields the last position, so a
/// capture that kept every position would have to discard all but one to
/// compare against it.
///
/// Cloneable, and clones share one map: the caller keeps a handle to read the
/// rows after `set_tap` has taken ownership of the model's copy.
///
/// "Most recent" means the highest position seen since the last `clear`, not
/// the last row written. The distinction is invisible while the forward pass
/// is position-major, and load-bearing the moment a batched prefill walks
/// layer-major: under that order the final write for a layer is whichever
/// position the batch happened to finish on. Callers reset per example
/// (`truncate(0)` then `clear`), so positions always restart from zero
/// alongside the cache they index.
/// One retained row per key, carrying the position it came from. Shared
/// between a `LastRowTap` and its clones.
type Held<K> = Arc<Mutex<HashMap<K, (usize, Vec<f32>)>>>;

#[derive(Clone)]
pub struct LastRowTap {
    layers: Vec<usize>,
    /// Position alongside the row, so the comparison is against the position
    /// the row came from rather than against write order.
    rows: Held<usize>,
    /// Whether to also keep the named stages that precede layer 0. Off by
    /// default: `activation-capture` writes probe training data and has no
    /// use for them, and keeping them would copy `n_layers * ple_dim` floats
    /// per position for nobody.
    prelude: bool,
    stages: Held<String>,
}

impl LastRowTap {
    /// The site whose rows this tap keeps.
    ///
    /// Public because the capture manifest records it: a shard that names
    /// only its layers says half of where its rows came from, and the reader
    /// cannot recover the other half from the file.
    pub const SITE: Site = Site::LayerOutput;

    pub fn new(layers: Vec<usize>) -> Self {
        Self {
            layers,
            rows: Arc::new(Mutex::new(HashMap::new())),
            prelude: false,
            stages: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Also keep the named pre-stack stages, `PRELUDE_STAGES`.
    pub fn capturing_prelude(mut self) -> Self {
        self.prelude = true;
        self
    }

    /// A captured stage, or `None` when the prelude was not requested or the
    /// checkpoint does not reach that stage.
    pub fn stage(&self, name: &str) -> Option<Vec<f32>> {
        self.stages
            .lock()
            .expect("tap map not poisoned")
            .get(name)
            .map(|(_, row)| row.clone())
    }

    /// The captured row for `layer`, or `None` when the layer was not
    /// requested or no prefill has run since the last `clear`.
    pub fn row(&self, layer: usize) -> Option<Vec<f32>> {
        self.rows
            .lock()
            .expect("tap map not poisoned")
            .get(&layer)
            .map(|(_, row)| row.clone())
    }

    /// Drop every captured row, between examples.
    pub fn clear(&self) {
        self.rows.lock().expect("tap map not poisoned").clear();
        self.stages.lock().expect("tap map not poisoned").clear();
    }
}

impl Tap for LastRowTap {
    fn on_residual(&mut self, layer: usize, site: Site, position: usize, x: &[f32]) {
        // Keyed by layer alone, so a second site would collide with the first
        // in `rows`: take only the site the capture format names.
        if site != LastRowTap::SITE || !self.layers.contains(&layer) {
            return;
        }
        let mut rows = self.rows.lock().expect("tap map not poisoned");
        match rows.get_mut(&layer) {
            Some(held) if held.0 > position => {}
            Some(held) => {
                held.0 = position;
                held.1.clear();
                held.1.extend_from_slice(x);
            }
            None => {
                rows.insert(layer, (position, x.to_vec()));
            }
        }
    }

    fn wants_stages(&self) -> bool {
        self.prelude
    }

    fn on_stage(&mut self, stage: &str, position: usize, x: &[f32]) {
        if !self.prelude {
            return;
        }
        let mut stages = self.stages.lock().expect("tap map not poisoned");
        match stages.get_mut(stage) {
            Some(held) if held.0 > position => {}
            Some(held) => {
                held.0 = position;
                held.1.clear();
                held.1.extend_from_slice(x);
            }
            None => {
                stages.insert(stage.to_string(), (position, x.to_vec()));
            }
        }
    }
}

/// One captured attention row: the first visible key position, and the
/// distribution over `first ..= position`.
///
/// `first` travels with the weights because it is not recoverable from them.
/// Two rows of equal length can name different spans -- a global layer early
/// in the sequence and a sliding layer past its window -- so a collector that
/// kept only the vector would hold a histogram it could not place.
pub type AttentionRow = (usize, Vec<f32>);

/// Which attention row: layer, query head, query position.
///
/// Head before position so the natural sort of a collected batch groups one
/// head's whole sequence together, which is the order a reader plots.
pub type AttentionKey = (usize, usize, usize);

/// Post-softmax attention weights for chosen layers and query heads.
///
/// The counterpart of `LastRowTap` at the attention site, and deliberately
/// unlike it in what it keeps: `LastRowTap` holds one row per layer because a
/// probe reads the last prompt token, while watching a sliding window against
/// a global layer means watching how the span changes *along* the sequence.
/// So this keeps every position, and the filters are how the cost is bounded
/// instead.
///
/// That cost is quadratic and worth stating in the type's own documentation:
/// a full capture is `4 * n^2 / 2` bytes per (layer, head) pair, so one head
/// of one layer over an 8k context is about 134 MB. Neither filter is
/// optional at the command line for that reason.
#[derive(Clone)]
pub struct AttentionTap {
    layers: Vec<usize>,
    heads: Vec<usize>,
    rows: Arc<Mutex<HashMap<AttentionKey, AttentionRow>>>,
}

impl AttentionTap {
    /// Capture `heads` of `layers`. Both lists are required: see the type's
    /// note on the quadratic volume.
    pub fn new(layers: Vec<usize>, heads: Vec<usize>) -> Self {
        Self {
            layers,
            heads,
            rows: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Every captured row, sorted by (layer, head, position).
    ///
    /// Sorted rather than handed out as a map because the caller writes them
    /// into an array whose row index is the position: an iteration order that
    /// depended on the hasher would write a different file each run from the
    /// same activations.
    pub fn rows(&self) -> Vec<(AttentionKey, AttentionRow)> {
        let held = self.rows.lock().expect("tap map not poisoned");
        let mut out: Vec<_> = held
            .iter()
            .map(|(&(l, h, p), row)| ((l, h, p), row.clone()))
            .collect();
        out.sort_by_key(|&((l, h, p), _)| (l, h, p));
        out
    }

    /// Drop every captured row, between prompts.
    pub fn clear(&self) {
        self.rows.lock().expect("tap map not poisoned").clear();
    }
}

impl Tap for AttentionTap {
    fn on_residual(&mut self, _layer: usize, _site: Site, _position: usize, _x: &[f32]) {}

    fn wants_attention(&self) -> bool {
        true
    }

    fn on_attention(
        &mut self,
        layer: usize,
        position: usize,
        head: usize,
        first: usize,
        w: &[f32],
    ) {
        if !self.layers.contains(&layer) || !self.heads.contains(&head) {
            return;
        }
        // Keyed by all three, so a re-prefill of the same span overwrites
        // rather than accumulating: `truncate(0)` resets the model but not
        // this map, and `clear` is the caller's explicit reset.
        self.rows
            .lock()
            .expect("tap map not poisoned")
            .insert((layer, head, position), (first, w.to_vec()));
    }
}

/// The tensor-name prefix a control-vector image uses, one f32 tensor per
/// steered block.
const BLOCK_PREFIX: &str = "block.";

/// Additive steering directions: a safetensors image holding one 1-D f32
/// tensor per steered block, named `block.<index>`.
///
/// `block.N` steers decoder block `N`, **0-based**, so `block.0` is a legal
/// tensor that steers the first block. `scale * direction` is added to the
/// residual after that block, before the tap fires, so captures observe the
/// steered stream that downstream blocks actually consume.
///
/// What changed at R10 is which indices are legal, not where a direction
/// lands. The retired `direction.` images llama.cpp read also named
/// the 0-based block a direction was added to, but refused index 0: repeng
/// trains key `k` on the output of block `k` and upstream numbers from 1, so
/// the first block was unsteerable for no reason this backend shares. With
/// that consumer gone the floor is dropped, and the naming matches
/// `schemas/probe_artifact.json`'s 0-based `layer` — one convention across
/// the interp artifacts rather than two spellings of the same index.
///
/// The prefix was renamed in the same step, so an image written before R10
/// is refused for holding no blocks rather than loading. Nothing about the
/// mapping moved, so a stale file would otherwise have loaded and steered
/// correctly right up until someone changed the base, and the rename is what
/// makes the format's generation legible at the reader.
pub struct ControlVector {
    scale: f32,
    /// Indexed by 0-based block; None for blocks without a direction.
    directions: Vec<Option<Vec<f32>>>,
}

impl ControlVector {
    /// Parse a control-vector image, keeping only blocks inside `range`.
    ///
    /// `range` is an inclusive `(start, end)` pair of 0-based block indices,
    /// the harness-side spelling of llama.cpp's
    /// `--control-vector-layer-range`. It is applied here, at load, rather
    /// than tested per token in the forward loop: a direction outside the
    /// range is dropped, so the hot path sees only what it must add.
    pub fn from_image(
        image: &Image,
        scale: f32,
        range: Option<(usize, usize)>,
    ) -> Result<ControlVector> {
        if let Some((start, end)) = range {
            ensure!(
                start <= end,
                "control-vector layer range {start}..={end} is empty"
            );
        }
        let mut directions: Vec<Option<Vec<f32>>> = Vec::new();
        for name in image.names() {
            let Some(suffix) = name.strip_prefix(BLOCK_PREFIX) else {
                continue;
            };
            let block: usize = suffix
                .parse()
                .with_context(|| format!("tensor name {name}: non-numeric block index"))?;
            if let Some((start, end)) = range {
                if block < start || block > end {
                    continue;
                }
            }
            // Image::vector refuses a rank other than 1 and widens to f32, so
            // the shape and dtype checks the GGUF reader spelled out by hand
            // are inherited from the one envelope reader.
            let v = image.vector(&name)?;
            if block >= directions.len() {
                directions.resize(block + 1, None);
            }
            directions[block] = Some(v);
        }
        ensure!(
            directions.iter().any(Option::is_some),
            "control vector holds no {BLOCK_PREFIX}<index> tensors{}",
            match range {
                Some((start, end)) => format!(" within layer range {start}..={end}"),
                None => String::new(),
            }
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
    /// Public so the cross-language contract test can assert the block
    /// mapping directly (`schemas/control_vector_artifact.json`). An
    /// off-by-one here steers the wrong block with no error anywhere —
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
    /// One matmul's output in feature-major order, `[out_dim][n]`.
    ///
    /// The batched kernel fixes an output feature and walks the positions, so
    /// the `n` results it produces at once are adjacent here and can be handed
    /// to a worker as one contiguous run. The panel is transposed back to
    /// position-major immediately after, which costs `n * out_dim` moves
    /// against the matmul's `n * out_dim * in_dim` folds — under a tenth of a
    /// percent at E4B's widths, and the price of every elementwise stage
    /// downstream keeping the contiguous per-position slices it has today.
    ///
    /// Sized for one position at load, like the rest of `Scratch`.
    fm: Vec<f32>,
    /// Positions the buffers above are currently sized for. `ensure` grows
    /// them and never shrinks: a model that has prefilled once keeps the
    /// allocation for the next prefill rather than reallocating per batch.
    batch: usize,
}

/// Per-position width of `Scratch::buf`: the residual stream plus every view
/// `forward_batch` carves after it, in carving order — x, normed, q, k, v,
/// attn_out, projected, post, gate, up, mixed, down, ple_gate, ple_out. Sized
/// against the widest layer, not layer 0 or a model-wide scalar: Gemma 4's
/// global layers carry twice the head width of its sliding layers and E2B
/// doubles the feed-forward width over its trailing layers, so a buffer cut to
/// either one alone is short for the other. The carving asserts it never
/// exceeds this.
fn scratch_row(spec: &ModelSpec) -> usize {
    let c = &spec.config;
    let hd = spec.max_head_dim();
    let (q_dim, kv_dim) = (c.n_heads * hd, c.n_kv_heads * hd);
    6 * c.embed_dim + 2 * q_dim + 2 * kv_dim + 3 * spec.max_ffn_dim() + c.ple_dim.unwrap_or(0)
}

/// Widest output any batched matmul produces, which is what `Scratch::fm` is
/// sized by.
///
/// The tied LM head is deliberately absent. It is the one projection whose
/// output is the vocabulary rather than a model width, and reserving
/// `n * vocab_size` would be 268 MB of scratch at E4B and a 256-token batch
/// for a buffer the head does not use: it runs per position through
/// `fill_dots`, for the rows `RowsWanted` actually asks for.
fn panel_row(spec: &ModelSpec) -> usize {
    let c = &spec.config;
    let hd = spec.max_head_dim();
    let ple_rows = c.n_layers * c.ple_dim.unwrap_or(0);
    [
        c.embed_dim,
        c.n_heads * hd,
        c.n_kv_heads * hd,
        spec.max_ffn_dim(),
        ple_rows,
    ]
    .into_iter()
    .max()
    .expect("non-empty")
}

impl Scratch {
    /// Grow the buffers to hold `n` positions in flight.
    ///
    /// Called once per batch rather than per layer, so the batched pass makes
    /// at most three allocations however many layers it walks. Never shrinks:
    /// the decode path asks for one position after a prefill asked for many,
    /// and returning the memory only to reallocate it on the next prefill
    /// would trade a bounded resident set for a per-batch allocation.
    fn ensure(&mut self, spec: &ModelSpec, n: usize) {
        if n <= self.batch {
            return;
        }
        let c = &spec.config;
        self.buf.resize(n * scratch_row(spec), 0.0);
        self.ple
            .resize(n * c.n_layers * c.ple_dim.unwrap_or(0), 0.0);
        self.fm.resize(n * panel_row(spec), 0.0);
        self.batch = n;
    }
}

impl Scratch {
    /// Sized for one position, which is what decode needs. A batched prefill
    /// calls `ensure` to widen the buffers to its own batch.
    fn new(spec: &ModelSpec) -> Scratch {
        let c = &spec.config;
        Scratch {
            buf: vec![0.0; scratch_row(spec)],
            scores: Vec::new(),
            ple: vec![0.0; c.n_layers * c.ple_dim.unwrap_or(0)],
            fm: vec![0.0; panel_row(spec)],
            batch: 1,
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
    /// Width of the row split in `par_rows`. Defaulted from the machine and
    /// carried across forks. Not a numerics parameter: see `par_rows`.
    threads: usize,
    /// Which dot product the matrix-vector products use. Carried across forks
    /// like `threads`, and unlike `threads` it can change the logits.
    dot: DotKind,
    /// Positions per prefill forward pass. Carried across forks, and like
    /// `threads` it cannot change the logits.
    batch: BatchKind,
    /// Forward passes made, once a caller has asked for them.
    ///
    /// `None` until `record_passes` is called, so a run wanting no alignment
    /// pays neither the allocation nor the growth. Shared out through an `Arc`
    /// for the same reason `MonitorState` is: by the time a run ends the model
    /// has been moved into a `Session` and possibly wrapped in a
    /// `RecordingModel`, and the writer that needs the record cannot reach
    /// back through either.
    ///
    /// Not carried across forks: a fork is a separate branch of the
    /// conversation, and merging two branches' pass indices into one sequence
    /// would name passes that never happened in that order.
    passes: Option<Arc<Mutex<Vec<align::Pass>>>>,
}

/// Output rows below which `par_rows` stays serial.
///
/// A row split costs one thread spawn per worker, on the order of ten
/// microseconds, against a saving proportional to the weight bytes the rows
/// cover. E4B's smallest projection is a 2048-wide fold over 2048 output rows
/// and clears this by three orders of magnitude; the Tier T fixture's are a
/// few thousand elements and do not, which is why the fixture goldens still
/// exercise the serial path. Chosen so a matvec must be worth at least a few
/// hundred microseconds before it pays for its workers.
const PARALLEL_MIN_ROWS: usize = 512;

/// Write every element of `out`, splitting the index range across at most
/// `threads` workers.
///
/// **Bit-identical to the serial loop at every width**, and that is the whole
/// reason this sits in the hot path rather than behind a flag. f32 addition is
/// not associative, so splitting a *reduction* across threads changes the
/// result; splitting across *output elements* does not. Each element is still
/// produced by one sequential fold in the original order, and `f` sees the
/// same global index it would have seen serially. The partition is
/// `chunks_mut` over a width derived from `out.len()` and `threads` alone, so
/// it depends on neither the scheduler, nor how many workers the OS actually
/// runs at once, nor the order in which they finish.
///
/// The consequence worth stating: the golden tests pin the parallel path
/// directly. There is no reference fallback to maintain and no tolerance
/// comparison, because there is no second answer.
fn par_rows(out: &mut [f32], threads: usize, f: impl Fn(usize, &mut f32) + Sync) {
    par_row_runs(out, threads, |base, run| {
        for (i, dst) in run.iter_mut().enumerate() {
            f(base + i, dst);
        }
    });
}

/// `par_rows`, handing each worker its whole contiguous run rather than one
/// element at a time.
///
/// Split out when the block folds arrived: a worker that fills rows `R` at a
/// time needs to see the run, not be called per row. The partition is
/// unchanged and still depends only on `out.len()` and `threads`, so
/// everything `par_rows` argues above continues to hold — including for a
/// caller whose block boundaries fall wherever the run happens to start,
/// because a block is bit-identical to the rows it covers folded singly.
fn par_row_runs(out: &mut [f32], threads: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let n = out.len();
    // `threads.min(n)` keeps every worker non-empty, so `chunks_mut` yields
    // exactly the number of chunks intended rather than a trailing empty one.
    let threads = threads.min(n);
    if threads <= 1 || n < PARALLEL_MIN_ROWS {
        f(0, out);
        return;
    }
    let rows_per = n.div_ceil(threads);
    let f = &f;
    std::thread::scope(|s| {
        for (chunk, part) in out.chunks_mut(rows_per).enumerate() {
            let base = chunk * rows_per;
            s.spawn(move || f(base, part));
        }
    });
}

/// Which dot product the forward pass uses for its matrix-vector products.
///
/// An experiment switch, `Widen4` by default. For every kind but one it is a
/// *performance* switch, and for that one it is a *numerics* switch -- which
/// is why a run reports it beside the thread width instead of leaving it
/// implicit, and why `is_bit_identical` exists to tell a caller which it is
/// holding. The default is allowed to be a tuned kind precisely because the
/// tuned kinds are bit-identical: choosing one is choosing a speed, not a
/// number. `Sequential` remains the reference the others are compared to and
/// the kind the goldens were produced under, which is a different job from
/// being the default. `--threads` may be
/// swept freely because every width produces the same bits; this may not.
///
/// The two families differ in what they cost, not only in speed:
///
/// - `Rows4`, `Rows8`, `Rows16` fold that many output rows at once, one
///   accumulator each. **Bit-identical to `Sequential`**, because separate
///   output rows never shared an accumulator; see `Weight::row_dot_block`.
///   Nothing downstream can tell these apart from the default.
/// - `Lanes16` splits the reduction itself into sixteen interleaved partial
///   sums. Deterministic, and by the usual error bounds more accurate, but
///   **not bit-identical** -- the goldens, the snapshots and the recorded
///   replays were all produced under `Sequential`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DotKind {
    /// One sequential fold per output row. The reference the other kinds are
    /// measured and compared against, and the slowest of them -- 1.87x behind
    /// `Rows4` on E4B. Not the default; see `Widen4`.
    Sequential,
    /// Four output rows per block.
    Rows4,
    /// Eight output rows per block.
    Rows8,
    /// Sixteen output rows per block.
    Rows16,
    /// Four output rows per block, widened once instead of once per position.
    ///
    /// The default, because it is the fastest measured kind on both paths at
    /// once: above `WIDEN_MIN_POSITIONS` it takes the hoisted widen (5.2% over
    /// `Rows4` on prefill), and below it routes back to `Rows4`'s kernel, so
    /// decode keeps its 1.87x and pays nothing for the variant it is not
    /// using. One default is therefore correct for prefill and decode both.
    #[default]
    Widen4,
    /// Sixteen interleaved partial sums within one row. Changes the bits.
    Lanes16,
}

impl DotKind {
    /// Every kind, in the order a sweep should walk them.
    pub const ALL: [DotKind; 6] = [
        DotKind::Sequential,
        DotKind::Rows4,
        DotKind::Rows8,
        DotKind::Rows16,
        DotKind::Widen4,
        DotKind::Lanes16,
    ];

    /// The spelling a command line and a run header use.
    pub fn label(self) -> &'static str {
        match self {
            DotKind::Sequential => "sequential",
            DotKind::Rows4 => "rows4",
            DotKind::Rows8 => "rows8",
            DotKind::Rows16 => "rows16",
            DotKind::Widen4 => "widen4",
            DotKind::Lanes16 => "lanes16",
        }
    }

    pub fn parse(s: &str) -> Option<DotKind> {
        DotKind::ALL.into_iter().find(|k| k.label() == s)
    }

    /// Whether this kind produces the same bits as `Sequential`.
    ///
    /// Exposed so a caller that records or compares numerics can say which it
    /// is holding, rather than every such caller re-deriving it from the
    /// variant and one of them getting it wrong.
    pub fn is_bit_identical(self) -> bool {
        self != DotKind::Lanes16
    }
}

/// Which positions of a batched pass need their logits computed.
///
/// The tied LM head is the largest tensor in the checkpoint, so this is not a
/// micro-optimization: `prefill` wants one row and would otherwise fold the
/// embedding table once per prefilled position and discard all but the last.
///
/// Two callers discard rows, and they discard different shapes of them.
/// `prefill` wants the tail alone, which is `Last` and `None`. Perplexity
/// wants a *suffix* — `eval::per_token_nll` scores the second half of each
/// window and reads nothing before it — which is `From`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowsWanted {
    /// Every position from this one on. `From(0)` is every position, which is
    /// what `prefill_all` returns.
    From(usize),
    /// The final position alone.
    Last,
    /// No position: a prefill group before the first row any caller wants.
    None,
}

/// How many positions a prefill puts through the forward pass at once.
///
/// An A/B switch on the `DotKind` pattern, and unlike `DotKind` every variant
/// here is bit-identical: `Stepped` and `Batched(n)` differ in when a dot
/// product runs and which weight rows are resident while it runs, never in
/// what it sums or in what order. `Stepped` is kept as the reference the
/// batched path is measured against, not as a fallback for a numerics
/// difference — there is none to fall back from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchKind {
    /// One position per forward pass. The reference.
    Stepped,
    /// Up to this many positions per forward pass.
    Batched(usize),
    /// The whole prefill in one pass, however long it is.
    Whole,
}

/// Positions per pass by default.
///
/// A cap rather than `Whole` because scratch scales with the batch, and a
/// prefill that filled a long context in one pass would scale it by the
/// context length.
///
/// **64 because the reuse saturates there, measured rather than argued.** On
/// E4B at `--size quick`, batch 64 and batch 256 are the same run — L3 54.88s
/// against 55.02s, inside noise — so quadrupling the reuse buys nothing while
/// costing four times the scratch. This shipped at 256 on the reasoning that
/// amortizing further is free; the reasoning assumed a memory-traffic limit
/// that the same measurement refutes above 64.
const DEFAULT_BATCH: usize = 64;

impl Default for BatchKind {
    fn default() -> BatchKind {
        BatchKind::Batched(DEFAULT_BATCH)
    }
}

impl BatchKind {
    /// The spelling a command line and a run header use: `stepped`, `whole`,
    /// or a positive integer.
    pub fn label(self) -> String {
        match self {
            BatchKind::Stepped => "stepped".to_string(),
            BatchKind::Whole => "whole".to_string(),
            BatchKind::Batched(n) => n.to_string(),
        }
    }

    pub fn parse(s: &str) -> Option<BatchKind> {
        match s {
            "stepped" => Some(BatchKind::Stepped),
            "whole" => Some(BatchKind::Whole),
            _ => match s.parse::<usize>() {
                Ok(0) | Err(_) => None,
                Ok(1) => Some(BatchKind::Stepped),
                Ok(n) => Some(BatchKind::Batched(n)),
            },
        }
    }

    /// Positions per pass for a prefill of `tokens` tokens.
    ///
    /// Public because the KV budget depends on it: a windowed layer holds
    /// `window - 1 + chunk` rows, so a run header reporting cache bytes has to
    /// ask the same question the forward pass does rather than assume the
    /// default.
    pub fn chunk(self, tokens: usize) -> usize {
        match self {
            BatchKind::Stepped => 1,
            BatchKind::Batched(n) => n.min(tokens),
            BatchKind::Whole => tokens,
        }
    }

    /// Whether this kind produces the same bits as `Stepped`. Constant, and
    /// present so a caller recording numerics states the property rather than
    /// assuming it.
    pub fn is_bit_identical(self) -> bool {
        true
    }
}

/// Fill one contiguous run of output rows, `R` at a time.
///
/// `post` runs on each element after its dot, inside the worker: the tied LM
/// head's softcap is elementwise and there is no reason to pay a second pass
/// over the vocabulary for it.
fn fill_run_blocked<const R: usize>(
    w: &Weight,
    x: &[f32],
    base: usize,
    run: &mut [f32],
    post: &(impl Fn(f32) -> f32 + Sync),
) {
    let full = run.len() / R * R;
    let (head, tail) = run.split_at_mut(full);
    for (b, block) in head.chunks_exact_mut(R).enumerate() {
        let mut got = [0.0f32; R];
        w.row_dot_block::<R>(base + b * R, x, &mut got);
        for (dst, g) in block.iter_mut().zip(got) {
            *dst = post(g);
        }
    }
    // Rows past the last full block, which need no rule of their own: the
    // single-row fold gives the same bits the block would have.
    for (i, dst) in tail.iter_mut().enumerate() {
        *dst = post(w.row_dot(base + full + i, x));
    }
}

/// `out[o] = post(dot(W[o], x))` for every `o`, under the selected dot.
///
/// The one place the `DotKind` fan-out lives, so `matvec_into` and the tied
/// head cannot drift into supporting different sets of kinds.
fn fill_dots(
    w: &Weight,
    x: &[f32],
    out: &mut [f32],
    threads: usize,
    dot: DotKind,
    post: impl Fn(f32) -> f32 + Sync,
) {
    // Opened on the caller before the scope spawns and closed after it joins,
    // so the span measures the whole parallel region; the workers carry no
    // span of their own.
    let _kernel = tracing::trace_span!(
        "kernel.lm_head",
        rows = out.len(),
        in_dim = x.len(),
        threads
    )
    .entered();
    match dot {
        DotKind::Sequential => par_rows(out, threads, |o, dst| *dst = post(w.row_dot(o, x))),
        DotKind::Lanes16 => par_rows(out, threads, |o, dst| *dst = post(w.row_dot_lanes(o, x))),
        // One vector, so there is no second position to amortize a widened
        // block across and `Widen4` has nothing to hoist. It folds from stored
        // weights here, which is what `Rows4` already is.
        DotKind::Rows4 | DotKind::Widen4 => {
            par_row_runs(out, threads, |b, run| {
                fill_run_blocked::<4>(w, x, b, run, &post)
            });
        }
        DotKind::Rows8 => {
            par_row_runs(out, threads, |b, run| {
                fill_run_blocked::<8>(w, x, b, run, &post)
            });
        }
        DotKind::Rows16 => {
            par_row_runs(out, threads, |b, run| {
                fill_run_blocked::<16>(w, x, b, run, &post)
            });
        }
    }
}

/// `par_row_runs` over output *features* of a feature-major panel, handing
/// each worker a whole number of `[n]` runs.
///
/// The partition depends only on the feature count, `n` and `threads`, so
/// everything `par_rows` argues about determinism carries over unchanged: a
/// worker's run is a contiguous block of features, no feature is split across
/// workers, and no accumulator is shared.
fn par_feature_runs(
    fm: &mut [f32],
    n: usize,
    threads: usize,
    f: impl Fn(usize, &mut [f32]) + Sync,
) {
    debug_assert_eq!(fm.len() % n, 0, "panel is a whole number of features");
    let feats = fm.len() / n;
    let threads = threads.min(feats);
    if threads <= 1 || feats < PARALLEL_MIN_ROWS {
        f(0, fm);
        return;
    }
    let per = feats.div_ceil(threads);
    let f = &f;
    std::thread::scope(|s| {
        for (chunk, part) in fm.chunks_mut(per * n).enumerate() {
            let base = chunk * per;
            s.spawn(move || f(base, part));
        }
    });
}

/// One feature-major run of a batched matmul, `R` output features at a time.
///
/// **Bit-identical to the per-position matvec, and that is the entire design
/// constraint of the batched pass.** Each output element is still one
/// sequential fold over `in_dim` in index order, because position `p`'s dot
/// against weight row `o` sums the same products in the same order however
/// many positions are in flight. What changes is only when it is computed and
/// what is resident: the `R` weight rows are read once and reused across all
/// `n` positions, which is the whole point — decode reads roughly 14 GiB of
/// weights per position, and a 256-token batch reads them once for 256
/// positions.
///
/// Note what is *not* here: no accumulation across positions, and no
/// `mul_add`. A conventional GEMM would fuse the multiply and add and roughly
/// double achievable peak FLOPS; `mul_add` keeps a wider intermediate and is
/// prohibited workspace-wide, so this kernel forgoes that. A known and
/// accepted cost of bit-identical replay, stated here rather than discovered
/// as a disappointing benchmark.
fn fill_panel_blocked<const R: usize>(
    w: &Weight,
    x: &[f32],
    in_dim: usize,
    n: usize,
    base: usize,
    run: &mut [f32],
) {
    let feats = run.len() / n;
    let full = feats / R * R;
    let (head, tail) = run.split_at_mut(full * n);
    for (b, block) in head.chunks_exact_mut(R * n).enumerate() {
        let o = base + b * R;
        let mut got = [0.0f32; R];
        for p in 0..n {
            w.row_dot_block::<R>(o, &x[p * in_dim..(p + 1) * in_dim], &mut got);
            for (j, g) in got.iter().enumerate() {
                block[j * n + p] = *g;
            }
        }
    }
    // Features past the last full block, which need no rule of their own: the
    // single-row fold gives the same bits the block would have.
    for (i, acc) in tail.chunks_exact_mut(n).enumerate() {
        let o = base + full + i;
        for (p, dst) in acc.iter_mut().enumerate() {
            *dst = w.row_dot(o, &x[p * in_dim..(p + 1) * in_dim]);
        }
    }
}

/// `fill_panel_blocked` with the widening hoisted out of the position loop.
///
/// **The experiment this exists to run.** `fill_panel_blocked` calls
/// `row_dot_block` once per position, and each of those calls widens the same
/// `R` stored weight rows to f32 again. At `n` positions every weight element
/// is converted `n` times to be used `n` times. Batching removed the redundant
/// *reads* of those rows and left the redundant *conversions* untouched. This
/// converts the block once and folds the `n` positions against the result.
///
/// Bit-identical, and not marginally so: `widened_row_dot_block` is the same
/// `block_fold` the stored path runs, over values that widening produces
/// exactly either way. See its doc comment.
///
/// What could make it slower rather than faster, stated because the answer is
/// not obvious: the widened block is twice the bytes of the stored one. At
/// `in_dim` 2048 that is 32 KB against 16 KB and both are L1-resident, but at
/// the FFN widths it is 256 KB against 128 KB and the traffic is real. A win
/// at the attention projections and a loss at the FFN would net out to
/// approximately nothing, which is a result rather than a bug.
///
/// The scratch is allocated per call rather than held in `Scratch` because
/// this is an experiment and the allocation is one per thread per matmul,
/// nowhere near the inner loop. A version that ships hoists it.
fn fill_panel_widened<const R: usize>(
    w: &Weight,
    x: &[f32],
    in_dim: usize,
    n: usize,
    base: usize,
    run: &mut [f32],
) {
    let feats = run.len() / n;
    let full = feats / R * R;
    let (head, tail) = run.split_at_mut(full * n);
    let mut wide = vec![0.0f32; R * in_dim];
    for (b, block) in head.chunks_exact_mut(R * n).enumerate() {
        let o = base + b * R;
        for j in 0..R {
            w.widen_row_into(o + j, &mut wide[j * in_dim..(j + 1) * in_dim]);
        }
        let mut got = [0.0f32; R];
        for p in 0..n {
            widened_row_dot_block::<R>(&wide, &x[p * in_dim..(p + 1) * in_dim], &mut got);
            for (j, g) in got.iter().enumerate() {
                block[j * n + p] = *g;
            }
        }
    }
    // The same remainder rule as the stored path, and the same reason: the
    // single-row fold gives the bits a partial block would have.
    for (i, acc) in tail.chunks_exact_mut(n).enumerate() {
        let o = base + full + i;
        for (p, dst) in acc.iter_mut().enumerate() {
            *dst = w.row_dot(o, &x[p * in_dim..(p + 1) * in_dim]);
        }
    }
}

/// Positions below which `Widen4` folds from stored weights instead.
///
/// Hoisting the conversion only pays if the conversion is reused, and at one
/// position it is not reused at all: decode would widen `R` rows to use each
/// element exactly once, paying a copy and an allocation for nothing. So the
/// decode path takes the stored kernel whatever the flag says.
///
/// **No test defends this number, and none can.** Both kernels produce the
/// same bits, so the threshold is invisible to every correctness test in the
/// tree -- which is exactly what makes it safe to apply automatically rather
/// than expose as a flag, and exactly why it needs saying here. It is a
/// performance switch, and only a measurement can be wrong about it.
const WIDEN_MIN_POSITIONS: usize = 8;

/// `fill_panel_blocked` for the kinds that fold one row at a time.
fn fill_panel_single(
    w: &Weight,
    x: &[f32],
    in_dim: usize,
    n: usize,
    base: usize,
    run: &mut [f32],
    lanes: bool,
) {
    for (i, acc) in run.chunks_exact_mut(n).enumerate() {
        let o = base + i;
        for (p, dst) in acc.iter_mut().enumerate() {
            let xp = &x[p * in_dim..(p + 1) * in_dim];
            *dst = if lanes {
                w.row_dot_lanes(o, xp)
            } else {
                w.row_dot(o, xp)
            };
        }
    }
}

/// `out[p] = W x[p]` for `n` positions at once.
///
/// `x` and `out` are position-major (`[n][dim]`, so every elementwise stage
/// downstream keeps the contiguous per-position slice it has today); `fm` is
/// the feature-major staging panel the kernel writes, transposed back into
/// `out` here. See `Scratch::fm` for why the transpose is affordable.
fn matmul_panel(
    w: &Weight,
    x: &[f32],
    out: &mut [f32],
    fm: &mut [f32],
    n: usize,
    threads: usize,
    dot: DotKind,
) {
    let (in_dim, out_dim) = (x.len() / n, out.len() / n);
    // No projection name: the enclosing block span says which one this is.
    let _kernel = tracing::trace_span!("kernel.matmul", out_dim, in_dim, n, threads).entered();
    assert_eq!(
        w.len(),
        in_dim * out_dim,
        "matmul_panel: weight of {} elements for a {in_dim} x {out_dim} product",
        w.len()
    );
    let fm = &mut fm[..out_dim * n];
    match dot {
        DotKind::Sequential => par_feature_runs(fm, n, threads, |base, run| {
            fill_panel_single(w, x, in_dim, n, base, run, false);
        }),
        DotKind::Lanes16 => par_feature_runs(fm, n, threads, |base, run| {
            fill_panel_single(w, x, in_dim, n, base, run, true);
        }),
        DotKind::Rows4 => par_feature_runs(fm, n, threads, |base, run| {
            fill_panel_blocked::<4>(w, x, in_dim, n, base, run);
        }),
        DotKind::Rows8 => par_feature_runs(fm, n, threads, |base, run| {
            fill_panel_blocked::<8>(w, x, in_dim, n, base, run);
        }),
        DotKind::Rows16 => par_feature_runs(fm, n, threads, |base, run| {
            fill_panel_blocked::<16>(w, x, in_dim, n, base, run);
        }),
        // Below the threshold this is `Rows4`, deliberately and silently. The
        // two agree bit for bit, so nothing downstream can observe the switch;
        // see `WIDEN_MIN_POSITIONS`.
        DotKind::Widen4 if n < WIDEN_MIN_POSITIONS => {
            par_feature_runs(fm, n, threads, |base, run| {
                fill_panel_blocked::<4>(w, x, in_dim, n, base, run);
            })
        }
        DotKind::Widen4 => par_feature_runs(fm, n, threads, |base, run| {
            fill_panel_widened::<4>(w, x, in_dim, n, base, run);
        }),
    }
    for (o, feature) in fm.chunks_exact(n).enumerate() {
        for (p, v) in feature.iter().enumerate() {
            out[p * out_dim + o] = *v;
        }
    }
}

/// `out = W x` with GGUF layout `[in, out]`: rows of `in` contiguous per
/// output element. The output width is `out.len()`, and `out` is written in
/// full, so its previous contents never reach the result.
///
/// The shape check is a real assertion rather than a `debug_assert`: a
/// matrix longer than `in_dim * out_dim` stays in bounds and silently uses
/// its leading rows, so the compiled-out version left release builds — the
/// builds that matter — with no guard at all. `CpuModel::from_checkpoint` now
/// makes this unreachable from a loaded checkpoint, and one integer
/// comparison against `in_dim * out_dim` multiply-accumulates is free. It
/// also catches a `Scratch` buffer sized against the wrong dimension, which
/// would otherwise show up only as wrong logits.
///
/// `threads` bounds the row split and does not change the result; see
/// `par_rows`.
///
/// Test-only since the batched pass arrived, and kept deliberately: it is the
/// straightforward per-position matvec that `matmul_panel` must agree with
/// bit-for-bit, so retiring it would retire the independent statement of what
/// the panel kernel is supposed to compute.
#[cfg(test)]
fn matvec_into(w: &Weight, x: &[f32], out: &mut [f32], threads: usize, dot: DotKind) {
    let (in_dim, out_dim) = (x.len(), out.len());
    assert_eq!(
        w.len(),
        in_dim * out_dim,
        "matvec: weight of {} elements for a {in_dim} x {out_dim} product",
        w.len()
    );
    fill_dots(w, x, out, threads, dot, |v| v);
}

/// Ceiling on the default row-split width, whatever the core count.
///
/// Decode is memory-bandwidth-bound, so workers past the point where the
/// memory controller saturates queue rather than compute, while still paying
/// to be created and joined. The first threaded E4B ladder measured the cost
/// of ignoring that: at width 32 on a 32-core workstation, 4.10x wall-clock
/// against 3.5x more CPU consumed than the serial run, and system time up
/// 107x — 3.95s to 424.64s, which is `clone` and futex and nothing else.
/// Only 14.6 of the 32 cores were ever busy.
///
/// Measured, not adopted. The `SIZE=quick` E4B sweep on the 32-core
/// workstation put the knee at exactly this width:
///
/// | threads | wall | user | cpu |
/// |---|---|---|---|
/// | 4 | 319.44s | 1166.85s | 371% |
/// | 8 | 236.43s | 1548.63s | 672% |
/// | 12 | 237.21s | 2244.08s | 975% |
/// | 16 | 221.72s | 2636.56s | 1231% |
///
/// Twelve is slower than eight while burning 45% more CPU, and sixteen buys 6%
/// of wall clock for 70% more. All four beat the width-32 run's 265.43s.
/// llama.cpp on the same class of machine optimizes near 12, but that is a
/// different engine with a batched prefill, so the number did have to be
/// measured here. Override with `--threads`.
const DEFAULT_MAX_THREADS: usize = 8;

/// Row-split width for a freshly loaded model.
///
/// A hardware-derived default would ordinarily be a determinism defect, and it
/// is not one here — the width does not reach the logits, so a 4-core sandbox
/// and a 32-core workstation produce identical bits from identical inputs.
/// That equivalence is what `a_threaded_forward_pass_produces_bit_identical_logits`
/// pins, and it is what makes a default this code cannot predict acceptable.
/// The same property is why capping it is a free choice: a cap trades speed
/// for nothing else, and never trades away an answer.
///
/// Falls back to serial where the count is unavailable, which is the safe
/// direction: slow rather than wrong about how much parallelism exists.
fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().min(DEFAULT_MAX_THREADS))
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

/// Attention logit for one query head against one cached position.
///
/// The plain dot product, deliberately without the `1/sqrt(head_dim)` an
/// ordinary transformer applies here. Gemma 4 sets `self.scaling = 1.0`
/// (upstream:1181) and passes it explicitly, so upstream's
/// `eager_attention_forward` never reaches its
/// `scaling = module.head_dim**-0.5` fallback (upstream:829-835); `q_norm`
/// and `k_norm` set the magnitude instead.
///
/// A named function rather than an inline `.sum()` so the absence of the
/// divide is a property a test can assert. It went missing for nine packages
/// because nothing could state it.
fn attn_logit(q: &[f32], k: &[f32]) -> f32 {
    q.iter().zip(k).map(|(a, b)| a * b).sum()
}

/// Gemma RMSNorm with no learnable scale, for the value states.
///
/// Upstream builds this as `Gemma4RMSNorm(head_dim, eps, with_scale=False)`
/// (upstream:1201), and `with_scale=False` allocates no `weight` parameter at
/// all. So the norm exists in the computation but contributes nothing to the
/// checkpoint, and every audit that works by comparing tensor names against
/// the model walks straight past it.
///
/// Written as its own function rather than a call with a unit-weight slice:
/// the absence of a weight is the fact worth stating, and allocating 256 ones
/// to multiply by would state the opposite.
fn rms_norm_scaleless_in_place(x: &mut [f32], eps: f32) {
    let inv = rms_inv(x, eps);
    for v in x.iter_mut() {
        *v *= inv;
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
    /// Load a model from a checkpoint directory, holding no payload open.
    ///
    /// The one entry point every binary should use. `from_checkpoint` reads
    /// each tensor from the open file at its recorded offset, so a caller that
    /// keeps `CheckpointFiles` alive afterwards holds a file handle it will
    /// never read again. Nothing above the loader needs it.
    ///
    /// The snapshot fingerprint comes back alongside the model because it is
    /// the one thing a caller *would* have kept the checkpoint open for, and
    /// it is cheap to take here: `checkpoint::fingerprint` reads the
    /// safetensors header, not the weights.
    ///
    /// Written as a function rather than as advice because advice was not
    /// enough. Each binary grew its own three-line load sequence, the map
    /// outlived the load in all of them, and fixing one did not fix the
    /// others -- the measurement was taken with `ladder` while the repair sat
    /// in `harness`. Here the borrow checker settles it: `Checkpoint` borrows
    /// from `CheckpointFiles`, both die at the closing brace, and a caller
    /// cannot hold either.
    pub fn load(model: &std::path::Path, text_only: bool) -> Result<(CpuModel, u64)> {
        // Same name as the obs phase around `build_generator`, on purpose:
        // the phase is written on every run, the span only under a trace
        // subscriber, and each stage below is a child of the span so a trace
        // shows where a slow load spent its time.
        let _load = tracing::info_span!("model.load", text_only).entered();
        let files = {
            let _stage = tracing::debug_span!("checkpoint.read").entered();
            crate::checkpoint::CheckpointFiles::read(model)
                .with_context(|| format!("read model {}", model.display()))?
        };
        let mut ck = {
            let _stage = tracing::debug_span!("checkpoint.parse").entered();
            crate::checkpoint::parse(&files).context("parse checkpoint")?
        };
        ck.text_only = text_only;
        let model = {
            let _stage = tracing::debug_span!("model.from_checkpoint").entered();
            CpuModel::from_checkpoint(&ck).context("load CPU model")?
        };
        let _stage = tracing::debug_span!("checkpoint.fingerprint").entered();
        Ok((model, crate::checkpoint::fingerprint(&files)))
    }

    /// Load from a Hugging Face checkpoint: `config.json` and
    /// `model.safetensors`.
    ///
    /// Tensors are checked by *shape*, not element count. A transposed matrix
    /// has the same element count, so a count check cannot see one, and
    /// nothing downstream would catch it: the forward pass slices by strides
    /// from the config and returns finite, wrong logits.
    ///
    /// RMSNorm weights load as the checkpoint stores them, with no shift.
    ///
    /// This applied `1 + w` until the Tier T fixture measured it. Gemma 1
    /// through Gemma 3 store a parameter initialized to zero and add one at
    /// every use, and that convention was carried forward here; Gemma 4's
    /// `Gemma4RMSNorm` initializes `weight` to `torch.ones` and multiplies by
    /// plain `w`. Adding one to a weight that is already about one scaled
    /// every normalized activation by roughly two.
    ///
    /// llama.cpp's converter agreed, for what that was worth: it writes
    /// `norm_shift` 0.0 for gemma3n and gemma4, so the shift was ours alone.
    /// Nothing in this repository could say so, because `synth.rs` wrote
    /// `v - 1.0` into the fixture and `reference_logits.py` added one back --
    /// one belief, spelled at three ends, agreeing with itself.
    pub fn from_checkpoint(ck: &crate::checkpoint::Checkpoint) -> Result<CpuModel> {
        use crate::checkpoint::hf_tensor_name;

        let spec = ModelSpec::from_checkpoint(&ck.config)?;
        let config = spec.config;
        check_supported_checkpoint(ck, &spec)?;

        // Named by the llama.cpp spelling and translated through the one
        // shared table, so the loader and the fixture writer cannot disagree
        // about a name. That spelling is now only an internal vocabulary --
        // the short names are shorter and every call site already used them,
        // so renaming to the published spelling was not worth the churn.
        let hf = |gguf: &str| -> Result<String> {
            hf_tensor_name(gguf).with_context(|| format!("no published name for tensor {gguf}"))
        };
        let tensor =
            |gguf: &str, want: &[usize]| -> Result<Vec<f32>> { ck.tensor(&hf(gguf)?, want) };
        // The large matrices, kept in the width the file declares rather than
        // widened to f32. Bit-identical either way -- `tensor::Weight` carries
        // the argument -- so the split between this and `tensor` is about
        // bytes resident and bytes read per token, nothing else.
        let weight = |gguf: &str, want: &[usize]| -> Result<Weight> { ck.weight(&hf(gguf)?, want) };
        // A norm weight is the effective multiplier already: see the note on
        // this function. Kept as a named binding rather than folded into
        // `tensor` so the debug assertion still says which tensors are norms.
        let norm = |gguf: &str, want: &[usize]| -> Result<Vec<f32>> {
            debug_assert!(gguf.ends_with("norm.weight"), "{gguf} is not a norm");
            tensor(gguf, want)
        };

        let embed_dim = config.embed_dim;
        let embed = weight("token_embd.weight", &[config.vocab_size, embed_dim])?;
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
                let shape = ck.shape(&name)?;
                let ple_vocab = match shape.as_slice() {
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
                    token_embd: weight(PLE_TOP_LEVEL_TENSORS[0], &[ple_vocab, row])?,
                    ple_vocab,
                    model_proj: weight(PLE_TOP_LEVEL_TENSORS[1], &[row, embed_dim])?,
                    proj_norm: norm(PLE_TOP_LEVEL_TENSORS[2], &[pd])?,
                })
            }
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        for (i, ls) in spec.layers.iter().enumerate() {
            let _layer_span =
                tracing::debug_span!("model.load.layer", layer = i, owns_kv = ls.kv_source == i)
                    .entered();
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
            let w = |suffix: &str, want: &[usize]| weight(&format!("blk.{i}.{suffix}"), want);
            let n = |suffix: &str, want: &[usize]| norm(&format!("blk.{i}.{suffix}"), want);
            let output_scale = match ck.has(&hf(&format!("blk.{i}.layer_output_scale.weight"))?) {
                true => t("layer_output_scale.weight", &[1])?[0],
                false => 1.0,
            };
            let ple_layer = match config.ple_dim {
                Some(pd) => Some(PleLayer {
                    inp_gate: w(PLE_LAYER_TENSORS[0], &[pd, embed_dim])?,
                    proj: w(PLE_LAYER_TENSORS[1], &[embed_dim, pd])?,
                    post_norm: n(PLE_LAYER_TENSORS[2], &[embed_dim])?,
                }),
                None => None,
            };
            // Left unloaded on a shared-KV layer: the modeling code never
            // evaluates them, and an allocated-but-unread weight is the shape
            // of a silent wrong answer. The test
            // `a_shared_layers_own_key_and_value_weights_are_deliberately_unread`
            // holds the decision.
            let (wk, wv, k_norm) = match spec.owns_cache(i) {
                true => (
                    w("attn_k.weight", &[kv_dim, embed_dim])?,
                    w("attn_v.weight", &[kv_dim, embed_dim])?,
                    n("attn_k_norm.weight", &[head_dim])?,
                ),
                // An empty `Weight` needs a variant, and the choice is
                // arbitrary: nothing reads these, and `is_empty` is what the
                // guards ask.
                false => (Weight::F32(Vec::new()), Weight::F32(Vec::new()), Vec::new()),
            };
            layers.push(LayerWeights {
                spec: *ls,
                rope_freqs: rope_frequencies(head_dim, ls.rope_freq_base, divisors),
                output_scale,
                ple: ple_layer,
                attn_norm: n("attn_norm.weight", &[embed_dim])?,
                wq: w("attn_q.weight", &[q_dim, embed_dim])?,
                wk,
                wv,
                q_norm: n("attn_q_norm.weight", &[head_dim])?,
                k_norm,
                wo: w("attn_output.weight", &[embed_dim, q_dim])?,
                post_attn_norm: n("post_attention_norm.weight", &[embed_dim])?,
                ffn_norm: n("ffn_norm.weight", &[embed_dim])?,
                w_gate: w("ffn_gate.weight", &[ffn_dim, embed_dim])?,
                w_up: w("ffn_up.weight", &[ffn_dim, embed_dim])?,
                w_down: w("ffn_down.weight", &[embed_dim, ffn_dim])?,
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
            threads: default_threads(),
            dot: DotKind::default(),
            batch: BatchKind::default(),
            passes: None,
        })
    }

    /// Per-layer geometry, in layer order.
    pub fn layer_specs(&self) -> &[LayerSpec] {
        &self.spec.layers
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Fix the width of the row split, in place of the machine-derived
    /// default. `1` restores the strictly serial path.
    ///
    /// Serial is what rr replay wants, and not for determinism: `par_rows` is
    /// bit-identical at every width, so a replay at four threads produces the
    /// same logits as a recording at one. rr serializes threads onto a single
    /// core regardless, so a parallel recording is slower to take and harder
    /// to read, with nothing bought.
    ///
    /// A width of zero is one, not a panic: a caller computing it from a
    /// subtraction should get the serial path rather than an abort deep in a
    /// decode loop.
    pub fn set_threads(&mut self, n: usize) {
        self.threads = n.max(1);
    }

    /// The width in force, for a runner that reports what it ran.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Select the dot product the matrix-vector products use.
    ///
    /// Unlike `set_threads` this can change the logits: `DotKind::Lanes16`
    /// reorders the reduction. The block kinds do not, and
    /// `DotKind::is_bit_identical` is the check a caller should make rather
    /// than matching the variant itself.
    pub fn set_dot(&mut self, dot: DotKind) {
        self.dot = dot;
    }

    /// The dot in force, for a runner that reports what it ran.
    pub fn dot(&self) -> DotKind {
        self.dot
    }

    /// Select how many positions a prefill puts through the forward pass at
    /// once.
    ///
    /// Like `set_threads` and unlike `set_dot` this cannot change the logits;
    /// see `BatchKind`. It is exposed anyway because it changes the resident
    /// set and the wall clock by a large factor, and a run that reports one
    /// should report the other.
    pub fn set_batch(&mut self, batch: BatchKind) {
        self.batch = batch;
    }

    /// The batch in force, for a runner that reports what it ran.
    pub fn batch(&self) -> BatchKind {
        self.batch
    }

    /// Begin recording forward passes, returning the handle they accumulate
    /// into. Discards any already recorded.
    ///
    /// Opt-in because the record grows with the run and only an alignment
    /// artifact reads it. `prefill_groups` is the one place a pass boundary is
    /// decided, so this is the only honest place to observe one: a caller
    /// above the `Model` seam sees a prefill of 150 tokens and cannot tell how
    /// many passes served it.
    pub fn record_passes(&mut self) -> Arc<Mutex<Vec<align::Pass>>> {
        let handle = Arc::new(Mutex::new(Vec::new()));
        self.passes = Some(Arc::clone(&handle));
        handle
    }

    /// The passes recorded so far, empty when recording was never enabled.
    pub fn passes(&self) -> Vec<align::Pass> {
        match self.passes.as_ref() {
            Some(p) => p.lock().expect("pass record lock").clone(),
            None => Vec::new(),
        }
    }

    /// The absolute positions layer `i` currently holds, as `base..end`.
    ///
    /// `0..0` for a shared-KV layer, which stores nothing. Public because the
    /// storage cap is otherwise unobservable from outside this module: every
    /// other test in the suite compares logits, and logits are exactly what
    /// the cap must not change, so a suite without this accessor would pass
    /// unaltered against a backend that quietly kept the whole sequence. An
    /// optimization nothing can see is an optimization nothing can confirm.
    pub fn cache_extent(&self, i: usize) -> std::ops::Range<usize> {
        let row = self.spec.cache_row(i);
        if row == 0 {
            return 0..0;
        }
        let base = self.kv[i].base;
        base..base + self.kv[i].k.len() / row
    }

    /// How far behind the tip a rewind can still reach once the windows have
    /// filled, or `None` for a model with no capped layer.
    ///
    /// A constant of the configuration rather than of the current cache, which
    /// is what makes it reportable at startup — before any position exists to
    /// ask `rewindable_from` about. The narrowest capped layer sets it, and
    /// the surplus over its window is what the slack in `kv::window_capacity`
    /// buys.
    pub fn rewind_horizon(&self, max_context: usize) -> Option<usize> {
        let group = self.batch.chunk(max_context);
        (0..self.spec.layers.len())
            .filter_map(|i| self.spec.retention_window(i))
            .map(|win| crate::kv::window_capacity(win, group) - (win - 1))
            .min()
    }

    /// Heap bytes the weights occupy, and the width they are held at.
    ///
    /// Reported at startup so the claim that a bf16 checkpoint is not expanded
    /// is checkable from the run rather than from `ps`. `None` for the width
    /// on a checkpoint whose tensors are not all one dtype -- possible in the
    /// format, absent from every checkpoint this loads, and a mixed answer is
    /// better than a confident wrong one.
    pub fn weight_bytes(&self) -> (usize, Option<&'static str>) {
        let w = &self.weights;
        let mut total = 0;
        let mut width: Option<&'static str> = None;
        let mut mixed = false;
        let mut count = |t: &Weight| {
            if t.is_empty() {
                return;
            }
            total += t.stored_bytes();
            let name = match t {
                Weight::F32(_) => "f32",
                Weight::F16(_) => "f16",
                Weight::Bf16(_) => "bf16",
            };
            match width {
                Some(seen) if seen != name => mixed = true,
                _ => width = Some(name),
            }
        };
        count(&w.embed);
        if let Some(p) = w.ple.as_ref() {
            count(&p.token_embd);
            count(&p.model_proj);
        }
        for l in &w.layers {
            for t in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
                count(t);
            }
            if let Some(p) = l.ple.as_ref() {
                count(&p.inp_gate);
                count(&p.proj);
            }
        }
        (total, if mixed { None } else { width })
    }

    /// `Model::fork`'s body, returning the concrete type.
    ///
    /// Inherent rather than written inline in the trait impl so a test can
    /// read the child's fields without a downcast: `fork` hands back a
    /// `Box<dyn Model>`, and the `Model` trait deliberately exposes no
    /// `as_any`. Weights are shared and the cache deep-copied; prefix sharing
    /// stays deferred per the runtime design.
    fn fork_cpu(&self) -> CpuModel {
        CpuModel {
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
            // A subagent decodes as wide as its parent. Not a correctness
            // matter -- the width does not reach the logits -- but a fork that
            // silently reverted to one core would read as the model getting
            // slower after a spawn.
            threads: self.threads,
            dot: self.dot,
            batch: self.batch,
            // Deliberately not inherited. Pass indices are ordinals within one
            // branch, and a fork's passes interleaved with its parent's would
            // describe an order neither branch ran in.
            passes: None,
        }
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

    /// One decode position, as the batch of one.
    ///
    /// Test-only since the batched pass arrived: decode reaches
    /// `forward_batch` through `prefill` of a single token, so a second
    /// production entry point would be a second thing to keep in step. It
    /// survives because several tests are about a position rather than about a
    /// prefill, and read better saying so.
    #[cfg(test)]
    fn forward_one(&mut self, token: Token, position: usize) -> Result<Logits> {
        Ok(self
            .forward_batch(&[token], position, RowsWanted::From(0))?
            .pop()
            .expect("a batch of one returns one row"))
    }

    /// `n` consecutive positions in one pass, starting at absolute position
    /// `start`: append every token's K/V to every layer cache and return the
    /// logits for the positions `want` asks for.
    ///
    /// **Bit-identical to running the positions one at a time.** The argument
    /// is in `fill_panel_blocked`, and the gate is
    /// `batched_prefill_matches_stepped_decode`. Batching changes when each
    /// dot product is computed and which weight rows are resident while it
    /// happens; it changes no summation order, so no bit moves.
    ///
    /// The traversal is layer-major rather than position-major: every position
    /// passes through layer `i` before any reaches layer `i + 1`. That is what
    /// lets the layer's weights be read once for the whole batch, and it is
    /// visible at exactly one seam — a `Tap` sees `(0, 0), (0, 1), ..., (1,
    /// 0)` instead of `(0, 0), (1, 0), (0, 1)`. Every pair still fires exactly
    /// once, and positions still ascend within a layer.
    fn forward_batch(
        &mut self,
        tokens: &[Token],
        start: usize,
        want: RowsWanted,
    ) -> Result<Vec<Logits>> {
        let c = self.config;
        let n = tokens.len();
        assert!(n > 0, "forward_batch of zero tokens");
        // Copied out before the field split below, which borrows `self`
        // mutably and would otherwise put these out of reach.
        let threads = self.threads;
        let dot = self.dot;
        // One atomic increment, so the cache-eviction bounds stay reachable
        // after the field split below hands `self` out piecewise.
        let spec = Arc::clone(&self.spec);
        for &token in tokens {
            self.check_token(token)?;
        }
        self.scratch.ensure(&self.spec, n);
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

        // `x` is the residual stream, one row per position, and outlives the
        // layer loop; everything else is re-carved per layer, because head
        // width and feed-forward width vary by layer and a single carving
        // would size them all to whichever layer happened to be first.
        let Scratch {
            buf,
            scores,
            ple,
            fm,
            ..
        } = scratch;
        let (x_all, work) = buf.split_at_mut(n * c.embed_dim);

        // Asked once for the whole pass, not once per head. A tap cannot be
        // installed or removed while `forward_batch` holds `&mut self`, so the
        // answer cannot change underneath the layer loop, and what is left at
        // the fire site is a compare against a local. That is one predictable
        // test per (position, head) rather than zero -- the alternative is a
        // duplicated head loop, which would buy nothing measurable beside the
        // `hd`-wide dot product already running per key and would put the
        // attention arithmetic in two places.
        let want_attention = tap.as_ref().is_some_and(|t| t.wants_attention());

        // Every span below nests under `model.forward` through the current
        // span on this thread; the block spans end with an explicit drop so
        // their boundaries are the comments that already divide the layer.
        let prelude = tracing::debug_span!("model.prelude", n, ple = w.ple.is_some()).entered();

        // Embedding, scaled by sqrt(embed_dim) per Gemma convention. Widened
        // into the residual buffer and scaled there: one multiply per element,
        // exactly as when the table was already f32.
        let scale = (c.embed_dim as f32).sqrt();
        for (p, &token) in tokens.iter().enumerate() {
            let x = &mut x_all[p * c.embed_dim..(p + 1) * c.embed_dim];
            w.embed.widen_row_into(token as usize, x);
            for dst in x.iter_mut() {
                *dst *= scale;
            }
            if let Some(tap) = tap.as_mut() {
                tap.on_stage(STAGE_EMBED, start + p, x);
            }
        }

        // Per-layer input embeddings, computed once per position from the
        // embedded input and the token identity, before the decoder stack.
        // Layer `i` consumes row `i` of the position's result.
        let ple_stride = c.n_layers * c.ple_dim.unwrap_or(0);
        if let (Some(p_w), Some(pd)) = (w.ple.as_ref(), c.ple_dim) {
            let ple_panel = &mut ple[..n * ple_stride];
            // Context component: project the embedded input to one row per
            // layer, scale, and RMSNorm each row over its own `pd` dimensions.
            //
            // The scale is `hidden_size ** -0.5` and lands immediately before
            // an RMSNorm over the same values, which is scale invariant, so it
            // very nearly cancels — but only very nearly: `rms_eps` is added
            // to the mean square in absolute terms, so the scale does change
            // the result and is applied rather than folded away.
            matmul_panel(&p_w.model_proj, x_all, ple_panel, fm, n, threads, dot);
            let proj_scale = 1.0 / (c.embed_dim as f32).sqrt();
            for v in ple_panel.iter_mut() {
                *v *= proj_scale;
            }
            for row in ple_panel.chunks_mut(pd) {
                rms_norm_in_place(row, &p_w.proj_norm, c.rms_eps);
            }
            let embed_scale = (pd as f32).sqrt();
            for (p, &token) in tokens.iter().enumerate() {
                let rows = &mut ple_panel[p * ple_stride..(p + 1) * ple_stride];
                if let Some(tap) = tap.as_mut() {
                    tap.on_stage(STAGE_PLE_CONTEXT, start + p, rows);
                }
                // Token-identity component: one packed row of the per-layer
                // table, scaled by sqrt(ple_dim) -- x16 at Gemma 4's width of
                // 256, an embed scale that is easy to miss because it lives on
                // the embedding module rather than at the call site
                // (`docs/models/gemma4-upstream.txt`).
                //
                // The row stride is the packed width, not `pd`: the table is
                // one tensor to split into `n_layers` rows, not a stack of
                // per-layer tables.
                debug_assert!(
                    (token as usize) < p_w.ple_vocab,
                    "token bound checked at entry"
                );
                let identity = token as usize * ple_stride;
                // The identity half has no buffer of its own -- it is read
                // straight out of the packed table and summed in place -- so
                // reporting it costs an allocation, taken only when a tap has
                // asked for stages.
                if tap.as_ref().is_some_and(|t| t.wants_stages()) {
                    let scaled: Vec<f32> = (0..ple_stride)
                        .map(|j| p_w.token_embd.get(identity + j) * embed_scale)
                        .collect();
                    let tap = tap.as_mut().expect("checked above");
                    tap.on_stage(STAGE_PLE_IDENTITY, start + p, &scaled);
                }
                for (j, dst) in rows.iter_mut().enumerate() {
                    *dst =
                        (*dst + p_w.token_embd.get(identity + j) * embed_scale) * PLE_INPUT_SCALE;
                }
                if let Some(tap) = tap.as_mut() {
                    tap.on_stage(STAGE_PLE, start + p, rows);
                }
            }
        }
        let ple_rows: &[f32] = ple;
        drop(prelude);

        for (i, layer) in w.layers.iter().enumerate() {
            let hd = layer.spec.head_dim;
            let (q_dim, kv_dim) = (c.n_heads * hd, c.n_kv_heads * hd);
            let ffn_dim = layer.spec.ffn_dim;
            let owns_kv = layer.spec.kv_source == i;
            let _layer_span = tracing::debug_span!(
                "model.layer",
                layer = i,
                n,
                kv_source = layer.spec.kv_source,
                owns_kv
            )
            .entered();
            // Reborrowed per iteration so each layer carves the same storage
            // to its own widths. `Scratch::ensure` reserves for the widest
            // layer and `carve` panics if a view runs short, so a scratch
            // sized against the wrong layer fails here rather than as wrong
            // logits.
            let mut rest: &mut [f32] = &mut *work;
            let normed = carve(&mut rest, n * c.embed_dim);
            let q = carve(&mut rest, n * q_dim);
            let k = carve(&mut rest, n * kv_dim);
            let v = carve(&mut rest, n * kv_dim);
            let attn_out = carve(&mut rest, n * q_dim);
            let projected = carve(&mut rest, n * c.embed_dim);
            let post = carve(&mut rest, n * c.embed_dim);
            let gate = carve(&mut rest, n * ffn_dim);
            let up = carve(&mut rest, n * ffn_dim);
            let mixed = carve(&mut rest, n * ffn_dim);
            let down = carve(&mut rest, n * c.embed_dim);
            let ple_gate = carve(&mut rest, n * c.ple_dim.unwrap_or(0));
            let ple_out = carve(&mut rest, n * c.embed_dim);

            // Attention block.
            let attn_span =
                tracing::debug_span!("model.layer.attn", owns_kv, kv_len = start + n).entered();
            for (xp, np) in x_all
                .chunks_exact(c.embed_dim)
                .zip(normed.chunks_exact_mut(c.embed_dim))
            {
                rms_norm_into(xp, &layer.attn_norm, c.rms_eps, np);
            }
            matmul_panel(&layer.wq, normed, q, fm, n, threads, dot);
            for (p, qp) in q.chunks_exact_mut(q_dim).enumerate() {
                for head in qp.chunks_mut(hd) {
                    rms_norm_in_place(head, &layer.q_norm, c.rms_eps);
                    rope(head, start + p, &layer.rope_freqs);
                }
            }
            // K and V only where this layer owns its cache. A shared-KV layer
            // computes its query and nothing else, then attends over the
            // donor's rows -- which are already post-k_norm and post-rotary,
            // and already carry every position of this batch, because the
            // donor is an earlier layer in this same loop.
            let src = layer.spec.kv_source;
            if src == i {
                matmul_panel(&layer.wk, normed, k, fm, n, threads, dot);
                matmul_panel(&layer.wv, normed, v, fm, n, threads, dot);
                for (p, kp) in k.chunks_exact_mut(kv_dim).enumerate() {
                    for head in kp.chunks_mut(hd) {
                        rms_norm_in_place(head, &layer.k_norm, c.rms_eps);
                        rope(head, start + p, &layer.rope_freqs);
                    }
                }
                // Value states are normed and never rotated: position enters
                // attention through q and k alone (upstream:1253-1254). The
                // norm carries no learnable scale, so it puts no tensor in the
                // checkpoint and no name-based audit of the weights can find
                // it missing -- which is how it went missing.
                for head in v.chunks_mut(hd) {
                    rms_norm_scaleless_in_place(head, c.rms_eps);
                }
                // Every position's K/V lands before any attention at this
                // layer, so the rows an earlier member of the batch appended
                // are visible to a later one -- which is what makes the
                // triangular mask below both necessary and sufficient.
                //
                // It is also why the eviction bound is `start`, the group's
                // first query position, and not `start + n - 1`. Row `start`
                // has not attended yet and still reaches back `win - 1` from
                // where it sits; evicting against the tail would drop rows
                // this very call is about to read.
                let (retain, capacity) = match spec.retention_window(i) {
                    Some(win) => (
                        (start + 1).saturating_sub(win),
                        crate::kv::window_capacity(win, n),
                    ),
                    None => (0, 0),
                };
                kv[i].append(k, v, kv_dim, start + n, retain, capacity);
            }
            // The donor shares this layer's type and therefore its head width,
            // so the reading layer's stride is the donor's stride. Asserted
            // rather than assumed: a donor rule that crossed layer types would
            // read the cache at the wrong stride and produce finite garbage.
            debug_assert_eq!(
                w.layers[src].spec.head_dim, hd,
                "donor layer {src} and reader layer {i} disagree on head width"
            );
            // The same rule for the window, which used to be harmless and is
            // not any more. A reader looking further back than its donor
            // retains once read rows that happened to still be there; against
            // a capped cache those rows are gone and the index below is out of
            // bounds. `retention_window` is what keeps this true, by sizing a
            // donor for the widest of its readers rather than for itself.
            debug_assert!(
                match (layer.spec.kind, w.layers[src].spec.kind) {
                    (_, LayerKind::Global) => true,
                    (LayerKind::Global, LayerKind::SlidingWindow(_)) => false,
                    (LayerKind::SlidingWindow(r), LayerKind::SlidingWindow(d)) => r <= d,
                },
                "reader layer {i} looks further back than donor layer {src} retains"
            );
            let cache = &kv[src];

            // Causal attention over the cache; sliding-window layers see
            // only the trailing `window` positions including this one. The
            // bound comes from the *reading* layer's own kind: sharing supplies
            // the states, it does not change which of them are visible.
            //
            // Both ends of the span are the batch member's own absolute
            // position, which is what makes the mask triangular within the
            // batch rather than square: position `start + p` attends over
            // `first ..= start + p` and never over a later member, even though
            // that member's row is already in the cache.
            //
            // So the grouping `prefill_groups` chose is not observable here,
            // and must not become observable anywhere else. That obligation is
            // stated once, for every backend rather than for this one, in
            // `crate::conformance` -- restating it as a comment on each site
            // that has to honour it is how the copies start to disagree.
            //
            // No 1/sqrt(head_dim) here. Gemma 4 sets `self.scaling = 1.0`
            // (upstream:1181) and passes it explicitly, so upstream's
            // `eager_attention_forward` never reaches the
            // `scaling = module.head_dim**-0.5` fallback it would use for an
            // ordinary transformer (upstream:829-835); `q_norm` and `k_norm`
            // fix the magnitude instead. Dividing here would scale every logit
            // by 1/16 on a 256-wide head and flatten the softmax toward
            // uniform. Stated as a comment rather than a `let scale = 1.0`,
            // which reads as an oversight and invites the divide back.
            attn_out.fill(0.0);
            let attn_kernel =
                tracing::trace_span!("kernel.attention", n, kv_len = start + n, heads = c.n_heads)
                    .entered();
            for p in 0..n {
                let position = start + p;
                let first = match layer.spec.kind {
                    LayerKind::Global => 0,
                    LayerKind::SlidingWindow(win) => (position + 1).saturating_sub(win),
                };
                for qh in 0..c.n_heads {
                    let kvh = qh / heads_per_kv;
                    let qv = &q[p * q_dim + qh * hd..p * q_dim + (qh + 1) * hd];
                    // `first ..= position` stays absolute -- the mask is
                    // easier to read against the sequence than against the
                    // buffer -- and `- cache.base` is the one translation into
                    // storage. A windowed cache has relocated its tail, so row
                    // `t` sits at `t - base`; a global cache has `base == 0`
                    // and the subtraction is a no-op.
                    scores.clear();
                    scores.extend((first..=position).map(|t| {
                        let r = (t - cache.base) * kv_dim;
                        let kt = &cache.k[r + kvh * hd..r + (kvh + 1) * hd];
                        attn_logit(qv, kt)
                    }));
                    softmax_in_place(scores);
                    // Between the softmax and the weighted-V fold, which is
                    // the only point where the distribution exists as a value.
                    // The slice is borrowed straight out of `Scratch::scores`,
                    // so an installed tap costs a dynamic dispatch and no copy,
                    // and the arithmetic below reads the same buffer it would
                    // have read with no tap present.
                    if want_attention {
                        if let Some(t) = tap.as_mut() {
                            t.on_attention(i, position, qh, first, scores);
                        }
                    }
                    let out = &mut attn_out[p * q_dim + qh * hd..p * q_dim + (qh + 1) * hd];
                    for (s, t) in scores.iter().zip(first..=position) {
                        let r = (t - cache.base) * kv_dim;
                        let vt = &cache.v[r + kvh * hd..r + (kvh + 1) * hd];
                        for (o, val) in out.iter_mut().zip(vt) {
                            *o += s * val;
                        }
                    }
                }
            }
            drop(attn_kernel);
            matmul_panel(&layer.wo, attn_out, projected, fm, n, threads, dot);
            for (pr, po) in projected
                .chunks_exact(c.embed_dim)
                .zip(post.chunks_exact_mut(c.embed_dim))
            {
                rms_norm_into(pr, &layer.post_attn_norm, c.rms_eps, po);
            }
            for (a, b) in x_all.iter_mut().zip(post.iter()) {
                *a += b;
            }
            drop(attn_span);

            // GeGLU MLP block.
            let mlp_span = tracing::debug_span!("model.layer.mlp", ffn_dim).entered();
            for (xp, np) in x_all
                .chunks_exact(c.embed_dim)
                .zip(normed.chunks_exact_mut(c.embed_dim))
            {
                rms_norm_into(xp, &layer.ffn_norm, c.rms_eps, np);
            }
            matmul_panel(&layer.w_gate, normed, gate, fm, n, threads, dot);
            matmul_panel(&layer.w_up, normed, up, fm, n, threads, dot);
            for ((m, g), u) in mixed.iter_mut().zip(gate.iter()).zip(up.iter()) {
                *m = gelu(*g) * u;
            }
            matmul_panel(&layer.w_down, mixed, down, fm, n, threads, dot);
            for (dn, po) in down
                .chunks_exact(c.embed_dim)
                .zip(post.chunks_exact_mut(c.embed_dim))
            {
                rms_norm_into(dn, &layer.post_ffw_norm, c.rms_eps, po);
            }
            for (a, b) in x_all.iter_mut().zip(post.iter()) {
                *a += b;
            }
            drop(mlp_span);

            // Per-layer input embedding block: a third residual update, after
            // the feed-forward block rather than before or inside attention.
            // The gate narrows the residual stream to `ple_dim`, the embedding
            // modulates it there elementwise, and the projection widens it
            // back (`docs/models/gemma4-upstream.txt`).
            if let (Some(pl), Some(pd)) = (layer.ple.as_ref(), c.ple_dim) {
                let _ple_span = tracing::debug_span!("model.layer.ple", ple_dim = pd).entered();
                matmul_panel(&pl.inp_gate, x_all, ple_gate, fm, n, threads, dot);
                for (p, gp) in ple_gate.chunks_exact_mut(pd).enumerate() {
                    let row = &ple_rows[p * ple_stride + i * pd..p * ple_stride + (i + 1) * pd];
                    for (g, e) in gp.iter_mut().zip(row) {
                        *g = gelu(*g) * e;
                    }
                }
                matmul_panel(&pl.proj, ple_gate, ple_out, fm, n, threads, dot);
                for (po_in, po) in ple_out
                    .chunks_exact(c.embed_dim)
                    .zip(post.chunks_exact_mut(c.embed_dim))
                {
                    rms_norm_into(po_in, &pl.post_norm, c.rms_eps, po);
                }
                for (a, b) in x_all.iter_mut().zip(post.iter()) {
                    *a += b;
                }
            }

            // The layer's last operation, scaling everything the layer
            // produced including the residual carried through it. Upstream
            // applies it after all of the layer's residual updates and
            // returns immediately, so it is the boundary between this layer
            // and the next.
            if layer.output_scale != 1.0 {
                for a in x_all.iter_mut() {
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
                let cv_scale = control.as_ref().expect("checked above").scale;
                for xp in x_all.chunks_exact_mut(c.embed_dim) {
                    for (a, b) in xp.iter_mut().zip(dir) {
                        *a += cv_scale * b;
                    }
                }
            }

            // Interp tap seam: once per position, in ascending order. The
            // call names the site; that it lands after layer_output_scale
            // rather than after the last residual add is the Python-parity
            // constraint, pinned by
            // the_tap_observes_the_layer_output_after_its_scale.
            if let Some(tap) = tap.as_mut() {
                for (p, xp) in x_all.chunks_exact(c.embed_dim).enumerate() {
                    tap.on_residual(i, Site::LayerOutput, start + p, xp);
                }
            }
        }

        // Final norm and tied LM head: logits[t] = embed_row(t) . x.
        //
        // Only for the positions `want` asks for, because the head is the tied
        // embedding table -- 1.07 GB at E4B, about 7 percent of the weights but
        // 84 percent of a scored window's weight traffic, since this form reads
        // the whole table once per row it produces. Not computing a row is
        // therefore the cheapest thing that can be done about it, and `want` is
        // how a caller says which rows it will actually read.
        //
        // Still per position rather than as a panel, and that is now a measured
        // choice rather than an absent caller. Panelizing would take the table
        // reads from one per position to one per group, but it leaves the
        // arithmetic untouched, and the head is only 7.6 percent of a scored
        // window's flops against a body that is already batched. The estimate
        // is 1.05-1.12x for a chunked staging panel on the bit-identity
        // critical path; see the rust-harness optimization candidates document,
        // which carries the roofline and the design.
        let mut rest: &mut [f32] = work;
        let normed = carve(&mut rest, n * c.embed_dim);
        let rows: Vec<usize> = match want {
            RowsWanted::From(first) => (first.min(n)..n).collect(),
            RowsWanted::Last => vec![n - 1],
            RowsWanted::None => Vec::new(),
        };
        let _head_span = tracing::debug_span!("model.lm_head", rows = rows.len(), n).entered();
        for &p in &rows {
            let (xp, np) = (
                &x_all[p * c.embed_dim..(p + 1) * c.embed_dim],
                &mut normed[p * c.embed_dim..(p + 1) * c.embed_dim],
            );
            rms_norm_into(xp, &w.output_norm, c.rms_eps, np);
        }
        let hidden: &[f32] = normed;
        let mut out = Vec::with_capacity(rows.len());
        for &p in &rows {
            // Row-split like every other projection, and for a larger share of
            // the token than any of them: the tied head folds the whole
            // embedding table once per position. Left serial it would be the
            // Amdahl floor under the rest of the threading.
            let mut logits = vec![0.0f32; c.vocab_size];
            // Softcap, when the checkpoint declares one: a smooth squash
            // toward +/-cap rather than a clamp. Monotone, so it moves no
            // argmax and greedy decoding cannot detect its absence --
            // perplexity, sampled decoding, and every captured logprob can,
            // which is why omitting it is a silent defect and why its tests
            // are not written against argmax.
            //
            // Applied per element inside the worker rather than in a pass of
            // its own: it is elementwise, and a second walk of the vocabulary
            // to run a few hundred thousand `tanh` calls serially would cost
            // more than the cap does.
            fill_dots(
                &w.embed,
                &hidden[p * c.embed_dim..(p + 1) * c.embed_dim],
                &mut logits,
                threads,
                dot,
                |logit| match c.logit_softcap {
                    Some(cap) => (logit / cap).tanh() * cap,
                    None => logit,
                },
            );
            out.push(Logits(logits));
        }
        Ok(out)
    }

    /// `first` is the first appended position whose logits the caller will
    /// read, and is ignored when `last_only` is set — `prefill` wants the tail
    /// alone, which is not a suffix of length greater than one.
    fn prefill_groups(
        &mut self,
        tokens: &[Token],
        last_only: bool,
        first: usize,
    ) -> Result<Vec<Logits>> {
        ensure!(!tokens.is_empty(), "prefill of zero tokens");
        ensure!(
            first <= tokens.len(),
            "prefill from position {first} of only {} tokens",
            tokens.len()
        );
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
        let chunk = self.batch.chunk(tokens.len());
        let groups = tokens.len().div_ceil(chunk);
        // `groups` is on the span because the spans inside it are per group,
        // not per forward: a prefill wider than the batch runs the layer
        // stack once per group, and a trace reader relating `model.layer` to
        // `model.forward` needs the count without knowing the batch width.
        let span = tracing::info_span!(
            "model.forward",
            start = self.cached_len,
            len = tokens.len(),
            groups
        );
        let _enter = span.enter();
        let entry = self.cached_len;
        let mut rows = Vec::with_capacity(tokens.len() - first);
        for (g, group) in tokens.chunks(chunk).enumerate() {
            let start = self.cached_len;
            // `first` is an index into the whole call; each group needs it
            // relative to its own span. A group entirely before it wants
            // nothing, one entirely after it wants everything, and the group
            // straddling it wants a suffix.
            let off = g * chunk;
            let want = match (last_only, g + 1 == groups) {
                (true, true) => RowsWanted::Last,
                (true, false) => RowsWanted::None,
                (false, _) if off + group.len() <= first => RowsWanted::None,
                (false, _) => RowsWanted::From(first.saturating_sub(off)),
            };
            // Backstop for any error a future forward pass raises: roll the
            // cache back to the length prefill was called at, so a failed
            // batch leaves the model where it found it — including the groups
            // that already succeeded.
            //
            // A capped cache cannot always do that. Once a later group has
            // evicted, the rows `entry` names are gone and no rollback
            // restores them, so the fallback is a full clear. That is a
            // weakening of the `Model::prefill` contract and it is stated
            // there: all of the tokens land, or the cache is left at some
            // prefix of what it held, possibly empty. `Session::sync`
            // reconciles against `cached_len` rather than trusting its own
            // mirror, which is what makes an empty cache recoverable instead
            // of a desync — and is why that reconciliation is not redundant.
            let logits = match self.forward_batch(group, start, want) {
                Ok(logits) => logits,
                Err(e) => {
                    if self.truncate(entry).is_err() {
                        self.truncate(0)
                            .expect("a full clear is always within the rewind floor");
                    }
                    return Err(e);
                }
            };
            // Recorded after the pass succeeded, so a rolled-back group leaves
            // no trace of positions the cache no longer holds.
            if let Some(handle) = self.passes.as_ref() {
                let mut passes = handle.lock().expect("pass record lock");
                let index = passes.len();
                passes.push(align::Pass {
                    index,
                    start,
                    len: group.len(),
                });
            }
            self.cached_len += group.len();
            rows.extend(logits);
        }
        Ok(rows)
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
        // `last_only`, so the groups before the final one skip the tied LM
        // head entirely rather than folding the embedding table for logits
        // this caller is about to drop.
        Ok(self
            .prefill_groups(tokens, true, 0)?
            .pop()
            .expect("non-empty prefill"))
    }

    fn prefill_all(&mut self, tokens: &[Token]) -> Result<Vec<Logits>> {
        self.prefill_from(tokens, 0)
    }

    fn prefill_from(&mut self, tokens: &[Token], first: usize) -> Result<Vec<Logits>> {
        self.prefill_groups(tokens, false, first)
    }

    fn rewindable_from(&self) -> usize {
        self.kv
            .iter()
            .enumerate()
            .filter_map(|(i, cache)| {
                let win = self.spec.retention_window(i)?;
                // A layer that has never evicted holds position 0 and bounds
                // nothing. One that has holds `base` onward, and the earliest
                // position it can still *answer* for is `win - 1` later --
                // that position's own window is the first that fits entirely
                // inside what survives.
                (cache.base > 0).then(|| cache.base + win - 1)
            })
            .max()
            .unwrap_or(0)
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        ensure!(len <= self.cached_len, "truncate beyond cached length");
        // A full clear is always legal and bypasses the floor: it is a reset,
        // not a rewind, and several callers depend on being able to hand back
        // a genuinely fresh model -- `Session`'s restore-mismatch recovery
        // most of all, where returning an error would leave exactly the skew
        // the recovery exists to prevent.
        if len == 0 {
            for cache in self.kv.iter_mut() {
                cache.clear();
            }
            self.cached_len = 0;
            return Ok(());
        }
        // Strictly below, not at. After a single-token decode the floor is
        // exactly `cached_len - 1`, which is what `Session::ensure_tail_logits`
        // truncates to on every restore and every invalidated tail; a `<=`
        // here would refuse the ordinary replay rather than an overreach.
        let floor = self.rewindable_from();
        ensure!(
            len >= floor,
            "truncate to {len} is past the sliding window: this cache can no longer \
             return to any position before {floor}, because the rows are evicted \
             rather than masked. Ask `rewindable_from` before rewinding, or clear \
             the cache with `truncate(0)` and re-prefill"
        );
        for (i, cache) in self.kv.iter_mut().enumerate() {
            cache.truncate_to(len, self.spec.cache_row(i));
        }
        self.cached_len = len;
        Ok(())
    }

    fn fork(&self) -> Result<Box<dyn Model>> {
        // Weights shared, cache deep-copied: correctness first, prefix
        // sharing stays deferred per the runtime design.
        Ok(Box::new(self.fork_cpu()))
    }

    // Cache layout on disk: n_layers u64, cached_len u64, then per layer a
    // `base` u64 followed by the K vector and the V vector as little-endian
    // f32.
    //
    // `base` is why the per-layer header exists. Lengths used to be implied by
    // `cached_len`, on the reasoning that the backend stored the full sequence
    // in every layer it stored at all; capping sliding layers at the window
    // falsified that, and a layer's row count is now `cached_len - base`.
    // Writing it per layer rather than deriving it from the spec keeps the
    // reader honest about a payload it did not produce, and keeps a restored
    // cache reporting the same `rewindable_from` the saved one did -- a
    // snapshot that silently re-based to zero would claim a rewind it cannot
    // perform.
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
            let rows = self.cached_len - cache.base;
            ensure!(
                cache.base <= self.cached_len
                    && cache.k.len() == rows * kv_dim
                    && cache.v.len() == rows * kv_dim,
                "layer {i} KV cache length inconsistent with cached_len"
            );
            w.write_all(&(cache.base as u64).to_le_bytes())?;
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
            r.read_exact(&mut u64buf)?;
            let base = u64::from_le_bytes(u64buf) as usize;
            // A snapshot is untrusted input, and `base` reaches straight into
            // the attention indexing: `t - base` underflows for any `base`
            // past the position being read, which in release wraps to an
            // enormous index and panics in the slice. Every constraint the
            // writer satisfies is therefore checked here rather than assumed.
            ensure!(
                base <= cached_len,
                "layer {i} declares base {base} beyond the cached length {cached_len}"
            );
            match self.spec.retention_window(i) {
                // An uncapped layer never evicts, so a non-zero base is either
                // corruption or a snapshot from a differently-configured
                // model. Both are refusals.
                None => ensure!(
                    base == 0,
                    "layer {i} declares base {base} but retains its whole sequence"
                ),
                // A capped layer must still hold a full window behind the tip,
                // or it is restored already unable to answer for its own last
                // position -- and the first thing `Session` does after a
                // restore is replay that position.
                Some(win) => ensure!(
                    base == 0 || base + win <= cached_len,
                    "layer {i} declares base {base} with window {win}, which leaves \
                     fewer rows than position {} needs",
                    cached_len.saturating_sub(1)
                ),
            }
            let rows = cached_len - base;
            let elements = rows
                .checked_mul(self.spec.cache_row(i))
                .context("layer row count overflows")?;
            cache.base = base;
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
    use crate::checkpoint::TEXT_PREFIX;
    use crate::synth::TinySpec;

    /// The synthetic checkpoint for `spec`, in memory.
    ///
    /// Delegates to `synth::tiny_checkpoint`, which the integration suite in
    /// `runtime/tests/long_context.rs` also calls: an inline copy here would
    /// be a second construction path for the same fixture, and the two would
    /// drift.
    fn checkpoint_of(spec: &TinySpec) -> crate::checkpoint::CheckpointFiles {
        crate::synth::tiny_checkpoint(spec).unwrap()
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
        crate::checkpoint::CheckpointFiles::in_memory(
            crate::synth::build_tiny_config_json(spec).unwrap(),
            safetensors::serialize(views, None).unwrap(),
            std::path::PathBuf::from("<memory>"),
        )
        .unwrap()
    }

    /// Published name of a tensor, from its llama.cpp spelling.
    ///
    /// Tests name tensors through the shared table rather than spelling a
    /// published name literally, because a literal is a second copy of the
    /// map -- and when the map moved to `model.language_model.` for the real
    /// multimodal checkpoint, every literal became a test asserting against a
    /// name no file contains.
    fn hf(gguf: &str) -> String {
        crate::checkpoint::hf_tensor_name(gguf).expect("a tensor the table maps")
    }

    /// A checkpoint whose stored values have been edited in place.
    ///
    /// `edit` sees each tensor's Hugging Face name and its stored values, as
    /// the file holds them. Perturbing a weight is how a test shows which
    /// layers actually read it, and doing it through the writer rather than
    /// by patching serialized bytes keeps the test independent of where
    /// safetensors chose to place a payload.
    ///
    /// This must write exactly what `build_tiny_safetensors` writes, or a
    /// perturbed checkpoint differs from the unperturbed one in ways the edit
    /// did not ask for -- which is what happened when the sink stopped
    /// unshifting norms and this did not: every norm came out a full 1.0 low
    /// and the comparison blamed the perturbation.
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
            let mut values: Vec<f32> = t.values.clone();
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
        crate::checkpoint::CheckpointFiles::in_memory(
            crate::synth::build_tiny_config_json(spec).unwrap(),
            safetensors::serialize(views, None).unwrap(),
            std::path::PathBuf::from("<memory>"),
        )
        .unwrap()
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

    /// One tensor's values, named in the llama.cpp spelling the tests use.
    ///
    /// Independent-recomputation tests want the numbers the forward pass
    /// sees, not the geometry, so there is no shape argument. That used to
    /// mean adding 1.0 to norm weights here, mirroring the loader; the loader
    /// no longer shifts, so neither does this, and the stored value is what
    /// the forward pass sees.
    fn tensor_of(ck: &crate::checkpoint::Checkpoint, gguf_name: &str) -> Vec<f32> {
        let hf = crate::checkpoint::hf_tensor_name(gguf_name)
            .unwrap_or_else(|| panic!("no published name for {gguf_name}"));
        // Shape read from the file and handed straight back, so `tensor`'s
        // shape check is a no-op here and the name resolution still runs.
        let shape = ck
            .shape(&hf)
            .unwrap_or_else(|e| panic!("missing {hf}: {e:?}"));
        ck.tensor(&hf, &shape)
            .unwrap_or_else(|e| panic!("read {hf}: {e:?}"))
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

    /// Rewrite a checkpoint with every tensor widened to f32.
    ///
    /// The twin half of the storage-preserving claim. Widening here uses the
    /// same `bf16_to_f32` the loader would have used, so the two images hold
    /// *the same numbers* at two widths -- which is the only comparison that
    /// isolates storage. Regenerating an f32 fixture from the spec instead
    /// would compare truncated values against unrounded ones and fail for a
    /// reason that has nothing to do with this code.
    fn widened_to_f32(
        files: &crate::checkpoint::CheckpointFiles,
    ) -> crate::checkpoint::CheckpointFiles {
        use safetensors::tensor::{Dtype, TensorView};

        let ck = crate::checkpoint::parse(files).unwrap();
        let payloads: Vec<(String, Vec<usize>, Vec<u8>)> = ck
            .names()
            .iter()
            .map(|name| {
                let shape = ck.shape(name).unwrap();
                let bytes = ck
                    .weight(name, &shape)
                    .unwrap()
                    .to_f32()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                (name.clone(), shape, bytes)
            })
            .collect();
        let views: Vec<(&str, TensorView)> = payloads
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        crate::checkpoint::CheckpointFiles::in_memory(
            files.config.clone(),
            safetensors::serialize(views, None).unwrap(),
            std::path::PathBuf::from("<widened>"),
        )
        .unwrap()
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
    fn an_attention_logit_is_the_plain_dot_product_with_no_head_width_divide() {
        // Gemma 4 passes `scaling = 1.0` (upstream:1181), so the fallback
        // `head_dim**-0.5` in upstream:829-835 never runs. The width here is
        // E4B's real global head width, so the constant this test would show
        // under the old code -- 512/sqrt(512), or 22.6 -- is the actual error
        // that was applied to every attention logit on the workstation.
        let q = vec![1.0f32; 512];
        let k = vec![1.0f32; 512];
        let got = attn_logit(&q, &k);
        assert_eq!(
            got,
            512.0,
            "a 1/sqrt(head_dim) divide would report {}",
            512.0 / (512.0f32).sqrt()
        );

        // Asymmetric inputs, so a test passing on all-ones alone cannot hide
        // a divide that happens to cancel.
        assert_eq!(attn_logit(&[3.0, -4.0], &[2.0, 0.5]), 4.0);
    }

    #[test]
    fn the_scaleless_norm_divides_by_the_rms_and_multiplies_by_nothing() {
        // `Gemma4RMSNorm(..., with_scale=False)` (upstream:1201) allocates no
        // weight, so this must agree exactly with the weighted form given
        // unit weights -- and exactness is the property, hence `assert_eq`
        // rather than a tolerance.
        let x = [3.0f32, -4.0, 0.0, 5.0];
        let mut got = x;
        rms_norm_scaleless_in_place(&mut got, 0.0);
        assert_eq!(got.to_vec(), rms_norm(&x, &[1.0; 4], 0.0));

        // The defining property, stated directly: unit RMS out.
        let rms = (got.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-6, "{rms}");
    }

    #[test]
    fn cached_value_heads_are_unit_rms_after_prefill() {
        // The forward-pass guard for the norm above: an unnormalized value
        // row is what shipped, and it reads as a plausible activation from
        // every angle except this one. Random fixture weights make the
        // pre-norm RMS something other than 1 with certainty, so the
        // assertion has content.
        let spec = TinySpec::default();
        let mut model = model_of(&spec);
        model.prefill_all(&[1, 2, 3]).unwrap();

        for (i, layer) in model.weights.layers.iter().enumerate() {
            // Only cache-owning layers write rows; a sharing layer's vector
            // is empty and reads the donor's.
            if layer.spec.kv_source != i {
                continue;
            }
            let hd = layer.spec.head_dim;
            let v = &model.kv[i].v;
            assert!(!v.is_empty(), "layer {i} owns its cache but wrote no rows");
            for (h, head) in v.chunks(hd).enumerate() {
                let rms = (head.iter().map(|x| x * x).sum::<f32>() / hd as f32).sqrt();
                assert!(
                    (rms - 1.0).abs() < 1e-4,
                    "layer {i} head {h}: value RMS {rms}, so v_norm did not run"
                );
            }
        }
    }

    #[test]
    fn the_prelude_tap_reports_the_scaled_embedding_and_the_composed_ple_rows() {
        // The seam layer-diff needs to tell "the stack was handed the wrong
        // input" apart from "layer 0 computed the wrong thing", which the
        // per-layer table cannot do because layer 0 is its first row.
        //
        // Asserted against values recomputed here rather than against a
        // golden: a golden would only pin the tap to whatever it emitted
        // first, and the property that matters is that the embedding row is
        // the table row times sqrt(embed_dim) and nothing else.
        //
        // The E4B-shaped spec rather than the default one, because the
        // default carries no per-layer embeddings and would leave half of
        // this untested.
        let spec = TinySpec::e4b_shaped();
        let mut model = model_of(&spec);
        let c = *model.config();
        let tap = LastRowTap::new(vec![]).capturing_prelude();
        model.set_tap(Box::new(tap.clone()));

        let tokens = [1u32, 2, 3];
        model.prefill_all(&tokens).unwrap();

        let embed = tap.stage(STAGE_EMBED).expect("prelude capture requested");
        assert_eq!(embed.len(), c.embed_dim);
        let last = *tokens.last().unwrap() as usize;
        let scale = (c.embed_dim as f32).sqrt();
        for (j, got) in embed.iter().enumerate() {
            let want = model.weights.embed.get(last * c.embed_dim + j) * scale;
            assert_eq!(*got, want, "embedding element {j}");
        }

        // The per-layer rows are one flat block of n_layers * ple_dim, the
        // shape the decoder stack indexes into -- not a per-layer stack, and
        // the difference is the row stride that made this worth tapping.
        let pd = c.ple_dim.expect("the E4B-shaped fixture carries PLE");
        for name in [STAGE_PLE_IDENTITY, STAGE_PLE_CONTEXT, STAGE_PLE] {
            let row = tap.stage(name).expect("prelude capture requested");
            assert_eq!(row.len(), c.n_layers * pd, "{name}");
        }

        // The composition, checked as the arithmetic it is. This is the stage
        // that diverged against a released E4B while both of its halves are
        // yet to be judged, so the relationship between the three is the
        // thing worth pinning rather than any one of them.
        let (identity, context, composed) = (
            tap.stage(STAGE_PLE_IDENTITY).unwrap(),
            tap.stage(STAGE_PLE_CONTEXT).unwrap(),
            tap.stage(STAGE_PLE).unwrap(),
        );
        for j in 0..composed.len() {
            let want = (context[j] + identity[j]) * PLE_INPUT_SCALE;
            assert!(
                (composed[j] - want).abs() < 1e-6,
                "element {j}: {} is not ({} + {}) / sqrt(2)",
                composed[j],
                context[j],
                identity[j]
            );
        }

        // And the opt-in is real: the default tap keeps no stage at all.
        let plain = LastRowTap::new(vec![]);
        model.set_tap(Box::new(plain.clone()));
        model.prefill_all(&[4]).unwrap();
        for name in PRELUDE_STAGES {
            assert!(plain.stage(name).is_none(), "{name}");
        }
    }

    /// Deterministic non-trivial floats, spread across magnitudes so that a
    /// reordered summation would round differently and the equality assertions
    /// below would have something to catch.
    fn spread(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let m = ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
                let e = ((s >> 20) & 0xf) as i32 - 8;
                (m * 2f64.powi(e)) as f32
            })
            .collect()
    }

    /// Storage width is not a numerics decision.
    ///
    /// The claim the storage-preserving loader rests on, measured the only way
    /// that isolates it: one checkpoint held as bf16, and a second holding the
    /// same values widened to f32, run through the same forward pass. Bit
    /// equality, not a tolerance -- widening is a shift, so anything but exact
    /// agreement means an operand or an order changed.
    #[test]
    fn a_bf16_checkpoint_and_its_widened_twin_produce_identical_logits() {
        let spec = TinySpec {
            storage: crate::synth::TensorStorage::Bf16,
            ..TinySpec::e4b_shaped()
        };
        let narrow_files = checkpoint_of(&spec);
        let wide_files = widened_to_f32(&narrow_files);

        let load = |files: &crate::checkpoint::CheckpointFiles| {
            let ck = crate::checkpoint::parse(files).unwrap();
            CpuModel::from_checkpoint(&ck).unwrap()
        };
        let (mut narrow, mut wide) = (load(&narrow_files), load(&wide_files));

        // Non-vacuity: if both loaded at the same width the test compares a
        // model with itself and would pass through any defect here.
        assert!(
            matches!(narrow.weights.embed, Weight::Bf16(_)),
            "the bf16 fixture did not load as bf16"
        );
        assert!(
            matches!(wide.weights.embed, Weight::F32(_)),
            "the widened twin did not load as f32"
        );
        assert!(
            narrow.weight_bytes().0 * 2 == wide.weight_bytes().0,
            "{} against {}",
            narrow.weight_bytes().0,
            wide.weight_bytes().0
        );

        let prompt: Vec<Token> = vec![7, 3, 42, 11, 5];
        let a = narrow.prefill_all(&prompt).unwrap();
        let b = wide.prefill_all(&prompt).unwrap();
        assert_eq!(a.len(), b.len(), "position counts differ");
        for (position, (x, y)) in a.iter().zip(&b).enumerate() {
            let diff = x.0.iter().zip(&y.0).position(|(p, q)| p != q);
            assert!(
                diff.is_none(),
                "position {position} differs at logit {}: {:?} against {:?}",
                diff.unwrap(),
                x.0[diff.unwrap()],
                y.0[diff.unwrap()]
            );
        }
    }

    /// An f32 checkpoint keeps its low mantissa bits.
    ///
    /// The guard against a future simplification that stores everything as
    /// bf16 to halve memory unconditionally. Storage-preserving means never
    /// narrowing, and a fixture value with bits below the bf16 boundary is
    /// what makes the difference observable.
    #[test]
    fn an_f32_checkpoint_is_not_narrowed() {
        let model = tiny();
        assert!(
            matches!(model.weights.embed, Weight::F32(_)),
            "an f32 checkpoint did not load as f32"
        );
        let truncated = |v: f32| f32::from_bits(v.to_bits() & 0xffff_0000);
        let table = model.weights.embed.to_f32();
        assert!(
            table.iter().any(|v| *v != truncated(*v)),
            "no fixture weight carries bits below the bf16 boundary, so this proves nothing"
        );
    }

    /// `CpuModel::load` is the manual load sequence, plus the drop.
    ///
    /// The drop itself is not observable from a unit test -- it is a peak-RSS
    /// property of a 15 GB checkpoint. What is testable, and what would
    /// actually break if `load` were wrong, is that routing every binary
    /// through it changed nothing: same weights, same fingerprint, same
    /// logits as the three-line sequence it replaced.
    #[test]
    fn the_shared_loader_matches_the_load_sequence_it_replaced() {
        let spec = TinySpec::default();
        let config = crate::synth::build_tiny_config_json(&spec).unwrap();
        let weights = crate::synth::build_tiny_safetensors(&spec).unwrap();
        let files = crate::checkpoint::CheckpointFiles::in_memory(
            config.clone(),
            weights.clone(),
            "<memory>".into(),
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), &config).unwrap();
        std::fs::write(dir.path().join("model.safetensors"), &weights).unwrap();

        let (loaded, fingerprint) = CpuModel::load(dir.path(), false).unwrap();
        assert_eq!(
            fingerprint,
            crate::checkpoint::fingerprint(&files),
            "the loader's fingerprint must be the one a snapshot was written with"
        );

        let mut direct = model_of(&spec);
        let mut loaded = loaded;
        for (position, t) in [1u32, 2, 3].into_iter().enumerate() {
            assert_eq!(
                loaded.forward_one(t, position).unwrap(),
                direct.forward_one(t, position).unwrap(),
                "the shared loader produced different logits for token {t}"
            );
        }
    }

    /// `row_dot` folds the widened values, in index order, for every dtype.
    #[test]
    fn weight_row_dot_matches_the_f32_loop_for_every_dtype() {
        use crate::tensor::{bf16_to_f32, f16_to_f32};

        let (in_dim, out_dim) = (37, 5);
        let n = in_dim * out_dim;
        let x = spread(in_dim, 9);

        // Built per dtype rather than from one shared bit pattern, because the
        // two formats divide the same 16 bits differently: a word that is an
        // ordinary bf16 is an infinity or a NaN read as f16, and `assert_eq!`
        // on a NaN fails whatever the code does.
        let bf16: Vec<u16> = spread(n, 4)
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let f16: Vec<u16> = (0..n)
            .map(|i| {
                let sign = ((i % 2) as u16) << 15;
                // Exponent in 1..=30 -- normal, never the subnormal 0 or the
                // infinity-and-NaN 31.
                let exponent = 1 + (i as u16 % 30);
                sign | (exponent << 10) | ((i as u16).wrapping_mul(2027) & 0x03ff)
            })
            .collect();

        let check = |label: &str, w: &Weight, bits: &[u16], widen: fn(u16) -> f32| {
            for o in 0..out_dim {
                let want: f32 = bits[o * in_dim..(o + 1) * in_dim]
                    .iter()
                    .zip(&x)
                    .map(|(&a, b)| widen(a) * b)
                    .sum();
                let got = w.row_dot(o, &x);
                assert_eq!(got, want, "{label} row {o}: {got} against {want}");
                assert!(got.is_finite(), "{label}: degenerate fixture at row {o}");
            }
        };
        check("bf16", &Weight::Bf16(bf16.clone()), &bf16, bf16_to_f32);
        check("f16", &Weight::F16(f16.clone()), &f16, f16_to_f32);
    }

    #[test]
    fn par_rows_is_bit_identical_at_every_width() {
        // The property the whole threading change rests on. Splitting a dot
        // product across threads reorders a float sum and changes the last
        // bits; splitting across output rows reorders nothing, because each
        // row is still one sequential fold. `assert_eq!` rather than a
        // tolerance: exactness is the assertion, not an approximation of it.
        let out_dim = PARALLEL_MIN_ROWS * 2;
        let in_dim = 37;
        let w = Weight::F32(spread(out_dim * in_dim, 1));
        let x = spread(in_dim, 2);

        let mut serial = vec![0.0f32; out_dim];
        matvec_into(&w, &x, &mut serial, 1, DotKind::Sequential);
        // Not all-zero, or every width would agree vacuously.
        assert!(serial.iter().any(|v| *v != 0.0), "degenerate fixture");

        for threads in 2..=8 {
            let mut got = vec![0.0f32; out_dim];
            matvec_into(&w, &x, &mut got, threads, DotKind::Sequential);
            let diff = serial.iter().zip(&got).position(|(a, b)| a != b);
            assert!(
                diff.is_none(),
                "width {threads} differs at row {}: {:?} against {:?}",
                diff.unwrap(),
                serial[diff.unwrap()],
                got[diff.unwrap()]
            );
        }
    }

    #[test]
    fn par_rows_leaves_no_output_element_unwritten() {
        // `chunks_mut` over `div_ceil` covers the range exactly, but the
        // arithmetic is off-by-one territory and a missed tail would read as
        // a plausible logit rather than as an error. Checked at a width that
        // does not divide the length, which is where a partition drops rows.
        for threads in [1, 3, 5, 7] {
            let mut out = vec![f32::NAN; PARALLEL_MIN_ROWS + 13];
            par_rows(&mut out, threads, |i, dst| *dst = i as f32);
            for (i, v) in out.iter().enumerate() {
                assert_eq!(*v, i as f32, "width {threads} row {i}");
            }
        }
    }

    #[test]
    fn par_rows_below_the_threshold_stays_serial() {
        // Pins the threshold's existence, so removing it is a deliberate act
        // rather than a refactor. Observed through the thread ids `f` runs on,
        // which is the only externally visible difference the split has --
        // the values are identical either way, by construction.
        let seen = std::sync::Mutex::new(std::collections::HashSet::new());
        let mut small = vec![0.0f32; PARALLEL_MIN_ROWS - 1];
        par_rows(&mut small, 8, |_, dst| {
            seen.lock().unwrap().insert(std::thread::current().id());
            *dst = 1.0;
        });
        assert_eq!(
            seen.lock().unwrap().len(),
            1,
            "a {}-row split spawned workers below the {PARALLEL_MIN_ROWS}-row threshold",
            small.len()
        );
    }

    #[test]
    fn the_block_dots_produce_bit_identical_logits_at_every_width() {
        // `Weight::row_dot_block` is pinned in isolation by its own test in
        // `tensor`. This is the other half: that the eleven call sites in
        // `forward_one` reach it through `fill_dots` without disturbing the
        // arithmetic, and that a block boundary landing anywhere inside a
        // worker's run is still bit-identical.
        //
        // The thread widths are the point of the sweep. `par_row_runs` hands
        // each worker a contiguous run, so the width decides where the runs
        // begin and therefore where every block boundary falls. A block fold
        // that quietly depended on its origin would agree at one width and
        // disagree at another.
        //
        // Widened past `PARALLEL_MIN_ROWS` for the same reason the threaded
        // test is: at the committed fixture's geometry every matvec takes the
        // serial path and the sweep would compare five identical runs.
        let spec = TinySpec {
            vocab_size: 1024,
            embed_dim: 24,
            ffn_dim: 640,
            ..TinySpec::e4b_shaped()
        };
        // 640 and 1024 are both multiples of 16, so a partial trailing block
        // needs a width that does not divide the run. 3 and 6 supply it:
        // 640 over 3 workers is runs of 214, and 214 % 16 is 6.
        let prompt: Vec<Token> = vec![7, 300, 42, 261, 900, 3];
        let run = |threads: usize, dot: DotKind| {
            let mut m = model_of(&spec);
            m.set_threads(threads);
            m.set_dot(dot);
            m.prefill_all(&prompt).unwrap()
        };

        let want = run(1, DotKind::Sequential);
        for dot in [DotKind::Rows4, DotKind::Rows8, DotKind::Rows16] {
            assert!(dot.is_bit_identical(), "{} claims to reorder", dot.label());
            for threads in [1usize, 3, 4, 6, 8] {
                let got = run(threads, dot);
                for (pos, (a, b)) in want.iter().zip(&got).enumerate() {
                    let at = a.0.iter().zip(&b.0).position(|(p, q)| p != q);
                    assert!(
                        at.is_none(),
                        "{} at {threads} threads, position {pos}, logit {}: \
                         sequential {:?} against {:?}",
                        dot.label(),
                        at.unwrap(),
                        a.0[at.unwrap()],
                        b.0[at.unwrap()]
                    );
                }
            }
        }
    }

    #[test]
    fn the_lane_dot_changes_the_logits_and_says_so() {
        // The negative of the test above, and it earns its place by failing
        // if someone "fixes" `row_dot_lanes` into a bit-identical alias: the
        // experiment would then be measuring nothing while reporting a
        // difference of zero as a result.
        let spec = TinySpec {
            vocab_size: 1024,
            embed_dim: 24,
            ffn_dim: 640,
            ..TinySpec::e4b_shaped()
        };
        let prompt: Vec<Token> = vec![7, 300, 42, 261, 900, 3];
        let run = |dot: DotKind| {
            let mut m = model_of(&spec);
            m.set_dot(dot);
            m.prefill_all(&prompt).unwrap()
        };
        assert!(!DotKind::Lanes16.is_bit_identical());
        let (want, got) = (run(DotKind::Sequential), run(DotKind::Lanes16));

        let differs = want
            .iter()
            .zip(&got)
            .any(|(a, b)| a.0.iter().zip(&b.0).any(|(p, q)| p != q));
        assert!(
            differs,
            "the lane fold produced identical bits, so either it stopped \
             reordering or the fixture is too small to accumulate a difference"
        );
        // Different, but only by rounding: a difference large enough to move
        // an argmax would mean a bug rather than a reordering.
        for (pos, (a, b)) in want.iter().zip(&got).enumerate() {
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .map(|(i, _)| i)
            };
            assert_eq!(
                argmax(&a.0),
                argmax(&b.0),
                "position {pos}: the lane fold moved the argmax"
            );
        }
    }

    #[test]
    fn a_threaded_forward_pass_produces_bit_identical_logits() {
        // The test that lets the goldens stay pinned on the parallel path.
        // Everything above checks `par_rows` in isolation; this checks that
        // the eleven call sites in `forward_one` -- ten projections and the
        // tied head -- were wired to it without disturbing the arithmetic.
        //
        // Widened past `PARALLEL_MIN_ROWS`, because at the committed fixture's
        // geometry every matvec is under the threshold and both runs would
        // silently take the serial path. That vacuous version of this test is
        // worse than no test: it is the only thing standing between a future
        // threading change and the goldens, and it would report green.
        let spec = TinySpec {
            vocab_size: 1024,
            embed_dim: 24,
            ffn_dim: 640,
            ..TinySpec::e4b_shaped()
        };
        assert!(
            spec.ffn_dim >= PARALLEL_MIN_ROWS && spec.vocab_size >= PARALLEL_MIN_ROWS,
            "fixture below the {PARALLEL_MIN_ROWS}-row threshold: both runs would be serial"
        );

        let prompt: Vec<Token> = vec![7, 300, 42, 261, 900, 3];
        let run = |threads: usize| {
            let mut m = model_of(&spec);
            m.set_threads(threads);
            m.prefill_all(&prompt).unwrap()
        };

        let serial = run(1);
        let parallel = run(4);
        for (pos, (a, b)) in serial.iter().zip(&parallel).enumerate() {
            let diff = a.0.iter().zip(&b.0).position(|(p, q)| p != q);
            assert!(
                diff.is_none(),
                "position {pos} token {}: serial {:?}, 4 threads {:?}",
                diff.unwrap(),
                a.0[diff.unwrap()],
                b.0[diff.unwrap()]
            );
        }
    }

    #[test]
    fn a_fork_decodes_as_wide_as_its_parent() {
        let mut parent = tiny();
        parent.set_threads(3);
        let child = parent.fork_cpu();
        assert_eq!(child.threads, 3, "fork reverted to {}", child.threads);
    }

    #[test]
    fn a_zero_thread_width_is_serial_rather_than_a_panic() {
        let mut m = tiny();
        m.set_threads(0);
        assert_eq!(m.threads, 1);
    }

    #[test]
    #[should_panic(expected = "matvec")]
    fn matvec_rejects_a_weight_matrix_that_does_not_match_the_declared_shape() {
        // A weight longer than in_dim x out_dim stays in bounds and silently
        // uses its leading rows, so a release build had no guard here at all:
        // debug_assert is compiled out exactly where wrong numerics matter.
        // The check costs one integer comparison against in_dim x out_dim
        // multiply-accumulates.
        matvec_into(
            &Weight::F32(vec![1.0; 16]),
            &[1.0; 4],
            &mut [0.0; 2],
            1,
            DotKind::Sequential,
        );
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
            &[&hf("per_layer_token_embd.weight")],
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
        let files = checkpoint_with(&spec, &[], &[], |n| n != hf("blk.1.inp_gate.weight"));
        let err = verdict(&files).unwrap_err().to_string();
        assert!(
            err.contains(&format!(
                "{TEXT_PREFIX}layers.*.per_layer_input_gate.weight (1 of 2 layers)"
            )),
            "{err}"
        );
    }

    #[test]
    fn per_layer_embedding_tensors_without_their_width_are_refused() {
        // The tensors alone do not say how to split the packed row. Inferring
        // the width from the shape would work only if `n_layers` and the
        // per-layer width were the sole factorization of it, which they are
        // not.
        let spec = TinySpec {
            ple_dim: Some(4),
            ..TinySpec::default()
        };
        let mut config: serde_json::Value =
            serde_json::from_slice(&crate::synth::build_tiny_config_json(&spec).unwrap()).unwrap();
        config["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("hidden_size_per_layer_input");
        let files = crate::checkpoint::CheckpointFiles::in_memory(
            serde_json::to_vec(&config).unwrap(),
            crate::synth::build_tiny_safetensors(&spec).unwrap(),
            std::path::PathBuf::from("<memory>"),
        )
        .unwrap();
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
            n != hf("blk.1.attn_v.weight")
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
            // Everything the reader can still see, which is not the same as
            // everything: a windowed donor evicts, and after this package it
            // is only a short prefill that keeps every row. What must hold at
            // any length is that the donor covers `base ..= cached_len`, so
            // the reader's own window bound never names a row that is gone.
            assert_eq!(
                model.kv[src].k.len(),
                (model.cached_len - model.kv[src].base) * model.spec.kv_dim(src),
                "donor {src} must retain every position it claims for reader {i}"
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
        let bumped = checkpoint_bumping(&spec, &hf("blk.0.attn_k.weight"), 0.5);
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
        // `docs/models/gemma-4.md`, not a settled fact.
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
            hf("blk.2.attn_k.weight"),
            hf("blk.2.attn_v.weight"),
            hf("blk.3.attn_k.weight"),
        ] {
            assert_eq!(
                before,
                run(&checkpoint_bumping(&spec, &name, 0.5)),
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
        // at four positions, plus the two u64 header fields and one `base`
        // u64 per layer. A shared layer still costs its `base` -- the reader
        // walks every index, and a header that skipped the tail would put the
        // payload out of step with the layer it belongs to.
        let owned: usize = (0..4)
            .filter(|&i| model.spec.owns_cache(i))
            .map(|i| model.spec.kv_dim(i))
            .sum();
        assert_eq!(snapshot.len(), 16 + 4 * 8 + 2 * owned * tokens.len() * 4);

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
            .map(|i| format!("{TEXT_PREFIX}layers.{i}.self_attn.v_norm.weight"))
            .collect();
        let refs: Vec<&str> = extra.iter().map(String::as_str).collect();
        let err = verdict(&checkpoint_with(&spec, &refs, &[], |_| true))
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 tensor family"), "{err}");
        assert!(
            err.contains(&format!("{TEXT_PREFIX}layers.*.self_attn.v_norm.weight")),
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
        let (embed, table, proj_norm) = (
            t("token_embd.weight"),
            t("per_layer_token_embd.weight"),
            t("per_layer_proj_norm.weight"),
        );
        let post_norm = t("blk.0.post_norm.weight");
        // The three matrices this reimplementation multiplies by, in the form
        // `matvec_into` now takes. Built once: the loop below runs per token.
        let model_proj = Weight::F32(t("per_layer_model_proj.weight"));
        let inp_gate = Weight::F32(t("blk.0.inp_gate.weight"));
        let proj = Weight::F32(t("blk.0.proj.weight"));
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
            matvec_into(&model_proj, &embedded, &mut context, 1, DotKind::Sequential);
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
            matvec_into(&inp_gate, residual, &mut gate, 1, DotKind::Sequential);
            for (g, e) in gate.iter_mut().zip(&rows[0..pd]) {
                *g = gelu(*g) * e;
            }
            let mut widened = vec![0.0f32; c.embed_dim];
            matvec_into(&proj, &gate, &mut widened, 1, DotKind::Sequential);
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
                &hf("blk.1.ffn_up.weight"),
                &[spec.ffn_dim - 8, spec.embed_dim],
            )],
            |_| true,
        );
        let err = verdict(&files).unwrap_err().to_string();
        assert!(err.contains(&hf("blk.1.ffn_up.weight")), "{err}");
        assert!(err.contains("feed-forward width"), "{err}");
        assert!(err.contains("layer 1"), "{err}");
    }

    /// The checkpoint loader shifts nothing: a norm weight loads as stored.
    ///
    /// This test asserted the opposite until the Tier T fixture measured it
    /// (`runtime/tests/ladder_tiny.rs`). Gemma 1 through 3 store a norm
    /// parameter initialized to zero and add one at every use; Gemma 4's
    /// `Gemma4RMSNorm` initializes `weight` to ones and multiplies by plain
    /// `w`, so the `1 + w` carried over from the older models doubled every
    /// normalized activation.
    ///
    /// Kept as a test rather than deleted with the code, because the shift is
    /// the natural thing to reintroduce: it is what Gemma is famous for and
    /// what three of this repository's own files used to do. This is the
    /// statement that Gemma 4 is the exception.
    #[test]
    fn the_checkpoint_loader_does_not_shift_norm_weights() {
        let spec = TinySpec::default();
        let files = crate::checkpoint::CheckpointFiles::in_memory(
            crate::synth::build_tiny_config_json(&spec).unwrap(),
            crate::synth::build_tiny_safetensors(&spec).unwrap(),
            std::path::PathBuf::from("<memory>"),
        )
        .unwrap();
        let ck = crate::checkpoint::parse(&files).unwrap();
        let model = CpuModel::from_checkpoint(&ck).unwrap();

        // Through the public reader rather than the header map: `tensor`
        // widens and shape-checks but applies no arithmetic, which is exactly
        // the claim under test.
        let stored = |name: &str| -> Vec<f32> {
            let shape = ck.shape(name).unwrap();
            ck.tensor(name, &shape).unwrap()
        };

        // A norm arrives exactly as stored.
        let file = stored(&hf("blk.0.attn_norm.weight"));
        assert_eq!(
            model.weights.layers[0].attn_norm, file,
            "attn_norm was transformed on the way in"
        );
        // And the stored values sit near one, so a stray `1 + w` would be a
        // visible doubling rather than a rounding difference. Without this
        // the assertion above would still pass on a fixture whose norms were
        // near zero, where both conventions nearly agree.
        assert!(
            file.iter().all(|w| (0.5..=2.0).contains(w)),
            "the fixture's norms are not near one, so this proves little"
        );

        // A projection is likewise untouched.
        assert_eq!(
            model.weights.layers[0].wq.to_f32(),
            stored(&hf("blk.0.attn_q.weight")),
            "q_proj was transformed"
        );
        assert_eq!(
            model.weights.embed.to_f32(),
            stored(&hf("token_embd.weight")),
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
        // The gate for the whole batched pass. Exact equality, not a
        // tolerance: batching reorders no summation, so the only admissible
        // result is the same bits. Every batch width is checked against the
        // one-position-at-a-time reference, including 2 and 4, which do not
        // divide the token count and therefore exercise a short trailing
        // group -- the case where an off-by-one in the position arithmetic
        // would otherwise hide.
        let tokens = [1u32, 5, 9, 13, 3];
        let mut stepped = tiny();
        let reference: Vec<Logits> = tokens
            .iter()
            .map(|&t| stepped.prefill(&[t]).unwrap())
            .collect();

        for batch in [
            BatchKind::Stepped,
            BatchKind::Batched(1),
            BatchKind::Batched(2),
            BatchKind::Batched(4),
            BatchKind::Batched(5),
            BatchKind::Batched(64),
            BatchKind::Whole,
        ] {
            let mut m = tiny();
            m.set_batch(batch);
            let rows = m.prefill_all(&tokens).unwrap();
            assert_eq!(rows.len(), tokens.len(), "batch {}", batch.label());
            for (i, row) in rows.iter().enumerate() {
                assert_eq!(row, &reference[i], "batch {} position {i}", batch.label());
            }
            assert!(batch.is_bit_identical());

            // `prefill` takes the last-row-only path through the same groups,
            // so it must land on the same final logits as `prefill_all` did.
            let mut last = tiny();
            last.set_batch(batch);
            assert_eq!(
                last.prefill(&tokens).unwrap(),
                reference[tokens.len() - 1],
                "batch {} via prefill",
                batch.label()
            );
        }
    }

    #[test]
    fn prefill_from_returns_exactly_the_tail_of_prefill_all() {
        // Skipping the tied LM head for rows nobody reads is only sound if the
        // rows that remain are the same bits at the same positions. Exact
        // equality, not a tolerance: `first` selects which rows are computed
        // and changes nothing about how any of them is folded.
        //
        // Every `first` against every batch width, because the group mapping
        // is the whole of the new logic and it has three cases -- a group
        // entirely before `first` (wants nothing), one entirely after (wants
        // everything), and the group straddling it (wants a suffix). At
        // production widths the straddling case does not arise: `CHUNK` 256
        // with `warmup` 128 and `DEFAULT_BATCH` 64 puts `first` exactly on a
        // group boundary. It is reachable only from a test, which is the
        // reason this one enumerates rather than samples.
        //
        // Mutation check performed when this was written: replacing the
        // group-relative offset with `first.min(group.len())` -- a plausible
        // way to write the same clamp wrongly -- fails this test and *nothing
        // else in the suite* (324 passed, 1 failed). So this is not a
        // restatement of coverage elsewhere; it is the only coverage there is.
        let tokens = [1u32, 5, 9, 13, 3, 8, 21, 2, 17];
        let mut whole = tiny();
        let reference = whole.prefill_all(&tokens).unwrap();

        for batch in [
            BatchKind::Stepped,
            BatchKind::Batched(1),
            BatchKind::Batched(2),
            BatchKind::Batched(4),
            BatchKind::Batched(9),
            BatchKind::Batched(64),
            BatchKind::Whole,
        ] {
            for first in 0..=tokens.len() {
                let mut m = tiny();
                m.set_batch(batch);
                let rows = m.prefill_from(&tokens, first).unwrap();
                assert_eq!(
                    rows.len(),
                    tokens.len() - first,
                    "batch {} first {first}",
                    batch.label()
                );
                for (i, row) in rows.iter().enumerate() {
                    assert_eq!(
                        row,
                        &reference[first + i],
                        "batch {} first {first} row {i} (position {})",
                        batch.label(),
                        first + i
                    );
                }
                // The cache advanced over every token, not only the ones whose
                // logits were asked for: skipping the head must not skip the
                // forward pass, or the next call attends to a short prefix.
                assert_eq!(
                    m.cached_len(),
                    tokens.len(),
                    "batch {} first {first}",
                    batch.label()
                );
            }
        }
    }

    #[test]
    fn prefill_from_past_the_end_is_an_error_naming_both_numbers() {
        let mut m = tiny();
        let err = m.prefill_from(&[1, 5, 9], 4).unwrap_err().to_string();
        assert!(err.contains('4') && err.contains('3'), "{err}");
    }

    #[test]
    fn a_batch_sees_its_own_earlier_positions() {
        // The intra-batch causality claim, asserted on its own rather than
        // only through the gate above. Every position of a batch has its K/V
        // in the cache before any attention runs at that layer, so the mask is
        // the only thing stopping position 0 from attending to position 1 --
        // and the only thing letting position 1 attend to position 0.
        //
        // Mutation check performed when this was written: ending the span at
        // the batch's last position instead of the reading position's own
        // fails this test, so the bound is load-bearing rather than
        // incidentally correct.
        let mut stepped = tiny();
        stepped.set_batch(BatchKind::Stepped);
        let first = stepped.prefill(&[1]).unwrap();
        let second = stepped.prefill(&[5]).unwrap();

        let mut whole = tiny();
        whole.set_batch(BatchKind::Whole);
        let rows = whole.prefill_all(&[1, 5]).unwrap();
        assert_eq!(rows[0], first, "position 0 must not see position 1");
        assert_eq!(rows[1], second, "position 1 must see position 0");
        assert_ne!(rows[0], rows[1], "the two positions differ at all");
    }

    #[test]
    fn the_batch_kind_round_trips_through_its_label() {
        for kind in [
            BatchKind::Stepped,
            BatchKind::Whole,
            BatchKind::Batched(64),
            BatchKind::Batched(256),
        ] {
            assert_eq!(BatchKind::parse(&kind.label()), Some(kind));
        }
        // A width of one is the stepped path spelled as a number, not a
        // distinct kind that would report itself as "1" forever after.
        assert_eq!(BatchKind::parse("1"), Some(BatchKind::Stepped));
        assert_eq!(BatchKind::parse("0"), None);
        assert_eq!(BatchKind::parse("-4"), None);
        assert_eq!(BatchKind::parse("all"), None);
        assert_eq!(BatchKind::default(), BatchKind::Batched(DEFAULT_BATCH));
    }

    #[test]
    fn recorded_passes_tile_the_positions_with_no_gap_or_overlap() {
        // The property an alignment artifact rests on: every position belongs
        // to exactly one pass, at every batch width. A gap would leave a
        // position with no pass to name; an overlap would let two passes claim
        // it and make "computed alongside" unanswerable.
        let tokens: Vec<u32> = (0..17u32).map(|i| i % 15 + 1).collect();
        for batch in [
            BatchKind::Stepped,
            BatchKind::Batched(1),
            BatchKind::Batched(4),
            BatchKind::Batched(64),
            BatchKind::Whole,
        ] {
            let mut m = tiny();
            m.set_batch(batch);
            m.record_passes();
            m.prefill_all(&tokens).unwrap();

            let passes = m.passes();
            let label = batch.label();
            let mut next = 0;
            for (i, p) in passes.iter().enumerate() {
                assert_eq!(p.index, i, "batch {label}: pass index {p:?}");
                assert_eq!(p.start, next, "batch {label}: gap or overlap at {p:?}");
                assert!(p.len > 0, "batch {label}: empty pass {p:?}");
                next = p.start + p.len;
            }
            assert_eq!(next, tokens.len(), "batch {label}: passes {passes:?}");

            // Every position is covered by exactly one.
            for pos in 0..tokens.len() {
                let n = passes.iter().filter(|p| p.covers(pos)).count();
                assert_eq!(n, 1, "batch {label}: position {pos} in {n} passes");
            }
        }
    }

    #[test]
    fn a_stepped_run_records_one_single_position_pass_per_token() {
        // Decode reaches the backend as a prefill of one token, so this is
        // also the shape a generation loop produces -- and the reason `phase`
        // can be derived from the pass width rather than tracked separately.
        let tokens = [1u32, 5, 9];
        let mut m = tiny();
        m.set_batch(BatchKind::Stepped);
        m.record_passes();
        m.prefill_all(&tokens).unwrap();
        assert_eq!(m.passes().len(), tokens.len(), "{:?}", m.passes());
        for p in m.passes() {
            assert_eq!(p.len, 1, "{p:?}");
            assert_eq!(p.phase(), "decode", "{p:?}");
        }
    }

    #[test]
    fn which_rows_were_wanted_does_not_change_the_recorded_passes() {
        // `RowsWanted` selects which positions get an LM-head row. It must not
        // reach the pass record, or an artifact would describe a different
        // batching for a run that asked for fewer logits.
        let tokens: Vec<u32> = (0..10u32).map(|i| i + 1).collect();
        let passes_of = |f: &dyn Fn(&mut CpuModel)| {
            let mut m = tiny();
            m.set_batch(BatchKind::Batched(4));
            m.record_passes();
            f(&mut m);
            m.passes().to_vec()
        };
        let all = passes_of(&|m| {
            m.prefill_all(&tokens).unwrap();
        });
        let last = passes_of(&|m| {
            m.prefill(&tokens).unwrap();
        });
        let from = passes_of(&|m| {
            m.prefill_from(&tokens, 7).unwrap();
        });
        assert_eq!(all, last, "last-row-only changed the passes");
        assert_eq!(all, from, "a row offset changed the passes");
    }

    #[test]
    fn passes_are_not_recorded_until_asked_for_and_a_fork_starts_its_own() {
        let mut m = tiny();
        m.prefill_all(&[1u32, 2, 3]).unwrap();
        assert!(m.passes().is_empty(), "recorded without being asked");

        m.record_passes();
        m.prefill_all(&[4u32]).unwrap();
        assert_eq!(m.passes().len(), 1, "{:?}", m.passes());

        // A fork is a separate branch; its pass ordinals are its own.
        let fork = m.fork_cpu();
        assert!(fork.passes().is_empty(), "{:?}", fork.passes());
    }

    #[test]
    fn the_default_dot_kind_is_the_fast_bit_identical_one() {
        // A derive nobody reads was how `Sequential` stayed the default for
        // every caller but the ladder long after `Rows4` measured 1.87x
        // faster at the same bits. Stating it here makes the next change to
        // it deliberate.
        assert_eq!(DotKind::default(), DotKind::Widen4);
        // The half of the choice that is not about speed: a default that
        // moved the numerics would silently invalidate every golden in the
        // tree, so the default must be on the bit-identical side.
        //
        // Mutation check performed when this was written: moving `#[default]`
        // to `Lanes16` fails this test and *nothing else in the suite* --
        // every other test either sets the kind explicitly or runs a fixture
        // small enough that the reordered fold lands on the same bits. So
        // this assertion is not a restatement of coverage elsewhere; it is
        // the only coverage there is.
        assert!(DotKind::default().is_bit_identical());
    }

    #[test]
    fn the_batched_panel_matches_the_per_position_matvec() {
        // `matmul_panel` against `matvec_into` directly, one layer below the
        // forward pass, so a kernel regression is localized here rather than
        // showing up only as a moved logit.
        let (in_dim, out_dim, n) = (37usize, 21usize, 5usize);
        let w = Weight::F32(
            (0..in_dim * out_dim)
                .map(|i| ((i % 19) as f32 - 9.0) / 7.0)
                .collect(),
        );
        let x: Vec<f32> = (0..n * in_dim)
            .map(|i| ((i % 13) as f32 - 6.0) / 5.0)
            .collect();

        let mut expected = vec![0.0f32; n * out_dim];
        for p in 0..n {
            matvec_into(
                &w,
                &x[p * in_dim..(p + 1) * in_dim],
                &mut expected[p * out_dim..(p + 1) * out_dim],
                1,
                DotKind::Sequential,
            );
        }

        for dot in DotKind::ALL {
            if !dot.is_bit_identical() {
                continue;
            }
            for threads in [1usize, 3, 8] {
                let mut got = vec![0.0f32; n * out_dim];
                let mut fm = vec![0.0f32; n * out_dim];
                matmul_panel(&w, &x, &mut got, &mut fm, n, threads, dot);
                assert_eq!(got, expected, "dot {} threads {threads}", dot.label());
            }
        }
    }

    #[test]
    fn widen4_matches_rows4_on_both_sides_of_its_threshold() {
        // The panel sweep above runs at n = 5, below `WIDEN_MIN_POSITIONS`, so
        // it exercises `Widen4`'s fallback and never its kernel. This walks
        // the boundary itself: a threshold covered only incidentally is one
        // that can move without any test noticing.
        //
        // bf16 rather than f32 because widening an f32 row is a copy, and a
        // copy cannot expose a conversion hoisted to the wrong place.
        let (in_dim, out_dim) = (37usize, 21usize);
        let stored: Vec<u16> = (0..in_dim * out_dim)
            .map(|i| ((((i % 19) as f32 - 9.0) / 7.0).to_bits() >> 16) as u16)
            .collect();
        let w = Weight::Bf16(stored);

        // 4 and 21 are also the block width and the feature count: out_dim is
        // not a multiple of 4, so every case runs the remainder path too.
        for n in [WIDEN_MIN_POSITIONS - 1, WIDEN_MIN_POSITIONS, 64] {
            let x: Vec<f32> = (0..n * in_dim)
                .map(|i| ((i % 13) as f32 - 6.0) / 5.0)
                .collect();
            for threads in [1usize, 3, 8] {
                let (mut a, mut b) = (vec![0.0f32; n * out_dim], vec![0.0f32; n * out_dim]);
                let mut fm = vec![0.0f32; n * out_dim];
                matmul_panel(&w, &x, &mut a, &mut fm, n, threads, DotKind::Rows4);
                matmul_panel(&w, &x, &mut b, &mut fm, n, threads, DotKind::Widen4);
                assert_eq!(a, b, "n {n} threads {threads}");
            }
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
        assert_eq!(
            after.len(),
            16 + m.kv.len() * 8,
            "a failed restore left rows in the cache"
        );
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
        // error. Cut 132 of this 416-byte snapshot is one such point: it
        // leaves layer 0's K fully restored and gave a first logit of
        // -0.11380069 where a fresh model gives 0.32625324. It was cut 124 of
        // 400 until each layer gained an eight-byte `base`, which moved every
        // payload offset without changing what the cut demonstrates.
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
        assert_eq!(full.len(), 416, "snapshot geometry assumed by this test");

        let mut fresh = model();
        let mut empty_cache = Vec::new();
        fresh.save_state(&mut empty_cache).unwrap();
        let expected = fresh.prefill(&[4]).unwrap();

        // The named point first: layer 0's K fully restored, the payload cut
        // inside layer 0's V. Compared as a scalar so the failure names the
        // wrong number instead of printing 288 floats twice; the whole row
        // is compared in the scan below.
        let mut wedged = model();
        wedged.load_state(&mut &full[..132]).unwrap_err();
        assert_eq!(wedged.cached_len(), 0, "cut 132 reports fresh");
        assert_eq!(
            wedged.prefill(&[4]).unwrap().0[0],
            expected.0[0],
            "cut 132: the first token attended to the orphaned rows"
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
        let over = m.config().context_length + 1;
        let long: Vec<Token> = (0..over).map(|i| (i % 30) as Token).collect();
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
        fn on_residual(&mut self, layer: usize, site: Site, position: usize, x: &[f32]) {
            if site != Site::LayerOutput {
                return;
            }
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
        let order: Vec<(usize, usize)> = calls.iter().map(|(l, p, _)| (*l, *p)).collect();

        // The contract, which holds under any traversal: every (layer,
        // position) pair fires exactly once, and positions ascend within a
        // layer so a tap keeping the last row keeps the last position.
        let mut seen = order.clone();
        seen.sort_unstable();
        let expected: Vec<(usize, usize)> =
            (0..2).flat_map(|l| (0..3).map(move |p| (l, p))).collect();
        assert_eq!(
            seen, expected,
            "every layer at every position, exactly once"
        );
        for layer in 0..2 {
            let positions: Vec<usize> = order
                .iter()
                .filter(|(l, _)| *l == layer)
                .map(|(_, p)| *p)
                .collect();
            assert!(
                positions.windows(2).all(|w| w[0] < w[1]),
                "layer {layer} positions must ascend, got {positions:?}"
            );
        }

        // The traversal itself, pinned separately: a batched prefill is
        // layer-major, every position through layer 0 before any reaches
        // layer 1. That order is what lets a layer's weights be read once for
        // the whole batch, so it is a property of the optimization rather than
        // an accident of the loop nesting.
        assert_eq!(order, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]);
        assert!(calls.iter().all(|(_, _, x)| x.len() == 16));
        assert!(calls
            .iter()
            .all(|(_, _, x)| x.iter().all(|v| v.is_finite())));
    }

    /// `TapCalls` with the site kept, for the tap that filters nothing.
    type SitedCalls = std::sync::Arc<std::sync::Mutex<Vec<(usize, Site, usize, Vec<f32>)>>>;

    /// Records every site it is handed, so what a filtering tap drops is
    /// visible rather than inferred.
    #[derive(Clone, Default)]
    struct EverySiteTap(SitedCalls);

    impl Tap for EverySiteTap {
        fn on_residual(&mut self, layer: usize, site: Site, position: usize, x: &[f32]) {
            self.0
                .lock()
                .unwrap()
                .push((layer, site, position, x.to_vec()));
        }
    }

    #[test]
    fn filtering_on_the_site_observes_what_accepting_every_site_does() {
        // The widening is meant to be inert: a tap that takes only
        // LayerOutput must see exactly the rows it saw when the trait had no
        // site at all, bit-for-bit. Demonstrated rather than asserted in
        // prose, by running the filtering tap against one that filters
        // nothing -- so the day a second fire site lands, this test fails and
        // names the taps that have to decide about it.
        let mut m = tiny();
        let filtering = CollectingTap::default();
        m.set_tap(Box::new(filtering.clone()));
        m.prefill_all(&[1, 5, 9]).unwrap();

        let mut m = tiny();
        let everything = EverySiteTap::default();
        m.set_tap(Box::new(everything.clone()));
        m.prefill_all(&[1, 5, 9]).unwrap();

        let all = everything.0.lock().unwrap();
        assert!(
            all.iter().all(|(_, s, _, _)| *s == Site::LayerOutput),
            "the seam fires a site the filtering taps drop: {:?}",
            all.iter()
                .map(|(_, s, _, _)| *s)
                .filter(|s| *s != Site::LayerOutput)
                .collect::<Vec<_>>()
        );
        let dropped: Vec<(usize, usize, Vec<f32>)> =
            all.iter().map(|(l, _, p, x)| (*l, *p, x.clone())).collect();
        assert_eq!(
            *filtering.0.lock().unwrap(),
            dropped,
            "filtering on the site changed what a tap observes"
        );
    }

    #[test]
    fn a_site_round_trips_through_its_stored_name() {
        // The artifact spelling. A capture manifest stores this text and a
        // reader parses it back, so the two directions are one decision and a
        // rename that touched only Display would be silent until a reader
        // hit an old file.
        assert_eq!(Site::LayerOutput.to_string(), "layer_output");
        assert_eq!(
            "layer_output".parse::<Site>().unwrap(),
            Site::LayerOutput,
            "the name Display writes must parse back"
        );
        assert!(
            "post_attention".parse::<Site>().is_err(),
            "an unknown site must be an error, not a default"
        );
    }

    #[test]
    fn a_stepped_prefill_still_fires_the_tap_position_major() {
        // The other half of the traversal contract: under BatchKind::Stepped
        // the firing order is the one the pass had before batching, so a
        // caller that needs position-major has a way to ask for it and the two
        // orders are pinned against each other rather than one replacing the
        // other silently.
        let mut m = tiny();
        m.set_batch(BatchKind::Stepped);
        let tap = CollectingTap::default();
        m.set_tap(Box::new(tap.clone()));
        m.prefill_all(&[1, 5, 9]).unwrap();
        let order: Vec<(usize, usize)> = tap
            .0
            .lock()
            .unwrap()
            .iter()
            .map(|(l, p, _)| (*l, *p))
            .collect();
        assert_eq!(order, [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn last_row_tap_keeps_the_highest_position_whatever_the_call_order() {
        // Layer-major batched prefill finishes a layer on whichever position
        // the batch ends on, so "last written" and "last position" part ways.
        // Feed the rows out of order and require the position to decide.
        let mut tap = LastRowTap::new(vec![0]).capturing_prelude();
        tap.on_residual(0, Site::LayerOutput, 1, &[1.0]);
        tap.on_residual(0, Site::LayerOutput, 2, &[2.0]);
        tap.on_residual(0, Site::LayerOutput, 0, &[0.0]);
        assert_eq!(tap.row(0), Some(vec![2.0]));

        tap.on_stage("embedding", 2, &[2.0]);
        tap.on_stage("embedding", 0, &[0.0]);
        assert_eq!(tap.stage("embedding"), Some(vec![2.0]));

        // A clear restarts the comparison, so the next example's position 0
        // is not shadowed by the previous example's position 2.
        tap.clear();
        tap.on_residual(0, Site::LayerOutput, 0, &[9.0]);
        assert_eq!(tap.row(0), Some(vec![9.0]));

        // Unrequested layers stay absent.
        tap.on_residual(1, Site::LayerOutput, 5, &[3.0]);
        assert_eq!(tap.row(1), None);
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

    /// One `on_attention` call: layer, position, query head, first key
    /// position, and the distribution itself.
    type AttnCall = (usize, usize, usize, usize, Vec<f32>);

    /// Shared collector, on `CollectingTap`'s reasoning: the model owns the tap
    /// once `set_tap` takes it, and `Tap: Send` rules out `Rc`.
    type AttnCalls = std::sync::Arc<std::sync::Mutex<Vec<AttnCall>>>;

    /// A tap that opts into attention weights and records every fire.
    ///
    /// Separate from `CollectingTap`, which must keep answering `false` to
    /// `wants_attention` -- that default is what
    /// `a_tap_that_does_not_want_attention_is_never_offered_it` asserts, and a
    /// single helper opting in would make that test vacuous.
    #[derive(Clone, Default)]
    struct AttnTap(AttnCalls);

    impl Tap for AttnTap {
        fn on_residual(&mut self, _layer: usize, _site: Site, _position: usize, _x: &[f32]) {}

        fn wants_attention(&self) -> bool {
            true
        }

        fn on_attention(
            &mut self,
            layer: usize,
            position: usize,
            head: usize,
            first: usize,
            w: &[f32],
        ) {
            self.0
                .lock()
                .unwrap()
                .push((layer, position, head, first, w.to_vec()));
        }
    }

    /// Every attention fire from one prefill of `tokens` over the mixed
    /// sliding/global fixture, with the model returned so a test can read the
    /// declared geometry back out of it rather than restate the schedule.
    fn attention_calls(tokens: &[Token]) -> (CpuModel, Vec<AttnCall>) {
        let mut m = model_of(&TinySpec::e4b_shaped());
        let tap = AttnTap::default();
        m.set_tap(Box::new(tap.clone()));
        m.prefill_all(tokens).unwrap();
        let calls = tap.0.lock().unwrap().clone();
        (m, calls)
    }

    #[test]
    fn attention_weights_are_a_distribution_over_the_visible_span() {
        // Covers the accumulated error of a softmax over a span this short.
        const TOLERANCE: f32 = 1e-5;
        let (_m, calls) = attention_calls(&[1, 5, 9, 2, 7]);
        assert!(!calls.is_empty(), "the attention seam never fired");
        for (l, p, h, first, w) in &calls {
            assert!(
                !w.is_empty(),
                "layer {l} position {p} head {h}: empty distribution"
            );
            let sum: f32 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < TOLERANCE,
                "layer {l} position {p} head {h} first {first}: weights sum to {sum}, not 1"
            );
            for (i, x) in w.iter().enumerate() {
                assert!(
                    x.is_finite() && *x >= 0.0,
                    "layer {l} position {p} head {h}: weight {i} is {x}"
                );
            }
        }
    }

    #[test]
    fn the_attention_span_matches_the_layer_kind() {
        // Five positions against `e4b_shaped`'s window of 3, so the sliding
        // layers reach their cap and the two regimes disagree about the span
        // rather than both seeing everything.
        let (m, calls) = attention_calls(&[1, 5, 9, 2, 7]);
        assert!(!calls.is_empty(), "the attention seam never fired");

        // Read the schedule off the loaded model. Recomputing it here from
        // `sliding_window_pattern` would assert the fixture against itself:
        // `docs/models/gemma-4.md` is explicit that the schedule is an array to
        // be read, not a stride to be derived.
        let mut saw_global = false;
        let mut saw_sliding = false;
        for (l, p, h, first, w) in &calls {
            let (want_first, kind) = match m.spec.layers[*l].kind {
                LayerKind::Global => {
                    saw_global = true;
                    (0, "global")
                }
                LayerKind::SlidingWindow(win) => {
                    saw_sliding = true;
                    ((p + 1).saturating_sub(win), "sliding")
                }
            };
            assert_eq!(
                *first, want_first,
                "{kind} layer {l} position {p} head {h}: first key {first}, expected {want_first}"
            );
            assert_eq!(
                w.len(),
                p - want_first + 1,
                "{kind} layer {l} position {p} head {h}: {} weights over span {want_first}..={p}",
                w.len()
            );
        }
        assert!(
            saw_global && saw_sliding,
            "the fixture must exercise both regimes, saw global {saw_global} sliding {saw_sliding}"
        );

        // The window actually bit somewhere, so the assertions above are not
        // all trivially the global case.
        let clipped = calls.iter().any(|(_, _, _, first, _)| *first > 0);
        assert!(clipped, "no sliding layer dropped a position at window 3");
    }

    #[test]
    fn installing_the_attention_tap_does_not_perturb_logits() {
        let spec = TinySpec::e4b_shaped();
        let mut tapped = model_of(&spec);
        tapped.set_tap(Box::new(AttnTap::default()));
        let mut plain = model_of(&spec);
        assert_eq!(
            tapped.prefill_all(&[1, 4, 7, 2, 9]).unwrap(),
            plain.prefill_all(&[1, 4, 7, 2, 9]).unwrap()
        );
    }

    #[test]
    fn a_tap_that_does_not_want_attention_is_never_offered_it() {
        // `CollectingTap` takes the trait's default `wants_attention`, so this
        // pins the gate rather than the tap: a site whose cost is quadratic in
        // the context must not be delivered to a tap that never asked.
        let mut m = model_of(&TinySpec::e4b_shaped());
        let tap = CollectingTap::default();
        m.set_tap(Box::new(tap.clone()));
        m.prefill_all(&[1, 5, 9, 2]).unwrap();
        assert!(
            !CollectingTap::default().wants_attention(),
            "the default must stay false or this test asserts nothing"
        );
        assert!(
            !tap.0.lock().unwrap().is_empty(),
            "the residual seam must still fire, or the model never ran"
        );
    }

    #[test]
    fn every_query_head_fires_once_per_position() {
        let tokens = [1, 5, 9, 2];
        let (m, calls) = attention_calls(&tokens);
        let c = *m.config();
        let mut seen: Vec<(usize, usize, usize)> =
            calls.iter().map(|(l, p, h, _, _)| (*l, *p, *h)).collect();
        seen.sort_unstable();
        let expected: Vec<(usize, usize, usize)> = (0..c.n_layers)
            .flat_map(|l| {
                (0..tokens.len()).flat_map(move |p| (0..c.n_heads).map(move |h| (l, p, h)))
            })
            .collect();
        assert_eq!(
            seen, expected,
            "every (layer, position, query head) exactly once"
        );
    }

    #[test]
    fn a_stepped_prefill_reports_the_same_attention_as_one_batch() {
        // The `crate::conformance` obligation reaching the new site: how a
        // request was internally divided may not be observable. A per-position
        // fire site is where that could leak, because the batch member index
        // and the absolute position are different numbers and only the second
        // is meaningful.
        let tokens = [1, 5, 9, 2, 7];
        let (_m, batched) = attention_calls(&tokens);

        let mut m = model_of(&TinySpec::e4b_shaped());
        let tap = AttnTap::default();
        m.set_tap(Box::new(tap.clone()));
        for t in tokens {
            m.prefill_all(&[t]).unwrap();
        }
        let mut stepped = tap.0.lock().unwrap().clone();

        // Sorting is the comparison, not a weakening of it: the traversal order
        // differs by construction -- batched prefill is layer-major over the
        // whole span, a stepped one is position-major -- while the *set* of
        // (layer, position, head, first, weights) must be identical.
        let key = |c: &AttnCall| (c.0, c.1, c.2);
        let mut batched = batched;
        batched.sort_by_key(key);
        stepped.sort_by_key(key);
        assert_eq!(
            batched.len(),
            stepped.len(),
            "{} batched fires against {} stepped",
            batched.len(),
            stepped.len()
        );
        for (b, s) in batched.iter().zip(&stepped) {
            assert_eq!(
                b,
                s,
                "batched {:?} against stepped {:?}",
                (b.0, b.1, b.2, b.3),
                (s.0, s.1, s.2, s.3)
            );
        }
    }

    /// Control-vector safetensors image with `block.<index>` (0-based)
    /// tensors, through the same writer the probe fixtures use.
    fn control_image(directions: &[(usize, Vec<f32>)]) -> Image {
        let named: Vec<(String, &[f32])> = directions
            .iter()
            .map(|(block, v)| (format!("{BLOCK_PREFIX}{block}"), v.as_slice()))
            .collect();
        let borrowed: Vec<(&str, &[f32])> = named.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        let bytes = crate::synth::build_f32_safetensors(&borrowed).unwrap();
        Image::parse(bytes, "control vector").unwrap()
    }

    fn control(directions: &[(usize, Vec<f32>)], scale: f32) -> ControlVector {
        ControlVector::from_image(&control_image(directions), scale, None).unwrap()
    }

    /// The message from a control-vector image the reader must refuse.
    ///
    /// `unwrap_err` would need `Debug` on `ControlVector`, which holds the
    /// direction vectors and has no reason to be printable.
    fn control_error(image: &Image, range: Option<(usize, usize)>) -> String {
        match ControlVector::from_image(image, 1.0, range) {
            Ok(cv) => panic!("image accepted, width {}", cv.width()),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn zero_control_vector_is_bit_identical_to_unsteered() {
        // The tiny model has blocks 0 and 1. Both are steerable under the
        // 0-based numbering; block 1 is used here so the assertion is about
        // the zero direction and not about which block it landed on.
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
    fn block_n_steers_block_n_and_leaves_earlier_blocks_untouched() {
        // The exporter names a tensor for the block whose output it was
        // fitted on, and this reader must place it there. An off-by-one
        // steers a neighbouring block and returns a plausible answer with no
        // error anywhere; observing the residual through the tap is the only
        // way to tell the two mappings apart.
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
        // Block beyond the model. block.2 names block 2, which a two-block
        // model does not have.
        let err = m
            .set_control_vector(control(&[(2, vec![0.0; 16])], 1.0))
            .unwrap_err();
        assert!(err.to_string().contains("blocks"), "{err}");
        // Disagreeing widths are refused at load, before the model sees them.
        let image = control_image(&[(0, vec![0.0; 16]), (1, vec![0.0; 8])]);
        let err = control_error(&image, None);
        assert!(err.contains("width"), "{err}");
        // An image naming no blocks at all is a load error.
        let err = control_error(&control_image(&[]), None);
        assert!(err.contains("no block."), "{err}");
    }

    #[test]
    fn block_zero_is_steerable() {
        // The retired `direction.` naming refused index 0, so block 0
        // could not be steered even though the backend can steer it. This is
        // the assertion that the floor is gone rather than merely unused.
        let cv =
            ControlVector::from_image(&control_image(&[(0, vec![1.0; 16])]), 1.0, None).unwrap();
        assert_eq!(cv.direction(0).map(<[f32]>::len), Some(16));
    }

    #[test]
    fn an_image_carrying_the_retired_direction_prefix_is_refused() {
        // Refusing the old prefix is what keeps the format's generation
        // legible. A pre-R10 image names the same blocks this reader would
        // place correctly, so accepting it would work -- until the base or
        // the layout moves and a stale file steers the wrong block with no
        // error anywhere. The reader is the only place that can see it.
        let bytes =
            crate::synth::build_f32_safetensors(&[("direction.1", &[0.0f32; 16][..])]).unwrap();
        let image = Image::parse(bytes, "control vector").unwrap();
        let err = control_error(&image, None);
        assert!(err.contains("no block."), "{err}");
    }

    #[test]
    fn a_layer_range_drops_the_blocks_outside_it() {
        let image = control_image(&[(0, vec![1.0; 16]), (1, vec![2.0; 16])]);
        let cv = ControlVector::from_image(&image, 1.0, Some((1, 1))).unwrap();
        assert!(cv.direction(0).is_none(), "block 0 survived the range");
        assert_eq!(cv.direction(1), Some(&[2.0f32; 16][..]));
        // A range that admits nothing is an error rather than an unsteered
        // model: the caller asked for steering and got none, silently.
        let err = control_error(&image, Some((5, 6)));
        assert!(err.contains("within layer range"), "{err}");
        // An inverted range is refused for what it is, not for being empty.
        let err = control_error(&image, Some((2, 1)));
        assert!(err.contains("is empty"), "{err}");
    }

    #[test]
    fn a_non_numeric_block_suffix_is_refused_naming_the_tensor() {
        let bytes =
            crate::synth::build_f32_safetensors(&[("block.last", &[0.0f32; 16][..])]).unwrap();
        let image = Image::parse(bytes, "control vector").unwrap();
        let err = control_error(&image, None);
        assert!(err.contains("block.last"), "{err}");
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

    /// Verdict over a checkpoint read with `text_only` set either way, so the
    /// tower tests below differ in exactly the flag they are about.
    fn verdict_text_only(
        files: &crate::checkpoint::CheckpointFiles,
        text_only: bool,
    ) -> Result<()> {
        let mut ck = crate::checkpoint::parse(files)?;
        ck.text_only = text_only;
        let spec = ModelSpec::from_checkpoint(&ck.config)?;
        check_supported_checkpoint(&ck, &spec)
    }

    /// The default is refusal, and it names the towers and the flag.
    ///
    /// A real E4B carries an audio and a vision tower -- 1409 of its 2130
    /// tensors -- and the generic unconsumed-tensor message would report that
    /// as though the backend had met something it could not parse. It has not:
    /// it has met a multimodal release, and declining the other towers is a
    /// decision the caller makes rather than one this code makes silently.
    #[test]
    fn a_multimodal_checkpoint_is_refused_naming_the_towers_and_the_flag() {
        let files = checkpoint_with(
            &TinySpec::default(),
            &[
                "model.audio_tower.layers.0.norm_out.weight",
                "model.vision_tower.layers.0.norm_out.weight",
                "model.embed_audio.embedding_projection.weight",
            ],
            &[],
            |_| true,
        );
        let err = verdict_text_only(&files, false).unwrap_err().to_string();
        assert!(err.contains("multimodal"), "{err}");
        assert!(err.contains("--text-only"), "{err}");
        // The towers by prefix, and how much was found in them -- not every
        // family, which on a real release is 114 lines of noise around the one
        // sentence the reader needs.
        assert!(err.contains("model.audio_tower."), "{err}");
        assert!(err.contains("model.vision_tower."), "{err}");
        assert!(err.contains("model.embed_audio."), "{err}");
        assert!(err.contains("3 tensor families"), "{err}");
        // The one tower the file does not carry stays out of the message.
        assert!(!err.contains("model.embed_vision."), "{err}");
        // Not reported as unparseable structure, which is the other message.
        assert!(!err.contains("does not consume"), "{err}");

        // The same file loads once the caller has said so.
        verdict_text_only(&files, true).unwrap();
    }

    /// `--text-only` declines the four named towers and nothing else.
    ///
    /// The escape hatch is a list, not "ignore what you do not recognize": a
    /// fifth tower in a future release is structure this backend has never
    /// seen, and it must still be a refusal.
    #[test]
    fn an_unknown_tensor_family_is_refused_even_with_text_only() {
        let files = checkpoint_with(
            &TinySpec::default(),
            &["model.depth_tower.layers.0.norm_out.weight"],
            &[],
            |_| true,
        );
        let err = verdict_text_only(&files, true).unwrap_err().to_string();
        assert!(err.contains("does not consume"), "{err}");
        assert!(
            err.contains("model.depth_tower.layers.*.norm_out.weight"),
            "{err}"
        );
        assert!(!err.contains("--text-only"), "{err}");
    }

    /// A text-only checkpoint is unaffected by the flag in either position,
    /// which is what keeps the refusal from becoming a tax on every load.
    #[test]
    fn a_text_only_checkpoint_loads_with_the_flag_either_way() {
        let files = checkpoint_of(&TinySpec::e4b_shaped());
        verdict_text_only(&files, false).unwrap();
        verdict_text_only(&files, true).unwrap();
        assert!(non_text_families(&crate::checkpoint::parse(&files).unwrap()).is_empty());
    }
}
