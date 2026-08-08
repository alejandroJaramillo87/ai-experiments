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
 * That is still a subset of the published Gemma 4 architecture, which adds
 * per-layer input embeddings, a shared KV cache, a per-layer output scale,
 * final logit softcapping, and value normalization. A checkpoint carrying
 * any of those is refused by `check_supported` rather than loaded with the
 * extra structure silently dropped: an ignored weight tensor still produces
 * fluent text, so a silent load would be an undetectable wrong answer.
 * See docs/rust-harness/gemma-4-ingestion.md.
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
 * The K/V cache stores the full sequence for every layer; the sliding
 * window is applied as an attention mask only. Capping sliding-window
 * storage to the kv.rs budget (a ring buffer) is deferred to the
 * real-model work, where the memory matters.
 */

use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};

use super::{Logits, Model, Token};
use crate::gguf::{GgufFile, MetaValue};
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
    pub sliding_window_pattern: usize,
    pub rms_eps: f32,
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
}

impl Config {
    pub fn from_gguf(file: &GgufFile) -> Result<Config> {
        let arch = match file.metadata.get("general.architecture") {
            Some(MetaValue::Str(a)) => a.as_str(),
            _ => bail!("missing general.architecture"),
        };
        ensure!(
            arch.starts_with("gemma"),
            "unsupported architecture {arch}, expected a gemma variant"
        );
        let u = |key: &str| -> Result<usize> {
            match file.metadata.get(&format!("{arch}.{key}")) {
                Some(MetaValue::U32(v)) => Ok(*v as usize),
                Some(other) => bail!("{arch}.{key}: expected u32, got {other:?}"),
                None => bail!("missing {arch}.{key}"),
            }
        };
        // A bool array marks each layer sliding-window (true) or global
        // (false); `layer_kind` makes every `stride`-th layer global. The two
        // agree exactly when the global layers are periodic, which is the
        // only form Gemma 4 ships -- E2B stride 5, E4B 6, the mtp head 4.
        let schedule = |key: &str| -> Result<usize> {
            let Some(MetaValue::ArrBool(pattern)) = file.metadata.get(&format!("{arch}.{key}"))
            else {
                return u(key);
            };
            let Some(first_global) = pattern.iter().position(|sliding| !sliding) else {
                bail!(
                    "{arch}.{key} marks no layer global, so it encodes no periodic \
                     stride; an all-sliding-window schedule is not expressible here"
                );
            };
            let stride = first_global + 1;
            for (i, sliding) in pattern.iter().enumerate() {
                if *sliding == (i + 1).is_multiple_of(stride) {
                    bail!(
                        "{arch}.{key} is not periodic: it implies stride {stride} from \
                         layer {first_global}, but layer {i} contradicts it"
                    );
                }
            }
            Ok(stride)
        };
        let f = |key: &str| -> Result<f32> {
            match file.metadata.get(&format!("{arch}.{key}")) {
                Some(MetaValue::F32(v)) => Ok(*v),
                Some(other) => bail!("{arch}.{key}: expected f32, got {other:?}"),
                None => bail!("missing {arch}.{key}"),
            }
        };
        let vocab_size = match file.metadata.get("tokenizer.ggml.tokens") {
            Some(MetaValue::ArrStr(v)) => v.len(),
            _ => bail!("missing tokenizer.ggml.tokens"),
        };
        let config = Config {
            vocab_size,
            embed_dim: u("embedding_length")?,
            n_layers: u("block_count")?,
            n_heads: u("attention.head_count")?,
            n_kv_heads: u("attention.head_count_kv")?,
            context_length: u("context_length")?,
            sliding_window: u("attention.sliding_window")?,
            sliding_window_pattern: schedule("attention.sliding_window_pattern")?,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
        };
        // Each dimension below divides or strides something in the forward
        // pass, so a zero reaches the arithmetic instead of a guard: 0 is
        // even, so key_length = 0 cleared the RoPE check and then aborted
        // the process at `chunks_mut(0)` on the first token. Positivity is
        // checked first so the error names the key.
        for (key, value) in [
            ("embedding_length", config.embed_dim),
            ("block_count", config.n_layers),
            ("attention.head_count", config.n_heads),
        ] {
            ensure!(value > 0, "{arch}.{key} is 0, must be positive");
        }
        // The window keys are dimensions too, but a zero in either has a
        // specific wrong reading here. Pattern 0 makes every layer
        // sliding-window in `layer_kind` while `kv::interleaved_layers`
        // asserts a positive period and aborts, so the loader, the budget
        // calculator, and gguf-info disagree about one file. Window 0 means
        // windowing is disabled everywhere else in the ecosystem (the numpy
        // reference raises on it) but yields an empty attention span here:
        // a context-blind model that still reads as fluent.
        ensure!(
            config.sliding_window > 0,
            "{arch}.attention.sliding_window is 0, must be positive; \
             a zero-token window is not disabled windowing"
        );
        ensure!(
            config.sliding_window_pattern > 0,
            "{arch}.attention.sliding_window_pattern is 0, must be positive"
        );
        ensure!(
            config.n_heads.is_multiple_of(config.n_kv_heads.max(1)) && config.n_kv_heads > 0,
            "head_count {} not divisible by head_count_kv {}",
            config.n_heads,
            config.n_kv_heads
        );
        Ok(config)
    }

    /// Attention kind of layer `i`, matching `kv::interleaved_layers`.
    pub fn layer_kind(&self, i: usize) -> LayerKind {
        if (i + 1).is_multiple_of(self.sliding_window_pattern) {
            LayerKind::Global
        } else {
            LayerKind::SlidingWindow(self.sliding_window)
        }
    }
}

impl ModelSpec {
    /// Read the model scalars and then one `LayerSpec` per layer.
    ///
    /// The per-layer axes come from paired keys: llama.cpp writes the
    /// full-attention value unsuffixed and the sliding-window value with an
    /// `_swa` suffix (`attention.key_length` and `attention.key_length_swa`,
    /// `rope.freq_base` and `rope.freq_base_swa`). A checkpoint that declares
    /// no `_swa` variant is uniform on that axis, and every layer takes the
    /// unsuffixed value.
    pub fn from_gguf(file: &GgufFile) -> Result<ModelSpec> {
        let config = Config::from_gguf(file)?;
        let arch = architecture(file)?;
        let u = |key: &str| -> Result<usize> {
            match file.metadata.get(&format!("{arch}.{key}")) {
                Some(MetaValue::U32(v)) => Ok(*v as usize),
                Some(other) => bail!("{arch}.{key}: expected u32, got {other:?}"),
                None => bail!("missing {arch}.{key}"),
            }
        };
        // Optional paired key. Absent means the axis is uniform, which is a
        // fact about the checkpoint rather than a defaulted guess; a present
        // key of the wrong type is still an error, because silently ignoring
        // it would apply the global value to the sliding layers.
        let u_opt = |key: &str| -> Result<Option<usize>> {
            match file.metadata.get(&format!("{arch}.{key}")) {
                Some(MetaValue::U32(v)) => Ok(Some(*v as usize)),
                Some(other) => bail!("{arch}.{key}: expected u32, got {other:?}"),
                None => Ok(None),
            }
        };
        let f = |key: &str| -> Result<f32> {
            match file.metadata.get(&format!("{arch}.{key}")) {
                Some(MetaValue::F32(v)) => Ok(*v),
                Some(other) => bail!("{arch}.{key}: expected f32, got {other:?}"),
                None => bail!("missing {arch}.{key}"),
            }
        };
        let f_opt = |key: &str| -> Result<Option<f32>> {
            match file.metadata.get(&format!("{arch}.{key}")) {
                Some(MetaValue::F32(v)) => Ok(Some(*v)),
                Some(other) => bail!("{arch}.{key}: expected f32, got {other:?}"),
                None => Ok(None),
            }
        };

        let head_dim_global = u("attention.key_length")?;
        let head_dim_sliding = u_opt("attention.key_length_swa")?.unwrap_or(head_dim_global);
        let rope_base_global = f("rope.freq_base")?;
        // The sliding regime's base, two orders of magnitude below the global
        // one on Gemma 4. Absent means one regime, which is the pre-Gemma-4
        // case and the shape of the committed fixture.
        let rope_base_sliding = f_opt("rope.freq_base_swa")?.unwrap_or(rope_base_global);

        // Feed-forward width is a uint32 scalar on E4B and an int32 array of
        // block_count entries on E2B, where the trailing layers are twice as
        // wide. Both forms are read into the same per-layer vector, so the
        // caller never learns which spelling the converter chose.
        let ffn_dims: Vec<usize> = match file.metadata.get(&format!("{arch}.feed_forward_length")) {
            Some(MetaValue::ArrI32(widths)) => {
                ensure!(
                    widths.len() == config.n_layers,
                    "{arch}.feed_forward_length has {} entries for {} layers",
                    widths.len(),
                    config.n_layers
                );
                widths
                    .iter()
                    .map(|w| usize::try_from(*w).unwrap_or(0))
                    .collect()
            }
            _ => vec![u("feed_forward_length")?; config.n_layers],
        };

        let layers: Vec<LayerSpec> = (0..config.n_layers)
            .map(|i| {
                let kind = config.layer_kind(i);
                let (head_dim, rope_freq_base) = match kind {
                    LayerKind::Global => (head_dim_global, rope_base_global),
                    LayerKind::SlidingWindow(_) => (head_dim_sliding, rope_base_sliding),
                };
                LayerSpec {
                    kind,
                    head_dim,
                    ffn_dim: ffn_dims[i],
                    rope_freq_base,
                }
            })
            .collect();

        // Each dimension divides or strides something in the forward pass, so
        // a zero reaches the arithmetic instead of a guard rather than failing
        // here: 0 is even, so a zero head width cleared the RoPE check and
        // then aborted the process at `chunks_mut(0)` on the first token.
        // Errors name the metadata key rather than only the layer, because
        // the key is what a reader has to go and fix.
        for (i, layer) in layers.iter().enumerate() {
            let head_key = match layer.kind {
                LayerKind::Global => "attention.key_length",
                LayerKind::SlidingWindow(_) => {
                    if file
                        .metadata
                        .contains_key(&format!("{arch}.attention.key_length_swa"))
                    {
                        "attention.key_length_swa"
                    } else {
                        "attention.key_length"
                    }
                }
            };
            ensure!(
                layer.head_dim > 0,
                "{arch}.{head_key} is 0 for layer {i}, must be positive"
            );
            ensure!(
                layer.head_dim.is_multiple_of(2),
                "{arch}.{head_key} is {} for layer {i}, must be even for RoPE",
                layer.head_dim
            );
            ensure!(
                layer.ffn_dim > 0,
                "{arch}.feed_forward_length is 0 for layer {i}, must be positive"
            );
        }
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

/// Proportional-RoPE frequency divisors, one per rotating pair of a
/// full-attention head.
///
/// Optional: a checkpoint whose global regime rotates its full width carries
/// no such tensor, and every frequency divides by an implicit 1.0. Present,
/// it is consumed rather than tolerated — see the file banner for why
/// treating it as inert was a latent wrong answer.
const ROPE_FREQS_TENSOR: &str = "rope_freqs.weight";

/// Collapse a tensor name to its family, replacing a leading `blk.<i>.` with
/// `blk.*.`, so a diagnostic over 42 layers reports one line per family.
pub fn tensor_family(name: &str) -> String {
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_string();
    };
    match rest.split_once('.') {
        Some((index, suffix)) if !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()) => {
            format!("blk.*.{suffix}")
        }
        _ => name.to_string(),
    }
}

/// Refuse a checkpoint whose structure the forward pass does not implement.
///
/// Loading is name-directed, so an unimplemented feature would otherwise
/// present as ignored tensors and plausible, wrong logits. Diagnoses the four
/// Gemma 4 features that change numerics by name before falling back to a
/// generic unconsumed-tensor report, so the message says which feature is
/// missing rather than only which tensor was unused.
///
/// Called by `CpuModel::from_gguf` and by the `gguf-info` binary, so the
/// inspection tool and the loader cannot disagree.
pub fn check_supported(file: &GgufFile, spec: &ModelSpec) -> Result<()> {
    let config = &spec.config;
    let names: Vec<&str> = file.tensors.iter().map(|t| t.name.as_str()).collect();

    // Per-layer embeddings: a token-identity table plus a projection, summed
    // and added as an extra residual update. Roughly 2.8B of E4B's parameters.
    if let Some(hit) = names.iter().find(|n| n.contains("per_layer")) {
        bail!(
            "checkpoint uses per-layer embeddings (tensor {hit}), which this \
             backend does not implement; loading it would ignore the per-layer \
             tables and produce wrong logits"
        );
    }

    // Shared KV cache: the trailing layers reuse K/V from the last non-shared
    // layer, so they carry no attn_k/attn_v of their own.
    for i in 0..config.n_layers {
        let has = |suffix: &str| names.contains(&format!("blk.{i}.{suffix}.weight").as_str());
        if has("attn_q") && !(has("attn_k") && has("attn_v")) {
            bail!(
                "layer {i} has no attn_k/attn_v tensors, the shared-KV-cache \
                 signature; this backend gives every layer its own cache and \
                 does not implement KV sharing"
            );
        }
    }

    // Per-layer feed-forward widths are supported, so this is a consistency
    // check rather than a refusal: the declared width for a layer must be the
    // width its weights actually carry. A disagreement means the metadata and
    // the tensors describe different models, and the forward pass slices by
    // the metadata.
    //
    // Not "MatFormer": that word appears nowhere in the architecture
    // reference. E2B varies the width because `use_double_wide_mlp` doubles
    // its trailing layers, and E4B is uniform despite sharing KV over the same
    // region, so the widening is not implied by the tail (docs/models/gemma-4.md).
    for info in &file.tensors {
        let Some(rest) = info.name.strip_prefix("blk.") else {
            continue;
        };
        let Some((index, suffix)) = rest.split_once('.') else {
            continue;
        };
        if suffix != "ffn_up.weight" {
            continue;
        }
        let Ok(layer) = index.parse::<usize>() else {
            continue;
        };
        let Some(declared) = spec.layers.get(layer) else {
            continue;
        };
        let width = info.dims.last().copied().unwrap_or(0);
        ensure!(
            width == declared.ffn_dim as u64,
            "tensor {} has feed-forward width {width}, but {}.feed_forward_length \
             declares {} for layer {layer}",
            info.name,
            architecture(file)?,
            declared.ffn_dim
        );
    }

    // RoPE configuration. Both regimes are implemented, so the paired keys
    // are accepted and checked rather than refused; anything else under the
    // `rope.` prefix is still a scaling scheme the forward pass would ignore.
    //
    // The allowlist stays an allowlist. It is the only thing standing between
    // a converter that adds a third regime and a silent wrong rotation, and
    // it is deliberately not widened to "anything that looks descriptive".
    let arch = architecture(file)?;
    let rope_prefix = format!("{arch}.rope.");
    // Rotation width is checked against the regime that key governs: the
    // unsuffixed key is the full-attention value, `_swa` the sliding one.
    // Comparing both against one width is how a 256-wide sliding regime gets
    // read as a partial rotation of a 512-wide head, or vice versa.
    let regime_width = |global: bool| -> usize {
        spec.layers
            .iter()
            .filter(|l| matches!(l.kind, LayerKind::Global) == global)
            .map(|l| l.head_dim)
            .next()
            .unwrap_or(0)
    };
    for key in file.metadata.keys() {
        let Some(suffix) = key.strip_prefix(&rope_prefix) else {
            continue;
        };
        let width = match suffix {
            "freq_base" | "freq_base_swa" => continue,
            "dimension_count" => regime_width(true),
            "dimension_count_swa" => regime_width(false),
            _ => bail!(
                "metadata key {key} configures RoPE beyond the two frequency bases and \
                 rotation widths this backend implements"
            ),
        };
        // No layer in that regime means the key describes nothing here, which
        // is not a disagreement. Typed exhaustively rather than with a bare
        // `if let`: a value decoded as anything but u32 used to fall through
        // and be accepted as inert, so a rotation width the forward pass
        // ignores reached the numerics unremarked.
        if width == 0 {
            continue;
        }
        match file.metadata.get(key) {
            Some(MetaValue::U32(v)) => ensure!(
                *v as usize == width,
                "{key} is {v} but that regime's head width is {width}; a rotation \
                 narrower than the head is partial rotary embedding applied by slicing, \
                 which this backend does not implement -- Gemma 4 expresses its partial \
                 rotation through rope_freqs.weight at full width instead"
            ),
            Some(other) => bail!("{key}: expected u32, got {other:?}"),
            None => {}
        }
    }

    // Proportional-RoPE divisors, if present. One entry per rotating pair of
    // a global head: 64 ones followed by 192 values of 1e30 on E4B, which
    // drive those angles to zero. Length is checked here because the loader
    // divides by it elementwise and a short tensor would leave the tail
    // rotating at full frequency.
    if let Some(info) = file.tensors.iter().find(|t| t.name == ROPE_FREQS_TENSOR) {
        let want = regime_width(true) / 2;
        let got = info.dims.iter().product::<u64>() as usize;
        ensure!(
            want > 0 && got == want,
            "{ROPE_FREQS_TENSOR} has {got} entries, but the global head width implies {want}"
        );
    }

    // Numeric-affecting metadata carries no tensor, so the unconsumed-tensor
    // backstop below cannot see it: Gemma-2 publishes both softcapping keys,
    // and either one changes every logit with nothing left over to notice.
    // The refused keys are named rather than inverted into an allowlist of
    // inert keys, which would also refuse every descriptive key a converter
    // adds later — a far larger behavioral change than closing this gap.
    let arch_prefix = format!("{arch}.");
    for key in file.metadata.keys() {
        let Some(suffix) = key.strip_prefix(&arch_prefix) else {
            continue;
        };
        if suffix.contains("softcapping") || matches!(suffix, "attention.scale" | "logit_scale") {
            bail!(
                "metadata key {key} caps or rescales attention scores or logits, which \
                 this forward pass does not implement; it carries no tensor, so loading \
                 would change every logit silently"
            );
        }
    }

    // Generic backstop: anything left unconsumed is structure this backend
    // does not know about, including features published after this code.
    let mut unconsumed: Vec<String> = Vec::new();
    for name in &names {
        if TOP_LEVEL_TENSORS.contains(name) || *name == ROPE_FREQS_TENSOR {
            continue;
        }
        let consumed = (0..config.n_layers).any(|i| {
            let prefix = format!("blk.{i}.");
            name.strip_prefix(&prefix)
                .is_some_and(|s| LAYER_TENSORS.contains(&s))
        });
        if !consumed {
            let family = tensor_family(name);
            if !unconsumed.contains(&family) {
                unconsumed.push(family);
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
    Ok(())
}

/// Architecture string, the prefix of every model metadata key.
fn architecture(file: &GgufFile) -> Result<&str> {
    match file.metadata.get("general.architecture") {
        Some(MetaValue::Str(a)) => Ok(a.as_str()),
        _ => bail!("missing general.architecture"),
    }
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

/// Observer of the residual stream, the interp tap seam. Fires after each
/// decoder layer's final residual add — the hidden state nnsight exposes as
/// `block.output[0]` — so probes trained on Python-captured activations and
/// on rust-captured activations see the same tensor. Any backend added later
/// (CUDA) must preserve this seam and its semantics.
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
        // down. The carving asserts it never exceeds this.
        let elements = 5 * c.embed_dim + 2 * q_dim + 2 * kv_dim + 3 * spec.max_ffn_dim();
        Scratch {
            buf: vec![0.0; elements],
            scores: Vec::new(),
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
    pub fn from_gguf(file: &GgufFile, data: &[u8]) -> Result<CpuModel> {
        let spec = ModelSpec::from_gguf(file)?;
        let config = spec.config;
        // Refuse unimplemented architecture features before loading a single
        // weight: a partial load emits fluent, wrong text.
        check_supported(file, &spec)?;
        // Every tensor is length-checked against the config as it loads.
        // Loading is name-directed and the forward pass slices by strides
        // computed from the metadata, so a tensor whose shape disagrees with
        // the config is caught nowhere later: it reads the wrong rows and
        // returns finite, wrong logits, or slices past the end and panics
        // mid-forward. Both are silent to a caller that only sees fluent
        // text, which is what this backend refuses to be.
        let tensor = |name: &str, want: usize| -> Result<Vec<f32>> {
            let info = file
                .tensors
                .iter()
                .find(|t| t.name == name)
                .with_context(|| format!("missing tensor {name}"))?;
            let values = load_tensor(file, data, info)?;
            ensure!(
                values.len() == want,
                "tensor {name} has {} elements, the configuration implies {want}",
                values.len()
            );
            Ok(values)
        };
        let embed_dim = config.embed_dim;
        let embed = tensor("token_embd.weight", config.vocab_size * embed_dim)?;
        // Divisors apply to the full-attention regime only: the sliding
        // regime rotates its full width, and its rotation width is already
        // the whole head (docs/rust-harness/gemma-4-ingestion.md).
        let rope_divisors: Option<Vec<f32>> =
            match file.tensors.iter().find(|t| t.name == ROPE_FREQS_TENSOR) {
                Some(info) => Some(load_tensor(file, data, info)?),
                None => None,
            };
        let mut layers = Vec::with_capacity(config.n_layers);
        for (i, ls) in spec.layers.iter().enumerate() {
            // Widths come from this layer's own spec, so a checkpoint that
            // interleaves head widths or feed-forward widths is length-checked
            // against what it declares for each layer rather than against the
            // first one.
            let (head_dim, ffn_dim) = (ls.head_dim, ls.ffn_dim);
            let q_dim = config.n_heads * head_dim;
            let kv_dim = config.n_kv_heads * head_dim;
            let divisors = match (ls.kind, rope_divisors.as_deref()) {
                (LayerKind::Global, Some(d)) => Some(d),
                _ => None,
            };
            let t = |suffix: &str, want: usize| tensor(&format!("blk.{i}.{suffix}"), want);
            layers.push(LayerWeights {
                spec: *ls,
                rope_freqs: rope_frequencies(head_dim, ls.rope_freq_base, divisors),
                attn_norm: t("attn_norm.weight", embed_dim)?,
                wq: t("attn_q.weight", embed_dim * q_dim)?,
                wk: t("attn_k.weight", embed_dim * kv_dim)?,
                wv: t("attn_v.weight", embed_dim * kv_dim)?,
                q_norm: t("attn_q_norm.weight", head_dim)?,
                k_norm: t("attn_k_norm.weight", head_dim)?,
                wo: t("attn_output.weight", q_dim * embed_dim)?,
                post_attn_norm: t("post_attention_norm.weight", embed_dim)?,
                ffn_norm: t("ffn_norm.weight", embed_dim)?,
                w_gate: t("ffn_gate.weight", embed_dim * ffn_dim)?,
                w_up: t("ffn_up.weight", embed_dim * ffn_dim)?,
                w_down: t("ffn_down.weight", ffn_dim * embed_dim)?,
                post_ffw_norm: t("post_ffw_norm.weight", embed_dim)?,
            });
        }
        let output_norm = tensor("output_norm.weight", embed_dim)?;
        Ok(CpuModel {
            kv: vec![LayerKv::default(); config.n_layers],
            config,
            scratch: Scratch::new(&spec),
            spec: Arc::new(spec),
            weights: Arc::new(Weights {
                embed,
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

    /// Cached K (or V) elements per token in layer `i`. Varies with the
    /// layer's head width, so every cache walk has to ask per layer.
    fn kv_dim(&self, layer: usize) -> usize {
        self.spec.kv_dim(layer)
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
        let Scratch { buf, scores } = scratch;
        let (x, work) = buf.split_at_mut(c.embed_dim);

        // Embedding, scaled by sqrt(embed_dim) per Gemma convention.
        let row = &w.embed[token as usize * c.embed_dim..(token as usize + 1) * c.embed_dim];
        let scale = (c.embed_dim as f32).sqrt();
        for (dst, e) in x.iter_mut().zip(row) {
            *dst = e * scale;
        }

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

            // Attention block.
            rms_norm_into(x, &layer.attn_norm, c.rms_eps, normed);
            matvec_into(&layer.wq, normed, q);
            matvec_into(&layer.wk, normed, k);
            matvec_into(&layer.wv, normed, v);
            for head in q.chunks_mut(hd) {
                rms_norm_in_place(head, &layer.q_norm, c.rms_eps);
                rope(head, position, &layer.rope_freqs);
            }
            for head in k.chunks_mut(hd) {
                rms_norm_in_place(head, &layer.k_norm, c.rms_eps);
                rope(head, position, &layer.rope_freqs);
            }
            let cache = &mut kv[i];
            cache.k.extend_from_slice(k);
            cache.v.extend_from_slice(v);

            // Causal attention over the cache; sliding-window layers see
            // only the trailing `window` positions including this one.
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

            // Steering before the tap, so captures see the steered stream.
            if let Some(dir) = control.as_ref().and_then(|cv| cv.direction(i)) {
                let scale = control.as_ref().expect("checked above").scale;
                for (a, b) in x.iter_mut().zip(dir) {
                    *a += scale * b;
                }
            }

            // Interp tap seam: the layer's complete residual output.
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
                row.iter().zip(hidden).map(|(a, b)| a * b).sum()
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
            let kv_dim = self.spec.kv_dim(i);
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
    // backend stores the full sequence in every layer, windowed or not), so a
    // length mismatch on load is a hard error rather than a guess.
    //
    // The per-layer head width is why the row length is recomputed per layer
    // rather than hoisted: a checkpoint interleaving two head widths writes
    // rows of two sizes, and one hoisted stride would read the second regime
    // at the first regime's offset.
    fn save_state(&self, w: &mut dyn std::io::Write) -> Result<()> {
        w.write_all(&(self.kv.len() as u64).to_le_bytes())?;
        w.write_all(&(self.cached_len as u64).to_le_bytes())?;
        for (i, cache) in self.kv.iter().enumerate() {
            let kv_dim = self.kv_dim(i);
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
            let elements = cached_len * self.spec.kv_dim(i);
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
    use crate::synth::{build_tiny_model, TinySpec};

    fn tiny() -> CpuModel {
        let image = build_tiny_model(&TinySpec::default());
        let file = crate::gguf::parse(&image).unwrap();
        CpuModel::from_gguf(&file, &image).unwrap()
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
    fn the_two_rope_regimes_take_their_own_base_and_head_width() {
        // A1-R04 was refuted only because every path refused a second base
        // before reaching the numerics. Now that both regimes load, the
        // per-layer assignment is the thing under test rather than the
        // refusal that used to stand in front of it.
        let (_, mut file, _) = tiny_parsed();
        file.metadata
            .insert("gemma4.rope.freq_base_swa".into(), MetaValue::F32(10000.0));
        file.metadata
            .insert("gemma4.rope.freq_base".into(), MetaValue::F32(1_000_000.0));
        let spec = ModelSpec::from_gguf(&file).unwrap();
        // The tiny fixture is layer 0 sliding, layer 1 global.
        assert_eq!(spec.layers[0].kind, LayerKind::SlidingWindow(4));
        assert_eq!(spec.layers[1].kind, LayerKind::Global);
        assert_eq!(spec.layers[0].rope_freq_base, 10000.0);
        assert_eq!(spec.layers[1].rope_freq_base, 1_000_000.0);
    }

    #[test]
    fn a_sliding_head_width_is_read_from_the_swa_key() {
        let (_, mut file, _) = tiny_parsed();
        file.metadata
            .insert("gemma4.attention.key_length_swa".into(), MetaValue::U32(4));
        let spec = ModelSpec::from_gguf(&file).unwrap();
        assert_eq!(
            spec.layers[0].head_dim, 4,
            "sliding layer takes the _swa key"
        );
        assert_eq!(spec.layers[1].head_dim, 8, "global layer keeps key_length");
        assert_eq!(spec.max_head_dim(), 8);
    }

    #[test]
    fn each_rotation_width_is_checked_against_its_own_regime() {
        // One width checked against both regimes reads a 4-wide sliding
        // rotation as a partial rotation of an 8-wide head. Give the two
        // regimes different widths and confirm each key is held to its own.
        let (_, mut file, _) = tiny_parsed();
        file.metadata
            .insert("gemma4.attention.key_length_swa".into(), MetaValue::U32(4));
        file.metadata.insert(
            "gemma4.rope.dimension_count".into(),
            MetaValue::U32(8), // global head width
        );
        file.metadata.insert(
            "gemma4.rope.dimension_count_swa".into(),
            MetaValue::U32(4), // sliding head width
        );
        let spec = ModelSpec::from_gguf(&file).unwrap();
        check_supported(&file, &spec).unwrap();

        // Swapping them makes each key wrong for the regime it governs.
        file.metadata
            .insert("gemma4.rope.dimension_count_swa".into(), MetaValue::U32(8));
        let err = check_supported(&file, &spec).unwrap_err().to_string();
        assert!(err.contains("dimension_count_swa"), "{err}");
        assert!(err.contains("head width is 4"), "{err}");
    }

    #[test]
    fn an_unknown_rope_key_is_still_refused() {
        // The allowlist is what stands between a converter that adds a third
        // regime and a silent wrong rotation. Widening it to the two Gemma 4
        // regimes must not have widened it to anything else.
        let (_, mut file, spec) = tiny_parsed();
        file.metadata
            .insert("gemma4.rope.scaling.factor".into(), MetaValue::F32(8.0));
        let err = check_supported(&file, &spec).unwrap_err().to_string();
        assert!(err.contains("gemma4.rope.scaling.factor"), "{err}");
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
            let image = build_tiny_model(spec);
            let file = crate::gguf::parse(&image).unwrap();
            let mut model = CpuModel::from_gguf(&file, &image).unwrap();
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
        let image = build_tiny_model(&spec);
        let file = crate::gguf::parse(&image).unwrap();
        let model = CpuModel::from_gguf(&file, &image).unwrap();
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
    fn a_rope_divisor_tensor_of_the_wrong_length_is_refused() {
        // The loader divides elementwise, so a short tensor would either
        // panic or leave the tail rotating at full frequency depending on
        // where it ran out.
        let (_, mut file, spec) = tiny_parsed();
        let mut info = extra_tensor(ROPE_FREQS_TENSOR);
        info.dims = vec![3]; // global head width 8 implies 4
        file.tensors.push(info);
        let err = check_supported(&file, &spec).unwrap_err().to_string();
        assert!(err.contains(ROPE_FREQS_TENSOR), "{err}");
        assert!(err.contains("3") && err.contains("4"), "{err}");
    }

    #[test]
    fn config_parses_from_the_tiny_model() {
        let m = tiny();
        let c = m.config();
        assert_eq!(c.vocab_size, 288);
        assert_eq!(c.n_layers, 2);
        assert_eq!(c.n_heads, 2);
        assert_eq!(c.n_kv_heads, 1);
        assert_eq!(c.layer_kind(0), LayerKind::SlidingWindow(4));
        assert_eq!(c.layer_kind(1), LayerKind::Global);
    }

    /// Parsed synthetic checkpoint, for mutating the tensor directory or
    /// metadata in place. `check_supported` reads names, dims, and metadata
    /// only, so a mutated directory needs no re-encoded payload.
    fn tiny_parsed() -> (Vec<u8>, crate::gguf::GgufFile, ModelSpec) {
        let image = build_tiny_model(&TinySpec::default());
        let file = crate::gguf::parse(&image).unwrap();
        let spec = ModelSpec::from_gguf(&file).unwrap();
        (image, file, spec)
    }

    fn extra_tensor(name: &str) -> crate::gguf::TensorInfo {
        crate::gguf::TensorInfo {
            name: name.to_string(),
            dims: vec![4],
            ggml_type: crate::tensor::GGML_F32,
            offset: 0,
        }
    }

    #[test]
    fn the_synthetic_fixture_consumes_every_tensor_it_ships() {
        // Couples the guard's tensor lists to the loader: a tensor the loader
        // reads but the lists omit makes the guard refuse this fixture.
        let (_, file, spec) = tiny_parsed();
        check_supported(&file, &spec).unwrap();
    }

    #[test]
    fn per_layer_embedding_tensors_are_refused() {
        let (_, mut file, spec) = tiny_parsed();
        file.tensors
            .push(extra_tensor("per_layer_token_embd.weight"));
        let err = check_supported(&file, &spec).unwrap_err().to_string();
        assert!(err.contains("per-layer embeddings"), "{err}");
    }

    #[test]
    fn a_layer_without_kv_tensors_is_refused_as_shared_kv() {
        // Diagnosed as KV sharing, not as a bare missing tensor, and refused
        // by the loader before any weight is read.
        let (image, mut file, _) = tiny_parsed();
        file.tensors.retain(|t| t.name != "blk.1.attn_v.weight");
        let err = match CpuModel::from_gguf(&file, &image) {
            Ok(_) => panic!("a checkpoint missing attn_v must not load"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("shared-KV-cache"), "{err}");
        assert!(err.contains("layer 1"), "{err}");
    }

    #[test]
    fn a_rotation_narrower_than_the_head_is_refused() {
        // Gemma 4 does rotate a quarter of each global head, but it expresses
        // that at full width through rope_freqs.weight. A dimension_count
        // below the head width means the other encoding -- rotate a slice,
        // leave the remainder alone -- which is a different operation and is
        // not implemented.
        let (_, mut file, spec) = tiny_parsed();
        file.metadata.insert(
            "gemma4.rope.dimension_count".into(),
            MetaValue::U32(spec.max_head_dim() as u32 / 2),
        );
        let err = check_supported(&file, &spec).unwrap_err().to_string();
        assert!(err.contains("dimension_count"), "{err}");
        assert!(err.contains("rope_freqs.weight"), "{err}");
        // The same key equal to the head width is a full rotation: accepted.
        file.metadata.insert(
            "gemma4.rope.dimension_count".into(),
            MetaValue::U32(spec.max_head_dim() as u32),
        );
        check_supported(&file, &spec).unwrap();
    }

    #[test]
    fn a_rope_key_outside_the_two_implemented_regimes_is_refused() {
        // gemma3's local base arrives under a different spelling from
        // gemma4's `freq_base_swa`, and nothing in the forward pass would
        // read it.
        let (_, mut file, spec) = tiny_parsed();
        file.metadata.insert(
            "gemma4.rope.local_freq_base".into(),
            MetaValue::F32(10000.0),
        );
        let err = check_supported(&file, &spec).unwrap_err().to_string();
        assert!(err.contains("gemma4.rope.local_freq_base"), "{err}");
        assert!(err.contains("two frequency bases"), "{err}");
    }

    #[test]
    fn unknown_tensor_families_are_refused_and_collapsed_by_layer() {
        let (_, mut file, spec) = tiny_parsed();
        for i in 0..spec.config.n_layers {
            file.tensors
                .push(extra_tensor(&format!("blk.{i}.attn_v_norm.weight")));
        }
        let err = check_supported(&file, &spec).unwrap_err().to_string();
        assert!(err.contains("1 tensor family"), "{err}");
        assert!(err.contains("blk.*.attn_v_norm.weight"), "{err}");
    }

    #[test]
    fn inert_tensors_do_not_trip_the_guard() {
        let (_, mut file, spec) = tiny_parsed();
        file.tensors.push(extra_tensor("rope_freqs.weight"));
        check_supported(&file, &spec).unwrap();
    }

    #[test]
    fn numeric_metadata_with_no_tensor_to_notice_is_refused() {
        // A pure-metadata feature leaves no tensor for the unconsumed-tensor
        // backstop to catch, so a Gemma-2 checkpoint's softcapping was
        // ignored and every logit came out wrong with nothing to notice.
        for (key, value) in [
            ("gemma4.attn_logit_softcapping", 50.0f32),
            ("gemma4.final_logit_softcapping", 30.0),
            ("gemma4.attention.scale", 0.0625),
            ("gemma4.logit_scale", 1.5),
        ] {
            let (_, mut file, spec) = tiny_parsed();
            file.metadata.insert(key.into(), MetaValue::F32(value));
            let err = match check_supported(&file, &spec) {
                Ok(()) => panic!("{key} must not load"),
                Err(e) => e.to_string(),
            };
            assert!(err.contains(key), "{err}");
        }
        // The guard names the numeric keys instead of inverting the rope
        // check into an allowlist, so a descriptive key still loads.
        let (_, mut file, spec) = tiny_parsed();
        file.metadata
            .insert("gemma4.vocab_size".into(), MetaValue::U32(288));
        check_supported(&file, &spec).unwrap();
    }

    #[test]
    fn a_consumed_tensor_of_the_wrong_shape_is_refused_naming_the_tensor() {
        // Self-consistent checkpoint: the directory declares the smaller
        // shape and the payload supplies exactly that many elements, so no
        // later stage can notice. The forward pass then either strides
        // through the wrong rows and returns finite, wrong logits (a norm
        // tensor) or slices past the end and panics mid-forward (a per-head
        // norm). ffn_up is absent because its width already had a check.
        for (name, elements, want) in [
            ("blk.0.attn_norm.weight", 8, 16),
            ("blk.1.post_ffw_norm.weight", 4, 16),
            ("blk.1.attn_q_norm.weight", 4, 8),
            ("blk.0.attn_k_norm.weight", 2, 8),
            ("blk.0.attn_q.weight", 128, 256),
            ("blk.0.attn_k.weight", 64, 128),
            ("blk.1.attn_v.weight", 64, 128),
            ("blk.0.attn_output.weight", 128, 256),
            ("blk.0.ffn_gate.weight", 256, 512),
            ("blk.0.ffn_down.weight", 256, 512),
            ("blk.1.ffn_norm.weight", 8, 16),
            ("blk.1.post_attention_norm.weight", 8, 16),
            ("output_norm.weight", 8, 16),
        ] {
            let (image, mut file, _) = tiny_parsed();
            let info = file.tensors.iter_mut().find(|t| t.name == name).unwrap();
            info.dims = vec![elements as u64];
            let err = match CpuModel::from_gguf(&file, &image) {
                Ok(_) => panic!("{name} at {elements} elements must not load"),
                Err(e) => e.to_string(),
            };
            assert!(err.contains(name), "{err}");
            assert!(err.contains(&elements.to_string()), "{err}");
            assert!(err.contains(&want.to_string()), "{err}");
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
        let image = build_tiny_model(&spec);
        let file = crate::gguf::parse(&image).unwrap();
        let mut m = CpuModel::from_gguf(&file, &image).unwrap();
        let logits = m.prefill(&[1, 5, 9]).unwrap();
        assert_eq!(logits.0.len(), spec.vocab_size);
        assert!(logits.0.iter().all(|v| v.is_finite()));
    }

    /// The three shipped Gemma 4 schedules, as `gguf-info` reports them:
    /// (name, per-layer bool array, the stride it is equivalent to).
    /// `false` marks a global-attention layer.
    fn shipped_schedules() -> Vec<(&'static str, Vec<bool>, usize)> {
        let repeat = |period: usize, layers: usize| -> Vec<bool> {
            (0..layers)
                .map(|i| !(i + 1).is_multiple_of(period))
                .collect()
        };
        vec![
            ("E2B", repeat(5, 35), 5),
            ("E4B", repeat(6, 42), 6),
            ("mtp", repeat(4, 4), 4),
        ]
    }

    #[test]
    fn a_bool_attention_schedule_decodes_to_the_stride_it_encodes() {
        // llama.cpp writes Gemma 4's schedule as a per-layer bool array where
        // this backend models a u32 stride. The two are the same schedule in
        // different encodings, so the array must decode rather than be
        // refused -- but only the periodic case, which is what ships.
        for (name, pattern, want_stride) in shipped_schedules() {
            let (_, mut file, _) = tiny_parsed();
            file.metadata.insert(
                "gemma4.attention.sliding_window_pattern".into(),
                MetaValue::ArrBool(pattern.clone()),
            );
            let config = Config::from_gguf(&file)
                .unwrap_or_else(|e| panic!("{name} schedule must decode: {e}"));
            assert_eq!(config.sliding_window_pattern, want_stride, "{name}");
            // The stride is only correct if it reproduces the array it came
            // from; asserting the number alone would pass on an off-by-one.
            for (i, &sliding) in pattern.iter().enumerate() {
                let kind = config.layer_kind(i);
                let global = matches!(kind, LayerKind::Global);
                assert_eq!(!sliding, global, "{name} layer {i}: {kind:?}");
            }
        }
    }

    #[test]
    fn a_bool_schedule_that_is_not_a_stride_is_refused_naming_the_key() {
        // Deriving a period from the first global layer and then not checking
        // the rest would silently mis-schedule attention for every later
        // layer, which reads as a fluent but context-blind model rather than
        // as an error. Both shapes below share a first-false index with a
        // schedule that is periodic.
        for pattern in [
            vec![true, true, false, true, false, true], // period drifts
            vec![true, true, true, true],               // no global layer at all
        ] {
            let (_, mut file, _) = tiny_parsed();
            file.metadata.insert(
                "gemma4.attention.sliding_window_pattern".into(),
                MetaValue::ArrBool(pattern.clone()),
            );
            let err = match Config::from_gguf(&file) {
                Ok(c) => panic!(
                    "{pattern:?} must not decode, got stride {}",
                    c.sliding_window_pattern
                ),
                Err(e) => e.to_string(),
            };
            assert!(err.contains("sliding_window_pattern"), "{err}");
            // Without this, the test passes against the unfixed loader, which
            // refuses every bool array with a bare type error that happens to
            // name the key. The refusal has to be about periodicity.
            assert!(
                !err.contains("expected u32"),
                "refused as a type error: {err}"
            );
            assert!(err.contains("periodic"), "{err}");
        }
    }

    #[test]
    fn a_uniform_per_layer_feed_forward_array_decodes_to_that_width() {
        // E4B writes one width as a scalar; E2B writes an array. Both reach
        // the same per-layer vector, so nothing downstream learns which
        // spelling the converter chose.
        let (_, mut file, spec) = tiny_parsed();
        let width = spec.layers[0].ffn_dim;
        let n = spec.config.n_layers;
        file.metadata.insert(
            "gemma4.feed_forward_length".into(),
            MetaValue::ArrI32(vec![width as i32; n]),
        );
        let reparsed = ModelSpec::from_gguf(&file).unwrap();
        assert!(
            reparsed.layers.iter().all(|l| l.ffn_dim == width),
            "{:?}",
            reparsed.layers
        );
    }

    #[test]
    fn a_per_layer_feed_forward_array_decodes_to_one_width_per_layer() {
        // E2B's feed-forward width varies by layer: 6144 for layers 0..=14 and
        // 12288 for 15..=34. The cause is use_double_wide_mlp, which doubles
        // the MLP over the KV-shared tail (num_hidden_layers -
        // num_kv_shared_layers = 35 - 20 = 15). E4B sets the flag false and
        // carries one width despite sharing KV over the same region, so the
        // widening follows the flag and not the tail. Neither is a MatFormer
        // slice; that word appears nowhere in the architecture reference.
        // See schemas/gemma4_facts.json and docs/models/gemma-4.md.
        let (_, mut file, spec) = tiny_parsed();
        let narrow = spec.layers[0].ffn_dim;
        let wide = narrow * 2;
        assert_eq!(spec.config.n_layers, 2, "fixture shape assumed below");
        file.metadata.insert(
            "gemma4.feed_forward_length".into(),
            MetaValue::ArrI32(vec![narrow as i32, wide as i32]),
        );
        let reparsed = ModelSpec::from_gguf(&file).unwrap();
        assert_eq!(reparsed.layers[0].ffn_dim, narrow, "{:?}", reparsed.layers);
        assert_eq!(reparsed.layers[1].ffn_dim, wide, "{:?}", reparsed.layers);
        assert_eq!(reparsed.max_ffn_dim(), wide);
    }

    #[test]
    fn a_feed_forward_array_of_the_wrong_length_is_refused_naming_both_counts() {
        // An array is one entry per layer by definition. A shorter or longer
        // one describes a different model, and indexing it by layer would
        // either panic or silently read a neighbour's width.
        let (_, mut file, spec) = tiny_parsed();
        let width = spec.layers[0].ffn_dim as i32;
        file.metadata.insert(
            "gemma4.feed_forward_length".into(),
            MetaValue::ArrI32(vec![width; spec.config.n_layers + 1]),
        );
        let err = ModelSpec::from_gguf(&file).unwrap_err().to_string();
        assert!(err.contains("feed_forward_length"), "{err}");
        assert!(err.contains("3") && err.contains("2"), "{err}");
    }

    #[test]
    fn a_declared_width_that_the_weights_contradict_is_refused_naming_the_layer() {
        // Per-layer widths are supported, so the guard is no longer a refusal
        // of the feature but a check that metadata and tensors agree. The
        // forward pass slices by the metadata, so a disagreement reads the
        // wrong rows.
        let (_, mut file, spec) = tiny_parsed();
        let t = file
            .tensors
            .iter_mut()
            .find(|t| t.name == "blk.1.ffn_up.weight")
            .unwrap();
        *t.dims.last_mut().unwrap() += 8;
        let err = check_supported(&file, &spec).unwrap_err().to_string();
        assert!(err.contains("blk.1.ffn_up.weight"), "{err}");
        assert!(err.contains("feed-forward width"), "{err}");
        assert!(err.contains("layer 1"), "{err}");
    }

    #[test]
    fn a_zero_load_bearing_dimension_is_a_load_error_not_a_panic() {
        // attention.key_length = 0 clears the only dimension check there was
        // (even, for RoPE) and then reaches `chunks_mut(0)`, aborting the
        // process on the first token; the others divide or stride by zero
        // the same way. Hostile metadata owes a positioned error naming the
        // key, not exit 101.
        for key in [
            "embedding_length",
            "block_count",
            "attention.head_count",
            "attention.key_length",
            "feed_forward_length",
        ] {
            let (_, mut file, _) = tiny_parsed();
            file.metadata
                .insert(format!("gemma4.{key}"), MetaValue::U32(0));
            // Through ModelSpec, which carries the whole declared shape: head
            // width and feed-forward width are per-layer axes now, validated
            // there rather than on the model-wide Config.
            let err = match ModelSpec::from_gguf(&file) {
                Ok(_) => panic!("gemma4.{key} = 0 must not parse"),
                Err(e) => e.to_string(),
            };
            assert!(err.contains(key), "{err}");
        }
    }

    #[test]
    fn a_zero_sliding_window_or_pattern_is_a_load_error() {
        // Pattern 0 makes every layer sliding-window in `layer_kind` while
        // `kv::interleaved_layers` asserts a positive period and aborts, so
        // the loader, the budget calculator, and gguf-info disagreed about
        // one file. Window 0 reads as disabled windowing across the
        // ecosystem (the numpy reference raises on it) but yields an empty
        // attention span here: a context-blind model that still reads as
        // fluent.
        for key in [
            "attention.sliding_window",
            "attention.sliding_window_pattern",
        ] {
            let (_, mut file, _) = tiny_parsed();
            file.metadata
                .insert(format!("gemma4.{key}"), MetaValue::U32(0));
            let err = match Config::from_gguf(&file) {
                Ok(_) => panic!("gemma4.{key} = 0 must not parse"),
                Err(e) => e.to_string(),
            };
            assert!(err.contains(key), "{err}");
        }
    }

    #[test]
    fn tensor_family_collapses_only_numeric_layer_indices() {
        assert_eq!(tensor_family("blk.17.ffn_up.weight"), "blk.*.ffn_up.weight");
        assert_eq!(tensor_family("token_embd.weight"), "token_embd.weight");
        assert_eq!(tensor_family("blk.x.weight"), "blk.x.weight");
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
        let image = build_tiny_model(&spec);
        let file = crate::gguf::parse(&image).unwrap();
        let mut m = CpuModel::from_gguf(&file, &image).unwrap();
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
        let image = build_tiny_model(&TinySpec::default());
        let file = crate::gguf::parse(&image).unwrap();
        let model = || CpuModel::from_gguf(&file, &image).unwrap();

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
        let image = build_tiny_model(&spec);
        let file = crate::gguf::parse(&image).unwrap();
        let mut a = CpuModel::from_gguf(&file, &image).unwrap();
        let mut b = CpuModel::from_gguf(&file, &image).unwrap();
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
