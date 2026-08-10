/*
 * checkpoint — Hugging Face checkpoint reader: config.json and model.safetensors.
 *
 * The published form of a Gemma 4 checkpoint, and the format the CPU backend
 * loads weights from. Read alongside `gguf.rs`, which stays for the two
 * artifacts the Python interp exporters write (probe images and control
 * vectors) and no longer serves model weights.
 *
 * The distinction worth carrying into the reader: llama.cpp's converter has
 * to encode Gemma 4's architecture in metadata keys that predate it, so
 * several facts survive only as something else -- partial rotation as a
 * precomputed divisor tensor, the attention schedule as a stride that an
 * aperiodic model could not express, the vocabulary size as the length of a
 * tokenizer array. `config.json` states each of them as a named field. This
 * module is correspondingly smaller than the GGUF path it replaces, and the
 * places it is smaller are the places the GGUF path was guessing.
 */

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

use crate::tensor::{bf16_to_f32, f16_to_f32};

/// Weights file name inside a checkpoint directory.
pub const WEIGHTS_FILE: &str = "model.safetensors";
/// Architecture file name inside a checkpoint directory.
pub const CONFIG_FILE: &str = "config.json";

/// The activation the CPU backend implements. Gemma uses the tanh
/// approximation of GELU, and a checkpoint asking for anything else would be
/// silently mis-evaluated rather than refused.
const SUPPORTED_ACTIVATION: &str = "gelu_pytorch_tanh";

/// Prefix of every text-stack tensor.
///
/// Gemma 4 publishes as `Gemma4ForConditionalGeneration`, so the language
/// model is one tower among three and its tensors sit a level deeper than a
/// text-only release would put them. The audio and vision towers live beside
/// it under `model.` and this backend does not read them
/// (`cpu::NON_TEXT_PREFIXES`).
pub const TEXT_PREFIX: &str = "model.language_model.";

/// Attention regime of one layer, as `config.json` spells it.
const FULL_ATTENTION: &str = "full_attention";
const SLIDING_ATTENTION: &str = "sliding_attention";

/// RoPE regimes this backend implements, as `rope_parameters.*.rope_type`
/// spells them.
///
/// `proportional` is the name transformers gives partial rotation, and needs
/// no separate code path here: `_compute_proportional_rope_parameters` divides
/// the exponent by the full `head_dim` and zero-pads the remainder, so the
/// rotating set is the leading frequencies of an ordinary RoPE and the rest is
/// a no-op rotation. That is what `rope_rotating_pairs` and the divisor table
/// in `CpuModel::from_checkpoint` already build.
const ROPE_TYPE_DEFAULT: &str = "default";
const ROPE_TYPE_PROPORTIONAL: &str = "proportional";

/// One attention regime's RoPE parameters, as transformers 5.5 writes them.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeRegime {
    pub rope_theta: f32,
    /// Absent means the whole head rotates.
    pub partial_rotary_factor: Option<f64>,
    /// Read and checked rather than ignored: an unrecognized value is a
    /// rotation this backend does not implement, and treating it as the
    /// default would produce finite, wrong logits.
    pub rope_type: String,
}

/// The `rope_parameters` block, one entry per attention regime.
///
/// transformers 5.5 moved the RoPE fields here from the top of `text_config`,
/// which is the only reason a real Gemma 4 checkpoint failed to parse against
/// the flat spelling this module was written for.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub full_attention: RopeRegime,
    /// Absent means a single attention regime.
    pub sliding_attention: Option<RopeRegime>,
}

/// Feed-forward width, which is a scalar on a uniform model and a per-layer
/// list where the widths differ.
///
/// Gemma 4's `use_double_wide_mlp` doubles the width over the shared-KV tail,
/// so E2B needs the list form and E4B does not. Both are read into one
/// per-layer vector by `intermediate_sizes`, and no caller learns which
/// spelling the checkpoint used.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Widths {
    Uniform(usize),
    PerLayer(Vec<usize>),
}

/// The `text_config` block of a Gemma `config.json`.
///
/// Deserialized rather than key-probed: the field list is the schema, and a
/// missing required field fails with serde naming it. Optional fields are
/// `Option` where absence is a real statement about the checkpoint -- one
/// attention regime rather than two, no softcapping, no per-layer embeddings
/// -- and never where absence would mean a defaulted guess.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Head width of the sliding-window regime, and of every layer when
    /// `global_head_dim` is absent.
    pub head_dim: usize,
    /// Head width of the global regime where it differs. Gemma 4 E4B runs
    /// wider global heads than sliding ones; the GGUF spelling of this pair
    /// is `attention.key_length` and `attention.key_length_swa`.
    pub global_head_dim: Option<usize>,
    /// One entry per layer, `full_attention` or `sliding_attention`.
    ///
    /// An explicit list, so an aperiodic schedule loads. The GGUF path
    /// collapsed this to a stride and refused anything that was not periodic.
    pub layer_types: Vec<String>,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    /// The three RoPE fields in their pre-5.5 flat spelling, read through
    /// `rope_theta()`, `rope_local_base_freq()` and `partial_rotary_factor()`
    /// so no caller has to know which spelling the file used. Private for
    /// that reason: a caller reaching a raw field would see `None` on a real
    /// checkpoint and a value on a fixture.
    #[serde(rename = "rope_theta")]
    flat_rope_theta: Option<f32>,
    #[serde(rename = "rope_local_base_freq")]
    flat_rope_local_base_freq: Option<f32>,
    #[serde(rename = "partial_rotary_factor")]
    flat_partial_rotary_factor: Option<f64>,
    /// The 5.5 nested spelling, which wins where a file carries both.
    rope_parameters: Option<RopeParameters>,
    pub hidden_activation: String,
    pub tie_word_embeddings: bool,
    pub intermediate_size: Widths,
    /// Trailing layers that read a donor's K/V rather than storing their own.
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    pub final_logit_softcapping: Option<f32>,
    /// Width of one layer's slice of the per-layer input embedding. Absent
    /// means the model has no per-layer embeddings.
    pub hidden_size_per_layer_input: Option<usize>,
}

/// A Gemma `config.json`, whose architectural fields live under `text_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct CheckpointConfig {
    pub text_config: TextConfig,
}

/// The text architecture from a `config.json`, nested or flat.
///
/// Two spellings, both written by transformers. A multimodal release --
/// `Gemma4ForConditionalGeneration`, which is what gemma-4-E4B publishes as --
/// nests the text tower under `text_config` beside `vision_config` and
/// `audio_config`. A text-only release -- `Gemma4ForCausalLM`, which is what
/// `save_pretrained` writes for the Tier T fixture -- has one architecture and
/// puts its fields at the top level, tagged `model_type: gemma4_text`.
///
/// Only the nested spelling was accepted until the fixture existed, so the
/// loader could not read a checkpoint transformers itself had just written.
/// That gap was invisible because every checkpoint this workspace had was
/// either the released E4B or one `synth.rs` wrote in the released E4B's
/// image -- the same one-belief-written-twice that the fixture exists to
/// break.
///
/// The document is walked as a `Value` first, rather than the two shapes being
/// an untagged enum, so that a genuinely malformed config still fails with
/// serde naming the field it could not find. Untagged would collapse every
/// such failure into "did not match any variant", which is the property the
/// `TextConfig` doc comment above claims and this would quietly remove.
pub fn parse_text_config(bytes: &[u8]) -> Result<TextConfig> {
    let doc: serde_json::Value =
        serde_json::from_slice(bytes).context("config.json is not valid JSON")?;
    let nested = doc.get("text_config").is_some();
    let text = match doc.get("text_config") {
        Some(value) => value.clone(),
        None => doc,
    };
    serde_json::from_value(text).with_context(|| {
        if nested {
            "read the text_config block of config.json"
        } else {
            "read config.json as a text-only checkpoint (it has no text_config \
             block, so its top level is the text architecture)"
        }
    })
}

/// A block of checkpoint bytes, however it was obtained.
///
/// `read` maps the weights file; tests and in-memory callers build one from a
/// `Vec<u8>`. Both deref to `[u8]`, so nothing downstream of this type knows
/// which it holds -- `parse` and `widen` see one slice either way.
#[derive(Debug)]
pub enum Bytes {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl std::ops::Deref for Bytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Owned(v) => v,
            Bytes::Mapped(m) => m,
        }
    }
}

impl From<Vec<u8>> for Bytes {
    fn from(v: Vec<u8>) -> Bytes {
        Bytes::Owned(v)
    }
}

/// The bytes of a checkpoint.
///
/// Separate from `Checkpoint` because `safetensors::SafeTensors` borrows the
/// buffer it parses, so the owner and the view cannot be one struct. This
/// mirrors the GGUF path, where callers hold a `Vec<u8>` and pass it beside
/// the parsed header.
///
/// `config` is a plain `Vec<u8>` because it is a few kilobytes and mapping it
/// would buy nothing; only `weights` is large enough for the distinction to
/// matter.
#[derive(Debug)]
pub struct CheckpointFiles {
    pub config: Vec<u8>,
    pub weights: Bytes,
    /// Directory the two were read from, for error messages.
    pub dir: PathBuf,
}

/// A parsed checkpoint: the architecture and a view over the weight bytes.
pub struct Checkpoint<'a> {
    pub config: TextConfig,
    pub tensors: safetensors::SafeTensors<'a>,
    pub dir: PathBuf,
    /// Consume the text stack and decline the checkpoint's other towers.
    ///
    /// `parse` sets this false, so a caller that says nothing gets the
    /// refusal rather than a model quietly serving a third of the file. It is
    /// a statement about how the checkpoint is being read, not a property of
    /// the checkpoint, which is why it lives here and not in `TextConfig`.
    pub text_only: bool,
    /// The checkpoint spells its text tensors `model.` and not
    /// `model.language_model.`.
    ///
    /// Set by `parse` from the file rather than from the config, and never
    /// read outside this module's three accessors. Everything above them
    /// speaks the multimodal name -- `hf_tensor_name` mints it, the
    /// unconsumed-tensor guard compares it, the diagnostics print it -- and
    /// the file's own spelling is resolved at the boundary. One vocabulary
    /// inside the engine, two on disk.
    flat_text_names: bool,
}

impl TextConfig {
    /// Per-layer feed-forward widths, whichever spelling the checkpoint used.
    pub fn intermediate_sizes(&self) -> Result<Vec<usize>> {
        match &self.intermediate_size {
            Widths::Uniform(w) => Ok(vec![*w; self.num_hidden_layers]),
            Widths::PerLayer(ws) => {
                ensure!(
                    ws.len() == self.num_hidden_layers,
                    "intermediate_size has {} entries for {} layers",
                    ws.len(),
                    self.num_hidden_layers
                );
                Ok(ws.clone())
            }
        }
    }

    /// Head width of the global regime, defaulting to `head_dim` when the
    /// checkpoint declares one width for both.
    pub fn global_head_dim(&self) -> usize {
        self.global_head_dim.unwrap_or(self.head_dim)
    }

    /// RoPE base of the global regime.
    ///
    /// transformers 5.5 nests this under `rope_parameters.full_attention`;
    /// before that it was flat in `text_config`. Both spell the same model, so
    /// both load, and the nested one wins where a file carries both. A file
    /// carrying neither is an error rather than a default, because every base
    /// this backend could pick is a rotation the forward pass would apply
    /// without complaint.
    pub fn rope_theta(&self) -> Result<f32> {
        if let Some(rope) = &self.rope_parameters {
            return Ok(rope.full_attention.rope_theta);
        }
        self.flat_rope_theta.context(
            "config.json states no RoPE base: expected rope_parameters.full_attention.rope_theta \
             (transformers 5.5 and later) or a flat rope_theta",
        )
    }

    /// RoPE base of the sliding regime, two orders of magnitude below the
    /// global one on Gemma 4. `None` means a single regime.
    pub fn rope_local_base_freq(&self) -> Option<f32> {
        match &self.rope_parameters {
            Some(rope) => rope.sliding_attention.as_ref().map(|r| r.rope_theta),
            None => self.flat_rope_local_base_freq,
        }
    }

    /// Fraction of each global head that rotates.
    ///
    /// `None` means the whole head rotates. The GGUF carries the divisors this
    /// implies as a tensor; here the factor itself is stated and the divisors
    /// are derived at load, which is what transformers does.
    pub fn partial_rotary_factor(&self) -> Option<f64> {
        match &self.rope_parameters {
            Some(rope) => rope.full_attention.partial_rotary_factor,
            None => self.flat_partial_rotary_factor,
        }
    }

    /// Overwrite the global RoPE base in whichever spelling this config
    /// carries.
    ///
    /// Test-only, and spelling-agnostic on purpose: a setter that wrote the
    /// flat field would be silently inert on a config parsed from the nested
    /// form, and every fixture now writes the nested form. A test asserting a
    /// refusal would then pass while asserting nothing.
    #[cfg(test)]
    pub fn set_rope_theta(&mut self, base: f32) {
        match &mut self.rope_parameters {
            Some(rope) => rope.full_attention.rope_theta = base,
            None => self.flat_rope_theta = Some(base),
        }
    }

    /// Overwrite the partial rotary factor. Test-only, as `set_rope_theta`.
    #[cfg(test)]
    pub fn set_partial_rotary_factor(&mut self, factor: Option<f64>) {
        match &mut self.rope_parameters {
            Some(rope) => rope.full_attention.partial_rotary_factor = factor,
            None => self.flat_partial_rotary_factor = factor,
        }
    }

    /// Whether layer `i` is a full-attention layer.
    ///
    /// Errors on an unrecognized entry rather than treating it as sliding:
    /// a new regime name is a model this backend has not implemented, and
    /// guessing would run it as something else.
    pub fn is_global(&self, i: usize) -> Result<bool> {
        let kind = self
            .layer_types
            .get(i)
            .with_context(|| format!("layer_types has no entry for layer {i}"))?;
        match kind.as_str() {
            FULL_ATTENTION => Ok(true),
            SLIDING_ATTENTION => Ok(false),
            other => bail!(
                "layer_types[{i}] is {other:?}, expected {FULL_ATTENTION:?} \
                 or {SLIDING_ATTENTION:?}"
            ),
        }
    }

    /// Rotating pairs of a global head, from `partial_rotary_factor`.
    ///
    /// The factor is a fraction of the head width; the count of rotating
    /// dimension *pairs* is what the frequency table needs. Absent means the
    /// whole head rotates.
    pub fn rope_rotating_pairs(&self) -> Result<usize> {
        let half = self.global_head_dim() / 2;
        let Some(factor) = self.partial_rotary_factor() else {
            return Ok(half);
        };
        ensure!(
            factor.is_finite() && factor > 0.0 && factor <= 1.0,
            "partial_rotary_factor is {factor}, must be in (0, 1]"
        );
        // Floor, matching upstream's `int(rope_proportion * head_dim // 2)`.
        // The two agree at 0.25 on a 512-wide head (64 pairs of 256) and
        // diverge only on a factor landing between two pair counts, where
        // matching transformers is the only defensible tie-break.
        let rotating = (factor * self.global_head_dim() as f64 / 2.0).floor() as usize;
        ensure!(
            rotating > 0 && rotating <= half,
            "partial_rotary_factor {factor} implies {rotating} rotating pairs \
             of the {half} in a head of {}",
            self.global_head_dim()
        );
        Ok(rotating)
    }

    /// Everything that must hold before any tensor is read.
    ///
    /// Each dimension below divides or strides something in the forward pass,
    /// so a zero reaches the arithmetic rather than a guard. The GGUF path
    /// learned this the hard way: zero is even, so a zero head width cleared
    /// the RoPE parity check and then aborted the process at `chunks_mut(0)`
    /// on the first token.
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("num_hidden_layers", self.num_hidden_layers),
            ("hidden_size", self.hidden_size),
            ("num_attention_heads", self.num_attention_heads),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            ensure!(value > 0, "{field} is 0, must be positive");
        }
        // A zero window means disabled windowing everywhere else in the
        // ecosystem, but yields an empty attention span here: a context-blind
        // model that still reads as fluent.
        ensure!(
            self.sliding_window > 0,
            "sliding_window is 0, must be positive; \
             a zero-token window is not disabled windowing"
        );
        ensure!(
            self.num_key_value_heads > 0
                && self
                    .num_attention_heads
                    .is_multiple_of(self.num_key_value_heads),
            "num_attention_heads {} not divisible by num_key_value_heads {}",
            self.num_attention_heads,
            self.num_key_value_heads
        );
        ensure!(
            self.layer_types.len() == self.num_hidden_layers,
            "layer_types has {} entries for {} layers",
            self.layer_types.len(),
            self.num_hidden_layers
        );
        for (field, width) in [
            ("head_dim", self.head_dim),
            ("global_head_dim", self.global_head_dim()),
        ] {
            ensure!(width > 0, "{field} is 0, must be positive");
            ensure!(
                width.is_multiple_of(2),
                "{field} is {width}, must be even for RoPE"
            );
        }
        // Every regime the file declares, before either base is read: an
        // unrecognized `rope_type` is a rotation this backend does not
        // implement, and running it as `default` yields finite, wrong logits.
        if let Some(rope) = &self.rope_parameters {
            for (regime, spec) in [
                (FULL_ATTENTION, Some(&rope.full_attention)),
                (SLIDING_ATTENTION, rope.sliding_attention.as_ref()),
            ] {
                let Some(spec) = spec else { continue };
                ensure!(
                    spec.rope_type == ROPE_TYPE_DEFAULT || spec.rope_type == ROPE_TYPE_PROPORTIONAL,
                    "rope_parameters.{regime}.rope_type is {:?}, this backend implements \
                     only {ROPE_TYPE_DEFAULT:?} and {ROPE_TYPE_PROPORTIONAL:?}",
                    spec.rope_type
                );
            }
        }
        for (field, base) in [
            ("rope_theta", Some(self.rope_theta()?)),
            ("rope_local_base_freq", self.rope_local_base_freq()),
        ] {
            if let Some(b) = base {
                ensure!(
                    b.is_finite() && b > 0.0,
                    "{field} is {b}, must be finite and positive"
                );
            }
        }
        ensure!(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps >= 0.0,
            "rms_norm_eps is {}, must be finite and non-negative",
            self.rms_norm_eps
        );
        // A zero or negative cap divides the logits by zero or reflects them
        // through the origin. Neither is a cap, and both stay finite enough
        // to read as a working model.
        if let Some(cap) = self.final_logit_softcapping {
            ensure!(
                cap.is_finite() && cap > 0.0,
                "final_logit_softcapping is {cap}, must be positive"
            );
        }
        // Zero would make the per-layer table one row of nothing per layer
        // and divide by zero in its RMSNorm.
        if let Some(pd) = self.hidden_size_per_layer_input {
            ensure!(pd > 0, "hidden_size_per_layer_input is 0, must be positive");
        }
        ensure!(
            self.num_kv_shared_layers <= self.num_hidden_layers,
            "num_kv_shared_layers is {}, more than the {} layers \
             num_hidden_layers declares",
            self.num_kv_shared_layers,
            self.num_hidden_layers
        );
        for w in self.intermediate_sizes()? {
            ensure!(w > 0, "intermediate_size contains 0, must be positive");
        }
        // Refusals for features the backend does not implement. Both would
        // otherwise load and produce plausible numbers that are not this
        // model's.
        ensure!(
            self.hidden_activation == SUPPORTED_ACTIVATION,
            "hidden_activation is {:?}, this backend implements only \
             {SUPPORTED_ACTIVATION:?}",
            self.hidden_activation
        );
        ensure!(
            self.tie_word_embeddings,
            "tie_word_embeddings is false, so the checkpoint carries a separate \
             LM head; this backend reads logits from the tied embedding table"
        );
        // Checked last, so the more specific errors above are reported first.
        self.rope_rotating_pairs()?;
        Ok(())
    }
}

/// Hugging Face name for a llama.cpp tensor name, or `None` when the tensor is
/// a converter artifact with no published counterpart.
///
/// One table, shared by the loader that inverts it and the fixture writer
/// that applies it, because two copies of a naming map drift.
///
/// Every entry is now verified against the key list of a released checkpoint
/// -- `google/gemma-4-E4B`, 2130 tensors, read 2026-08-10 -- rather than
/// reconstructed from the llama.cpp converter's renaming, which is how the
/// three per-layer-embedding names were derived before. Two of those three
/// were right (`per_layer_input_gate`, `per_layer_projection`); the third
/// published as `post_per_layer_input_norm`, not `..._layernorm`. Do not
/// re-derive any of this from the converter: where the two disagree, the
/// checkpoint is the authority, because it is what gets read.
pub fn hf_tensor_name(gguf_name: &str) -> Option<String> {
    // Not in the checkpoint: llama.cpp precomputes the proportional-RoPE
    // divisors at conversion time, while transformers derives them from
    // partial_rotary_factor at load time.
    if gguf_name == "rope_freqs.weight" {
        return None;
    }
    if gguf_name == "token_embd.weight" {
        return Some(format!("{TEXT_PREFIX}embed_tokens.weight"));
    }
    if gguf_name == "output_norm.weight" {
        return Some(format!("{TEXT_PREFIX}norm.weight"));
    }
    if let Some(rest) = gguf_name.strip_prefix("per_layer_") {
        let tail = match rest {
            "token_embd.weight" => "embed_tokens_per_layer.weight",
            "model_proj.weight" => "per_layer_model_projection.weight",
            "proj_norm.weight" => "per_layer_projection_norm.weight",
            other => other,
        };
        return Some(format!("{TEXT_PREFIX}{tail}"));
    }
    let rest = gguf_name.strip_prefix("blk.")?;
    let (index, suffix) = rest.split_once('.')?;
    let tail = match suffix {
        "attn_norm.weight" => "input_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
        // The one published name without a `.weight` suffix: upstream declares
        // it `nn.Buffer(torch.ones(1))` and multiplies it into the layer
        // output, so it is a buffer rather than a parameter. The internal
        // GGUF-side key keeps the suffix, because that vocabulary is uniform.
        //
        // Initialized to ones and trained away from them: the released
        // checkpoints carry 42 distinct per-layer values between 0.049 and
        // 0.879 (`docs/models/gemma4-safetensors.txt`). A loader treating this
        // as an inert buffer would scale every layer output wrong.
        "layer_output_scale.weight" => "layer_scalar",
        // The converter drops the `per_layer` substring from these three.
        "inp_gate.weight" => "per_layer_input_gate.weight",
        "proj.weight" => "per_layer_projection.weight",
        "post_norm.weight" => "post_per_layer_input_norm.weight",
        _ => return None,
    };
    Some(format!("{TEXT_PREFIX}layers.{index}.{tail}"))
}

/// Locate a checkpoint from a `--model` argument.
///
/// Accepts the directory itself or a path to the weights file inside it, so a
/// caller that names either gets the same checkpoint. Follows
/// `tokenizer::resolve_tokenizer`: probe candidates, then fail naming every
/// path that was looked at.
pub fn resolve(model: &Path) -> Result<PathBuf> {
    if model.is_dir() {
        return Ok(model.to_path_buf());
    }
    if model.is_file() {
        return model
            .parent()
            .map(Path::to_path_buf)
            .with_context(|| format!("{} has no parent directory", model.display()));
    }
    bail!(
        "no checkpoint at {}: expected a directory holding {CONFIG_FILE} and \
         {WEIGHTS_FILE}, or a path to the weights file itself",
        model.display()
    )
}

impl CheckpointFiles {
    /// Read `config.json` and map `model.safetensors` from a checkpoint.
    ///
    /// The weights are mapped rather than read because `from_checkpoint`
    /// widens every tensor into an owned `Vec<f32>` anyway: an `fs::read`
    /// would hold a private copy of the whole file alongside that expansion,
    /// which on a 15 GB E4B checkpoint is 15 GB of anonymous memory that is
    /// never read twice. A map costs page cache instead, which the kernel can
    /// evict.
    ///
    /// What this does *not* do is make loading lazy. `widen` copies out
    /// regardless, so the f32 expansion is unchanged and every page a tensor
    /// touches is faulted in. The tools that only read the header --
    /// `checkpoint-info` -- are the ones that get the whole benefit.
    pub fn read(model: &Path) -> Result<CheckpointFiles> {
        let dir = resolve(model)?;
        let config_path = dir.join(CONFIG_FILE);
        let weights_path = dir.join(WEIGHTS_FILE);
        for path in [&config_path, &weights_path] {
            ensure!(
                path.is_file(),
                "checkpoint {} is missing {}",
                dir.display(),
                path.file_name().unwrap_or_default().to_string_lossy()
            );
        }
        let config = std::fs::read(&config_path)
            .with_context(|| format!("read {}", config_path.display()))?;
        let file = std::fs::File::open(&weights_path)
            .with_context(|| format!("open {}", weights_path.display()))?;
        // SAFETY: the obligation `Mmap::map` carries is not pointer validity
        // -- it hands back a slice the borrow checker tracks -- but external
        // mutation. If another process truncates or rewrites this file while
        // the map is live, reads through the slice are undefined behaviour,
        // and no Rust-level guard can prevent that.
        //
        // The assumption being made, rather than proved, is that a checkpoint
        // is a published read-only artifact: nothing in this repository writes
        // under a model directory while a harness runs, and the training
        // export pipeline writes a new directory rather than editing one in
        // place. That assumption is the price of not copying 15 GB, and it is
        // stated here so a future writer of an in-place model updater knows
        // what it would break.
        #[allow(unsafe_code)]
        let weights = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("map {}", weights_path.display()))?;
        Ok(CheckpointFiles {
            config,
            weights: Bytes::Mapped(weights),
            dir,
        })
    }
}

/// Parse a checkpoint's architecture and index its tensors.
pub fn parse(files: &CheckpointFiles) -> Result<Checkpoint<'_>> {
    let config = parse_text_config(&files.config)
        .with_context(|| format!("parse {}", files.dir.join(CONFIG_FILE).display()))?;
    config.validate()?;
    let tensors = safetensors::SafeTensors::deserialize(&files.weights)
        .map_err(|e| anyhow::anyhow!("parse {}: {e}", files.dir.join(WEIGHTS_FILE).display()))?;
    // Which spelling this file uses, decided by looking. A text-only release
    // -- `Gemma4ForCausalLM`, what save_pretrained writes -- has one tower and
    // puts it at `model.`; the released multimodal E4B nests it under
    // `model.language_model.` beside the vision and audio towers. Detected
    // from the presence of the deeper prefix rather than from the config's
    // `architectures`, because the tensors are what actually get read and a
    // config can disagree with them.
    let flat_text_names = !tensors.names().iter().any(|n| n.starts_with(TEXT_PREFIX));

    Ok(Checkpoint {
        config,
        tensors,
        dir: files.dir.clone(),
        text_only: false,
        flat_text_names,
    })
}

impl Checkpoint<'_> {
    /// Widen a tensor to f32, checking its shape.
    ///
    /// Shape and not element count: the two differ exactly where a transposed
    /// matrix would slip through, which is the defect this format change is
    /// most exposed to. `[out, in]` here is row-major with the input index
    /// contiguous, the layout `matvec_into` already assumes -- so a loader
    /// that transposes is wrong, and this is the check that says so.
    /// The canonical name as this file spells it.
    ///
    /// The only place the two spellings meet. Errors and diagnostics keep
    /// reporting the canonical name the caller asked for, because that is the
    /// name the rest of the codebase, the tests and the docs all use; the
    /// on-disk variant would be a second vocabulary for a reader to learn.
    fn on_disk<'n>(&self, name: &'n str) -> std::borrow::Cow<'n, str> {
        match name.strip_prefix(TEXT_PREFIX) {
            Some(tail) if self.flat_text_names => std::borrow::Cow::Owned(format!("model.{tail}")),
            _ => std::borrow::Cow::Borrowed(name),
        }
    }

    pub fn tensor(&self, name: &str, want: &[usize]) -> Result<Vec<f32>> {
        let view = self
            .tensors
            .tensor(&self.on_disk(name))
            .map_err(|e| anyhow::anyhow!("missing tensor {name}: {e}"))?;
        ensure!(
            view.shape() == want,
            "tensor {name} has shape {:?}, the configuration implies {want:?}",
            view.shape()
        );
        widen(name, view.dtype(), view.data())
    }

    /// A tensor's declared shape, without widening it.
    ///
    /// For the one axis a config does not state: the per-layer embedding
    /// table's own vocabulary, which is read from the table rather than
    /// assumed equal to `vocab_size`. An accessor rather than a reach into
    /// `tensors`, because a caller that indexes the raw map bypasses the
    /// name resolution and fails on exactly the checkpoints that need it.
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        self.tensors
            .tensor(&self.on_disk(name))
            .map(|v| v.shape().to_vec())
            .map_err(|e| anyhow::anyhow!("missing tensor {name}: {e}"))
    }

    /// Whether the checkpoint carries a tensor, for optional families.
    pub fn has(&self, name: &str) -> bool {
        self.tensors.tensor(&self.on_disk(name)).is_ok()
    }

    /// Every tensor name, sorted, for the unconsumed-tensor guard.
    ///
    /// Reported in the canonical spelling, so the guard's set arithmetic is
    /// against the same names `hf_tensor_name` produces. A flat checkpoint
    /// that returned its own spelling here would look like a file of entirely
    /// unknown tensors.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .tensors
            .names()
            .into_iter()
            .map(|n| match n.strip_prefix("model.") {
                Some(tail) if self.flat_text_names => format!("{TEXT_PREFIX}{tail}"),
                _ => n.to_string(),
            })
            .collect();
        names.sort();
        names
    }
}

/// Widen a safetensors payload to f32.
///
/// Only the three float widths a 16-bit-precision harness loads. A quantized
/// checkpoint is refused by name rather than approximated, because this
/// workspace computes at bf16 or wider on purpose: the numerics have to line
/// up with what the interpretability pipelines record.
fn widen(name: &str, dtype: safetensors::Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    use safetensors::Dtype;

    let width = match dtype {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        other => bail!(
            "tensor {name} has dtype {other:?}; this backend loads F32, F16 and \
             BF16 only, and computes at 16-bit precision or wider by design"
        ),
    };
    ensure!(
        bytes.len().is_multiple_of(width),
        "tensor {name}: {} bytes is not a whole number of {dtype:?} elements",
        bytes.len()
    );
    let values = match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        // Unreachable: `width` above already refused every other dtype.
        other => bail!("tensor {name}: unexpected dtype {other:?}"),
    };
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{build_tiny_config_json, build_tiny_safetensors, TinySpec};

    fn config_of(spec: &TinySpec) -> TextConfig {
        let bytes = build_tiny_config_json(spec).unwrap();
        parse_text_config(&bytes).unwrap()
    }

    /// The flat spelling, read off the artifact transformers actually wrote.
    ///
    /// Hand-building a flat document here would test this reader against this
    /// reader's idea of what `save_pretrained` emits, which is the mistake the
    /// Tier T fixture exists to stop making. The committed bytes are the
    /// claim; the manifest test next door is what keeps them honest.
    #[test]
    fn a_text_only_checkpoint_reads_without_a_text_config_block() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../fixtures/tiny-hf-gemma4/config.json");
        let Ok(bytes) = std::fs::read(&path) else {
            crate::testlog::skip(
                "flat config parse",
                "fixtures/tiny-hf-gemma4 has not been generated",
            );
            return;
        };
        let raw: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            raw.get("text_config").is_none(),
            "the fixture grew a text_config block, so it no longer covers the \
             flat spelling this test is here for"
        );

        let config = parse_text_config(&bytes).expect("read the flat config");
        config.validate().expect("the flat config is a valid model");
        assert_eq!(config.num_hidden_layers, 4);
        assert_eq!(config.hidden_size, 32);
        assert_eq!(config.hidden_size_per_layer_input, Some(4));
        // Scalar, and it broadcasts. The released checkpoint carries a scalar
        // too; `PerLayer` is reachable only from a config synth.rs writes.
        assert_eq!(config.intermediate_sizes().unwrap(), vec![128; 4]);
    }

    /// A malformed config still names the field it is missing.
    ///
    /// The reason `parse_text_config` walks a `Value` instead of being an
    /// untagged enum over the two spellings. Untagged reports "did not match
    /// any variant" and names nothing, which would turn every real config
    /// mistake into a hunt.
    #[test]
    fn a_missing_field_is_reported_by_name_in_either_spelling() {
        let spec = TinySpec::e4b_shaped();
        let bytes = build_tiny_config_json(&spec).unwrap();
        let mut doc: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        doc["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("num_hidden_layers");

        let nested = doc.to_string();
        let flat = doc["text_config"].to_string();
        for (spelling, text) in [("nested", nested), ("flat", flat)] {
            // `{:#}` and not `{}`: anyhow's plain Display prints only the
            // outermost context, and the field name serde reports is the
            // source beneath it. The alternate form is the whole chain, which
            // is what a caller surfaces -- `main` returning Err prints the
            // chain too -- so this is the message a human actually reads.
            let err = format!(
                "{:#}",
                parse_text_config(text.as_bytes())
                    .expect_err("a config with no num_hidden_layers is not a model")
            );
            assert!(
                err.contains("num_hidden_layers"),
                "{spelling} spelling lost the field name: {err}"
            );
        }
    }

    /// The writer and the reader agree, field for field. `synth` builds this
    /// config from a `TinySpec`, so a round trip through the real serializer
    /// is what pins the two halves of the mapping together.
    #[test]
    fn the_emitted_config_reads_back_as_the_spec_that_wrote_it() {
        let spec = TinySpec::e4b_shaped();
        let c = config_of(&spec);
        assert_eq!(c.num_hidden_layers, spec.n_layers);
        assert_eq!(c.hidden_size, spec.embed_dim);
        assert_eq!(c.vocab_size, spec.vocab_size);
        assert_eq!(c.num_attention_heads, spec.n_heads);
        assert_eq!(c.num_key_value_heads, spec.n_kv_heads);
        assert_eq!(c.head_dim, spec.head_dim);
        assert_eq!(c.global_head_dim(), spec.head_dim_global.unwrap());
        assert_eq!(c.sliding_window, spec.sliding_window);
        assert_eq!(c.num_kv_shared_layers, spec.shared_kv_layers);
        assert_eq!(c.rope_theta().unwrap(), spec.rope_freq_base);
        assert_eq!(c.rope_local_base_freq(), spec.rope_freq_base_swa);
        assert_eq!(c.final_logit_softcapping, spec.logit_softcap);
        assert_eq!(c.hidden_size_per_layer_input, spec.ple_dim);
        c.validate().unwrap();

        // The schedule is read from the list, not recomputed from a stride.
        for i in 0..spec.n_layers {
            let want = (i + 1).is_multiple_of(spec.sliding_window_pattern);
            assert_eq!(c.is_global(i).unwrap(), want, "layer {i}");
        }
        // The double-wide tail arrives as a list and reads back per layer.
        let widths = c.intermediate_sizes().unwrap();
        let boundary = spec.n_layers - spec.shared_kv_layers;
        for (i, w) in widths.iter().enumerate() {
            let want = if spec.double_wide_mlp && i >= boundary {
                spec.ffn_dim * 2
            } else {
                spec.ffn_dim
            };
            assert_eq!(*w, want, "layer {i} width");
        }
    }

    /// A uniform scalar width is accepted alongside the per-layer list, since
    /// E4B writes the scalar and E2B the list.
    #[test]
    fn a_scalar_intermediate_size_broadcasts_over_the_layers() {
        let mut c = config_of(&TinySpec::default());
        c.intermediate_size = Widths::Uniform(64);
        assert_eq!(
            c.intermediate_sizes().unwrap(),
            vec![64; c.num_hidden_layers]
        );
    }

    /// An aperiodic schedule loads. The GGUF path could not express one: it
    /// inferred a stride from the first global layer and refused any layer
    /// that contradicted it, so a real Gemma 4 with an irregular schedule
    /// would have been rejected as malformed.
    #[test]
    fn an_aperiodic_attention_schedule_is_accepted() {
        let mut c = config_of(&TinySpec::e4b_shaped());
        c.layer_types = vec![
            SLIDING_ATTENTION.to_string(),
            FULL_ATTENTION.to_string(),
            FULL_ATTENTION.to_string(),
            SLIDING_ATTENTION.to_string(),
        ];
        c.validate().unwrap();
        let got: Vec<bool> = (0..4).map(|i| c.is_global(i).unwrap()).collect();
        assert_eq!(got, vec![false, true, true, false]);
    }

    /// An unknown regime name is refused rather than read as sliding.
    #[test]
    fn an_unrecognized_layer_type_is_refused_naming_it() {
        let mut c = config_of(&TinySpec::default());
        c.layer_types[0] = "chunked_attention".to_string();
        let err = c.is_global(0).unwrap_err().to_string();
        assert!(err.contains("chunked_attention"), "{err}");
    }

    /// `partial_rotary_factor` reproduces the pair count the GGUF carried as a
    /// divisor tensor. E4B rotates half of each 256-wide head.
    #[test]
    fn the_rotary_factor_yields_the_rotating_pair_count() {
        let mut c = config_of(&TinySpec::default());
        c.head_dim = 256;
        c.global_head_dim = Some(256);
        c.set_partial_rotary_factor(None);
        assert_eq!(c.rope_rotating_pairs().unwrap(), 128, "absent means whole");
        c.set_partial_rotary_factor(Some(0.5));
        assert_eq!(c.rope_rotating_pairs().unwrap(), 64);
        c.set_partial_rotary_factor(Some(0.25));
        assert_eq!(c.rope_rotating_pairs().unwrap(), 32);
    }

    /// Out-of-range factors are refused naming the value, since each one
    /// yields a rotation the forward pass would apply without complaint.
    #[test]
    fn an_out_of_range_rotary_factor_is_refused() {
        let mut c = config_of(&TinySpec::default());
        for bad in [0.0, -0.5, 1.5, f64::NAN] {
            c.set_partial_rotary_factor(Some(bad));
            let err = c.rope_rotating_pairs().unwrap_err().to_string();
            assert!(
                err.contains("partial_rotary_factor"),
                "factor {bad} gave {err}"
            );
        }
    }

    /// Each refusal names the field, because the field is what a reader has
    /// to go and fix.
    #[test]
    fn validation_refuses_the_values_that_would_reach_the_arithmetic() {
        /// One mutation that should make `validate` name `field`.
        type Case = (&'static str, Box<dyn Fn(&mut TextConfig)>);

        let base = config_of(&TinySpec::e4b_shaped());
        let cases: Vec<Case> = vec![
            (
                "hidden_size",
                Box::new(|c: &mut TextConfig| c.hidden_size = 0),
            ),
            (
                "sliding_window",
                Box::new(|c: &mut TextConfig| c.sliding_window = 0),
            ),
            ("head_dim", Box::new(|c: &mut TextConfig| c.head_dim = 7)),
            (
                "num_key_value_heads",
                Box::new(|c: &mut TextConfig| c.num_key_value_heads = 3),
            ),
            (
                "layer_types",
                Box::new(|c: &mut TextConfig| c.layer_types.pop().map(|_| ()).unwrap()),
            ),
            (
                "num_kv_shared_layers",
                Box::new(|c: &mut TextConfig| c.num_kv_shared_layers = 99),
            ),
            (
                "final_logit_softcapping",
                Box::new(|c: &mut TextConfig| c.final_logit_softcapping = Some(0.0)),
            ),
            (
                "hidden_size_per_layer_input",
                Box::new(|c: &mut TextConfig| c.hidden_size_per_layer_input = Some(0)),
            ),
            (
                "hidden_activation",
                Box::new(|c: &mut TextConfig| c.hidden_activation = "silu".to_string()),
            ),
            (
                "tie_word_embeddings",
                Box::new(|c: &mut TextConfig| c.tie_word_embeddings = false),
            ),
            (
                "rope_theta",
                Box::new(|c: &mut TextConfig| c.set_rope_theta(0.0)),
            ),
        ];
        for (field, mutate) in cases {
            let mut c = base.clone();
            mutate(&mut c);
            let err = c.validate().unwrap_err().to_string();
            assert!(err.contains(field), "mutating {field} gave {err:?}");
        }
        base.validate().unwrap();
    }

    /// A missing required field fails at deserialization, naming it, rather
    /// than defaulting to something the forward pass would run with.
    #[test]
    fn a_missing_required_field_is_refused_naming_it() {
        let bytes = build_tiny_config_json(&TinySpec::default()).unwrap();
        let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        v["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("num_attention_heads");
        let err = serde_json::from_value::<CheckpointConfig>(v)
            .unwrap_err()
            .to_string();
        assert!(err.contains("num_attention_heads"), "{err}");
    }

    /// The reader widens the three float dtypes and refuses the rest by name.
    #[test]
    fn tensor_reads_check_shape_and_widen_the_float_dtypes() {
        let spec = TinySpec::default();
        let weights = build_tiny_safetensors(&spec).unwrap();
        let files = CheckpointFiles {
            config: build_tiny_config_json(&spec).unwrap(),
            weights: weights.into(),
            dir: PathBuf::from("<memory>"),
        };
        let ck = parse(&files).unwrap();

        // Named through the shared table, never spelled literally: a test
        // hard-coding a published name is a second copy of the map.
        let embed_name = hf_tensor_name("token_embd.weight").unwrap();
        // Row-major [out, in]: the embedding table is [vocab, hidden].
        let embed = ck
            .tensor(&embed_name, &[spec.vocab_size, spec.embed_dim])
            .unwrap();
        assert_eq!(embed.len(), spec.vocab_size * spec.embed_dim);

        // A transposed request is refused, which element counts could not do.
        let err = ck
            .tensor(&embed_name, &[spec.embed_dim, spec.vocab_size])
            .unwrap_err()
            .to_string();
        assert!(err.contains("shape"), "{err}");

        let err = ck
            .tensor("model.nonexistent.weight", &[1])
            .unwrap_err()
            .to_string();
        assert!(err.contains("model.nonexistent.weight"), "{err}");

        assert!(ck.has(&hf_tensor_name("output_norm.weight").unwrap()));
        assert!(!ck.has("model.absent.weight"));
        assert!(
            ck.names().windows(2).all(|w| w[0] <= w[1]),
            "names unsorted"
        );
    }

    /// bf16 weights widen through the same shift the GGUF path uses, so the
    /// two containers agree on a non-norm tensor bit for bit.
    #[test]
    fn a_bf16_checkpoint_widens_to_the_same_values_as_the_gguf_path() {
        let spec = TinySpec {
            storage: crate::synth::TensorStorage::Bf16,
            ..TinySpec::default()
        };
        let files = CheckpointFiles {
            config: build_tiny_config_json(&spec).unwrap(),
            weights: build_tiny_safetensors(&spec).unwrap().into(),
            dir: PathBuf::from("<memory>"),
        };
        let ck = parse(&files).unwrap();
        let got = ck
            .tensor(
                &hf_tensor_name("token_embd.weight").unwrap(),
                &[spec.vocab_size, spec.embed_dim],
            )
            .unwrap();
        let want: Vec<f32> = crate::synth::tiny_tensors(&spec)
            .into_iter()
            .find(|t| t.gguf_name == "token_embd.weight")
            .unwrap()
            .values
            .iter()
            .map(|v| f32::from_bits(v.to_bits() & 0xffff_0000))
            .collect();
        assert_eq!(got, want);
    }

    /// A checkpoint is locatable from the directory or from the weights file
    /// inside it, and a path that is neither fails naming what was expected.
    #[test]
    fn a_checkpoint_resolves_from_either_the_directory_or_the_weights_file() {
        let dir = tempfile::tempdir().unwrap();
        let spec = TinySpec::default();
        std::fs::write(
            dir.path().join(CONFIG_FILE),
            build_tiny_config_json(&spec).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join(WEIGHTS_FILE),
            build_tiny_safetensors(&spec).unwrap(),
        )
        .unwrap();

        for model in [dir.path().to_path_buf(), dir.path().join(WEIGHTS_FILE)] {
            let files = CheckpointFiles::read(&model).unwrap();
            let ck = parse(&files).unwrap();
            assert_eq!(ck.config.hidden_size, spec.embed_dim, "{}", model.display());
        }

        let err = CheckpointFiles::read(&dir.path().join("absent"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(CONFIG_FILE) && err.contains(WEIGHTS_FILE),
            "{err}"
        );
    }

    /// A directory missing one of the two files names the one that is absent,
    /// rather than failing later on a parse of nothing.
    #[test]
    fn a_checkpoint_missing_its_weights_names_the_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(CONFIG_FILE),
            build_tiny_config_json(&TinySpec::default()).unwrap(),
        )
        .unwrap();
        let err = CheckpointFiles::read(dir.path()).unwrap_err().to_string();
        assert!(err.contains(WEIGHTS_FILE), "{err}");
    }

    /// The mapped and the owned byte source are the same checkpoint.
    ///
    /// `read` maps the weights while every in-memory caller wraps a `Vec<u8>`,
    /// so the two `Bytes` variants are exercised by different halves of the
    /// suite and nothing else compares them. A map that handed back a
    /// truncated or misaligned view would widen to different floats here and
    /// pass everywhere else.
    #[test]
    fn a_mapped_checkpoint_and_an_owned_one_widen_to_the_same_values() {
        let dir = tempfile::tempdir().unwrap();
        let spec = TinySpec::default();
        let config = build_tiny_config_json(&spec).unwrap();
        let weights = build_tiny_safetensors(&spec).unwrap();
        std::fs::write(dir.path().join(CONFIG_FILE), &config).unwrap();
        std::fs::write(dir.path().join(WEIGHTS_FILE), &weights).unwrap();

        let mapped = CheckpointFiles::read(dir.path()).unwrap();
        assert!(
            matches!(mapped.weights, Bytes::Mapped(_)),
            "read did not map the weights"
        );
        let owned = CheckpointFiles {
            config,
            weights: weights.into(),
            dir: PathBuf::from("<memory>"),
        };

        let mapped_ck = parse(&mapped).unwrap();
        let owned_ck = parse(&owned).unwrap();
        let names = mapped_ck.names();
        assert_eq!(names, owned_ck.names());
        assert!(!names.is_empty(), "the fixture wrote no tensors");
        for name in &names {
            let want = mapped_ck.tensors.tensor(name).unwrap();
            let shape = want.shape().to_vec();
            // Exact comparison: the two paths read the same bytes through the
            // same widener, so any difference is a defect rather than drift.
            assert_eq!(
                mapped_ck.tensor(name, &shape).unwrap(),
                owned_ck.tensor(name, &shape).unwrap(),
                "{name}"
            );
        }
    }

    /// A `text_config` value with the RoPE fields removed, so a test can put
    /// back exactly the spelling it wants to exercise.
    fn text_config_without_rope() -> serde_json::Value {
        let bytes = build_tiny_config_json(&TinySpec::default()).unwrap();
        let mut doc: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let text = doc["text_config"].as_object_mut().unwrap();
        text.remove("rope_parameters");
        text.remove("rope_theta");
        text.remove("rope_local_base_freq");
        text.remove("partial_rotary_factor");
        doc["text_config"].clone()
    }

    fn config_from(text: serde_json::Value) -> Result<TextConfig> {
        let doc = serde_json::json!({ "text_config": text });
        let parsed: CheckpointConfig = serde_json::from_value(doc)?;
        Ok(parsed.text_config)
    }

    /// The flat spelling is what `text_config` carried before transformers
    /// 5.5, and checkpoints written against it must keep loading.
    #[test]
    fn the_flat_rope_spelling_still_loads() {
        let mut text = text_config_without_rope();
        let obj = text.as_object_mut().unwrap();
        obj.insert("rope_theta".into(), 1_000_000.0.into());
        obj.insert("rope_local_base_freq".into(), 10_000.0.into());
        obj.insert("partial_rotary_factor".into(), 0.25.into());
        let c = config_from(text).unwrap();
        assert_eq!(c.rope_theta().unwrap(), 1_000_000.0);
        assert_eq!(c.rope_local_base_freq(), Some(10_000.0));
        assert_eq!(c.partial_rotary_factor(), Some(0.25));
    }

    /// The nested spelling, with the values a released E4B carries. The three
    /// accessors are what let the loader read either without knowing which.
    #[test]
    fn the_nested_rope_spelling_loads_with_the_same_values() {
        let mut text = text_config_without_rope();
        text.as_object_mut().unwrap().insert(
            "rope_parameters".into(),
            serde_json::json!({
                "full_attention": {
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.25,
                    "rope_type": "proportional",
                },
                "sliding_attention": { "rope_theta": 10_000.0, "rope_type": "default" },
            }),
        );
        let c = config_from(text).unwrap();
        assert_eq!(c.rope_theta().unwrap(), 1_000_000.0);
        assert_eq!(c.rope_local_base_freq(), Some(10_000.0));
        assert_eq!(c.partial_rotary_factor(), Some(0.25));
    }

    /// A file carrying both is resolved one way, not left to field order.
    #[test]
    fn the_nested_rope_spelling_wins_over_the_flat_one() {
        let mut text = text_config_without_rope();
        let obj = text.as_object_mut().unwrap();
        obj.insert("rope_theta".into(), 111.0.into());
        obj.insert("rope_local_base_freq".into(), 222.0.into());
        obj.insert("partial_rotary_factor".into(), 0.5.into());
        obj.insert(
            "rope_parameters".into(),
            serde_json::json!({
                "full_attention": {
                    "rope_theta": 333.0,
                    "partial_rotary_factor": 0.25,
                    "rope_type": "proportional",
                },
                "sliding_attention": { "rope_theta": 444.0, "rope_type": "default" },
            }),
        );
        let c = config_from(text).unwrap();
        assert_eq!(c.rope_theta().unwrap(), 333.0);
        assert_eq!(c.rope_local_base_freq(), Some(444.0));
        assert_eq!(c.partial_rotary_factor(), Some(0.25));
    }

    /// Neither spelling is an error naming both, rather than a defaulted base
    /// the forward pass would rotate by without complaint.
    #[test]
    fn a_config_stating_no_rope_base_is_refused_naming_both_spellings() {
        let c = config_from(text_config_without_rope()).unwrap();
        let err = c.validate().unwrap_err().to_string();
        assert!(
            err.contains("rope_parameters.full_attention.rope_theta"),
            "{err}"
        );
        assert!(err.contains("rope_theta"), "{err}");
    }

    /// An unimplemented rotation is refused naming the value. Running it as
    /// the default would produce finite, wrong logits.
    #[test]
    fn an_unrecognized_rope_type_is_refused_naming_it() {
        for (regime, other) in [
            ("full_attention", "sliding_attention"),
            ("sliding_attention", "full_attention"),
        ] {
            let mut text = text_config_without_rope();
            text.as_object_mut().unwrap().insert(
                "rope_parameters".into(),
                serde_json::json!({
                    regime: { "rope_theta": 10_000.0, "rope_type": "yarn" },
                    other: { "rope_theta": 10_000.0, "rope_type": "default" },
                }),
            );
            let err = config_from(text)
                .unwrap()
                .validate()
                .unwrap_err()
                .to_string();
            assert!(err.contains("yarn"), "{regime}: {err}");
            assert!(err.contains(regime), "{regime}: {err}");
        }
    }

    /// The published names, asserted literally in the one place a literal
    /// belongs: these are facts read off `google/gemma-4-E4B`, and a test is
    /// where a fact goes. Everywhere else names come from this table.
    #[test]
    fn the_published_names_are_the_ones_a_released_checkpoint_carries() {
        for (gguf, published) in [
            (
                "token_embd.weight",
                "model.language_model.embed_tokens.weight",
            ),
            ("output_norm.weight", "model.language_model.norm.weight"),
            (
                "per_layer_token_embd.weight",
                "model.language_model.embed_tokens_per_layer.weight",
            ),
            (
                "blk.7.attn_q.weight",
                "model.language_model.layers.7.self_attn.q_proj.weight",
            ),
            (
                "blk.7.inp_gate.weight",
                "model.language_model.layers.7.per_layer_input_gate.weight",
            ),
            (
                "blk.7.proj.weight",
                "model.language_model.layers.7.per_layer_projection.weight",
            ),
            // Not `..._layernorm`, which is what the converter's renaming
            // implied and what this table said until a real key list settled it.
            (
                "blk.7.post_norm.weight",
                "model.language_model.layers.7.post_per_layer_input_norm.weight",
            ),
            // The one entry with no `.weight` suffix: a buffer, not a parameter.
            (
                "blk.7.layer_output_scale.weight",
                "model.language_model.layers.7.layer_scalar",
            ),
        ] {
            assert_eq!(hf_tensor_name(gguf).as_deref(), Some(published), "{gguf}");
        }
    }
}
