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

/// Attention regime of one layer, as `config.json` spells it.
const FULL_ATTENTION: &str = "full_attention";
const SLIDING_ATTENTION: &str = "sliding_attention";

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
    /// RoPE base of the global regime.
    pub rope_theta: f32,
    /// RoPE base of the sliding regime, two orders of magnitude below the
    /// global one on Gemma 4. Absent means a single regime.
    pub rope_local_base_freq: Option<f32>,
    /// Fraction of each global head that rotates.
    ///
    /// Absent means the whole head rotates. The GGUF carries the divisors this
    /// implies as a tensor; here the factor itself is stated and the divisors
    /// are derived at load, which is what transformers does.
    pub partial_rotary_factor: Option<f64>,
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

/// The bytes of a checkpoint, owned.
///
/// Separate from `Checkpoint` because `safetensors::SafeTensors` borrows the
/// buffer it parses, so the owner and the view cannot be one struct. This
/// mirrors the GGUF path, where callers hold a `Vec<u8>` and pass it beside
/// the parsed header.
#[derive(Debug)]
pub struct CheckpointFiles {
    pub config: Vec<u8>,
    pub weights: Vec<u8>,
    /// Directory the two were read from, for error messages.
    pub dir: PathBuf,
}

/// A parsed checkpoint: the architecture and a view over the weight bytes.
pub struct Checkpoint<'a> {
    pub config: TextConfig,
    pub tensors: safetensors::SafeTensors<'a>,
    pub dir: PathBuf,
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
        let Some(factor) = self.partial_rotary_factor else {
            return Ok(half);
        };
        ensure!(
            factor.is_finite() && factor > 0.0 && factor <= 1.0,
            "partial_rotary_factor is {factor}, must be in (0, 1]"
        );
        let rotating = (factor * self.global_head_dim() as f64 / 2.0).round() as usize;
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
        for (field, base) in [
            ("rope_theta", Some(self.rope_theta)),
            ("rope_local_base_freq", self.rope_local_base_freq),
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
/// that applies it, because two copies of a naming map drift. The
/// attention and feed-forward families are the Gemma naming the transformers
/// implementation uses. The per-layer-embedding names are provisional: they are
/// reconstructed from the converter's own renaming and have not been checked
/// against a real checkpoint's key list, which is a workstation step. Nothing
/// reads the safetensors output yet, so a wrong entry here is a one-line fix
/// rather than a defect in flight.
pub fn hf_tensor_name(gguf_name: &str) -> Option<String> {
    // Not in the checkpoint: llama.cpp precomputes the proportional-RoPE
    // divisors at conversion time, while transformers derives them from
    // partial_rotary_factor at load time.
    if gguf_name == "rope_freqs.weight" {
        return None;
    }
    if gguf_name == "token_embd.weight" {
        return Some("model.embed_tokens.weight".to_string());
    }
    if gguf_name == "output_norm.weight" {
        return Some("model.norm.weight".to_string());
    }
    if let Some(rest) = gguf_name.strip_prefix("per_layer_") {
        let tail = match rest {
            "token_embd.weight" => "embed_tokens_per_layer.weight",
            "model_proj.weight" => "per_layer_model_projection.weight",
            "proj_norm.weight" => "per_layer_projection_norm.weight",
            other => other,
        };
        return Some(format!("model.{tail}"));
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
        "layer_output_scale.weight" => "layer_output_scale.weight",
        // Provisional, as above: the converter drops the `per_layer` substring
        // from these three, so the published names are reconstructed.
        "inp_gate.weight" => "per_layer_input_gate.weight",
        "proj.weight" => "per_layer_projection.weight",
        "post_norm.weight" => "post_per_layer_input_layernorm.weight",
        _ => return None,
    };
    Some(format!("model.layers.{index}.{tail}"))
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
    /// Read `config.json` and `model.safetensors` from a checkpoint.
    ///
    /// `fs::read` rather than a memory map: the `runtime` crate forbids
    /// unsafe at the crate level and `memmap2::Mmap::map` is unsafe. This
    /// matches what the GGUF path did, so it is not a regression, but a real
    /// E4B checkpoint is fully resident and then widened to f32 on top.
    /// Reducing that is its own piece of work.
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
        Ok(CheckpointFiles {
            config: std::fs::read(&config_path)
                .with_context(|| format!("read {}", config_path.display()))?,
            weights: std::fs::read(&weights_path)
                .with_context(|| format!("read {}", weights_path.display()))?,
            dir,
        })
    }
}

/// Parse a checkpoint's architecture and index its tensors.
pub fn parse(files: &CheckpointFiles) -> Result<Checkpoint<'_>> {
    let parsed: CheckpointConfig = serde_json::from_slice(&files.config)
        .with_context(|| format!("parse {}", files.dir.join(CONFIG_FILE).display()))?;
    parsed.text_config.validate()?;
    let tensors = safetensors::SafeTensors::deserialize(&files.weights)
        .map_err(|e| anyhow::anyhow!("parse {}: {e}", files.dir.join(WEIGHTS_FILE).display()))?;
    Ok(Checkpoint {
        config: parsed.text_config,
        tensors,
        dir: files.dir.clone(),
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
    pub fn tensor(&self, name: &str, want: &[usize]) -> Result<Vec<f32>> {
        let view = self
            .tensors
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("missing tensor {name}: {e}"))?;
        ensure!(
            view.shape() == want,
            "tensor {name} has shape {:?}, the configuration implies {want:?}",
            view.shape()
        );
        widen(name, view.dtype(), view.data())
    }

    /// Whether the checkpoint carries a tensor, for optional families.
    pub fn has(&self, name: &str) -> bool {
        self.tensors.tensor(name).is_ok()
    }

    /// Every tensor name, sorted, for the unconsumed-tensor guard.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.names().into_iter().map(String::from).collect();
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
        let parsed: CheckpointConfig = serde_json::from_slice(&bytes).unwrap();
        parsed.text_config
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
        assert_eq!(c.rope_theta, spec.rope_freq_base);
        assert_eq!(c.rope_local_base_freq, spec.rope_freq_base_swa);
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
        c.partial_rotary_factor = None;
        assert_eq!(c.rope_rotating_pairs().unwrap(), 128, "absent means whole");
        c.partial_rotary_factor = Some(0.5);
        assert_eq!(c.rope_rotating_pairs().unwrap(), 64);
        c.partial_rotary_factor = Some(0.25);
        assert_eq!(c.rope_rotating_pairs().unwrap(), 32);
    }

    /// Out-of-range factors are refused naming the value, since each one
    /// yields a rotation the forward pass would apply without complaint.
    #[test]
    fn an_out_of_range_rotary_factor_is_refused() {
        let mut c = config_of(&TinySpec::default());
        for bad in [0.0, -0.5, 1.5, f64::NAN] {
            c.partial_rotary_factor = Some(bad);
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
                Box::new(|c: &mut TextConfig| c.rope_theta = 0.0),
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
            weights,
            dir: PathBuf::from("<memory>"),
        };
        let ck = parse(&files).unwrap();

        // Row-major [out, in]: the embedding table is [vocab, hidden].
        let embed = ck
            .tensor(
                "model.embed_tokens.weight",
                &[spec.vocab_size, spec.embed_dim],
            )
            .unwrap();
        assert_eq!(embed.len(), spec.vocab_size * spec.embed_dim);

        // A transposed request is refused, which element counts could not do.
        let err = ck
            .tensor(
                "model.embed_tokens.weight",
                &[spec.embed_dim, spec.vocab_size],
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("shape"), "{err}");

        let err = ck
            .tensor("model.nonexistent.weight", &[1])
            .unwrap_err()
            .to_string();
        assert!(err.contains("model.nonexistent.weight"), "{err}");

        assert!(ck.has("model.norm.weight"));
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
            weights: build_tiny_safetensors(&spec).unwrap(),
            dir: PathBuf::from("<memory>"),
        };
        let ck = parse(&files).unwrap();
        let got = ck
            .tensor(
                "model.embed_tokens.weight",
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
}
