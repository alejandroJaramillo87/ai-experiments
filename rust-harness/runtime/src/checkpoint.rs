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
/// Both are read into one per-layer vector by `intermediate_sizes`, and no
/// caller learns which spelling the checkpoint used.
///
/// The list form is not how the published E-series expresses a non-uniform
/// model, and an earlier version of this comment claimed it was. E2B declares
/// the scalar 6144 with `use_double_wide_mlp`, and its real widths are 6144
/// over layers 0-14 and 12288 over 15-34; a reader that took the scalar at
/// face value refused the checkpoint at layer 15. `intermediate_sizes` now
/// applies the flag. The list form is retained because it is what `synth.rs`
/// writes and what a per-layer model could legitimately use.
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
    /// Doubles the feed-forward width over the shared-KV tail. E2B sets it,
    /// E4B does not, and a file that omits it is uniform.
    ///
    /// Read rather than inferred, because the doubling is not spelled out in
    /// the widths: E2B declares the scalar 6144 and this flag, and the loader
    /// is expected to apply the rule. `intermediate_sizes` does.
    #[serde(default)]
    pub use_double_wide_mlp: bool,
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

/// Bytes of the little-endian `u64` that prefixes a safetensors JSON header.
const LEN_PREFIX: usize = 8;

/// The JSON header of a safetensors image, deserialized.
///
/// **This reads the container envelope, not the tensor schema.** A safetensors
/// file is a little-endian `u64` byte count followed by that many bytes of
/// JSON; those two fields are what this type knows. Every field *inside* an
/// entry -- the dtype spelling, the shape, the payload offsets -- is
/// deserialized into `safetensors::tensor::TensorInfo`, so the part that could
/// drift against the format is still the upstream crate's, per the dependency
/// argument in /.styles/RUST_STYLE.md.
///
/// Hand-read rather than taken from `SafeTensors::read_metadata` because that
/// function asserts `payload_end + 8 + n == buffer.len()` and returns
/// `MetadataIncompleteBuffer` otherwise -- it cannot be given a header. Only a
/// memory map supplies a whole-file buffer cheaply, and not needing one is the
/// point: see `CheckpointFiles::read`.
/// `the_envelope_reader_agrees_with_safetensors_own_reader` is what keeps this
/// honest against a container change upstream.
///
/// `__metadata__` is named only because serde has to be told to keep the
/// framework's free-form block out of the flattened tensor map. Nothing reads
/// it.
#[derive(Deserialize)]
struct Header {
    #[serde(rename = "__metadata__")]
    _framework: Option<std::collections::HashMap<String, String>>,
    #[serde(flatten)]
    tensors: std::collections::HashMap<String, safetensors::tensor::TensorInfo>,
}

/// A safetensors header, and where the payload it describes begins.
#[derive(Debug)]
pub struct Envelope {
    tensors: std::collections::HashMap<String, safetensors::tensor::TensorInfo>,
    /// `LEN_PREFIX + json.len()`, which is what `data_offsets` count from.
    data_start: u64,
    /// The framed header bytes, kept because `fingerprint` hashes them and
    /// re-reading them from the file to do so would be the same bytes twice.
    framed: Vec<u8>,
}

impl Envelope {
    /// Parse a framed header: the length prefix followed by exactly that many
    /// bytes of JSON.
    fn parse(framed: Vec<u8>, name: &str) -> Result<Envelope> {
        let json = framed.get(LEN_PREFIX..).unwrap_or_default();
        let header: Header = serde_json::from_slice(json)
            .with_context(|| format!("parse the safetensors header of {name}"))?;
        Ok(Envelope {
            tensors: header.tensors,
            data_start: framed.len() as u64,
            framed,
        })
    }

    /// The framed header, read from the front of an in-memory image.
    fn framed_from(bytes: &[u8], name: &str) -> Result<Vec<u8>> {
        let prefix = bytes.get(..LEN_PREFIX).with_context(|| {
            format!(
                "{name} is {} bytes, too short for a safetensors length prefix",
                bytes.len()
            )
        })?;
        let len = declared_len(prefix, name)?;
        let end = LEN_PREFIX
            .checked_add(len)
            .filter(|e| *e <= bytes.len())
            .with_context(|| {
                format!(
                    "{name} declares a {len}-byte header but the image is {} bytes",
                    bytes.len()
                )
            })?;
        Ok(bytes[..end].to_vec())
    }

    /// The framed header, read from the front of a file.
    fn framed_from_file(file: &std::fs::File, name: &str) -> Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
        let mut prefix = [0u8; LEN_PREFIX];
        file.read_exact_at(&mut prefix, 0)
            .with_context(|| format!("read the safetensors length prefix of {name}"))?;
        let len = declared_len(&prefix, name)?;
        let mut framed = vec![0u8; LEN_PREFIX + len];
        framed[..LEN_PREFIX].copy_from_slice(&prefix);
        file.read_exact_at(&mut framed[LEN_PREFIX..], LEN_PREFIX as u64)
            .with_context(|| format!("read the {len}-byte safetensors header of {name}"))?;
        Ok(framed)
    }

    /// The byte range of one tensor's payload, checked against the image.
    ///
    /// Replaces `safetensors::tensor::Metadata::validate`, which this loader
    /// no longer reaches. Stronger in one respect: `image_len` comes from the
    /// filesystem rather than from the length of a slice someone already
    /// mapped, so a file that shrank between header and payload is caught
    /// here rather than being undefined behaviour through a stale mapping.
    fn payload_range(&self, name: &str, image_len: u64) -> Result<(u64, usize)> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let (start, end) = info.data_offsets;
        let len = end.checked_sub(start).with_context(|| {
            format!("tensor {name}: payload ends at {end}, before its start at {start}")
        })?;
        let at = u64::try_from(start)
            .ok()
            .and_then(|s| s.checked_add(self.data_start))
            .with_context(|| format!("tensor {name}: payload offset {start} overflows"))?;
        let stop = at
            .checked_add(len as u64)
            .filter(|s| *s <= image_len)
            .with_context(|| {
                format!(
                    "tensor {name}: payload at offset {at} ({len} bytes) exceeds an image \
                     of {image_len} bytes"
                )
            })?;
        debug_assert!(
            stop <= image_len,
            "payload_range returned a range past the image"
        );
        Ok((at, len))
    }
}

/// The header length a safetensors image declares in its first eight bytes.
fn declared_len(prefix: &[u8], name: &str) -> Result<usize> {
    let arr: [u8; LEN_PREFIX] = prefix
        .try_into()
        .map_err(|_| anyhow::anyhow!("{name}: length prefix is {} bytes", prefix.len()))?;
    usize::try_from(u64::from_le_bytes(arr))
        .map_err(|_| anyhow::anyhow!("{name} declares a header larger than this machine's usize"))
}

/// Where a checkpoint's payload bytes come from.
///
/// `read` keeps the open file and reads each tensor at its offset; `synth.rs`
/// and the tests build an image in memory. Nothing downstream distinguishes
/// them -- `payload_at` serves both and hands `widen` a plain slice either
/// way.
///
/// The file arm is why this type exists at all. Holding a `File` rather than
/// its contents is what keeps the loader's peak resident set at the size of
/// the weights it is building, instead of that plus a second copy of the
/// checkpoint it is building them from.
#[derive(Debug)]
pub enum Payload {
    Owned(Vec<u8>),
    File {
        handle: std::fs::File,
        /// Length at open. A file that shrinks afterwards fails its next read
        /// with a short-read error naming the tensor, which is the behaviour
        /// a mapping could not offer.
        len: u64,
    },
}

impl Payload {
    /// Total bytes of the image, header included.
    fn len(&self) -> u64 {
        match self {
            Payload::Owned(v) => v.len() as u64,
            Payload::File { len, .. } => *len,
        }
    }

    /// `len` bytes starting at `at`, copied out.
    fn at(&self, at: u64, len: usize, name: &str) -> Result<Vec<u8>> {
        match self {
            Payload::Owned(v) => {
                let lo = usize::try_from(at)
                    .map_err(|_| anyhow::anyhow!("tensor {name}: offset {at} overflows"))?;
                let hi = lo
                    .checked_add(len)
                    .filter(|h| *h <= v.len())
                    .with_context(|| {
                        format!("tensor {name}: payload at {lo} ({len} bytes) exceeds the image")
                    })?;
                Ok(v[lo..hi].to_vec())
            }
            Payload::File { handle, .. } => {
                use std::os::unix::fs::FileExt;
                let mut buf = vec![0u8; len];
                handle
                    .read_exact_at(&mut buf, at)
                    .with_context(|| format!("read tensor {name}: {len} bytes at offset {at}"))?;
                Ok(buf)
            }
        }
    }
}

impl From<Vec<u8>> for Payload {
    fn from(v: Vec<u8>) -> Payload {
        Payload::Owned(v)
    }
}

/// A checkpoint's config bytes, its parsed header, and its payload source.
///
/// Separate from `Checkpoint` because the two have different jobs rather than
/// different lifetimes: this owns the handle and the offsets, `Checkpoint`
/// owns the validated architecture. Since the payload is read on demand,
/// neither borrows the other's buffer and `Checkpoint` carries no lifetime.
///
/// `config` is a plain `Vec<u8>` because it is a few kilobytes read once.
#[derive(Debug)]
pub struct CheckpointFiles {
    pub config: Vec<u8>,
    envelope: Envelope,
    payload: Payload,
    /// Directory the two were read from, for error messages.
    pub dir: PathBuf,
}

impl CheckpointFiles {
    /// Build one over an image already in memory, for `synth.rs` and tests.
    pub fn in_memory(config: Vec<u8>, image: Vec<u8>, dir: PathBuf) -> Result<CheckpointFiles> {
        let name = dir.join(WEIGHTS_FILE).display().to_string();
        let framed = Envelope::framed_from(&image, &name)?;
        Ok(CheckpointFiles {
            config,
            envelope: Envelope::parse(framed, &name)?,
            payload: Payload::Owned(image),
            dir,
        })
    }
}

/// A parsed checkpoint: the architecture, and the offsets to read tensors by.
pub struct Checkpoint<'a> {
    pub config: TextConfig,
    /// Borrowed rather than cloned: the offset map for a 42-layer checkpoint
    /// is not large, but copying it per `parse` would be a copy for nothing.
    files: &'a CheckpointFiles,
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
    ///
    /// Two spellings reach the same vector. A list states every width. A
    /// scalar states one, and `use_double_wide_mlp` doubles it over the final
    /// `num_kv_shared_layers` layers -- the tail that shares a K/V cache
    /// receives a wider feed-forward in compensation. On E2B that is 6144 for
    /// layers 0-14 and 12288 for 15-34.
    ///
    /// The flag is applied only to the scalar form. A file that lists its
    /// widths has already said what each layer is, and doubling those again
    /// would mis-shape the tail in the other direction.
    pub fn intermediate_sizes(&self) -> Result<Vec<usize>> {
        match &self.intermediate_size {
            Widths::Uniform(w) => {
                let mut ws = vec![*w; self.num_hidden_layers];
                if self.use_double_wide_mlp {
                    ensure!(
                        self.num_kv_shared_layers <= self.num_hidden_layers,
                        "num_kv_shared_layers {} exceeds {} layers, so the \
                         double-wide tail has no start",
                        self.num_kv_shared_layers,
                        self.num_hidden_layers
                    );
                    let boundary = self.num_hidden_layers - self.num_kv_shared_layers;
                    for width in &mut ws[boundary..] {
                        *width *= 2;
                    }
                }
                Ok(ws)
            }
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
    /// Read `config.json` and the header of `model.safetensors`.
    ///
    /// **The payload is not read here, and not mapped.** Only the framed JSON
    /// header comes into memory; each tensor is read at its offset when a
    /// caller asks for it, straight into the buffer that will own it. That is
    /// what keeps the loader's peak resident set at the size of the weights
    /// being built rather than that plus a second copy of the file they come
    /// from -- 14 GiB rather than 28 on E4B, measured.
    ///
    /// This replaced a memory map, and the map is worth a sentence because
    /// removing it looks like a regression from a distance. The map existed to
    /// avoid `fs::read`'s private copy of the whole file, back when it also
    /// served zero-copy tensor views. Since the storage-preserving loader,
    /// every tensor is copied out regardless, so the map's only remaining
    /// effect was to make the checkpoint's pages resident against this process
    /// while the copy loop walked them. Positioned reads put those bytes in
    /// page cache the kernel owns instead, and they end the map's one
    /// unresolvable hazard: a file rewritten underneath a live mapping is
    /// undefined behaviour, while a file rewritten underneath a `pread` is an
    /// ordinary short-read error naming the tensor.
    ///
    /// The size of what gets copied is `weight`'s business, not this
    /// function's: it keeps a tensor in its published width, so a bf16
    /// checkpoint is held at 15 GB rather than expanded to 28.
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
        let handle = std::fs::File::open(&weights_path)
            .with_context(|| format!("open {}", weights_path.display()))?;
        let len = handle
            .metadata()
            .with_context(|| format!("stat {}", weights_path.display()))?
            .len();
        let name = weights_path.display().to_string();
        let framed = Envelope::framed_from_file(&handle, &name)?;
        Ok(CheckpointFiles {
            config,
            envelope: Envelope::parse(framed, &name)?,
            payload: Payload::File { handle, len },
            dir,
        })
    }
}

/// Identify a checkpoint for snapshot compatibility, without reading its weights.
///
/// Hashes the safetensors header -- every tensor's name, dtype, shape and byte
/// range -- together with `config.json` and the weights file's length. That
/// covers the whole class of change which would make a restored KV cache mean
/// something else: a different architecture, a retyped or reshaped tensor, a
/// tensor added or removed, a file of a different size.
///
/// What it deliberately does not catch is an in-place edit of weight *values*
/// leaving the layout untouched. That is a cost-of-hashing decision, and it is
/// now the only reason: hashing the payload meant reading 15 GB at every
/// startup, for a check whose own documentation calls itself corruption
/// detection rather than a cryptographic guarantee. It used to also inherit an
/// assumption from the memory map -- that a checkpoint is a published
/// read-only artifact -- but that map is gone, and a file rewritten underneath
/// this loader now produces a read error rather than undefined behaviour. The
/// blind spot is narrower than it was: a same-size, same-layout edit of values
/// between two runs that share a KV snapshot.
///
/// A file too short to carry its own header cannot reach here: `read` and
/// `in_memory` both fail first, with a message about the format.
pub fn fingerprint(files: &CheckpointFiles) -> u64 {
    // The framed header -- every tensor's name, dtype, shape and byte range --
    // was already read to build the offset map, so identifying the checkpoint
    // costs no further I/O.
    let hash = crate::session::fingerprint(&files.config);
    let hash = crate::session::fingerprint_from(hash, &files.envelope.framed);
    crate::session::fingerprint_from(hash, &files.payload.len().to_le_bytes())
}

/// Parse a checkpoint's architecture and index its tensors.
pub fn parse(files: &CheckpointFiles) -> Result<Checkpoint<'_>> {
    let config = parse_text_config(&files.config)
        .with_context(|| format!("parse {}", files.dir.join(CONFIG_FILE).display()))?;
    config.validate()?;
    // Which spelling this file uses, decided by looking. A text-only release
    // -- `Gemma4ForCausalLM`, what save_pretrained writes -- has one tower and
    // puts it at `model.`; the released multimodal E4B nests it under
    // `model.language_model.` beside the vision and audio towers. Detected
    // from the presence of the deeper prefix rather than from the config's
    // `architectures`, because the tensors are what actually get read and a
    // config can disagree with them.
    let flat_text_names = !files
        .envelope
        .tensors
        .keys()
        .any(|n| n.starts_with(TEXT_PREFIX));

    Ok(Checkpoint {
        config,
        files,
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

    /// The header entry for a tensor, by its canonical name.
    fn info(&self, name: &str) -> Result<&safetensors::tensor::TensorInfo> {
        let on_disk = self.on_disk(name);
        self.files
            .envelope
            .tensors
            .get(on_disk.as_ref())
            .with_context(|| format!("missing tensor {name}"))
    }

    /// One tensor's stored bytes, shape-checked, read at its offset.
    ///
    /// The single place a payload enters this module, so the bounds check and
    /// the shape check each exist once. `tensor` and `weight` differ only in
    /// what they do with the bytes afterwards.
    fn stored(&self, name: &str, want: &[usize]) -> Result<(safetensors::Dtype, Vec<u8>)> {
        let info = self.info(name)?;
        ensure!(
            info.shape == want,
            "tensor {name} has shape {:?}, the configuration implies {want:?}",
            info.shape
        );
        let dtype = info.dtype;
        let on_disk = self.on_disk(name).into_owned();
        let (at, len) = self
            .files
            .envelope
            .payload_range(&on_disk, self.files.payload.len())?;
        Ok((dtype, self.files.payload.at(at, len, name)?))
    }

    pub fn tensor(&self, name: &str, want: &[usize]) -> Result<Vec<f32>> {
        let (dtype, bytes) = self.stored(name, want)?;
        widen(name, dtype, &bytes)
    }

    /// Read a tensor in the width the file declares, checking its shape.
    ///
    /// The counterpart to `tensor` for the large matrices: same name
    /// resolution, same shape check, but the bytes are kept as stored instead
    /// of expanded into f32. See `tensor::Weight` for why that is a bandwidth
    /// decision and not a numerics one.
    ///
    /// `tensor` remains for the small tensors -- every norm -- and for callers
    /// that genuinely want f32, where the expansion costs nothing measurable
    /// and an f32 slice is the simpler thing to hand a caller.
    pub fn weight(&self, name: &str, want: &[usize]) -> Result<crate::tensor::Weight> {
        use crate::tensor::Weight;
        use safetensors::Dtype;

        let (dtype, bytes) = self.stored(name, want)?;
        let narrow = || -> Vec<u16> {
            bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        // The width check `widen` performs, restated here because the two
        // return different things from the same bytes and neither can call
        // the other without allocating what the caller did not ask for.
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
            "tensor {name}: {} bytes is not a whole number of {:?} elements",
            bytes.len(),
            dtype
        );
        Ok(match dtype {
            Dtype::F32 => Weight::F32(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            Dtype::F16 => Weight::F16(narrow()),
            Dtype::BF16 => Weight::Bf16(narrow()),
            // Unreachable: `width` above already refused every other dtype.
            other => bail!("tensor {name}: unexpected dtype {other:?}"),
        })
    }

    /// A tensor's declared shape, without widening it.
    ///
    /// For the one axis a config does not state: the per-layer embedding
    /// table's own vocabulary, which is read from the table rather than
    /// assumed equal to `vocab_size`. An accessor rather than a reach into
    /// `tensors`, because a caller that indexes the raw map bypasses the
    /// name resolution and fails on exactly the checkpoints that need it.
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        self.info(name).map(|i| i.shape.clone())
    }

    /// A tensor's stored dtype, without reading it.
    ///
    /// For `checkpoint-info`, which reports what a file declares rather than
    /// loading it. Beside `shape` for the same reason: the alternative is a
    /// caller reaching into the header map and losing the name resolution.
    pub fn dtype(&self, name: &str) -> Result<safetensors::Dtype> {
        self.info(name).map(|i| i.dtype)
    }

    /// Whether the checkpoint carries a tensor, for optional families.
    pub fn has(&self, name: &str) -> bool {
        self.info(name).is_ok()
    }

    /// Every tensor name, sorted, for the unconsumed-tensor guard.
    ///
    /// Reported in the canonical spelling, so the guard's set arithmetic is
    /// against the same names `hf_tensor_name` produces. A flat checkpoint
    /// that returned its own spelling here would look like a file of entirely
    /// unknown tensors.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .files
            .envelope
            .tensors
            .keys()
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

    /// A uniform scalar width is accepted alongside the per-layer list. Both
    /// published E-series models write the scalar; only E2B pairs it with
    /// `use_double_wide_mlp`.
    #[test]
    fn a_scalar_intermediate_size_broadcasts_over_the_layers() {
        let mut c = config_of(&TinySpec::default());
        c.intermediate_size = Widths::Uniform(64);
        assert_eq!(
            c.intermediate_sizes().unwrap(),
            vec![64; c.num_hidden_layers]
        );
    }

    /// The real E2B geometry, from `/mnt/ai-data/models/hf/gemma-4-E2B`'s
    /// `config.json`: 35 layers, `intermediate_size` 6144,
    /// `use_double_wide_mlp` true, `num_kv_shared_layers` 20.
    ///
    /// A regression test in the strict sense. Before the flag was read, this
    /// config produced 6144 for all 35 layers and the loader refused the real
    /// checkpoint at layer 15, whose `up_proj` is 12288 wide. The widths and
    /// the boundary here are the artifact's, not a fixture's, which is the
    /// point: the fixture and the reader were written from the same reading
    /// of the documents and agreed with each other while both diverged from
    /// the file.
    #[test]
    fn e2b_doubles_the_feed_forward_width_over_its_shared_kv_tail() {
        let mut c = config_of(&TinySpec::default());
        c.num_hidden_layers = 35;
        c.intermediate_size = Widths::Uniform(6144);
        c.use_double_wide_mlp = true;
        c.num_kv_shared_layers = 20;

        let ws = c.intermediate_sizes().unwrap();
        assert_eq!(ws.len(), 35);
        for (i, w) in ws.iter().enumerate() {
            let want = if i >= 15 { 12288 } else { 6144 };
            assert_eq!(*w, want, "layer {i} width");
        }
    }

    /// E4B's spelling: the same scalar, the flag absent. Nothing doubles,
    /// even though it also has a shared-KV tail.
    #[test]
    fn a_shared_kv_tail_alone_does_not_widen_the_feed_forward() {
        let mut c = config_of(&TinySpec::default());
        c.num_hidden_layers = 42;
        c.intermediate_size = Widths::Uniform(10240);
        c.use_double_wide_mlp = false;
        c.num_kv_shared_layers = 18;
        assert_eq!(c.intermediate_sizes().unwrap(), vec![10240; 42]);
    }

    /// A listed width is already per-layer, so the flag must not double it
    /// again. `synth.rs` writes both the list and the flag.
    #[test]
    fn the_double_wide_flag_does_not_reapply_to_listed_widths() {
        let mut c = config_of(&TinySpec::default());
        c.num_hidden_layers = 4;
        c.intermediate_size = Widths::PerLayer(vec![32, 32, 64, 64]);
        c.use_double_wide_mlp = true;
        c.num_kv_shared_layers = 2;
        assert_eq!(c.intermediate_sizes().unwrap(), vec![32, 32, 64, 64]);
    }

    /// A tail longer than the stack has no start. Caught rather than
    /// underflowing the boundary subtraction.
    #[test]
    fn a_shared_tail_longer_than_the_stack_is_an_error() {
        let mut c = config_of(&TinySpec::default());
        c.num_hidden_layers = 4;
        c.intermediate_size = Widths::Uniform(32);
        c.use_double_wide_mlp = true;
        c.num_kv_shared_layers = 5;
        let e = c.intermediate_sizes().unwrap_err().to_string();
        assert!(e.contains("num_kv_shared_layers 5"), "{e}");
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
        let files = CheckpointFiles::in_memory(
            build_tiny_config_json(&spec).unwrap(),
            weights,
            PathBuf::from("<memory>"),
        )
        .unwrap();
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
        let files = CheckpointFiles::in_memory(
            build_tiny_config_json(&spec).unwrap(),
            build_tiny_safetensors(&spec).unwrap(),
            PathBuf::from("<memory>"),
        )
        .unwrap();
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

    /// The test that licenses reading the container envelope in-tree.
    ///
    /// `Envelope` hand-reads the length prefix and the JSON frame because
    /// `SafeTensors::read_metadata` demands a whole-file buffer. That is only
    /// defensible while the two agree, so this compares them on the same image
    /// -- name for name, dtype, shape, and both payload offsets. If the
    /// container ever gains a field or moves one, the upstream reader will
    /// follow it and this will go red, which is the whole point of writing it.
    #[test]
    fn the_envelope_reader_agrees_with_safetensors_own_reader() {
        for spec in [TinySpec::default(), TinySpec::e4b_shaped()] {
            let image = build_tiny_safetensors(&spec).unwrap();
            let framed = Envelope::framed_from(&image, "<memory>").unwrap();
            let ours = Envelope::parse(framed, "<memory>").unwrap();

            let (header_len, theirs) = safetensors::SafeTensors::read_metadata(&image).unwrap();
            assert_eq!(
                ours.data_start,
                (LEN_PREFIX + header_len) as u64,
                "the payload start disagrees with the crate's header length"
            );

            let theirs = theirs.tensors();
            assert_eq!(
                ours.tensors.len(),
                theirs.len(),
                "the two readers found different numbers of tensors"
            );
            for (name, want) in &theirs {
                let got = ours
                    .tensors
                    .get(name.as_str())
                    .unwrap_or_else(|| panic!("the envelope reader missed tensor {name}"));
                assert_eq!(got.dtype, want.dtype, "{name}: dtype");
                assert_eq!(got.shape, want.shape, "{name}: shape");
                assert_eq!(got.data_offsets, want.data_offsets, "{name}: offsets");
            }
        }
    }

    /// The check that replaces `Metadata::validate` for a reader that never
    /// holds the whole image. A payload running past the file has to be named,
    /// not silently truncated or panicked on.
    #[test]
    fn a_tensor_whose_payload_exceeds_the_image_is_named_in_the_error() {
        let image = build_tiny_safetensors(&TinySpec::default()).unwrap();
        let framed = Envelope::framed_from(&image, "<memory>").unwrap();
        let envelope = Envelope::parse(framed, "<memory>").unwrap();
        let name = envelope.tensors.keys().next().unwrap().clone();

        // The real image is fine at its real length.
        envelope
            .payload_range(&name, image.len() as u64)
            .unwrap_or_else(|e| panic!("a whole image should validate: {e:#}"));

        // One byte short of the end is not, and the message says which tensor
        // and by reference to what size.
        let err = envelope
            .payload_range(&name, envelope.data_start)
            .unwrap_err()
            .to_string();
        assert!(err.contains(&name), "error does not name the tensor: {err}");
        assert!(
            err.contains(&envelope.data_start.to_string()),
            "error does not state the image size: {err}"
        );
    }

    /// A header that declares more than the file holds is an error at read
    /// time, not a panic and not a silently short map.
    #[test]
    fn a_header_longer_than_its_image_is_refused() {
        let image = build_tiny_safetensors(&TinySpec::default()).unwrap();
        let mut lying = image.clone();
        lying[..LEN_PREFIX].copy_from_slice(&(image.len() as u64 + 1).to_le_bytes());
        let err = Envelope::framed_from(&lying, "<memory>")
            .unwrap_err()
            .to_string();
        assert!(err.contains("header"), "{err}");

        // And an image too short to hold even the prefix.
        let err = Envelope::framed_from(&image[..4], "<memory>")
            .unwrap_err()
            .to_string();
        assert!(err.contains("length prefix"), "{err}");
    }

    /// An in-memory checkpoint from a spec, for the fingerprint tests.
    fn files_of(spec: &TinySpec) -> CheckpointFiles {
        CheckpointFiles::in_memory(
            build_tiny_config_json(spec).unwrap(),
            build_tiny_safetensors(spec).unwrap(),
            PathBuf::from("<memory>"),
        )
        .unwrap()
    }

    /// Storage width moves the header, so it moves the fingerprint.
    ///
    /// The strongest of the header-only cases: the two images hold the same
    /// model at the same shapes and differ only in how each tensor is typed.
    /// A KV cache computed from one is not a KV cache of the other, and the
    /// dtype is exactly the kind of change a payload hash would have caught
    /// for free and a header hash has to be shown to catch.
    #[test]
    fn a_checkpoint_fingerprint_changes_with_a_tensors_dtype() {
        let f32_spec = TinySpec::default();
        let bf16_spec = TinySpec {
            storage: crate::synth::TensorStorage::Bf16,
            ..TinySpec::default()
        };
        let (a, b) = (
            fingerprint(&files_of(&f32_spec)),
            fingerprint(&files_of(&bf16_spec)),
        );
        assert_ne!(a, b, "f32 and bf16 images share a fingerprint: {a:#x}");
    }

    /// A different architecture is a different checkpoint, weights aside.
    #[test]
    fn a_checkpoint_fingerprint_changes_with_the_config() {
        let base = files_of(&TinySpec::default());
        let mut edited = files_of(&TinySpec::default());
        edited.config.extend(b"  ");
        assert_ne!(fingerprint(&base), fingerprint(&edited));
    }

    /// The same bytes hash the same, which is the only reason it is usable.
    ///
    /// A snapshot is written by one process and read by the next, so a
    /// fingerprint that varied with anything but the file -- an address, an
    /// iteration order, a timestamp -- would reject every restore.
    #[test]
    fn a_checkpoint_fingerprint_is_stable_across_reads() {
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

        let first = fingerprint(&CheckpointFiles::read(dir.path()).unwrap());
        let second = fingerprint(&CheckpointFiles::read(dir.path()).unwrap());
        assert_eq!(first, second, "{first:#x} then {second:#x}");
        // And the mapped and owned sources agree, so a test that builds its
        // checkpoint in memory is testing the same function the harness runs.
        assert_eq!(first, fingerprint(&files_of(&spec)));
    }

    /// Editing a weight value in place does not move the fingerprint.
    ///
    /// Asserted rather than lamented. This is the cost of not reading the
    /// payload -- 15 GB at every startup, for a check that calls itself
    /// corruption detection -- and pinning it here means a future reader finds
    /// a decision instead of an oversight. It no longer rides on the memory
    /// map's read-only-artifact assumption, because there is no map: a
    /// checkpoint rewritten under a live loader now fails its next read
    /// instead of being undefined behaviour. What remains uncovered is
    /// narrower than it was -- a same-size, same-layout edit of values between
    /// two runs that share a KV snapshot.
    #[test]
    fn a_checkpoint_fingerprint_does_not_see_an_in_place_weight_edit() {
        let spec = TinySpec::default();
        let base = files_of(&spec);
        let mut bytes = build_tiny_safetensors(&spec).unwrap();
        let header_end = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert!(header_end < bytes.len(), "fixture has no payload to edit");
        bytes[header_end] ^= 0xff;
        let edited =
            CheckpointFiles::in_memory(base.config.clone(), bytes, PathBuf::from("<memory>"))
                .unwrap();
        assert_eq!(fingerprint(&base), fingerprint(&edited));
    }

    /// A file too short to carry its own frame is named, not panicked on.
    ///
    /// This used to assert that `fingerprint` returned anyway, because the
    /// hash was taken over a slice and a short slice is merely a short hash.
    /// With the header parsed at read time there is no `CheckpointFiles` to
    /// fingerprint until the frame is valid, so the refusal moves earlier and
    /// the property worth pinning becomes the message.
    #[test]
    fn a_truncated_checkpoint_is_refused_with_a_message_about_the_frame() {
        for len in [0usize, 4, 8] {
            let err = CheckpointFiles::in_memory(
                Vec::new(),
                vec![0xabu8; len],
                PathBuf::from("<memory>"),
            )
            .unwrap_err()
            .to_string();
            assert!(
                err.contains("length prefix") || err.contains("header"),
                "a {len}-byte image was refused without saying why: {err}"
            );
        }
    }

    /// The two payload sources are the same checkpoint.
    ///
    /// `read` holds an open file and reads each tensor at its offset; every
    /// in-memory caller holds a `Vec<u8>`. The two arms of `Payload` are
    /// exercised by different halves of the suite and nothing else compares
    /// them, so a positioned read that landed one tensor at the wrong offset
    /// would widen to different floats here and pass everywhere else. This is
    /// the test that would have caught the offset arithmetic being wrong.
    #[test]
    fn a_checkpoint_read_from_disk_and_one_held_in_memory_agree_tensor_for_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let spec = TinySpec::e4b_shaped();
        let config = build_tiny_config_json(&spec).unwrap();
        let weights = build_tiny_safetensors(&spec).unwrap();
        std::fs::write(dir.path().join(CONFIG_FILE), &config).unwrap();
        std::fs::write(dir.path().join(WEIGHTS_FILE), &weights).unwrap();

        let from_disk = CheckpointFiles::read(dir.path()).unwrap();
        assert!(
            matches!(from_disk.payload, Payload::File { .. }),
            "read did not keep the file open"
        );
        let in_memory =
            CheckpointFiles::in_memory(config, weights, PathBuf::from("<memory>")).unwrap();

        // The identifying hash must agree too: it is taken over the framed
        // header and the image length, and both sources have the same file.
        assert_eq!(
            fingerprint(&from_disk),
            fingerprint(&in_memory),
            "the same checkpoint fingerprinted differently by payload source"
        );

        let disk_ck = parse(&from_disk).unwrap();
        let memory_ck = parse(&in_memory).unwrap();
        let names = disk_ck.names();
        assert_eq!(names, memory_ck.names());
        assert!(!names.is_empty(), "the fixture wrote no tensors");
        for name in &names {
            let shape = disk_ck.shape(name).unwrap();
            // Exact comparison: the two paths read the same bytes through the
            // same widener, so any difference is a defect rather than drift.
            assert_eq!(
                disk_ck.tensor(name, &shape).unwrap(),
                memory_ck.tensor(name, &shape).unwrap(),
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
