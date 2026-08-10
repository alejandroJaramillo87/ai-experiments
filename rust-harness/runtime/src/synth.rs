/*
 * synth — synthetic tiny-model GGUF builder for tests and sandbox smoke runs.
 *
 * Produces a complete, valid GGUF image of a two-layer Gemma-4-style model
 * with seeded pseudo-random weights, small enough to commit as a fixture and
 * run in a sandbox. The metadata keys, tensor names, and layouts follow the
 * llama.cpp GGUF conventions the real E4B checkpoint uses, so the CPU
 * backend exercises the identical load path. Weights are deterministic in
 * the seed: the committed Python reference recomputes logits from the same
 * file bytes.
 */

use super::tensor::{GGML_BF16, GGML_F32};

/// SentencePiece space marker, U+2581. The normalizer substitutes it for a
/// space and the decoder substitutes it back, so it appears in both halves of
/// the emitted pipeline.
const SPACE_PIECE_STR: &str = "\u{2581}";

/// Storage type for the synthetic model's weight tensors.
///
/// The committed fixture is F32. `Bf16` exists so the loader's bfloat16 path
/// is exercised end to end on a model whose exact logits are known, without
/// needing a real checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorStorage {
    F32,
    Bf16,
}

/// Dimensions of the synthetic model. Small enough that a full forward pass
/// is hand-checkable and the GGUF stays a few tens of kilobytes.
#[derive(Debug, Clone, Copy)]
pub struct TinySpec {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// Head width of the sliding-window regime, and of every layer when
    /// `head_dim_global` is `None`.
    pub head_dim: usize,
    /// Head width of the global regime, when it differs. `Some(n)` writes
    /// `attention.key_length` as `n` and `attention.key_length_swa` as
    /// `head_dim`; `None` writes only `attention.key_length`, the
    /// single-width shape the committed fixture ships.
    ///
    /// The two regimes then carry differently shaped `attn_q`, `attn_k`,
    /// `attn_v`, `attn_output`, `attn_q_norm`, and `attn_k_norm`, which is
    /// the layout E4B ships and the one a uniform fixture cannot exercise.
    pub head_dim_global: Option<usize>,
    pub ffn_dim: usize,
    /// Doubles the feed-forward width over the shared-KV tail and writes
    /// `feed_forward_length` as a per-layer int32 array rather than a scalar.
    ///
    /// This is Gemma 4's `use_double_wide_mlp`, which E2B sets and E4B does
    /// not. The rule is reproduced here because the emitter stands in for the
    /// converter; the loader reads the widths from the array and uses the
    /// rule only as a check.
    pub double_wide_mlp: bool,
    pub context_length: usize,
    pub sliding_window: usize,
    /// Every `sliding_window_pattern`-th layer is global (kv.rs convention).
    pub sliding_window_pattern: usize,
    pub rope_freq_base: f32,
    /// Frequency base for the sliding-window regime. `None` writes no
    /// `rope.freq_base_swa` key, the single-regime shape the committed
    /// fixture ships.
    pub rope_freq_base_swa: Option<f32>,
    /// Rotating pairs of a global head, when the checkpoint expresses a
    /// partial rotation. `Some(n)` writes `rope_freqs.weight` as `n` ones
    /// followed by `head_dim / 2 - n` divisors of 1e30, the construction
    /// llama.cpp's converter emits for Gemma 4's proportional RoPE. Held as
    /// a count rather than a vector so the spec stays `Copy`, and because
    /// the count is the parameter the converter itself is given.
    pub rope_rotating_pairs: Option<usize>,
    /// Ceiling written to `final_logit_softcapping`. `None` writes no key.
    pub logit_softcap: Option<f32>,
    /// Writes a `layer_output_scale.weight` of this value on every layer.
    ///
    /// Uniform rather than varied per layer so that `Some(1.0)` is a true
    /// identity and can be compared against `None` bit for bit. Which layer
    /// the scale lands on is pinned by the tap instead: RMSNorm is scale
    /// invariant, so a scale applied at the wrong layer does not reappear as
    /// a clean multiple downstream and could not be read off the logits
    /// anyway.
    pub layer_output_scale: Option<f32>,
    /// Width of one layer's slice of the per-layer input embedding. `Some(n)`
    /// writes `embedding_length_per_layer_input` and the six tensor families
    /// the feature needs; `None` writes none of them, the shape the committed
    /// fixture ships.
    ///
    /// The token-identity table is emitted with exactly `vocab_size` rows.
    /// The real checkpoints carry their own `vocab_size_per_layer_input`,
    /// which the loader reads off the tensor rather than from metadata, so a
    /// wider table is a case worth a fixture of its own rather than the
    /// default here.
    pub ple_dim: Option<usize>,
    /// Trailing layers that read a donor's K/V instead of storing their own,
    /// written as `attention.shared_kv_layers`. Zero writes no key, the shape
    /// the committed fixture ships.
    ///
    /// The key alone changes the model: every layer's `attn_k` and `attn_v`
    /// are still emitted, exactly as the released checkpoints carry them over
    /// their own shared tails. A `shared_kv_layers: 0` image and a `2` image
    /// are therefore bit-identical but for this one metadata value, which is
    /// what makes a difference in their outputs attributable to sharing alone.
    pub shared_kv_layers: usize,
    pub rms_eps: f32,
    pub seed: u64,
    /// Weight storage type. Defaults to F32, which is what the committed
    /// fixture ships; changing it changes the image bytes.
    pub storage: TensorStorage,
}

impl Default for TinySpec {
    fn default() -> Self {
        Self {
            vocab_size: 288, // 3 specials + 256 byte tokens + 29 pieces
            embed_dim: 16,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 8,
            head_dim_global: None,
            ffn_dim: 32,
            double_wide_mlp: false,
            context_length: 1024,
            sliding_window: 4,
            sliding_window_pattern: 2,
            rope_freq_base: 10000.0,
            rope_freq_base_swa: None,
            rope_rotating_pairs: None,
            logit_softcap: None,
            layer_output_scale: None,
            ple_dim: None,
            shared_kv_layers: 0,
            rms_eps: 1e-6,
            seed: 42,
            storage: TensorStorage::F32,
        }
    }
}

impl TinySpec {
    /// An E4B-shaped tiny model: every architectural feature the E-series
    /// declares, at a geometry chosen so each one is observable.
    ///
    /// The geometry is not arbitrary. Four layers on a period of two put the
    /// two-layer shared-KV tail across one sliding and one global layer,
    /// which is the only arrangement where a single donor and the per-type
    /// donor rule disagree. `head_dim` differs from `embed_dim / n_heads`
    /// (8 and 12 against 6), so the two candidate QK scales separate rather
    /// than coincide. Four query heads over two KV heads make the blocked and
    /// interleaved GQA mappings disagree. The committed uniform fixture has
    /// none of those three properties, which is QA finding A1-15: it cannot
    /// tell either convention apart.
    pub fn e4b_shaped() -> Self {
        Self {
            embed_dim: 24,
            n_layers: 4,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 8,
            head_dim_global: Some(12),
            ffn_dim: 32,
            double_wide_mlp: true,
            // Narrower than the prompt, so the sliding layers actually drop
            // positions and the two regimes attend over different spans.
            sliding_window: 3,
            sliding_window_pattern: 2,
            rope_freq_base: 1_000_000.0,
            rope_freq_base_swa: Some(10_000.0),
            // Two rotating pairs of the global head's six: the proportional
            // rotation E4B expresses through rope_freqs.weight.
            rope_rotating_pairs: Some(2),
            logit_softcap: Some(30.0),
            layer_output_scale: Some(0.75),
            ple_dim: Some(4),
            shared_kv_layers: 2,
            // Distinct from the uniform fixture's seed so the two files share
            // no weights, and a golden copied between them fails loudly.
            seed: 4242,
            ..Self::default()
        }
    }
}

/// Vocabulary of the synthetic model: three control specials, byte tokens
/// for every byte value (so arbitrary text encodes via byte fallback, chat
/// template included), then printable pieces. Token types follow the
/// llama.cpp convention (3 = control, 6 = byte, 1 = normal).
pub fn tiny_vocab(vocab_size: usize) -> (Vec<String>, Vec<f32>, Vec<i32>) {
    assert!(
        vocab_size >= 260,
        "vocab needs specials, bytes, and content"
    );
    let mut pieces = vec!["<pad>".to_string(), "<s>".to_string(), "</s>".to_string()];
    let mut kinds = vec![3, 3, 3];
    for b in 0..=255u8 {
        pieces.push(format!("<0x{b:02X}>"));
        kinds.push(6);
    }
    // Alternate word-initial and continuation pieces over the alphabet.
    //
    // The letter advances every second piece, so the two forms of one letter
    // are adjacent and every piece is distinct. Indexing by `i` instead would
    // repeat after 26 pieces -- and did: the vocabulary carried three
    // duplicates until the Hugging Face port surfaced them, because a BPE
    // vocabulary is a map and cannot hold two pieces with the same text, while
    // the unigram encoder it replaced silently kept the first occurrence.
    let content = vocab_size - 259;
    assert!(
        content <= 52,
        "vocab of {vocab_size} needs {content} content pieces, \
         more than the 52 distinct forms of the alphabet"
    );
    for i in 0..content {
        let c = (b'a' + (i / 2) as u8) as char;
        if i % 2 == 0 {
            pieces.push(format!("{SPACE_PIECE_STR}{c}"));
        } else {
            pieces.push(c.to_string());
        }
        kinds.push(1);
    }
    let scores: Vec<f32> = (0..vocab_size).map(|i| -(i as f32)).collect();
    (pieces, scores, kinds)
}

/// Build a `tokenizer.json` from raw vocabulary arrays.
///
/// The pipeline shape matches the published Gemma 4 file: a BPE model with
/// byte fallback, behind a `Replace` normalizer that marks spaces and a
/// `Split` pre-tokenizer that isolates them, with a decoder that inverts both.
/// `kinds` carries the llama.cpp token-type codes `tiny_vocab` returns; only
/// the byte code matters here, since byte fallback is a model-level flag
/// rather than a per-token one.
///
/// Deliberately emits **no added-tokens table.** The specials are ordinary
/// vocabulary entries, so `tokenizers` has no AddedVocabulary to extract from
/// input, and the only path from a special literal to its id is the caller's
/// explicit allowlist in `tokenizer::encode_prompt`. Handing that decision to
/// the library would reopen the injection the chat template's sanitizer exists
/// to prevent.
///
/// Errors rather than panics: a caller assembling a vocabulary by hand can get
/// it wrong, and the message should name what.
pub fn tokenizer_json_from_parts(
    pieces: &[String],
    merges: &[(String, String)],
    unk: &str,
) -> anyhow::Result<Vec<u8>> {
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::decoders::fuse::Fuse;
    use tokenizers::decoders::sequence::Sequence as DecoderSequence;
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::Replace;
    use tokenizers::pre_tokenizers::split::Split;
    use tokenizers::{DecoderWrapper, SplitDelimiterBehavior, Tokenizer};

    let vocab: tokenizers::models::bpe::Vocab = pieces
        .iter()
        .enumerate()
        .map(|(i, p)| (p.clone(), i as u32))
        .collect();
    anyhow::ensure!(
        vocab.len() == pieces.len(),
        "vocabulary has duplicate pieces: {} entries collapsed to {}",
        pieces.len(),
        vocab.len()
    );
    anyhow::ensure!(
        vocab.contains_key(unk),
        "unknown-token piece {unk:?} is not in the vocabulary"
    );

    let model = BPE::builder()
        .vocab_and_merges(vocab, merges.to_vec())
        .unk_token(unk.to_string())
        .byte_fallback(true)
        .build()
        .map_err(|e| anyhow::anyhow!("build BPE model: {e}"))?;

    let mut tokenizer = Tokenizer::new(model);
    let _ = tokenizer.with_normalizer(Some(
        Replace::new(" ", SPACE_PIECE_STR).map_err(|e| anyhow::anyhow!("normalizer: {e}"))?,
    ));
    let _ = tokenizer.with_pre_tokenizer(Some(
        Split::new(SPACE_PIECE_STR, SplitDelimiterBehavior::Isolated, false)
            .map_err(|e| anyhow::anyhow!("pre-tokenizer: {e}"))?,
    ));
    // Inverse of the pipeline above, in order: reassemble the raw bytes that
    // byte-fallback tokens stand for, fuse the pieces, then turn the space
    // marker back into a space. Without a decoder, `decode` returns "<0x41>"
    // where it should return "A".
    let _ = tokenizer.with_decoder(Some(DecoderWrapper::Sequence(DecoderSequence::new(vec![
        DecoderWrapper::ByteFallback(ByteFallback::new()),
        DecoderWrapper::Fuse(Fuse::new()),
        DecoderWrapper::Replace(
            Replace::new(SPACE_PIECE_STR, " ").map_err(|e| anyhow::anyhow!("decoder: {e}"))?,
        ),
    ]))));

    Ok(tokenizer
        .to_string(true)
        .map_err(|e| anyhow::anyhow!("serialize tokenizer: {e}"))?
        .into_bytes())
}

/// Deterministic weight stream: the same 64-bit LCG the sampler uses, mapped
/// to small values so activations stay well-conditioned at f32.
struct WeightStream(u64);

impl WeightStream {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9e37_79b9_7f4a_7c15))
    }

    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((self.0 >> 40) as f32) / ((1u64 << 24) as f32);
        (unit - 0.5) * 0.2 // uniform in [-0.1, 0.1)
    }

    fn tensor(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }

    /// Norm weights, centered on 1.0.
    ///
    /// A GGUF stores the effective RMSNorm multiplier, so a norm tensor
    /// drawn from the same near-zero range as the projections would scale
    /// the residual stream to nothing and make the golden insensitive to
    /// the norms. Consumes the stream at the same rate as `tensor`, so the
    /// weights that follow are unaffected.
    fn norm_tensor(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 1.0 + self.next()).collect()
    }
}

/// GGUF image writer (v3, little-endian). Public so tests and fixture
/// generation share one encoder; the gguf.rs parser is its inverse.
#[derive(Default)]
pub struct GgufWriter {
    kv: Vec<u8>,
    kv_count: u64,
    directory: Vec<u8>,
    tensor_count: u64,
    data: Vec<u8>,
    alignment: usize,
}

impl GgufWriter {
    pub fn new() -> Self {
        Self {
            alignment: 32,
            ..Self::default()
        }
    }

    fn string_into(buf: &mut Vec<u8>, s: &str) {
        buf.extend((s.len() as u64).to_le_bytes());
        buf.extend(s.as_bytes());
    }

    fn kv_header(&mut self, key: &str, value_type: u32) {
        Self::string_into(&mut self.kv, key);
        self.kv.extend(value_type.to_le_bytes());
        self.kv_count += 1;
    }

    pub fn kv_u32(&mut self, key: &str, v: u32) {
        self.kv_header(key, 4);
        self.kv.extend(v.to_le_bytes());
    }

    pub fn kv_f32(&mut self, key: &str, v: f32) {
        self.kv_header(key, 6);
        self.kv.extend(v.to_le_bytes());
    }

    pub fn kv_str(&mut self, key: &str, v: &str) {
        self.kv_header(key, 8);
        Self::string_into(&mut self.kv, v);
    }

    pub fn kv_arr_str(&mut self, key: &str, vs: &[String]) {
        self.kv_header(key, 9);
        self.kv.extend(8u32.to_le_bytes());
        self.kv.extend((vs.len() as u64).to_le_bytes());
        for v in vs {
            Self::string_into(&mut self.kv, v);
        }
    }

    pub fn kv_arr_f32(&mut self, key: &str, vs: &[f32]) {
        self.kv_header(key, 9);
        self.kv.extend(6u32.to_le_bytes());
        self.kv.extend((vs.len() as u64).to_le_bytes());
        for v in vs {
            self.kv.extend(v.to_le_bytes());
        }
    }

    pub fn kv_arr_i32(&mut self, key: &str, vs: &[i32]) {
        self.kv_header(key, 9);
        self.kv.extend(5u32.to_le_bytes());
        self.kv.extend((vs.len() as u64).to_le_bytes());
        for v in vs {
            self.kv.extend(v.to_le_bytes());
        }
    }

    /// Append an f32 tensor: directory entry plus aligned payload.
    pub fn tensor_f32(&mut self, name: &str, dims: &[u64], values: &[f32]) {
        self.tensor(name, dims, values, TensorStorage::F32);
    }

    /// Append a tensor in the given storage type.
    ///
    /// Bfloat16 values are written by truncating the low 16 mantissa bits
    /// rather than rounding. Truncation makes `bf16_to_f32` of the result
    /// exactly `f32::from_bits(v.to_bits() & 0xffff_0000)`, so a caller can
    /// state the expected f32 model without reimplementing a rounding rule.
    pub fn tensor(&mut self, name: &str, dims: &[u64], values: &[f32], storage: TensorStorage) {
        assert_eq!(
            dims.iter().product::<u64>() as usize,
            values.len(),
            "tensor {name}: dims do not match value count"
        );
        while !self.data.len().is_multiple_of(self.alignment) {
            self.data.push(0);
        }
        Self::string_into(&mut self.directory, name);
        self.directory.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            self.directory.extend(d.to_le_bytes());
        }
        let ggml_type = match storage {
            TensorStorage::F32 => GGML_F32,
            TensorStorage::Bf16 => GGML_BF16,
        };
        self.directory.extend(ggml_type.to_le_bytes());
        self.directory
            .extend((self.data.len() as u64).to_le_bytes());
        self.tensor_count += 1;
        match storage {
            TensorStorage::F32 => {
                for v in values {
                    self.data.extend(v.to_le_bytes());
                }
            }
            TensorStorage::Bf16 => {
                for v in values {
                    self.data.extend(((v.to_bits() >> 16) as u16).to_le_bytes());
                }
            }
        }
    }

    /// Assemble the final image: header, metadata, directory, aligned data.
    pub fn finish(self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(super::gguf::GGUF_MAGIC.to_le_bytes());
        out.extend(super::gguf::GGUF_VERSION.to_le_bytes());
        out.extend(self.tensor_count.to_le_bytes());
        out.extend(self.kv_count.to_le_bytes());
        out.extend(&self.kv);
        out.extend(&self.directory);
        while out.len() % self.alignment != 0 {
            out.push(0);
        }
        out.extend(&self.data);
        out
    }
}

/// One synthetic tensor, in both naming conventions.
///
/// The GGUF and safetensors emitters are two sinks over the same list rather
/// than two geometry walks, because two walks drift and the loader inherits
/// the drift. The weight stream is drawn here exactly once, in emission order,
/// so the two containers describe the same model.
///
/// They do not spell it identically. `values` is the GGUF convention — for a
/// norm tensor the effective multiplier, with Gemma's runtime `+1` already
/// folded in — and the safetensors sink unshifts those on the way out. See
/// `build_tiny_safetensors`.
pub struct SynthTensor {
    /// llama.cpp name, for example `blk.0.attn_q.weight`.
    pub gguf_name: String,
    /// Hugging Face name, for example `model.layers.0.self_attn.q_proj.weight`.
    /// `None` for a tensor that exists only as a converter artifact and has no
    /// counterpart in the published checkpoint.
    pub hf_name: Option<String>,
    /// GGUF `ne` order, fastest axis first: `[in, out]` for a weight matrix.
    ///
    /// The payload is row-major with `ne0` contiguous, so the C-order shape is
    /// this reversed — `[out, in]`, which is already the Hugging Face
    /// convention. The safetensors emitter therefore reverses the dims and
    /// writes the same bytes; there is no data transpose between the two.
    pub dims: Vec<u64>,
    /// Effective RMSNorm multiplier for a norm tensor: the GGUF convention.
    /// The safetensors sink unshifts these; see `build_tiny_safetensors`.
    pub values: Vec<f32>,
    pub storage: TensorStorage,
    /// Whether this is an RMSNorm weight, which the two containers spell
    /// differently. Set from `is_norm_tensor`, the same predicate that chose
    /// the draw, so the emitted values and the flag cannot disagree.
    pub norm: bool,
}

/// Whether a tensor name denotes an RMSNorm weight.
///
/// Every norm in the Gemma tensor set ends `norm.weight` and nothing else
/// does, so the suffix is exact rather than a heuristic. It is a function
/// rather than an inline check because two things depend on agreeing about
/// it: which stream draw a tensor takes, and whether the safetensors sink
/// unshifts it.
fn is_norm_tensor(gguf_name: &str) -> bool {
    gguf_name.ends_with("norm.weight")
}

impl SynthTensor {
    /// Row-major shape, the order safetensors and Hugging Face use.
    pub fn hf_shape(&self) -> Vec<usize> {
        self.dims.iter().rev().map(|&d| d as usize).collect()
    }
}

/// Hugging Face name for a llama.cpp tensor name, or `None` when the tensor is
/// a converter artifact with no published counterpart.
///
/// One table, because every entry the loader will invert lives here. The
/// attention and feed-forward families are the Gemma naming the transformers
/// implementation uses. The per-layer-embedding names are provisional: they are
/// reconstructed from the converter's own renaming and have not been checked
/// against a real checkpoint's key list, which is a workstation step. Nothing
/// reads the safetensors output yet, so a wrong entry here is a one-line fix
/// rather than a defect in flight.
fn hf_tensor_name(gguf_name: &str) -> Option<String> {
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

/// Every weight tensor of `spec`, in emission order.
///
/// Order is load-bearing twice over: it fixes the GGUF byte layout, and it
/// fixes the sequence the weight stream is drawn in. Appending a feature's
/// tensors at the end rather than interleaving them is what keeps a
/// feature-off and a feature-on image bit-identical everywhere else.
pub fn tiny_tensors(spec: &TinySpec) -> Vec<SynthTensor> {
    let d = spec.embed_dim as u64;
    let hd_global = spec.head_dim_global.unwrap_or(spec.head_dim);
    let is_global = |i: usize| (i + 1).is_multiple_of(spec.sliding_window_pattern);
    let head_dim_of = |i: usize| match is_global(i) {
        true => hd_global,
        false => spec.head_dim,
    };
    let boundary = spec.n_layers - spec.shared_kv_layers;
    let ffn_dim_of = |i: usize| match spec.double_wide_mlp && i >= boundary {
        true => spec.ffn_dim * 2,
        false => spec.ffn_dim,
    };

    let mut ws = WeightStream::new(spec.seed);
    let n = |dims: &[u64]| dims.iter().product::<u64>() as usize;
    let mut out: Vec<SynthTensor> = Vec::new();
    let mut push = |name: String, dims: Vec<u64>, values: Vec<f32>, storage: TensorStorage| {
        out.push(SynthTensor {
            hf_name: hf_tensor_name(&name),
            norm: is_norm_tensor(&name),
            gguf_name: name,
            dims,
            values,
            storage,
        });
    };

    let embd_dims = vec![d, spec.vocab_size as u64];
    let embd = ws.tensor(n(&embd_dims));
    push(
        "token_embd.weight".to_string(),
        embd_dims,
        embd,
        spec.storage,
    );

    for i in 0..spec.n_layers {
        let hd = head_dim_of(i) as u64;
        let q_out = spec.n_heads as u64 * hd;
        let kv_out = spec.n_kv_heads as u64 * hd;
        let ffn = ffn_dim_of(i) as u64;
        // Weight matrices: GGUF layout [ne0 = input dim, ne1 = output dim],
        // rows of ne0 contiguous per output element.
        for (suffix, dims) in [
            ("attn_norm.weight", vec![d]),
            ("attn_q.weight", vec![d, q_out]),
            ("attn_k.weight", vec![d, kv_out]),
            ("attn_v.weight", vec![d, kv_out]),
            ("attn_q_norm.weight", vec![hd]),
            ("attn_k_norm.weight", vec![hd]),
            ("attn_output.weight", vec![q_out, d]),
            ("post_attention_norm.weight", vec![d]),
            ("ffn_norm.weight", vec![d]),
            ("ffn_gate.weight", vec![d, ffn]),
            ("ffn_up.weight", vec![d, ffn]),
            ("ffn_down.weight", vec![ffn, d]),
            ("post_ffw_norm.weight", vec![d]),
        ] {
            let values = if is_norm_tensor(suffix) {
                ws.norm_tensor(n(&dims))
            } else {
                ws.tensor(n(&dims))
            };
            push(format!("blk.{i}.{suffix}"), dims, values, spec.storage);
        }
        // Written F32 whatever `spec.storage` says, matching the converter:
        // the published GGUF carries `blk.{n}.layer_output_scale.weight [1]
        // F32` even in an otherwise BF16 file.
        if let Some(scale) = spec.layer_output_scale {
            push(
                format!("blk.{i}.layer_output_scale.weight"),
                vec![1],
                vec![scale],
                TensorStorage::F32,
            );
        }
    }

    // Proportional-RoPE divisors, matching the converter's own construction:
    // ones over the rotating pairs, 1e30 over the rest. Written F32
    // regardless of `spec.storage`, as llama.cpp writes it: the padded value
    // is far outside bfloat16's range and rounds to infinity, which would
    // turn the reference's very small residual angle into an exact zero.
    // Sized against the global head width, not the sliding one: the divisors
    // apply to the global regime only, and the loader length-checks them
    // against `attention.key_length / 2`.
    if let Some(rotating) = spec.rope_rotating_pairs {
        let half = hd_global / 2;
        assert!(
            rotating <= half,
            "rope_rotating_pairs {rotating} exceeds the {half} pairs of a head"
        );
        let mut divisors = vec![1.0f32; rotating];
        divisors.extend(std::iter::repeat_n(1e30f32, half - rotating));
        push(
            "rope_freqs.weight".to_string(),
            vec![half as u64],
            divisors,
            TensorStorage::F32,
        );
    }

    let output_norm = ws.norm_tensor(spec.embed_dim);
    push(
        "output_norm.weight".to_string(),
        vec![d],
        output_norm,
        spec.storage,
    );

    // Per-layer input embeddings, drawn last so that turning the feature on
    // appends to the weight stream instead of shifting it. Every other tensor
    // is then bit-identical between a `ple_dim: Some(n)` image and a `None`
    // one, which is what makes the difference between their outputs
    // attributable to this feature alone rather than to a reshuffled stream.
    //
    // Tensor order within the file is not otherwise meaningful: the loader
    // finds tensors by name.
    if let Some(pd) = spec.ple_dim {
        let pd64 = pd as u64;
        // The token-identity table's row is the whole stack's worth,
        // `n_layers * ple_dim`, as in the published files -- one tensor for
        // the loader to split into rows, not a stack of per-layer tables.
        let packed = spec.n_layers as u64 * pd64;
        let table_dims = vec![packed, spec.vocab_size as u64];
        let table = ws.tensor(n(&table_dims));
        push(
            "per_layer_token_embd.weight".to_string(),
            table_dims,
            table,
            spec.storage,
        );
        let proj_dims = vec![d, packed];
        let model_proj = ws.tensor(n(&proj_dims));
        push(
            "per_layer_model_proj.weight".to_string(),
            proj_dims,
            model_proj,
            spec.storage,
        );
        let proj_norm = ws.norm_tensor(pd);
        push(
            "per_layer_proj_norm.weight".to_string(),
            vec![pd64],
            proj_norm,
            spec.storage,
        );
        // The three per-block families, under the names the GGUF converter
        // gives them: `per_layer_input_gate` becomes `inp_gate`,
        // `per_layer_projection` becomes `proj`, and
        // `post_per_layer_input_norm` becomes `post_norm`. None of the three
        // keeps the `per_layer` substring, which is the trap the loader's
        // explicit tensor list exists to avoid.
        for i in 0..spec.n_layers {
            for (suffix, dims) in [
                ("inp_gate.weight", vec![d, pd64]),
                ("proj.weight", vec![pd64, d]),
                ("post_norm.weight", vec![d]),
            ] {
                let values = if is_norm_tensor(suffix) {
                    ws.norm_tensor(n(&dims))
                } else {
                    ws.tensor(n(&dims))
                };
                push(format!("blk.{i}.{suffix}"), dims, values, spec.storage);
            }
        }
    }
    // LM head is tied to token_embd, llama.cpp Gemma convention: no
    // output.weight tensor.
    out
}

/// Build the complete synthetic model image for `spec`.
pub fn build_tiny_model(spec: &TinySpec) -> Vec<u8> {
    let hd_global = spec.head_dim_global.unwrap_or(spec.head_dim);
    // Double-wide applies to the shared-KV tail, the region
    // `use_double_wide_mlp` widens. With no shared tail declared there is no
    // region to widen, and the widths stay uniform.
    let boundary = spec.n_layers - spec.shared_kv_layers;
    let ffn_dim_of = |i: usize| match spec.double_wide_mlp && i >= boundary {
        true => spec.ffn_dim * 2,
        false => spec.ffn_dim,
    };
    let mut w = GgufWriter::new();

    w.kv_str("general.architecture", "gemma4");
    w.kv_u32("gemma4.block_count", spec.n_layers as u32);
    w.kv_u32("gemma4.context_length", spec.context_length as u32);
    w.kv_u32("gemma4.embedding_length", spec.embed_dim as u32);
    // Scalar when uniform, per-layer int32 array when the tail is widened --
    // the two forms the converter emits, and both read by the loader.
    match spec.double_wide_mlp {
        true => w.kv_arr_i32(
            "gemma4.feed_forward_length",
            &(0..spec.n_layers)
                .map(|i| ffn_dim_of(i) as i32)
                .collect::<Vec<_>>(),
        ),
        false => w.kv_u32("gemma4.feed_forward_length", spec.ffn_dim as u32),
    }
    w.kv_u32("gemma4.attention.head_count", spec.n_heads as u32);
    w.kv_u32("gemma4.attention.head_count_kv", spec.n_kv_heads as u32);
    w.kv_u32("gemma4.attention.key_length", hd_global as u32);
    if spec.head_dim_global.is_some() {
        w.kv_u32("gemma4.attention.key_length_swa", spec.head_dim as u32);
    }
    w.kv_u32(
        "gemma4.attention.sliding_window",
        spec.sliding_window as u32,
    );
    w.kv_u32(
        "gemma4.attention.sliding_window_pattern",
        spec.sliding_window_pattern as u32,
    );
    w.kv_f32("gemma4.attention.layer_norm_rms_epsilon", spec.rms_eps);
    w.kv_f32("gemma4.rope.freq_base", spec.rope_freq_base);
    if let Some(swa) = spec.rope_freq_base_swa {
        w.kv_f32("gemma4.rope.freq_base_swa", swa);
    }
    if let Some(cap) = spec.logit_softcap {
        w.kv_f32("gemma4.final_logit_softcapping", cap);
    }
    if let Some(pd) = spec.ple_dim {
        w.kv_u32("gemma4.embedding_length_per_layer_input", pd as u32);
    }
    if spec.shared_kv_layers > 0 {
        w.kv_u32(
            "gemma4.attention.shared_kv_layers",
            spec.shared_kv_layers as u32,
        );
    }

    let (pieces, scores, kinds) = tiny_vocab(spec.vocab_size);
    w.kv_arr_str("tokenizer.ggml.tokens", &pieces);
    w.kv_arr_f32("tokenizer.ggml.scores", &scores);
    w.kv_arr_i32("tokenizer.ggml.token_type", &kinds);
    w.kv_u32("tokenizer.ggml.bos_token_id", 1);
    w.kv_u32("tokenizer.ggml.eos_token_id", 2);

    for t in tiny_tensors(spec) {
        w.tensor(&t.gguf_name, &t.dims, &t.values, t.storage);
    }
    w.finish()
}

/// The synthetic model's `tokenizer.json`.
///
/// Built from `tiny_vocab`, whose 288 pieces are three specials, 256 byte
/// tokens and 29 single-character pieces. The merge list is therefore
/// **empty**: a merge can only produce a token the vocabulary already holds,
/// and no piece here is longer than one character. Merge-order behaviour is
/// covered by `fixtures/tiny-bpe.tokenizer.json` instead, whose vocabulary is
/// built for exactly that purpose — keeping this one aligned with the GGUF
/// fixture the numpy oracle reads, which must not change.
pub fn build_tiny_tokenizer_json(spec: &TinySpec) -> anyhow::Result<Vec<u8>> {
    let (pieces, _scores, _kinds) = tiny_vocab(spec.vocab_size);
    tokenizer_json_from_parts(&pieces, &[], "<pad>")
}

/// A tokenizer whose merge list is the whole answer.
///
/// `tiny_vocab` cannot carry merges: every content piece is one character, and
/// a merge can only produce a token the vocabulary already holds. So
/// merge-order behaviour gets a fixture of its own rather than a change to the
/// model vocabulary, which is pinned to the GGUF the numpy oracle reads.
///
/// The vocabulary is built so the algorithms disagree. `ab` and `abc` both
/// exist, and the merge list reaches `abc` from `a`+`b`+`c` in a fixed order.
/// A unigram or longest-match encoder over the same pieces segments `cab`
/// differently than BPE does — which is exactly the failure that shipped, and
/// which no fixture in this tree could previously detect.
pub fn merge_fixture_tokenizer_json() -> anyhow::Result<Vec<u8>> {
    let mut pieces: Vec<String> = ["<pad>", "<s>", "</s>"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    // Byte tokens, so anything outside the content pieces still round-trips.
    pieces.extend((0..=255u8).map(|b| format!("<0x{b:02X}>")));
    pieces.extend(
        ["a", "b", "c", "ab", "abc", SPACE_PIECE_STR]
            .iter()
            .map(|s| s.to_string()),
    );
    let merges = vec![
        ("a".to_string(), "b".to_string()),
        ("ab".to_string(), "c".to_string()),
    ];
    tokenizer_json_from_parts(&pieces, &merges, "<pad>")
}

/// The synthetic model's `config.json`, in Hugging Face `text_config` field
/// names.
///
/// This is the inverse of the mapping the loader will apply, so the two are
/// worth reading together. Note what the checkpoint states directly that the
/// GGUF only implies: `partial_rotary_factor` rather than a precomputed
/// divisor tensor, and `layer_types` as an explicit per-layer list rather than
/// a stride to be inferred from `sliding_window_pattern`.
pub fn build_tiny_config_json(spec: &TinySpec) -> anyhow::Result<Vec<u8>> {
    let hd_global = spec.head_dim_global.unwrap_or(spec.head_dim);
    let is_global = |i: usize| (i + 1).is_multiple_of(spec.sliding_window_pattern);
    let layer_types: Vec<&str> = (0..spec.n_layers)
        .map(|i| match is_global(i) {
            true => "full_attention",
            false => "sliding_attention",
        })
        .collect();
    let boundary = spec.n_layers - spec.shared_kv_layers;
    let intermediate: Vec<usize> = (0..spec.n_layers)
        .map(|i| match spec.double_wide_mlp && i >= boundary {
            true => spec.ffn_dim * 2,
            false => spec.ffn_dim,
        })
        .collect();

    let mut text = serde_json::json!({
        "model_type": "gemma4",
        "num_hidden_layers": spec.n_layers,
        "hidden_size": spec.embed_dim,
        "vocab_size": spec.vocab_size,
        "num_attention_heads": spec.n_heads,
        "num_key_value_heads": spec.n_kv_heads,
        "head_dim": spec.head_dim,
        "global_head_dim": hd_global,
        "layer_types": layer_types,
        "sliding_window": spec.sliding_window,
        "max_position_embeddings": spec.context_length,
        "rms_norm_eps": spec.rms_eps,
        "rope_theta": spec.rope_freq_base,
        "hidden_activation": "gelu_pytorch_tanh",
        "tie_word_embeddings": true,
        "use_double_wide_mlp": spec.double_wide_mlp,
        "num_kv_shared_layers": spec.shared_kv_layers,
        // Per-layer, matching what `use_double_wide_mlp` produces, so the
        // loader reads the widths rather than re-deriving them from the flag.
        "intermediate_size": intermediate,
    });
    let obj = text
        .as_object_mut()
        .expect("json! literal above is an object");
    if let Some(swa) = spec.rope_freq_base_swa {
        obj.insert("rope_local_base_freq".to_string(), swa.into());
    }
    if let Some(cap) = spec.logit_softcap {
        obj.insert("final_logit_softcapping".to_string(), cap.into());
    }
    if let Some(pd) = spec.ple_dim {
        obj.insert("hidden_size_per_layer_input".to_string(), pd.into());
    }
    // The GGUF carries precomputed divisors; the checkpoint carries the factor
    // they were computed from. `rope_rotating_pairs` counts rotating pairs out
    // of `hd_global / 2`, and the factor is the fraction of the head rotated.
    if let Some(rotating) = spec.rope_rotating_pairs {
        let factor = (2 * rotating) as f64 / hd_global as f64;
        obj.insert("partial_rotary_factor".to_string(), factor.into());
    }

    Ok(serde_json::to_vec_pretty(&serde_json::json!({
        "model_type": "gemma4",
        "text_config": text,
    }))?)
}

/// The synthetic model's `model.safetensors`.
///
/// Three things differ from the GGUF image and they are the whole risk of the
/// loader that follows. Names are the Hugging Face ones (`hf_tensor_name`),
/// and shapes are row-major `[out, in]` — which is the *same byte layout* the
/// GGUF already holds, since GGUF dims are `ne` order with `ne0` contiguous.
/// So this reverses the dims and writes the identical payload; there is no
/// data transpose, and a future loader that transposes is wrong.
///
/// The third is RMSNorm. A published checkpoint stores the *unshifted*
/// parameter and transformers adds 1.0 at runtime; the GGUF converter
/// resolves that shift before writing, so `SynthTensor::values` holds the
/// already-shifted effective multiplier. This sink therefore subtracts 1.0
/// from every norm tensor, which is the converter's transformation run
/// backwards. `cpu.rs` applies `x * (1 + w)` to what it reads here and lands
/// on the same effective multiplier the GGUF path uses, which is why the
/// committed logits goldens are unchanged by the move between containers.
///
/// In f32 the round trip is exact, not merely close: the draw is
/// `1.0 + u` for `u` in `[-0.1, 0.1)`, so every norm weight lies in
/// `[0.5, 2.0]` where Sterbenz's lemma makes `w - 1.0` exactly
/// representable, and `1.0 + (w - 1.0)` recovers `w` bit for bit. The
/// equality test in `cpu_reference` asserts that rather than assuming it.
///
/// Under `TensorStorage::Bf16` the two containers deliberately do *not* agree
/// on norm tensors: truncating `w - 1.0` keeps far more significant bits than
/// truncating `w`, because the value sits near zero instead of near one. That
/// is the real relationship too — a converted GGUF loses precision on norms
/// that the published checkpoint retains — so this sink is the more faithful
/// of the two rather than the divergent one.
///
/// Tensors with no published counterpart are skipped, which today means the
/// precomputed RoPE divisors.
pub fn build_tiny_safetensors(spec: &TinySpec) -> anyhow::Result<Vec<u8>> {
    use safetensors::tensor::{Dtype, TensorView};

    let tensors = tiny_tensors(spec);
    // Payloads must outlive the views that borrow them.
    let payloads: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .filter_map(|t| {
            let name = t.hf_name.clone()?;
            // Unshift before narrowing, so a bf16 norm keeps the precision of
            // the small value rather than of the value near one.
            let shift = if t.norm { 1.0f32 } else { 0.0f32 };
            let values = t.values.iter().map(|v| v - shift);
            let (dtype, bytes) = match t.storage {
                TensorStorage::F32 => (
                    Dtype::F32,
                    values.flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
                ),
                // Truncating, matching `GgufWriter::tensor`.
                TensorStorage::Bf16 => (
                    Dtype::BF16,
                    values
                        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                        .collect::<Vec<u8>>(),
                ),
            };
            Some((name, dtype, t.hf_shape(), bytes))
        })
        .collect();

    let views: Vec<(&str, TensorView)> = payloads
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            let view = TensorView::new(*dtype, shape.clone(), bytes)
                .map_err(|e| anyhow::anyhow!("tensor {name}: {e}"))?;
            Ok((name.as_str(), view))
        })
        .collect::<anyhow::Result<_>>()?;

    Ok(safetensors::serialize(views, None)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf;

    #[test]
    fn tiny_model_parses_with_expected_shape() {
        let image = build_tiny_model(&TinySpec::default());
        let file = gguf::parse(&image).unwrap();
        assert_eq!(
            file.metadata["general.architecture"],
            gguf::MetaValue::Str("gemma4".into())
        );
        assert_eq!(file.metadata["gemma4.block_count"], gguf::MetaValue::U32(2));
        // 1 embedding + 2 layers x 13 tensors + 1 output norm.
        assert_eq!(file.tensors.len(), 28);
        // Every tensor payload is loadable and dequantizes to finite values.
        for info in &file.tensors {
            let values = crate::tensor::load_tensor(&file, &image, info).unwrap();
            assert!(values.iter().all(|v| v.is_finite()), "{}", info.name);
        }
    }

    #[test]
    fn same_seed_reproduces_identical_bytes() {
        let a = build_tiny_model(&TinySpec::default());
        let b = build_tiny_model(&TinySpec::default());
        assert_eq!(a, b);
        let c = build_tiny_model(&TinySpec {
            seed: 7,
            ..TinySpec::default()
        });
        assert_ne!(a, c);
    }

    /// The safetensors image holds the same weights as the GGUF one, tensor for
    /// tensor, once the norm shift the two containers spell differently is put
    /// back. This is what stands in for a loader at the writer's own level:
    /// without it the writer could emit wrong shapes for a whole package before
    /// anything noticed.
    ///
    /// Norm tensors are stored unshifted here and reconstructed as `1 + w`, the
    /// arithmetic `cpu.rs` performs. Under F32 that is exact and asserted as
    /// equality; under Bf16 the two containers genuinely differ, and the
    /// expectation is built from the same truncation the writer applied rather
    /// than from the GGUF value.
    #[test]
    fn safetensors_holds_the_same_tensors_as_the_gguf_image() {
        for (label, spec) in [
            ("uniform", TinySpec::default()),
            ("e4b", TinySpec::e4b_shaped()),
        ] {
            let image = build_tiny_safetensors(&spec).unwrap();
            let parsed = safetensors::SafeTensors::deserialize(&image).unwrap();
            let expected = tiny_tensors(&spec);

            let published: Vec<&SynthTensor> =
                expected.iter().filter(|t| t.hf_name.is_some()).collect();
            assert_eq!(
                parsed.names().len(),
                published.len(),
                "{label}: safetensors holds {} tensors, the spec publishes {}",
                parsed.names().len(),
                published.len()
            );

            for t in published {
                let name = t.hf_name.as_ref().unwrap();
                let view = parsed
                    .tensor(name)
                    .unwrap_or_else(|e| panic!("{label}: {name} missing: {e:?}"));
                assert_eq!(
                    view.shape(),
                    t.hf_shape().as_slice(),
                    "{label}: {name} shape is {:?}, expected {:?} (GGUF dims {:?} reversed)",
                    view.shape(),
                    t.hf_shape(),
                    t.dims
                );
                // The payload is compared through the decoder the loader will
                // use, so a byte-order or dtype error shows as a value error.
                let ggml_type = match t.storage {
                    TensorStorage::F32 => GGML_F32,
                    TensorStorage::Bf16 => GGML_BF16,
                };
                let stored =
                    crate::tensor::dequantize(ggml_type, view.data(), t.values.len()).unwrap();
                // What the loader will compute from what it read.
                let shift = if t.norm { 1.0f32 } else { 0.0f32 };
                let got: Vec<f32> = stored.iter().map(|v| v + shift).collect();
                let want: Vec<f32> = match t.storage {
                    TensorStorage::F32 => t.values.clone(),
                    TensorStorage::Bf16 => t
                        .values
                        .iter()
                        .map(|v| f32::from_bits((v - shift).to_bits() & 0xffff_0000) + shift)
                        .collect(),
                };
                for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        g, w,
                        "{label}: {name} element {i}: reconstructed {g}, expected {w}"
                    );
                }
            }
        }
    }

    /// The norm flag selects exactly the tensors drawn from `norm_tensor`, and
    /// those are exactly the RMSNorm weights of the architecture.
    ///
    /// The flag decides an arithmetic transformation in the safetensors sink,
    /// so a tensor wrongly marked would be off by one for the whole run while
    /// still loading, still having the right shape, and still producing
    /// fluent output. Naming the expected set here means adding a norm to the
    /// architecture and forgetting the shift fails a test rather than moving
    /// a golden.
    #[test]
    fn the_norm_flag_marks_every_rmsnorm_weight_and_nothing_else() {
        let spec = TinySpec {
            ple_dim: Some(4),
            ..TinySpec::e4b_shaped()
        };
        let expected_suffixes = [
            "attn_norm.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "post_attention_norm.weight",
            "ffn_norm.weight",
            "post_ffw_norm.weight",
            "post_norm.weight",
        ];
        let tensors = tiny_tensors(&spec);
        let mut seen: Vec<&str> = Vec::new();
        for t in &tensors {
            let want = t.gguf_name == "output_norm.weight"
                || t.gguf_name == "per_layer_proj_norm.weight"
                || expected_suffixes
                    .iter()
                    .any(|s| t.gguf_name.ends_with(&format!(".{s}")));
            assert_eq!(
                t.norm, want,
                "{} is marked norm={}, expected {want}",
                t.gguf_name, t.norm
            );
            if t.norm && !seen.contains(&t.gguf_name.as_str()) {
                seen.push(&t.gguf_name);
            }
        }
        // Every family is actually present, so the assertion above is not
        // passing vacuously over a spec that happens to carry no norms.
        assert_eq!(
            seen.len(),
            2 + expected_suffixes.len() * spec.n_layers,
            "norm families found: {seen:?}"
        );
    }

    /// Unshifting a norm and adding the shift back recovers the GGUF value
    /// bit for bit at f32, which is what lets the logits goldens survive the
    /// move between containers unchanged.
    ///
    /// This is a property of the draw, not luck: `norm_tensor` yields
    /// `1.0 + u` for `u` in `[-0.1, 0.1)`, so every value lies in `[0.5, 2.0]`
    /// where Sterbenz's lemma makes the subtraction exact.
    #[test]
    fn the_norm_unshift_round_trips_exactly_at_f32() {
        let mut ws = WeightStream::new(7);
        let weights = ws.norm_tensor(4096);
        for (i, &w) in weights.iter().enumerate() {
            assert!(
                (0.5..=2.0).contains(&w),
                "norm draw {i} is {w}, outside the range the exactness argument assumes"
            );
            assert_eq!(
                1.0f32 + (w - 1.0f32),
                w,
                "element {i} of {w} did not recover"
            );
        }
    }

    /// Every tensor the GGUF image carries has a published name, except the
    /// precomputed RoPE divisors, which transformers derives at load time from
    /// `partial_rotary_factor` instead. A silent `None` anywhere else would
    /// drop a weight from the checkpoint.
    #[test]
    fn only_the_converter_artifact_lacks_a_published_name() {
        for (label, spec) in [
            ("uniform", TinySpec::default()),
            ("e4b", TinySpec::e4b_shaped()),
        ] {
            let unmapped: Vec<&str> = tiny_tensors(&spec)
                .iter()
                .filter(|t| t.hf_name.is_none())
                .map(|t| t.gguf_name.as_str())
                .collect::<Vec<_>>()
                .iter()
                .map(|s| Box::leak(s.to_string().into_boxed_str()) as &str)
                .collect();
            assert!(
                unmapped.iter().all(|n| *n == "rope_freqs.weight"),
                "{label}: unexpected tensors without a Hugging Face name: {unmapped:?}"
            );
        }
    }

    /// `config.json` states the geometry the GGUF only implies, so the fields
    /// the loader will read are asserted rather than assumed.
    #[test]
    fn config_json_states_the_geometry_the_gguf_implies() {
        let spec = TinySpec::e4b_shaped();
        let doc: serde_json::Value =
            serde_json::from_slice(&build_tiny_config_json(&spec).unwrap()).unwrap();
        let text = &doc["text_config"];

        assert_eq!(text["num_hidden_layers"], spec.n_layers);
        assert_eq!(text["hidden_size"], spec.embed_dim);
        assert_eq!(text["vocab_size"], spec.vocab_size);
        assert_eq!(text["head_dim"], spec.head_dim);
        assert_eq!(
            text["global_head_dim"],
            spec.head_dim_global.unwrap_or(spec.head_dim)
        );
        assert_eq!(text["num_kv_shared_layers"], spec.shared_kv_layers);

        // An explicit per-layer list, not a stride to infer.
        let types = text["layer_types"].as_array().unwrap();
        assert_eq!(types.len(), spec.n_layers, "layer_types: {types:?}");
        for (i, t) in types.iter().enumerate() {
            let want = match (i + 1).is_multiple_of(spec.sliding_window_pattern) {
                true => "full_attention",
                false => "sliding_attention",
            };
            assert_eq!(t, want, "layer {i} in {types:?}");
        }

        // The factor the divisors were computed from, not the divisors.
        let rotating = spec.rope_rotating_pairs.unwrap();
        let half = spec.head_dim_global.unwrap_or(spec.head_dim) / 2;
        let want = (2 * rotating) as f64 / (2 * half) as f64;
        assert_eq!(
            text["partial_rotary_factor"].as_f64().unwrap(),
            want,
            "partial_rotary_factor for {rotating} of {half} pairs"
        );
    }

    /// The emitted tokenizer.json is loadable by `tokenizers` and round-trips
    /// text through encode and decode. The decoder is the part that is easy to
    /// omit: without it, decode returns "<0x41>" where it should return "A".
    #[test]
    fn tiny_tokenizer_json_round_trips_text() {
        let json = build_tiny_tokenizer_json(&TinySpec::default()).unwrap();
        let doc: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(doc["model"]["type"], "BPE", "model: {}", doc["model"]);
        assert_eq!(doc["model"]["byte_fallback"], true);
        assert_eq!(doc["normalizer"]["type"], "Replace");
        assert_eq!(doc["pre_tokenizer"]["type"], "Split");
        assert_eq!(doc["decoder"]["type"], "Sequence");
        // No added-tokens table: the only path from a special literal to its id
        // is the caller's allowlist, never the library's extraction.
        assert!(
            doc["added_tokens"].as_array().is_none_or(|a| a.is_empty()),
            "unexpected added tokens: {}",
            doc["added_tokens"]
        );

        let tokenizer = tokenizers::Tokenizer::from_bytes(&json)
            .unwrap_or_else(|e| panic!("tokenizer.json did not load: {e}"));
        for text in ["abc", "a b c", "hello there"] {
            let encoding = tokenizer.encode(text, false).unwrap();
            let decoded = tokenizer.decode(encoding.get_ids(), false).unwrap();
            assert_eq!(
                decoded,
                text,
                "round trip failed: {text:?} became {decoded:?} via {:?}",
                encoding.get_tokens()
            );
        }
    }

    /// Byte fallback reaches characters the vocabulary does not hold. The tiny
    /// vocabulary carries only a-z pieces, so anything else must survive as
    /// bytes rather than becoming the unknown token.
    #[test]
    fn text_outside_the_vocabulary_survives_byte_fallback() {
        let json = build_tiny_tokenizer_json(&TinySpec::default()).unwrap();
        let tokenizer = tokenizers::Tokenizer::from_bytes(&json)
            .unwrap_or_else(|e| panic!("tokenizer.json did not load: {e}"));
        for text in ["Z9!", "\u{00e9}"] {
            let encoding = tokenizer.encode(text, false).unwrap();
            let decoded = tokenizer.decode(encoding.get_ids(), false).unwrap();
            assert_eq!(
                decoded,
                text,
                "byte fallback lost {text:?}: became {decoded:?} via {:?}",
                encoding.get_tokens()
            );
        }
    }

    #[test]
    fn tokenizer_loads_from_the_tiny_model() {
        let json = build_tiny_tokenizer_json(&TinySpec::default()).unwrap();
        let tok = crate::tokenizer::Tokenizer::from_json(&json).unwrap();
        assert_eq!(tok.bos(), 1);
        assert_eq!(tok.eos(), 2);
        assert_eq!(tok.vocab_len(), TinySpec::default().vocab_size);
    }
}
