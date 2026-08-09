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
    for i in 0..vocab_size - 259 {
        let c = (b'a' + (i as u8) % 26) as char;
        if i % 2 == 0 {
            pieces.push(format!("\u{2581}{c}"));
        } else {
            pieces.push(c.to_string());
        }
        kinds.push(1);
    }
    let scores: Vec<f32> = (0..vocab_size).map(|i| -(i as f32)).collect();
    (pieces, scores, kinds)
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

/// Build the complete synthetic model image for `spec`.
pub fn build_tiny_model(spec: &TinySpec) -> Vec<u8> {
    let d = spec.embed_dim as u64;
    let hd_global = spec.head_dim_global.unwrap_or(spec.head_dim);
    // Layer `i` is global when `i + 1` divides by the pattern, matching
    // `Config::layer_kind` and `kv::interleaved_layers`.
    let is_global = |i: usize| (i + 1).is_multiple_of(spec.sliding_window_pattern);
    let head_dim_of = |i: usize| match is_global(i) {
        true => hd_global,
        false => spec.head_dim,
    };
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

    let mut ws = WeightStream::new(spec.seed);
    let n = |dims: &[u64]| dims.iter().product::<u64>() as usize;

    let embd_dims = [d, spec.vocab_size as u64];
    let embd = ws.tensor(n(&embd_dims));
    w.tensor("token_embd.weight", &embd_dims, &embd, spec.storage);
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
            let values = if suffix.ends_with("norm.weight") {
                ws.norm_tensor(n(&dims))
            } else {
                ws.tensor(n(&dims))
            };
            w.tensor(&format!("blk.{i}.{suffix}"), &dims, &values, spec.storage);
        }
        // Written F32 whatever `spec.storage` says, matching the converter:
        // the published GGUF carries `blk.{n}.layer_output_scale.weight [1]
        // F32` even in an otherwise BF16 file.
        if let Some(scale) = spec.layer_output_scale {
            w.tensor_f32(
                &format!("blk.{i}.layer_output_scale.weight"),
                &[1],
                &[scale],
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
        w.tensor_f32("rope_freqs.weight", &[half as u64], &divisors);
    }

    let output_norm = ws.norm_tensor(spec.embed_dim);
    w.tensor("output_norm.weight", &[d], &output_norm, spec.storage);

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
        let table_dims = [packed, spec.vocab_size as u64];
        let table = ws.tensor(n(&table_dims));
        w.tensor(
            "per_layer_token_embd.weight",
            &table_dims,
            &table,
            spec.storage,
        );
        let proj_dims = [d, packed];
        let model_proj = ws.tensor(n(&proj_dims));
        w.tensor(
            "per_layer_model_proj.weight",
            &proj_dims,
            &model_proj,
            spec.storage,
        );
        let proj_norm = ws.norm_tensor(pd);
        w.tensor(
            "per_layer_proj_norm.weight",
            &[pd64],
            &proj_norm,
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
                let values = if suffix.ends_with("norm.weight") {
                    ws.norm_tensor(n(&dims))
                } else {
                    ws.tensor(n(&dims))
                };
                w.tensor(&format!("blk.{i}.{suffix}"), &dims, &values, spec.storage);
            }
        }
    }
    // LM head is tied to token_embd, llama.cpp Gemma convention: no
    // output.weight tensor.
    w.finish()
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

    #[test]
    fn tokenizer_loads_from_the_tiny_model() {
        let image = build_tiny_model(&TinySpec::default());
        let file = gguf::parse(&image).unwrap();
        let tok = crate::tokenizer::Tokenizer::from_gguf(&file).unwrap();
        assert_eq!(tok.bos(), 1);
        assert_eq!(tok.eos(), 2);
    }
}
