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
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub context_length: usize,
    pub sliding_window: usize,
    /// Every `sliding_window_pattern`-th layer is global (kv.rs convention).
    pub sliding_window_pattern: usize,
    pub rope_freq_base: f32,
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
            ffn_dim: 32,
            context_length: 1024,
            sliding_window: 4,
            sliding_window_pattern: 2,
            rope_freq_base: 10000.0,
            rms_eps: 1e-6,
            seed: 42,
            storage: TensorStorage::F32,
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
    let hd = spec.head_dim as u64;
    let q_out = spec.n_heads as u64 * hd;
    let kv_out = spec.n_kv_heads as u64 * hd;
    let ffn = spec.ffn_dim as u64;
    let mut w = GgufWriter::new();

    w.kv_str("general.architecture", "gemma4");
    w.kv_u32("gemma4.block_count", spec.n_layers as u32);
    w.kv_u32("gemma4.context_length", spec.context_length as u32);
    w.kv_u32("gemma4.embedding_length", spec.embed_dim as u32);
    w.kv_u32("gemma4.feed_forward_length", spec.ffn_dim as u32);
    w.kv_u32("gemma4.attention.head_count", spec.n_heads as u32);
    w.kv_u32("gemma4.attention.head_count_kv", spec.n_kv_heads as u32);
    w.kv_u32("gemma4.attention.key_length", spec.head_dim as u32);
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
    }
    let output_norm = ws.norm_tensor(spec.embed_dim);
    w.tensor("output_norm.weight", &[d], &output_norm, spec.storage);
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
