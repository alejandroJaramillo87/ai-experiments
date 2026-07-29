/*
 * cpu — hand-rolled CPU reference backend for Gemma-4-style checkpoints.
 *
 * Correctness-first: plain single-threaded f32 loops, no SIMD, no wall
 * clock, no allocation tricks — deterministic by construction so runs
 * record and replay under rr. This backend is the reference the CUDA
 * backend is diffed against (through the phase 1 flight recorder) and makes
 * the full harness runnable in a sandbox against synthetic weights.
 *
 * Architecture (Gemma-3-style, to be reconciled against the real E4B GGUF
 * dump): RMSNorm with (1 + weight) scaling, embedding scaled by
 * sqrt(embed_dim), grouped-query attention with per-head QK-norm and NEOX
 * RoPE, interleaved sliding-window/global layers, GeGLU MLP with tanh GELU,
 * four norms per layer, LM head tied to the token embedding.
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

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub context_length: usize,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
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
            head_dim: u("attention.key_length")?,
            ffn_dim: u("feed_forward_length")?,
            context_length: u("context_length")?,
            sliding_window: u("attention.sliding_window")?,
            sliding_window_pattern: u("attention.sliding_window_pattern")?,
            rope_freq_base: f("rope.freq_base")?,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
        };
        ensure!(
            config.n_heads.is_multiple_of(config.n_kv_heads.max(1)) && config.n_kv_heads > 0,
            "head_count {} not divisible by head_count_kv {}",
            config.n_heads,
            config.n_kv_heads
        );
        ensure!(
            config.head_dim.is_multiple_of(2),
            "head_dim {} must be even for RoPE",
            config.head_dim
        );
        Ok(config)
    }

    /// Attention kind of layer `i`, matching `kv::interleaved_layers`.
    fn layer_kind(&self, i: usize) -> LayerKind {
        if (i + 1).is_multiple_of(self.sliding_window_pattern) {
            LayerKind::Global
        } else {
            LayerKind::SlidingWindow(self.sliding_window)
        }
    }
}

struct LayerWeights {
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

pub struct CpuModel {
    config: Config,
    weights: Arc<Weights>,
    kv: Vec<LayerKv>,
    cached_len: usize,
    /// None on construction and after fork (a `Box<dyn Tap>` cannot be
    /// cloned); absent taps cost one branch per layer.
    tap: Option<Box<dyn Tap>>,
}

/// `y = W x` with GGUF layout `[in, out]`: rows of `in` contiguous per
/// output element.
fn matvec(w: &[f32], x: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    debug_assert_eq!(w.len(), in_dim * out_dim);
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            row.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// Gemma RMSNorm: `x / rms(x) * (1 + w)`.
fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * (1.0 + g)).collect()
}

/// Tanh-approximated GELU, the Gemma activation.
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

/// NEOX-style RoPE over one head in place: pairs are (j, j + head_dim/2).
fn rope(head: &mut [f32], position: usize, freq_base: f32) {
    let half = head.len() / 2;
    for j in 0..half {
        let freq = freq_base.powf(-2.0 * j as f32 / head.len() as f32);
        let angle = position as f32 * freq;
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
        let config = Config::from_gguf(file)?;
        let tensor = |name: &str| -> Result<Vec<f32>> {
            let info = file
                .tensors
                .iter()
                .find(|t| t.name == name)
                .with_context(|| format!("missing tensor {name}"))?;
            load_tensor(file, data, info)
        };
        let embed = tensor("token_embd.weight")?;
        ensure!(
            embed.len() == config.vocab_size * config.embed_dim,
            "token_embd.weight has {} elements, config needs {}",
            embed.len(),
            config.vocab_size * config.embed_dim
        );
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let t = |suffix: &str| tensor(&format!("blk.{i}.{suffix}"));
            layers.push(LayerWeights {
                attn_norm: t("attn_norm.weight")?,
                wq: t("attn_q.weight")?,
                wk: t("attn_k.weight")?,
                wv: t("attn_v.weight")?,
                q_norm: t("attn_q_norm.weight")?,
                k_norm: t("attn_k_norm.weight")?,
                wo: t("attn_output.weight")?,
                post_attn_norm: t("post_attention_norm.weight")?,
                ffn_norm: t("ffn_norm.weight")?,
                w_gate: t("ffn_gate.weight")?,
                w_up: t("ffn_up.weight")?,
                w_down: t("ffn_down.weight")?,
                post_ffw_norm: t("post_ffw_norm.weight")?,
            });
        }
        let output_norm = tensor("output_norm.weight")?;
        Ok(CpuModel {
            kv: vec![LayerKv::default(); config.n_layers],
            config,
            weights: Arc::new(Weights {
                embed,
                layers,
                output_norm,
            }),
            cached_len: 0,
            tap: None,
        })
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

    /// One decode position: append the token's K/V to every layer cache and
    /// return the logits at this position.
    fn forward_one(&mut self, token: Token, position: usize) -> Result<Logits> {
        let c = self.config;
        let w = Arc::clone(&self.weights);
        ensure!(
            (token as usize) < c.vocab_size,
            "token {token} outside vocabulary of {}",
            c.vocab_size
        );
        let (hd, kv_dim) = (c.head_dim, c.n_kv_heads * c.head_dim);
        let heads_per_kv = c.n_heads / c.n_kv_heads;

        // Embedding, scaled by sqrt(embed_dim) per Gemma convention.
        let row = &w.embed[token as usize * c.embed_dim..(token as usize + 1) * c.embed_dim];
        let scale = (c.embed_dim as f32).sqrt();
        let mut x: Vec<f32> = row.iter().map(|v| v * scale).collect();

        for (i, layer) in w.layers.iter().enumerate() {
            // Attention block.
            let h = rms_norm(&x, &layer.attn_norm, c.rms_eps);
            let mut q = matvec(&layer.wq, &h, c.n_heads * hd);
            let mut k = matvec(&layer.wk, &h, kv_dim);
            let v = matvec(&layer.wv, &h, kv_dim);
            for head in q.chunks_mut(hd) {
                let normed = rms_norm(head, &layer.q_norm, c.rms_eps);
                head.copy_from_slice(&normed);
                rope(head, position, c.rope_freq_base);
            }
            for head in k.chunks_mut(hd) {
                let normed = rms_norm(head, &layer.k_norm, c.rms_eps);
                head.copy_from_slice(&normed);
                rope(head, position, c.rope_freq_base);
            }
            let cache = &mut self.kv[i];
            cache.k.extend_from_slice(&k);
            cache.v.extend_from_slice(&v);

            // Causal attention over the cache; sliding-window layers see
            // only the trailing `window` positions including this one.
            let first = match c.layer_kind(i) {
                LayerKind::Global => 0,
                LayerKind::SlidingWindow(win) => (position + 1).saturating_sub(win),
            };
            let attn_scale = 1.0 / (hd as f32).sqrt();
            let mut attn_out = vec![0.0f32; c.n_heads * hd];
            for qh in 0..c.n_heads {
                let kvh = qh / heads_per_kv;
                let qv = &q[qh * hd..(qh + 1) * hd];
                let mut scores: Vec<f32> = (first..=position)
                    .map(|p| {
                        let kp = &cache.k[p * kv_dim + kvh * hd..p * kv_dim + (kvh + 1) * hd];
                        qv.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>() * attn_scale
                    })
                    .collect();
                softmax_in_place(&mut scores);
                let out = &mut attn_out[qh * hd..(qh + 1) * hd];
                for (s, p) in scores.iter().zip(first..=position) {
                    let vp = &cache.v[p * kv_dim + kvh * hd..p * kv_dim + (kvh + 1) * hd];
                    for (o, val) in out.iter_mut().zip(vp) {
                        *o += s * val;
                    }
                }
            }
            let projected = matvec(&layer.wo, &attn_out, c.embed_dim);
            let post = rms_norm(&projected, &layer.post_attn_norm, c.rms_eps);
            for (a, b) in x.iter_mut().zip(&post) {
                *a += b;
            }

            // GeGLU MLP block.
            let h = rms_norm(&x, &layer.ffn_norm, c.rms_eps);
            let gate = matvec(&layer.w_gate, &h, c.ffn_dim);
            let up = matvec(&layer.w_up, &h, c.ffn_dim);
            let mixed: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| gelu(*g) * u).collect();
            let down = matvec(&layer.w_down, &mixed, c.embed_dim);
            let post = rms_norm(&down, &layer.post_ffw_norm, c.rms_eps);
            for (a, b) in x.iter_mut().zip(&post) {
                *a += b;
            }

            // Interp tap seam: the layer's complete residual output.
            if let Some(tap) = self.tap.as_mut() {
                tap.on_residual(i, position, &x);
            }
        }

        // Final norm and tied LM head: logits[t] = embed_row(t) . x.
        let x = rms_norm(&x, &w.output_norm, c.rms_eps);
        let logits = (0..c.vocab_size)
            .map(|t| {
                let row = &w.embed[t * c.embed_dim..(t + 1) * c.embed_dim];
                row.iter().zip(&x).map(|(a, b)| a * b).sum()
            })
            .collect();
        Ok(Logits(logits))
    }
}

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
        let span =
            tracing::info_span!("model.forward", start = self.cached_len, len = tokens.len());
        let _enter = span.enter();
        let mut rows = Vec::with_capacity(tokens.len());
        for &t in tokens {
            let position = self.cached_len;
            let logits = self.forward_one(t, position)?;
            self.cached_len += 1;
            rows.push(logits);
        }
        Ok(rows)
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        ensure!(len <= self.cached_len, "truncate beyond cached length");
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;
        for cache in &mut self.kv {
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
            weights: Arc::clone(&self.weights),
            kv: self.kv.clone(),
            cached_len: self.cached_len,
            tap: None,
        }))
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
    #[derive(Clone, Default)]
    struct CollectingTap(std::sync::Arc<std::sync::Mutex<Vec<(usize, usize, Vec<f32>)>>>);

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
        assert!(calls.iter().all(|(_, _, x)| x.iter().all(|v| v.is_finite())));
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
