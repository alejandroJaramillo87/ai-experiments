/*
 * kv — KV cache budgeting for interleaved sliding-window/global attention.
 *
 * Gemma interleaves sliding-window and global attention layers. Windowed
 * layers cap their cache at the window size regardless of context length, so
 * the total budget is exactly computable at startup: fixed cost for windowed
 * layers plus a per-token cost for global layers only. Allocation happens
 * once per session against this budget.
 */

/// Attention kind of one transformer layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerKind {
    Global,
    SlidingWindow(usize),
}

/// Per-layer cache geometry, shared by every layer of a model.
#[derive(Debug, Clone, Copy)]
pub struct KvGeometry {
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub bytes_per_element: usize,
}

impl KvGeometry {
    /// Bytes one cached token costs in one layer: keys plus values.
    pub fn bytes_per_token(&self) -> usize {
        2 * self.n_kv_heads * self.head_dim * self.bytes_per_element
    }
}

/// Total cache bytes for `max_context` tokens across `layers`.
pub fn budget_bytes(layers: &[LayerKind], geometry: KvGeometry, max_context: usize) -> usize {
    let per_token = geometry.bytes_per_token();
    layers
        .iter()
        .map(|layer| match layer {
            LayerKind::Global => max_context * per_token,
            LayerKind::SlidingWindow(w) => max_context.min(*w) * per_token,
        })
        .sum()
}

/// Gemma-style layer pattern: one global layer every `period` layers,
/// sliding-window layers otherwise. The final layer of each period is global.
pub fn interleaved_layers(n_layers: usize, period: usize, window: usize) -> Vec<LayerKind> {
    assert!(period > 0, "interleave period must be positive");
    (0..n_layers)
        .map(|i| {
            if (i + 1) % period == 0 {
                LayerKind::Global
            } else {
                LayerKind::SlidingWindow(window)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-calculated reference geometry: 4 KV heads x 128 dim x 2 bytes
    // (f16) x 2 (K and V) = 2048 bytes per token per layer.
    const GEO: KvGeometry = KvGeometry {
        n_kv_heads: 4,
        head_dim: 128,
        bytes_per_element: 2,
    };

    #[test]
    fn bytes_per_token_counts_keys_and_values() {
        assert_eq!(GEO.bytes_per_token(), 2048);
    }

    #[test]
    fn global_layers_scale_with_context_and_windowed_layers_cap() {
        // 2 global + 2 windowed(1024) at 8192 context:
        // global: 2 x 8192 x 2048; windowed: 2 x 1024 x 2048.
        let layers = [
            LayerKind::Global,
            LayerKind::SlidingWindow(1024),
            LayerKind::Global,
            LayerKind::SlidingWindow(1024),
        ];
        let expected = 2 * 8192 * 2048 + 2 * 1024 * 2048;
        assert_eq!(budget_bytes(&layers, GEO, 8192), expected);
    }

    #[test]
    fn short_context_never_exceeds_the_window() {
        let layers = [LayerKind::SlidingWindow(1024), LayerKind::Global];
        // Context below the window: both layers cost context x per-token.
        assert_eq!(budget_bytes(&layers, GEO, 512), 2 * 512 * 2048);
    }

    #[test]
    fn interleaved_pattern_places_one_global_per_period() {
        let layers = interleaved_layers(6, 3, 512);
        assert_eq!(
            layers,
            vec![
                LayerKind::SlidingWindow(512),
                LayerKind::SlidingWindow(512),
                LayerKind::Global,
                LayerKind::SlidingWindow(512),
                LayerKind::SlidingWindow(512),
                LayerKind::Global,
            ]
        );
    }

    #[test]
    #[should_panic(expected = "period")]
    fn zero_period_is_rejected() {
        interleaved_layers(4, 0, 512);
    }

    #[test]
    fn zero_layers_and_zero_context_are_empty_budgets() {
        assert_eq!(interleaved_layers(0, 3, 512), Vec::<LayerKind>::new());
        assert_eq!(budget_bytes(&[], GEO, 8192), 0);
        assert_eq!(budget_bytes(&[LayerKind::Global], GEO, 0), 0);
    }

    #[test]
    fn windowed_budget_is_flat_beyond_the_window() {
        // The whole point: doubling context only grows the global layers.
        let layers = interleaved_layers(6, 3, 1024);
        let at_8k = budget_bytes(&layers, GEO, 8192);
        let at_16k = budget_bytes(&layers, GEO, 16384);
        let global_growth = 2 * 8192 * 2048; // two global layers, 8192 more tokens
        assert_eq!(at_16k - at_8k, global_growth);
    }
}
