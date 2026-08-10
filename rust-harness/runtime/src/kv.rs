/*
 * kv — KV cache sizing for interleaved sliding-window/global attention.
 *
 * Gemma interleaves sliding-window and global attention layers, and the two
 * functions here answer two different questions about what that costs.
 *
 * `budget_bytes` is the design: a windowed layer that discards tokens older
 * than its window needs a fixed cache regardless of context length, so the
 * total is a fixed cost for windowed layers plus a per-token cost for global
 * layers only.
 *
 * `resident_bytes` is the present: `model::cpu` applies the window as an
 * attention mask and stores the full sequence in every cache-owning layer
 * (see the `LayerKv` comment there). Nothing allocates against
 * `budget_bytes` today, so a startup estimate that used it would report less
 * memory than the process goes on to take.
 *
 * Both take one geometry per layer rather than one per model, because Gemma
 * 4's global layers run wider heads than its sliding ones.
 */

/// Attention kind of one transformer layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerKind {
    Global,
    SlidingWindow(usize),
}

/// Cache geometry of one layer.
///
/// `head_dim` is per layer, not per model: Gemma 4's global layers are twice
/// as wide as its sliding layers, so one model-wide value under-reports the
/// budget by the difference across every global layer.
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

/// Cache bytes for `max_context` tokens across `layers` **if windowed layers
/// capped their storage at the window**.
///
/// This is the intended design, not the current one. `model::cpu` stores the
/// full sequence in every layer, so this understates what a run actually
/// holds -- use `resident_bytes` for anything reported to a user or compared
/// against available memory. Kept because it is the target a windowed-storage
/// backend would be measured against, and the gap between the two functions
/// is exactly what implementing that would save.
pub fn budget_bytes(layers: &[(LayerKind, KvGeometry)], max_context: usize) -> usize {
    layers
        .iter()
        .map(|(layer, geometry)| {
            let per_token = geometry.bytes_per_token();
            match layer {
                LayerKind::Global => max_context * per_token,
                LayerKind::SlidingWindow(w) => max_context.min(*w) * per_token,
            }
        })
        .sum()
}

/// Cache bytes a backend that stores the full sequence in every layer
/// actually holds at `max_context` tokens.
///
/// The window is irrelevant here, which is the whole difference from
/// `budget_bytes`: `model::cpu` masks old positions out of the attention
/// score rather than dropping them from the cache, so a sliding layer costs
/// exactly what a global one of the same geometry costs.
///
/// Pass only cache-*owning* layers. A shared-KV layer reads a donor's cache
/// and stores nothing, so including one over-reports by a whole layer -- 18
/// of E4B's 42, 20 of E2B's 35.
pub fn resident_bytes(layers: &[(LayerKind, KvGeometry)], max_context: usize) -> usize {
    layers
        .iter()
        .map(|(_, geometry)| max_context * geometry.bytes_per_token())
        .sum()
}

/// Pair a uniform geometry with every layer, for a model whose head width
/// does not vary.
pub fn uniform(layers: &[LayerKind], geometry: KvGeometry) -> Vec<(LayerKind, KvGeometry)> {
    layers.iter().map(|l| (*l, geometry)).collect()
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
        assert_eq!(budget_bytes(&uniform(&layers, GEO), 8192), expected);
    }

    #[test]
    fn short_context_never_exceeds_the_window() {
        let layers = [LayerKind::SlidingWindow(1024), LayerKind::Global];
        // Context below the window: both layers cost context x per-token.
        assert_eq!(budget_bytes(&uniform(&layers, GEO), 512), 2 * 512 * 2048);
    }

    /// What `model::cpu` actually holds ignores the window entirely, so a
    /// windowed layer and a global one of the same geometry cost the same.
    /// This is the property that makes `budget_bytes` an understatement.
    #[test]
    fn resident_bytes_charges_a_windowed_layer_as_much_as_a_global_one() {
        let layers = [LayerKind::SlidingWindow(1024), LayerKind::Global];
        let paired = uniform(&layers, GEO);
        assert_eq!(resident_bytes(&paired, 8192), 2 * 8192 * 2048);
        // And the gap it opens against the design, which is what a
        // windowed-storage backend would recover.
        assert_eq!(
            resident_bytes(&paired, 8192) - budget_bytes(&paired, 8192),
            (8192 - 1024) * 2048
        );
    }

    /// Below the window nothing has been discarded yet, so the design and the
    /// present agree. A divergence here would mean one of them is wrong about
    /// something other than the window.
    #[test]
    fn resident_and_budget_agree_when_the_context_fits_the_window() {
        let layers = interleaved_layers(6, 3, 1024);
        let paired = uniform(&layers, GEO);
        for context in [0, 1, 512, 1024] {
            assert_eq!(
                resident_bytes(&paired, context),
                budget_bytes(&paired, context),
                "context {context}"
            );
        }
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
        assert_eq!(budget_bytes(&[], 8192), 0);
        assert_eq!(budget_bytes(&uniform(&[LayerKind::Global], GEO), 0), 0);
    }

    #[test]
    fn windowed_budget_is_flat_beyond_the_window() {
        // The whole point: doubling context only grows the global layers.
        let layers = interleaved_layers(6, 3, 1024);
        let at_8k = budget_bytes(&uniform(&layers, GEO), 8192);
        let at_16k = budget_bytes(&uniform(&layers, GEO), 16384);
        let global_growth = 2 * 8192 * 2048; // two global layers, 8192 more tokens
        assert_eq!(at_16k - at_8k, global_growth);
    }
}
