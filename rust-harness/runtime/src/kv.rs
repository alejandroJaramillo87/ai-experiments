/*
 * kv — KV cache sizing for interleaved sliding-window/global attention.
 *
 * Gemma interleaves sliding-window and global attention layers. A windowed
 * layer discards tokens older than its window, so its cache is a fixed cost
 * regardless of context length; a global layer pays per token. `budget_bytes`
 * is that sum, and `model::cpu` allocates against it -- the two agree, which
 * they did not before the storage cap landed. What the cap recovers is
 * `(context - window)` per windowed layer, which on E4B at its trained context
 * is most of the cache.
 *
 * A windowed layer holds more than its window, and `window_capacity` is the
 * reason: a batched forward pass appends every position of its group before
 * any of them attends, so the group's *first* query row still needs the row
 * `window - 1` before it while the group's last row is already resident. The
 * residency is therefore `window - 1 + group`, and sizing a cache at the
 * window alone under-reports by the batch width. That off-by-a-batch is not
 * academic: it is the eviction bug `runtime/tests/long_context.rs` exists to
 * catch, seen from the memory side rather than the correctness side.
 *
 * Every function takes one geometry per layer rather than one per model,
 * because Gemma 4's global layers run wider heads than its sliding ones.
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

/// Rows a windowed layer must hold to answer correctly, prefilling in groups
/// of `group`.
///
/// `window - 1 + group`, because a group's first query row attends over the
/// `window - 1` rows preceding it while the group's own appends have already
/// pushed the tail forward by `group`. Both spans are resident at once.
///
/// This is the floor, not the allocation: dropping to exactly this on every
/// append would relocate the whole retained tail once per token. See
/// `window_capacity`.
pub fn window_residency(window: usize, group: usize) -> usize {
    window.saturating_sub(1).saturating_add(group)
}

/// Rows a windowed layer is actually allocated: the residency plus a quarter
/// of it as slack.
///
/// The slack is what makes compaction amortized. A compacting buffer that
/// relocated its tail the moment it exceeded the floor would memmove the
/// entire window on every single-token decode; freeing a quarter of the
/// buffer at once instead spreads one relocation over that many appends.
///
/// It is charged for rather than hidden. `model::cpu` sizes its buffers from
/// this and `budget_bytes` reports it, so the number a run prints and the
/// number it allocates cannot drift apart -- which is the entire reason this
/// module exists as arithmetic separate from the backend.
pub fn window_capacity(window: usize, group: usize) -> usize {
    let need = window_residency(window, group);
    need + need.div_ceil(4)
}

/// Cache bytes for `max_context` tokens across `layers`, prefilling in groups
/// of `group` positions.
///
/// What `model::cpu` allocates. A windowed layer is charged
/// `window_capacity`, a global layer the full context, and neither exceeds
/// `max_context` -- below the window nothing has been discarded, so the two
/// kinds agree there.
///
/// `group` matters and is not a detail: at `BatchKind::Whole` it is the whole
/// prefill, which makes a windowed layer cost what a global one costs and the
/// cap save nothing. That is a real property of the batch knob rather than an
/// approximation here, and reporting it is the point.
///
/// Pass only cache-*owning* layers. A shared-KV layer reads a donor's cache
/// and stores nothing, so including one over-reports by a whole layer -- 18
/// of E4B's 42, 20 of E2B's 35.
pub fn budget_bytes(layers: &[(LayerKind, KvGeometry)], max_context: usize, group: usize) -> usize {
    layers
        .iter()
        .map(|(layer, geometry)| {
            let per_token = geometry.bytes_per_token();
            match layer {
                LayerKind::Global => max_context * per_token,
                LayerKind::SlidingWindow(w) => {
                    max_context.min(window_capacity(*w, group)) * per_token
                }
            }
        })
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

    /// Stepped decode is the smallest group, so the residency is the window
    /// itself: `window - 1 + 1`.
    #[test]
    fn a_stepped_group_needs_exactly_the_window() {
        assert_eq!(window_residency(1024, 1), 1024);
        assert_eq!(window_residency(3, 1), 3);
    }

    /// The batch term, which is the whole reason this is a function rather
    /// than a bare `min(context, window)`. A group's first query row needs
    /// rows its last row has already pushed past.
    #[test]
    fn a_batched_group_needs_the_window_plus_the_batch_less_one() {
        assert_eq!(window_residency(512, 64), 575);
    }

    /// The slack is a quarter on top, and it is allocated rather than
    /// notional -- a caller comparing the budget against free memory is
    /// comparing against what the process takes.
    #[test]
    fn capacity_is_the_residency_plus_a_quarter() {
        assert_eq!(window_capacity(1024, 1), 1024 + 256);
        assert_eq!(window_capacity(512, 64), 575 + 144);
    }

    /// Slack never rounds to nothing, or a tiny-window model would relocate
    /// its whole buffer on every append.
    #[test]
    fn even_a_tiny_window_gets_a_row_of_slack() {
        assert_eq!(window_residency(3, 1), 3);
        assert_eq!(window_capacity(3, 1), 4);
    }

    /// A window of zero is not a model this backend loads, but the helpers
    /// are arithmetic and must not underflow on the way to the error that
    /// rejects it.
    #[test]
    fn a_zero_window_does_not_underflow() {
        assert_eq!(window_residency(0, 1), 1);
        assert_eq!(window_residency(0, 0), 0);
        assert_eq!(window_capacity(0, 0), 0);
    }

    #[test]
    fn global_layers_scale_with_context_and_windowed_layers_cap() {
        // 2 global + 2 windowed(1024) at 8192 context, stepped:
        // global: 2 x 8192 x 2048; windowed: 2 x 1280 x 2048.
        let layers = [
            LayerKind::Global,
            LayerKind::SlidingWindow(1024),
            LayerKind::Global,
            LayerKind::SlidingWindow(1024),
        ];
        let expected = 2 * 8192 * 2048 + 2 * 1280 * 2048;
        assert_eq!(budget_bytes(&uniform(&layers, GEO), 8192, 1), expected);
    }

    /// The batch width is charged for, once per windowed layer. A budget that
    /// ignored it would report less than the process goes on to take, which
    /// is the failure mode this whole module exists to avoid.
    #[test]
    fn the_batch_width_is_charged_once_per_windowed_layer() {
        let layers = [
            LayerKind::Global,
            LayerKind::SlidingWindow(1024),
            LayerKind::Global,
            LayerKind::SlidingWindow(1024),
        ];
        let paired = uniform(&layers, GEO);
        let widened = window_capacity(1024, 64) - window_capacity(1024, 1);
        assert_eq!(
            budget_bytes(&paired, 8192, 64) - budget_bytes(&paired, 8192, 1),
            2 * widened * 2048
        );
    }

    #[test]
    fn short_context_never_exceeds_the_window() {
        let layers = [LayerKind::SlidingWindow(1024), LayerKind::Global];
        // Context below the window: both layers cost context x per-token.
        assert_eq!(
            budget_bytes(&uniform(&layers, GEO), 512, 64),
            2 * 512 * 2048
        );
    }

    /// `BatchKind::Whole` makes the group the whole prefill, so a windowed
    /// layer costs what a global one costs and the cap saves nothing.
    /// Asserted rather than left as a remark, because it is the one
    /// configuration in which the headline number is not the design.
    #[test]
    fn a_whole_prefill_costs_what_full_storage_cost() {
        let layers = interleaved_layers(6, 3, 1024);
        let paired = uniform(&layers, GEO);
        assert_eq!(budget_bytes(&paired, 8192, 8192), 6 * 8192 * 2048);
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
        assert_eq!(budget_bytes(&[], 8192, 64), 0);
        assert_eq!(budget_bytes(&uniform(&[LayerKind::Global], GEO), 0, 64), 0);
    }

    #[test]
    fn windowed_budget_is_flat_beyond_the_window() {
        // The whole point: doubling context only grows the global layers.
        let layers = interleaved_layers(6, 3, 1024);
        let at_8k = budget_bytes(&uniform(&layers, GEO), 8192, 64);
        let at_16k = budget_bytes(&uniform(&layers, GEO), 16384, 64);
        let global_growth = 2 * 8192 * 2048; // two global layers, 8192 more tokens
        assert_eq!(at_16k - at_8k, global_growth);
    }
}
