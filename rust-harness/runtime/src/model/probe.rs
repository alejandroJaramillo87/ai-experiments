/*
 * probe — in-band linear probe monitor over the residual-stream tap.
 *
 * Loads the probe GGUF that interp/probes/export.py writes (architecture
 * "probe": f32 tensors probe.weights/probe.mu/probe.sigma, KV metadata
 * probe.bias/probe.layer/probe.threshold/probe.name/probe.model_hash) and
 * scores every position of the probe's layer as it flows through the
 * forward pass. The scoring math mirrors interp/probes/score.py exactly:
 * sigmoid(((x - mu) / sigma) . w + bias), flagged when strictly above the
 * trained threshold. Observation only — the monitor never alters the
 * stream, so monitored and unmonitored runs produce identical logits.
 */

use std::sync::{Arc, Mutex};

use anyhow::{bail, ensure, Context, Result};

use super::cpu::Tap;
use crate::gguf::{GgufFile, MetaValue};
use crate::tensor::load_tensor;

/// A trained logistic-regression probe: standardization statistics plus a
/// weight vector over one decoder layer's residual stream.
pub struct Probe {
    pub name: String,
    pub model_hash: String,
    /// 0-based decoder layer whose post-block residual the probe scores.
    pub layer: usize,
    pub threshold: f32,
    weights: Vec<f32>,
    mu: Vec<f32>,
    sigma: Vec<f32>,
    bias: f32,
}

impl Probe {
    pub fn from_gguf(file: &GgufFile, data: &[u8]) -> Result<Probe> {
        let s = |key: &str| -> Result<String> {
            match file.metadata.get(key) {
                Some(MetaValue::Str(v)) => Ok(v.clone()),
                Some(other) => bail!("{key}: expected string, got {other:?}"),
                None => bail!("missing {key}"),
            }
        };
        let f = |key: &str| -> Result<f32> {
            match file.metadata.get(key) {
                Some(MetaValue::F32(v)) => Ok(*v),
                Some(other) => bail!("{key}: expected f32, got {other:?}"),
                None => bail!("missing {key}"),
            }
        };
        let layer = match file.metadata.get("probe.layer") {
            Some(MetaValue::U32(v)) => *v as usize,
            Some(other) => bail!("probe.layer: expected u32, got {other:?}"),
            None => bail!("missing probe.layer"),
        };
        let tensor = |name: &str| -> Result<Vec<f32>> {
            let info = file
                .tensors
                .iter()
                .find(|t| t.name == name)
                .with_context(|| format!("probe GGUF is missing tensor {name}"))?;
            load_tensor(file, data, info)
        };
        let weights = tensor("probe.weights")?;
        let mu = tensor("probe.mu")?;
        let sigma = tensor("probe.sigma")?;
        ensure!(
            weights.len() == mu.len() && mu.len() == sigma.len() && !weights.is_empty(),
            "probe tensors disagree on width: weights {}, mu {}, sigma {}",
            weights.len(),
            mu.len(),
            sigma.len()
        );
        ensure!(
            sigma.iter().all(|v| *v > 0.0),
            "probe.sigma must be strictly positive"
        );
        let threshold = f("probe.threshold")?;
        ensure!(
            threshold.is_finite(),
            "probe.threshold must be finite, got {threshold}"
        );
        Ok(Probe {
            name: s("probe.name")?,
            model_hash: s("probe.model_hash")?,
            layer,
            threshold,
            weights,
            mu,
            sigma,
            bias: f("probe.bias")?,
        })
    }

    pub fn width(&self) -> usize {
        self.weights.len()
    }

    /// Sigmoid probe score for one residual row, the score.py math.
    pub fn score(&self, x: &[f32]) -> f32 {
        debug_assert_eq!(x.len(), self.weights.len());
        let logit: f32 = x
            .iter()
            .zip(&self.mu)
            .zip(&self.sigma)
            .zip(&self.weights)
            .map(|(((x, mu), sigma), w)| (x - mu) / sigma * w)
            .sum::<f32>()
            + self.bias;
        1.0 / (1.0 + (-logit).exp())
    }
}

/// One over-threshold position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flag {
    pub position: usize,
    pub score: f32,
}

/// Monitor results, shared out through an Arc so the caller can drain them
/// after the run (`set_tap` takes ownership of the tap itself).
#[derive(Default)]
pub struct MonitorState {
    /// Positions of the probe's layer that were scored.
    pub scored: usize,
    pub flags: Vec<Flag>,
}

/// Tap implementation scoring the probe's layer at every position. Other
/// layers cost one integer comparison.
pub struct ProbeMonitor {
    probe: Arc<Probe>,
    state: Arc<Mutex<MonitorState>>,
}

impl ProbeMonitor {
    pub fn new(probe: Arc<Probe>) -> (ProbeMonitor, Arc<Mutex<MonitorState>>) {
        let state = Arc::new(Mutex::new(MonitorState::default()));
        (
            ProbeMonitor {
                probe,
                state: Arc::clone(&state),
            },
            state,
        )
    }
}

impl Tap for ProbeMonitor {
    fn on_residual(&mut self, layer: usize, position: usize, x: &[f32]) {
        if layer != self.probe.layer || x.len() != self.probe.width() {
            return;
        }
        let score = self.probe.score(x);
        let mut state = self.state.lock().expect("monitor state lock");
        state.scored += 1;
        if score > self.probe.threshold {
            state.flags.push(Flag { position, score });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_probe(threshold: f32) -> Probe {
        Probe {
            name: "test".into(),
            model_hash: "hash".into(),
            layer: 1,
            threshold,
            weights: vec![1.0, -2.0, 0.5],
            mu: vec![0.5, 0.0, 1.0],
            sigma: vec![1.0, 2.0, 0.5],
            bias: 0.25,
        }
    }

    #[test]
    fn score_matches_a_hand_computed_sigmoid() {
        let probe = test_probe(0.5);
        // logit = (1.5-0.5)/1*1 + (1-0)/2*(-2) + (1-1)/0.5*0.5 + 0.25 = 0.25
        let score = probe.score(&[1.5, 1.0, 1.0]);
        let expected = 1.0 / (1.0 + (-0.25f32).exp());
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn monitor_flags_only_scores_strictly_above_threshold() {
        let mut probe = test_probe(0.0);
        probe.threshold = probe.score(&[1.5, 1.0, 1.0]); // exact score of position 0
        let (mut monitor, state) = ProbeMonitor::new(Arc::new(probe));
        monitor.on_residual(1, 0, &[1.5, 1.0, 1.0]); // score == threshold: not flagged
        monitor.on_residual(1, 1, &[9.0, 1.0, 1.0]); // large positive logit: flagged
        let state = state.lock().unwrap();
        assert_eq!(state.scored, 2);
        assert_eq!(state.flags.len(), 1);
        assert_eq!(state.flags[0].position, 1);
    }

    #[test]
    fn monitor_ignores_other_layers_and_width_mismatches() {
        let (mut monitor, state) = ProbeMonitor::new(Arc::new(test_probe(0.5)));
        monitor.on_residual(0, 0, &[1.5, 1.0, 1.0]); // wrong layer
        monitor.on_residual(1, 0, &[1.5, 1.0]); // wrong width
        assert_eq!(state.lock().unwrap().scored, 0);
    }
}
