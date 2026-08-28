/*
 * probe — in-band linear probe monitor over the residual-stream tap.
 *
 * Loads the probe artifact that interp/probes/export.py writes -- a
 * safetensors image of the f32 vectors weights/mu/sigma, and a JSON sidecar
 * beside it holding the scalars (name, model_hash, layer, site, bias,
 * threshold, width) -- and scores every position of the probe's layer as it
 * flows through the forward pass. The sidecar is mandatory: without it the
 * tensors name no layer and no decision threshold, so a probe missing its
 * sidecar is unusable rather than merely unchecked. That is the opposite of
 * the control vector's sibling manifest.json, which carries only an identity
 * check and whose absence is a warning.
 *
 * The scoring math mirrors interp/probes/score.py exactly:
 * sigmoid(((x - mu) / sigma) . w + bias), flagged when strictly above the
 * trained threshold. Probe parameters must be finite at load, and a score
 * that is still not a number is flagged rather than passed as clean, since
 * no comparison against a threshold can clear it. Observation only — the
 * monitor never alters the stream, so monitored and unmonitored runs
 * produce identical logits.
 */

use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

use super::cpu::{Site, Tap};
use crate::checkpoint::Image;

/// The sidecar layout this reader understands, bumped on a breaking change.
pub const SCHEMA_VERSION: u32 = 1;

/// The sidecar's tap site when it names none.
///
/// Every probe written before the key existed was trained at the block output,
/// which is where the seam has always fired; an absent `site` is that fact and
/// not an unknown one.
const DEFAULT_SITE: Site = Site::LayerOutput;

/// The three parameter vectors, in the order the sidecar's `width` describes.
const TENSORS: [&str; 3] = ["weights", "mu", "sigma"];

/// The JSON beside a probe's safetensors image.
///
/// Unknown fields are accepted: the envelope from `common/reports.py` puts
/// `schema_version`, `artifact`, `timestamp` and `git_sha` at the top level
/// beside these, and a writer that adds a field must not break a reader that
/// does not need it. What the reader will not accept is a *wrong* one, so the
/// two envelope keys that identify the file are deserialized and checked.
#[derive(Deserialize)]
struct Sidecar {
    schema_version: u32,
    artifact: String,
    name: String,
    model_hash: String,
    layer: usize,
    bias: f32,
    threshold: f32,
    /// Restated from the tensors deliberately: it is the sidecar's one
    /// falsifiable claim about the image beside it. A sidecar paired with the
    /// wrong `.safetensors` is otherwise loadable, and a probe scored at the
    /// wrong width is a truncated dot product returning a plausible number.
    width: usize,
    site: Option<String>,
}

/// A trained logistic-regression probe: standardization statistics plus a
/// weight vector over one decoder layer's residual stream.
#[derive(Debug)]
pub struct Probe {
    pub name: String,
    pub model_hash: String,
    /// 0-based decoder layer whose residual the probe scores, at `site`.
    pub layer: usize,
    /// The tap site the probe was trained on, from the sidecar.
    pub site: Site,
    pub threshold: f32,
    weights: Vec<f32>,
    mu: Vec<f32>,
    sigma: Vec<f32>,
    bias: f32,
}

impl Probe {
    /// Read a probe: the safetensors image at `path`, and the JSON sidecar
    /// beside it at the same path with a `.json` extension.
    ///
    /// The sidecar is required, and its absence names the path that was
    /// looked for. A probe copied away from its sidecar carries the vectors
    /// but not the layer, the site or the threshold, and there is no default
    /// for any of those that would be safer than refusing.
    pub fn load(path: &Path) -> Result<Probe> {
        let sidecar_path = path.with_extension("json");
        let text = std::fs::read_to_string(&sidecar_path).with_context(|| {
            format!(
                "read the probe sidecar {}: a probe is its safetensors image \
                 plus the JSON beside it",
                sidecar_path.display()
            )
        })?;
        let sidecar: Sidecar = serde_json::from_str(&text)
            .with_context(|| format!("parse {}", sidecar_path.display()))?;
        let bytes =
            std::fs::read(path).with_context(|| format!("read probe {}", path.display()))?;
        Probe::from_parts(sidecar, bytes, &path.display().to_string())
    }

    /// The reader proper, over bytes already in hand.
    ///
    /// Split from `load` so the tests exercise the validation without writing
    /// a temporary directory for every case, and so a caller holding an image
    /// from somewhere other than a file can still use it.
    fn from_parts(sidecar: Sidecar, image: Vec<u8>, name: &str) -> Result<Probe> {
        ensure!(
            sidecar.artifact == "probe",
            "{name}: the sidecar names artifact {:?}, not \"probe\"",
            sidecar.artifact
        );
        ensure!(
            sidecar.schema_version == SCHEMA_VERSION,
            "{name}: sidecar schema_version {} is not the {SCHEMA_VERSION} this \
             reader understands",
            sidecar.schema_version
        );
        let image = Image::parse(image, name)?;
        let [weights, mu, sigma] = TENSORS.map(|t| image.vector(t));
        let (weights, mu, sigma) = (weights?, mu?, sigma?);
        ensure!(
            weights.len() == mu.len() && mu.len() == sigma.len() && !weights.is_empty(),
            "probe tensors disagree on width: weights {}, mu {}, sigma {}",
            weights.len(),
            mu.len(),
            sigma.len()
        );
        ensure!(
            weights.len() == sidecar.width,
            "{name}: the sidecar declares width {} but the tensors are {} wide, \
             so the sidecar and the image do not belong together",
            sidecar.width,
            weights.len()
        );
        // A non-finite parameter poisons every score, and a NaN score fails
        // the strict threshold comparison, so the monitor would report every
        // position scored and none flagged — the same summary a clean run
        // prints. Refuse the file rather than monitor with it.
        for (tensor, values) in TENSORS.iter().zip([&weights, &mu, &sigma]) {
            if let Some(i) = values.iter().position(|v| !v.is_finite()) {
                bail!("{tensor} must be finite, got {} at index {i}", values[i]);
            }
        }
        ensure!(
            sigma.iter().all(|v| *v > 0.0),
            "sigma must be strictly positive"
        );
        ensure!(
            sidecar.threshold.is_finite(),
            "threshold must be finite, got {}",
            sidecar.threshold
        );
        ensure!(
            sidecar.bias.is_finite(),
            "bias must be finite, got {}",
            sidecar.bias
        );
        let site = match sidecar.site.as_deref() {
            Some(s) => Site::from_str(s).with_context(|| format!("{name}: probe site"))?,
            None => DEFAULT_SITE,
        };
        Ok(Probe {
            name: sidecar.name,
            model_hash: sidecar.model_hash,
            layer: sidecar.layer,
            site,
            threshold: sidecar.threshold,
            weights,
            mu,
            sigma,
            bias: sidecar.bias,
        })
    }

    pub fn width(&self) -> usize {
        self.weights.len()
    }

    /// Sigmoid probe score for one residual row, the score.py math.
    ///
    /// The width check is an error rather than a `debug_assert`: a release
    /// build compiles the assert out, and `zip` then scores a truncated dot
    /// product that looks like an ordinary score. One length comparison per
    /// scored position is nothing beside the dot product it guards.
    pub fn score(&self, x: &[f32]) -> Result<f32> {
        ensure!(
            x.len() == self.weights.len(),
            "residual row of width {} does not match probe width {}",
            x.len(),
            self.weights.len()
        );
        let logit: f32 = x
            .iter()
            .zip(&self.mu)
            .zip(&self.sigma)
            .zip(&self.weights)
            .map(|(((x, mu), sigma), w)| (x - mu) / sigma * w)
            .sum::<f32>()
            + self.bias;
        Ok(1.0 / (1.0 + (-logit).exp()))
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
    /// Every score, in the order the positions were computed, when the
    /// monitor was asked to keep them.
    ///
    /// Empty otherwise, which is what it has always been: a flag-only record
    /// answers "did anything cross the threshold" and cannot answer "what was
    /// the probe doing while the model wrote this", because the second
    /// question is about the clean positions too. Kept behind a flag because
    /// the cost is proportional to the run and only an alignment artifact
    /// reads it.
    pub scores: Vec<Flag>,
}

/// Tap implementation scoring the probe's layer at every position. Other
/// layers cost one integer comparison.
pub struct ProbeMonitor {
    probe: Arc<Probe>,
    state: Arc<Mutex<MonitorState>>,
    keep_scores: bool,
}

impl ProbeMonitor {
    pub fn new(probe: Arc<Probe>) -> (ProbeMonitor, Arc<Mutex<MonitorState>>) {
        ProbeMonitor::with_scores(probe, false)
    }

    /// A monitor that also keeps every clean score, for a caller writing an
    /// alignment artifact. `new` is the flag-only behaviour and stays the
    /// default, so a run that only wants `probe-flags.jsonl` is unchanged.
    pub fn with_scores(
        probe: Arc<Probe>,
        keep_scores: bool,
    ) -> (ProbeMonitor, Arc<Mutex<MonitorState>>) {
        let state = Arc::new(Mutex::new(MonitorState::default()));
        (
            ProbeMonitor {
                probe,
                state: Arc::clone(&state),
                keep_scores,
            },
            state,
        )
    }
}

impl Tap for ProbeMonitor {
    fn on_residual(&mut self, layer: usize, site: Site, position: usize, x: &[f32]) {
        if site != self.probe.site || layer != self.probe.layer {
            return;
        }
        // A row of another width belongs to a model this probe was not
        // trained on; `score` refuses it and the position goes unscored.
        let Ok(score) = self.probe.score(x) else {
            return;
        };
        let mut state = self.state.lock().expect("monitor state lock");
        state.scored += 1;
        if self.keep_scores {
            state.scores.push(Flag { position, score });
        }
        // Parameters are finite by load-time check, so a non-finite score
        // means a non-finite residual. `score > threshold` is false for NaN,
        // so leaving it unflagged would report the position clean: a monitor
        // that could not score a position must not claim it passed.
        if !score.is_finite() || score > self.probe.threshold {
            state.flags.push(Flag { position, score });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sidecar `interp/probes/export.py` writes, as JSON text, with the
    /// scalars under test. Written as text rather than built through serde so
    /// the tests exercise the same parse a real file takes.
    fn sidecar_json(bias: f32, width: usize, site: Option<&str>) -> String {
        let site = site
            .map(|s| format!(", \"site\": \"{s}\""))
            .unwrap_or_default();
        format!(
            r#"{{"schema_version": {SCHEMA_VERSION}, "artifact": "probe",
               "timestamp": "2026-01-01T00:00:00+0000", "git_sha": "abcdef0",
               "name": "test", "model_hash": "hash", "layer": 1,
               "bias": {bias}, "threshold": 0.5, "width": {width}{site}}}"#
        )
    }

    /// A probe assembled from parts, the way `load` does after reading them.
    fn load_parts(
        weights: &[f32],
        mu: &[f32],
        sigma: &[f32],
        bias: f32,
        width: usize,
        site: Option<&str>,
    ) -> Result<Probe> {
        let image = crate::synth::build_f32_safetensors(&[
            ("weights", weights),
            ("mu", mu),
            ("sigma", sigma),
        ])?;
        let sidecar = serde_json::from_str(&sidecar_json(bias, width, site))?;
        Probe::from_parts(sidecar, image, "test-probe.safetensors")
    }

    /// The common case: a well-formed probe of the given vectors.
    fn load(weights: &[f32], mu: &[f32], sigma: &[f32], bias: f32) -> Result<Probe> {
        load_parts(weights, mu, sigma, bias, weights.len(), None)
    }

    fn test_probe(threshold: f32) -> Probe {
        Probe {
            name: "test".into(),
            model_hash: "hash".into(),
            layer: 1,
            site: Site::LayerOutput,
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
        let score = probe.score(&[1.5, 1.0, 1.0]).unwrap();
        let expected = 1.0 / (1.0 + (-0.25f32).exp());
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn monitor_flags_only_scores_strictly_above_threshold() {
        let mut probe = test_probe(0.0);
        probe.threshold = probe.score(&[1.5, 1.0, 1.0]).unwrap(); // exact score of position 0
        let (mut monitor, state) = ProbeMonitor::new(Arc::new(probe));
        monitor.on_residual(1, Site::LayerOutput, 0, &[1.5, 1.0, 1.0]); // score == threshold: not flagged
        monitor.on_residual(1, Site::LayerOutput, 1, &[9.0, 1.0, 1.0]); // large positive logit: flagged
        let state = state.lock().unwrap();
        assert_eq!(state.scored, 2);
        assert_eq!(state.flags.len(), 1);
        assert_eq!(state.flags[0].position, 1);
    }

    #[test]
    fn monitor_ignores_other_layers_and_width_mismatches() {
        let (mut monitor, state) = ProbeMonitor::new(Arc::new(test_probe(0.5)));
        monitor.on_residual(0, Site::LayerOutput, 0, &[1.5, 1.0, 1.0]); // wrong layer
        monitor.on_residual(1, Site::LayerOutput, 0, &[1.5, 1.0]); // wrong width
        assert_eq!(state.lock().unwrap().scored, 0);
    }

    #[test]
    fn a_finite_probe_loads_and_scores() {
        let probe = load(&[1.0, -2.0], &[0.5, 0.0], &[1.0, 2.0], 0.25).expect("finite probe");
        assert_eq!(probe.width(), 2);
        assert_eq!(probe.name, "test");
        assert_eq!(probe.layer, 1);
        assert!(probe.score(&[1.5, 1.0]).unwrap().is_finite());
    }

    #[test]
    fn load_refuses_a_non_finite_probe_tensor() {
        // A non-finite parameter makes every score NaN, and NaN fails the
        // strict threshold comparison, so the run would report every position
        // scored and none flagged — exactly what a clean run reports.
        for (name, weights, mu, sigma) in [
            (
                "weights",
                vec![1.0, f32::NAN],
                vec![0.0, 0.0],
                vec![1.0, 1.0],
            ),
            (
                "mu",
                vec![1.0, 1.0],
                vec![0.0, f32::INFINITY],
                vec![1.0, 1.0],
            ),
            (
                "sigma",
                vec![1.0, 1.0],
                vec![0.0, 0.0],
                vec![1.0, f32::INFINITY],
            ),
        ] {
            let err = load(&weights, &mu, &sigma, 0.25).expect_err("expected error");
            assert!(
                err.to_string().contains(&format!("{name} must be finite")),
                "{err}"
            );
        }
    }

    #[test]
    fn load_refuses_a_non_positive_sigma() {
        let err = load(&[1.0, 1.0], &[0.0, 0.0], &[1.0, 0.0], 0.25).expect_err("expected error");
        assert!(
            err.to_string().contains("sigma must be strictly positive"),
            "{err}"
        );
    }

    #[test]
    fn load_refuses_a_non_finite_bias() {
        // Reached by constructing the sidecar directly, because JSON has no
        // spelling for NaN: the test below covers the other half, where a
        // writer emits Python's non-standard `NaN` literal and the parse is
        // what refuses it. Both paths must refuse; only together do they
        // cover a scalar that would poison every score.
        let sidecar = Sidecar {
            schema_version: SCHEMA_VERSION,
            artifact: "probe".into(),
            name: "test".into(),
            model_hash: "hash".into(),
            layer: 1,
            bias: f32::NAN,
            threshold: 0.5,
            width: 2,
            site: None,
        };
        let image = crate::synth::build_f32_safetensors(&[
            ("weights", &[1.0, 1.0][..]),
            ("mu", &[0.0, 0.0][..]),
            ("sigma", &[1.0, 1.0][..]),
        ])
        .unwrap();
        let err = Probe::from_parts(sidecar, image, "p.safetensors").expect_err("expected error");
        assert!(err.to_string().contains("bias must be finite"), "{err}");
    }

    #[test]
    fn a_sidecar_spelling_a_scalar_as_nan_is_refused_at_the_parse() {
        // `json.dump(float("nan"))` writes a bare `NaN`, which is not JSON.
        // serde_json refuses it, so a probe whose threshold never fitted
        // cannot be loaded and silently flag nothing.
        let text = sidecar_json(0.25, 2, None).replace("\"threshold\": 0.5", "\"threshold\": NaN");
        assert!(
            serde_json::from_str::<Sidecar>(&text).is_err(),
            "a NaN threshold parsed as a number"
        );
    }

    #[test]
    fn a_sidecar_declaring_another_width_is_refused() {
        // The pairing check: these two files are individually well-formed and
        // only their combination is wrong. Without it the probe would score a
        // truncated dot product and return a number that looks ordinary.
        let err = load_parts(&[1.0, 1.0], &[0.0, 0.0], &[1.0, 1.0], 0.25, 3, None)
            .expect_err("expected error");
        assert!(err.to_string().contains("do not belong together"), "{err}");
    }

    #[test]
    fn a_sidecar_from_another_artifact_or_another_version_is_refused() {
        let image = crate::synth::build_f32_safetensors(&[
            ("weights", &[1.0][..]),
            ("mu", &[0.0][..]),
            ("sigma", &[1.0][..]),
        ])
        .unwrap();
        for (text, want) in [
            (
                sidecar_json(0.25, 1, None)
                    .replace(r#""artifact": "probe""#, r#""artifact": "controlvector""#),
                "not \"probe\"",
            ),
            (
                sidecar_json(0.25, 1, None).replace(
                    &format!(r#""schema_version": {SCHEMA_VERSION}"#),
                    r#""schema_version": 99"#,
                ),
                "is not the 1 this reader understands",
            ),
        ] {
            let sidecar = serde_json::from_str(&text).unwrap();
            let err = Probe::from_parts(sidecar, image.clone(), "p.safetensors")
                .expect_err("expected error");
            assert!(err.to_string().contains(want), "{err}");
        }
    }

    #[test]
    fn the_site_comes_from_the_sidecar_and_defaults_when_absent() {
        // The gap this format closes: the GGUF named the layer and left the
        // site to a constant on this type, so a probe file said half of where
        // it belonged.
        let named = load_parts(&[1.0], &[0.0], &[1.0], 0.25, 1, Some("layer_output"))
            .expect("a named site loads");
        assert_eq!(named.site, Site::LayerOutput);

        let absent = load(&[1.0], &[0.0], &[1.0], 0.25).expect("an absent site loads");
        assert_eq!(absent.site, DEFAULT_SITE);

        let err = load_parts(&[1.0], &[0.0], &[1.0], 0.25, 1, Some("mlp_output"))
            .expect_err("an unknown site is refused");
        // `{:#}` renders the whole chain: `Site::from_str`'s message is the
        // cause under the context naming which file failed.
        assert!(format!("{err:#}").contains("unknown tap site"), "{err:#}");
    }

    #[test]
    fn a_probe_without_its_sidecar_names_the_path_it_looked_for() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("probe.safetensors");
        std::fs::write(
            &path,
            crate::synth::build_f32_safetensors(&[("weights", &[1.0][..])]).unwrap(),
        )
        .unwrap();
        let err = Probe::load(&path).expect_err("expected error");
        assert!(
            err.to_string().contains("probe.json"),
            "the error does not name the sidecar it wanted: {err}"
        );
    }

    #[test]
    fn a_written_probe_round_trips_through_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("probe.safetensors");
        std::fs::write(
            &path,
            crate::synth::build_f32_safetensors(&[
                ("weights", &[1.0, -2.0][..]),
                ("mu", &[0.5, 0.0][..]),
                ("sigma", &[1.0, 2.0][..]),
            ])
            .unwrap(),
        )
        .unwrap();
        std::fs::write(path.with_extension("json"), sidecar_json(0.25, 2, None)).unwrap();

        let probe = Probe::load(&path).expect("round trip");
        assert_eq!(probe.model_hash, "hash");
        assert_eq!(probe.threshold, 0.5);
        assert_eq!(probe.width(), 2);
    }

    #[test]
    fn a_non_finite_score_is_flagged_rather_than_counted_as_clean() {
        // Parameters are finite by load-time check, so a NaN score means a
        // NaN residual. `score > threshold` is false for NaN, so an unflagged
        // position would read as clean in the "N scored, 0 flagged" summary.
        let (mut monitor, state) = ProbeMonitor::new(Arc::new(test_probe(0.5)));
        monitor.on_residual(1, Site::LayerOutput, 3, &[f32::NAN, 1.0, 1.0]);
        let state = state.lock().unwrap();
        assert_eq!(state.scored, 1);
        assert_eq!(state.flags.len(), 1);
        assert_eq!(state.flags[0].position, 3);
        assert!(state.flags[0].score.is_nan());
    }

    #[test]
    fn score_refuses_a_residual_row_of_another_width() {
        // A release build compiles a `debug_assert` out and `zip` then scores
        // a truncated dot product, which looks like an ordinary score.
        let probe = test_probe(0.5);
        let err = probe.score(&[1.5, 1.0]).expect_err("expected error");
        assert!(
            err.to_string().contains("does not match probe width"),
            "{err}"
        );
    }
}
