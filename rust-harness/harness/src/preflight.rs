/*
 * preflight — what a run would fail on, answered without loading weights.
 *
 * Every check here already exists inside `build_generator`, and every one of
 * them runs there *after* `CpuModel::load` has spent ninety seconds and 14 GiB
 * finding out the model is fine. A probe missing its JSON sidecar, a control
 * vector fitted on other weights, an unwritable log directory: all knowable
 * from the checkpoint's config and safetensors header, which
 * `runtime::checkpoint` reads without touching a weight byte.
 *
 * Two properties this module is built around:
 *
 * `checks` returns data and prints nothing. The caller renders. That is why
 * `compare_identity` exists beside `check_model_identity` -- the latter warns
 * inline for the run path, and a preflight that called it would emit stray
 * stderr beside its own report.
 *
 * A check whose input failed reports `Skip`, not `Error`. When `config.json`
 * does not parse there is no embedding width to compare a probe against, and
 * saying so is honest where a second `Error` would invent a second failure
 * from one cause.
 *
 * Not a second `checkpoint-info`. The one question the two share -- can the
 * CPU backend serve these weights -- is delegated to the same
 * `cpu::check_supported_checkpoint` rather than re-derived, so they cannot
 * disagree. `checkpoint-info` inspects a checkpoint and prints its tensor
 * table; this answers whether one particular invocation would start.
 *
 * Deliberately not covered, because it needs a loaded model rather than a
 * header: `set_control_vector`'s width and block-count validation
 * (`runtime/src/model/cpu.rs`). Transcribing those rules here would create a
 * second copy to drift. `--status` narrows the failure surface; it does not
 * promise to eliminate it.
 */

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Sentinel an interp artifact carries when the weights it was fitted on
/// have no recorded identity (`interp/probes/capture.py` defaults
/// `model_hash` to it, and the exporters pass it through).
pub const UNKNOWN_MODEL_HASH: &str = "unknown";

/// How one check came out.
///
/// `Skip` is the fourth because three cannot express "not checked". It is
/// distinct from `Warn` -- a warning is something observed, a skip is
/// something not looked at -- and the exit code ignores both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    Warn,
    Error,
    Skip,
}

/// One check and what it found.
///
/// `detail` may be multi-line. The renderer puts the first line on the tagged
/// line and indents the rest untagged.
#[derive(Clone, Debug)]
pub struct Check {
    pub name: &'static str,
    pub outcome: Outcome,
    pub detail: String,
}

impl Check {
    fn new(name: &'static str, outcome: Outcome, detail: impl Into<String>) -> Check {
        Check {
            name,
            outcome,
            detail: detail.into(),
        }
    }
}

/// The resolved command line, as far as parsing got.
///
/// Not a `RunHeader`: that is built from a loaded `CpuModel` and every field
/// of it costs the load this module exists to avoid. These are the values
/// `main` already holds before it decides to load anything.
pub struct Request<'a> {
    pub model: &'a Path,
    pub probe: Option<&'a Path>,
    pub control_vector: Option<&'a Path>,
    pub control_scale: f32,
    pub control_layer_range: Option<(usize, usize)>,
    pub interp_log: Option<&'a Path>,
    pub text_only: bool,
    pub deep: bool,
}

/// The result of comparing an artifact's recorded model hash to the served one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Identity {
    /// The artifact records the `unknown` sentinel; there is nothing to compare.
    Sentinel,
    Match,
    Differs {
        artifact: String,
        served: String,
    },
}

/// Compare an artifact's recorded model hash against the served weights.
///
/// The pure half of `check_model_identity`: it decides, and the caller
/// reports. The served digest arrives as a closure because producing it
/// hashes multi-gigabyte weights, a cost the sentinel case must not pay.
pub fn compare_identity(
    artifact_hash: &str,
    served_sha256: impl FnOnce() -> Result<String>,
) -> Result<Identity> {
    if artifact_hash == UNKNOWN_MODEL_HASH {
        return Ok(Identity::Sentinel);
    }
    let served = served_sha256()?;
    Ok(match artifact_hash == served {
        true => Identity::Match,
        false => Identity::Differs {
            artifact: artifact_hash.to_string(),
            served,
        },
    })
}

/// Refuse an interp artifact fitted on weights other than the served ones.
///
/// A probe scored against a different model, or a control vector added to
/// its residual stream, produces meaningless output indistinguishable from
/// a genuine flag or steer, so the pairing is refused the way every Python
/// consumer refuses it (`interp/probes/registry.py` `require_match`,
/// `interp/steering/wiring.py`). Only two real hashes can disagree: the
/// `unknown` sentinel downgrades to a skipped check, which is what keeps
/// Python-captured artifacts loading.
///
/// The run path's renderer over `compare_identity`. It warns inline, which is
/// why `checks` uses the comparison directly instead.
pub fn check_model_identity(
    artifact: &str,
    artifact_hash: &str,
    served_sha256: impl FnOnce() -> Result<String>,
) -> Result<()> {
    match compare_identity(artifact_hash, served_sha256)? {
        Identity::Sentinel => {
            runtime::status::warn(&format!(
                "Model identity: {artifact} records model hash \
                 \"{UNKNOWN_MODEL_HASH}\", check skipped"
            ));
            Ok(())
        }
        Identity::Match => Ok(()),
        Identity::Differs {
            artifact: a,
            served,
        } => anyhow::bail!(
            "{artifact} was trained on a different model: {artifact} model_hash \
             {a}, served model sha256 {served}"
        ),
    }
}

/// The model hash recorded beside a control vector, when its artifact
/// directory is intact. The image itself carries tensors and nothing else,
/// so the sibling `manifest.json` the exporter writes
/// (`interp/steering/export.py`) is the only hash available. A vector copied
/// away from its manifest, or one whose manifest is unreadable, stays
/// loadable and unchecked — the deliberate asymmetry with the probe's
/// mandatory sidecar, which `schemas/control_vector_artifact.json` records.
pub fn control_vector_model_hash(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path.parent()?.join("manifest.json")).ok()?;
    let manifest: serde_json::Value = serde_json::from_str(&text).ok()?;
    manifest.get("model_hash")?.as_str().map(str::to_string)
}

/// True when any check found something that would stop the run.
pub fn failed(checks: &[Check]) -> bool {
    checks.iter().any(|c| c.outcome == Outcome::Error)
}

/// Run every check the request calls for.
pub fn checks(req: &Request) -> Vec<Check> {
    let weights = runtime::checkpoint::resolve(req.model)
        .map(|dir| dir.join(runtime::checkpoint::WEIGHTS_FILE));
    checks_with_digest(req, || match &weights {
        Ok(path) => {
            runtime::digest::sha256_file(path).with_context(|| format!("hash {}", path.display()))
        }
        // Flattened to a message: the resolve failure is already reported by
        // the first check, and this arm exists so the digest producer is
        // total rather than to report it twice.
        Err(e) => anyhow::bail!("{e:#}"),
    })
}

/// `checks`, with the served digest injected.
///
/// Split out so a test can count how many times the weights are hashed. With
/// both a probe and a control vector under `--deep` the naive shape reads
/// E4B's weights twice, ~28 GiB, and no assertion over the returned vector
/// would notice.
fn checks_with_digest<F>(req: &Request, mut digest: F) -> Vec<Check>
where
    F: FnMut() -> Result<String>,
{
    let mut out: Vec<Check> = Vec::new();
    let mut cache: Option<std::result::Result<String, String>> = None;

    let resolved = runtime::checkpoint::resolve(req.model);
    out.push(match &resolved {
        Ok(dir) => Check::new("model directory", Outcome::Ok, dir.display().to_string()),
        Err(e) => Check::new("model directory", Outcome::Error, format!("{e:#}")),
    });

    let files = resolved
        .as_ref()
        .ok()
        .map(|_| runtime::checkpoint::CheckpointFiles::read(req.model));
    match &files {
        None => out.push(Check::new(
            "checkpoint files",
            Outcome::Skip,
            "not checked: the model path did not resolve",
        )),
        Some(Ok(f)) => out.push(Check::new(
            "checkpoint files",
            Outcome::Ok,
            format!(
                "{} and {} present, header names {} tensors",
                runtime::checkpoint::CONFIG_FILE,
                runtime::checkpoint::WEIGHTS_FILE,
                runtime::checkpoint::parse(f)
                    .map(|ck| ck.names().len())
                    .unwrap_or(0)
            ),
        )),
        Some(Err(e)) => out.push(Check::new(
            "checkpoint files",
            Outcome::Error,
            format!("{e:#}"),
        )),
    }

    // Parsed once and borrowed by the config, compatibility and probe-geometry
    // checks. `parse` leaves `text_only` false so a caller that says nothing
    // gets the refusal rather than a model serving a third of the file; the
    // preflight must ask the question the run would ask, which is the one the
    // flag decides.
    let parsed = match &files {
        Some(Ok(f)) => Some(runtime::checkpoint::parse(f).map(|mut ck| {
            ck.text_only = req.text_only;
            ck
        })),
        _ => None,
    };
    match &parsed {
        None => out.push(Check::new(
            "config.json",
            Outcome::Skip,
            "not checked: the checkpoint files were unreadable",
        )),
        Some(Ok(ck)) => out.push(Check::new(
            "config.json",
            Outcome::Ok,
            format!(
                "{} layers, hidden size {}, context {}",
                ck.config.num_hidden_layers,
                ck.config.hidden_size,
                ck.config.max_position_embeddings
            ),
        )),
        Some(Err(e)) => out.push(Check::new("config.json", Outcome::Error, format!("{e:#}"))),
    }

    out.push(match &parsed {
        Some(Ok(ck)) => {
            match runtime::model::cpu::ModelSpec::from_checkpoint(&ck.config)
                .and_then(|spec| runtime::model::cpu::check_supported_checkpoint(ck, &spec))
            {
                Ok(()) => Check::new("backend compatibility", Outcome::Ok, "supported"),
                Err(e) => Check::new("backend compatibility", Outcome::Error, format!("{e:#}")),
            }
        }
        _ => Check::new(
            "backend compatibility",
            Outcome::Skip,
            "not checked: the checkpoint did not parse",
        ),
    });

    let probe = req.probe.map(runtime::model::probe::Probe::load);
    match &probe {
        None => out.push(Check::new("probe", Outcome::Skip, "no --probe given")),
        Some(Ok(p)) => out.push(Check::new(
            "probe",
            Outcome::Ok,
            format!(
                "{}, layer {}, site {:?}, width {}",
                p.name,
                p.layer,
                p.site,
                p.width()
            ),
        )),
        Some(Err(e)) => out.push(Check::new("probe", Outcome::Error, format!("{e:#}"))),
    }

    out.push(match (&probe, &parsed) {
        (Some(Ok(p)), Some(Ok(ck))) => {
            let mut faults: Vec<String> = Vec::new();
            if p.width() != ck.config.hidden_size {
                faults.push(format!(
                    "probe width {} does not match model embedding dimension {}",
                    p.width(),
                    ck.config.hidden_size
                ));
            }
            if p.layer >= ck.config.num_hidden_layers {
                faults.push(format!(
                    "probe layer {} out of range for a {}-layer model",
                    p.layer, ck.config.num_hidden_layers
                ));
            }
            match faults.is_empty() {
                true => Check::new(
                    "probe geometry",
                    Outcome::Ok,
                    format!("width {} and layer {} fit the model", p.width(), p.layer),
                ),
                false => Check::new("probe geometry", Outcome::Error, faults.join("\n")),
            }
        }
        (None, _) => Check::new("probe geometry", Outcome::Skip, "no --probe given"),
        _ => Check::new(
            "probe geometry",
            Outcome::Skip,
            "not checked: the probe or the config was unreadable",
        ),
    });

    out.push(match &probe {
        Some(Ok(p)) => identity_check(
            "probe identity",
            &p.model_hash,
            req.deep,
            &mut cache,
            &mut digest,
        ),
        Some(Err(_)) => Check::new(
            "probe identity",
            Outcome::Skip,
            "not checked: the probe did not load",
        ),
        None => Check::new("probe identity", Outcome::Skip, "no --probe given"),
    });

    let vector = req.control_vector.map(|path| {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read control vector {}", path.display()))?;
        let image = runtime::checkpoint::Image::parse(bytes, "control vector")
            .context("parse control-vector image")?;
        runtime::model::cpu::ControlVector::from_image(
            &image,
            req.control_scale,
            req.control_layer_range,
        )
        .context("load control vector")
    });
    match &vector {
        None => out.push(Check::new(
            "control vector",
            Outcome::Skip,
            "no --control-vector given",
        )),
        Some(Ok(cv)) => out.push(Check::new(
            "control vector",
            Outcome::Ok,
            format!("width {}, scale {}", cv.width(), req.control_scale),
        )),
        Some(Err(e)) => out.push(Check::new(
            "control vector",
            Outcome::Error,
            format!("{e:#}"),
        )),
    }

    out.push(match (&vector, req.control_vector) {
        (Some(Ok(_)), Some(path)) => match control_vector_model_hash(path) {
            Some(hash) => identity_check(
                "control vector identity",
                &hash,
                req.deep,
                &mut cache,
                &mut digest,
            ),
            // Warn, never Error: a vector copied away from its manifest stays
            // loadable and unchecked, and the preflight must report the run
            // that would actually happen.
            None => Check::new(
                "control vector identity",
                Outcome::Warn,
                format!(
                    "no readable manifest.json beside {}, the vector loads unchecked",
                    path.display()
                ),
            ),
        },
        (Some(Err(_)), _) => Check::new(
            "control vector identity",
            Outcome::Skip,
            "not checked: the control vector did not load",
        ),
        _ => Check::new(
            "control vector identity",
            Outcome::Skip,
            "no --control-vector given",
        ),
    });

    out.push(match req.interp_log {
        Some(dir) => {
            let (outcome, detail) = directory_state(dir);
            Check::new(
                "interp log directory",
                outcome,
                format!("{detail} (written only under --serve)"),
            )
        }
        None => Check::new(
            "interp log directory",
            Outcome::Skip,
            "no --interp-log given",
        ),
    });

    // Resolution mirrors `obs::RunLog::begin`. An unwritable root is capped at
    // Warn on purpose: the log degrades to a silent no-op handle by contract,
    // so the run proceeds and calling it an Error would claim otherwise.
    let obs_root = std::env::var_os("OBS_LOG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/mnt/ai-data/logs"))
        .join("harness");
    let (outcome, detail) = directory_state(&obs_root);
    out.push(match outcome {
        Outcome::Ok => Check::new("obs log directory", Outcome::Ok, detail),
        _ => Check::new(
            "obs log directory",
            Outcome::Warn,
            format!("{detail}; run records degrade to a no-op"),
        ),
    });

    out
}

/// One artifact's identity check, sharing the memoized served digest.
fn identity_check<F>(
    name: &'static str,
    artifact_hash: &str,
    deep: bool,
    cache: &mut Option<std::result::Result<String, String>>,
    digest: &mut F,
) -> Check
where
    F: FnMut() -> Result<String>,
{
    if !deep {
        return Check::new(
            name,
            Outcome::Skip,
            "not checked: hashing the weights is opt-in, pass --deep",
        );
    }
    if artifact_hash == UNKNOWN_MODEL_HASH {
        return Check::new(
            name,
            Outcome::Warn,
            format!("the artifact records model hash \"{UNKNOWN_MODEL_HASH}\", nothing to compare"),
        );
    }
    match served(cache, digest) {
        Err(e) => Check::new(name, Outcome::Error, e),
        Ok(served) if served == artifact_hash => Check::new(
            name,
            Outcome::Ok,
            format!("matches the served weights, sha256 {served}"),
        ),
        Ok(served) => Check::new(
            name,
            Outcome::Error,
            format!(
                "trained on a different model: artifact model_hash {artifact_hash}, \
                 served model sha256 {served}"
            ),
        ),
    }
}

/// The memoized served digest. Free function rather than a closure so both
/// identity checks can call it while it holds `&mut` state.
fn served<F>(
    cache: &mut Option<std::result::Result<String, String>>,
    digest: &mut F,
) -> std::result::Result<String, String>
where
    F: FnMut() -> Result<String>,
{
    if cache.is_none() {
        *cache = Some(digest().map_err(|e| format!("{e:#}")));
    }
    cache.clone().expect("filled immediately above")
}

/// Whether a directory is one the harness could write into.
///
/// Tested by writing. Permission bits and ownership do not settle it under
/// ACLs, a read-only mount, or a full filesystem, and the question here is
/// exactly whether the write would succeed.
///
/// An absent directory is not an error: the run creates it. What is checked
/// instead is the nearest existing ancestor, since a directory whose parent
/// cannot be written is one that will not appear.
fn directory_state(dir: &Path) -> (Outcome, String) {
    if !dir.exists() {
        let mut ancestor = dir.parent();
        while let Some(path) = ancestor {
            if path.exists() {
                return match write_probe(path) {
                    Ok(()) => (
                        Outcome::Warn,
                        format!("{} does not exist; the run creates it", dir.display()),
                    ),
                    Err(e) => (
                        Outcome::Error,
                        format!(
                            "{} cannot be created: {} is not writable: {e}",
                            dir.display(),
                            path.display()
                        ),
                    ),
                };
            }
            ancestor = path.parent();
        }
        return (
            Outcome::Error,
            format!("{}: no existing ancestor directory", dir.display()),
        );
    }
    if !dir.is_dir() {
        return (
            Outcome::Error,
            format!("{} exists and is not a directory", dir.display()),
        );
    }
    match write_probe(dir) {
        Ok(()) => (Outcome::Ok, format!("{} is writable", dir.display())),
        Err(e) => (
            Outcome::Error,
            format!("{} is not writable: {e}", dir.display()),
        ),
    }
}

/// Create and remove a uniquely named file, to settle writability by writing.
fn write_probe(dir: &Path) -> std::io::Result<()> {
    let path = dir.join(format!(".harness-preflight-{}", std::process::id()));
    std::fs::write(&path, b"")?;
    std::fs::remove_file(&path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime::digest::sha256_hex;

    /// The committed checkpoint every positive case is measured against.
    fn fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/tiny-gemma4")
    }

    fn request(model: &Path) -> Request<'_> {
        Request {
            model,
            probe: None,
            control_vector: None,
            control_scale: 1.0,
            control_layer_range: None,
            interp_log: None,
            text_only: false,
            deep: false,
        }
    }

    fn find<'a>(checks: &'a [Check], name: &str) -> &'a Check {
        checks
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no check named {name} among {checks:?}"))
    }

    // The three moved from main.rs unmodified, which is what says the move of
    // `check_model_identity` was a move.

    #[test]
    fn model_identity_accepts_an_artifact_hash_equal_to_the_served_digest() {
        let served = sha256_hex(b"gguf bytes");
        check_model_identity("probe", &served, || Ok(sha256_hex(b"gguf bytes")))
            .expect("an artifact fitted on the served weights is the matching case");
    }

    #[test]
    fn model_identity_refuses_a_mismatch_and_names_both_hashes() {
        let artifact = sha256_hex(b"weights the probe was trained on");
        let served = sha256_hex(b"weights actually loaded");
        let err = check_model_identity("probe", &artifact, || Ok(served.clone()))
            .expect_err("two real hashes that disagree are a refusal")
            .to_string();
        assert!(err.contains(&artifact), "{err}");
        assert!(err.contains(&served), "{err}");
    }

    #[test]
    fn model_identity_skips_the_unknown_sentinel_without_hashing_the_model() {
        let mut hashed = false;
        check_model_identity("probe", UNKNOWN_MODEL_HASH, || {
            hashed = true;
            Ok(String::new())
        })
        .expect("unknown is the capture default, not a mismatch");
        assert!(!hashed, "a skipped check must not hash the model");
    }

    #[test]
    fn control_vector_model_hash_reads_the_manifest_beside_the_vector() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vector = dir.path().join("vector.safetensors");
        assert_eq!(control_vector_model_hash(&vector), None);
        std::fs::write(
            dir.path().join("manifest.json"),
            r#"{"artifact": "control_vector", "model_hash": "deadbeef"}"#,
        )
        .expect("write manifest");
        assert_eq!(
            control_vector_model_hash(&vector),
            Some("deadbeef".to_string())
        );
    }

    #[test]
    fn a_missing_model_directory_is_one_error_naming_the_path() {
        let checks = checks(&request(Path::new("/absent/checkpoint")));
        let dir = find(&checks, "model directory");
        assert_eq!(dir.outcome, Outcome::Error, "{dir:?}");
        assert!(dir.detail.contains("/absent/checkpoint"), "{dir:?}");
        // Everything downstream is unmeasured, not failed: one cause, one error.
        assert_eq!(
            checks
                .iter()
                .filter(|c| c.outcome == Outcome::Error)
                .count(),
            1,
            "{checks:?}"
        );
    }

    #[test]
    fn the_committed_fixture_passes_every_check_that_needs_no_artifact() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model = fixture();
        let mut req = request(&model);
        req.interp_log = Some(dir.path());
        let checks = checks(&req);
        assert!(!failed(&checks), "{checks:?}");
        for name in [
            "model directory",
            "checkpoint files",
            "config.json",
            "backend compatibility",
        ] {
            assert_eq!(find(&checks, name).outcome, Outcome::Ok, "{checks:?}");
        }
    }

    #[test]
    fn a_checkpoint_without_its_config_skips_the_checks_that_would_need_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::copy(
            fixture().join(runtime::checkpoint::WEIGHTS_FILE),
            dir.path().join(runtime::checkpoint::WEIGHTS_FILE),
        )
        .expect("copy weights");
        let checks = checks(&request(dir.path()));
        assert_eq!(
            find(&checks, "checkpoint files").outcome,
            Outcome::Error,
            "{checks:?}"
        );
        // The point of the fourth outcome: one cause must not read as three
        // independent failures.
        for name in ["config.json", "backend compatibility"] {
            assert_eq!(find(&checks, name).outcome, Outcome::Skip, "{checks:?}");
        }
    }

    #[test]
    fn a_probe_without_its_json_sidecar_is_an_error_naming_the_sidecar_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let probe = dir.path().join("probe.safetensors");
        std::fs::write(
            &probe,
            runtime::synth::build_f32_safetensors(&[
                ("weights", &[0.1f32, 0.2]),
                ("mu", &[0.0f32, 0.0]),
                ("sigma", &[1.0f32, 1.0]),
            ])
            .expect("build probe image"),
        )
        .expect("write probe");
        let model = fixture();
        let mut req = request(&model);
        req.probe = Some(&probe);
        let checks = checks(&req);
        let check = find(&checks, "probe");
        assert_eq!(check.outcome, Outcome::Error, "{check:?}");
        assert!(check.detail.contains("probe.json"), "{check:?}");
    }

    #[test]
    fn a_probe_wider_than_the_model_reports_both_widths() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model = fixture();
        let hidden = hidden_size(&model);
        let probe = write_probe_artifact(dir.path(), hidden + 3, 0, UNKNOWN_MODEL_HASH);
        let mut req = request(&model);
        req.probe = Some(&probe);
        let checks = checks(&req);
        let check = find(&checks, "probe geometry");
        assert_eq!(check.outcome, Outcome::Error, "{check:?}");
        assert!(
            check.detail.contains(&(hidden + 3).to_string()),
            "{check:?}"
        );
        assert!(check.detail.contains(&hidden.to_string()), "{check:?}");
    }

    #[test]
    fn a_probe_layer_past_the_last_block_names_the_layer_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model = fixture();
        let layers = layer_count(&model);
        let probe =
            write_probe_artifact(dir.path(), hidden_size(&model), layers, UNKNOWN_MODEL_HASH);
        let mut req = request(&model);
        req.probe = Some(&probe);
        let checks = checks(&req);
        let check = find(&checks, "probe geometry");
        assert_eq!(check.outcome, Outcome::Error, "{check:?}");
        assert!(check.detail.contains(&layers.to_string()), "{check:?}");
    }

    #[test]
    fn identity_without_deep_names_the_flag_that_would_check_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model = fixture();
        let probe = write_probe_artifact(dir.path(), hidden_size(&model), 0, "abc123");
        let mut req = request(&model);
        req.probe = Some(&probe);
        let checks = checks(&req);
        let check = find(&checks, "probe identity");
        assert_eq!(check.outcome, Outcome::Skip, "{check:?}");
        assert!(check.detail.contains("--deep"), "{check:?}");
    }

    #[test]
    fn an_artifact_trained_on_other_weights_is_an_error_naming_both_hashes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model = fixture();
        let probe = write_probe_artifact(dir.path(), hidden_size(&model), 0, "abc123");
        let mut req = request(&model);
        req.probe = Some(&probe);
        req.deep = true;
        let checks = checks_with_digest(&req, || Ok("def456".to_string()));
        let check = find(&checks, "probe identity");
        assert_eq!(check.outcome, Outcome::Error, "{check:?}");
        assert!(check.detail.contains("abc123"), "{check:?}");
        assert!(check.detail.contains("def456"), "{check:?}");
    }

    #[test]
    fn an_unknown_model_hash_downgrades_the_identity_check_to_a_warning() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model = fixture();
        let probe = write_probe_artifact(dir.path(), hidden_size(&model), 0, UNKNOWN_MODEL_HASH);
        let mut req = request(&model);
        req.probe = Some(&probe);
        req.deep = true;
        let mut hashed = 0;
        let checks = checks_with_digest(&req, || {
            hashed += 1;
            Ok("def456".to_string())
        });
        assert_eq!(find(&checks, "probe identity").outcome, Outcome::Warn);
        assert_eq!(hashed, 0, "the sentinel must not hash the weights");
    }

    #[test]
    fn the_weights_are_hashed_once_for_a_probe_and_a_control_vector_together() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model = fixture();
        let probe = write_probe_artifact(dir.path(), hidden_size(&model), 0, "abc123");
        let vector = dir.path().join("vector.safetensors");
        std::fs::write(
            &vector,
            // `block.<index>`, the spelling `ControlVector::from_image` reads.
            // Written wrong the first time, and the test still passed: a
            // vector that fails to load skips its identity check, so the
            // hash count came from the probe alone. The assertions below are
            // what make the count mean what it says.
            runtime::synth::build_f32_safetensors(&[(
                "block.0",
                &vec![0.1f32; hidden_size(&model)],
            )])
            .expect("build vector image"),
        )
        .expect("write vector");
        std::fs::write(
            dir.path().join("manifest.json"),
            r#"{"artifact": "control_vector", "model_hash": "abc123"}"#,
        )
        .expect("write manifest");
        let mut req = request(&model);
        req.probe = Some(&probe);
        req.control_vector = Some(&vector);
        req.deep = true;
        let mut hashed = 0;
        let checks = checks_with_digest(&req, || {
            hashed += 1;
            Ok("abc123".to_string())
        });
        // Both artifacts must have reached their identity check, or the count
        // below is a count of one.
        for name in [
            "control vector",
            "probe identity",
            "control vector identity",
        ] {
            assert_eq!(find(&checks, name).outcome, Outcome::Ok, "{checks:?}");
        }
        assert_eq!(
            hashed, 1,
            "two artifacts must share one digest, not read the weights twice: {checks:?}"
        );
    }

    #[test]
    fn an_unwritable_obs_log_dir_never_raises_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("not-a-directory");
        std::fs::write(&file, b"").expect("write file");
        // A file where the log root should be is what `obs.rs` degrades on.
        let checks = temp_env_obs(&file, || checks(&request(&fixture())));
        let check = find(&checks, "obs log directory");
        assert_eq!(check.outcome, Outcome::Warn, "{check:?}");
        assert!(check.detail.contains("no-op"), "{check:?}");
    }

    #[test]
    fn no_two_checks_in_one_run_share_a_name() {
        let checks = checks(&request(&fixture()));
        let mut names: Vec<&str> = checks.iter().map(|c| c.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "{checks:?}");
        for check in &checks {
            assert!(!check.name.is_empty(), "{check:?}");
            assert!(!check.detail.is_empty(), "{check:?}");
        }
    }

    /// `hidden_size` from the fixture's own config, so the probe tests move
    /// with the fixture rather than pinning a number beside it.
    fn hidden_size(model: &Path) -> usize {
        let files = runtime::checkpoint::CheckpointFiles::read(model).expect("read fixture");
        runtime::checkpoint::parse(&files)
            .expect("parse fixture")
            .config
            .hidden_size
    }

    fn layer_count(model: &Path) -> usize {
        let files = runtime::checkpoint::CheckpointFiles::read(model).expect("read fixture");
        runtime::checkpoint::parse(&files)
            .expect("parse fixture")
            .config
            .num_hidden_layers
    }

    /// A probe in the layout `interp/probes/export.py` writes: the image and
    /// the mandatory sidecar beside it.
    fn write_probe_artifact(dir: &Path, width: usize, layer: usize, model_hash: &str) -> PathBuf {
        let path = dir.join("probe.safetensors");
        let weights: Vec<f32> = (0..width).map(|j| 0.1 * j as f32).collect();
        let mu = vec![0.0f32; width];
        let sigma = vec![1.0f32; width];
        std::fs::write(
            &path,
            runtime::synth::build_f32_safetensors(&[
                ("weights", &weights),
                ("mu", &mu),
                ("sigma", &sigma),
            ])
            .expect("build probe image"),
        )
        .expect("write probe");
        std::fs::write(
            path.with_extension("json"),
            format!(
                r#"{{"schema_version": 1, "artifact": "probe",
                   "timestamp": "2026-01-01T00:00:00+0000", "git_sha": "abcdef0",
                   "name": "preflight-probe", "model_hash": "{model_hash}",
                   "layer": {layer}, "bias": 0.1, "threshold": 0.5, "width": {width}}}"#
            ),
        )
        .expect("write sidecar");
        path
    }

    /// Run `f` with `OBS_LOG_DIR` pointed somewhere, then restore it.
    ///
    /// The environment is process-global and cargo runs tests in threads, so
    /// only one test may use this; it is the one that needs an unwritable
    /// root. Restoring rather than leaking keeps the rest of the module's
    /// tests reading whatever the gate set.
    fn temp_env_obs<T>(value: &Path, f: impl FnOnce() -> T) -> T {
        let previous = std::env::var_os("OBS_LOG_DIR");
        std::env::set_var("OBS_LOG_DIR", value);
        let out = f();
        match previous {
            Some(v) => std::env::set_var("OBS_LOG_DIR", v),
            None => std::env::remove_var("OBS_LOG_DIR"),
        }
        out
    }
}
