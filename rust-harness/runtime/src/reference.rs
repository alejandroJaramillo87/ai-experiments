/*
 * reference — reading the artifacts the ladder compares against.
 *
 * One module for every reference format, because the ladder's rungs are only
 * as trustworthy as the claim that both sides are looking at the same run. An
 * artifact carries the token IDs rather than the prompt text for exactly that
 * reason: the tokenizer is itself under test at L0, so a loader that
 * re-tokenized the prompt would quietly agree with whatever we did wrong.
 *
 * Artifacts are absent more often than present -- Tier E's are generated per
 * run on the workstation and not committed -- so every loader distinguishes
 * "not there" from "there and broken". The first is a skipped rung, the
 * second is an error.
 */

use std::path::Path;

use anyhow::{ensure, Context, Result};

use crate::ladder::{Activations, TokenizerCase};
use crate::model::Token;

/// A reference forward pass: what was fed in, and what came out at each stage.
#[derive(Debug, Clone)]
pub struct Capture {
    pub token_ids: Vec<Token>,
    /// The precision the reference ran in, for the report header. A capture
    /// in bfloat16 and one in float32 are not comparable to the same
    /// tolerance, and the number alone does not say which it was.
    pub dtype: String,
    pub activations: Activations,
    /// The final-position logits, when the capture recorded them.
    ///
    /// L2's reference for Tier T, and optional because the capture format
    /// predates L2 -- a layers-only artifact is a usable L1 reference and
    /// should skip L2 rather than fail it.
    pub logits: Option<Vec<f32>>,
}

/// A float array under `key`, or `None` when the capture does not carry it.
fn optional_row(v: &serde_json::Value, key: &str) -> Result<Option<Vec<f32>>> {
    let Some(node) = v.get(key) else {
        return Ok(None);
    };
    let row = node
        .as_array()
        .with_context(|| format!("{key} is present but not an array"))?;
    row.iter()
        .map(|x| {
            x.as_f64()
                .map(|f| f as f32)
                .with_context(|| format!("{key} holds a non-number"))
        })
        .collect::<Result<Vec<f32>>>()
        .map(Some)
}

/// Load a capture written by `fixtures/reference_capture.py`.
///
/// Returns `Ok(None)` when the file is absent, so a caller can report a
/// skipped rung without deciding what a missing artifact means. Any other
/// failure -- unreadable, malformed, missing a layer the header promises --
/// is an error, because a half-read reference compared against is worse than
/// no reference at all.
pub fn load_capture(path: &Path, stages: &[&'static str]) -> Result<Option<Capture>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read reference {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse reference {}", path.display()))?;

    let ids = v
        .get("token_ids")
        .and_then(|t| t.as_array())
        .context("reference carries no token_ids array")?;
    let token_ids: Vec<Token> = ids
        .iter()
        .map(|t| {
            t.as_u64()
                .map(|n| n as Token)
                .context("token_ids holds a non-integer")
        })
        .collect::<Result<_>>()?;

    let n_layers = v
        .get("n_layers")
        .and_then(serde_json::Value::as_u64)
        .context("reference carries no n_layers")? as usize;
    let rows = v
        .get("layers")
        .and_then(|t| t.as_object())
        .context("reference carries no layers object")?;

    // Indexed by the declared count rather than by iterating the object, so a
    // capture that lost a middle layer is an error instead of a shorter table
    // that still looks complete.
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let row = rows
            .get(&i.to_string())
            .and_then(|t| t.as_array())
            .with_context(|| format!("reference is missing layer {i}"))?;
        layers.push(
            row.iter()
                .map(|x| {
                    x.as_f64()
                        .map(|f| f as f32)
                        .with_context(|| format!("layer {i} holds a non-number"))
                })
                .collect::<Result<_>>()?,
        );
    }

    let dtype = v
        .get("dtype")
        .and_then(|t| t.as_str())
        .unwrap_or("unknown")
        .to_string();
    let mut named = Vec::new();
    for name in stages {
        if let Some(row) = optional_row(&v, name)? {
            named.push(((*name).to_string(), row));
        }
    }

    Ok(Some(Capture {
        token_ids,
        dtype,
        logits: optional_row(&v, "logits")?,
        activations: Activations {
            stages: named,
            layers,
        },
    }))
}

/// The token ids under `key`, which must be a present array of integers.
///
/// Unlike `optional_row`, absence is an error: every caller here reads a field
/// the format requires, and a case with no ids would compare our encoding
/// against nothing and call it agreement.
fn required_ids(v: &serde_json::Value, key: &str) -> Result<Vec<Token>> {
    let node = v
        .get(key)
        .and_then(|t| t.as_array())
        .with_context(|| format!("{key} is missing or not an array"))?;
    node.iter()
        .map(|t| {
            t.as_u64()
                .map(|n| n as Token)
                .with_context(|| format!("{key} holds a non-integer"))
        })
        .collect()
}

/// Load the L0 parity cases written by `fixtures/reference_tokenizer.py`.
///
/// Same contract as `load_capture`: absent is `Ok(None)` and a skipped rung,
/// present and malformed is an error.
///
/// `chat_template_version` is checked against the harness's own constant. The
/// rendered-turn case is the whole reason this artifact exists, and a
/// reference generated under one template revision compared against a renderer
/// at another would report a tokenizer defect for what is a version skew.
pub fn load_tokenizer_parity(
    path: &Path,
    chat_template_version: u32,
) -> Result<Option<Vec<TokenizerCase>>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read reference {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse reference {}", path.display()))?;

    let want = v
        .get("chat_template_version")
        .and_then(serde_json::Value::as_u64)
        .context("parity artifact carries no chat_template_version")? as u32;
    ensure!(
        want == chat_template_version,
        "parity artifact was generated at chat template version {want}, \
         and this harness renders version {chat_template_version}",
    );

    let cases = v
        .get("cases")
        .and_then(|t| t.as_array())
        .context("parity artifact carries no cases array")?;

    cases
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let name = c
                .get("name")
                .and_then(|t| t.as_str())
                .with_context(|| format!("case {i} has no name"))?
                .to_string();
            let text = c
                .get("text")
                .and_then(|t| t.as_str())
                .with_context(|| format!("case {name} has no text"))?
                .to_string();
            // Absent means the empty allowlist, which is `encode`: a case that
            // vouches for nothing is the ordinary one, and requiring the field
            // would make the common case the verbose one.
            let allowed = match c.get("allowed") {
                None => Vec::new(),
                Some(a) => a
                    .as_array()
                    .with_context(|| format!("case {name}: allowed is not an array"))?
                    .iter()
                    .map(|s| {
                        s.as_str()
                            .map(str::to_string)
                            .with_context(|| format!("case {name}: allowed holds a non-string"))
                    })
                    .collect::<Result<_>>()?,
            };
            let ids = required_ids(c, "ids").with_context(|| format!("case {name}"))?;
            Ok(TokenizerCase {
                name,
                text,
                allowed,
                ids,
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, body: &str) -> std::path::PathBuf {
        let path = dir.join("capture.json");
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn an_absent_artifact_is_none_rather_than_an_error() {
        // The distinction the whole module turns on: Tier E artifacts are
        // generated per run and not committed, so absence is the ordinary
        // case and has to reach the caller as a skipped rung.
        let dir = tempfile::tempdir().unwrap();
        let got = load_capture(&dir.path().join("nothing.json"), &[]).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn a_capture_round_trips_its_tokens_layers_and_named_stages() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(
            dir.path(),
            r#"{"token_ids": [2, 105], "n_layers": 2, "dtype": "float32",
                "layers": {"0": [1.0, 2.0], "1": [3.0, 4.0]},
                "embed": [0.5, 0.25]}"#,
        );
        let c = load_capture(&path, &["embed", "ple"]).unwrap().unwrap();
        assert_eq!(c.token_ids, vec![2, 105]);
        assert_eq!(c.dtype, "float32");
        assert_eq!(c.activations.layers, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        // "ple" was asked for and is not in the file: absent, not empty.
        assert_eq!(c.activations.stages.len(), 1);
        assert_eq!(c.activations.stages[0].0, "embed");
    }

    #[test]
    fn a_capture_missing_a_layer_it_declares_is_an_error() {
        // Indexed by n_layers rather than by whatever the object holds, so a
        // capture that lost layer 1 fails instead of producing a two-row
        // table that reads as complete.
        let dir = tempfile::tempdir().unwrap();
        let path = write(
            dir.path(),
            r#"{"token_ids": [1], "n_layers": 3,
                "layers": {"0": [1.0], "2": [3.0]}}"#,
        );
        let err = load_capture(&path, &[]).expect_err("expected an error");
        assert!(err.to_string().contains("missing layer 1"), "{err}");
    }

    fn write_parity(dir: &Path, body: &str) -> std::path::PathBuf {
        let path = dir.join("tokenizer_parity.json");
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn an_absent_parity_artifact_is_none_rather_than_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let got = load_tokenizer_parity(&dir.path().join("nothing.json"), 2).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn parity_cases_round_trip_their_text_allowlist_and_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_parity(
            dir.path(),
            r#"{"chat_template_version": 2, "cases": [
                {"name": "plain", "text": "hi", "ids": [7, 8]},
                {"name": "turn", "text": "<|turn>u", "allowed": ["<|turn>"], "ids": [3]}]}"#,
        );
        let cases = load_tokenizer_parity(&path, 2).unwrap().unwrap();
        assert_eq!(cases.len(), 2, "{cases:?}");
        // Absent `allowed` is the empty allowlist, not a parse failure.
        assert!(cases[0].allowed.is_empty(), "{:?}", cases[0]);
        assert_eq!(cases[1].allowed, vec!["<|turn>".to_string()]);
        assert_eq!(cases[1].ids, vec![3]);
    }

    #[test]
    fn a_parity_artifact_from_another_chat_template_version_is_an_error() {
        // The rendered-turn case is the reason the artifact exists. Comparing
        // it across a template revision would report a tokenizer defect for
        // what is a version skew, which is the wrong thing to send a reader
        // looking for.
        let dir = tempfile::tempdir().unwrap();
        let path = write_parity(
            dir.path(),
            r#"{"chat_template_version": 1, "cases": [{"name": "a", "text": "b", "ids": [1]}]}"#,
        );
        let err = load_tokenizer_parity(&path, 2).expect_err("expected an error");
        assert!(
            err.to_string().contains("version 1") && err.to_string().contains("version 2"),
            "{err}"
        );
    }

    #[test]
    fn a_parity_case_with_no_ids_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_parity(
            dir.path(),
            r#"{"chat_template_version": 2, "cases": [{"name": "a", "text": "b"}]}"#,
        );
        let err = load_tokenizer_parity(&path, 2).expect_err("expected an error");
        assert!(err.to_string().contains("case a"), "{err}");
    }

    #[test]
    fn a_present_but_malformed_artifact_is_an_error_not_a_skip() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), "{not json");
        let err = load_capture(&path, &[]).expect_err("expected an error");
        assert!(err.to_string().contains("parse reference"), "{err}");

        let path = write(dir.path(), r#"{"n_layers": 0, "layers": {}}"#);
        let err = load_capture(&path, &[]).expect_err("expected an error");
        assert!(err.to_string().contains("no token_ids"), "{err}");
    }
}
