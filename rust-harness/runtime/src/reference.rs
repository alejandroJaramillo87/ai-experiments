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

use crate::ladder::{Activations, GreedyReference, TokenizerCase};
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

/// Load the L4 greedy continuations written by `fixtures/reference_greedy.py`.
///
/// `want` prompts are taken from the front of the artifact, which is what
/// makes a `deep` reference a valid `quick` one: the generator writes prompts
/// in the order `prompts.txt` lists them, and every size is a prefix of that
/// list. An artifact with fewer than `want` is an error rather than a shorter
/// run, because the header would otherwise claim a prompt count nothing
/// measured.
pub fn load_greedy(path: &Path, want: usize) -> Result<Option<Vec<GreedyReference>>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read reference {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse reference {}", path.display()))?;

    let prompts = v
        .get("prompts")
        .and_then(|t| t.as_array())
        .context("greedy artifact carries no prompts array")?;
    ensure!(
        prompts.len() >= want,
        "greedy artifact holds {} prompts and this size needs {want} -- \
         regenerate it at this size or larger",
        prompts.len()
    );

    prompts
        .iter()
        .take(want)
        .enumerate()
        .map(|(i, p)| {
            Ok(GreedyReference {
                prompt: required_ids(p, "ids").with_context(|| format!("greedy prompt {i}"))?,
                continuation: required_ids(p, "continuation_ids")
                    .with_context(|| format!("greedy prompt {i}"))?,
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// The corpus tokens and per-token NLLs written by
/// `fixtures/reference_perplexity.py`.
///
/// Returns the reference beside the token ids it scored: the caller feeds
/// those ids to the engine rather than re-tokenizing the corpus, because the
/// tokenizer is itself under test at L0 and a loader that re-derived them
/// would agree with whatever we did wrong.
///
/// `corpus` is hashed and checked against what the artifact recorded. The
/// corpus is committed and the artifact is not, so the pair drifts in exactly
/// one direction -- an artifact outliving an edit to the text it scored -- and
/// the resulting per-token comparison would be against a different document
/// with no sign of it in the output.
pub fn load_perplexity(
    path: &Path,
    corpus: &Path,
) -> Result<Option<(Vec<Token>, crate::ladder::PplReference)>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read reference {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse reference {}", path.display()))?;

    let want = v
        .get("corpus_sha256")
        .and_then(|t| t.as_str())
        .context("perplexity artifact carries no corpus_sha256")?;
    let bytes =
        std::fs::read(corpus).with_context(|| format!("read corpus {}", corpus.display()))?;
    let got = crate::digest::sha256_hex(&bytes);
    ensure!(
        got == want,
        "perplexity artifact scored a corpus with sha256 {want} and {} hashes to {got}",
        corpus.display()
    );

    let chunk = v
        .get("chunk")
        .and_then(serde_json::Value::as_u64)
        .context("perplexity artifact carries no chunk")? as usize;
    let token_ids = required_ids(&v, "token_ids")?;

    let rows = v
        .get("scored")
        .and_then(|t| t.as_array())
        .context("perplexity artifact carries no scored array")?;
    let scored = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let pair = r
                .as_array()
                .filter(|p| p.len() == 2)
                .with_context(|| format!("scored entry {i} is not a [position, nll] pair"))?;
            let position = pair[0]
                .as_u64()
                .with_context(|| format!("scored entry {i} has a non-integer position"))?
                as usize;
            let nll = pair[1]
                .as_f64()
                .with_context(|| format!("scored entry {i} has a non-number nll"))?;
            Ok((position, nll))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Some((
        token_ids,
        crate::ladder::PplReference { chunk, scored },
    )))
}

/// Load the L5 long-context rows written by
/// `fixtures/reference_long_context.py`.
///
/// Same contract as the loaders above: absent is `Ok(None)` and a skipped
/// rung, present and malformed is an error.
///
/// `sliding_window` is required rather than defaulted. It is what the
/// reference attended under, and L5's verdict is a comparison against the
/// window this run read from the same config -- an artifact that does not
/// state it cannot take part in that comparison at all, and a default would
/// silently supply one side of it.
pub fn load_long_context(path: &Path) -> Result<Option<crate::ladder::LongContextReference>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read reference {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse reference {}", path.display()))?;

    let sliding_window =
        v.get("sliding_window")
            .and_then(serde_json::Value::as_u64)
            .context("long-context artifact carries no sliding_window")? as usize;
    let token_ids = required_ids(&v, "token_ids")?;

    let rows = v
        .get("rows")
        .and_then(|t| t.as_array())
        .context("long-context artifact carries no rows array")?;
    let mut positions = Vec::with_capacity(rows.len());
    let mut logits = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        positions.push(
            row.get("position")
                .and_then(serde_json::Value::as_u64)
                .with_context(|| format!("row {i} has no integer position"))? as usize,
        );
        logits.push(
            optional_row(row, "logits")?.with_context(|| format!("row {i} carries no logits"))?,
        );
    }

    Ok(Some(crate::ladder::LongContextReference {
        sliding_window,
        token_ids,
        positions,
        rows: logits,
    }))
}

/// Load the L6 attention rows written by `fixtures/reference_attention.py`.
///
/// Same contract as the loaders above: absent is `Ok(None)` and a skipped
/// rung, present and malformed is an error. `layers`, `heads` and
/// `positions` are required for L5's reason with `sliding_window`: the rung
/// refuses an artifact whose choices differ from its own, and a defaulted
/// field would silently supply one side of that comparison.
pub fn load_attention(path: &Path) -> Result<Option<crate::ladder::AttentionReference>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read reference {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse reference {}", path.display()))?;

    let count = |key: &str| -> Result<usize> {
        Ok(v.get(key)
            .and_then(serde_json::Value::as_u64)
            .with_context(|| format!("attention artifact carries no integer {key}"))?
            as usize)
    };
    let indices = |key: &str| -> Result<Vec<usize>> {
        v.get(key)
            .and_then(|t| t.as_array())
            .with_context(|| format!("attention artifact carries no {key} array"))?
            .iter()
            .enumerate()
            .map(|(i, x)| {
                Ok(x.as_u64()
                    .with_context(|| format!("{key}[{i}] is not an integer"))?
                    as usize)
            })
            .collect()
    };
    let sliding_window = count("sliding_window")?;
    let heads = count("heads")?;
    let layers = indices("layers")?;
    let positions = indices("positions")?;
    let token_ids = required_ids(&v, "token_ids")?;

    let rows = v
        .get("rows")
        .and_then(|t| t.as_array())
        .context("attention artifact carries no rows array")?;
    let mut out = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let field = |key: &str| -> Result<usize> {
            Ok(row
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .with_context(|| format!("row {i} has no integer {key}"))? as usize)
        };
        out.push(crate::ladder::AttentionReferenceRow {
            layer: field("layer")?,
            head: field("head")?,
            position: field("position")?,
            weights: optional_row(row, "weights")?
                .with_context(|| format!("row {i} carries no weights"))?,
        });
    }

    Ok(Some(crate::ladder::AttentionReference {
        sliding_window,
        layers,
        heads,
        positions,
        token_ids,
        rows: out,
    }))
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

    fn write_greedy(dir: &Path, body: &str) -> std::path::PathBuf {
        let path = dir.join("greedy.json");
        std::fs::write(&path, body).unwrap();
        path
    }

    const GREEDY: &str = r#"{"tokens": 2, "prompts": [
        {"ids": [1, 2], "continuation_ids": [7, 8]},
        {"ids": [3], "continuation_ids": [9, 10]},
        {"ids": [4], "continuation_ids": [11, 12]}]}"#;

    #[test]
    fn a_larger_greedy_artifact_serves_a_smaller_size_from_its_front() {
        // The nested-prefix property, at the artifact rather than the corpus:
        // the generator writes prompts in prompts.txt order and every size is
        // a prefix of that list, so a deep reference is a valid quick one and
        // the two runs measure the same first rows.
        let dir = tempfile::tempdir().unwrap();
        let path = write_greedy(dir.path(), GREEDY);
        let got = load_greedy(&path, 2).unwrap().unwrap();
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[0].prompt, vec![1, 2]);
        assert_eq!(got[1].continuation, vec![9, 10]);
    }

    #[test]
    fn a_greedy_artifact_with_too_few_prompts_is_an_error() {
        // Not a shorter run: the header would claim a prompt count that
        // nothing measured.
        let dir = tempfile::tempdir().unwrap();
        let path = write_greedy(dir.path(), GREEDY);
        let err = load_greedy(&path, 8).expect_err("expected an error");
        assert!(
            err.to_string().contains("holds 3 prompts") && err.to_string().contains("needs 8"),
            "{err}"
        );
    }

    #[test]
    fn a_greedy_prompt_missing_its_continuation_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_greedy(dir.path(), r#"{"prompts": [{"ids": [1]}]}"#);
        let err = load_greedy(&path, 1).expect_err("expected an error");
        assert!(err.to_string().contains("greedy prompt 0"), "{err}");
    }

    /// A corpus file and an artifact that scored it, agreeing by construction.
    fn ppl_pair(
        dir: &Path,
        corpus_text: &str,
        recorded: Option<&str>,
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let corpus = dir.join("corpus.txt");
        std::fs::write(&corpus, corpus_text).unwrap();
        let hash = recorded
            .map(str::to_string)
            .unwrap_or_else(|| crate::digest::sha256_hex(corpus_text.as_bytes()));
        let path = dir.join("ppl.json");
        std::fs::write(
            &path,
            format!(
                r#"{{"chunk": 4, "corpus_sha256": "{hash}", "token_ids": [1, 2, 3, 4],
                    "scored": [[2, 0.5], [3, 1.5]]}}"#
            ),
        )
        .unwrap();
        (path, corpus)
    }

    #[test]
    fn a_perplexity_artifact_round_trips_its_corpus_chunk_and_scored_pairs() {
        let dir = tempfile::tempdir().unwrap();
        let (path, corpus) = ppl_pair(dir.path(), "the corpus", None);
        let (ids, reference) = load_perplexity(&path, &corpus).unwrap().unwrap();
        assert_eq!(ids, vec![1, 2, 3, 4]);
        assert_eq!(reference.chunk, 4);
        assert_eq!(reference.scored, vec![(2, 0.5), (3, 1.5)]);
    }

    #[test]
    fn a_perplexity_artifact_that_scored_a_different_corpus_is_an_error() {
        // The corpus is committed and the artifact is not, so the pair drifts
        // in exactly one direction: an artifact outliving an edit to the text
        // it scored. Without this the per-token comparison would run against
        // a different document and say nothing about it.
        let dir = tempfile::tempdir().unwrap();
        let (path, corpus) = ppl_pair(dir.path(), "the corpus", None);
        std::fs::write(&corpus, "a different corpus").unwrap();
        let err = load_perplexity(&path, &corpus).expect_err("expected an error");
        assert!(err.to_string().contains("hashes to"), "{err}");
    }

    #[test]
    fn a_perplexity_artifact_with_a_ragged_scored_entry_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let corpus = dir.path().join("corpus.txt");
        std::fs::write(&corpus, "x").unwrap();
        let hash = crate::digest::sha256_hex(b"x");
        let path = dir.path().join("ppl.json");
        std::fs::write(
            &path,
            format!(
                r#"{{"chunk": 4, "corpus_sha256": "{hash}", "token_ids": [1],
                    "scored": [[2]]}}"#
            ),
        )
        .unwrap();
        let err = load_perplexity(&path, &corpus).expect_err("expected an error");
        assert!(err.to_string().contains("scored entry 0"), "{err}");
    }

    #[test]
    fn a_long_context_artifact_round_trips_its_window_tokens_and_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long_context.json");
        std::fs::write(
            &path,
            r#"{"sliding_window": 3, "token_ids": [2, 7, 9, 4],
                "rows": [{"position": 2, "logits": [0.5, 0.25]},
                         {"position": 3, "logits": [1.5, 2.0]}]}"#,
        )
        .unwrap();
        let r = load_long_context(&path).unwrap().unwrap();
        assert_eq!(r.sliding_window, 3);
        assert_eq!(r.token_ids, vec![2, 7, 9, 4]);
        assert_eq!(r.positions, vec![2, 3]);
        assert_eq!(r.rows[1], vec![1.5, 2.0]);
    }

    #[test]
    fn a_long_context_artifact_without_a_window_is_an_error() {
        // The window is half of L5's comparison. Defaulting it would supply
        // the reference's side of a disagreement the rung exists to find.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long_context.json");
        std::fs::write(&path, r#"{"token_ids": [1], "rows": []}"#).unwrap();
        let err = load_long_context(&path).expect_err("expected an error");
        assert!(err.to_string().contains("no sliding_window"), "{err}");
    }

    #[test]
    fn an_absent_long_context_artifact_is_none_rather_than_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let got = load_long_context(&dir.path().join("nothing.json")).unwrap();
        assert!(got.is_none());
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
