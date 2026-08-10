/*
 * layer-diff — compare the CPU backend's residual stream against transformers.
 *
 * Reads a capture written by fixtures/reference_capture.py, prefills the same
 * token IDs through the CPU backend, and prints one row per layer: cosine
 * similarity, max absolute difference, and relative L2. It names the first
 * layer to exceed tolerance and exits non-zero.
 *
 * First, not worst. Once layer k diverges every later layer is wrong
 * downstream of it, so a table read by magnitude points at the end of the
 * stack rather than at the cause. The first row over tolerance is the layer
 * holding the bug.
 *
 * This exists because the goldens cannot establish correctness. They compare
 * the Rust against fixtures/reference_logits.py on random weights, and both
 * were transcribed from the same documents by the same reader -- so a
 * misreading is present in both and they agree about it. Exactly that
 * happened: a 1/sqrt(head_dim) Gemma 4 does not apply, and a missing
 * scale-free norm on the value states, survived nine work packages of green
 * gate. Real weights and an oracle written by someone else are what catch a
 * belief; a second transcription only catches a slip.
 */

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{ensure, Context, Result};

use runtime::model::cpu::{CpuModel, LastRowTap, PRELUDE_STAGES};
use runtime::model::{Model, Token};

/// Cosine similarity below which a layer is reported as divergent.
///
/// Cosine rather than an absolute difference, because the residual stream's
/// magnitude spans orders of magnitude across the stack -- E2B's per-layer
/// scales alone range over 47x -- so one absolute threshold would be
/// meaninglessly loose at one end and unsatisfiable at the other.
///
/// 0.9999 against a float32 reference: our accumulation order differs from
/// torch's, which moves the last bits and nothing more. A real defect moves
/// cosine far further than this, as bug A's factor of 16 on every attention
/// logit did.
const COSINE_FLOOR: f64 = 0.9999;

/// Relative L2 above which a layer is reported as divergent, checked
/// alongside the cosine floor.
///
/// Cosine measures direction and nothing else, so a residual stream that is
/// correct in shape and wrong in size passes it perfectly -- which is the
/// exact signature of a missing `layer_scalar`, a tensor the released
/// checkpoints carry at values as low as 0.049. Direction and magnitude are
/// independent failures and each needs its own threshold.
const REL_L2_CEILING: f64 = 1e-3;

struct Reference {
    token_ids: Vec<Token>,
    dtype: String,
    layers: Vec<Vec<f32>>,
    /// The named pre-stack stages the capture carried, in `PRELUDE_STAGES`
    /// order. A stage is absent from an older capture, and from a model that
    /// does not reach it.
    stages: Vec<(&'static str, Vec<f32>)>,
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

fn load_reference(path: &Path) -> Result<Reference> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read reference {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text).context("parse reference JSON")?;

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
    let mut stages = Vec::new();
    for name in PRELUDE_STAGES {
        if let Some(row) = optional_row(&v, name)? {
            stages.push((name, row));
        }
    }
    Ok(Reference {
        token_ids,
        dtype,
        layers,
        stages,
    })
}

/// Cosine similarity, max absolute difference, and relative L2 of `got`
/// against `want`. f64 throughout: the sums run over thousands of terms and
/// the quantity being measured is itself a small difference of large numbers.
fn compare(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    let mut dot = 0.0f64;
    let mut ng = 0.0f64;
    let mut nw = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut diff_sq = 0.0f64;
    for (a, b) in got.iter().zip(want) {
        let (a, b) = (*a as f64, *b as f64);
        dot += a * b;
        ng += a * a;
        nw += b * b;
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        diff_sq += (a - b) * (a - b);
    }
    // A zero row on either side has no direction, so cosine is undefined
    // rather than 1.0. Reported as 0.0, which fails the floor and prints,
    // instead of silently passing as a perfect match of nothing.
    let cosine = match ng > 0.0 && nw > 0.0 {
        true => dot / (ng.sqrt() * nw.sqrt()),
        false => 0.0,
    };
    let rel_l2 = match nw > 0.0 {
        true => diff_sq.sqrt() / nw.sqrt(),
        false => f64::INFINITY,
    };
    (cosine, max_abs, rel_l2)
}

/// Print one table row, and report whether it diverged.
///
/// Either failure alone is a divergence, and the mark says which, so a
/// wrong-magnitude row is not read as a wrong-direction one.
fn report(label: &str, got: &[f32], want: &[f32], floor: f64, ceiling: f64) -> Result<bool> {
    ensure!(
        got.len() == want.len(),
        "{label}: backend row is {} wide, reference row is {} wide",
        got.len(),
        want.len()
    );
    let (cosine, max_abs, rel_l2) = compare(got, want);
    let mark = match (cosine < floor, rel_l2 > ceiling) {
        (true, true) => "  <-- diverges",
        (true, false) => "  <-- diverges (direction)",
        (false, true) => "  <-- diverges (magnitude)",
        (false, false) => "",
    };
    println!("{label:>12}  {cosine:<11.8}  {max_abs:<11.3e}  {rel_l2:<11.3e}{mark}");
    Ok(!mark.is_empty())
}

fn run(
    model_path: &Path,
    reference_path: &Path,
    text_only: bool,
    floor: f64,
    ceiling: f64,
) -> Result<bool> {
    let reference = load_reference(reference_path)?;
    ensure!(
        !reference.token_ids.is_empty(),
        "reference carries no tokens"
    );

    let files = runtime::checkpoint::CheckpointFiles::read(model_path)
        .with_context(|| format!("read model {}", model_path.display()))?;
    let mut ck = runtime::checkpoint::parse(&files).context("parse checkpoint")?;
    ck.text_only = text_only;
    let mut model = CpuModel::from_checkpoint(&ck).context("load CPU model")?;
    let n_layers = model.config().n_layers;
    ensure!(
        n_layers == reference.layers.len(),
        "reference captured {} layers, the checkpoint has {n_layers}: the two are not \
         the same model",
        reference.layers.len()
    );

    let tap = LastRowTap::new((0..n_layers).collect()).capturing_prelude();
    model.set_tap(Box::new(tap.clone()));
    model
        .prefill(&reference.token_ids)
        .context("prefill the reference tokens")?;

    println!(
        "Reference: {} tokens, {} layers, {} precision",
        reference.token_ids.len(),
        n_layers,
        reference.dtype
    );
    println!("Cosine floor: {floor}, relative-L2 ceiling: {ceiling}");
    println!();
    println!("{:>12}  cosine       max_abs      rel_l2", "stage/layer");

    // The prelude rows first, and they are the rows to read first. A wrong
    // input to the stack makes every layer wrong without any layer being at
    // fault, and the layer table alone cannot say so: layer 0 is the first
    // row it has, so it takes the blame either way.
    // Stages are a chain, so the first divergence is the term rather than
    // any sum it feeds: `ple` is wrong whenever either half is.
    let mut prelude_bad: Option<&str> = None;
    for (name, want) in &reference.stages {
        let got = tap
            .stage(name)
            .with_context(|| format!("{name}: the tap captured no row"))?;
        if report(name, &got, want, floor, ceiling)? && prelude_bad.is_none() {
            prelude_bad = Some(name);
        }
    }

    let mut first_bad: Option<usize> = None;
    for i in 0..n_layers {
        let got = tap
            .row(i)
            .with_context(|| format!("layer {i}: the tap captured no row"))?;
        if report(&i.to_string(), &got, &reference.layers[i], floor, ceiling)?
            && first_bad.is_none()
        {
            first_bad = Some(i);
        }
    }

    println!();
    if let Some(stage) = prelude_bad {
        // Reported before the layer verdict and instead of it: with the input
        // to layer 0 already wrong, the layer table below measures a stack
        // fed bad data and says nothing about the layers themselves.
        let cause = match stage {
            "embed" => {
                "the token IDs, the embedding table, or the sqrt(embed_dim) scale -- \
                 not any decoder layer"
            }
            "ple_identity" => {
                "the packed per-layer table: its row stride, or the sqrt(ple_dim) \
                 embed scale"
            }
            "ple_context" => {
                "the per-layer projection: its weight orientation, the \
                 1/sqrt(embed_dim) scale, or the norm over each row"
            }
            // Both halves clean and the sum wrong leaves only the sum.
            _ => "the composition of the two halves, which is only the 1/sqrt(2) scale",
        };
        println!("[ERROR] Residual stream: the stack's input diverges at '{stage}'");
        println!("  Look at {cause}.");
        println!(
            "  Every layer below is downstream of this; the layer rows are not \
             evidence about the layers until this is clean."
        );
        return Ok(false);
    }
    match first_bad {
        None => {
            println!("[OK] Residual stream: all {n_layers} layers within tolerance");
            Ok(true)
        }
        Some(i) => {
            // Naming the layer's own geometry here saves the reader a trip to
            // checkpoint-info, and it is usually the answer: the first Global
            // layer points at head width or RoPE base, and a layer at or past
            // the sharing point points at the donor path.
            let spec = model.layer_specs()[i];
            println!(
                "[ERROR] Residual stream: first divergence at layer {i} ({:?}, head {}, \
                 ffn {}, rope base {}, kv source {})",
                spec.kind, spec.head_dim, spec.ffn_dim, spec.rope_freq_base, spec.kv_source
            );
            println!(
                "  Every later layer is downstream of this one; fix layer {i} before \
                 reading the rest of the table."
            );
            Ok(false)
        }
    }
}

const USAGE: &str = "usage: layer-diff [--text-only] [--cosine-floor <f>] \
                     [--rel-l2-ceiling <f>] --model <checkpoint-dir> \
                     --reference <capture.json>";

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    let text_only = match args.iter().position(|a| a == "--text-only") {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    };

    let mut floor = COSINE_FLOOR;
    let mut ceiling = REL_L2_CEILING;
    let mut model: Option<PathBuf> = None;
    let mut reference: Option<PathBuf> = None;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let Some(value) = it.next() else {
            eprintln!("[ERROR] {arg}: missing value");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        };
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--reference" => reference = Some(PathBuf::from(value)),
            "--cosine-floor" => match value.parse::<f64>() {
                Ok(v) => floor = v,
                Err(e) => {
                    eprintln!("[ERROR] --cosine-floor {value}: {e}");
                    return ExitCode::from(2);
                }
            },
            "--rel-l2-ceiling" => match value.parse::<f64>() {
                Ok(v) => ceiling = v,
                Err(e) => {
                    eprintln!("[ERROR] --rel-l2-ceiling {value}: {e}");
                    return ExitCode::from(2);
                }
            },
            other => {
                eprintln!("[ERROR] Unknown argument: {other}");
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }

    let (Some(model), Some(reference)) = (model, reference) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };

    match run(&model, &reference, text_only, floor, ceiling) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[ERROR] {e:#}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_rows_compare_as_a_perfect_match() {
        let x = [1.0f32, -2.0, 3.5, 0.25];
        let (cosine, max_abs, rel_l2) = compare(&x, &x);
        assert!((cosine - 1.0).abs() < 1e-12, "{cosine}");
        assert_eq!(max_abs, 0.0);
        assert_eq!(rel_l2, 0.0);
    }

    #[test]
    fn a_uniformly_scaled_row_keeps_its_direction_and_fails_on_magnitude() {
        // The shape a missing output scale produces: same direction, wrong
        // size. Cosine alone would pass it, which is why the table prints
        // rel_l2 beside it rather than cosine on its own.
        let x = [1.0f32, -2.0, 3.5, 0.25];
        let scaled: Vec<f32> = x.iter().map(|v| v * 0.5).collect();
        let (cosine, _, rel_l2) = compare(&scaled, &x);
        assert!((cosine - 1.0).abs() < 1e-12, "{cosine}");
        assert!((rel_l2 - 0.5).abs() < 1e-12, "{rel_l2}");
    }

    #[test]
    fn a_zero_row_is_reported_as_no_match_rather_than_a_perfect_one() {
        // 0/0 would be NaN, and NaN < floor is false -- so an unhooked layer
        // capturing zeros would read as passing. Reported as 0.0 instead,
        // which fails the floor.
        let (cosine, _, rel_l2) = compare(&[0.0; 4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cosine, 0.0);
        assert!(rel_l2.is_finite(), "{rel_l2}");
    }
}
