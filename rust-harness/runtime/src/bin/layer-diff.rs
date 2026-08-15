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

use runtime::cli::{Flag, Kind, Spec};
use runtime::metrics::vec_diff;
use runtime::model::cpu::{CpuModel, LastRowTap, PRELUDE_STAGES};
use runtime::model::Model;

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
    let d = vec_diff(got, want);
    let (cosine, max_abs, rel_l2) = (d.cosine, d.max_abs, d.rel_l2);
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
    single_thread: bool,
    floor: f64,
    ceiling: f64,
) -> Result<bool> {
    let reference = runtime::reference::load_capture(reference_path, &PRELUDE_STAGES)?
        .with_context(|| format!("reference {} does not exist", reference_path.display()))?;
    ensure!(
        !reference.token_ids.is_empty(),
        "reference carries no tokens"
    );

    // `load` drops the checkpoint mapping before returning; see its doc
    // comment for why no caller here should open one itself.
    let (mut model, _fingerprint) = CpuModel::load(model_path, text_only)?;
    if single_thread {
        model.set_threads(1);
    }
    let n_layers = model.config().n_layers;
    ensure!(
        n_layers == reference.activations.layers.len(),
        "reference captured {} layers, the checkpoint has {n_layers}: the two are not \
         the same model",
        reference.activations.layers.len()
    );

    let tap = LastRowTap::new((0..n_layers).collect()).capturing_prelude();
    model.set_tap(Box::new(tap.clone()));
    model
        .prefill(&reference.token_ids)
        .context("prefill the reference tokens")?;

    // These two describe the run rather than reporting a measurement, so
    // they are status and leave stdout to the table. The cost is real and
    // worth stating: on a terminal they still print immediately above the
    // table they qualify, but a redirect now separates them, and the
    // tolerances are what make the table's `ok`/`FAIL` marks readable.
    // `2>&1` is how a capture keeps them together.
    runtime::status::info(&format!(
        "Reference: {} tokens, {} layers, {} precision",
        reference.token_ids.len(),
        n_layers,
        reference.dtype
    ));
    runtime::status::info(&format!(
        "Cosine floor: {floor}, relative-L2 ceiling: {ceiling}"
    ));
    println!("{:>12}  cosine       max_abs      rel_l2", "stage/layer");

    // The prelude rows first, and they are the rows to read first. A wrong
    // input to the stack makes every layer wrong without any layer being at
    // fault, and the layer table alone cannot say so: layer 0 is the first
    // row it has, so it takes the blame either way.
    // Stages are a chain, so the first divergence is the term rather than
    // any sum it feeds: `ple` is wrong whenever either half is.
    let mut prelude_bad: Option<&str> = None;
    for (name, want) in &reference.activations.stages {
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
        if report(
            &i.to_string(),
            &got,
            &reference.activations.layers[i],
            floor,
            ceiling,
        )? && first_bad.is_none()
        {
            first_bad = Some(i);
        }
    }

    println!();
    if let Some(stage) = prelude_bad {
        // Reported before the layer verdict and instead of it: with the input
        // to layer 0 already wrong, the layer table below measures a stack
        // fed bad data and says nothing about the layers themselves.
        // From the ladder rather than restated here: L1 reports the same
        // divergence, and two wordings for one failure is how a reader ends
        // up believing they are two failures.
        let cause = runtime::ladder::stage_cause(stage);
        runtime::status::error(&format!(
            "Residual stream: the stack's input diverges at '{stage}'"
        ));
        runtime::status::plain(&format!("  Look at {cause}."));
        runtime::status::plain(
            "  Every layer below is downstream of this; the layer rows are not \
             evidence about the layers until this is clean.",
        );
        return Ok(false);
    }
    match first_bad {
        None => {
            runtime::status::ok(&format!(
                "Residual stream: all {n_layers} layers within tolerance"
            ));
            Ok(true)
        }
        Some(i) => {
            // Naming the layer's own geometry here saves the reader a trip to
            // checkpoint-info, and it is usually the answer: the first Global
            // layer points at head width or RoPE base, and a layer at or past
            // the sharing point points at the donor path.
            let spec = model.layer_specs()[i];
            runtime::status::error(&format!(
                "Residual stream: first divergence at layer {i} ({:?}, head {}, \
                 ffn {}, rope base {}, kv source {})",
                spec.kind, spec.head_dim, spec.ffn_dim, spec.rope_freq_base, spec.kv_source
            ));
            runtime::status::plain(&format!(
                "  Every later layer is downstream of this one; fix layer {i} before \
                 reading the rest of the table."
            ));
            Ok(false)
        }
    }
}

const SPEC: Spec = Spec {
    program: "layer-diff",
    summary: "Report per-layer divergence of the residual stream against a reference\n\
              capture, and pass or fail it against the tolerances.",
    positionals: "",
    required: &["--model", "--reference"],
    flags: &[
        Flag {
            name: "--help",
            kind: Kind::Bare,
            help: "print this help and exit",
        },
        Flag {
            name: "-h",
            kind: Kind::Bare,
            help: "print this help and exit",
        },
        Flag {
            name: "--model",
            kind: Kind::Value("<checkpoint-dir>"),
            help: "the checkpoint to run; required",
        },
        Flag {
            name: "--reference",
            kind: Kind::Value("<capture.json>"),
            help: "the reference capture to compare against; required",
        },
        Flag {
            name: "--cosine-floor",
            kind: Kind::Value("<f>"),
            help: "lowest per-layer cosine similarity that still passes",
        },
        Flag {
            name: "--rel-l2-ceiling",
            kind: Kind::Value("<f>"),
            help: "highest per-layer relative L2 error that still passes",
        },
        Flag {
            name: "--text-only",
            kind: Kind::Bare,
            help: "treat a multimodal checkpoint as its text tower",
        },
        // Serial rather than the machine's width. Not a numerics switch: the
        // row split is bit-identical at every width, so the cosine and rel-L2
        // figures are the same either way. Present for rr, which serializes
        // threads regardless, and for timing one path against the other.
        Flag {
            name: "--single-thread",
            kind: Kind::Bare,
            help: "run the row split on one thread; the figures are unchanged",
        },
    ],
};

/// Parse a tolerance, naming the flag and the value that could not be read.
fn tolerance(name: &str, given: Option<String>, fallback: f64) -> Result<f64, String> {
    match given {
        None => Ok(fallback),
        Some(v) => v.parse().map_err(|e| format!("{name} {v}: {e}")),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if runtime::cli::handled_help(&SPEC, &args) {
        return ExitCode::SUCCESS;
    }
    let (mut model, mut reference, mut floor_arg, mut ceiling_arg) = (None, None, None, None);
    let parsed = {
        let mut slots: [(&'static str, &mut Option<String>); 4] = [
            ("--model", &mut model),
            ("--reference", &mut reference),
            ("--cosine-floor", &mut floor_arg),
            ("--rel-l2-ceiling", &mut ceiling_arg),
        ];
        match runtime::cli::extract(&SPEC, args, &mut slots) {
            Ok(p) => p,
            Err(e) => return runtime::cli::usage_error(&SPEC, &e),
        }
    };
    if let Some(extra) = parsed.rest.first() {
        return runtime::cli::usage_error(&SPEC, &format!("Unexpected argument: {extra}"));
    }
    if let Some(absent) = runtime::cli::missing_required(&SPEC, &parsed) {
        return runtime::cli::usage_error(&SPEC, &absent);
    }
    let (floor, ceiling) = match tolerance("--cosine-floor", floor_arg, COSINE_FLOOR)
        .and_then(|f| tolerance("--rel-l2-ceiling", ceiling_arg, REL_L2_CEILING).map(|c| (f, c)))
    {
        Ok(pair) => pair,
        Err(e) => return runtime::cli::usage_error(&SPEC, &e),
    };
    let expect = |v: Option<String>| PathBuf::from(v.expect("checked by missing_required above"));
    let (model, reference) = (expect(model), expect(reference));

    match run(
        &model,
        &reference,
        parsed.has("--text-only"),
        parsed.has("--single-thread"),
        floor,
        ceiling,
    ) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            runtime::status::error(&format!("{e:#}"));
            ExitCode::from(1)
        }
    }
}
