/*
 * ladder — run the five-rung correctness gate against a real checkpoint.
 *
 * Loads whichever reference artifacts exist under --reference, climbs the
 * rungs in order, prints one table, and exits non-zero on the first failure.
 * A rung whose artifact is absent reports `skipped` and does not stop the
 * climb: missing evidence is not contrary evidence, and Tier E artifacts are
 * generated per run rather than committed, so absence is the ordinary case.
 *
 * The verdict logic lives in runtime::ladder, which is unit-tested against a
 * scripted mock with no model at all. This binary is the part that cannot be
 * tested that way -- loading a checkpoint, driving a real forward pass -- and
 * it is kept thin for exactly that reason.
 */

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};

use runtime::ladder::{self, Rung, RungResult, Size, Tolerance};
use runtime::model::cpu::{CpuModel, LastRowTap, PRELUDE_STAGES};
use runtime::model::Model;

/// Run one rung, or `None` when its reference artifact is absent.
///
/// Every arm names the file it looked for, because "skipped" without a path
/// is an invitation to assume the rung is unimplemented rather than
/// unreferenced.
fn run_rung(
    rung: Rung,
    model: &mut CpuModel,
    reference_dir: &Path,
    tol: Tolerance,
) -> Result<Option<RungResult>> {
    match rung {
        Rung::L1Layers => {
            let path = reference_dir.join("layers.json");
            let Some(reference) = runtime::reference::load_capture(&path, &PRELUDE_STAGES)? else {
                return Ok(None);
            };
            let n_layers = model.config().n_layers;
            let tap = LastRowTap::new((0..n_layers).collect()).capturing_prelude();
            model.set_tap(Box::new(tap.clone()));
            model.truncate(0)?;
            model
                .prefill(&reference.token_ids)
                .context("prefill the reference tokens")?;

            // Only the stages the reference measured. Asking the tap for a
            // stage it never saw would be an error inside the rung, where it
            // would read as a divergence rather than as a capture that did
            // not run.
            let mut stages = Vec::new();
            for (name, _) in &reference.activations.stages {
                let row = tap
                    .stage(name)
                    .with_context(|| format!("{name}: the tap captured no row"))?;
                stages.push((name.clone(), row));
            }
            let mut layers = Vec::with_capacity(n_layers);
            for i in 0..n_layers {
                layers.push(
                    tap.row(i)
                        .with_context(|| format!("layer {i}: the tap captured no row"))?,
                );
            }
            let ours = ladder::Activations { stages, layers };
            ladder::rung_layers(&ours, &reference.activations, tol).map(Some)
        }
        // The remaining rungs' artifacts do not have a format yet: L0's
        // tokenizer parity file and the tiny transformers fixture land with
        // the generator that writes them. Reported as skipped, which is
        // exactly what they are -- no reference, rather than no rung.
        _ => Ok(None),
    }
}

fn run(
    model_path: &Path,
    reference_dir: &Path,
    size: Size,
    only: Option<Rung>,
    text_only: bool,
) -> Result<bool> {
    let files = runtime::checkpoint::CheckpointFiles::read(model_path)
        .with_context(|| format!("read model {}", model_path.display()))?;
    let mut ck = runtime::checkpoint::parse(&files).context("parse checkpoint")?;
    ck.text_only = text_only;
    let mut model = CpuModel::from_checkpoint(&ck).context("load CPU model")?;

    // A rung that errors is not a rung that failed: an unreadable or
    // mismatched artifact says nothing about the engine. It is raised rather
    // than folded into the table, so it cannot be read as a red rung.
    let mut error = None;
    let l = ladder::climb(
        size,
        model_path.display().to_string(),
        reference_dir.display().to_string(),
        |rung| match error {
            // --rung isolates one rung; the rest report skipped rather than
            // vanishing, so the table still shows what was not run.
            _ if only.is_some_and(|r| r != rung) => None,
            Some(_) => None,
            None => match run_rung(rung, &mut model, reference_dir, ladder::TIER_E) {
                Ok(result) => result,
                Err(e) => {
                    error = Some(e.context(rung.label()));
                    None
                }
            },
        },
    );
    if let Some(e) = error {
        return Err(e);
    }
    print!("{}", ladder::render(&l));
    // Every rung skipped is not a pass. The ladder library is right to treat
    // a skip as neutral -- it does not know whether the caller asked for one
    // rung -- but a run that measured nothing at all and exited 0 would read
    // as a green gate, and the usual cause is a --reference that points
    // somewhere empty.
    if l.skipped() == ladder::Rung::ALL.len() {
        eprintln!(
            "[ERROR] Ladder: no reference artifacts under {} -- nothing was measured",
            reference_dir.display()
        );
        eprintln!("  See rust-harness/fixtures/reference/README.md for how to generate them.");
        return Ok(false);
    }
    Ok(l.ok())
}

const USAGE: &str = "usage: ladder [--text-only] [--size quick|standard|deep] \
                     [--rung L0|L1|L2|L3|L4] --model <checkpoint-dir> \
                     --reference <reference-dir>";

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    let text_only = match args.iter().position(|a| a == "--text-only") {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    };

    let mut size = Size::Quick;
    let mut only: Option<Rung> = None;
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
            "--size" => match Size::parse(&value) {
                Some(s) => size = s,
                None => {
                    eprintln!("[ERROR] --size {value}: expected quick, standard, or deep");
                    return ExitCode::from(2);
                }
            },
            "--rung" => match Rung::ALL.into_iter().find(|r| r.label() == value) {
                Some(r) => only = Some(r),
                None => {
                    eprintln!("[ERROR] --rung {value}: expected one of L0 L1 L2 L3 L4");
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

    match run(&model, &reference, size, only, text_only) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[ERROR] {e:#}");
            ExitCode::from(1)
        }
    }
}
