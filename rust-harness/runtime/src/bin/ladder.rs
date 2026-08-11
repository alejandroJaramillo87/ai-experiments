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
use runtime::model::cpu::{CpuModel, PRELUDE_STAGES};

/// Run one rung, or `None` when its reference artifact is absent.
///
/// Every arm names the file it looked for, because "skipped" without a path
/// is an invitation to assume the rung is unimplemented rather than
/// unreferenced.
fn run_rung(
    rung: Rung,
    model: &mut CpuModel,
    model_path: &Path,
    reference_dir: &Path,
    size: Size,
    tol: Tolerance,
    // The prompt length L1 and L2 scored, reported back so the table can say
    // how many tokens the agreement covers. Only these rungs know it -- they
    // are the ones that read the capture -- and it is not the sized corpus.
    prompt_tokens: &mut Option<usize>,
) -> Result<Option<RungResult>> {
    match rung {
        Rung::L0Tokenizer => {
            let path = reference_dir.join("tokenizer_parity.json");
            let Some(cases) = runtime::reference::load_tokenizer_parity(
                &path,
                runtime::tokenizer::CHAT_TEMPLATE_VERSION,
            )?
            else {
                return Ok(None);
            };
            // Resolved and loaded inside the arm: a run with no L0 artifact --
            // the ordinary case for a reference directory written before this
            // rung existed -- should not pay to read a 30 MB tokenizer.json.
            let tokenizer_path = runtime::tokenizer::resolve_tokenizer(model_path, None)?;
            let tok = runtime::tokenizer::Tokenizer::from_file(&tokenizer_path)?;
            ladder::rung_tokenizer(&tok, &cases).map(Some)
        }
        Rung::L1Layers => {
            let path = reference_dir.join("layers.json");
            let Some(reference) = runtime::reference::load_capture(&path, &PRELUDE_STAGES)? else {
                return Ok(None);
            };
            *prompt_tokens = Some(reference.token_ids.len());
            let ours = ladder::capture_ours(model, &reference.token_ids, &reference.activations)?;
            ladder::rung_layers(&ours, &reference.activations, tol).map(Some)
        }
        Rung::L2Logits => {
            let path = reference_dir.join("layers.json");
            let Some(reference) = runtime::reference::load_capture(&path, &PRELUDE_STAGES)? else {
                return Ok(None);
            };
            // A layers-only capture is a usable L1 reference and no L2
            // reference at all. Skipped rather than failed: the artifact says
            // nothing about the distribution, which is not the same as saying
            // the distribution is wrong.
            let Some(row) = reference.logits else {
                return Ok(None);
            };
            *prompt_tokens = Some(reference.token_ids.len());
            let last = reference.token_ids.len().saturating_sub(1);
            let logits = ladder::LogitReference {
                positions: vec![last],
                rows: vec![row],
            };
            ladder::rung_logits(model, &reference.token_ids, &logits, tol).map(Some)
        }
        Rung::L3Perplexity => {
            let path = reference_dir.join("ppl.json");
            let corpus_path =
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/reference/corpus.txt");
            let Some((corpus, reference)) =
                runtime::reference::load_perplexity(&path, &corpus_path)?
            else {
                return Ok(None);
            };
            // The artifact scored a prefix of the corpus chosen by its own
            // --size. A mismatch here means the reference and this run
            // disagree about how much text is being measured, which would
            // otherwise surface as a scored-token-count error inside the rung
            // and read as a numerics problem.
            anyhow::ensure!(
                corpus.len() == size.tokens(),
                "perplexity artifact scored {} tokens and size {} reads {} -- \
                 regenerate it at this size",
                corpus.len(),
                size.label(),
                size.tokens()
            );
            ladder::rung_perplexity(model, &corpus, &reference).map(Some)
        }
        Rung::L4Greedy => {
            let path = reference_dir.join("greedy.json");
            let Some(references) = runtime::reference::load_greedy(&path, size.prompts())? else {
                return Ok(None);
            };
            ladder::rung_greedy(model, &references).map(Some)
        } // Every rung is wired; the match stays exhaustive by arm rather than
          // by wildcard so a sixth rung has to be handled here rather than
          // silently reporting skipped forever.
    }
}

fn run(
    model_path: &Path,
    reference_dir: &Path,
    size: Size,
    only: Option<Rung>,
    text_only: bool,
    threads: Option<usize>,
) -> Result<bool> {
    let files = runtime::checkpoint::CheckpointFiles::read(model_path)
        .with_context(|| format!("read model {}", model_path.display()))?;
    let mut ck = runtime::checkpoint::parse(&files).context("parse checkpoint")?;
    ck.text_only = text_only;
    let mut model = CpuModel::from_checkpoint(&ck).context("load CPU model")?;
    if let Some(n) = threads {
        model.set_threads(n);
    }

    // A rung that errors is not a rung that failed: an unreadable or
    // mismatched artifact says nothing about the engine. It is raised rather
    // than folded into the table, so it cannot be read as a red rung.
    let mut error = None;
    let mut prompt_tokens = None;
    let mut l = ladder::climb(
        size,
        model_path.display().to_string(),
        reference_dir.display().to_string(),
        |rung| match error {
            // --rung isolates one rung; the rest report skipped rather than
            // vanishing, so the table still shows what was not run.
            _ if only.is_some_and(|r| r != rung) => None,
            Some(_) => None,
            None => match run_rung(
                rung,
                &mut model,
                model_path,
                reference_dir,
                size,
                ladder::TIER_E,
                &mut prompt_tokens,
            ) {
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
    l.prompt_tokens = prompt_tokens;
    l.threads = Some(model.threads());
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

const USAGE: &str = "usage: ladder [--text-only] [--single-thread] \
                     [--threads <n>] [--size quick|standard|deep] \
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
    // Neither this nor --threads is a numerics switch: the row split is
    // bit-identical at every width, so every table this run prints matches
    // every other width's digit for digit. --single-thread exists because rr
    // serializes threads anyway; --threads exists because the width that is
    // actually fastest on a given machine has to be swept for, and rebuilding
    // between widths would put a recompile inside the measurement.
    let mut threads = match args.iter().position(|a| a == "--single-thread") {
        Some(i) => {
            args.remove(i);
            Some(1)
        }
        None => None,
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
            "--threads" => match value.parse::<usize>() {
                Ok(n) if n >= 1 => threads = Some(n),
                _ => {
                    eprintln!("[ERROR] --threads {value}: expected a positive integer");
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

    match run(&model, &reference, size, only, text_only, threads) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[ERROR] {e:#}");
            ExitCode::from(1)
        }
    }
}
