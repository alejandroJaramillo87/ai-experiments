/*
 * ladder — run the seven-rung correctness gate against a real checkpoint.
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

use anyhow::Result;

use runtime::cli::{Flag, Kind, Spec};
use runtime::kv::LayerKind;
use runtime::ladder::{self, Rung, RungResult, Size, Tolerance};
use runtime::model::cpu::{BatchKind, CpuModel, DotKind, PRELUDE_STAGES};

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
        }
        Rung::L5LongContext => {
            let path = reference_dir.join("long_context.json");
            let Some(reference) = runtime::reference::load_long_context(&path)? else {
                return Ok(None);
            };
            // The window this run read from the checkpoint, not the one the
            // artifact recorded. Comparing the two is half of what L5
            // measures, so the rung has to be handed our own answer.
            let window = model.config().sliding_window;
            ladder::rung_long_context(model, window, &reference, tol).map(Some)
        }
        Rung::L6Attention => {
            let path = reference_dir.join("attention.json");
            let Some(reference) = runtime::reference::load_attention(&path)? else {
                return Ok(None);
            };
            let captured = ladder::capture_attention(model, &reference)?;
            // Our own reading of the schedule, the head count and the window:
            // the rung recomputes which rows to expect from these and refuses
            // the artifact's choices where they differ.
            let kinds: Vec<LayerKind> = model.layer_specs().iter().map(|l| l.kind).collect();
            let config = model.config();
            ladder::rung_attention(
                &captured,
                &kinds,
                config.n_heads,
                config.sliding_window,
                &reference,
                tol,
            )
            .map(Some)
        } // Every rung is wired; the match stays exhaustive by arm rather than
          // by wildcard so an eighth rung has to be handled here rather than
          // silently reporting skipped forever.
    }
}

/// The engine knobs a run reports in its header.
///
/// Grouped rather than passed one at a time because they travel together:
/// every one of them is set on the model before the climb and printed after
/// it, and the next one added should join them rather than lengthen a
/// signature.
struct Tuning {
    threads: Option<usize>,
    dot: DotKind,
    batch: BatchKind,
}

/// The resolved configuration of a ladder run, as obs fields.
///
/// **Not `run.header`.** That event's field list is asserted for key-set
/// equality in both directions against a real harness run
/// (`harness/tests/obs_contract.rs`), and a ladder has no sampler, no KV
/// cache and no context window to report -- it would have to leave `seed`,
/// `temperature`, `max_tokens`, `max_context`, `kv_bytes` and
/// `kv_batch_group` unwritten. Admitting both writers under one event means
/// moving all six to `optional`, which retires exactly the assertion U2 added
/// them for: that a field silently dropped from the harness header fails a
/// test. Two events, each pinned exactly, is the stronger contract.
///
/// The keys this does share with `run.header` are spelled identically and
/// carry the same units, so one `jq` filter reads both.
fn ladder_header(
    model_path: &Path,
    model_hash: u64,
    reference_dir: &Path,
    size: Size,
    only: Option<Rung>,
    text_only: bool,
    model: &CpuModel,
) -> Vec<(&'static str, obs::Value)> {
    let (weight_bytes, width) = model.weight_bytes();
    let mut f = vec![
        ("model", obs::Value::Str(model_path.display().to_string())),
        ("model_hash", obs::Value::Str(format!("{model_hash:016x}"))),
        ("n_layers", obs::Value::Int(model.config().n_layers as u64)),
        (
            "dtype",
            obs::Value::Str(width.unwrap_or("mixed widths").to_string()),
        ),
        ("weight_bytes", obs::Value::Int(weight_bytes as u64)),
        (
            "reference",
            obs::Value::Str(reference_dir.display().to_string()),
        ),
        ("size", obs::Value::Str(size.label().to_string())),
        ("corpus_tokens", obs::Value::Int(size.tokens() as u64)),
        ("text_only", obs::Value::Bool(text_only)),
        ("threads", obs::Value::Int(model.threads() as u64)),
        ("dot", obs::Value::Str(model.dot().label().to_string())),
        ("batch", obs::Value::Str(model.batch().label())),
        (
            "binary_version",
            obs::Value::Str(env!("CARGO_PKG_VERSION").to_string()),
        ),
        ("git_sha", obs::Value::Str(runtime::envelope::git_sha())),
    ];
    // Absent rather than null when the whole ladder ran: `--rung` names the
    // one rung a run isolated to, and "no isolation" is not a rung.
    if let Some(r) = only {
        f.push(("only_rung", obs::Value::Str(r.label().to_string())));
    }
    f
}

/// A measure's display name as a JSON key: `worst rel_l2` -> `worst_rel_l2`.
///
/// The names are written for a table column, so they carry spaces and
/// hyphens. Slugified rather than given a second `key` field on `Measure`,
/// because a second name is a second thing to keep in step and the table is
/// the one a human reads. The coupling is real and worth stating: renaming a
/// column renames a metrics key.
fn slug(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        match c {
            'a'..='z' | '0'..='9' => out.push(c),
            'A'..='Z' => out.extend(c.to_lowercase()),
            _ if out.ends_with('_') => {}
            _ => out.push('_'),
        }
    }
    out.trim_matches('_').to_string()
}

fn run(
    model_path: &Path,
    reference_dir: &Path,
    size: Size,
    only: Option<Rung>,
    text_only: bool,
    tuning: Tuning,
    log: &mut obs::RunLog,
) -> Result<bool> {
    // `load` drops the checkpoint mapping before returning; see its doc
    // comment for why no caller here should open one itself.
    let (mut model, fingerprint) = {
        let _span = log.phase("model.load");
        CpuModel::load(model_path, text_only)?
    };
    if let Some(n) = tuning.threads {
        model.set_threads(n);
    }
    model.set_dot(tuning.dot);
    model.set_batch(tuning.batch);

    // After the tuning is applied, so the header reports the fold and the
    // width the climb actually used rather than the ones it was asked for.
    // That distinction is the whole reason this record exists: `DotKind`
    // defaulted to `Sequential` for months with nothing printing the fold.
    let header = ladder_header(
        model_path,
        fingerprint,
        reference_dir,
        size,
        only,
        text_only,
        &model,
    );
    log.event(
        "info",
        "ladder.header",
        "resolved ladder configuration",
        &header,
    );
    log.metrics_header(&header);

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
            None => {
                // Only around a rung the closure actually attempts. A rung
                // that `--rung` isolated away, or that a prior failure
                // stopped, did no work, and a zero-length span in the event
                // log would say it ran and was instant.
                let _span = log.phase(&format!("rung.{}", rung.label()));
                match run_rung(
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
                }
            }
        },
    );
    if let Some(e) = error {
        return Err(e);
    }
    l.prompt_tokens = prompt_tokens;
    l.threads = Some(model.threads());
    l.dot = Some(model.dot().label().to_string());
    l.batch = Some(model.batch().label());

    // One metrics record per rung, written from the finished ladder rather
    // than from inside the climb: `elapsed` is stamped by `climb` around the
    // closure, so it is the only figure here no rung body can under-report.
    //
    // `step` is the rung index, which makes this a category axis wearing a
    // continuum's clothes -- seven points, ordered but not spaced. That is
    // honest for `rung_duration_s`, the series the dashboard charts, because
    // the rungs really do run in that order. It is why the measures ride
    // along un-charted: `worst kl` and `ppl delta` are different quantities
    // in different units, and a line through them would mean nothing. They
    // are here to be read with `jq`, and to be diffed between two runs.
    for (i, r) in l.results.iter().enumerate() {
        let keys: Vec<String> = r.measures.iter().map(|m| slug(m.name)).collect();
        let mut fields = vec![
            ("rung", obs::Value::Str(r.rung.label().to_string())),
            ("status", obs::Value::Str(r.status.label().to_string())),
        ];
        // Absent for a rung the climb never reached. An absent duration and a
        // zero duration are different claims, which is why `elapsed` is an
        // Option in the first place.
        if let Some(d) = r.elapsed {
            fields.push((
                "rung_duration_s",
                obs::Value::Num(obs::round3(d.as_secs_f64())),
            ));
        }
        for (k, m) in keys.iter().zip(&r.measures) {
            fields.push((k.as_str(), obs::Value::Num(m.value)));
        }
        log.metric(i as u64, &fields);
    }
    // The table is the ladder's answer and stays on stdout, which is the
    // payload channel. The verdict is status: it goes to stderr through
    // `runtime::status`, so a redirect that captures the table for comparison
    // does not also capture a line that says whether it passed.
    print!("{}", ladder::render(&l));
    for (tag, line) in ladder::summary(&l) {
        match tag {
            Some(runtime::status::Tag::Ok) => runtime::status::ok(&line),
            Some(runtime::status::Tag::Warn) => runtime::status::warn(&line),
            Some(runtime::status::Tag::Error) => runtime::status::error(&line),
            // The verdict emits only those three; anything else, and this
            // arm is what says so rather than a silently dropped line.
            Some(other) => runtime::status::plain(&runtime::status::line(
                other,
                &line,
                0,
                runtime::status::color_enabled(),
            )),
            None => runtime::status::plain(&line),
        }
    }
    // Every rung skipped is not a pass. The ladder library is right to treat
    // a skip as neutral -- it does not know whether the caller asked for one
    // rung -- but a run that measured nothing at all and exited 0 would read
    // as a green gate, and the usual cause is a --reference that points
    // somewhere empty.
    if l.skipped() == ladder::Rung::ALL.len() {
        runtime::status::error(&format!(
            "Ladder: no reference artifacts under {} -- nothing was measured",
            reference_dir.display()
        ));
        runtime::status::plain(
            "  See rust-harness/fixtures/reference/README.md for how to generate them.",
        );
        return Ok(false);
    }
    Ok(l.ok())
}

const SPEC: Spec = Spec {
    program: "ladder",
    summary: "Run the seven-rung correctness gate against a transformers reference.",
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
            help: "the checkpoint to climb; required",
        },
        Flag {
            name: "--reference",
            kind: Kind::Value("<reference-dir>"),
            help: "directory of transformers reference artifacts to compare against; required",
        },
        Flag {
            name: "--size",
            kind: Kind::Value("quick|standard|deep"),
            help: "how much corpus the perplexity and greedy rungs read",
        },
        Flag {
            name: "--rung",
            kind: Kind::Value("L0|L1|L2|L3|L4|L5|L6"),
            help: "climb one rung rather than all seven",
        },
        Flag {
            name: "--dot",
            kind: Kind::Value("<kind>"),
            help: "the dot-product kernel; every kind is bit-identical",
        },
        Flag {
            name: "--batch",
            kind: Kind::Value("stepped|whole|<n>"),
            help: "how many positions one forward pass covers",
        },
        Flag {
            name: "--threads",
            kind: Kind::Value("<n>"),
            help: "row-split width; the tables are bit-identical at every width",
        },
        Flag {
            name: "--text-only",
            kind: Kind::Bare,
            help: "treat a multimodal checkpoint as its text tower",
        },
        Flag {
            name: "--single-thread",
            kind: Kind::Bare,
            help: "shorthand for --threads 1, for rr, which serializes threads anyway",
        },
    ],
};

/// Report the run's peak resident set size.
///
/// The ladder is where memory claims get settled, so it states the number
/// itself instead of leaving it to whoever remembers `/usr/bin/time -v`.
/// Silent where `/proc` cannot answer -- an absent measurement must not be
/// mistaken for a small one.
fn report_peak_rss() {
    if let Some(bytes) = runtime::peak_rss_bytes() {
        runtime::status::info(&format!(
            "Peak RSS at exit: {:.2} GiB",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // Answered before anything else, and in particular before `RunLog::begin`
    // below: a query is not a run, so asking how to use the ladder must leave
    // no row in the catalog. The old pair loop consumed a value for every
    // token it saw, so `--help` alone reported "missing value".
    if runtime::cli::handled_help(&SPEC, &args) {
        return ExitCode::SUCCESS;
    }
    // The record carries what was typed rather than what survived parsing.
    let argv_full = args.clone();

    let (mut model, mut reference, mut size_arg, mut rung_arg) = (None, None, None, None);
    let (mut dot_arg, mut batch_arg, mut threads_arg) = (None, None, None);
    let parsed = {
        let mut slots: [(&'static str, &mut Option<String>); 7] = [
            ("--model", &mut model),
            ("--reference", &mut reference),
            ("--size", &mut size_arg),
            ("--rung", &mut rung_arg),
            ("--dot", &mut dot_arg),
            ("--batch", &mut batch_arg),
            ("--threads", &mut threads_arg),
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

    // Neither --single-thread nor --threads is a numerics switch: the row split
    // is bit-identical at every width, so every table this run prints matches
    // every other width's digit for digit. --single-thread exists because rr
    // serializes threads anyway; --threads exists because the width that is
    // actually fastest on a given machine has to be swept for, and rebuilding
    // between widths would put a recompile inside the measurement.
    let mut threads = parsed.has("--single-thread").then_some(1);
    let mut dot = DotKind::default();
    let mut batch = BatchKind::default();
    let mut size = Size::Quick;
    let mut only: Option<Rung> = None;
    // Each value keeps the error message naming its own accepted set, which is
    // the reason `cli::extract` hands back strings rather than parsing them.
    if let Some(value) = size_arg {
        match Size::parse(&value) {
            Some(s) => size = s,
            None => {
                return runtime::cli::usage_error(
                    &SPEC,
                    &format!("--size {value}: expected quick, standard, or deep"),
                )
            }
        }
    }
    if let Some(value) = dot_arg {
        match DotKind::parse(&value) {
            Some(k) => dot = k,
            None => {
                return runtime::cli::usage_error(
                    &SPEC,
                    &format!(
                        "--dot {value}: expected one of {}",
                        DotKind::ALL.map(|k| k.label()).join(", ")
                    ),
                )
            }
        }
    }
    if let Some(value) = batch_arg {
        match BatchKind::parse(&value) {
            Some(b) => batch = b,
            None => {
                return runtime::cli::usage_error(
                    &SPEC,
                    &format!("--batch {value}: expected stepped, whole, or a positive integer"),
                )
            }
        }
    }
    if let Some(value) = threads_arg {
        match value.parse::<usize>() {
            Ok(n) if n >= 1 => threads = Some(n),
            _ => {
                return runtime::cli::usage_error(
                    &SPEC,
                    &format!("--threads {value}: expected a positive integer"),
                )
            }
        }
    }
    if let Some(value) = rung_arg {
        match Rung::ALL.into_iter().find(|r| r.label() == value) {
            Some(r) => only = Some(r),
            None => {
                return runtime::cli::usage_error(
                    &SPEC,
                    &format!("--rung {value}: expected one of L0 L1 L2 L3 L4 L5 L6"),
                )
            }
        }
    }
    let text_only = parsed.has("--text-only");
    let expect = |v: Option<String>| PathBuf::from(v.expect("checked by missing_required above"));
    let (model, reference) = (expect(model), expect(reference));

    let tuning = Tuning {
        threads,
        dot,
        batch,
    };
    // Opened only once the arguments are known good, so a usage mistake
    // leaves no row in the catalog. Same rule as `harness --help` and
    // `--status`: a question that never started a climb is not a run.
    //
    // `cli` is "harness" for the whole workspace (obs::CLI_NAME) with the
    // binary name as `command`, so one catalog and one `runs` view serve
    // every tool here.
    let mut log = obs::RunLog::begin("ladder", &argv_full);
    let verdict = run(&model, &reference, size, only, text_only, tuning, &mut log);
    // After the run, so the figure is the whole run's high-water mark rather
    // than the loader's. Printed even on failure: a rung that failed because
    // the machine was swapping is worth being able to see.
    report_peak_rss();
    match verdict {
        Ok(true) => {
            log.end(0);
            ExitCode::SUCCESS
        }
        // A red rung is a completed run that answered "no", not a crash, so
        // it takes the ordinary exit path and no `fail` record. The failing
        // rung is already in the table and in this run's metrics.
        Ok(false) => {
            log.end(1);
            ExitCode::from(1)
        }
        Err(e) => {
            runtime::status::error(&format!("{e:#}"));
            log.fail(&e, 1);
            log.end(1);
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::slug;

    /// The four shapes the real measure names take. These are the actual
    /// strings `runtime::ladder` builds, so this is a golden of the keys that
    /// reach `metrics.jsonl` rather than an exercise of the function on
    /// invented input -- which is the difference between pinning the contract
    /// and pinning the implementation.
    #[test]
    fn the_measure_names_slugify_to_the_keys_the_metrics_file_carries() {
        assert_eq!(slug("worst cosine"), "worst_cosine");
        assert_eq!(slug("worst rel_l2"), "worst_rel_l2");
        assert_eq!(slug("argmax agreement"), "argmax_agreement");
        // Built at runtime from the configured k, and the hyphen is why this
        // needs a slug at all rather than a `replace(' ', "_")`.
        assert_eq!(slug("worst top-50 overlap"), "worst_top_50_overlap");
        assert_eq!(slug("mean common prefix"), "mean_common_prefix");
        assert_eq!(slug("ppl delta"), "ppl_delta");
        assert_eq!(slug("captured layers agree"), "captured_layers_agree");
        assert_eq!(slug("weight outside window"), "weight_outside_window");
        assert_eq!(slug("worst row max abs diff"), "worst_row_max_abs_diff");
        assert_eq!(slug("worst top-10 overlap"), "worst_top_10_overlap");
    }

    /// A run of separators collapses, and neither end keeps one. A key like
    /// `worst__top_50_` is legal JSON and parses fine, which is exactly why
    /// nothing downstream would report it -- it would simply be a different
    /// key from the one a saved `jq` filter names.
    #[test]
    fn separators_collapse_and_never_bound_the_key() {
        assert_eq!(slug("  worst   top-50  "), "worst_top_50");
        assert_eq!(slug("(scaled)"), "scaled");
    }

    /// Uppercase folds rather than being dropped. `Status::label` returns
    /// `PASS`, and while that is a value here rather than a key, a measure
    /// named with a capital would otherwise lose the letter entirely and
    /// silently shorten its key.
    #[test]
    fn capitals_fold_rather_than_vanish() {
        assert_eq!(slug("KL divergence"), "kl_divergence");
    }
}
