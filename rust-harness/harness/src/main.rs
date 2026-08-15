/*
 * harness — single-binary local coding agent.
 *
 * Wires the full stack: permissions from permissions.yaml, a durable JSONL
 * transcript, the agent loop, and a SessionGenerator over the runtime. The
 * generator runs the hand-rolled CPU backend against the checkpoint passed
 * via --model; the cuda backend (phase A) will slot in at the same seam.
 */

// See the note in lib.rs: the binary is its own crate root, so the attribute
// has to be repeated to cover it.
#![forbid(unsafe_code)]

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context, Result};

use harness::agent::Agent;
use harness::generator::SessionGenerator;
use harness::interplog::{InterpLog, MonitorHandle};
use harness::permissions::Permissions;
use harness::tools::ToolCall;
use harness::transcript::Transcript;

/// Interactive Ask resolution: prompt on stderr, read one line from stdin.
///
/// The one stderr write in this binary that does not go through
/// `runtime::status`. A prompt is not a status line: it must leave the cursor
/// on the same line for the answer, and `status` has no emitter that omits the
/// newline -- deliberately, since a tag whose line never ends is unparseable.
fn ask_stdin(call: &ToolCall) -> bool {
    eprint!("allow {} {}? [y/N] ", call.name(), call.target());
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return false;
    }
    matches!(line.trim(), "y" | "yes")
}

/// The forward-pass record a run accumulates, when one was asked for.
///
/// `None` rather than an empty vector, so "alignment was not requested" and
/// "the model made no passes" stay distinguishable in the writer.
type PassHandle = Option<std::sync::Arc<std::sync::Mutex<Vec<runtime::align::Pass>>>>;

// `UNKNOWN_MODEL_HASH`, `check_model_identity` and `control_vector_model_hash`
// live in `harness::preflight`, which needs them to answer the same questions
// before a model is loaded. They are `pub` there rather than crate-private
// because this binary is a separate crate from the library it consumes.

/// Print a preflight report and turn it into an exit code.
///
/// The only renderer over `preflight::Check`, which is why `checks` returns
/// data and tags nothing itself.
///
/// A check's first detail line goes on the tagged line; any further lines are
/// indented and **untagged**. That is the repo's error shape
/// (`rust-harness/Makefile:70-86`): one tagged statement of what happened,
/// then detail. `status::error_at` would tag every continuation, turning one
/// finding into what reads like several.
///
/// Exit 1, not 2, when something would stop the run: 2 is reserved for a
/// command line that was wrong to ask, and the whole value of `--status` is
/// that the two are different answers. Warnings and skips do not fail.
fn report(checks: &[harness::preflight::Check]) -> ExitCode {
    use harness::preflight::Outcome;
    for check in checks {
        let mut lines = check.detail.lines();
        let first = lines.next().unwrap_or_default();
        let head = format!("{}: {first}", check.name);
        match check.outcome {
            Outcome::Ok => runtime::status::ok(&head),
            Outcome::Warn => runtime::status::warn(&head),
            Outcome::Error => runtime::status::error(&head),
            Outcome::Skip => runtime::status::skip(&head),
        }
        for line in lines {
            runtime::status::plain(&format!("        {line}"));
        }
    }
    match harness::preflight::failed(checks) {
        true => ExitCode::from(1),
        false => ExitCode::SUCCESS,
    }
}

/// Parse `--control-vector-layer-range` as an inclusive pair of 0-based
/// block indices.
///
/// The value is one argument holding two numbers, `"START END"`, rather than
/// two arguments. That is the spelling `interp/steering/wiring.py`'s
/// `steering_env` already emits into `CONTROL_VECTOR_LAYER_RANGE`, so the
/// environment variable can be forwarded to this flag unmodified; llama.cpp
/// took the same string as two shell words instead. Either separator run of
/// whitespace is accepted because the variable is assembled by shell.
fn parse_layer_range(value: &str) -> Result<(usize, usize), String> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    let [start, end] = parts[..] else {
        return Err(format!(
            "--control-vector-layer-range: expected two numbers \"START END\", got {value:?}"
        ));
    };
    let parsed = |s: &str| {
        s.parse::<usize>()
            .map_err(|_| format!("--control-vector-layer-range: {s:?} is not a block index"))
    };
    let (start, end) = (parsed(start)?, parsed(end)?);
    if start > end {
        return Err(format!(
            "--control-vector-layer-range: start {start} is above end {end}"
        ));
    }
    Ok((start, end))
}

/// Construct the generator over the CPU backend for the checkpoint at
/// `model_path`, wrapping the model in the flight recorder when `record`
/// names a path. There is no mock fallback in the shipping binary.
#[allow(clippy::too_many_arguments)]
fn build_generator(
    model_path: Option<&Path>,
    tokenizer_path: Option<&Path>,
    record: Option<&Path>,
    control_vector: Option<&Path>,
    control_scale: f32,
    control_layer_range: Option<(usize, usize)>,
    probe: Option<&Path>,
    sampling: runtime::sampler::SamplingParams,
    max_tokens: usize,
    context: Option<usize>,
    snapshot: Option<&Path>,
    text_only: bool,
    single_thread: bool,
    align: bool,
    keep_scores: bool,
) -> Result<(
    SessionGenerator,
    Option<MonitorHandle>,
    u64,
    PassHandle,
    harness::runheader::RunHeader,
)> {
    use runtime::model::{
        cpu::{ControlVector, CpuModel},
        probe::{Probe, ProbeMonitor},
        record::RecordingModel,
        Model,
    };
    let Some(model_path) = model_path else {
        anyhow::bail!(
            "no model: pass --model <checkpoint-dir> to run on the CPU backend; \
             the cuda backend (roadmap phase A) has not landed"
        )
    };
    // `load` drops the checkpoint mapping before returning, so nothing below
    // holds 15 GiB of page cache it will never read; see its doc comment.
    let (mut cpu, model_hash) = CpuModel::load(model_path, text_only)?;
    if single_thread {
        cpu.set_threads(1);
    }
    // Model identity for the interp artifacts. The weights alone, not
    // config.json: a checkpoint is a directory, and the weights are what an
    // artifact was fitted on. Hashed from the file rather than from memory,
    // and only when an artifact is actually present -- see
    // `check_model_identity` on why the digest arrives as a closure.
    let weights_path =
        runtime::checkpoint::resolve(model_path)?.join(runtime::checkpoint::WEIGHTS_FILE);
    if let Some(path) = control_vector {
        let cv_data = std::fs::read(path)
            .with_context(|| format!("read control vector {}", path.display()))?;
        let cv_image = runtime::checkpoint::Image::parse(cv_data, "control vector")
            .context("parse control-vector image")?;
        match harness::preflight::control_vector_model_hash(path) {
            Some(hash) => {
                harness::preflight::check_model_identity("control vector", &hash, || {
                    runtime::digest::sha256_file(&weights_path)
                        .with_context(|| format!("hash {}", weights_path.display()))
                })?;
            }
            None => runtime::status::warn(&format!(
                "Model identity: no readable manifest.json beside {}, \
                 control vector unchecked",
                path.display()
            )),
        }
        let cv = ControlVector::from_image(&cv_image, control_scale, control_layer_range)
            .context("load control vector")?;
        cpu.set_control_vector(cv).context("apply control vector")?;
    }
    // The monitor observes the steered stream: set_control_vector first,
    // then the tap, matching the documented tap semantics.
    let monitor = match probe {
        Some(path) => {
            let p = std::sync::Arc::new(Probe::load(path).context("load probe")?);
            harness::preflight::check_model_identity("probe", &p.model_hash, || {
                runtime::digest::sha256_file(&weights_path)
                    .with_context(|| format!("hash {}", weights_path.display()))
            })?;
            anyhow::ensure!(
                p.width() == cpu.config().embed_dim,
                "probe width {} does not match model embedding dimension {}",
                p.width(),
                cpu.config().embed_dim
            );
            anyhow::ensure!(
                p.layer < cpu.config().n_layers,
                "probe layer {} out of range for a {}-layer model",
                p.layer,
                cpu.config().n_layers
            );
            // Keeping every clean score is what the alignment artifact's
            // second channel is; `probe-flags.jsonl` needs only the flags.
            // Serve mode asks for it too: the interp log's per-request
            // maximum is a maximum over every position, and taken over the
            // flagged ones alone it would be a maximum over "already above
            // threshold" and describe nothing.
            let (tap, state) = ProbeMonitor::with_scores(std::sync::Arc::clone(&p), keep_scores);
            cpu.set_tap(Box::new(tap));
            Some(MonitorHandle { probe: p, state })
        }
        None => None,
    };
    // --context caps the cache below the model's trained window; raising it
    // above the checkpoint's max_position_embeddings is refused because the
    // model was not trained there. The estimate below then warns before a
    // context the CPU backend cannot realistically hold.
    let trained = cpu.config().context_length;
    let max_context = match context {
        Some(n) => {
            anyhow::ensure!(
                n <= trained,
                "--context {n} exceeds the model's trained context length {trained}"
            );
            n
        }
        None => trained,
    };
    // An expression block, not a bare one: it yields the header. Every borrow
    // of `cpu` still ends at its close, which matters because `cpu` is wrapped
    // and moved into the session a few lines below -- lifting these locals out
    // instead would extend `c: &Config` past that move.
    let header = {
        let c = cpu.config();
        let n_kv_heads = c.n_kv_heads;
        let n_layers = c.n_layers;
        // Geometry per layer, taken from the loaded spec rather than from one
        // model-wide head width. Gemma 4's global layers are twice as wide as
        // its sliding layers, so a single width silently under-reports the
        // budget across every global layer -- enough to keep the warning below
        // from firing on a checkpoint that would exceed it.
        //
        // Shared-KV layers are skipped rather than listed: they store nothing,
        // and counting them over-reports by the whole shared tail -- 18 of
        // E4B's 42 layers, 20 of E2B's 35.
        let layers: Vec<_> = cpu
            .layer_specs()
            .iter()
            .enumerate()
            .filter(|(i, l)| l.kv_source == *i)
            .map(|(_, l)| {
                (
                    l.kind,
                    runtime::kv::KvGeometry {
                        n_kv_heads,
                        head_dim: l.head_dim,
                        bytes_per_element: 4, // CPU backend caches f32
                    },
                )
            })
            .collect();
        // The batch width is an input to the budget, not a detail: a windowed
        // layer holds `window - 1 + group` rows, so the number below is only
        // true for the batch this run will actually use. At `--batch whole`
        // the group is the whole prefill and the cap saves nothing, which the
        // figure then says out loud rather than under-reporting.
        // Weights first, because it is the larger number and the one a reader
        // is checking against the checkpoint on disk. A bf16 checkpoint held
        // at its published width reports about the file's size; twice that
        // would mean the loader had expanded it.
        let (weight_bytes, width) = cpu.weight_bytes();
        // Peak sampled here rather than at construction below. A later sample
        // is a different number reported under the same name: this one is the
        // high-water mark of the load, which is the diagnostic -- against the
        // weight figure above it says whether the checkpoint mapping was
        // dropped, since a load that kept it reports roughly twice the weights.
        let peak_rss_bytes = runtime::peak_rss_bytes();
        let group = cpu.batch().chunk(max_context);
        let bytes = runtime::kv::budget_bytes(&layers, max_context, group);
        let header = harness::runheader::RunHeader {
            model: model_path.display().to_string(),
            model_hash: format!("{model_hash:016x}"),
            n_layers: n_layers as u64,
            dtype: width.unwrap_or("mixed widths").to_string(),
            weight_bytes: weight_bytes as u64,
            peak_rss_bytes,
            max_context: max_context as u64,
            kv_bytes: bytes as u64,
            kv_batch_group: group as u64,
            rewind_horizon: cpu.rewind_horizon(max_context).map(|n| n as u64),
            threads: cpu.threads() as u64,
            dot: cpu.dot().label().to_string(),
            batch: cpu.batch().label(),
            seed: sampling.seed,
            temperature: sampling.temperature,
            max_tokens: max_tokens as u64,
            binary_version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha: runtime::envelope::git_sha(),
            probe: monitor.as_ref().map(|m| m.probe.name.clone()),
            probe_layer: monitor.as_ref().map(|m| m.probe.layer as u64),
        };
        for line in header.lines() {
            runtime::status::info(&line);
        }
        // Outside `lines()`, and the boundary is worth stating: the header
        // renders facts about the configuration, and this is a judgement about
        // one of them. A warning that fired from a field getter would fire in
        // the record too, where it is not a fact anyone can compare across runs.
        if bytes > 2 << 30 {
            runtime::status::warn(
                "KV cache above 2 GiB: consider --context to cap it \
                 (compaction keeps long sessions inside the cap)",
            );
        }
        header
    };
    // Taken here because this is the last point the concrete backend is in
    // hand: below, `cpu` is wrapped and moved into the session.
    let passes: PassHandle = align.then(|| cpu.record_passes());
    let model: Box<dyn Model> = match record {
        // The sampling parameters go into the recording's header, so a replay
        // rebuilds the sampler that chose the tokens rather than a default
        // one that would choose others.
        Some(path) => Box::new(RecordingModel::to_file(
            Box::new(cpu),
            path,
            Some(sampling.into()),
        )?),
        None => Box::new(cpu),
    };
    let mut session = runtime::session::Session::new(model, max_context)?;
    // Snapshot restore: resume the KV cache from the previous clean exit.
    // A header or weight-hash mismatch is loud and the stale file is
    // removed; a missing file is the normal first run. Whether the restored
    // tokens still match the transcript is settled later by sync's prefix
    // matching, which degrades to a full re-prefill at worst. `model_hash` was
    // taken during the load, while the checkpoint was still open.
    if let Some(path) = snapshot {
        match std::fs::File::open(path) {
            Ok(f) => {
                let mut reader = std::io::BufReader::new(f);
                match session.restore(&mut reader, model_hash) {
                    Ok(()) => runtime::status::info(&format!(
                        "Session restored: {} cached tokens from {}",
                        session.cached_len(),
                        path.display()
                    )),
                    // A snapshot is a cache, and the only thing lost by
                    // discarding one is the prefill it would have saved. This
                    // used to abort and ask for a rerun, which removed the
                    // file and then made the user do by hand what the rerun
                    // would do anyway. It matters more now that the payload
                    // format is versioned against the KV layout: every
                    // snapshot written before the sliding-window storage cap
                    // is unreadable, and an upgrade should cost a line of
                    // output rather than a failed startup.
                    Err(e) => {
                        let _ = std::fs::remove_file(path);
                        runtime::status::warn(&format!(
                            "Session snapshot {} discarded: {e:#} \
                             (starting with an empty cache)",
                            path.display()
                        ));
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("open snapshot {}", path.display())),
        }
    }
    let tokenizer_path = runtime::tokenizer::resolve_tokenizer(model_path, tokenizer_path)?;
    let tokenizer =
        runtime::tokenizer::Tokenizer::from_file(&tokenizer_path).context("load tokenizer")?;
    // The alignment table itself is started by the caller, which owns the
    // handle it is written from; only the pass record has to be taken here.
    let generator = SessionGenerator::new(session, tokenizer, sampling, max_tokens)?;
    Ok((generator, monitor, model_hash, passes, header))
}

/// Drain the monitor state: write one JSONL record per flagged position to
/// `out`, print a one-line summary to stderr. A missing parent directory is
/// created rather than treated as a failure, since the flags exist only in
/// memory and are lost the moment this returns. Observation only, so this
/// runs after the task and never affects its result: the caller folds a
/// failure here into a warning through `fold_probe_report`.
fn report_probe(monitor: &MonitorHandle, out: &Path) -> Result<()> {
    let state = monitor.state.lock().expect("monitor state lock");
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create probe output directory {}", parent.display()))?;
    }
    let mut file = std::fs::File::create(out)
        .with_context(|| format!("create probe output {}", out.display()))?;
    for flag in &state.flags {
        let record = serde_json::json!({
            "probe": monitor.probe.name,
            "layer": monitor.probe.layer,
            "position": flag.position,
            "score": flag.score,
            "threshold": monitor.probe.threshold,
        });
        writeln!(file, "{record}").context("write probe flag record")?;
    }
    runtime::status::info(&format!(
        "Probe {}: {} positions scored, {} flagged (threshold {}), flags at {}",
        monitor.probe.name,
        state.scored,
        state.flags.len(),
        monitor.probe.threshold,
        out.display()
    ));
    Ok(())
}

/// Fold a probe report's outcome into the task's. Observation is not part
/// of the task, so a report failure leaves the task result alone and
/// surfaces as a warning line; propagating it instead turned a completed,
/// transcript-recorded task into exit 1 with the answer never printed.
/// Returning the line rather than printing it keeps the policy testable
/// without a live monitor. The message is untagged: `runtime::status` owns the
/// tag, and the caller applies it at the point of emission -- the same split
/// `cli::extract` and `cli::inert` already use.
fn fold_probe_report(
    task: Result<String>,
    reported: Result<()>,
) -> (Result<String>, Option<String>) {
    let warning = reported
        .err()
        .map(|e| format!("Probe flags not written: {e:#}"));
    (task, warning)
}

#[allow(clippy::too_many_arguments)]
fn run(
    root: &Path,
    task: &str,
    model: Option<&Path>,
    tokenizer: Option<&Path>,
    record: Option<&Path>,
    control_vector: Option<&Path>,
    control_scale: f32,
    control_layer_range: Option<(usize, usize)>,
    probe: Option<&Path>,
    probe_out: Option<&Path>,
    alignment_out: Option<&Path>,
    interp_log: Option<&Path>,
    // The log rather than its id. `run` is where the header becomes available
    // -- `main` does not call `build_generator`, and `run` returns
    // `Result<()>`, so there is nowhere to hand a header back up. Passing the
    // log makes the dependency what it already was: `run` needed a shard of a
    // RunLog that `main` still owns.
    //
    // This hands `run` the general ability to write events. U2 writes exactly
    // one, `run.header`, immediately below.
    run_log: &mut obs::RunLog,
    sampling: runtime::sampler::SamplingParams,
    max_tokens: usize,
    context: Option<usize>,
    compact_at: f32,
    chat: bool,
    serve: Option<u16>,
    serve_addr: std::net::IpAddr,
    text_only: bool,
    single_thread: bool,
) -> Result<()> {
    let transcript = Transcript::open(&root.join("session.jsonl")).context("session.jsonl")?;
    // HARNESS.md: persistent memory read once at startup and injected as a
    // synthetic leading turn. A missing file is fine; mid-session edits take
    // effect on the next start (the prompt prefix must stay stable so KV
    // delta prefill keeps working).
    let memory = std::fs::read_to_string(root.join("HARNESS.md")).ok();
    let snapshot_path = root.join("session.kv");
    let (mut generator, monitor, model_hash, passes, header) = build_generator(
        model,
        tokenizer,
        record,
        control_vector,
        control_scale,
        control_layer_range,
        probe,
        sampling,
        max_tokens,
        context,
        Some(&snapshot_path),
        text_only,
        single_thread,
        alignment_out.is_some(),
        // Serve mode keeps every clean score for the per-request maximum,
        // but must not record passes: the pass list is the one artifact with
        // no per-request drain, so under a long-lived server it would grow
        // for the life of the process.
        alignment_out.is_some() || serve.is_some(),
    )?;
    // The resolved configuration, recorded once. The terminal already saw it
    // as `header.lines()` inside `build_generator`; this is the same facts
    // through the same struct, which is the property the stage is for.
    run_log.event(
        "info",
        "run.header",
        "resolved run configuration",
        &header.fields(),
    );
    // Started here rather than in `build_generator`, which does not own the
    // handle the artifact is written from.
    let alignment = alignment_out.map(|_| generator.record_alignment());
    // The task's outcome is held rather than propagated: the bookkeeping
    // below runs either way, so flags collected before a failure survive it.
    let result = if let Some(port) = serve {
        // Serve mode: the OpenAI-shaped endpoint the interp pipeline calls.
        // The control vector and probe monitor are already installed on the
        // model, so a served request runs through the same taps a task does.
        //
        // This blocks until the process is signalled, so the end-of-run
        // bookkeeping below is unreachable in this mode. That is why
        // observation here is per request rather than per run: --interp-log
        // opens an append-only log and every completed request drains the
        // monitor into it. The session snapshot has no such treatment and is
        // simply not taken -- a served session's KV is the accumulation of
        // unrelated callers' traffic, which is not a thing to resume.
        let listener = std::net::TcpListener::bind((serve_addr, port))
            .with_context(|| format!("binding {serve_addr}:{port}"))?;
        let model_name = model
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "synthetic".to_string());
        runtime::status::info(&format!(
            "Serving on http://{serve_addr}:{port} (model {model_name})"
        ));
        let log = match interp_log {
            Some(dir) => {
                let log = InterpLog::open(dir, harness::server::SERVICE_NAME);
                if log.is_enabled() {
                    runtime::status::info(&format!("Interp log: {}", dir.display()));
                } else {
                    // A warning, not a failure: the endpoint's contract is to
                    // answer requests, and it can still do that.
                    runtime::status::warn(&format!(
                        "Interp log not opened under {}",
                        dir.display()
                    ));
                }
                log
            }
            None => InterpLog::disabled(),
        };
        if monitor.is_some() && !log.is_enabled() {
            runtime::status::warn(
                "--probe without --interp-log: scores are computed and discarded, \
                 because a served run has no end at which to report them",
            );
        }
        let state = std::sync::Arc::new(
            harness::server::ServerState::new(generator, model_name, sampling, max_tokens)
                .observing(log, monitor),
        );
        return harness::server::serve(listener, state);
    } else if chat {
        // Chat mode: REPL over the generator, no tools or permissions. The
        // probe monitor and control vector are already installed on the
        // model, so interp flags apply to chat turns unchanged.
        let mut transcript = transcript;
        let stdin = std::io::stdin();
        harness::chat::run_chat(
            &mut generator,
            &mut transcript,
            memory.as_deref(),
            compact_at,
            stdin.lock(),
            std::io::stdout(),
        )
        .map(|turns| {
            // Replies were already streamed to stdout; the summary is status.
            runtime::status::info(&format!("Chat ended: {turns} model turns"));
            String::new()
        })
    } else {
        let permissions = match std::fs::read_to_string(root.join("permissions.yaml")) {
            Ok(yaml) => Permissions::from_yaml(&yaml).context("permissions.yaml")?,
            // No config: every call goes through the interactive Ask path.
            Err(_) => Permissions::from_yaml("rules: []").expect("empty rule set"),
        };
        let mut agent = Agent::new(
            root.to_path_buf(),
            permissions,
            transcript,
            memory.as_deref(),
        );
        agent.compact_at = compact_at;
        agent.run_task(&mut generator, task, 64, ask_stdin)
    };
    if let Ok(answer) = &result {
        // The answer is the task's product: print it before the persistence
        // and observation bookkeeping, so neither can keep a completed
        // task's answer off stdout.
        if !answer.is_empty() {
            println!("{answer}");
        }
        // Clean-exit snapshot: save the KV cache beside the transcript so the
        // next start resumes without a full re-prefill. A save failure degrades
        // to a warning — a completed task must not fail over persistence.
        let saved = std::fs::File::create(&snapshot_path)
            .map_err(anyhow::Error::from)
            .and_then(|f| {
                let mut w = std::io::BufWriter::new(f);
                generator.session().save(&mut w, model_hash)?;
                std::io::Write::flush(&mut w)?;
                Ok(())
            });
        match saved {
            Ok(()) => runtime::status::info(&format!(
                "Session snapshot: {} cached tokens at {}",
                generator.session().cached_len(),
                snapshot_path.display()
            )),
            Err(e) => runtime::status::warn(&format!("Session snapshot not saved: {e:#}")),
        }
    }
    let result = match &monitor {
        Some(monitor) => {
            let default_out = root.join("probe-flags.jsonl");
            let reported = report_probe(monitor, probe_out.unwrap_or(&default_out));
            let (result, warning) = fold_probe_report(result, reported);
            if let Some(warning) = warning {
                runtime::status::warn(&warning);
            }
            result
        }
        None => result,
    };
    // Observation, like the probe flags: a failure here is a warning, never
    // the task's exit code. The same reasoning as `fold_probe_report` --
    // turning a completed, transcript-recorded task into exit 1 because an
    // artifact could not be written loses the answer as well as the artifact.
    if let (Some(out), Some(alignment)) = (alignment_out, alignment.as_ref()) {
        let a = alignment.lock().expect("alignment lock");
        let passes = passes
            .as_ref()
            .map(|p| p.lock().expect("pass record lock").clone())
            .unwrap_or_default();
        let scores = monitor
            .as_ref()
            .map(|m| m.state.lock().expect("monitor state lock").scores.clone())
            .unwrap_or_default();
        let probe_name = monitor.as_ref().map(|m| m.probe.name.clone());
        // A projection of the run header, not a second build of the same
        // facts. `alignment.rs` gains no dependency on `runheader.rs` -- it
        // stays the narrower artifact view -- but the values come from one
        // place. This also removes a sharp edge: `model_hash` used to be
        // `&format!(..)`, a borrow of a temporary living on lifetime
        // extension; it is now a plain field borrow.
        let run_id = run_log.run_id().unwrap_or("unknown").to_string();
        let stamp = harness::alignment::Header {
            run_id: &run_id,
            model_hash: &header.model_hash,
            probe: probe_name.as_deref(),
            probe_layer: monitor.as_ref().map(|m| m.probe.layer),
        };
        match harness::alignment::write(out, &stamp, &a, &passes, &scores) {
            Ok(()) => runtime::status::info(&format!(
                "Alignment: {} positions, {} passes -> {}",
                a.rows.len(),
                passes.len(),
                out.display()
            )),
            Err(e) => runtime::status::warn(&format!("Alignment not written: {e:#}")),
        }
    }
    result?;
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // The meta flags answer and exit before anything is parsed, loaded or
    // logged. They print to stdout, which does not weaken the stdout-is-payload
    // rule: requested help is this invocation's payload, exactly as argparse
    // treats it in the Python tiers. A usage *error* stays on stderr at exit 2.
    // Exiting here also means no obs record, matching the rule that a query is
    // not a run -- and matching the usage errors below, which likewise return
    // before RunLog::begin.
    //
    // Only the region before `--` is scanned, so a task containing the word
    // --help is still a task.
    let meta: Vec<&String> = args.iter().take_while(|a| *a != "--").collect();
    if meta.iter().any(|a| *a == "--help" || *a == "-h") {
        println!("{}", harness::cli::help());
        return ExitCode::SUCCESS;
    }
    if meta.iter().any(|a| *a == "--version" || *a == "-V") {
        println!("harness {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    drop(meta);
    // Snapshot before extraction consumes args: cmd.start records the
    // invocation as given, program name excluded (the shape common/obs logs).
    let argv_full = args.clone();
    let mut model: Option<std::path::PathBuf> = None;
    let mut tokenizer: Option<std::path::PathBuf> = None;
    let mut record: Option<std::path::PathBuf> = None;
    let mut trace_chrome: Option<std::path::PathBuf> = None;
    let mut control_vector: Option<std::path::PathBuf> = None;
    let mut control_scale_arg: Option<std::path::PathBuf> = None;
    let mut control_range_arg: Option<std::path::PathBuf> = None;
    let mut probe: Option<std::path::PathBuf> = None;
    let mut probe_out: Option<std::path::PathBuf> = None;
    let mut alignment_out: Option<std::path::PathBuf> = None;
    let mut interp_log: Option<std::path::PathBuf> = None;
    let mut temperature_arg: Option<std::path::PathBuf> = None;
    let mut top_p_arg: Option<std::path::PathBuf> = None;
    let mut top_k_arg: Option<std::path::PathBuf> = None;
    let mut seed_arg: Option<std::path::PathBuf> = None;
    let mut max_tokens_arg: Option<std::path::PathBuf> = None;
    let mut context_arg: Option<std::path::PathBuf> = None;
    let mut compact_at_arg: Option<std::path::PathBuf> = None;
    let mut serve_arg: Option<std::path::PathBuf> = None;
    let mut serve_addr_arg: Option<std::path::PathBuf> = None;
    // The scan borrows every slot mutably, so it lives in a block: the typed
    // parsing below reads those same locals, and the borrows have to be dead by
    // then. Deleting the braces produces a borrow error pointing at the first
    // reader rather than here.
    let parsed = {
        let mut slots: [(&'static str, &mut Option<std::path::PathBuf>); 20] = [
            ("--model", &mut model),
            ("--record", &mut record),
            ("--trace-chrome", &mut trace_chrome),
            ("--control-vector", &mut control_vector),
            ("--control-vector-scale", &mut control_scale_arg),
            ("--control-vector-layer-range", &mut control_range_arg),
            ("--probe", &mut probe),
            ("--tokenizer", &mut tokenizer),
            ("--probe-out", &mut probe_out),
            ("--alignment-out", &mut alignment_out),
            ("--interp-log", &mut interp_log),
            ("--temperature", &mut temperature_arg),
            ("--top-p", &mut top_p_arg),
            ("--top-k", &mut top_k_arg),
            ("--seed", &mut seed_arg),
            ("--max-tokens", &mut max_tokens_arg),
            ("--context", &mut context_arg),
            ("--compact-at", &mut compact_at_arg),
            ("--serve", &mut serve_arg),
            ("--serve-addr", &mut serve_addr_arg),
        ];
        match harness::cli::extract(args, &mut slots) {
            Ok(parsed) => parsed,
            Err(error) => {
                runtime::status::error(&error);
                runtime::status::plain(&harness::cli::usage());
                return ExitCode::from(2);
            }
        }
    };
    // Flags given without the flag they need. Generalized from the one
    // hand-written case (--serve-addr), so the steering companions are checked
    // too: --control-vector-scale without a vector used to be ignored.
    let missing = harness::cli::missing_requirements(&parsed.given);
    if !missing.is_empty() {
        for message in &missing {
            runtime::status::error(message);
        }
        return ExitCode::from(2);
    }
    let args = parsed.rest;
    let control_scale = match control_scale_arg {
        Some(v) => match v.to_string_lossy().parse::<f32>() {
            Ok(s) if s.is_finite() => s,
            _ => {
                runtime::status::error("--control-vector-scale: not a finite number");
                return ExitCode::from(2);
            }
        },
        None => 1.0,
    };
    let control_layer_range = match control_range_arg {
        Some(v) => match parse_layer_range(&v.to_string_lossy()) {
            Ok(range) => Some(range),
            Err(error) => {
                runtime::status::error(&error);
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    // Sampler::new treats out-of-range values as config bugs and panics, so
    // user input is range-checked here and rejected as a usage error instead.
    let mut sampling = runtime::sampler::SamplingParams::default();
    if let Some(v) = temperature_arg {
        match v.to_string_lossy().parse::<f32>() {
            Ok(t) if t.is_finite() && t >= 0.0 => sampling.temperature = t,
            _ => {
                runtime::status::error("--temperature: expected a finite number >= 0");
                return ExitCode::from(2);
            }
        }
    }
    if let Some(v) = top_p_arg {
        match v.to_string_lossy().parse::<f32>() {
            Ok(p) if p > 0.0 && p <= 1.0 => sampling.top_p = p,
            _ => {
                runtime::status::error("--top-p: expected a number in (0, 1]");
                return ExitCode::from(2);
            }
        }
    }
    if let Some(v) = top_k_arg {
        // Zero is a usage error rather than a synonym for absence: it would
        // otherwise have to mean either "keep nothing", which is not a
        // distribution, or "keep everything", which reads as its opposite.
        match v.to_string_lossy().parse::<usize>() {
            Ok(k) if k >= 1 => sampling.top_k = Some(k),
            _ => {
                runtime::status::error(
                    "--top-k: expected an integer >= 1 (omit the flag for no cutoff)",
                );
                return ExitCode::from(2);
            }
        }
    }
    if let Some(v) = seed_arg {
        match v.to_string_lossy().parse::<u64>() {
            Ok(s) => sampling.seed = s,
            _ => {
                runtime::status::error("--seed: expected an unsigned integer");
                return ExitCode::from(2);
            }
        }
    }
    let max_tokens = match max_tokens_arg {
        Some(v) => match v.to_string_lossy().parse::<usize>() {
            Ok(n) if n >= 1 => n,
            _ => {
                runtime::status::error("--max-tokens: expected an integer >= 1");
                return ExitCode::from(2);
            }
        },
        None => 256,
    };
    let context = match context_arg {
        Some(v) => match v.to_string_lossy().parse::<usize>() {
            Ok(n) if n >= 1 => Some(n),
            _ => {
                runtime::status::error("--context: expected an integer >= 1");
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    let compact_at = match compact_at_arg {
        Some(v) => match v.to_string_lossy().parse::<f32>() {
            Ok(f) if f > 0.0 && f <= 1.0 => f,
            _ => {
                runtime::status::error("--compact-at: expected a number in (0, 1]");
                return ExitCode::from(2);
            }
        },
        None => 0.8,
    };
    // The bare flags carry no value, so the scan recorded only that they were
    // given. Their help text lives in the table with everything else.
    let given = |name: &str| parsed.given.contains(&name);
    let text_only = given("--text-only");
    let single_thread = given("--single-thread");
    let chat = given("--chat");
    let status = given("--status");
    let deep = given("--deep");
    // --serve <port>: the OpenAI-shaped endpoint on loopback instead of a
    // task or a REPL. Mutually exclusive with both, and like --chat it takes
    // no positional task text.
    let serve = match serve_arg {
        Some(v) => match v.to_string_lossy().parse::<u16>() {
            Ok(p) if p >= 1 => Some(p),
            _ => {
                runtime::status::error("--serve: expected a port in 1..=65535");
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    if [chat, serve.is_some(), status]
        .iter()
        .filter(|m| **m)
        .count()
        > 1
    {
        runtime::status::error("--chat, --serve and --status are mutually exclusive");
        return ExitCode::from(2);
    }
    let serve_addr = match serve_addr_arg {
        Some(v) => match v.to_string_lossy().parse::<std::net::IpAddr>() {
            Ok(addr) => addr,
            Err(_) => {
                runtime::status::error(&format!(
                    "--serve-addr: expected an IP address, got {}",
                    v.display()
                ));
                return ExitCode::from(2);
            }
        },
        None => std::net::IpAddr::from([127, 0, 0, 1]),
    };
    // Flags accepted but with nothing to act on. A warning rather than an
    // error: setting sampling defaults uniformly and then pinning one run to
    // greedy is a coherent thing for a script to do, and the request log is
    // documented as ignored outside --serve rather than being lost. What was
    // wrong was doing it silently.
    if sampling.temperature == 0.0 {
        for message in harness::cli::inert(&parsed.given, harness::cli::GREEDY) {
            runtime::status::warn(&message);
        }
    }
    if serve.is_none() {
        for message in harness::cli::inert(&parsed.given, harness::cli::NOT_SERVING) {
            runtime::status::warn(&message);
        }
    }
    // The digest is produced only when an identity check asks for it, so
    // --deep with neither artifact reads nothing and reported nothing. The
    // laziness is right -- hashing 15 GiB for a comparison nobody made would
    // be the defect -- but a script that passed --deep and saw no complaint
    // would believe the weights had been verified. No `status &&` guard: the
    // requires edge has already refused --deep outside --status.
    if probe.is_none() && control_vector.is_none() {
        for message in harness::cli::inert(&parsed.given, harness::cli::NO_ARTIFACT) {
            runtime::status::warn(&message);
        }
    }
    // A task is positional text; the three flag modes take none. Exactly one
    // of the four must be selected.
    if args.is_empty() != (chat || serve.is_some() || status) {
        runtime::status::plain(&harness::cli::usage());
        return ExitCode::from(2);
    }
    // --status answers from the filesystem and returns. Placed here for two
    // reasons: after every parse and range check, so it reports on the run
    // that was actually asked for rather than on a half-validated one; and
    // before `RunLog::begin` below, because a query is not a run and
    // cataloguing it would pollute the runs table the way `command == "runs"`
    // does on the Python side. Before the trace guard too, so no subscriber is
    // installed and no trace file appears for a preflight.
    if status {
        // `REQUIRED` in the flag table feeds the usage line and nothing else,
        // so nothing has rejected a missing --model by this point. A run
        // discovers it inside `build_generator` and exits 1; here it is a
        // usage mistake, which is what exit 2 is for.
        let Some(model) = model.as_deref() else {
            runtime::status::error("--status: no model to inspect, pass --model <checkpoint-dir>");
            runtime::status::plain(&harness::cli::usage());
            return ExitCode::from(2);
        };
        return report(&harness::preflight::checks(&harness::preflight::Request {
            model,
            probe: probe.as_deref(),
            control_vector: control_vector.as_deref(),
            control_scale,
            control_layer_range,
            interp_log: interp_log.as_deref(),
            text_only,
            deep,
        }));
    }
    // The guard flushes the trace file on drop at the end of main.
    let _trace_guard = match trace_chrome {
        Some(path) => {
            use tracing_subscriber::layer::SubscriberExt;
            let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
                .file(path)
                .include_args(true)
                .build();
            tracing::subscriber::set_global_default(
                tracing_subscriber::registry::Registry::default().with(layer),
            )
            .expect("first and only subscriber installation");
            Some(guard)
        }
        None => None,
    };
    let task = args.join(" ");
    let root = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => {
            runtime::status::error(&format!("Working directory: {e}"));
            return ExitCode::from(1);
        }
    };
    // Run-level obs records in the shared common/obs contract. Usage errors
    // above return before a run id exists, matching the Python dispatch;
    // without a writable OBS_LOG_DIR the log degrades to a silent no-op.
    let kind = match (chat, serve.is_some()) {
        (_, true) => "serve",
        (true, _) => "chat",
        _ => "task",
    };
    let mut run_log = obs::RunLog::begin(kind, &argv_full);
    match run(
        &root,
        &task,
        model.as_deref(),
        tokenizer.as_deref(),
        record.as_deref(),
        control_vector.as_deref(),
        control_scale,
        control_layer_range,
        probe.as_deref(),
        probe_out.as_deref(),
        alignment_out.as_deref(),
        interp_log.as_deref(),
        &mut run_log,
        sampling,
        max_tokens,
        context,
        compact_at,
        chat,
        serve,
        serve_addr,
        text_only,
        single_thread,
    ) {
        // The answer, if any, was printed by `run` before its bookkeeping.
        Ok(()) => {
            run_log.end(0);
            ExitCode::SUCCESS
        }
        Err(e) => {
            runtime::status::error(&format!("Task failed: {e:#}"));
            run_log.fail(&e, 1);
            run_log.end(1);
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Still used by the probe fixtures below, which write a sidecar carrying
    // the sentinel so the identity check does not hash anything.
    use harness::preflight::UNKNOWN_MODEL_HASH;

    // The model-identity and control-vector-manifest tests moved to
    // `harness::preflight` with the functions they cover, unmodified.

    #[test]
    fn a_probe_report_failure_warns_and_leaves_a_completed_task_completed() {
        let (result, warning) = fold_probe_report(
            Ok("the answer".to_string()),
            Err(anyhow::anyhow!("create probe output /absent/flags.jsonl")),
        );
        assert_eq!(
            result.expect("a write failure must not fail the task"),
            "the answer"
        );
        let warning = warning.expect("the discarded flags must still be reported");
        // Untagged: the tag belongs to `runtime::status` at the emission
        // site, so what this pins is the wording and the fact that a line is
        // produced at all -- which is what the test was ever about.
        assert!(!warning.contains("[WARN]"), "{warning}");
        assert!(
            warning.starts_with("Probe flags not written: "),
            "{warning}"
        );
        assert!(warning.contains("create probe output"), "{warning}");
    }

    #[test]
    fn a_probe_report_never_rewrites_the_task_result_it_folds_into() {
        let (ok, warning) = fold_probe_report(Ok("kept".to_string()), Ok(()));
        assert_eq!(ok.expect("a clean report changes nothing"), "kept");
        assert_eq!(warning, None);
        // A failed task stays failed with its own error, not the report's.
        let (err, _) = fold_probe_report(
            Err(anyhow::anyhow!("task error")),
            Err(anyhow::anyhow!("report error")),
        );
        assert_eq!(
            err.expect_err("a failed task stays failed").to_string(),
            "task error"
        );
    }

    /// A minimal probe artifact in `dir`, in the layout
    /// `interp/probes/export.py` writes: the safetensors image and the JSON
    /// sidecar beside it. The scoring math is covered by the runtime's
    /// reference test, so the values here are arbitrary.
    fn write_probe(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("probe.safetensors");
        std::fs::write(
            &path,
            runtime::synth::build_f32_safetensors(&[
                ("weights", &[1.0, 1.0][..]),
                ("mu", &[0.0, 0.0][..]),
                ("sigma", &[1.0, 1.0][..]),
            ])
            .expect("write the probe image"),
        )
        .expect("write the probe image");
        std::fs::write(
            path.with_extension("json"),
            format!(
                r#"{{"schema_version": 1, "artifact": "probe",
                   "timestamp": "2026-01-01T00:00:00+0000", "git_sha": "abcdef0",
                   "name": "report-probe", "model_hash": "{UNKNOWN_MODEL_HASH}",
                   "layer": 0, "bias": 0.0, "threshold": 0.5, "width": 2}}"#
            ),
        )
        .expect("write the probe sidecar");
        path
    }

    #[test]
    fn probe_flags_are_written_under_a_parent_directory_that_does_not_exist_yet() {
        let dir = tempfile::tempdir().expect("tempdir");
        let probe = std::sync::Arc::new(
            runtime::model::probe::Probe::load(&write_probe(dir.path())).expect("load probe"),
        );
        let (_tap, state) = runtime::model::probe::ProbeMonitor::new(std::sync::Arc::clone(&probe));
        let monitor = MonitorHandle { probe, state };
        let out = dir.path().join("absent").join("flags.jsonl");
        report_probe(&monitor, &out).expect("a missing parent directory is created, not a failure");
        assert!(out.is_file(), "{}", out.display());
    }
}
