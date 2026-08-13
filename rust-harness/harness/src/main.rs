/*
 * harness — single-binary local coding agent.
 *
 * Wires the full stack: permissions from permissions.yaml, a durable JSONL
 * transcript, the agent loop, and a SessionGenerator over the runtime. The
 * generator runs the hand-rolled CPU backend against the checkpoint passed
 * via --model; the cuda backend (phase A) will slot in at the same seam.
 */

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context, Result};

use harness::agent::Agent;
use harness::generator::SessionGenerator;
use harness::permissions::Permissions;
use harness::tools::ToolCall;
use harness::transcript::Transcript;

/// Interactive Ask resolution: prompt on stderr, read one line from stdin.
fn ask_stdin(call: &ToolCall) -> bool {
    eprint!("allow {} {}? [y/N] ", call.name(), call.target());
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() {
        return false;
    }
    matches!(line.trim(), "y" | "yes")
}

/// A live probe monitor: the loaded probe plus the shared state its tap
/// fills during the run. Drained by `report_probe` after the task.
struct MonitorHandle {
    probe: std::sync::Arc<runtime::model::probe::Probe>,
    state: std::sync::Arc<std::sync::Mutex<runtime::model::probe::MonitorState>>,
}

/// The forward-pass record a run accumulates, when one was asked for.
///
/// `None` rather than an empty vector, so "alignment was not requested" and
/// "the model made no passes" stay distinguishable in the writer.
type PassHandle = Option<std::sync::Arc<std::sync::Mutex<Vec<runtime::align::Pass>>>>;

/// Sentinel an interp artifact carries when the weights it was fitted on
/// have no recorded identity (`interp/probes/capture.py` defaults
/// `model_hash` to it, and the exporters pass it through).
const UNKNOWN_MODEL_HASH: &str = "unknown";

/// Refuse an interp artifact fitted on weights other than the served ones.
/// A probe scored against a different model, or a control vector added to
/// its residual stream, produces meaningless output indistinguishable from
/// a genuine flag or steer, so the pairing is refused the way every Python
/// consumer refuses it (`interp/probes/registry.py` `require_match`,
/// `interp/steering/wiring.py`). Only two real hashes can disagree: the
/// `unknown` sentinel downgrades to a skipped check, which is what keeps
/// Python-captured artifacts loading. The served digest arrives as a
/// closure because producing it hashes multi-gigabyte weights, a cost a
/// skipped check must not pay.
fn check_model_identity(
    artifact: &str,
    artifact_hash: &str,
    served_sha256: impl FnOnce() -> Result<String>,
) -> Result<()> {
    if artifact_hash == UNKNOWN_MODEL_HASH {
        eprintln!(
            "[WARN] Model identity: {artifact} records model hash \
             \"{UNKNOWN_MODEL_HASH}\", check skipped"
        );
        return Ok(());
    }
    let served = served_sha256()?;
    anyhow::ensure!(
        artifact_hash == served,
        "{artifact} was trained on a different model: {artifact} model_hash \
         {artifact_hash}, served model sha256 {served}"
    );
    Ok(())
}

/// The model hash recorded beside a control vector, when its artifact
/// directory is intact. The image itself carries tensors and nothing else,
/// so the sibling `manifest.json` the exporter writes
/// (`interp/steering/export.py`) is the only hash available. A vector copied
/// away from its manifest, or one whose manifest is unreadable, stays
/// loadable and unchecked — the deliberate asymmetry with the probe's
/// mandatory sidecar, which `schemas/control_vector_artifact.json` records.
fn control_vector_model_hash(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path.parent()?.join("manifest.json")).ok()?;
    let manifest: serde_json::Value = serde_json::from_str(&text).ok()?;
    manifest.get("model_hash")?.as_str().map(str::to_string)
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
) -> Result<(SessionGenerator, Option<MonitorHandle>, u64, PassHandle)> {
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
        match control_vector_model_hash(path) {
            Some(hash) => {
                check_model_identity("control vector", &hash, || {
                    runtime::digest::sha256_file(&weights_path)
                        .with_context(|| format!("hash {}", weights_path.display()))
                })?;
            }
            None => eprintln!(
                "[WARN] Model identity: no readable manifest.json beside {}, \
                 control vector unchecked",
                path.display()
            ),
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
            check_model_identity("probe", &p.model_hash, || {
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
            let (tap, state) = ProbeMonitor::with_scores(std::sync::Arc::clone(&p), align);
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
    {
        let c = cpu.config();
        let n_kv_heads = c.n_kv_heads;
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
        eprintln!(
            "[INFO] Weights: {:.2} GiB resident at {}",
            weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            width.unwrap_or("mixed widths")
        );
        // Peak against the line above is the diagnostic, not the peak alone: a
        // load that left the checkpoint mapped reports roughly twice the
        // weights, because the map is still resident behind the copy.
        if let Some(peak) = runtime::peak_rss_bytes() {
            eprintln!(
                "[INFO] Peak RSS after load: {:.2} GiB",
                peak as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
        let group = cpu.batch().chunk(max_context);
        let bytes = runtime::kv::budget_bytes(&layers, max_context, group);
        let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "[INFO] KV cache: {max_context} tokens, {gib:.2} GiB f32 resident \
             (sliding layers capped at the window plus a batch of {group})"
        );
        // The rewind horizon, stated before the run rather than discovered
        // when a rewind fails. A windowed layer cannot return to a position it
        // has evicted, so once the cache is full the reachable past is bounded
        // -- and conversation compaction rewinds much further than this, which
        // is why `Session::sync` rebuilds instead of erroring.
        if let Some(horizon) = cpu.rewind_horizon(max_context) {
            eprintln!(
                "[INFO] Rewind horizon: {horizon} tokens behind the tip once the \
                 window fills (compaction past it re-prefills)"
            );
        }
        if bytes > 2 << 30 {
            eprintln!(
                "[WARN] KV cache above 2 GiB: consider --context to cap it \
                 (compaction keeps long sessions inside the cap)"
            );
        }
    }
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
                    Ok(()) => eprintln!(
                        "[INFO] Session restored: {} cached tokens from {}",
                        session.cached_len(),
                        path.display()
                    ),
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
                        eprintln!(
                            "[WARN] Session snapshot {} discarded: {e:#} \
                             (starting with an empty cache)",
                            path.display()
                        );
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
    Ok((generator, monitor, model_hash, passes))
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
    eprintln!(
        "[INFO] Probe {}: {} positions scored, {} flagged (threshold {}), flags at {}",
        monitor.probe.name,
        state.scored,
        state.flags.len(),
        monitor.probe.threshold,
        out.display()
    );
    Ok(())
}

/// Fold a probe report's outcome into the task's. Observation is not part
/// of the task, so a report failure leaves the task result alone and
/// surfaces as a warning line; propagating it instead turned a completed,
/// transcript-recorded task into exit 1 with the answer never printed.
/// Returning the line rather than printing it keeps the policy testable
/// without a live monitor.
fn fold_probe_report(
    task: Result<String>,
    reported: Result<()>,
) -> (Result<String>, Option<String>) {
    let warning = reported
        .err()
        .map(|e| format!("[WARN] Probe flags not written: {e:#}"));
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
    run_id: Option<&str>,
    sampling: runtime::sampler::SamplingParams,
    max_tokens: usize,
    context: Option<usize>,
    compact_at: f32,
    chat: bool,
    serve: Option<u16>,
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
    let (mut generator, monitor, model_hash, passes) = build_generator(
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
    )?;
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
        // This blocks until the process is signalled, so the snapshot and
        // probe-flag bookkeeping below is unreachable in this mode. Flushing
        // interp artifacts from a long-lived server is its own problem and is
        // not solved here; a served run's observation is the tap seam, not
        // the end-of-run report.
        let listener = std::net::TcpListener::bind(("127.0.0.1", port))
            .with_context(|| format!("binding 127.0.0.1:{port}"))?;
        let model_name = model
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "synthetic".to_string());
        eprintln!("[INFO] Serving on http://127.0.0.1:{port} (model {model_name})");
        let state = std::sync::Arc::new(harness::server::ServerState::new(
            generator, model_name, sampling, max_tokens,
        ));
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
            eprintln!("[INFO] Chat ended: {turns} model turns");
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
            Ok(()) => eprintln!(
                "[INFO] Session snapshot: {} cached tokens at {}",
                generator.session().cached_len(),
                snapshot_path.display()
            ),
            Err(e) => eprintln!("[WARN] Session snapshot not saved: {e:#}"),
        }
    }
    let result = match &monitor {
        Some(monitor) => {
            let default_out = root.join("probe-flags.jsonl");
            let reported = report_probe(monitor, probe_out.unwrap_or(&default_out));
            let (result, warning) = fold_probe_report(result, reported);
            if let Some(warning) = warning {
                eprintln!("{warning}");
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
        let header = harness::alignment::Header {
            run_id: run_id.unwrap_or("unknown"),
            model_hash: &format!("{model_hash:016x}"),
            probe: probe_name.as_deref(),
            probe_layer: monitor.as_ref().map(|m| m.probe.layer),
        };
        match harness::alignment::write(out, &header, &a, &passes, &scores) {
            Ok(()) => eprintln!(
                "[INFO] Alignment: {} positions, {} passes -> {}",
                a.rows.len(),
                passes.len(),
                out.display()
            ),
            Err(e) => eprintln!("[WARN] Alignment not written: {e:#}"),
        }
    }
    result?;
    Ok(())
}

/// Refuse a flag value that is itself a flag, naming both. The extractor
/// takes the word after a flag as its value, so an omitted value swallows
/// the flag that follows: the swallowed flag never applies, its own value
/// falls into the task text, and the run exits 0 having done something the
/// user never asked for. Only the `--` spelling this parser recognizes is
/// refused; a negative number or a path starting with `-` is a value.
fn flag_value_error(flag: &str, value: &str) -> Option<String> {
    value
        .starts_with("--")
        .then(|| format!("[ERROR] {flag}: expected a value, found the flag {value}"))
}

fn main() -> ExitCode {
    const USAGE: &str = "usage: harness --model <checkpoint-dir> [--tokenizer <path.json>] \
         [--record <path>] \
         [--trace-chrome <path>] [--control-vector <path.safetensors>] \
         [--control-vector-scale <f>] [--control-vector-layer-range \"<start> <end>\"] \
         [--probe <path.safetensors>] \
         [--probe-out <path>] [--alignment-out <path>] [--temperature <f>] [--top-p <f>] \
         [--top-k <n>] [--seed <n>] [--max-tokens <n>] [--context <n>] \
         [--compact-at <f>] [--text-only] [--single-thread] \
         (--chat | --serve <port> | <task description>)";
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // Snapshot before flag extraction mutates args: cmd.start records the
    // invocation as given, program name excluded (the shape common/obs logs).
    let argv_full = args.clone();
    // --model <checkpoint-dir>: config.json and model.safetensors for the
    // CPU backend. A path to the weights file resolves to its directory.
    // --record <path>: wrap the model in the flight recorder, appending every
    // forward call to a JSONL file replayable as a MockModel script.
    // --trace-chrome <path>: write a Perfetto-loadable trace of the span
    // timeline. No subscriber is installed otherwise, so spans cost nothing.
    // --control-vector <path.safetensors>: steering directions from
    // interp/steering/export.py, one `block.<index>` tensor per steered
    // decoder block, added to that block's output residual.
    // --control-vector-scale <f>: multiplier on the directions (default 1.0).
    // --control-vector-layer-range "<start> <end>": inclusive 0-based block
    // range; directions outside it are dropped at load.
    // --probe <path.safetensors>: linear probe (interp/probes/export.py
    // layout) scored in-band at every position of its layer through the tap
    // seam. The JSON sidecar beside it -- the same path with a .json
    // extension -- carries the layer, site and threshold, and is required.
    // --probe-out <path>: JSONL file for flagged positions (default
    // probe-flags.jsonl in the working directory).
    // --alignment-out <path>: JSONL file mapping every position to the byte
    // span of output text it produced, the forward pass that computed it, and
    // the probe score taken there. Off by default: it makes the model record
    // its passes and the monitor keep every clean score, both of which grow
    // with the run.
    // --temperature <f> / --top-p <f> / --seed <n>: sampling parameters
    // (defaults 0.8 / 0.95 / 0; seed 0 replays a run exactly).
    // --top-k <n>: rank cutoff applied before top-p, keeping at most the n
    // most probable tokens of the masked support. Omitted by default, which
    // is no rank cutoff at all; 0 is rejected rather than read as "disabled".
    // --max-tokens <n>: per-turn generation budget (default 256).
    // --context <n>: cap the context budget below the model's trained
    // window (never above it; max_position_embeddings is the default and
    // the maximum).
    // --compact-at <f>: automatic-compaction threshold as a fraction of the
    // context budget (default 0.8, must be in (0, 1]).
    // --serve <port>: serve /v1/chat/completions and /v1/health on loopback
    // instead of running a task, for the interp pipeline's HTTP clients. The
    // sampling flags above become the per-request defaults; a request may
    // override any of them, including the seed. Mutually exclusive with
    // --chat, and takes no task text.
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
    let mut temperature_arg: Option<std::path::PathBuf> = None;
    let mut top_p_arg: Option<std::path::PathBuf> = None;
    let mut top_k_arg: Option<std::path::PathBuf> = None;
    let mut seed_arg: Option<std::path::PathBuf> = None;
    let mut max_tokens_arg: Option<std::path::PathBuf> = None;
    let mut context_arg: Option<std::path::PathBuf> = None;
    let mut compact_at_arg: Option<std::path::PathBuf> = None;
    let mut serve_arg: Option<std::path::PathBuf> = None;
    for (flag, slot) in [
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
        ("--temperature", &mut temperature_arg),
        ("--top-p", &mut top_p_arg),
        ("--top-k", &mut top_k_arg),
        ("--seed", &mut seed_arg),
        ("--max-tokens", &mut max_tokens_arg),
        ("--context", &mut context_arg),
        ("--compact-at", &mut compact_at_arg),
        ("--serve", &mut serve_arg),
    ] {
        if let Some(i) = args.iter().position(|a| a == flag) {
            if i + 1 >= args.len() {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
            if let Some(error) = flag_value_error(flag, &args[i + 1]) {
                eprintln!("{error}");
                return ExitCode::from(2);
            }
            *slot = Some(std::path::PathBuf::from(args.remove(i + 1)));
            args.remove(i);
        }
    }
    let control_scale = match control_scale_arg {
        Some(v) => match v.to_string_lossy().parse::<f32>() {
            Ok(s) if s.is_finite() => s,
            _ => {
                eprintln!("[ERROR] --control-vector-scale: not a finite number");
                return ExitCode::from(2);
            }
        },
        None => 1.0,
    };
    let control_layer_range = match control_range_arg {
        Some(v) => match parse_layer_range(&v.to_string_lossy()) {
            Ok(range) => Some(range),
            Err(error) => {
                eprintln!("[ERROR] {error}");
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
                eprintln!("[ERROR] --temperature: expected a finite number >= 0");
                return ExitCode::from(2);
            }
        }
    }
    if let Some(v) = top_p_arg {
        match v.to_string_lossy().parse::<f32>() {
            Ok(p) if p > 0.0 && p <= 1.0 => sampling.top_p = p,
            _ => {
                eprintln!("[ERROR] --top-p: expected a number in (0, 1]");
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
                eprintln!(
                    "[ERROR] --top-k: expected an integer >= 1 (omit the flag for no cutoff)"
                );
                return ExitCode::from(2);
            }
        }
    }
    if let Some(v) = seed_arg {
        match v.to_string_lossy().parse::<u64>() {
            Ok(s) => sampling.seed = s,
            _ => {
                eprintln!("[ERROR] --seed: expected an unsigned integer");
                return ExitCode::from(2);
            }
        }
    }
    let max_tokens = match max_tokens_arg {
        Some(v) => match v.to_string_lossy().parse::<usize>() {
            Ok(n) if n >= 1 => n,
            _ => {
                eprintln!("[ERROR] --max-tokens: expected an integer >= 1");
                return ExitCode::from(2);
            }
        },
        None => 256,
    };
    let context = match context_arg {
        Some(v) => match v.to_string_lossy().parse::<usize>() {
            Ok(n) if n >= 1 => Some(n),
            _ => {
                eprintln!("[ERROR] --context: expected an integer >= 1");
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    let compact_at = match compact_at_arg {
        Some(v) => match v.to_string_lossy().parse::<f32>() {
            Ok(f) if f > 0.0 && f <= 1.0 => f,
            _ => {
                eprintln!("[ERROR] --compact-at: expected a number in (0, 1]");
                return ExitCode::from(2);
            }
        },
        None => 0.8,
    };
    // --text-only: load the text stack of a multimodal checkpoint and decline
    // its audio and vision towers. Required rather than implied, so nobody
    // runs a multimodal model text-blind without having said so.
    let text_only = if let Some(i) = args.iter().position(|a| a == "--text-only") {
        args.remove(i);
        true
    } else {
        false
    };
    // --single-thread: run the CPU backend's row split at width one. Not a
    // numerics switch -- the split is bit-identical at every width -- but rr
    // serializes threads anyway, so `make debug-record` wants it.
    let single_thread = if let Some(i) = args.iter().position(|a| a == "--single-thread") {
        args.remove(i);
        true
    } else {
        false
    };
    // --chat: interactive REPL instead of a one-shot task; the two are
    // mutually exclusive, so leftover positional args are a usage error.
    let chat = if let Some(i) = args.iter().position(|a| a == "--chat") {
        args.remove(i);
        true
    } else {
        false
    };
    // --serve <port>: the OpenAI-shaped endpoint on loopback instead of a
    // task or a REPL. Mutually exclusive with both, and like --chat it takes
    // no positional task text.
    let serve = match serve_arg {
        Some(v) => match v.to_string_lossy().parse::<u16>() {
            Ok(p) if p >= 1 => Some(p),
            _ => {
                eprintln!("[ERROR] --serve: expected a port in 1..=65535");
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    if chat && serve.is_some() {
        eprintln!("[ERROR] --chat and --serve are mutually exclusive");
        return ExitCode::from(2);
    }
    // A task is positional text; the two flag modes take none. Exactly one of
    // the three must be selected.
    if args.is_empty() != (chat || serve.is_some()) {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
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
            eprintln!("[ERROR] Working directory: {e}");
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
    let mut run_log = harness::obs::RunLog::begin(kind, &argv_full);
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
        run_log.run_id(),
        sampling,
        max_tokens,
        context,
        compact_at,
        chat,
        serve,
        text_only,
        single_thread,
    ) {
        // The answer, if any, was printed by `run` before its bookkeeping.
        Ok(()) => {
            run_log.end(0, &root);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[ERROR] Task failed: {e:#}");
            run_log.fail(&e, 1);
            run_log.end(1, &root);
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime::digest::sha256_hex;

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
        assert!(warning.starts_with("[WARN] "), "{warning}");
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

    #[test]
    fn a_flag_value_that_is_itself_a_flag_is_refused_naming_the_flag_and_the_value() {
        // `--probe-out --max-tokens 4 hello` without a path: the old parser
        // stored "--max-tokens" as the probe output, dropped the token
        // budget, and pushed "4" into the task text, all at exit 0.
        let error = flag_value_error("--probe-out", "--max-tokens")
            .expect("an omitted value must not swallow the following flag");
        assert!(error.starts_with("[ERROR] "), "{error}");
        assert!(error.contains("--probe-out"), "{error}");
        assert!(error.contains("--max-tokens"), "{error}");
    }

    #[test]
    fn a_flag_value_that_only_looks_flag_adjacent_is_accepted() {
        // Refusing more than the `--` spelling would reject legitimate
        // values: a negative scale, or a path whose name starts with a dash.
        assert_eq!(flag_value_error("--probe-out", "flags.jsonl"), None);
        assert_eq!(flag_value_error("--control-vector-scale", "-1.5"), None);
        assert_eq!(flag_value_error("--model", "./-weights.gguf"), None);
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
}
