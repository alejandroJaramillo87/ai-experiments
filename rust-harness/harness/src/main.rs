/*
 * harness — single-binary local coding agent.
 *
 * Wires the full stack: permissions from permissions.yaml, a durable JSONL
 * transcript, the agent loop, and a SessionGenerator over the runtime. The
 * generator runs the hand-rolled CPU backend against the GGUF passed via
 * --model; the cuda backend (phase A) will slot in at the same seam.
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

/// Construct the generator over the CPU backend for the GGUF at
/// `model_path`, wrapping the model in the flight recorder when `record`
/// names a path. There is no mock fallback in the shipping binary.
fn build_generator(
    model_path: Option<&Path>,
    record: Option<&Path>,
    control_vector: Option<&Path>,
    control_scale: f32,
    probe: Option<&Path>,
    sampling: runtime::sampler::SamplingParams,
    max_tokens: usize,
) -> Result<(SessionGenerator, Option<MonitorHandle>)> {
    use runtime::model::{
        cpu::{ControlVector, CpuModel},
        probe::{Probe, ProbeMonitor},
        record::RecordingModel,
        Model,
    };
    let Some(model_path) = model_path else {
        anyhow::bail!(
            "no model: pass --model <path.gguf> to run on the CPU backend; \
             the cuda backend (roadmap phase A) has not landed"
        )
    };
    let data = std::fs::read(model_path)
        .with_context(|| format!("read model {}", model_path.display()))?;
    let file = runtime::gguf::parse(&data).context("parse GGUF")?;
    let mut cpu = CpuModel::from_gguf(&file, &data).context("load CPU model")?;
    if let Some(path) = control_vector {
        let cv_data = std::fs::read(path)
            .with_context(|| format!("read control vector {}", path.display()))?;
        let cv_file = runtime::gguf::parse(&cv_data).context("parse control-vector GGUF")?;
        let cv = ControlVector::from_gguf(&cv_file, &cv_data, control_scale)
            .context("load control vector")?;
        cpu.set_control_vector(cv).context("apply control vector")?;
    }
    // The monitor observes the steered stream: set_control_vector first,
    // then the tap, matching the documented tap semantics.
    let monitor = match probe {
        Some(path) => {
            let p_data =
                std::fs::read(path).with_context(|| format!("read probe {}", path.display()))?;
            let p_file = runtime::gguf::parse(&p_data).context("parse probe GGUF")?;
            let p = std::sync::Arc::new(Probe::from_gguf(&p_file, &p_data).context("load probe")?);
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
            let (tap, state) = ProbeMonitor::new(std::sync::Arc::clone(&p));
            cpu.set_tap(Box::new(tap));
            Some(MonitorHandle { probe: p, state })
        }
        None => None,
    };
    let max_context = cpu.config().context_length;
    let model: Box<dyn Model> = match record {
        Some(path) => Box::new(RecordingModel::to_file(Box::new(cpu), path)?),
        None => Box::new(cpu),
    };
    let session = runtime::session::Session::new(model, max_context)?;
    let tokenizer = runtime::tokenizer::Tokenizer::from_gguf(&file).context("load tokenizer")?;
    let generator = SessionGenerator::new(session, tokenizer, sampling, max_tokens)?;
    Ok((generator, monitor))
}

/// Drain the monitor state: write one JSONL record per flagged position to
/// `out`, print a one-line summary to stderr. Observation only, so this
/// runs after the task and never affects its result.
fn report_probe(monitor: &MonitorHandle, out: &Path) -> Result<()> {
    let state = monitor.state.lock().expect("monitor state lock");
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

#[allow(clippy::too_many_arguments)]
fn run(
    root: &Path,
    task: &str,
    model: Option<&Path>,
    record: Option<&Path>,
    control_vector: Option<&Path>,
    control_scale: f32,
    probe: Option<&Path>,
    probe_out: Option<&Path>,
    sampling: runtime::sampler::SamplingParams,
    max_tokens: usize,
    chat: bool,
) -> Result<String> {
    let transcript = Transcript::open(&root.join("session.jsonl")).context("session.jsonl")?;
    let (mut generator, monitor) = build_generator(
        model,
        record,
        control_vector,
        control_scale,
        probe,
        sampling,
        max_tokens,
    )?;
    let answer = if chat {
        // Chat mode: REPL over the generator, no tools or permissions. The
        // probe monitor and control vector are already installed on the
        // model, so interp flags apply to chat turns unchanged.
        let mut transcript = transcript;
        let stdin = std::io::stdin();
        let turns = harness::chat::run_chat(
            &mut generator,
            &mut transcript,
            stdin.lock(),
            std::io::stdout(),
        )?;
        // Replies were already streamed to stdout; the summary is status.
        eprintln!("[INFO] Chat ended: {turns} model turns");
        String::new()
    } else {
        let permissions = match std::fs::read_to_string(root.join("permissions.yaml")) {
            Ok(yaml) => Permissions::from_yaml(&yaml).context("permissions.yaml")?,
            // No config: every call goes through the interactive Ask path.
            Err(_) => Permissions::from_yaml("rules: []").expect("empty rule set"),
        };
        let mut agent = Agent::new(root.to_path_buf(), permissions, transcript);
        agent.run_task(&mut generator, task, 64, ask_stdin)?
    };
    if let Some(monitor) = &monitor {
        let default_out = root.join("probe-flags.jsonl");
        report_probe(monitor, probe_out.unwrap_or(&default_out))?;
    }
    Ok(answer)
}

fn main() -> ExitCode {
    const USAGE: &str = "usage: harness --model <path.gguf> [--record <path>] \
         [--trace-chrome <path>] [--control-vector <path.gguf>] \
         [--control-vector-scale <f>] [--probe <path.gguf>] \
         [--probe-out <path>] [--temperature <f>] [--top-p <f>] \
         [--seed <n>] [--max-tokens <n>] (--chat | <task description>)";
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // Snapshot before flag extraction mutates args: cmd.start records the
    // invocation as given, program name excluded (the shape common/obs logs).
    let argv_full = args.clone();
    // --model <path.gguf>: checkpoint for the CPU backend.
    // --record <path>: wrap the model in the flight recorder, appending every
    // forward call to a JSONL file replayable as a MockModel script.
    // --trace-chrome <path>: write a Perfetto-loadable trace of the span
    // timeline. No subscriber is installed otherwise, so spans cost nothing.
    // --control-vector <path.gguf>: steering directions in the llama.cpp
    // control-vector layout, added to the residual stream per layer.
    // --control-vector-scale <f>: multiplier on the directions (default 1.0).
    // --probe <path.gguf>: linear probe (interp/probes/export.py layout)
    // scored in-band at every position of its layer through the tap seam.
    // --probe-out <path>: JSONL file for flagged positions (default
    // probe-flags.jsonl in the working directory).
    // --temperature <f> / --top-p <f> / --seed <n>: sampling parameters
    // (defaults 0.8 / 0.95 / 0; seed 0 replays a run exactly).
    // --max-tokens <n>: per-turn generation budget (default 256).
    let mut model: Option<std::path::PathBuf> = None;
    let mut record: Option<std::path::PathBuf> = None;
    let mut trace_chrome: Option<std::path::PathBuf> = None;
    let mut control_vector: Option<std::path::PathBuf> = None;
    let mut control_scale_arg: Option<std::path::PathBuf> = None;
    let mut probe: Option<std::path::PathBuf> = None;
    let mut probe_out: Option<std::path::PathBuf> = None;
    let mut temperature_arg: Option<std::path::PathBuf> = None;
    let mut top_p_arg: Option<std::path::PathBuf> = None;
    let mut seed_arg: Option<std::path::PathBuf> = None;
    let mut max_tokens_arg: Option<std::path::PathBuf> = None;
    for (flag, slot) in [
        ("--model", &mut model),
        ("--record", &mut record),
        ("--trace-chrome", &mut trace_chrome),
        ("--control-vector", &mut control_vector),
        ("--control-vector-scale", &mut control_scale_arg),
        ("--probe", &mut probe),
        ("--probe-out", &mut probe_out),
        ("--temperature", &mut temperature_arg),
        ("--top-p", &mut top_p_arg),
        ("--seed", &mut seed_arg),
        ("--max-tokens", &mut max_tokens_arg),
    ] {
        if let Some(i) = args.iter().position(|a| a == flag) {
            if i + 1 >= args.len() {
                eprintln!("{USAGE}");
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
    // --chat: interactive REPL instead of a one-shot task; the two are
    // mutually exclusive, so leftover positional args are a usage error.
    let chat = if let Some(i) = args.iter().position(|a| a == "--chat") {
        args.remove(i);
        true
    } else {
        false
    };
    if args.is_empty() != chat {
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
    let mut run_log = harness::obs::RunLog::begin(if chat { "chat" } else { "task" }, &argv_full);
    match run(
        &root,
        &task,
        model.as_deref(),
        record.as_deref(),
        control_vector.as_deref(),
        control_scale,
        probe.as_deref(),
        probe_out.as_deref(),
        sampling,
        max_tokens,
        chat,
    ) {
        Ok(answer) => {
            if !answer.is_empty() {
                println!("{answer}");
            }
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
