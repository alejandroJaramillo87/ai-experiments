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

/// Construct the generator over the CPU backend for the GGUF at
/// `model_path`, wrapping the model in the flight recorder when `record`
/// names a path. There is no mock fallback in the shipping binary.
fn build_generator(
    model_path: Option<&Path>,
    record: Option<&Path>,
    control_vector: Option<&Path>,
    control_scale: f32,
) -> Result<SessionGenerator> {
    use runtime::model::{
        cpu::{ControlVector, CpuModel},
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
    let max_context = cpu.config().context_length;
    let model: Box<dyn Model> = match record {
        Some(path) => Box::new(RecordingModel::to_file(Box::new(cpu), path)?),
        None => Box::new(cpu),
    };
    let session = runtime::session::Session::new(model, max_context)?;
    let tokenizer = runtime::tokenizer::Tokenizer::from_gguf(&file).context("load tokenizer")?;
    SessionGenerator::new(
        session,
        tokenizer,
        runtime::sampler::SamplingParams::default(),
        256,
    )
}

fn run(
    root: &Path,
    task: &str,
    model: Option<&Path>,
    record: Option<&Path>,
    control_vector: Option<&Path>,
    control_scale: f32,
) -> Result<String> {
    let permissions = match std::fs::read_to_string(root.join("permissions.yaml")) {
        Ok(yaml) => Permissions::from_yaml(&yaml).context("permissions.yaml")?,
        // No config: every call goes through the interactive Ask path.
        Err(_) => Permissions::from_yaml("rules: []").expect("empty rule set"),
    };
    let transcript = Transcript::open(&root.join("session.jsonl")).context("session.jsonl")?;
    let mut agent = Agent::new(root.to_path_buf(), permissions, transcript);
    let mut generator = build_generator(model, record, control_vector, control_scale)?;
    agent.run_task(&mut generator, task, 64, ask_stdin)
}

fn main() -> ExitCode {
    const USAGE: &str = "usage: harness --model <path.gguf> [--record <path>] \
         [--trace-chrome <path>] [--control-vector <path.gguf>] \
         [--control-vector-scale <f>] <task description>";
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // --model <path.gguf>: checkpoint for the CPU backend.
    // --record <path>: wrap the model in the flight recorder, appending every
    // forward call to a JSONL file replayable as a MockModel script.
    // --trace-chrome <path>: write a Perfetto-loadable trace of the span
    // timeline. No subscriber is installed otherwise, so spans cost nothing.
    // --control-vector <path.gguf>: steering directions in the llama.cpp
    // control-vector layout, added to the residual stream per layer.
    // --control-vector-scale <f>: multiplier on the directions (default 1.0).
    let mut model: Option<std::path::PathBuf> = None;
    let mut record: Option<std::path::PathBuf> = None;
    let mut trace_chrome: Option<std::path::PathBuf> = None;
    let mut control_vector: Option<std::path::PathBuf> = None;
    let mut control_scale_arg: Option<std::path::PathBuf> = None;
    for (flag, slot) in [
        ("--model", &mut model),
        ("--record", &mut record),
        ("--trace-chrome", &mut trace_chrome),
        ("--control-vector", &mut control_vector),
        ("--control-vector-scale", &mut control_scale_arg),
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
    if args.is_empty() {
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
    match run(
        &root,
        &task,
        model.as_deref(),
        record.as_deref(),
        control_vector.as_deref(),
        control_scale,
    ) {
        Ok(answer) => {
            println!("{answer}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[ERROR] Task failed: {e:#}");
            ExitCode::from(1)
        }
    }
}
