/*
 * harness — single-binary local coding agent.
 *
 * Wires the full stack: permissions from permissions.yaml, a durable JSONL
 * transcript, the agent loop, and a SessionGenerator over the runtime. The
 * generator requires the cuda backend (phase A); until it lands the binary
 * reports that and exits, everything above it being already testable.
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

/// Construct the generator over the real model, wrapping it in the flight
/// recorder (`runtime::model::record::RecordingModel::to_file`) when
/// `record` names a path. Requires the phase A cuda backend; there is no
/// mock fallback in the shipping binary.
fn build_generator(record: Option<&Path>) -> Result<SessionGenerator> {
    let _ = record; // consumed once the backend exists to be wrapped
    anyhow::bail!(
        "inference backend not built: the cuda backend (roadmap phase A) has not landed; \
         the harness stack above it is complete and runs under cargo test"
    )
}

fn run(root: &Path, task: &str, record: Option<&Path>) -> Result<String> {
    let permissions = match std::fs::read_to_string(root.join("permissions.yaml")) {
        Ok(yaml) => Permissions::from_yaml(&yaml).context("permissions.yaml")?,
        // No config: every call goes through the interactive Ask path.
        Err(_) => Permissions::from_yaml("rules: []").expect("empty rule set"),
    };
    let transcript = Transcript::open(&root.join("session.jsonl")).context("session.jsonl")?;
    let mut agent = Agent::new(root.to_path_buf(), permissions, transcript);
    let mut generator = build_generator(record)?;
    agent.run_task(&mut generator, task, 64, ask_stdin)
}

fn main() -> ExitCode {
    const USAGE: &str =
        "usage: harness [--record <path>] [--trace-chrome <path>] <task description>";
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // --record <path>: wrap the model in the flight recorder, appending every
    // forward call to a JSONL file replayable as a MockModel script.
    // --trace-chrome <path>: write a Perfetto-loadable trace of the span
    // timeline. No subscriber is installed otherwise, so spans cost nothing.
    let mut record: Option<std::path::PathBuf> = None;
    let mut trace_chrome: Option<std::path::PathBuf> = None;
    for (flag, slot) in [
        ("--record", &mut record),
        ("--trace-chrome", &mut trace_chrome),
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
    match run(&root, &task, record.as_deref()) {
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
