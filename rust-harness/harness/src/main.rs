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
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // --record <path>: wrap the model in the flight recorder, appending every
    // forward call to a JSONL file replayable as a MockModel script.
    let mut record: Option<std::path::PathBuf> = None;
    if let Some(i) = args.iter().position(|a| a == "--record") {
        if i + 1 >= args.len() {
            eprintln!("usage: harness [--record <path>] <task description>");
            return ExitCode::from(2);
        }
        record = Some(std::path::PathBuf::from(args.remove(i + 1)));
        args.remove(i);
    }
    if args.is_empty() {
        eprintln!("usage: harness [--record <path>] <task description>");
        return ExitCode::from(2);
    }
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
