/*
 * capture_obs — the three instrumented capture binaries' run records.
 *
 * `activation_capture.rs` and `attention_capture.rs` next door prove the
 * artifacts. This file proves the runs said what they were: that each leaves a
 * header naming the checkpoint, the layers and the thread width it actually
 * used, a `model.load` phase with a duration, and exactly one catalog row.
 * None of that is assertable from the library, because every one of those
 * records is written by the binary.
 *
 * `ladder_obs.rs` is the same shape for the fourth instrumented binary, and
 * this file deliberately mirrors its helpers rather than inventing a second
 * spelling of the same degradation.
 *
 * The three exist as one file rather than three because the interesting
 * assertions are the ones that hold across all of them -- one header, one
 * phase, one row, no run directory -- and a per-binary file would state each
 * three times and let one drift.
 */

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The declared field set for one event: `fields` plus `optional`.
///
/// `None` when the contract file is absent, which is the container build,
/// whose context is `rust-harness/` alone. Mirrors `ladder_obs.rs`'s helper
/// deliberately.
fn contract_event_keys(event: &str) -> Option<BTreeSet<String>> {
    let text = std::fs::read_to_string(repo_root().join("schemas/obs_record.json")).ok()?;
    let contract: serde_json::Value = serde_json::from_str(&text).expect("contract is valid JSON");
    let read = |k: &str| -> Vec<String> {
        contract["events"][event][k]
            .as_array()
            .unwrap_or_else(|| panic!("events.{event}.{k} is an array"))
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    };
    Some(read("fields").into_iter().chain(read("optional")).collect())
}

/// The eight keys every record carries, which no header declares.
const BASE_KEYS: [&str; 8] = [
    "cli", "command", "event", "level", "msg", "pid", "run_id", "ts",
];

/// What one run left behind. `run_dir` is an `Option` because these three
/// write none -- see `no_capture_binary_opens_a_run_directory`.
struct Recorded {
    exit_code: i32,
    events: Vec<serde_json::Value>,
    catalog: Vec<serde_json::Value>,
    run_dir: Option<PathBuf>,
}

impl Recorded {
    fn events_named(&self, event: &str) -> Vec<&serde_json::Value> {
        self.events.iter().filter(|e| e["event"] == event).collect()
    }

    fn one(&self, event: &str) -> &serde_json::Value {
        let found = self.events_named(event);
        assert_eq!(
            found.len(),
            1,
            "expected exactly one {event} among {:#?}",
            self.events
        );
        found[0]
    }

    fn phases(&self) -> Vec<String> {
        self.events_named("phase.start")
            .iter()
            .map(|e| e["phase"].as_str().expect("phase is a string").to_string())
            .collect()
    }
}

fn parse_lines(path: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(|l| serde_json::from_str(l).expect("each line is one JSON object"))
        .collect()
}

/// One binary, its header event, and a closure building a working command line
/// inside a scratch directory.
///
/// Named tuples rather than three near-identical spawn helpers: the assertions
/// below are the same sentence about each binary, and a table is what keeps a
/// fourth instrumented binary one row rather than one copied test.
struct Subject {
    command: &'static str,
    exe: &'static str,
    header: &'static str,
    /// Arguments, given a scratch directory to write outputs into.
    args: fn(&Path) -> Vec<String>,
    /// The fixture file whose absence makes this subject unrunnable.
    needs: fn() -> PathBuf,
}

fn subjects() -> Vec<Subject> {
    vec![
        Subject {
            command: "activation-capture",
            exe: env!("CARGO_BIN_EXE_activation-capture"),
            header: "activation-capture.header",
            args: |dir| {
                let dataset = dir.join("dataset.jsonl");
                std::fs::write(
                    &dataset,
                    "{\"text\": \"anchor harbor rope\", \"label\": 1}\n\
                     {\"text\": \"bridge river stone\", \"label\": 0}\n",
                )
                .unwrap();
                vec![
                    "--model".into(),
                    fixtures().join("tiny-gemma4").display().to_string(),
                    "--dataset".into(),
                    dataset.display().to_string(),
                    "--layers".into(),
                    "0,1".into(),
                    "--out".into(),
                    dir.join("capture").display().to_string(),
                ]
            },
            needs: || fixtures().join("tiny-gemma4/config.json"),
        },
        Subject {
            command: "attention-capture",
            exe: env!("CARGO_BIN_EXE_attention-capture"),
            header: "attention-capture.header",
            args: |dir| {
                vec![
                    "--model".into(),
                    fixtures().join("tiny-gemma4-e4b").display().to_string(),
                    "--prompt".into(),
                    "anchor harbor rope bridge".into(),
                    "--layers".into(),
                    "0".into(),
                    "--heads".into(),
                    "0".into(),
                    "--out".into(),
                    dir.join("attention").display().to_string(),
                ]
            },
            needs: || fixtures().join("tiny-gemma4-e4b/config.json"),
        },
        Subject {
            command: "layer-diff",
            exe: env!("CARGO_BIN_EXE_layer-diff"),
            header: "layer-diff.header",
            args: |_dir| {
                vec![
                    "--model".into(),
                    fixtures().join("tiny-hf-gemma4").display().to_string(),
                    "--reference".into(),
                    fixtures()
                        .join("tiny-hf-gemma4/capture.json")
                        .display()
                        .to_string(),
                ]
            },
            // The Tier T capture is generated on the workstation rather than
            // committed, so its absence is a skip and not a failure -- the
            // same condition `ladder_obs.rs` degrades on.
            needs: || fixtures().join("tiny-hf-gemma4/capture.json"),
        },
    ]
}

/// Run one subject with its log root at a fresh tempdir, or `None` with a
/// `[SKIP]` already logged.
fn record(subject: &Subject, extra: &[&str]) -> Option<(tempfile::TempDir, Recorded)> {
    if !(subject.needs)().exists() {
        runtime::testlog::skip(
            &format!("{} obs records", subject.command),
            &format!(
                "{} is absent; generate the fixture on the workstation",
                (subject.needs)().display()
            ),
        );
        return None;
    }
    let dir = tempfile::tempdir().unwrap();
    let logs = dir.path().join("logs");
    let out = Command::new(subject.exe)
        .args((subject.args)(dir.path()))
        .args(extra)
        // Nothing is resolved against the working directory, but a test that
        // does not say so is one stray relative path away from writing into
        // the worktree.
        .current_dir(dir.path())
        .env("OBS_LOG_DIR", &logs)
        .output()
        .expect("the binary builds as a test binary");

    let harness_dir = logs.join("harness");
    let entries: Vec<PathBuf> = std::fs::read_dir(&harness_dir)
        .unwrap_or_else(|e| panic!("{}: {e}", harness_dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    let events = entries
        .iter()
        .find(|p| p.is_file() && p.file_name().is_some_and(|n| n != "runs.jsonl"))
        .expect("the run wrote an event file")
        .clone();

    Some((
        dir,
        Recorded {
            exit_code: out.status.code().expect("the binary exited normally"),
            events: parse_lines(&events),
            catalog: parse_lines(&harness_dir.join("runs.jsonl")),
            run_dir: entries.into_iter().find(|p| p.is_dir()),
        },
    ))
}

/// The stage's central assertion, and the reason the header events are
/// per-binary rather than shared: key-set equality in **both** directions
/// against the declaration. Containment one way would let a field be declared
/// and never written; the other way would let a field be written and never
/// declared. Neither half is worth having alone.
#[test]
fn every_capture_header_matches_its_declared_fields() {
    let base: BTreeSet<String> = BASE_KEYS
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    for subject in subjects() {
        let Some(declared) = contract_event_keys(subject.header) else {
            runtime::testlog::skip(
                "capture header contract",
                "schemas/obs_record.json is outside this build's context",
            );
            return;
        };
        let Some((_dir, r)) = record(&subject, &[]) else {
            continue;
        };
        let written: BTreeSet<String> = r
            .one(subject.header)
            .as_object()
            .expect("the header is an object")
            .keys()
            .filter(|k| !base.contains(*k))
            .cloned()
            .collect();
        assert_eq!(
            written,
            declared,
            "{}: written but not declared {:?}; declared but not written {:?}",
            subject.command,
            written.difference(&declared).collect::<Vec<_>>(),
            declared.difference(&written).collect::<Vec<_>>()
        );
    }
}

/// The load is what these three spend most of their wall clock on, and until
/// U9c none of them measured it. One phase, opened and closed, with a real
/// number on it.
#[test]
fn every_capture_binary_times_the_model_load() {
    for subject in subjects() {
        let Some((_dir, r)) = record(&subject, &[]) else {
            continue;
        };
        assert_eq!(
            r.phases(),
            vec!["model.load".to_string()],
            "{}: {:#?}",
            subject.command,
            r.events
        );
        let end = r.one("phase.end");
        assert_eq!(end["phase"], "model.load", "{}", subject.command);
        assert!(
            end["duration_s"].is_number(),
            "{}: {end}",
            subject.command
        );
    }
}

/// One run, one row, named for the binary rather than for the crate. The
/// catalog is shared across the workspace under `cli = "harness"`, so the
/// `command` field is the only thing distinguishing a capture from a climb.
#[test]
fn every_capture_binary_leaves_exactly_one_catalog_row_naming_itself() {
    for subject in subjects() {
        let Some((_dir, r)) = record(&subject, &[]) else {
            continue;
        };
        assert_eq!(r.exit_code, 0, "{}: {:#?}", subject.command, r.events);
        assert_eq!(r.catalog.len(), 1, "{}: {:#?}", subject.command, r.catalog);
        assert_eq!(r.catalog[0]["command"], subject.command);
        assert_eq!(r.catalog[0]["exit_code"], 0, "{}", subject.command);
        let header = r.one(subject.header);
        assert_eq!(header["cli"], "harness", "{}", subject.command);
        assert_eq!(header["command"], subject.command);
        assert_eq!(r.one("cmd.end")["exit_code"], 0, "{}", subject.command);
    }
}

/// U9c is header-and-phase, deliberately not metrics: two of the three have no
/// series to plot, and a third writer of the run-directory contract is a cost
/// nobody asked for. Asserted rather than left to the reader, because adding a
/// `metrics_header` call is a one-line change whose consequence -- a directory
/// the retention pruner's `R-*.jsonl` glob never reclaims -- is not local to
/// the line.
#[test]
fn no_capture_binary_opens_a_run_directory() {
    for subject in subjects() {
        let Some((_dir, r)) = record(&subject, &[]) else {
            continue;
        };
        assert!(
            r.run_dir.is_none(),
            "{} opened {:?}",
            subject.command,
            r.run_dir
        );
    }
}

/// The header must report the width the run used, not the one it was
/// configured with by default -- which means it is built after `set_threads`
/// and not before. This is the specific lie the ordering rule prevents, and
/// the reason `--single-thread` is the flag to assert it with: it is the one
/// case where the two answers are guaranteed to differ on a multi-core box.
#[test]
fn the_header_reports_the_thread_width_the_run_actually_used() {
    for subject in subjects() {
        let Some((_dir, r)) = record(&subject, &["--single-thread"]) else {
            continue;
        };
        assert_eq!(
            r.one(subject.header)["threads"],
            1,
            "{}: the header was built before the tuning was applied",
            subject.command
        );
    }
}
