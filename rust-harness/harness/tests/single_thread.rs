/*
 * single_thread — the harness's own --single-thread reaches the model.
 *
 * The runtime binaries' flag of the same name is asserted in
 * `capture_obs.rs` and `ladder_flags.rs`; the harness's was help-listed and
 * passed once on the workstation under rr, and nothing read the width back.
 * The run header prints `model.threads()`, the width the loaded model will
 * split rows across, so a flag the parser accepted and `main.rs` dropped
 * would leave that field at the runtime's default width.
 */

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// The `run.header` event of one task run with `extra` appended.
fn header_with(extra: &[&str]) -> serde_json::Value {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let logs = dir.path().join("logs");
    let model = fixture_model();
    let out = Command::new(env!("CARGO_BIN_EXE_harness"))
        .args(["--model", model.to_str().unwrap(), "--max-tokens", "1"])
        .args(extra)
        .arg("say hi")
        .current_dir(dir.path())
        .env("OBS_LOG_DIR", &logs)
        .stdin(Stdio::null())
        .output()
        .expect("the harness builds as a test binary");
    assert!(
        out.status.success(),
        "{extra:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let harness_dir = logs.join("harness");
    std::fs::read_dir(&harness_dir)
        .unwrap_or_else(|e| panic!("{}: {e}", harness_dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.file_name().is_some_and(|n| n != "runs.jsonl"))
        .flat_map(|p| {
            std::fs::read_to_string(&p)
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str(l).expect("each line is one JSON object"))
                .collect::<Vec<serde_json::Value>>()
        })
        .find(|e| e["event"] == "run.header")
        .expect("the run wrote a run.header")
}

/// With the flag the header reports width one; without it, the runtime's
/// default, the machine's width under `DEFAULT_MAX_THREADS`' cap, which is
/// one only on a one-CPU host. Both are read so a header pinned to one would
/// fail on any wider machine. The oracle is the runtime's own function, not
/// `available_parallelism()`: the raw count agrees with the default only on
/// a host at or under the cap, which the sandbox is and the workstation is
/// not.
#[test]
fn single_thread_is_the_width_the_run_header_reports() {
    let narrow = header_with(&["--single-thread"]);
    assert_eq!(narrow["threads"].as_u64(), Some(1), "{narrow}");
    let wide = header_with(&[]);
    let default = runtime::model::cpu::default_threads() as u64;
    assert_eq!(wide["threads"].as_u64(), Some(default), "{wide}");
}
