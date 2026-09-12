/*
 * ladder_size — --size reaches the run, and is recorded as the run used it.
 *
 * The flag was passed on the workstation in RH.17 and RH.18 and by no test:
 * `ladder_flags.rs` sweeps the other four tuning flags and left this one,
 * because the rungs it sizes, L3 and L4, need reference artifacts the tiny
 * fixture does not carry. What the fixture can show is that the value
 * arrives: the run header names the size and the corpus length it implies,
 * and both move with the flag. A parser that read `standard` and ran at the
 * default would leave those fields at the default's figures.
 */

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/tiny-hf-gemma4")
}

/// The run header the ladder wrote for one climb with `extra` appended, or
/// `None` with a `[SKIP]` logged when the fixture has not been generated.
fn header_with(extra: &[&str]) -> Option<serde_json::Value> {
    let fixture = fixture_dir();
    if !fixture.join("capture.json").exists() {
        runtime::testlog::skip(
            "ladder --size",
            "fixtures/tiny-hf-gemma4 has not been generated; \
             run fixtures/build_tiny_reference.py on the workstation",
        );
        return None;
    }
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let reference = dir.path().join("reference");
    std::fs::create_dir_all(&reference).unwrap();
    std::fs::copy(fixture.join("capture.json"), reference.join("layers.json")).unwrap();
    let logs = dir.path().join("logs");
    let out = Command::new(env!("CARGO_BIN_EXE_ladder"))
        .args([
            "--model",
            fixture.to_str().unwrap(),
            "--reference",
            reference.to_str().unwrap(),
        ])
        .args(extra)
        .current_dir(dir.path())
        .env("OBS_LOG_DIR", &logs)
        .output()
        .expect("the ladder builds as a test binary");
    assert_eq!(
        out.status.code(),
        Some(0),
        "{extra:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let harness_dir = logs.join("harness");
    let header = std::fs::read_dir(&harness_dir)
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
        .find(|e| e["event"] == "ladder.header")
        .expect("the ladder wrote a ladder.header");
    Some(header)
}

/// The size and corpus length a header records.
fn size_of(header: &serde_json::Value) -> (String, u64) {
    (
        header["size"]
            .as_str()
            .unwrap_or_else(|| panic!("size is not a string: {header}"))
            .to_string(),
        header["corpus_tokens"]
            .as_u64()
            .unwrap_or_else(|| panic!("corpus_tokens is not a number: {header}")),
    )
}

/// Each spelling is recorded by name with the corpus length it implies, and
/// the default is `quick`. The figures are stated here rather than taken
/// from `Size::tokens`, so the expectation is independent of the type under
/// test.
#[test]
fn size_is_recorded_as_given_with_the_corpus_length_it_implies() {
    let Some(default) = header_with(&[]) else {
        return;
    };
    assert_eq!(size_of(&default), ("quick".to_string(), 512), "{default}");
    for (size, tokens) in [("quick", 512), ("standard", 2048), ("deep", 8192)] {
        let header = header_with(&["--size", size]).expect("the fixture was present a moment ago");
        assert_eq!(
            size_of(&header),
            (size.to_string(), tokens),
            "--size {size}: {header}"
        );
    }
}
