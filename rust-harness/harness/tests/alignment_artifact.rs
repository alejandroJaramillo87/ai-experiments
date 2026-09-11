/*
 * alignment_artifact — the written alignment.jsonl against its contract.
 *
 * The key sets are written out once here and once in schemas/alignment.json,
 * with no shared definition between them, so adding a field to the writer
 * without amending the contract fails here. This is the Rust half of the
 * guard; the Python half is tests/hardware/test_alignment.py, which reads
 * the artifact on the workstation against the real vocabulary.
 */

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use harness::alignment::{write, Alignment, Header};
use harness::template::{BOS, TURN_CLOSE, TURN_OPEN};
use runtime::align::Pass;
use runtime::model::probe::Flag;
use runtime::tokenizer::Tokenizer;

const SPECIALS: &[&str] = &[BOS, TURN_OPEN, TURN_CLOSE];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tokenizer() -> Tokenizer {
    Tokenizer::from_json(
        &runtime::synth::build_tiny_tokenizer_json(&runtime::synth::TinySpec::default()).unwrap(),
    )
    .unwrap()
}

/// A written artifact, as a header value and its rows.
fn artifact(with_probe: bool) -> (serde_json::Value, Vec<serde_json::Value>, Alignment) {
    let tok = tokenizer();
    let prompt = format!("{BOS}{TURN_OPEN}user\nabc{TURN_CLOSE}\n{TURN_OPEN}model\n");
    let generated = tok.encode("abc caf\u{00e9}").unwrap();
    let a = Alignment::build(&tok, &prompt, SPECIALS, &generated).unwrap();
    let n = a.rows.len();

    let passes = vec![
        Pass {
            index: 0,
            start: 0,
            len: n - generated.len(),
        },
        Pass {
            index: 1,
            start: n - generated.len(),
            len: generated.len(),
        },
    ];
    let scores: Vec<Flag> = if with_probe {
        a.rows
            .iter()
            .map(|r| Flag {
                position: r.position,
                score: 0.5,
            })
            .collect()
    } else {
        Vec::new()
    };

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("alignment.jsonl");
    write(
        &out,
        &Header {
            run_id: "R-20260101-000000-abcdef",
            model_hash: "0123456789abcdef",
            probe: with_probe.then_some("test-probe"),
            probe_layer: with_probe.then_some(2),
        },
        &a,
        &passes,
        &scores,
    )
    .unwrap();

    let body = std::fs::read_to_string(&out).unwrap();
    let mut lines = body.lines();
    let head = serde_json::from_str(lines.next().unwrap()).unwrap();
    let rows = lines.map(|l| serde_json::from_str(l).unwrap()).collect();
    // The temporary directory drops here; the artifact has been read.
    (head, rows, a)
}

fn keys(v: &serde_json::Value) -> BTreeSet<String> {
    v.as_object()
        .expect("object")
        .keys()
        .map(String::from)
        .collect()
}

fn contract() -> Option<serde_json::Value> {
    let path = repo_root().join("schemas/alignment.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        runtime::testlog::skip(
            "contract assertions",
            &format!("{} is absent", path.display()),
        );
        return None;
    };
    Some(serde_json::from_str(&text).expect("contract is valid JSON"))
}

fn named(contract: &serde_json::Value, section: &str, part: &str) -> BTreeSet<String> {
    contract[section][part]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect()
}

/// Hold one written object's key set to the contract's bound for `part`:
/// every required key present, and no key the contract does not name.
///
/// A function rather than an inlined loop because the artifact reaches this
/// file two ways -- from `write` directly and from a spawned binary -- and a
/// second copy of the bound would let the two paths be held to different
/// contracts, which is the drift this file exists to prevent.
fn assert_conforms(contract: &serde_json::Value, part: &str, got: &BTreeSet<String>) {
    let required = named(contract, "required", part);
    let optional = named(contract, "optional", part);
    assert!(
        required.is_subset(got),
        "{part} is missing required keys {:?}",
        required.difference(got).collect::<Vec<_>>()
    );
    let allowed: BTreeSet<String> = required.union(&optional).cloned().collect();
    assert!(
        got.is_subset(&allowed),
        "{part} carries keys the contract does not name: {:?}",
        got.difference(&allowed).collect::<Vec<_>>()
    );
}

#[test]
fn the_header_and_rows_carry_exactly_the_contract_key_sets() {
    let Some(contract) = contract() else { return };
    let (head, rows, _) = artifact(true);

    for (part, got) in [
        ("header", keys(&head)),
        ("row", keys(&rows[0])),
        // The last row is scored and the first prompt row may not be, so both
        // ends are checked against the same required-plus-optional bound.
        ("row", keys(rows.last().unwrap())),
    ] {
        assert_conforms(&contract, part, &got);
    }
}

#[test]
fn every_enumerated_field_holds_a_value_the_contract_lists() {
    let Some(contract) = contract() else { return };
    let (head, rows, _) = artifact(true);

    let legal = |field: &str| -> BTreeSet<String> {
        contract["values"][field]
            .as_array()
            .expect("values array")
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    };
    assert!(legal("artifact").contains(head["artifact"].as_str().unwrap()));
    for row in &rows {
        let role = row["role"].as_str().expect("role is a string");
        assert!(legal("role").contains(role), "role {role:?} in {row}");
        if let Some(phase) = row.get("phase") {
            let phase = phase.as_str().expect("phase is a string");
            assert!(legal("phase").contains(phase), "phase {phase:?} in {row}");
        }
    }
}

/// The round trip IF-2 asks for, taken from the written bytes rather than
/// from the in-memory table: a reader has only the file.
#[test]
fn concatenating_the_written_spans_reproduces_the_written_text() {
    let (head, rows, _) = artifact(false);
    let text = head["text"].as_str().expect("header carries the text");
    let mut joined = String::new();
    for row in &rows {
        let start = row["byte_start"].as_u64().unwrap() as usize;
        let end = row["byte_end"].as_u64().unwrap() as usize;
        assert!(start <= end && end <= text.len(), "{row}");
        joined.push_str(&text[start..end]);
    }
    assert_eq!(joined, text, "the written spans do not reproduce the text");
}

#[test]
fn positions_are_dense_from_zero_and_counted_in_the_header() {
    let (head, rows, a) = artifact(true);
    assert_eq!(head["positions"].as_u64().unwrap() as usize, rows.len());
    assert_eq!(rows.len(), a.rows.len());
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row["position"].as_u64().unwrap() as usize, i, "{row}");
    }
}

/// The id is what lets a reader other than the harness check a span: the
/// row's `token_id` is the id the run computed at that position, prompt and
/// generation alike.
#[test]
fn every_row_carries_the_id_the_run_computed_at_its_position() {
    let (_, rows, a) = artifact(false);
    let tok = tokenizer();
    let prompt = format!("{BOS}{TURN_OPEN}user\nabc{TURN_CLOSE}\n{TURN_OPEN}model\n");
    let prompt_ids = tok.encode_prompt(&prompt, SPECIALS).unwrap();
    assert_eq!(rows.len(), a.rows.len());
    for (i, row) in rows.iter().enumerate() {
        let id = row["token_id"].as_u64().expect("token_id is an integer") as u32;
        assert_eq!(id, a.rows[i].token, "{row}");
        if i < prompt_ids.len() {
            assert_eq!(id, prompt_ids[i], "prompt position {i}: {row}");
        }
    }
}

#[test]
fn a_scored_run_carries_the_score_channel_and_an_unscored_one_omits_it() {
    let (scored_head, scored, _) = artifact(true);
    let (plain_head, plain, _) = artifact(false);

    assert_eq!(scored_head["probe"], "test-probe");
    assert!(plain_head.get("probe").is_none(), "{plain_head}");
    assert!(
        scored.iter().all(|r| r.get("score").is_some()),
        "a scored run left a position without its score"
    );
    assert!(
        plain.iter().all(|r| r.get("score").is_none()),
        "a run with no probe wrote a score anyway"
    );
    // Spans and passes are the other channel and do not depend on the probe.
    assert!(plain.iter().all(|r| r.get("pass").is_some()), "{plain:?}");
}

/// The fixture checkpoint the spawned run below serves. Random weights, so
/// nothing here says a token is right; what it says is that a run produced an
/// artifact of the shape the contract describes.
fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// The artifact a spawned harness leaves at the path `--alignment-out` names.
///
/// Every test above calls `write` directly, so none of them can fail if the
/// flag is never wired to it: the writer would keep its contract while no run
/// ever produced a file. The file name is not the default of anything, so its
/// existence is attributable to the flag alone.
///
/// No probe, because the score channel is optional in the contract and the
/// flag is independent of it; one token, because the artifact's shape does not
/// depend on the budget. stdin is closed so a tool call cannot block on the
/// interactive prompt, as probe_identity.rs does for the same reason.
fn artifact_through_the_binary() -> (serde_json::Value, Vec<serde_json::Value>) {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("named-alignment.jsonl");
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_harness"))
        .current_dir(dir.path())
        .arg("--model")
        .arg(fixture_model())
        .arg("--alignment-out")
        .arg(&out)
        .arg("--max-tokens")
        .arg("1")
        .arg("say ok")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("spawn harness");
    let err = String::from_utf8_lossy(&run.stderr).to_string();
    assert!(run.status.success(), "the run failed: {err}");
    assert!(
        out.is_file(),
        "--alignment-out named {} and nothing was written there: {err}",
        out.display()
    );
    let body = std::fs::read_to_string(&out).unwrap();
    let mut lines = body.lines();
    let head = serde_json::from_str(lines.next().expect("a header line")).unwrap();
    let rows = lines.map(|l| serde_json::from_str(l).unwrap()).collect();
    // The temporary directory drops here; the artifact has been read.
    (head, rows)
}

#[test]
fn the_alignment_out_flag_writes_an_artifact_conforming_to_the_same_contract() {
    let Some(contract) = contract() else { return };
    let (head, rows) = artifact_through_the_binary();
    assert!(
        !rows.is_empty(),
        "the artifact carries no positions: {head}"
    );

    assert_conforms(&contract, "header", &keys(&head));
    for row in [&rows[0], rows.last().unwrap()] {
        assert_conforms(&contract, "row", &keys(row));
    }
    let artifact = head["artifact"].as_str().expect("artifact is a string");
    let legal: BTreeSet<String> = contract["values"]["artifact"]
        .as_array()
        .expect("values array")
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(legal.contains(artifact), "artifact {artifact:?} in {head}");
}

/// The two channels a library call cannot check: that the run recorded its
/// forward passes, and that the positions the file carries are the run's own
/// rather than a residue of the in-memory table.
#[test]
fn the_flagged_run_records_every_pass_and_counts_its_positions_in_the_header() {
    let (head, rows) = artifact_through_the_binary();
    assert_eq!(
        head["positions"].as_u64().unwrap() as usize,
        rows.len(),
        "the header counts {} positions against {} rows",
        head["positions"],
        rows.len()
    );
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row["position"].as_u64().unwrap() as usize, i, "{row}");
        // `record_alignment` is started only when the flag is given, so a pass
        // on every row is what says the recording was armed for the whole run
        // and not from some later point in it.
        assert!(row.get("pass").is_some(), "position {i} has no pass: {row}");
    }
}
