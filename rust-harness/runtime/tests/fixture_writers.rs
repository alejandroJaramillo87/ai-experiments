/*
 * fixture_writers — what tiny-model and long-context-golden actually wrote.
 *
 * These two binaries exist to produce committed fixtures, and everything else
 * in the tree that reads a fixture assumes the writer's output is loadable.
 * The coverage they had stopped at the command line: `binary_cli_surface.rs`
 * asserts that asking either for help writes nothing, which says nothing about
 * the case where they do write. An exit code cannot distinguish a checkpoint
 * from a file of the right name holding a truncated header.
 *
 * So each artifact is read back through the consumer that will read it for
 * real -- the checkpoint through `CpuModel::load` and a prefill, the tokenizer
 * through `runtime::tokenizer`, the golden through `serde_json` and the
 * constants the reading test shares with the writer. A round trip through the
 * writer's own structures would only prove it is self-consistent.
 *
 * Determinism is asserted alongside, because it is the property that makes
 * these fixtures committable at all: `tiny-model`'s header claims regenerating
 * reproduces identical bytes, and a fixture that did not would turn every
 * regeneration into a diff nobody can review.
 */

use std::path::Path;
use std::process::{Command, Output};

use runtime::model::cpu::CpuModel;
use runtime::model::Model;

/// Run a fixture writer in `dir`, so nothing it writes can reach the worktree.
fn run_in(dir: &Path, exe: &str, args: &[&str]) -> Output {
    Command::new(exe)
        .args(args)
        .current_dir(dir)
        .output()
        .expect("the binary builds as a test binary")
}

fn stderr(out: &Output) -> String {
    String::from_utf8(out.stderr.clone()).expect("stderr is UTF-8")
}

/// The checkpoint `tiny-model` writes loads, and answers a forward pass.
///
/// Both halves matter. A `config.json` that parses and a `model.safetensors`
/// whose tensors are the wrong shape gets through a loader that only reads the
/// header, and the first thing to notice would be a test elsewhere failing on
/// numerics with no sign of where the bad bytes came from.
#[test]
fn tiny_model_writes_a_checkpoint_that_loads_and_prefills() {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let out = run_in(dir.path(), env!("CARGO_BIN_EXE_tiny-model"), &["fixture"]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));

    let checkpoint = dir.path().join("fixture");
    for name in ["config.json", "model.safetensors"] {
        assert!(
            checkpoint.join(name).exists(),
            "no {name} under {}",
            checkpoint.display()
        );
    }
    assert!(dir.path().join("fixture.tokenizer.json").exists());

    let (mut model, fingerprint) =
        CpuModel::load(&checkpoint, false).expect("the written checkpoint loads");
    assert_ne!(fingerprint, 0, "the checkpoint hashed to nothing");
    let vocab = model.config().vocab_size;
    assert!(vocab > 0, "the checkpoint declares an empty vocabulary");

    let logits = model
        .prefill(&[1, 5, 9])
        .expect("prefill the written weights");
    assert_eq!(logits.0.len(), vocab, "the row is not vocabulary-wide");
    // Non-finite logits are the shape a plausible-looking but wrong tensor
    // takes, and are an error at the point of detection everywhere else in
    // this workspace.
    let bad = logits.0.iter().position(|v| !v.is_finite());
    assert!(bad.is_none(), "non-finite logit at index {bad:?}");
}

/// The tokenizer written beside the checkpoint is a tokenizer.
///
/// Read through `runtime::tokenizer`, which is the `tokenizers` crate's own
/// reader: the format belongs to HuggingFace, so anything else here would be a
/// second transcription agreeing with the first.
#[test]
fn tiny_model_writes_a_tokenizer_the_tokenizers_crate_reads() {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let out = run_in(dir.path(), env!("CARGO_BIN_EXE_tiny-model"), &["fixture"]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));

    let t = runtime::tokenizer::Tokenizer::from_file(&dir.path().join("fixture.tokenizer.json"))
        .expect("the written tokenizer loads");
    let ids = t.encode("hello world").expect("encode through the fixture");
    assert!(!ids.is_empty(), "the vocabulary encoded nothing");

    // Every id is addressable by the checkpoint written from the same spec.
    // The two files are one model or they are not a fixture set, and a
    // tokenizer whose ids overrun the embedding table is the way that fails.
    let (model, _) = CpuModel::load(&dir.path().join("fixture"), false).expect("load the fixture");
    let vocab = model.config().vocab_size as u32;
    let over = ids.iter().find(|&&id| id >= vocab);
    assert!(
        over.is_none(),
        "token id {over:?} is outside a {vocab} vocabulary"
    );
}

/// The `bpe` variant writes the merge-list vocabulary and no checkpoint.
///
/// It is the one fixture in the tree with a merge list, and the reason it is a
/// separate invocation is that no model fixture can carry one. A run that
/// quietly wrote weights too would be a different artifact from the one
/// `tokenizer_golden.rs` documents regenerating.
#[test]
fn tiny_model_bpe_writes_only_a_tokenizer() {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let out = run_in(
        dir.path(),
        env!("CARGO_BIN_EXE_tiny-model"),
        &["vocab.json", "bpe"],
    );
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));

    let written: Vec<String> = std::fs::read_dir(dir.path())
        .expect("the scratch directory is readable")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(written, vec!["vocab.json".to_string()], "{written:?}");

    let bytes = std::fs::read(dir.path().join("vocab.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&String::from_utf8(bytes).unwrap())
        .expect("the written vocabulary is JSON");
    let merges = json["model"]["merges"]
        .as_array()
        .expect("the bpe fixture carries a merge list");
    assert!(!merges.is_empty(), "the merge list is empty");

    let t = runtime::tokenizer::Tokenizer::from_file(&dir.path().join("vocab.json"))
        .expect("the written vocabulary loads");
    assert!(!t.encode("hello").unwrap().is_empty());
}

/// The `e4b` variant writes a differently shaped model, not the same one twice.
///
/// A variant flag that was accepted and ignored would leave every test that
/// asks for the wider shape measuring the narrow one.
#[test]
fn tiny_model_e4b_writes_a_different_architecture_from_the_default() {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    for args in [vec!["plain"], vec!["wide", "e4b"]] {
        let out = run_in(dir.path(), env!("CARGO_BIN_EXE_tiny-model"), &args);
        assert_eq!(out.status.code(), Some(0), "{args:?}: {}", stderr(&out));
    }
    let read = |name: &str| {
        let bytes = std::fs::read(dir.path().join(name).join("config.json")).unwrap();
        runtime::checkpoint::parse_text_config(&bytes).expect("the written config parses")
    };
    let (plain, wide) = (read("plain"), read("wide"));
    assert_ne!(
        (plain.layer_types.clone(), plain.num_kv_shared_layers),
        (wide.layer_types.clone(), wide.num_kv_shared_layers),
        "e4b produced the default architecture: {:?} vs {:?}",
        plain.layer_types,
        wide.layer_types
    );
    // The wider shape is the one with a shared-KV tail, which is the feature it
    // is asked for by name.
    assert!(
        wide.num_kv_shared_layers > 0,
        "the e4b fixture shares no KV layers"
    );
    CpuModel::load(&dir.path().join("wide"), false).expect("the e4b checkpoint loads");
}

/// Regenerating reproduces identical bytes, which is what makes the output
/// committable.
#[test]
fn tiny_model_regenerates_byte_identical_artifacts() {
    let mut runs = Vec::new();
    for _ in 0..2 {
        let dir = tempfile::tempdir().expect("a scratch working directory");
        let out = run_in(dir.path(), env!("CARGO_BIN_EXE_tiny-model"), &["fixture"]);
        assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
        let read = |p: &str| std::fs::read(dir.path().join(p)).unwrap();
        runs.push((
            read("fixture.tokenizer.json"),
            read("fixture/config.json"),
            read("fixture/model.safetensors"),
        ));
        // The directory is dropped here, after its bytes have been read.
    }
    assert_eq!(runs[0].0, runs[1].0, "the tokenizer is not reproducible");
    assert_eq!(runs[0].1, runs[1].1, "the config is not reproducible");
    assert!(
        runs[0].2 == runs[1].2,
        "model.safetensors is not reproducible: {} then {} bytes",
        runs[0].2.len(),
        runs[1].2.len()
    );
}

/// The golden `long-context-golden` writes is the fixture its reader expects.
///
/// Checked against `conformance::golden`, which the writer and
/// `runtime/tests/long_context.rs` both read: the assertion is that the file
/// describes the model and the positions that module names, since a golden
/// recorded over different tokens is a change detector for something nobody is
/// running.
#[test]
fn long_context_golden_writes_the_rows_it_claims() {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let out = run_in(
        dir.path(),
        env!("CARGO_BIN_EXE_long-context-golden"),
        &["golden.json"],
    );
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));

    let text = std::fs::read_to_string(dir.path().join("golden.json")).expect("the golden exists");
    let v: serde_json::Value = serde_json::from_str(&text).expect("the golden is JSON");

    let spec = runtime::conformance::golden::spec();
    assert_eq!(
        v["tokens"].as_u64(),
        Some(runtime::conformance::golden::TOKENS as u64),
        "{v}"
    );
    assert_eq!(
        v["sliding_window"].as_u64(),
        Some(spec.sliding_window as u64),
        "{v}"
    );
    assert_eq!(v["n_layers"].as_u64(), Some(spec.n_layers as u64), "{v}");
    assert_eq!(
        v["shared_kv_layers"].as_u64(),
        Some(spec.shared_kv_layers as u64),
        "{v}"
    );
    assert_eq!(v["seed"].as_u64(), Some(spec.seed), "{v}");

    let vocab = v["vocab_size"]
        .as_u64()
        .expect("the golden states a vocabulary") as usize;
    let rows = v["rows"].as_array().expect("the golden carries rows");
    assert_eq!(
        rows.len(),
        runtime::conformance::golden::POSITIONS.len(),
        "{rows:?}"
    );
    for (row, &want) in rows.iter().zip(&runtime::conformance::golden::POSITIONS) {
        assert_eq!(row["position"].as_u64(), Some(want as u64), "{row}");
        let logits = row["logits"].as_array().expect("a row carries logits");
        assert_eq!(
            logits.len(),
            vocab,
            "position {want} is not vocabulary-wide"
        );
        let bad = logits
            .iter()
            .position(|x| !x.as_f64().is_some_and(f64::is_finite));
        assert!(
            bad.is_none(),
            "position {want}: non-finite logit at {bad:?}"
        );
    }
    // Every recorded row is not the same row: a writer that sampled one
    // position six times would satisfy every assertion above.
    assert_ne!(rows[0]["logits"], rows[5]["logits"], "{rows:?}");
}

/// The golden is reproducible, which is what lets a reader tell a deliberate
/// re-recording from a drifting one.
#[test]
fn long_context_golden_regenerates_byte_identical_bytes() {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let mut written = Vec::new();
    for name in ["first.json", "second.json"] {
        let out = run_in(
            dir.path(),
            env!("CARGO_BIN_EXE_long-context-golden"),
            &[name],
        );
        assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
        written.push(std::fs::read(dir.path().join(name)).unwrap());
    }
    assert!(
        written[0] == written[1],
        "two runs disagreed: {} then {} bytes",
        written[0].len(),
        written[1].len()
    );
}
