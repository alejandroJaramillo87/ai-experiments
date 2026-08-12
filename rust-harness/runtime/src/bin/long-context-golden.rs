/*
 * long-context-golden — record the logits of a long prefill, as a tripwire.
 *
 * Companion to `tiny-model`: a utility binary whose only job is to write a
 * committed fixture. The fixture here is `fixtures/long-context-logits.json`,
 * read by `runtime/tests/long_context.rs`.
 *
 * Why a committed file rather than a regenerated one. Every other synthetic
 * fixture in this workspace is derivable, and its test regenerates it and
 * byte-compares, so drift is impossible. That construction cannot serve this
 * purpose: regenerating from the current engine and comparing against the
 * current engine agrees by definition. A change detector has to hold bytes
 * recorded at a point in time, so that a package rewriting how the cache is
 * stored has to either reproduce them or re-run this binary deliberately and
 * say why in the commit.
 *
 * That has happened once. Capping sliding-layer KV storage at the window
 * changed both the layout of the cache and how attention indexes it, and
 * these bytes came back unchanged -- the outcome the fixture exists to be
 * able to report, and one no other construct in the workspace could have.
 *
 * What it is not. It is not an oracle. We wrote the engine that produced it,
 * so it is evidence of nothing except that behaviour has not changed since it
 * was recorded, exactly as `/rust-harness/README.md` says of the other
 * goldens. Its standing comes from elsewhere: the window convention it
 * encodes was checked against transformers at 150 tokens on the Tier T
 * fixture (`runtime/tests/ladder_tiny.rs`), and this file was blessed only
 * after that passed. Blessing it first would have frozen whatever convention
 * we happened to hold and handed the storage cap a wrong answer to preserve.
 *
 * Usage, from the rust-harness directory:
 *
 *     cargo run -p runtime --bin long-context-golden -- \
 *         fixtures/long-context-logits.json
 */

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use runtime::model::Model;

fn main() -> ExitCode {
    let Some(out) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("[ERROR] Destination: no path given");
        eprintln!("  Solution: name the file to write");
        eprintln!(
            "  Command: cargo run -p runtime --bin long-context-golden -- \
             fixtures/long-context-logits.json"
        );
        return ExitCode::from(2);
    };
    match write(&out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[ERROR] {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn write(out: &PathBuf) -> Result<()> {
    let spec = runtime::conformance::golden::spec();
    let tokens = runtime::conformance::golden::tokens(runtime::conformance::golden::TOKENS);

    let files = runtime::synth::tiny_checkpoint(&spec).context("build the synthetic checkpoint")?;
    let ck = runtime::checkpoint::parse(&files).context("parse the synthetic checkpoint")?;
    let mut model =
        runtime::model::cpu::CpuModel::from_checkpoint(&ck).context("load the checkpoint")?;
    let rows = model.prefill_all(&tokens).context("prefill")?;

    let sampled: Vec<serde_json::Value> = runtime::conformance::golden::POSITIONS
        .iter()
        .map(|&p| {
            serde_json::json!({
                "position": p,
                "logits": rows[p].0,
            })
        })
        .collect();

    let payload = serde_json::json!({
        "note": "change detector, not an oracle -- see the binary that wrote it",
        "tokens": tokens.len(),
        "vocab_size": model.config().vocab_size,
        "sliding_window": spec.sliding_window,
        "n_layers": spec.n_layers,
        "shared_kv_layers": spec.shared_kv_layers,
        "seed": spec.seed,
        "rows": sampled,
    });
    std::fs::write(out, serde_json::to_string(&payload)?)
        .with_context(|| format!("write {}", out.display()))?;
    eprintln!(
        "[OK] Wrote {}: {} rows of {} over {} tokens",
        out.display(),
        runtime::conformance::golden::POSITIONS.len(),
        model.config().vocab_size,
        tokens.len()
    );
    Ok(())
}
