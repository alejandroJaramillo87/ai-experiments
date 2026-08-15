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
use runtime::cli::{Flag, Kind, Spec};
use runtime::model::Model;

const SPEC: Spec = Spec {
    program: "long-context-golden",
    summary: "Write the committed long-prefill logits fixture to the named file.",
    positionals: "<out.json>",
    required: &[],
    flags: &[
        Flag {
            name: "--help",
            kind: Kind::Bare,
            help: "print this help and exit",
        },
        Flag {
            name: "-h",
            kind: Kind::Bare,
            help: "print this help and exit",
        },
    ],
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if runtime::cli::handled_help(&SPEC, &args) {
        return ExitCode::SUCCESS;
    }
    // The destination goes through `extract` rather than being read off
    // `nth(1)`, which is what stops `-h` from becoming an output path. Until
    // U3b this program wrote the golden fixture to a file literally named `-h`
    // when asked for help, and silently discarded every argument past the
    // first.
    let parsed = match runtime::cli::extract(&SPEC, args, &mut []) {
        Ok(p) => p,
        Err(e) => return runtime::cli::usage_error(&SPEC, &e),
    };
    let out = match parsed.rest.as_slice() {
        [one] => PathBuf::from(one),
        [] => {
            runtime::status::error("Destination: no path given");
            // The two continuation lines are untagged by the repo's error shape
            // (rust-harness/Makefile:70-86): one tagged statement of what
            // failed, then indented remedy lines. `status::error_at` would tag
            // each of them -- `  [ERROR] Solution: ...` -- which is a different
            // message, so these stay `plain` with the indent in the literal.
            runtime::status::plain("  Solution: name the file to write");
            runtime::status::plain(
                "  Command: cargo run -p runtime --bin long-context-golden -- \
                 fixtures/long-context-logits.json",
            );
            return ExitCode::from(2);
        }
        many => {
            return runtime::cli::usage_error(
                &SPEC,
                &format!("one destination, {} given", many.len()),
            )
        }
    };
    match write(&out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            runtime::status::error(&format!("{e:#}"));
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
    runtime::status::ok(&format!(
        "Wrote {}: {} rows of {} over {} tokens",
        out.display(),
        runtime::conformance::golden::POSITIONS.len(),
        model.config().vocab_size,
        tokens.len()
    ));
    Ok(())
}
