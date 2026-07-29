/*
 * tiny-model — write the synthetic tiny-model GGUF fixture to a path.
 *
 * The output is deterministic in the default TinySpec seed, so regenerating
 * the committed fixture always reproduces identical bytes; the committed
 * Python reference (fixtures/reference_logits.py) reads the same file.
 */

use std::process::ExitCode;

use runtime::synth::{build_tiny_model, TinySpec};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path] = args.as_slice() else {
        eprintln!("usage: tiny-model <output.gguf>");
        return ExitCode::from(2);
    };
    let image = build_tiny_model(&TinySpec::default());
    if let Err(e) = std::fs::write(path, &image) {
        eprintln!("[ERROR] Write {path}: {e}");
        return ExitCode::from(1);
    }
    eprintln!("[OK] Wrote {} bytes to {path}", image.len());
    ExitCode::SUCCESS
}
