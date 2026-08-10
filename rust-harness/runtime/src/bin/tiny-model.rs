/*
 * tiny-model — write the synthetic tiny-model fixtures to a path.
 *
 * The output is deterministic in the default TinySpec seed, so regenerating
 * the committed fixture always reproduces identical bytes; the committed
 * Python reference (fixtures/reference_logits.py) reads the same file.
 *
 * Writes the GGUF image the CPU backend loads today, and beside it the
 * tokenizer.json the Hugging Face tokenizer reads. Both come from one
 * TinySpec, so a fixture pair cannot describe two different models.
 */

use std::path::Path;
use std::process::ExitCode;

use runtime::synth::{
    build_tiny_model, build_tiny_tokenizer_json, merge_fixture_tokenizer_json, TinySpec,
};

/// `<stem>.gguf` becomes `<stem>.tokenizer.json`, so the pair stays adjacent
/// and sorts together whatever the fixture is called.
fn tokenizer_path(model: &str) -> String {
    match Path::new(model).extension().map(|e| e == "gguf") {
        Some(true) => format!("{}.tokenizer.json", &model[..model.len() - 5]),
        _ => format!("{model}.tokenizer.json"),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (path, spec) = match args.as_slice() {
        [path] => (path, TinySpec::default()),
        [path, variant] if variant == "e4b" => (path, TinySpec::e4b_shaped()),
        // Tokenizer-only: a vocabulary with a merge list, which no model
        // fixture can carry. No weights, so no GGUF is written.
        [path, variant] if variant == "bpe" => {
            return match merge_fixture_tokenizer_json()
                .and_then(|bytes| Ok((std::fs::write(path, &bytes)?, bytes.len())))
            {
                Ok((_, n)) => {
                    eprintln!("[OK] Wrote {n} bytes to {path}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("[ERROR] Write {path}: {e:#}");
                    ExitCode::from(1)
                }
            };
        }
        _ => {
            eprintln!("usage: tiny-model <output.gguf> [e4b] | <output.json> bpe");
            return ExitCode::from(2);
        }
    };
    let image = build_tiny_model(&spec);
    if let Err(e) = std::fs::write(path, &image) {
        eprintln!("[ERROR] Write {path}: {e}");
        return ExitCode::from(1);
    }
    eprintln!("[OK] Wrote {} bytes to {path}", image.len());

    let tokenizer_path = tokenizer_path(path);
    let tokenizer = match build_tiny_tokenizer_json(&spec) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("[ERROR] Build tokenizer for {path}: {e:#}");
            return ExitCode::from(1);
        }
    };
    if let Err(e) = std::fs::write(&tokenizer_path, &tokenizer) {
        eprintln!("[ERROR] Write {tokenizer_path}: {e}");
        return ExitCode::from(1);
    }
    eprintln!("[OK] Wrote {} bytes to {tokenizer_path}", tokenizer.len());
    ExitCode::SUCCESS
}
