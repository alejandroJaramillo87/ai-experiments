/*
 * tiny-model — write the synthetic tiny-model fixtures to a path.
 *
 * The output is deterministic in the default TinySpec seed, so regenerating
 * the committed fixture always reproduces identical bytes; the committed
 * Python reference (fixtures/reference_logits.py) reads the same file.
 *
 * Given a stem, writes three artifacts describing one model:
 *
 *   <stem>.tokenizer.json          the vocabulary, for the tokenizers crate
 *   <stem>/config.json             the checkpoint's architecture
 *   <stem>/model.safetensors       the checkpoint's weights
 *
 * All three come from one TinySpec, so a fixture set cannot describe two
 * different models.
 */

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use runtime::synth::{
    build_tiny_config_json, build_tiny_safetensors, build_tiny_tokenizer_json,
    merge_fixture_tokenizer_json, TinySpec,
};

/// Drop a trailing `.gguf`, so the stem is the same whether the caller names
/// the image or the fixture. Every other artifact hangs off this.
fn stem_of(path: &str) -> &str {
    match Path::new(path).extension().map(|e| e == "gguf") {
        Some(true) => &path[..path.len() - 5],
        _ => path,
    }
}

/// Write `bytes` to `path`, reporting either way. Returns false on failure so
/// the caller can stop at the first one rather than leaving a half-written
/// fixture set that looks complete.
fn write(path: &Path, bytes: &[u8]) -> bool {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                eprintln!("[ERROR] Create {}: {e}", dir.display());
                return false;
            }
        }
    }
    match std::fs::write(path, bytes) {
        Ok(()) => {
            eprintln!("[OK] Wrote {} bytes to {}", bytes.len(), path.display());
            true
        }
        Err(e) => {
            eprintln!("[ERROR] Write {}: {e}", path.display());
            false
        }
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
            eprintln!("usage: tiny-model <output-stem> [e4b] | <output.json> bpe");
            return ExitCode::from(2);
        }
    };
    let stem = stem_of(path);
    let checkpoint = PathBuf::from(stem);

    let artifacts: Vec<(PathBuf, Vec<u8>)> = match (|| -> anyhow::Result<_> {
        Ok(vec![
            (
                PathBuf::from(format!("{stem}.tokenizer.json")),
                build_tiny_tokenizer_json(&spec)?,
            ),
            (
                checkpoint.join(runtime::checkpoint::CONFIG_FILE),
                build_tiny_config_json(&spec)?,
            ),
            (
                checkpoint.join(runtime::checkpoint::WEIGHTS_FILE),
                build_tiny_safetensors(&spec)?,
            ),
        ])
    })() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[ERROR] Build fixtures for {stem}: {e:#}");
            return ExitCode::from(1);
        }
    };
    // Built before any is written, so a failure leaves no partial set behind.
    for (path, bytes) in &artifacts {
        if !write(path, bytes) {
            return ExitCode::from(1);
        }
    }
    ExitCode::SUCCESS
}
