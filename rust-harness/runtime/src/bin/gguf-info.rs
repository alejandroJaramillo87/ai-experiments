/*
 * gguf-info — dump GGUF metadata and tensor directory.
 *
 * Inspection tool for resolving model-architecture questions against a real
 * checkpoint: run against the E4B GGUF on the workstation and report the
 * output verbatim. Long arrays (tokenizer vocab) are summarized by length.
 */

use std::process::ExitCode;

use runtime::gguf::{self, MetaValue};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path] = args.as_slice() else {
        eprintln!("usage: gguf-info <model.gguf>");
        return ExitCode::from(2);
    };
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[ERROR] Read {path}: {e}");
            return ExitCode::from(1);
        }
    };
    let file = match gguf::parse(&data) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[ERROR] Parse {path}: {e:#}");
            return ExitCode::from(1);
        }
    };

    println!("=== Metadata ({} keys) ===", file.metadata.len());
    for (key, value) in &file.metadata {
        match value {
            MetaValue::ArrStr(v) => println!("{key} = [str; {}]", v.len()),
            MetaValue::ArrF32(v) => println!("{key} = [f32; {}]", v.len()),
            MetaValue::ArrI32(v) => println!("{key} = [i32; {}]", v.len()),
            other => println!("{key} = {other:?}"),
        }
    }

    println!();
    println!(
        "=== Tensors ({}, data section at byte {}) ===",
        file.tensors.len(),
        file.data_offset
    );
    for t in &file.tensors {
        println!(
            "{}  dims={:?}  type={}  offset={}",
            t.name, t.dims, t.ggml_type, t.offset
        );
    }
    ExitCode::SUCCESS
}
