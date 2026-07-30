/*
 * gguf-info — dump GGUF metadata and tensor directory.
 *
 * Inspection tool for resolving model-architecture questions against a real
 * checkpoint: run against the E4B GGUF on the workstation and report the
 * output verbatim. Long arrays (tokenizer vocab) are summarized by length.
 *
 * The raw dump runs to several hundred lines for a 42-layer model, so a third
 * section derives the four answers runbook entry RH.1 needs — tensor
 * families, feed-forward widths, KV-bearing layers, quantization types — and
 * ends with the CPU backend's own compatibility verdict. The verdict calls
 * `cpu::check_supported`, the function the loader calls, so this tool and the
 * loader cannot disagree.
 */

use std::collections::BTreeMap;
use std::process::ExitCode;

use runtime::gguf::{self, GgufFile, MetaValue};
use runtime::model::cpu::{self, Config};

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

    println!();
    summarize(&file);
    ExitCode::SUCCESS
}

/// Derived summary: the architecture questions RH.1 has to answer, plus the
/// backend's verdict. Never fails the run — an unloadable checkpoint is the
/// finding, not an error.
fn summarize(file: &GgufFile) {
    println!("=== Architecture summary ===");

    let mut families: BTreeMap<String, usize> = BTreeMap::new();
    for t in &file.tensors {
        *families.entry(cpu::tensor_family(&t.name)).or_default() += 1;
    }
    println!("Tensor families:");
    for (family, count) in &families {
        println!("  {family}  x{count}");
    }

    let mut types: BTreeMap<u32, usize> = BTreeMap::new();
    for t in &file.tensors {
        *types.entry(t.ggml_type).or_default() += 1;
    }
    println!("Quantization types (implemented: 0 F32, 1 F16, 8 Q8_0):");
    for (ggml_type, count) in &types {
        let known = matches!(ggml_type, 0 | 1 | 8);
        println!(
            "  type {ggml_type}  x{count}{}",
            if known { "" } else { "  UNIMPLEMENTED" }
        );
    }

    // Feed-forward widths: one line when uniform, a table when MatFormer
    // slicing varies them per layer.
    let widths: Vec<(String, u64)> = file
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("blk.") && t.name.ends_with(".ffn_up.weight"))
        .map(|t| (t.name.clone(), t.dims.last().copied().unwrap_or(0)))
        .collect();
    match widths.first() {
        None => println!("Feed-forward widths: no ffn_up tensors found"),
        Some((_, first)) if widths.iter().all(|(_, w)| w == first) => {
            println!(
                "Feed-forward widths: uniform {first} across {} layers",
                widths.len()
            );
        }
        Some(_) => {
            println!("Feed-forward widths: NOT uniform (MatFormer per-layer slicing):");
            for (name, w) in &widths {
                println!("  {name}  {w}");
            }
        }
    }

    // Layers without their own K/V projections reuse another layer's cache.
    let block_count = match file
        .metadata
        .iter()
        .find(|(k, _)| k.ends_with(".block_count"))
    {
        Some((_, MetaValue::U32(v))) => *v as usize,
        _ => 0,
    };
    let names: Vec<&str> = file.tensors.iter().map(|t| t.name.as_str()).collect();
    let shared: Vec<usize> = (0..block_count)
        .filter(|i| {
            let has = |s: &str| names.contains(&format!("blk.{i}.{s}.weight").as_str());
            has("attn_q") && !(has("attn_k") && has("attn_v"))
        })
        .collect();
    if shared.is_empty() {
        println!("Shared KV cache: none, all {block_count} layers carry attn_k/attn_v");
    } else {
        println!(
            "Shared KV cache: {} of {block_count} layers lack attn_k/attn_v: {shared:?}",
            shared.len()
        );
    }

    println!("RoPE, window, and per-layer metadata keys:");
    let interesting = |k: &str| {
        ["rope", "window", "per_layer", "shared", "matformer"]
            .iter()
            .any(|needle| k.contains(needle))
    };
    let mut any = false;
    for (key, value) in file.metadata.iter().filter(|(k, _)| interesting(k)) {
        println!("  {key} = {value:?}");
        any = true;
    }
    if !any {
        println!("  (none)");
    }

    println!();
    match Config::from_gguf(file).and_then(|c| cpu::check_supported(file, &c)) {
        Ok(()) => println!("[OK] CPU backend compatibility: supported"),
        Err(e) => println!("[ERROR] CPU backend compatibility: {e:#}"),
    }
}
