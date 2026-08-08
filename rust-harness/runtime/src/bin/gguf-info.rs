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
use runtime::model::cpu;
use runtime::tensor::{GGML_BF16, GGML_F16, GGML_F32, GGML_Q8_0};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (check, rest): (Vec<&String>, Vec<&String>) =
        args.iter().partition(|a| a.as_str() == "--check");
    let [path] = rest.as_slice() else {
        eprintln!("usage: gguf-info [--check] <model.gguf>");
        return ExitCode::from(2);
    };
    let check = !check.is_empty();
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
            MetaValue::ArrBool(v) => println!("{key} = [bool; {}]", v.len()),
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
    let supported = summarize(&file);
    if check && !supported {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Derived summary: the architecture questions RH.1 has to answer, plus the
/// backend's verdict. Never fails the run — an unloadable checkpoint is the
/// finding, not an error. Returns whether the CPU backend accepts the file,
/// which only `--check` turns into an exit code.
fn summarize(file: &GgufFile) -> bool {
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
    println!(
        "Quantization types (implemented: {GGML_F32} F32, {GGML_F16} F16, \
         {GGML_Q8_0} Q8_0, {GGML_BF16} BF16):"
    );
    for (ggml_type, count) in &types {
        // Named constants, not literals: the loader and this tool must not
        // drift on which types are implemented.
        let known = matches!(*ggml_type, GGML_F32 | GGML_F16 | GGML_Q8_0 | GGML_BF16);
        println!(
            "  type {ggml_type}  x{count}{}",
            if known { "" } else { "  UNIMPLEMENTED" }
        );
    }

    // Feed-forward widths: one line when uniform, a table when they vary per
    // layer, as E2B's double-wide trailing region does.
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
            println!("Feed-forward widths: NOT uniform (per-layer widths):");
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
        // Tensor presence is not evidence of absence: the published E2B and
        // E4B files carry every projection for every layer and still declare
        // shared_kv_layers of 20 and 18. Reporting only the tensor view read
        // as "no KV sharing here", which is the opposite of what those files
        // say, so the declared count is printed alongside it.
        print!("Shared KV cache: no layer lacks attn_k/attn_v");
        match file
            .metadata
            .iter()
            .find(|(k, _)| k.ends_with(".attention.shared_kv_layers"))
        {
            Some((key, MetaValue::U32(n))) => println!(" (but {key} declares {n})"),
            _ => println!(", and no shared_kv_layers key is declared"),
        }
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
    match cpu::ModelSpec::from_gguf(file).and_then(|s| {
        // Print the geometry the loader derived before its verdict, so a
        // refusal is read beside the per-layer widths that produced it.
        println!("Per-layer geometry (head width, feed-forward width, RoPE base):");
        let mut runs: Vec<(cpu::LayerSpec, usize, usize)> = Vec::new();
        for (i, l) in s.layers.iter().enumerate() {
            match runs.last_mut() {
                Some((prev, _, end)) if prev == l => *end = i,
                _ => runs.push((*l, i, i)),
            }
        }
        for (l, start, end) in &runs {
            let span = if start == end {
                format!("layer {start}")
            } else {
                format!("layers {start}-{end}")
            };
            println!(
                "  {span}: {:?}, head {}, ffn {}, rope base {}",
                l.kind, l.head_dim, l.ffn_dim, l.rope_freq_base
            );
        }
        println!();
        cpu::check_supported(file, &s)
    }) {
        Ok(()) => {
            println!("[OK] CPU backend compatibility: supported");
            true
        }
        Err(e) => {
            println!("[ERROR] CPU backend compatibility: {e:#}");
            false
        }
    }
}
