/*
 * checkpoint-info — dump a Hugging Face checkpoint's architecture and tensors.
 *
 * Inspection tool for resolving model-architecture questions against a real
 * checkpoint: run against the E4B checkpoint on the workstation and report the
 * output verbatim.
 *
 * Three sections. The config echo and the tensor table are what the files say;
 * the architecture summary derives the answers runbook entry RH.1 needs --
 * tensor families, feed-forward widths, KV-bearing layers -- and ends with the
 * CPU backend's own verdict. The verdict calls `check_supported_checkpoint`
 * through `ModelSpec::from_checkpoint`, the same path the loader takes, so this
 * tool and the loader cannot disagree about a file.
 *
 * This replaces the GGUF inspector. The sections it lost went with the
 * container: a GGUF states its architecture in metadata keys that have to be
 * hunted for by suffix, and its dtypes as ggml type ids that a histogram had
 * to name. A
 * checkpoint states both directly, so the tool prints them instead of
 * reconstructing them.
 */

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

use runtime::checkpoint::{self, Checkpoint};
use runtime::cli::{Flag, Kind, Spec};
use runtime::model::cpu;

const SPEC: Spec = Spec {
    program: "checkpoint-info",
    summary: "Dump a checkpoint's config and tensor table, then the architecture summary\n\
              and the CPU backend's compatibility verdict.",
    positionals: "<checkpoint-dir>",
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
        Flag {
            name: "--check",
            kind: Kind::Bare,
            help: "exit 1 when the checkpoint is unsupported, rather than reporting and exiting 0",
        },
        Flag {
            name: "--text-only",
            kind: Kind::Bare,
            help: "treat a multimodal checkpoint as its text tower",
        },
    ],
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if runtime::cli::handled_help(&SPEC, &args) {
        return ExitCode::SUCCESS;
    }
    // The `partition` this replaces sorted every unrecognized token into the
    // positional pile, so a misspelled flag surfaced as the wrong argument
    // count and the message named neither it nor the count -- and a lone `-h`
    // was read as the checkpoint path.
    let parsed = match runtime::cli::extract(&SPEC, args, &mut []) {
        Ok(p) => p,
        Err(e) => return runtime::cli::usage_error(&SPEC, &e),
    };
    let [path] = parsed.rest.as_slice() else {
        return runtime::cli::usage_error(
            &SPEC,
            &format!("one checkpoint directory, {} given", parsed.rest.len()),
        );
    };
    let check = parsed.has("--check");
    // Reported either way (see `describe`); this decides only whether the
    // backend's verdict treats a multimodal checkpoint as supported.
    let text_only = parsed.has("--text-only");

    let files = match checkpoint::CheckpointFiles::read(Path::new(path)) {
        Ok(f) => f,
        Err(e) => {
            runtime::status::error(&format!("Read {path}: {e:#}"));
            return ExitCode::from(1);
        }
    };
    // A config the backend refuses still parses far enough to describe, and
    // describing it is the point of the tool -- but `parse` validates, so a
    // refusal surfaces here rather than in the summary. Reported as the
    // finding it is, and turned into an exit code only by `--check`.
    let mut ck = match checkpoint::parse(&files) {
        Ok(c) => c,
        Err(e) => {
            runtime::status::error(&format!("CPU backend compatibility: {e:#}"));
            return match check {
                true => ExitCode::from(1),
                false => ExitCode::SUCCESS,
            };
        }
    };

    ck.text_only = text_only;

    print_config(&files);
    print_tensors(&ck);
    print_non_text(&ck);

    println!();
    let supported = summarize(&ck);
    if check && !supported {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// The `config.json` as written. Echoed rather than reformatted from
/// `TextConfig`, so a field this backend does not read still appears -- which
/// is exactly the field a reader investigating an unsupported model needs.
fn print_config(files: &checkpoint::CheckpointFiles) {
    println!("=== config.json ===");
    match std::str::from_utf8(&files.config) {
        Ok(text) => println!("{}", text.trim_end()),
        Err(e) => println!("(not valid utf-8: {e})"),
    }
}

/// Tensor families outside the text stack, collapsed to one line each.
///
/// Printed whether or not `--text-only` was passed, because inspecting a file
/// should not require agreeing to load it: a reader asking what a checkpoint
/// contains wants the towers named either way. Silent on a text-only
/// checkpoint, which is most of them.
fn print_non_text(ck: &Checkpoint) {
    let families = cpu::non_text_families(ck);
    if families.is_empty() {
        return;
    }
    println!();
    println!("=== Non-text towers ({}) ===", families.len());
    for family in &families {
        println!("{family}");
    }
    println!("This backend serves text only; --text-only declines these and loads the rest.");
}

/// One line per tensor: name, dtype, shape. Shapes are row-major `[out, in]`.
fn print_tensors(ck: &Checkpoint) {
    let names = ck.names();
    println!();
    println!("=== Tensors ({}) ===", names.len());
    for name in &names {
        match (ck.dtype(name), ck.shape(name)) {
            (Ok(dtype), Ok(shape)) => println!("{name}  {dtype:?}  {shape:?}"),
            (Err(e), _) | (_, Err(e)) => println!("{name}  (unreadable: {e:#})"),
        }
    }
}

/// Derived summary: the architecture questions RH.1 has to answer, plus the
/// backend's verdict. Never fails the run -- an unloadable checkpoint is the
/// finding, not an error. Returns whether the CPU backend accepts it, which
/// only `--check` turns into an exit code.
fn summarize(ck: &Checkpoint) -> bool {
    println!("=== Architecture summary ===");

    let mut families: BTreeMap<String, usize> = BTreeMap::new();
    for name in ck.names() {
        *families.entry(cpu::hf_tensor_family(&name)).or_default() += 1;
    }
    println!("Tensor families:");
    for (family, count) in &families {
        println!("  {family}  x{count}");
    }

    let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
    for name in ck.names() {
        if let Ok(dtype) = ck.dtype(&name) {
            *dtypes.entry(format!("{dtype:?}")).or_default() += 1;
        }
    }
    println!("Dtypes (this backend loads F32, F16 and BF16):");
    for (dtype, count) in &dtypes {
        println!("  {dtype}  x{count}");
    }

    // Feed-forward widths: one line when uniform, a table when they vary per
    // layer, as E2B's double-wide trailing region does. Read from the config
    // rather than the tensors, because the config is what the forward pass
    // slices by and a disagreement is what `check_supported_checkpoint` is for.
    match ck.config.intermediate_sizes() {
        Err(e) => println!("Feed-forward widths: unreadable: {e:#}"),
        Ok(widths) => match widths.first() {
            None => println!("Feed-forward widths: none declared"),
            Some(first) if widths.iter().all(|w| w == first) => {
                println!(
                    "Feed-forward widths: uniform {first} across {} layers",
                    widths.len()
                );
            }
            Some(_) => {
                println!("Feed-forward widths: NOT uniform (per-layer widths):");
                for (i, w) in widths.iter().enumerate() {
                    println!("  layer {i}  {w}");
                }
            }
        },
    }

    // Cross-layer KV sharing is a declared count, not something inferred from
    // which tensors are present. The published E2B and E4B files carry every
    // projection for every layer and still declare 20 and 18 shared, so a
    // tensor-presence view reads as "no sharing here", the opposite of what
    // those files say.
    let n_layers = ck.config.num_hidden_layers;
    match ck.config.num_kv_shared_layers {
        0 => println!("Shared KV cache: num_kv_shared_layers is 0, every layer owns a cache"),
        n => println!(
            "Shared KV cache: num_kv_shared_layers is {n} of {n_layers}, \
             so layers {}-{} read a donor's",
            n_layers - n,
            n_layers - 1
        ),
    }

    println!(
        "Attention schedule: {}",
        match (0..n_layers)
            .map(|i| ck.config.is_global(i))
            .collect::<anyhow::Result<Vec<bool>>>()
        {
            Ok(kinds) => kinds
                .iter()
                .map(|g| match g {
                    true => "G",
                    false => "s",
                })
                .collect::<String>(),
            Err(e) => format!("unreadable: {e:#}"),
        }
    );
    println!("  (G = full attention, s = sliding window)");

    println!();
    match cpu::ModelSpec::from_checkpoint(&ck.config).and_then(|s| {
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
            let span = match start == end {
                true => format!("layer {start}"),
                false => format!("layers {start}-{end}"),
            };
            println!(
                "  {span}: {:?}, head {}, ffn {}, rope base {}",
                l.kind, l.head_dim, l.ffn_dim, l.rope_freq_base
            );
        }
        println!();
        cpu::check_supported_checkpoint(ck, &s)
    }) {
        Ok(()) => {
            runtime::status::ok("CPU backend compatibility: supported");
            true
        }
        Err(e) => {
            runtime::status::error(&format!("CPU backend compatibility: {e:#}"));
            false
        }
    }
}
