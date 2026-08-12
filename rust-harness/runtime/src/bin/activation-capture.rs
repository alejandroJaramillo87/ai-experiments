/*
 * activation-capture — write last-token residual activations in the interp
 * capture shard format.
 *
 * Produces the same artifact directory as interp/probes/capture.py (shards
 * of layer_<L> float32 rows plus labels, examples.jsonl, manifest.json), so
 * the Python probe toolkit (train, registry, evaluate) consumes rust-served
 * activations unchanged. Rows are the residual stream after each requested
 * decoder layer at the final prompt token, via the CpuModel Tap seam —
 * captured from the quantized serving path itself, not a separate
 * full-precision workbench.
 */

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{bail, ensure, Context, Result};
use runtime::digest::sha256_hex;

use runtime::model::cpu::{CpuModel, LastRowTap};
use runtime::model::Model;
use runtime::npz::{npy_f32, npy_i64, NpzWriter};
use runtime::tokenizer::Tokenizer;

/// Shard size fixed by interp/probes/capture.py (examples per npz file).
const SHARD_SIZE: usize = 2048;

struct Example {
    text: String,
    label: i64,
}

fn load_dataset(path: &Path) -> Result<Vec<Example>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read dataset {}", path.display()))?;
    let mut examples = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).with_context(|| format!("dataset line {}", i + 1))?;
        let text = v
            .get("text")
            .and_then(|t| t.as_str())
            .with_context(|| format!("dataset line {}: missing text", i + 1))?
            .to_string();
        let label = v
            .get("label")
            .and_then(serde_json::Value::as_i64)
            .with_context(|| format!("dataset line {}: missing integer label", i + 1))?;
        examples.push(Example { text, label });
    }
    ensure!(!examples.is_empty(), "dataset holds no examples");
    Ok(examples)
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn utc_now() -> String {
    // Seconds since the epoch rendered as UTC without a time library.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    let (h, m, s) = (secs % 86_400 / 3600, secs % 3600 / 60, secs % 60);
    // Civil-date conversion (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

/// The shard slicing assumes exactly one captured row per example per layer.
/// Checking that before anything is written keeps a future divergence between
/// the per-layer keying of `collected` and the capture loop's iteration from
/// producing an internally consistent artifact whose rows sit offset from
/// their labels — a failure no downstream reader can detect.
fn check_row_counts(
    collected: &HashMap<usize, Vec<f32>>,
    layers: &[usize],
    embed_dim: usize,
    examples: usize,
) -> Result<()> {
    for &l in layers {
        let buf = collected
            .get(&l)
            .with_context(|| format!("layer {l}: no captured rows"))?;
        ensure!(
            buf.len() == examples * embed_dim,
            "layer {l}: captured {} rows for {examples} examples",
            buf.len() / embed_dim
        );
    }
    Ok(())
}

fn run(
    model_path: &Path,
    dataset_path: &Path,
    layers: &[usize],
    out_dir: &Path,
    text_only: bool,
    single_thread: bool,
) -> Result<()> {
    // `load` drops the checkpoint mapping before returning; see its doc
    // comment for why no caller here should open one itself.
    let (mut model, _fingerprint) = CpuModel::load(model_path, text_only)?;
    // The manifest records the identity of the weights these activations were
    // read from, and the harness refuses a probe whose recorded hash does not
    // match what it serves -- so this digest and `harness`'s must stay the
    // same function over the same bytes. Both now stream the weights file
    // through `digest::sha256_file` rather than hashing a mapping, which is
    // what lets the mapping be dropped above.
    let weights_path =
        runtime::checkpoint::resolve(model_path)?.join(runtime::checkpoint::WEIGHTS_FILE);
    if single_thread {
        model.set_threads(1);
    }
    let config = *model.config();
    for &l in layers {
        ensure!(
            l < config.n_layers,
            "layer {l} out of range: model has {} layers",
            config.n_layers
        );
    }
    let tokenizer_path = runtime::tokenizer::resolve_tokenizer(model_path, None)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).context("load tokenizer")?;
    let examples = load_dataset(dataset_path)?;

    let tap = LastRowTap::new(layers.to_vec());
    model.set_tap(Box::new(tap.clone()));

    // One f32 row per example per layer, captured at the last prompt token
    // (position semantics match the Python capture's `position: "last"`).
    let mut collected: HashMap<usize, Vec<f32>> = layers.iter().map(|&l| (l, Vec::new())).collect();
    for (i, example) in examples.iter().enumerate() {
        model.truncate(0).context("reset KV cache")?;
        tap.clear();
        let mut tokens = vec![tokenizer.bos()];
        tokens.extend(tokenizer.encode(&example.text)?);
        ensure!(
            tokens.len() <= config.context_length,
            "example {}: {} tokens exceed the context length {}",
            i + 1,
            tokens.len(),
            config.context_length
        );
        model.prefill(&tokens)?;
        for &l in layers {
            let row = tap
                .row(l)
                .with_context(|| format!("example {}: no tap row for layer {l}", i + 1))?;
            collected
                .get_mut(&l)
                .expect("initialized above")
                .extend(row.iter().copied());
        }
        eprintln!(
            "[RUN] Example {}/{}: {} tokens",
            i + 1,
            examples.len(),
            tokens.len()
        );
    }
    check_row_counts(&collected, layers, config.embed_dim, examples.len())?;

    std::fs::create_dir_all(out_dir)?;
    // Mirror the Python writer's crash discipline: no manifest until every
    // shard is on disk, so a torn run cannot present as a valid capture.
    let stale = out_dir.join("manifest.json");
    if stale.is_file() {
        std::fs::remove_file(&stale)?;
    }
    let mut ex_file = std::fs::File::create(out_dir.join("examples.jsonl"))?;
    for e in &examples {
        writeln!(
            ex_file,
            "{}",
            serde_json::json!({ "text": e.text, "label": e.label })
        )?;
    }
    let mut shards = 0usize;
    for start in (0..examples.len()).step_by(SHARD_SIZE) {
        let end = (start + SHARD_SIZE).min(examples.len());
        let mut w = NpzWriter::new();
        for &l in layers {
            let rows = &collected[&l][start * config.embed_dim..end * config.embed_dim];
            w.add(
                &format!("layer_{l}"),
                npy_f32(end - start, config.embed_dim, rows),
            );
        }
        let labels: Vec<i64> = examples[start..end].iter().map(|e| e.label).collect();
        w.add("labels", npy_i64(&labels));
        w.write_to(&out_dir.join(format!("shard_{shards:05}.npz")))?;
        shards += 1;
    }

    // Fingerprint over what the rust stack actually used: the model file
    // name, vocabulary size, and no chat template (raw-text capture). Not
    // comparable to HF-derived fingerprints and not meant to be — it keeps
    // rust-captured artifacts internally consistent across capture, train,
    // and score.
    let model_name = model_path.display().to_string();
    let fingerprint = sha256_hex(format!("{model_name}|{}|", tokenizer.vocab_len()).as_bytes());
    let dataset_bytes = std::fs::read(dataset_path)?;
    let mut sorted_layers = layers.to_vec();
    sorted_layers.sort_unstable();
    let manifest = serde_json::json!({
        "schema_version": 1,
        "artifact": "activation_capture",
        "model_id": model_name,
        // The container the weights were read from, not a revision -- the
        // field name is the interp capture-shard schema's and predates this
        // backend. It said "gguf" while a GGUF was loaded and must not keep
        // saying so now that a checkpoint is, because the interp toolkit reads
        // this to know what produced a shard.
        "model_revision": "safetensors",
        "model_hash": runtime::digest::sha256_file(&weights_path)
            .with_context(|| format!("hash {}", weights_path.display()))?,
        "layers": sorted_layers,
        // Which point of each layer the rows came from. "layers" alone names
        // half of where an activation was taken, and a shard is readable
        // today only because there is exactly one site -- a property of this
        // moment, not of the format. From Site's own Display, so the stored
        // text cannot drift from the value a reader parses back.
        "site": LastRowTap::SITE.to_string(),
        "position": "last",
        "dataset_path": dataset_path.display().to_string(),
        "dataset_sha256": sha256_hex(&dataset_bytes),
        "tokenizer_fingerprint": fingerprint,
        "dtype": "float32",
        "count": examples.len(),
        "shards": shards,
        "capture_backend": "rust-harness-cpu",
        "git_sha": git_sha(),
        "created": utc_now(),
    });
    let tmp = out_dir.join("manifest.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&manifest)?)?;
    std::fs::rename(&tmp, out_dir.join("manifest.json"))?;
    eprintln!(
        "[OK] Capture: {} examples, {} layers, {} shards -> {}",
        examples.len(),
        layers.len(),
        shards,
        out_dir.display()
    );
    Ok(())
}

/// Parses `--layers`, rejecting a repeated layer instead of deduplicating it.
/// The capture loop appends one row per listed layer into buffers keyed by
/// layer, so a repeat stacks several rows per example into a single buffer and
/// the shard slicing then reads a window shifted against the labels. A repeat
/// in a generated layer list is a caller bug; a silent dedup would hide it.
fn parse_layers(arg: &str) -> Result<Vec<usize>> {
    let layers = arg
        .split(',')
        .map(|s| s.trim().parse::<usize>().context("bad --layers entry"))
        .collect::<Result<Vec<usize>>>()?;
    ensure!(!layers.is_empty(), "no layers requested");
    for (i, &l) in layers.iter().enumerate() {
        ensure!(!layers[..i].contains(&l), "duplicate layer {l} in --layers");
    }
    Ok(layers)
}

fn main() -> ExitCode {
    const USAGE: &str = "usage: activation-capture --model <checkpoint-dir> \
         --dataset <examples.jsonl> --layers <l0,l1,...> --out <dir> [--text-only] [--single-thread]";
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let parsed: Result<()> = (|| {
        let model = get("--model").map(PathBuf::from);
        let dataset = get("--dataset").map(PathBuf::from);
        let out = get("--out").map(PathBuf::from);
        let layers_arg = get("--layers");
        let (Some(model), Some(dataset), Some(out), Some(layers_arg)) =
            (model, dataset, out, layers_arg)
        else {
            bail!("missing required flag");
        };
        let layers = parse_layers(&layers_arg)?;
        // Bare presence flag, unlike the four above, which take a value.
        let text_only = args.iter().any(|a| a == "--text-only");
        let single_thread = args.iter().any(|a| a == "--single-thread");
        run(&model, &dataset, &layers, &out, text_only, single_thread)
    })();
    match parsed {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[ERROR] Capture failed: {e:#}");
            eprintln!("{USAGE}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_layers_rejects_a_repeated_layer_and_names_it() {
        let err = parse_layers("1,1").unwrap_err().to_string();
        assert!(err.contains("duplicate layer 1"), "{err}");
        let err = parse_layers("3, 5, 3").unwrap_err().to_string();
        assert!(err.contains("duplicate layer 3"), "{err}");
    }

    #[test]
    fn parse_layers_keeps_distinct_layers_in_the_order_given() {
        assert_eq!(parse_layers("5, 1,3").unwrap(), vec![5, 1, 3]);
    }

    #[test]
    fn check_row_counts_accepts_one_row_per_example_per_layer() {
        let collected = HashMap::from([(1, vec![0.0f32; 3 * 4]), (2, vec![0.0f32; 3 * 4])]);
        check_row_counts(&collected, &[1, 2], 4, 3).unwrap();
    }

    #[test]
    fn check_row_counts_rejects_a_layer_holding_two_rows_per_example() {
        // The buffer shape a duplicate --layers entry produced: one layer fed
        // twice per example, then sliced as if it held a single row each.
        let collected = HashMap::from([(1, vec![0.0f32; 2 * 3 * 4])]);
        let err = check_row_counts(&collected, &[1], 4, 3)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("layer 1: captured 6 rows for 3 examples"),
            "{err}"
        );
    }

    #[test]
    fn utc_now_renders_a_plausible_iso_timestamp() {
        let t = utc_now();
        assert_eq!(t.len(), 20);
        assert!(t.ends_with('Z') && t.contains('T'), "{t}");
        assert!(t.starts_with("20"), "{t}");
    }
}
