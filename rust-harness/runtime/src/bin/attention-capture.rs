/*
 * attention-capture — write post-softmax attention weights for chosen layers
 * and query heads over one prompt.
 *
 * The reason this binary exists rather than a flag on activation-capture:
 * that tool writes probe training data, one residual row per example at the
 * last prompt token, and its shard format is the interp toolkit's. This writes
 * every position of one prompt, because what attention weights are useful for
 * is watching the span change along a sequence — Gemma 4 interleaves sliding
 * and global layers, and the contrast between them only exists across
 * positions. Different question, different artifact.
 *
 * Volume is quadratic in the context, so both filters are required flags.
 */

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{ensure, Context, Result};
use runtime::cli::{Flag, Kind, Spec};

use runtime::model::cpu::{AttentionRow, AttentionTap, CpuModel};
use runtime::model::Model;
use runtime::npz::{npy_f32, npy_i64, NpzWriter};
use runtime::tokenizer::Tokenizer;

/// Bytes per f32, for the volume the binary reports before it allocates.
const F32_BYTES: usize = 4;

/// One (layer, head)'s captured rows: the query position and its row.
type Rows<'a> = Vec<(usize, &'a AttentionRow)>;

/// A dense `[n, n]` weight matrix for one (layer, head), row-major by query
/// position, with structural zeros outside each row's visible span.
///
/// Dense rather than ragged because the spans differ per row and per layer,
/// and a ragged encoding would need its own offset table in the npz — where
/// zero is already the mathematically correct weight for a masked key, and
/// `np.load` hands the reader a plain 2-D array. The cost is the upper
/// triangle plus, on a sliding layer, everything left of the window; the
/// `span_first` array below is what tells a reader which zeros are structural
/// rather than merely small.
fn dense_matrix(n: usize, rows: &[(usize, &AttentionRow)]) -> Result<Vec<f32>> {
    let mut m = vec![0.0f32; n * n];
    for (position, (first, w)) in rows {
        ensure!(
            *position < n,
            "position {position} outside the prompt of {n} tokens"
        );
        ensure!(
            first + w.len() == position + 1,
            "position {position}: {} weights over span starting at {first}",
            w.len()
        );
        m[position * n + first..position * n + first + w.len()].copy_from_slice(w);
    }
    Ok(m)
}

/// The resolved configuration of an attention capture, as obs fields.
///
/// A per-binary header event, for the reason `schemas/obs_record.json`'s
/// `per_tool_header_events` note gives and `activation-capture.rs::capture_header`
/// restates: each header's field list is asserted for key-set equality in both
/// directions, and a shared event would have to move every difference to
/// `optional`. `model_hash` is the u64 load fingerprint rather than the
/// manifest's sha256, for the same reason.
///
/// **The prompt is recorded as a digest, not as text.** `argv` on `cmd.start`
/// already carries the literal prompt, so a second copy in a header field
/// doubles a potentially long string in the event file for no new fact. The
/// digest is what makes two runs comparable without storing the text twice --
/// and it is more than the manifest carries, which records the token count and
/// never what was captured over.
fn attention_header(
    model_path: &Path,
    model_hash: u64,
    layers: &[usize],
    heads: &[usize],
    prompt: &str,
    prompt_tokens: usize,
    out_dir: &Path,
    text_only: bool,
    model: &CpuModel,
) -> Vec<(&'static str, obs::Value)> {
    let (weight_bytes, width) = model.weight_bytes();
    let list = |v: &[usize]| {
        v.iter()
            .map(usize::to_string)
            .collect::<Vec<String>>()
            .join(",")
    };
    vec![
        ("model", obs::Value::Str(model_path.display().to_string())),
        ("model_hash", obs::Value::Str(format!("{model_hash:016x}"))),
        ("n_layers", obs::Value::Int(model.config().n_layers as u64)),
        (
            "dtype",
            obs::Value::Str(width.unwrap_or("mixed widths").to_string()),
        ),
        ("weight_bytes", obs::Value::Int(weight_bytes as u64)),
        // In the order asked for, not the manifest's sorted spelling.
        ("layers", obs::Value::Str(list(layers))),
        ("heads", obs::Value::Str(list(heads))),
        (
            "prompt_sha256",
            obs::Value::Str(runtime::digest::sha256_hex(prompt.as_bytes())),
        ),
        ("prompt_tokens", obs::Value::Int(prompt_tokens as u64)),
        ("out", obs::Value::Str(out_dir.display().to_string())),
        ("text_only", obs::Value::Bool(text_only)),
        ("threads", obs::Value::Int(model.threads() as u64)),
        ("dot", obs::Value::Str(model.dot().label().to_string())),
        ("batch", obs::Value::Str(model.batch().label())),
        (
            "binary_version",
            obs::Value::Str(env!("CARGO_PKG_VERSION").to_string()),
        ),
        ("git_sha", obs::Value::Str(runtime::envelope::git_sha())),
    ]
}

fn run(
    model_path: &Path,
    prompt: &str,
    layers: &[usize],
    heads: &[usize],
    out_dir: &Path,
    text_only: bool,
    single_thread: bool,
    log: &mut obs::RunLog,
) -> Result<()> {
    let (mut model, fingerprint) = {
        let _span = log.phase("model.load");
        CpuModel::load(model_path, text_only)?
    };
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
    for &h in heads {
        ensure!(
            h < config.n_heads,
            "head {h} out of range: model has {} query heads",
            config.n_heads
        );
    }

    let tokenizer_path = runtime::tokenizer::resolve_tokenizer(model_path, None)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).context("load tokenizer")?;
    let mut tokens = vec![tokenizer.bos()];
    tokens.extend(tokenizer.encode(prompt)?);
    let n = tokens.len();
    ensure!(
        n <= config.context_length,
        "prompt of {n} tokens exceeds the context length {}",
        config.context_length
    );

    // Stated before the allocation rather than discovered as an OOM: the
    // quadratic term is the whole cost discipline of this site.
    let pairs = layers.len() * heads.len();
    let bytes = pairs * n * n * F32_BYTES;
    runtime::status::info(&format!(
        "Attention: {n} tokens, {} layers x {} heads = {pairs} matrices, {:.1} MiB",
        layers.len(),
        heads.len(),
        bytes as f64 / (1024.0 * 1024.0)
    ));

    // After `set_threads` and after the prompt is tokenized, so the header
    // reports the width the capture actually ran at and the token count it
    // actually captured over -- and still before the one `prefill` below, which
    // is the operation the record is worth having about.
    log.event(
        "info",
        "attention-capture.header",
        "resolved attention capture configuration",
        &attention_header(
            model_path, fingerprint, layers, heads, prompt, n, out_dir, text_only, &model,
        ),
    );

    let tap = AttentionTap::new(layers.to_vec(), heads.to_vec());
    model.set_tap(Box::new(tap.clone()));
    model.prefill(&tokens)?;

    // Group by (layer, head). `rows()` is sorted, so the grouping and every
    // array written below is a function of the activations alone and not of a
    // hash order that varies between runs.
    let mut grouped: BTreeMap<(usize, usize), Rows> = BTreeMap::new();
    let captured = tap.rows();
    for ((l, h, p), row) in &captured {
        grouped.entry((*l, *h)).or_default().push((*p, row));
    }

    // Checked before anything is written, so a partial artifact is never
    // left behind: a directory holding some of the requested pairs reads as a
    // complete capture of a smaller request.
    ensure!(
        grouped.len() == pairs,
        "captured {} of {pairs} requested (layer, head) pairs",
        grouped.len()
    );
    for (&(l, h), rows) in &grouped {
        ensure!(
            rows.len() == n,
            "layer {l} head {h}: {} rows captured for {n} positions",
            rows.len()
        );
    }

    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let mut npz = NpzWriter::new();
    for (&(l, h), rows) in &grouped {
        let m = dense_matrix(n, rows).with_context(|| format!("layer {l} head {h}"))?;
        npz.add(&format!("layer_{l}_head_{h}"), npy_f32(n, n, &m));
    }

    // The first visible key position per (layer, position), which is what
    // separates a structural zero from a small weight. Written per layer
    // because the span is a property of the layer's kind and not of the head,
    // and written at all because a reader that recomputed it would be
    // deriving the attention schedule from a stride constant -- the mistake
    // docs/models/gemma-4.md warns against by name.
    //
    // Taken from the first requested head, and cross-checked against the
    // others: heads of one layer share a mask, so a disagreement is not a
    // richer artifact but a broken assumption, and the file cannot express it.
    for &l in layers {
        let reference = &grouped[&(l, heads[0])];
        for &h in &heads[1..] {
            for ((p, (a, _)), (_, (b, _))) in reference.iter().zip(&grouped[&(l, h)]) {
                ensure!(
                    a == b,
                    "layer {l} position {p}: head {} sees from {a}, head {h} from {b}",
                    heads[0]
                );
            }
        }
        let first: Vec<i64> = reference.iter().map(|(_, (f, _))| *f as i64).collect();
        npz.add(&format!("span_first_layer_{l}"), npy_i64(&first));
    }
    npz.add(
        "tokens",
        npy_i64(&tokens.iter().map(|&t| i64::from(t)).collect::<Vec<_>>()),
    );
    npz.write_to(&out_dir.join("attention.npz"))?;

    let weights_path =
        runtime::checkpoint::resolve(model_path)?.join(runtime::checkpoint::WEIGHTS_FILE);
    let mut sorted_layers = layers.to_vec();
    sorted_layers.sort_unstable();
    let mut sorted_heads = heads.to_vec();
    sorted_heads.sort_unstable();
    let manifest = serde_json::json!({
        "schema_version": 1,
        "artifact": "attention_capture",
        "model_id": model_path.display().to_string(),
        "model_hash": runtime::digest::sha256_file(&weights_path)
            .with_context(|| format!("hash {}", weights_path.display()))?,
        "layers": sorted_layers,
        "heads": sorted_heads,
        // The tap site, spelled the way the seam spells it. Not Site's Display:
        // attention weights are not a residual site and deliberately carry no
        // Site variant, so this string is the artifact's own name for the
        // point and the contract in schemas/ is what pins it.
        "site": "attention_weights",
        "softmax": "post",
        "tokens": n,
        "dtype": "float32",
        "shard": "attention.npz",
        "capture_backend": "rust-harness-cpu",
        "git_sha": runtime::envelope::git_sha(),
        "created": runtime::envelope::utc_now(),
    });
    let tmp = out_dir.join("manifest.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&manifest)?)?;
    std::fs::rename(&tmp, out_dir.join("manifest.json"))?;
    runtime::status::ok(&format!(
        "Attention: {pairs} matrices of {n}x{n} -> {}",
        out_dir.display()
    ));
    Ok(())
}

/// Parses a comma-separated index list, rejecting a repeat instead of
/// deduplicating it.
///
/// A repeat is a caller bug and would otherwise pass silently: `AttentionTap`
/// keys its map by (layer, head, position), so the second copy overwrites the
/// first and the artifact looks correct while the request was not.
fn parse_indices(arg: &str, what: &str) -> Result<Vec<usize>> {
    let idx = arg
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .with_context(|| format!("bad --{what} entry {s:?}"))
        })
        .collect::<Result<Vec<usize>>>()?;
    ensure!(!idx.is_empty(), "no {what} requested");
    for (i, &l) in idx.iter().enumerate() {
        ensure!(!idx[..i].contains(&l), "duplicate {what} entry {l}");
    }
    Ok(idx)
}

const SPEC: Spec = Spec {
    program: "attention-capture",
    summary: "Write post-softmax attention weights for the requested layers and query\n\
              heads over one prompt.",
    positionals: "",
    required: &["--model", "--prompt", "--layers", "--heads", "--out"],
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
            name: "--model",
            kind: Kind::Value("<checkpoint-dir>"),
            help: "the checkpoint to capture from; required",
        },
        Flag {
            name: "--prompt",
            kind: Kind::Value("<text>"),
            help: "the single prompt the attention is captured over; required",
        },
        Flag {
            name: "--layers",
            kind: Kind::Value("<l0,l1,...>"),
            help: "decoder layers to capture, comma separated and distinct; required",
        },
        Flag {
            name: "--heads",
            kind: Kind::Value("<h0,h1,...>"),
            help: "query heads to capture, comma separated and distinct; required",
        },
        Flag {
            name: "--out",
            kind: Kind::Value("<dir>"),
            help: "directory to write the attention shards and manifest into; required",
        },
        Flag {
            name: "--text-only",
            kind: Kind::Bare,
            help: "treat a multimodal checkpoint as its text tower",
        },
        Flag {
            name: "--single-thread",
            kind: Kind::Bare,
            help: "run the row split on one thread; results are bit-identical at every width",
        },
    ],
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if runtime::cli::handled_help(&SPEC, &args) {
        return ExitCode::SUCCESS;
    }
    // Snapshotted before parsing, so the record carries what was typed rather
    // than what survived it.
    let argv_full = args.clone();
    let (mut model, mut prompt, mut layers_arg, mut heads_arg, mut out) =
        (None, None, None, None, None);
    let parsed = {
        let mut slots: [(&'static str, &mut Option<String>); 5] = [
            ("--model", &mut model),
            ("--prompt", &mut prompt),
            ("--layers", &mut layers_arg),
            ("--heads", &mut heads_arg),
            ("--out", &mut out),
        ];
        match runtime::cli::extract(&SPEC, args, &mut slots) {
            Ok(p) => p,
            Err(e) => return runtime::cli::usage_error(&SPEC, &e),
        }
    };
    if let Some(extra) = parsed.rest.first() {
        return runtime::cli::usage_error(&SPEC, &format!("Unexpected argument: {extra}"));
    }
    // A command line that cannot be acted on is exit 2 through `usage_error`; a
    // capture that was asked for correctly and then failed is exit 1 below.
    // This binary used to return 1 for both.
    if let Some(absent) = runtime::cli::missing_required(&SPEC, &parsed) {
        return runtime::cli::usage_error(&SPEC, &absent);
    }
    let expect = |v: Option<String>| v.expect("checked by missing_required above");
    let indices = |arg: String, what: &'static str| match parse_indices(&arg, what) {
        Ok(v) => Ok(v),
        Err(e) => Err(format!("--{what}: {e:#}")),
    };
    let (layers, heads) = match indices(expect(layers_arg), "layers")
        .and_then(|l| indices(expect(heads_arg), "heads").map(|h| (l, h)))
    {
        Ok(pair) => pair,
        Err(e) => return runtime::cli::usage_error(&SPEC, &e),
    };
    // Opened only once the arguments are known good, so a usage mistake leaves
    // no row in the catalog -- and after `handled_help` above, so a query is
    // not a run.
    let mut log = obs::RunLog::begin("attention-capture", &argv_full);
    let outcome = run(
        &PathBuf::from(expect(model)),
        &expect(prompt),
        &layers,
        &heads,
        &PathBuf::from(expect(out)),
        parsed.has("--text-only"),
        parsed.has("--single-thread"),
        &mut log,
    );
    match outcome {
        Ok(()) => {
            log.end(0);
            ExitCode::SUCCESS
        }
        Err(e) => {
            runtime::status::error(&format!("Attention capture failed: {e:#}"));
            log.fail(&e, 1);
            log.end(1);
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flag_table_is_well_formed() {
        SPEC.check().expect("this binary's own flag table");
    }

    #[test]
    fn parse_indices_rejects_a_repeat_and_names_it() {
        let err = parse_indices("1,1", "heads").unwrap_err().to_string();
        assert!(err.contains("duplicate heads entry 1"), "{err}");
        let err = parse_indices("3, 5, 3", "layers").unwrap_err().to_string();
        assert!(err.contains("duplicate layers entry 3"), "{err}");
    }

    #[test]
    fn parse_indices_names_the_entry_it_could_not_read() {
        let err = parse_indices("0,x", "heads").unwrap_err().to_string();
        assert!(err.contains("\"x\""), "{err}");
    }

    #[test]
    fn a_dense_matrix_zero_fills_outside_each_visible_span() {
        // Two positions of a window-2 sliding layer: position 0 sees only
        // itself, position 2 sees keys 1 and 2 and nothing before them.
        let r0 = (0usize, vec![1.0f32]);
        let r2 = (1usize, vec![0.25f32, 0.75]);
        let m = dense_matrix(3, &[(0, &r0), (2, &r2)]).unwrap();
        assert_eq!(
            m,
            vec![
                1.0, 0.0, 0.0, // position 0
                0.0, 0.0, 0.0, // position 1 not captured
                0.0, 0.25, 0.75, // position 2, key 0 outside the window
            ],
            "structural zeros must sit outside the span, got {m:?}"
        );
    }

    #[test]
    fn a_span_disagreeing_with_its_position_is_refused() {
        // Three weights claiming to end at position 1: the pair is internally
        // inconsistent, and writing it would produce a well-formed file whose
        // rows sit shifted against the keys they name.
        let bad = (0usize, vec![0.2f32, 0.3, 0.5]);
        let err = dense_matrix(4, &[(1, &bad)]).unwrap_err().to_string();
        assert!(err.contains("3 weights over span starting at 0"), "{err}");
    }

    #[test]
    fn a_position_past_the_prompt_is_refused() {
        let row = (0usize, vec![1.0f32]);
        let err = dense_matrix(1, &[(5, &row)]).unwrap_err().to_string();
        assert!(err.contains("position 5 outside"), "{err}");
    }
}
