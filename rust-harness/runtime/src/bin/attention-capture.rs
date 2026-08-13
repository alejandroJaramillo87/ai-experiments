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

use anyhow::{bail, ensure, Context, Result};

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

fn run(
    model_path: &Path,
    prompt: &str,
    layers: &[usize],
    heads: &[usize],
    out_dir: &Path,
    text_only: bool,
    single_thread: bool,
) -> Result<()> {
    let (mut model, _fingerprint) = CpuModel::load(model_path, text_only)?;
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
    eprintln!(
        "[INFO] Attention: {n} tokens, {} layers x {} heads = {pairs} matrices, {:.1} MiB",
        layers.len(),
        heads.len(),
        bytes as f64 / (1024.0 * 1024.0)
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
    eprintln!(
        "[OK] Attention: {pairs} matrices of {n}x{n} -> {}",
        out_dir.display()
    );
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

fn main() -> ExitCode {
    const USAGE: &str = "usage: attention-capture --model <checkpoint-dir> \
         --prompt <text> --layers <l0,l1,...> --heads <h0,h1,...> --out <dir> \
         [--text-only] [--single-thread]";
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let parsed: Result<()> = (|| {
        let (Some(model), Some(prompt), Some(out), Some(layers_arg), Some(heads_arg)) = (
            get("--model").map(PathBuf::from),
            get("--prompt"),
            get("--out").map(PathBuf::from),
            get("--layers"),
            get("--heads"),
        ) else {
            bail!("missing required flag");
        };
        let layers = parse_indices(&layers_arg, "layers")?;
        let heads = parse_indices(&heads_arg, "heads")?;
        let text_only = args.iter().any(|a| a == "--text-only");
        let single_thread = args.iter().any(|a| a == "--single-thread");
        run(
            &model,
            &prompt,
            &layers,
            &heads,
            &out,
            text_only,
            single_thread,
        )
    })();
    match parsed {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[ERROR] Attention capture failed: {e:#}");
            eprintln!("{USAGE}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
