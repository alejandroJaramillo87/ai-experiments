/*
 * runheader.rs — the resolved configuration of a run, built once and
 * rendered twice.
 *
 * Two bugs motivate this file, and they are the same bug. `DotKind`'s default
 * was `Sequential` while `set_dot` had one caller, so every path except the
 * ladder ran the reference fold for months; and RH.12's determinism runs
 * deviated from the documented `--seed 7`, caught only while the entry was
 * being written up. In both cases the tool knew exactly what it was doing and
 * never said so -- not on the terminal, and not in the run record, which
 * carried only `command` and the raw `argv`.
 *
 * The design rests on one property: `lines` derives from `fields`. Two match
 * arms over the same struct would reintroduce precisely the disagreement this
 * file exists to prevent -- a field added to the record and forgotten on the
 * terminal, or the reverse -- so `lines` is built from templates that name the
 * fields they consume, and an inline test asserts every field is named by
 * some template.
 *
 * Numbers stay numbers. "14.23 GiB" cannot be turned back into bytes, and a
 * header whose whole purpose is cross-run comparison is the record someone
 * will `jq` over. The Python writer already emits real numbers for
 * `duration_s` and `exit_code`; stringifying `n_layers` under the same
 * contract would diverge for nothing.
 */

use crate::obs::Value;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The facts about a run that are fixed once the model is loaded.
///
/// Owned, with no borrows: it outlives the scope that builds it (`cpu` is
/// wrapped and moved into the session immediately afterwards) and is read
/// again at the end of the run to stamp the alignment artifact.
pub struct RunHeader {
    /// Checkpoint directory as given on the command line.
    pub model: String,
    /// The u64 checkpoint fingerprint, `{:016x}`.
    ///
    /// Not the sha256 of the weights. `CpuModel::load` computes this one
    /// already and it costs nothing; the digest reads ~15 GiB and is taken
    /// only when an interp artifact has already forced it. Neither motivating
    /// bug needed it.
    pub model_hash: String,
    pub n_layers: u64,
    /// The storage width the weights are held at, or "mixed widths".
    pub dtype: String,
    pub weight_bytes: u64,
    /// Peak RSS sampled immediately after load. `None` where /proc cannot
    /// answer -- an absent measurement must not read as a small one.
    pub peak_rss_bytes: Option<u64>,
    pub max_context: u64,
    pub kv_bytes: u64,
    /// The batch group the KV budget was computed against. The budget is only
    /// true for the batch this run will use, so the two travel together.
    pub kv_batch_group: u64,
    /// Positions still reachable behind the tip once the window fills.
    /// `None` when no layer is windowed.
    pub rewind_horizon: Option<u64>,
    pub threads: u64,
    pub dot: String,
    pub batch: String,
    pub seed: u64,
    pub temperature: f64,
    pub max_tokens: u64,
    pub binary_version: String,
    pub git_sha: String,
    pub probe: Option<String>,
    pub probe_layer: Option<u64>,
}

/// One rendered line and the field names it drew on.
///
/// The names are what makes the fields/lines parity test possible. A line is
/// not one field: the KV line is three fields and a parenthetical in one
/// sentence, and splitting it into three lines to make the mapping trivial
/// would be worse output in service of an easier test.
type Template = (String, &'static [&'static str]);

impl RunHeader {
    /// The header as record fields, in the order they are written.
    pub fn fields(&self) -> Vec<(&'static str, Value)> {
        let mut f = vec![
            ("model", Value::Str(self.model.clone())),
            ("model_hash", Value::Str(self.model_hash.clone())),
            ("n_layers", Value::Int(self.n_layers)),
            ("dtype", Value::Str(self.dtype.clone())),
            ("weight_bytes", Value::Int(self.weight_bytes)),
            ("max_context", Value::Int(self.max_context)),
            ("kv_bytes", Value::Int(self.kv_bytes)),
            ("kv_batch_group", Value::Int(self.kv_batch_group)),
            ("threads", Value::Int(self.threads)),
            ("dot", Value::Str(self.dot.clone())),
            ("batch", Value::Str(self.batch.clone())),
            ("seed", Value::Int(self.seed)),
            ("temperature", Value::Num(self.temperature)),
            ("max_tokens", Value::Int(self.max_tokens)),
            ("binary_version", Value::Str(self.binary_version.clone())),
            ("git_sha", Value::Str(self.git_sha.clone())),
        ];
        // Optional fields are absent rather than null: `peak_rss_bytes` is
        // missing where /proc cannot answer, and the probe pair only exists
        // under --probe. A null would say "measured, and the answer is
        // nothing", which is a different claim.
        if let Some(n) = self.peak_rss_bytes {
            f.push(("peak_rss_bytes", Value::Int(n)));
        }
        if let Some(n) = self.rewind_horizon {
            f.push(("rewind_horizon", Value::Int(n)));
        }
        if let Some(p) = &self.probe {
            f.push(("probe", Value::Str(p.clone())));
        }
        if let Some(n) = self.probe_layer {
            f.push(("probe_layer", Value::Int(n)));
        }
        f
    }

    /// The header as status lines, derived from the same facts as `fields`.
    pub fn lines(&self) -> Vec<String> {
        self.templates().into_iter().map(|(l, _)| l).collect()
    }

    /// The lines paired with the field names each one states.
    ///
    /// Private, and read by exactly two callers: `lines`, which drops the
    /// names, and the parity test, which is the reason they exist.
    fn templates(&self) -> Vec<Template> {
        let mut t: Vec<Template> = vec![
            (
                format!(
                    "Model: {} hash={} layers={}",
                    self.model, self.model_hash, self.n_layers
                ),
                &["model", "model_hash", "n_layers"],
            ),
            (
                format!(
                    "Weights: {:.2} GiB resident at {}",
                    self.weight_bytes as f64 / GIB,
                    self.dtype
                ),
                &["weight_bytes", "dtype"],
            ),
        ];
        if let Some(peak) = self.peak_rss_bytes {
            t.push((
                format!("Peak RSS after load: {:.2} GiB", peak as f64 / GIB),
                &["peak_rss_bytes"],
            ));
        }
        t.push((
            format!(
                "KV cache: {} tokens, {:.2} GiB f32 resident \
                 (sliding layers capped at the window plus a batch of {})",
                self.max_context,
                self.kv_bytes as f64 / GIB,
                self.kv_batch_group
            ),
            &["max_context", "kv_bytes", "kv_batch_group"],
        ));
        if let Some(h) = self.rewind_horizon {
            t.push((
                format!(
                    "Rewind horizon: {h} tokens behind the tip once the \
                     window fills (compaction past it re-prefills)"
                ),
                &["rewind_horizon"],
            ));
        }
        // The fold. This is the line whose absence let DotKind default to
        // Sequential unnoticed: `dot` was reachable through a getter the
        // harness never called.
        t.push((
            format!(
                "Fold: dot={} batch={} threads={}",
                self.dot, self.batch, self.threads
            ),
            &["dot", "batch", "threads"],
        ));
        // And the line whose absence let RH.12 deviate from its documented
        // seed without anyone seeing it.
        t.push((
            format!(
                "Sampling: seed={} temperature={} max_tokens={}",
                self.seed, self.temperature, self.max_tokens
            ),
            &["seed", "temperature", "max_tokens"],
        ));
        if let Some(p) = &self.probe {
            t.push((
                match self.probe_layer {
                    Some(l) => format!("Probe {p}: layer {l}"),
                    None => format!("Probe {p}"),
                },
                &["probe", "probe_layer"],
            ));
        }
        t.push((
            format!("Build: {} git {}", self.binary_version, self.git_sha),
            &["binary_version", "git_sha"],
        ));
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A header with every optional field present, so a test that walks the
    /// field set walks all of it.
    fn full() -> RunHeader {
        RunHeader {
            model: "/models/e4b".to_string(),
            model_hash: "0123456789abcdef".to_string(),
            n_layers: 42,
            dtype: "bf16".to_string(),
            weight_bytes: 15_000_000_000,
            peak_rss_bytes: Some(15_300_000_000),
            max_context: 4096,
            kv_bytes: 1_073_741_824,
            kv_batch_group: 64,
            rewind_horizon: Some(1024),
            threads: 16,
            dot: "widen4".to_string(),
            batch: "64".to_string(),
            seed: 7,
            temperature: 0.0,
            max_tokens: 256,
            binary_version: "0.1.0".to_string(),
            git_sha: "abc1234".to_string(),
            probe: Some("refusal".to_string()),
            probe_layer: Some(20),
        }
    }

    #[test]
    fn every_recorded_field_is_named_by_some_printed_line() {
        // The property the whole file is built around. A field added to
        // `fields` and forgotten in `templates` is a fact in the record that
        // never reaches the terminal -- which is the shape of both bugs this
        // header exists to prevent, one level up.
        let h = full();
        let printed: Vec<&str> = h
            .templates()
            .into_iter()
            .flat_map(|(_, names)| names.iter().copied())
            .collect();
        for (name, _) in h.fields() {
            assert!(
                printed.contains(&name),
                "field {name:?} is recorded but no line states it: {printed:?}"
            );
        }
    }

    #[test]
    fn every_printed_line_names_only_fields_that_exist() {
        // The other direction. A template naming a field that `fields` does
        // not emit would satisfy the test above forever while describing a
        // record key nobody can query.
        let h = full();
        let recorded: Vec<&str> = h.fields().into_iter().map(|(n, _)| n).collect();
        for (line, names) in h.templates() {
            for n in names {
                assert!(
                    recorded.contains(n),
                    "line {line:?} names {n:?}, which is not a recorded field: {recorded:?}"
                );
            }
        }
    }

    #[test]
    fn an_absent_optional_field_is_omitted_from_both_renderings() {
        // Absent, not null. A `peak_rss_bytes: null` claims the measurement
        // was taken and came back empty; the truth is that /proc could not
        // answer, and the two are different facts about the run.
        let h = RunHeader {
            peak_rss_bytes: None,
            rewind_horizon: None,
            probe: None,
            probe_layer: None,
            ..full()
        };
        let names: Vec<&str> = h.fields().into_iter().map(|(n, _)| n).collect();
        for absent in ["peak_rss_bytes", "rewind_horizon", "probe", "probe_layer"] {
            assert!(!names.contains(&absent), "{absent} present: {names:?}");
        }
        let text = h.lines().join("\n");
        assert!(!text.contains("Peak RSS"), "{text}");
        assert!(!text.contains("Rewind horizon"), "{text}");
        assert!(!text.contains("Probe"), "{text}");
    }

    #[test]
    fn the_fold_and_the_seed_are_both_stated() {
        // Named individually rather than left to the parity test above,
        // because these two are the entire reason the stage exists: a run
        // that does not print its dot kind is how Sequential stayed the
        // default for months, and one that does not print its seed is how
        // RH.12's runs drifted off the documented value.
        let text = full().lines().join("\n");
        assert!(text.contains("dot=widen4"), "{text}");
        assert!(text.contains("seed=7"), "{text}");
    }
}
