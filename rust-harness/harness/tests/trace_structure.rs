/*
 * trace_structure — a Chrome trace from a real run has the documented span
 * shape, and its per-name totals are printed for the runbook.
 *
 * `tracing_spans.rs` proves the subscriber on traces it writes itself over
 * the tiny fixture. This one reads a trace `harness --trace-chrome` wrote
 * against the real checkpoint, a workstation artifact this tree does not
 * carry: the path arrives in HARNESS_TRACE, which `make trace-check
 * TRACE=<file>` sets. Under `make quality` the variable is unset and the test
 * skips, so the gate is not conditioned on a workstation run.
 *
 * The assertions are the count relations docs/rust-harness/observability.md
 * documents and the runbook's RH.6 reads by eye: one layer span per layer per
 * prefill group, the block spans equal to it, one load with its stages, one
 * sync per generation with its prefill forward outside any step. Durations
 * are printed, never asserted; they belong to the machine. The trace level is not recorded in the file, so the shape
 * decides which level's relations apply.
 */

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use runtime::{status, testlog};
use serde_json::Value;

/// One `B` event's identity: its name, its parent's name on the same thread
/// (the enclosing open span), and its `args` as written.
struct Span {
    name: String,
    parent: Option<String>,
    args: BTreeMap<String, String>,
}

/// The trace folded to what the assertions need: every span opened, and the
/// summed duration per name in microseconds.
struct Trace {
    spans: Vec<Span>,
    duration_us: HashMap<String, f64>,
}

impl Trace {
    fn count(&self, name: &str) -> usize {
        self.spans.iter().filter(|s| s.name == name).count()
    }

    /// Parents of every span named `name`, deduplicated, for asserting that
    /// a span nests where the documentation says it does.
    fn parents_of(&self, name: &str) -> Vec<Option<String>> {
        let mut parents: Vec<Option<String>> = self
            .spans
            .iter()
            .filter(|s| s.name == name)
            .map(|s| s.parent.clone())
            .collect();
        parents.sort();
        parents.dedup();
        parents
    }
}

/// Fold the event list with one stack of open spans per thread. Panics with
/// the event index on an `E` that closes nothing or closes the wrong name;
/// an unbalanced file is not a trace this test can say anything about.
fn fold(events: &[Value]) -> Trace {
    let mut open: HashMap<String, Vec<(usize, f64)>> = HashMap::new();
    let mut spans: Vec<Span> = Vec::new();
    let mut duration_us: HashMap<String, f64> = HashMap::new();
    for (i, e) in events.iter().enumerate() {
        let Some(ph) = e.get("ph").and_then(Value::as_str) else {
            continue;
        };
        if ph != "B" && ph != "E" {
            continue;
        }
        let tid = e.get("tid").map(Value::to_string).unwrap_or_default();
        let ts = e
            .get("ts")
            .and_then(Value::as_f64)
            .unwrap_or_else(|| panic!("event {i} has no numeric ts: {e}"));
        let name = e
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("event {i} has no name: {e}"));
        let stack = open.entry(tid).or_default();
        if ph == "B" {
            let parent = stack.last().map(|&(j, _)| spans[j].name.clone());
            let args = e
                .get("args")
                .and_then(Value::as_object)
                .map(|o| {
                    o.iter()
                        .map(|(k, v)| {
                            let v = v
                                .as_str()
                                .map(str::to_string)
                                .unwrap_or_else(|| v.to_string());
                            (k.clone(), v)
                        })
                        .collect::<BTreeMap<String, String>>()
                })
                .unwrap_or_default();
            stack.push((spans.len(), ts));
            spans.push(Span {
                name: name.to_string(),
                parent,
                args,
            });
        } else {
            let (j, start) = stack
                .pop()
                .unwrap_or_else(|| panic!("event {i} closes {name} with nothing open"));
            let opened: &Span = &spans[j];
            assert_eq!(
                opened.name, name,
                "event {i} closes {name} while {} is the open span",
                opened.name
            );
            *duration_us.entry(name.to_string()).or_default() += ts - start;
        }
    }
    let leftover: Vec<String> = open
        .values()
        .flatten()
        .map(|&(j, _)| spans[j].name.clone())
        .collect();
    assert!(leftover.is_empty(), "spans never closed: {leftover:?}");
    Trace { spans, duration_us }
}

/// The Chrome trace event array, from either shape a `.json` trace takes.
fn events(path: &Path) -> Vec<Value> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let value: Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    match value {
        Value::Array(events) => events,
        Value::Object(mut o) => match o.remove("traceEvents") {
            Some(Value::Array(events)) => events,
            _ => panic!("{}: neither an event array nor traceEvents", path.display()),
        },
        other => panic!("{}: unexpected top-level {other}", path.display()),
    }
}

fn assert_parent(trace: &Trace, name: &str, parent: &str) {
    let parents = trace.parents_of(name);
    assert!(
        parents.iter().all(|p| p.as_deref() == Some(parent)),
        "{name} nests under {parents:?}, expected only {parent:?}"
    );
}

#[test]
fn a_chrome_trace_from_a_real_run_has_the_documented_span_shape() {
    let Ok(path) = std::env::var("HARNESS_TRACE") else {
        testlog::skip(
            "chrome trace structure",
            "HARNESS_TRACE is not set; write a trace with harness --trace-chrome and \
             pass its path, or run make trace-check TRACE=<file>",
        );
        return;
    };
    let trace = fold(&events(Path::new(&path)));

    // The per-name table first, so a failing assertion below is read with
    // the counts in view.
    let mut names: Vec<&String> = trace.duration_us.keys().collect();
    names.sort_by(|a, b| trace.duration_us[*b].total_cmp(&trace.duration_us[*a]));
    for name in &names {
        status::info(&format!(
            "{name:24} {:6} calls {:9.2}s",
            trace.count(name),
            trace.duration_us[*name] / 1e6
        ));
    }

    // The four spans every generation carries, at any level.
    let forwards = trace.count("model.forward");
    let steps = trace.count("session.step");
    for name in [
        "session.sync",
        "session.step",
        "sampler.sample",
        "model.forward",
    ] {
        assert!(trace.count(name) >= 1, "{name} is absent from {path}");
    }
    // session.sync fires once per generation: once in a single-prompt run,
    // once per turn under --chat, where one process syncs each turn's delta
    // on top of the previous turn's cache. Each sync's prefill forward has no
    // enclosing step. A generation that hit its token budget forwarded on
    // every step, so its forwards are its steps plus one; one that stopped on
    // EOS sampled EOS in its last step and forwarded nothing after it, so the
    // two are equal. Over the whole trace that bounds forwards between steps
    // and steps plus syncs.
    let syncs = trace.count("session.sync");
    assert!(
        forwards >= steps && forwards <= steps + syncs,
        "model.forward {forwards} against session.step {steps} and session.sync {syncs}: \
         expected between steps and steps + syncs"
    );
    status::info(&format!(
        "session.sync {syncs} calls: {} stopped on EOS, {} on the token budget",
        steps + syncs - forwards,
        forwards - steps
    ));
    // Each sync's cached length, and the start and length of the prefill
    // forward that follows it, which is the delta it prefilled. The second
    // sync of a chat trace is RH.4's claim made in one process: its start is
    // the first turn's cached length, and its len the new turn alone.
    for (i, s) in trace.spans.iter().enumerate() {
        if s.name != "session.sync" {
            continue;
        }
        let forward = trace.spans[i + 1..]
            .iter()
            .find(|f| f.name == "model.forward");
        status::info(&format!(
            "session.sync args {:?}, prefill model.forward args {:?}",
            s.args,
            forward.map(|f| &f.args)
        ));
    }

    // model.load is written at INFO, its stages and everything inside the
    // forward at DEBUG or TRACE. An info-level trace therefore has the load
    // and none of its interior, and that is a level choice, not a defect.
    assert_eq!(trace.count("model.load"), 1, "one model.load per run");
    let layers = trace.count("model.load.layer");
    if trace.count("model.layer") == 0 {
        assert_eq!(layers, 0, "model.load.layer without model.layer");
        assert_eq!(
            trace.count("model.from_checkpoint"),
            0,
            "load stages without layer spans"
        );
        status::info("interior spans absent: an info-level trace");
        return;
    }

    for stage in [
        "checkpoint.read",
        "checkpoint.parse",
        "model.from_checkpoint",
        "checkpoint.fingerprint",
    ] {
        assert_eq!(trace.count(stage), 1, "{stage} once under model.load");
        assert_parent(&trace, stage, "model.load");
    }
    assert!(layers >= 1, "no model.load.layer span");
    assert_parent(&trace, "model.load.layer", "model.from_checkpoint");

    // The layer stack runs once per prefill group, not once per forward: a
    // forward wider than the batch splits into groups, and the spans inside
    // it are per group. Each forward's `groups` field carries the count; a
    // decode forward is one group, as is any prefill within the batch width,
    // which is why every single-prompt trace before the field existed read
    // as one `model.layer` per layer per forward.
    let batches: usize = trace
        .spans
        .iter()
        .filter(|s| s.name == "model.forward")
        .map(|s| {
            s.args
                .get("groups")
                .unwrap_or_else(|| panic!("model.forward without a groups field: {:?}", s.args))
                .parse::<usize>()
                .unwrap_or_else(|e| panic!("model.forward groups {:?}: {e}", s.args))
        })
        .sum();
    assert!(
        batches >= forwards,
        "prefill groups {batches} below model.forward {forwards}"
    );
    let layer_spans = trace.count("model.layer");
    assert_eq!(
        layer_spans,
        layers * batches,
        "model.layer {layer_spans} against {layers} layers x {batches} prefill groups"
    );
    assert_parent(&trace, "model.layer", "model.forward");
    for block in ["model.layer.attn", "model.layer.mlp"] {
        assert_eq!(
            trace.count(block),
            layer_spans,
            "{block} once per model.layer"
        );
        assert_parent(&trace, block, "model.layer");
    }
    let ple = trace.count("model.layer.ple");
    assert!(
        ple == 0 || ple == layer_spans,
        "model.layer.ple {ple}: every layer or none, against {layer_spans} layers"
    );
    for per_group in ["model.prelude", "model.lm_head"] {
        assert_eq!(
            trace.count(per_group),
            batches,
            "{per_group} once per prefill group"
        );
        assert_parent(&trace, per_group, "model.forward");
    }
    assert!(
        trace.count("tokenizer.encode") >= 1,
        "no tokenizer.encode span"
    );
    status::info(&format!(
        "layers {layers}, forwards {forwards}, prefill groups {batches}, \
         model.layer / group = {:.1}",
        layer_spans as f64 / batches as f64
    ));

    let kernels = ["kernel.matmul", "kernel.attention", "kernel.lm_head"];
    let kernel_count: usize = kernels.iter().map(|k| trace.count(k)).sum();
    if kernel_count == 0 {
        status::info("kernel spans absent: a debug-level trace");
    } else {
        assert_eq!(
            trace.count("kernel.attention"),
            layer_spans,
            "kernel.attention once per model.layer"
        );
        assert_eq!(
            trace.count("kernel.lm_head"),
            batches,
            "kernel.lm_head once per model.lm_head"
        );
        assert!(
            trace.count("kernel.matmul") >= layer_spans,
            "kernel.matmul below one per layer"
        );
        status::info(&format!("kernel spans {kernel_count}: a trace-level trace"));
    }
}
