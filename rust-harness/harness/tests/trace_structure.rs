/*
 * trace_structure — a Chrome trace has the documented span shape, and a
 * real run's per-name totals are printed for the runbook.
 *
 * `tracing_spans.rs` proves the subscriber on traces it writes itself over
 * the tiny fixture, one span name at a time. This file asserts the count
 * relations docs/rust-harness/observability.md documents and the runbook's
 * RH.6 reads by eye: one layer span per layer per prefill group, the block
 * spans equal to it, one load with its stages, one sync per generation with
 * its prefill forward outside any step. The same checker runs over two
 * traces:
 *
 * - One the gate writes itself: a three-turn `--chat` run of the harness
 *   binary over the fixture. It runs on every `cargo test`, so a relation
 *   that stops holding fails `make quality` here rather than on the
 *   workstation. A sync-once assertion once shipped on main because nothing
 *   in the gate produced a chat trace to contradict it.
 * - One `harness --trace-chrome` wrote against the real checkpoint, a
 *   workstation artifact this tree does not carry: the path arrives in
 *   HARNESS_TRACE, which `make trace-check TRACE=<file>` sets. Under `make
 *   quality` the variable is unset and that test skips.
 *
 * Durations are printed, never asserted; they belong to the machine. The
 * trace level is not recorded in the file, so the shape decides which
 * level's relations apply.
 */

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

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

    /// Every `session.sync` span, paired with the first `model.forward`
    /// opened after it: the prefill that fed the sync.
    fn syncs_with_prefill(&self) -> Vec<(&Span, Option<&Span>)> {
        self.spans
            .iter()
            .enumerate()
            .filter(|(_, s)| s.name == "session.sync")
            .map(|(i, s)| {
                let forward = self.spans[i + 1..]
                    .iter()
                    .find(|f| f.name == "model.forward");
                (s, forward)
            })
            .collect()
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

/// A span argument as the integer it was written as.
fn arg(span: &Span, field: &str) -> usize {
    span.args
        .get(field)
        .unwrap_or_else(|| panic!("{} without a {field} field: {:?}", span.name, span.args))
        .parse()
        .unwrap_or_else(|e| panic!("{} {field} {:?}: {e}", span.name, span.args))
}

/// The per-name table of calls and summed duration, most expensive first.
/// Printed for a workstation trace only: it is the runbook's evidence, and
/// the fixture's timings say nothing worth a screen of gate output.
fn report(trace: &Trace) {
    let mut names: Vec<&String> = trace.duration_us.keys().collect();
    names.sort_by(|a, b| trace.duration_us[*b].total_cmp(&trace.duration_us[*a]));
    for name in &names {
        status::info(&format!(
            "{name:24} {:6} calls {:9.2}s",
            trace.count(name),
            trace.duration_us[*name] / 1e6
        ));
    }
}

/// Every relation the documentation states, asserted against one trace.
/// `source` names the trace in failure messages.
fn check_shape(trace: &Trace, source: &str) {
    // The four spans every generation carries, at any level.
    let forwards = trace.count("model.forward");
    let steps = trace.count("session.step");
    for name in [
        "session.sync",
        "session.step",
        "sampler.sample",
        "model.forward",
    ] {
        assert!(trace.count(name) >= 1, "{name} is absent from {source}");
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
    for (sync, forward) in trace.syncs_with_prefill() {
        let forward = forward.unwrap_or_else(|| {
            panic!(
                "session.sync {:?} with no model.forward after it",
                sync.args
            )
        });
        assert_eq!(
            forward.parent.as_deref(),
            Some("session.sync"),
            "the prefill model.forward after session.sync {:?} nests under {:?}, \
             expected the sync itself and no step",
            sync.args,
            forward.parent
        );
        status::info(&format!(
            "session.sync args {:?}, prefill model.forward args {:?}",
            sync.args, forward.args
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
        assert_parent(trace, stage, "model.load");
    }
    assert!(layers >= 1, "no model.load.layer span");
    assert_parent(trace, "model.load.layer", "model.from_checkpoint");

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
        .map(|s| arg(s, "groups"))
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
    assert_parent(trace, "model.layer", "model.forward");
    for block in ["model.layer.attn", "model.layer.mlp"] {
        assert_eq!(
            trace.count(block),
            layer_spans,
            "{block} once per model.layer"
        );
        assert_parent(trace, block, "model.layer");
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
        assert_parent(trace, per_group, "model.forward");
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
        // Once per logit row, not once per model.lm_head: the head span opens
        // on every prefill group, and a group before the last of a prefill
        // produces no row, so a prompt wider than the batch has more head
        // spans than kernel calls. The first trace-level trace on real
        // weights failed the per-group form (RH.37).
        let rows: usize = trace
            .spans
            .iter()
            .filter(|s| s.name == "model.lm_head")
            .map(|s| arg(s, "rows"))
            .sum();
        assert_eq!(
            trace.count("kernel.lm_head"),
            rows,
            "kernel.lm_head once per logit row across {batches} model.lm_head spans"
        );
        assert!(
            trace.count("kernel.matmul") >= layer_spans,
            "kernel.matmul below one per layer"
        );
        status::info(&format!("kernel spans {kernel_count}: a trace-level trace"));
    }
}

/// Three chat turns over the fixture, in one harness process, traced. This
/// is the gate's own trace: it is what makes `check_shape` run without a
/// workstation. Random weights produce random tokens, which is fine, because
/// every assertion is a count relation.
fn chat_trace_over_the_fixture(dir: &Path) -> Trace {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures");
    let trace_path = dir.join("trace.json");
    let mut child = Command::new(env!("CARGO_BIN_EXE_harness"))
        .args([
            "--model".as_ref(),
            fixtures.join("tiny-gemma4").as_os_str(),
            "--tokenizer".as_ref(),
            fixtures.join("tiny-gemma4.tokenizer.json").as_os_str(),
            "--text-only".as_ref(),
            "--chat".as_ref(),
            "--trace-chrome".as_ref(),
            trace_path.as_os_str(),
            "--max-tokens".as_ref(),
            "8".as_ref(),
            "--seed".as_ref(),
            "0".as_ref(),
        ])
        // The run's observability records go to scratch, not the real root.
        .env("OBS_LOG_DIR", dir.join("obs"))
        // session.jsonl and session.kv land in cwd; see main.rs.
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the harness builds as a test binary");
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(b"first turn\nsecond turn\nthird turn\n/quit\n")
        .expect("the chat loop reads stdin");
    let out = child.wait_with_output().expect("the harness exits");
    assert!(
        out.status.success(),
        "chat run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    fold(&events(&trace_path))
}

/// One task over the fixture at `--trace-level trace`. The chat trace above
/// runs at the default level and so carries no kernel span; this is the
/// trace that puts the kernel relations in the gate. The task prompt is
/// wider than the batch, so its prefill splits into groups, which is the
/// case a decode-only trace never shows: `model.lm_head` opens once per
/// group and `kernel.lm_head` once per row it produces, and only the last
/// group of a prefill produces one.
fn traced_task_over_the_fixture(dir: &Path) -> Trace {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures");
    let trace_path = dir.join("trace.json");
    let out = Command::new(env!("CARGO_BIN_EXE_harness"))
        .args([
            "--model".as_ref(),
            fixtures.join("tiny-gemma4").as_os_str(),
            "--text-only".as_ref(),
            "--trace-chrome".as_ref(),
            trace_path.as_os_str(),
            "--trace-level".as_ref(),
            "trace".as_ref(),
            "--max-tokens".as_ref(),
            "4".as_ref(),
            "--seed".as_ref(),
            "0".as_ref(),
            "say hi".as_ref(),
        ])
        .env("OBS_LOG_DIR", dir.join("obs"))
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .expect("the harness builds as a test binary");
    assert!(
        out.status.success(),
        "traced task run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    fold(&events(&trace_path))
}

#[test]
fn a_trace_level_task_run_over_the_fixture_has_the_documented_span_shape() {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let trace = traced_task_over_the_fixture(dir.path());
    let groups: usize = trace
        .spans
        .iter()
        .filter(|s| s.name == "model.forward")
        .map(|s| arg(s, "groups"))
        .sum();
    assert!(
        groups > trace.count("model.forward"),
        "the task prompt fits one batch group, so no prefill splits: \
         the relation this test exists for is not exercised"
    );
    assert!(
        trace.count("kernel.matmul") >= 1,
        "no kernel span: the run was not traced at trace level"
    );
    check_shape(&trace, "the fixture task trace at trace level");
}

#[test]
fn a_chat_run_over_the_fixture_has_the_documented_span_shape() {
    // Held to the end: the trace is read back from it.
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let trace = chat_trace_over_the_fixture(dir.path());
    check_shape(&trace, "the fixture chat trace");

    // What a chat trace adds to the relations above: one sync per turn, each
    // carrying the cache length it started from and followed by a prefill
    // forward with the three fields the checker reads. The forward's start is
    // not asserted against the previous cached length: the fixture's sliding
    // window is four tokens, so every turn after the first rewinds to zero
    // (the "rewind past the sliding window" event). That is the fixture's
    // property; the runbook's RH.11 reads the real number.
    let syncs = trace.syncs_with_prefill();
    assert_eq!(syncs.len(), 3, "session.sync once per chat turn");
    for (sync, forward) in syncs {
        let cached = arg(sync, "cached_len");
        let forward = forward.expect("check_shape found a forward after every sync");
        let (start, len, groups) = (
            arg(forward, "start"),
            arg(forward, "len"),
            arg(forward, "groups"),
        );
        assert!(
            start <= cached && len >= 1 && groups >= 1,
            "prefill model.forward start {start} len {len} groups {groups} \
             after session.sync cached_len {cached}"
        );
    }
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
    // The table first, so a failing assertion is read with the counts in view.
    report(&trace);
    check_shape(&trace, &path);
}
