/*
 * tracing_spans — phase 2 acceptance (runtime side): the span schema is
 * asserted with a capturing subscriber over a full mock generation. Span
 * names and fields are the stable vocabulary shared with debugging and
 * benchmark analysis, so a rename here is a schema change, not a detail.
 */

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use runtime::grammar::{Grammar, ParamType, ToolSchema, Vocab};
use runtime::model::mock::MockModel;
use runtime::sampler::{Sampler, SamplingParams};
use runtime::session::Session;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

/// One captured span: its name and every field recorded on it, at creation
/// or later via `Span::record`.
#[derive(Debug, Clone)]
struct CapturedSpan {
    name: String,
    fields: Vec<(String, String)>,
}

#[derive(Default)]
struct CaptureState {
    spans: Vec<CapturedSpan>,
    // Registry span ids are reused after close; map only the currently open
    // ids to their chronological slot for the `on_record` merge.
    open: HashMap<u64, usize>,
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<CaptureState>>);

struct FieldVisitor(Vec<(String, String)>);

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0
            .push((field.name().to_string(), format!("{value:?}")));
    }
}

impl<S: Subscriber> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor(Vec::new());
        attrs.record(&mut visitor);
        let mut state = self.0.lock().unwrap();
        let idx = state.spans.len();
        state.spans.push(CapturedSpan {
            name: attrs.metadata().name().to_string(),
            fields: visitor.0,
        });
        state.open.insert(id.into_u64(), idx);
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        // Registry ids are reused after close; drop the mapping so a later
        // span with the same id cannot merge fields into an earlier slot.
        self.0.lock().unwrap().open.remove(&id.into_u64());
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor(Vec::new());
        values.record(&mut visitor);
        let mut state = self.0.lock().unwrap();
        if let Some(&idx) = state.open.get(&id.into_u64()) {
            state.spans[idx].fields.extend(visitor.0);
        }
    }
}

impl Capture {
    fn named(&self, name: &str) -> Vec<CapturedSpan> {
        self.0
            .lock()
            .unwrap()
            .spans
            .iter()
            .filter(|s| s.name == name)
            .cloned()
            .collect()
    }
}

fn field<'a>(span: &'a CapturedSpan, name: &str) -> Option<&'a str> {
    span.fields
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
}

/// Serializes the two tests in this binary. `tracing` caches per-callsite
/// interest process-globally, so a test driving these callsites with no
/// subscriber can race another test's subscriber registration and leave a
/// stale "never" interest behind, silently dropping spans for that run.
/// Cargo runs tests in one process on multiple threads, so the lock is the
/// fix rather than an ordering assumption.
static CALLSITE_LOCK: Mutex<()> = Mutex::new(());

/// Grammar-constrained generation over a character vocab, driving every
/// instrumented runtime path: sync, step, sample, and grammar mask.
fn traced_generation(capture: &Capture) {
    let mut tokens: Vec<String> = "{}\"argstol: ,ping123"
        .chars()
        .map(|c| c.to_string())
        .collect();
    tokens.push("</s>".to_string());
    let eos = (tokens.len() - 1) as u32;
    let vocab = Vocab::new(tokens, eos);
    let grammar = Grammar::compile(&ToolSchema {
        name: "ping".into(),
        params: vec![("n".into(), ParamType::Int)],
    });
    let script = vec![vec![1.0; vocab.len()]; 512];
    let subscriber = Registry::default().with(capture.clone());
    tracing::subscriber::with_default(subscriber, || {
        let mut session =
            Session::new(Box::new(MockModel::new(vocab.len(), script)), 4096).unwrap();
        let mut sampler = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            seed: 5,
        });
        session
            .generate(&[0, 1], &vocab, Some(&grammar), &mut sampler, 256, |_| {})
            .unwrap();
    });
}

#[test]
fn generation_emits_the_documented_span_schema() {
    let _serial = CALLSITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let capture = Capture::default();
    traced_generation(&capture);

    // session.sync: the prompt prefill with its delta and cache lengths.
    let syncs = capture.named("session.sync");
    assert_eq!(syncs.len(), 1);
    assert_eq!(field(&syncs[0], "cached_len"), Some("0"));
    assert_eq!(field(&syncs[0], "delta_len"), Some("2"));

    // session.step: one per generated token (plus the EOS step), positions
    // advancing from the prompt boundary, each with a chosen token.
    let steps = capture.named("session.step");
    assert!(steps.len() > 2, "expected a multi-token generation");
    assert_eq!(field(&steps[0], "position"), Some("2"));
    assert!(field(&steps[0], "token").is_some());

    // sampler.sample: one per step, carrying the configured temperature and
    // a recorded candidate count.
    let samples = capture.named("sampler.sample");
    assert_eq!(samples.len(), steps.len());
    assert_eq!(field(&samples[0], "temperature"), Some("1.0"));
    assert!(field(&samples[0], "candidates").is_some());

    // grammar.advance: one mask per step, with the automaton state and the
    // allowed-token count under the mask.
    let masks = capture.named("grammar.advance");
    assert_eq!(masks.len(), steps.len());
    assert!(field(&masks[0], "state").is_some());
    let allowed: usize = field(&masks[0], "allowed").unwrap().parse().unwrap();
    assert!(allowed >= 1);
}

#[test]
fn no_subscriber_means_no_capture_and_no_failure() {
    let _serial = CALLSITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Spans compile to disabled no-ops without a subscriber; a plain run
    // must behave identically.
    let vocab = Vocab::new(vec!["a".into(), "</s>".into()], 1);
    let script = vec![vec![1.0, 0.0]; 4];
    let mut session = Session::new(Box::new(MockModel::new(2, script)), 16).unwrap();
    let mut sampler = Sampler::new(SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        seed: 0,
    });
    let generation = session
        .generate(&[0], &vocab, None, &mut sampler, 2, |_| {})
        .unwrap();
    assert_eq!(generation.tokens.len(), 2);
}
