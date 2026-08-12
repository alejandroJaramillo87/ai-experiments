/*
 * long_context.rs — the forward pass at lengths where the window engages.
 *
 * Gemma 4's architectural interest is the interleaved sliding/global split and
 * the shared-KV tail, and none of it does anything until the sequence outruns
 * the sliding window. Before this file nothing in the workspace reached that
 * point. `batched_prefill_matches_stepped_decode` (cpu.rs) runs five tokens
 * against a window of four -- one position past it, and never as far as a
 * batch group boundary at DEFAULT_BATCH 64. The ladder is no better: L1 and L2
 * prompt about six tokens, and L3's windows are prefilled from an empty cache,
 * so even `--size deep` never holds more than CHUNK = 256 positions at once
 * against E4B's window of 512. No rung has crossed a real sliding window.
 *
 * This file was written before the KV storage cap, to catch it. The cap has
 * since landed and every assertion below held through it unchanged, golden
 * included; the arms under `Storage` were added with it, because everything
 * written first compares logits and the cap must not move a logit. A suite
 * that could not tell whether the cap exists is not a suite for the cap.
 *
 * Two failure shapes, needing two kinds of assertion:
 *
 * A *composition-dependent* truncation. In a batched group the early query
 * rows legitimately need cache rows that a naive ring buffer, sized to the
 * window and advanced to the group's last position, has already dropped. That
 * one is caught by comparing batch compositions against each other, because
 * a stepped pass never asks for an evicted row and a batched one does. It is
 * the trap mistral.rs documents, and it bites exactly at a group boundary.
 *
 * A *uniform* truncation, which every composition shares. Comparing the engine
 * against itself cannot see that at all: both sides carry the change. Only a
 * differential over *inputs* can -- perturb a token and require the logits to
 * move -- or a golden recorded before the change. Both are here, and the
 * distinction is why the tests are grouped the way they are below.
 *
 * These are self-comparisons, so none of them is evidence that the window
 * convention itself is right. That is pinned against transformers by
 * `ladder_tiny.rs` on the committed Tier T fixture, whose window is 3 and
 * whose capture runs 150 tokens -- fifty crossings of it. The golden here was
 * blessed only after that passed, so it freezes a convention a reference
 * implementation has agreed with rather than one we merely hold.
 *
 * Mutation checks performed when this file was written
 * ----------------------------------------------------
 *
 * Four mutations of the mask bound at cpu.rs, and which tests noticed. The
 * fourth is the one that justifies the file. All four also fail
 * `the_long_context_logits_match_the_committed_golden`, which is what a
 * change detector is for and is not repeated in each entry below.
 *
 *   1. `position.saturating_sub(win)` -- the window one wider. Fails
 *      `a_token_outside_the_composed_reach_..._changes_nothing` here, and the
 *      committed reference goldens in `cpu_reference.rs`.
 *
 *   2. `(position + 2).saturating_sub(win)` -- one narrower. Fails
 *      `a_token_just_inside_the_composed_reach_changes_the_logits` and
 *      `a_windowed_model_matches_a_full_attention_one_below_the_window`.
 *      Note that 1 and 2 fail *different* tests: eviction can only shorten
 *      reach, so the negative direction passes under a truncating bug and the
 *      positive one is what catches it. Neither test is redundant.
 *
 *   3. Clamping every group to a window-sized ring. Fails five tests here --
 *      and also three in `cpu.rs`, because at five tokens against a window of
 *      four that mutation already evicts a row position 0 needs. So this one
 *      does not show what the file is for.
 *
 *   4. The same clamp applied only from the second group onward and only when
 *      the group is wider than the window -- which is the shape a relocation
 *      bug in a windowed store actually takes, since the first group of a
 *      fresh cache has nothing to relocate. **Every other test binary in the
 *      workspace passes**: 330 unit tests and eight integration binaries, all
 *      green. Five tests in this file fail. That gap is the package.
 *
 * Mutation checks performed when the storage cap landed
 * ------------------------------------------------------
 *
 * The cap is now implemented, and the arms below `Storage` were added with
 * it. Four more mutations, of the eviction rather than of the mask. The
 * result is more interesting than "the tests catch it", so it is recorded as
 * what it is:
 *
 *   5. Retain from the group's *last* query position rather than its first
 *      -- the trap this file was written for. **Nothing fails.** Not a gap:
 *      eviction is driven by the capacity, and `capacity >= window_residency`
 *      makes the capacity-driven point strictly older than the retention
 *      point, so the retention bound is never what binds. It is a guard, not
 *      a computation, which is why `LayerKv::append` takes it as an assertion
 *      alongside the arithmetic instead of deriving the drop from it.
 *
 *   6. Size the capacity at `win`, ignoring the batch term. Sixteen of
 *      eighteen fail in a debug build, all on the `want <= retain`
 *      `debug_assert` in `LayerKv::append`.
 *
 *   7. The same mutation in `--release`, where that assertion is compiled
 *      out. **Everything passes**, including the golden. The `.min(retain)`
 *      clamp degrades an undersized capacity to the minimum safe residency:
 *      the logits stay right and the cache stays bounded, so there is nothing
 *      to see. Correctness has two independent guards and losing one is not
 *      observable -- which is the intended arrangement and the reason the
 *      clamp is not redundant with the assertion.
 *
 *   8. Both at once -- capacity at `win` and the clamp removed -- in
 *      `--release`. Sixteen of eighteen fail, the golden among them. Only the
 *      two arms that assert on a shared layer's empty cache survive, and they
 *      are about a layer that stores nothing.
 *
 * The moral for a CUDA backend: an eviction bug has to defeat both the
 * capacity and the clamp before it reaches a logit, and if it defeats only
 * the capacity the symptom is memory, not numerics. That is what
 * `a_sliding_layer_stops_growing_once_the_window_is_full` is for, and it is
 * the only arm here that would notice.
 *
 * Cost, measured: about 1.2 s wall across the eighteen tests, against a suite
 * that was 9.4 s before this file existed. The composition sweeps are trimmed
 * for this reason -- see `check_compositions`.
 */

use std::path::Path;

use anyhow::Result;
use runtime::conformance;
use runtime::model::cpu::{BatchKind, CpuModel};
use runtime::model::{Logits, Model, Token};
use runtime::synth::TinySpec;

/// Four layers, window 3, sliding and global interleaved, and a two-layer
/// shared-KV tail. Layers 0 and 2 are sliding, 1 and 3 global; 2 and 3 read a
/// donor's cache, so layer 2 is a *shared sliding* layer reading layer 0.
///
/// That is the configuration the storage cap has to survive, and the one E4B
/// ships at scale: 42 layers, window 512, 18 of them shared.
fn shared_spec() -> TinySpec {
    TinySpec::e4b_shaped()
}

/// Three layers, all sliding, window 4.
///
/// A stack with no global layer, so the reach of a position is finite and
/// computable -- with a global layer anywhere it is the whole prefix and the
/// bound tests below have nothing to assert. `sliding_window_pattern: 4` over
/// three layers makes every one sliding, since the pattern marks every fourth.
fn all_sliding_spec() -> TinySpec {
    TinySpec {
        n_layers: 3,
        sliding_window_pattern: 4,
        sliding_window: 4,
        ..TinySpec::default()
    }
}

/// Layers of `all_sliding_spec`, and its window. Named so the reach arithmetic
/// below reads against the spec rather than against two loose literals.
const SLIDING_LAYERS: usize = 3;
const SLIDING_WINDOW: usize = 4;

fn model_of(spec: &TinySpec) -> CpuModel {
    let files = runtime::synth::tiny_checkpoint(spec).expect("build the synthetic checkpoint");
    let ck = runtime::checkpoint::parse(&files).expect("parse the synthetic checkpoint");
    CpuModel::from_checkpoint(&ck).expect("load the synthetic checkpoint")
}

fn model_with_batch(spec: &TinySpec, batch: BatchKind) -> Box<dyn Model> {
    let mut m = model_of(spec);
    m.set_batch(batch);
    Box::new(m)
}

/// `n` deterministic in-vocabulary tokens, from the one definition of the
/// sequence, so this file and the committed golden cannot drift apart.
fn tokens_of(n: usize) -> Vec<Token> {
    conformance::golden::tokens(n)
}

/// An owned builder and the composition it sets up.
///
/// `conformance::Builder` is a borrowed `&dyn Fn`, so the closures have to be
/// kept alive somewhere while the slice of references into them is built. This
/// is that somewhere.
type OwnedComposition = (&'static str, Box<dyn Fn() -> Result<Box<dyn Model>>>);

/// The compositions worth comparing, as labelled builders.
///
/// Six, not the eight an exhaustive sweep would give. `prefill_groups` chunks
/// uniformly, so the structural cases are: one token at a time, a width that
/// divides the length, one that does not (a short trailing group), and one
/// that exceeds it (a single group). Widths 2 and 63 re-run arithmetic that 1
/// and 64 already cover.
fn compositions(spec: &TinySpec) -> Vec<OwnedComposition> {
    let widths: [(&'static str, BatchKind); 6] = [
        ("stepped", BatchKind::Stepped),
        ("batch 1", BatchKind::Batched(1)),
        ("batch 7", BatchKind::Batched(7)),
        ("batch 64", BatchKind::Batched(64)),
        ("batch 65", BatchKind::Batched(65)),
        ("whole", BatchKind::Whole),
    ];
    widths
        .into_iter()
        .map(|(label, batch)| {
            let spec = *spec;
            let build: Box<dyn Fn() -> Result<Box<dyn Model>>> =
                Box::new(move || Ok(model_with_batch(&spec, batch)));
            (label, build)
        })
        .collect()
}

/// Compare `keep` of the compositions at each length.
///
/// The subset exists for cost. A composition sweep at 200 tokens is six full
/// prefills, and `stepped` alone is 200 forward passes; the gate is run before
/// every push and this file is not worth a second of it. Widths are dropped
/// where they would re-run arithmetic the shorter lengths already covered,
/// never where they would drop a structural case.
fn check_compositions(spec: &TinySpec, lengths: &[usize], keep: &[&str]) {
    let built = compositions(spec);
    let builders: Vec<(&str, conformance::Builder)> = built
        .iter()
        .filter(|(label, _)| keep.contains(label))
        .map(|(label, build)| (*label, build.as_ref() as conformance::Builder))
        .collect();
    assert_eq!(
        builders.len(),
        keep.len(),
        "a composition name in {keep:?} matches nothing in `compositions`"
    );
    for &n in lengths {
        conformance::check_batch_composition_invariance(&builders, &tokens_of(n))
            .unwrap_or_else(|e| panic!("{n} tokens: {e:#}"));
    }
}

/// Every composition, at lengths cheap enough to run all of them: just past
/// twice the window, then one token past a single group so the trailing group
/// is short.
const SHORT: [usize; 2] = [7, 65];
const ALL_WIDTHS: [&str; 6] = [
    "stepped", "batch 1", "batch 7", "batch 64", "batch 65", "whole",
];

/// Three groups and a short trailing one. Only the widths that change the
/// grouping structure at this length -- one position at a time, the default
/// width, and one pass -- since 1, 7 and 65 differ from 64 only in where the
/// boundaries fall, which `SHORT` already varied.
const LONG: [usize; 1] = [200];
const LONG_WIDTHS: [&str; 3] = ["stepped", "batch 64", "whole"];

// The composition-dependent arm. These fail on a windowed store that evicts a
// row a batched group's early query positions still need.

/// The donor arm, and the one closest to E4B's shape.
///
/// `shared_spec` puts a shared *sliding* layer at depth 2 reading layer 0's
/// cache. Under a storage cap the rows it wants for the early positions of a
/// 64-wide group are the first thing a ring buffer drops, and a stepped pass
/// never asks for them -- so the two compositions diverge and this fails.
/// `cpu.rs`'s module header warns about the donor for this reason.
#[test]
fn every_batch_composition_agrees_past_the_sliding_window() {
    check_compositions(&shared_spec(), &SHORT, &ALL_WIDTHS);
    check_compositions(&shared_spec(), &LONG, &LONG_WIDTHS);
}

/// The same property with no global layer to mask a windowing error.
///
/// A global layer attends over the whole prefix, so it reproduces information
/// a broken sliding layer dropped and the logits can agree for the wrong
/// reason. Worth its own spec rather than folding into the test above.
#[test]
fn an_all_sliding_stack_agrees_across_batch_widths_past_the_window() {
    check_compositions(&all_sliding_spec(), &SHORT, &ALL_WIDTHS);
    check_compositions(&all_sliding_spec(), &LONG, &LONG_WIDTHS);
}

/// Asking for fewer rows must not change the rows that remain.
///
/// Offsets straddle group boundaries at width 64 rather than sweeping every
/// position: each costs a fresh model and a full prefill, and the offsets that
/// differ in anything are the ones landing inside a group rather than on its
/// edge.
#[test]
fn logit_rows_are_the_same_whether_or_not_neighbouring_rows_were_requested() {
    let spec = shared_spec();
    let tokens = tokens_of(130);
    let build = move || Ok(model_with_batch(&spec, BatchKind::Batched(64)));
    // Both ends, both sides of the first group boundary, and one offset
    // landing inside a group -- the case `prefill_groups` handles with a
    // group-relative offset and the only one where its arithmetic can slip.
    conformance::check_logit_rows_are_batch_isolated(&build, &tokens, &[0, 1, 63, 64, 100, 130])
        .unwrap_or_else(|e| panic!("{e:#}"));
}

// The window-is-the-only-difference arm. A genuine full-attention comparison
// with no second implementation of the forward pass to get wrong.

/// The precondition of the test below, asserted rather than assumed.
///
/// Widening the window is only a fair comparison if it changes nothing else.
/// It does not: `sliding_window` reaches the emitted config and nothing else,
/// while the weight stream is shaped by the layer count, the sliding *pattern*,
/// the shared-KV boundary and the MLP width. The comparison downstream is
/// worth exactly as much as this, so it is checked and not reasoned about.
#[test]
fn widening_the_window_changes_the_weights_of_nothing() {
    let narrow = all_sliding_spec();
    let wide = TinySpec {
        sliding_window: narrow.context_length,
        ..narrow
    };
    let a = runtime::synth::build_tiny_safetensors(&narrow).unwrap();
    let b = runtime::synth::build_tiny_safetensors(&wide).unwrap();
    assert_eq!(
        a.len(),
        b.len(),
        "widening the window resized the weight image: {} against {}",
        a.len(),
        b.len()
    );
    let differ = a.iter().zip(b.iter()).position(|(x, y)| x != y);
    assert!(
        differ.is_none(),
        "widening the window changed the weight image at byte {}",
        differ.unwrap()
    );
}

/// Below the window, a sliding layer is a full-attention layer.
///
/// `(position + 1).saturating_sub(window)` is 0 for every position inside the
/// window, which is literally the `LayerKind::Global` arm of the same match:
/// the same span, folded in the same order. So the two models must agree to
/// the last bit, and this is a full-attention reference obtained without
/// writing a second attention.
///
/// A hand-written naive mask would have been the obvious alternative and is
/// worth less: it would carry the same window convention this file's authors
/// hold, which is the oracle problem the README describes.
#[test]
fn a_windowed_model_matches_a_full_attention_one_below_the_window() {
    let narrow = all_sliding_spec();
    let wide = TinySpec {
        sliding_window: narrow.context_length,
        ..narrow
    };
    let tokens = tokens_of(SLIDING_WINDOW);
    let windowed = model_of(&narrow).prefill_all(&tokens).unwrap();
    let full = model_of(&wide).prefill_all(&tokens).unwrap();
    for (p, (w, f)) in windowed.iter().zip(full.iter()).enumerate() {
        assert_eq!(
            w,
            f,
            "position {p} of {} differs below the window",
            tokens.len()
        );
    }
}

/// And past it, they must not.
///
/// Without this the test above would also pass on an engine that ignored the
/// window entirely.
#[test]
fn a_windowed_model_diverges_from_a_full_attention_one_past_the_window() {
    let narrow = all_sliding_spec();
    let wide = TinySpec {
        sliding_window: narrow.context_length,
        ..narrow
    };
    let tokens = tokens_of(200);
    let windowed = model_of(&narrow).prefill_all(&tokens).unwrap();
    let full = model_of(&wide).prefill_all(&tokens).unwrap();
    assert_ne!(
        windowed.last().unwrap(),
        full.last().unwrap(),
        "the window changed nothing at position {}, so it is not being applied",
        tokens.len() - 1
    );
}

// The reach arm. Differentials over inputs, so a uniform truncation cannot
// hide in them.

/// The earliest position that can reach `p` through `SLIDING_LAYERS` layers.
///
/// Layer 0 at `q` reads `[q - w + 1, q]`; each further layer composes another
/// `w - 1`. So `p` is influenced by exactly `[p - layers * (w - 1), p]`, and
/// the index one before that is the first that cannot matter.
fn earliest_reaching(p: usize) -> usize {
    p - SLIDING_LAYERS * (SLIDING_WINDOW - 1)
}

/// Logits at the last position, with the token at `at` replaced.
fn logits_with_token_changed(spec: &TinySpec, n: usize, at: Option<usize>) -> Logits {
    let mut tokens = tokens_of(n);
    if let Some(i) = at {
        tokens[i] = (tokens[i] + 1) % 250 + 5;
    }
    model_of(spec).prefill_all(&tokens).unwrap().pop().unwrap()
}

#[test]
fn a_token_outside_the_composed_reach_of_the_sliding_stack_changes_nothing() {
    let spec = all_sliding_spec();
    let n = 200;
    let outside = earliest_reaching(n - 1) - 1;
    assert_eq!(
        logits_with_token_changed(&spec, n, None),
        logits_with_token_changed(&spec, n, Some(outside)),
        "token {outside} is outside the reach of position {} and must not \
         reach it through {SLIDING_LAYERS} layers of window {SLIDING_WINDOW}",
        n - 1
    );
}

/// The direction that a truncating storage bug breaks.
///
/// Eviction can only make reach *shorter*, so the negative test above passes
/// either way and this is the one that fails. Asserted at the exact boundary
/// index, because an off-by-one in the reach is the whole content of the pair.
#[test]
fn a_token_just_inside_the_composed_reach_changes_the_logits() {
    let spec = all_sliding_spec();
    let n = 200;
    let inside = earliest_reaching(n - 1);
    assert_ne!(
        logits_with_token_changed(&spec, n, None),
        logits_with_token_changed(&spec, n, Some(inside)),
        "token {inside} is the earliest that reaches position {}, so changing \
         it must change the logits",
        n - 1
    );
}

/// The committed golden, which is the only arm that fails on a change nobody
/// predicted.
///
/// Every other test here states a property and checks it, so it catches the
/// failures someone thought of. This one holds bytes recorded at a point in
/// time and notices any difference at all, which is what a package that
/// rewrites how the KV cache is stored actually needs pointed at it.
///
/// Not an oracle -- we wrote the engine that produced it. Its standing comes
/// from `ladder_tiny.rs`, which checked this window convention against
/// transformers at 150 tokens over 50 window crossings, and this file was
/// blessed only after that passed. Regenerate with
/// `cargo run -p runtime --bin long-context-golden -- fixtures/long-context-logits.json`,
/// and say in the commit message why the bytes moved.
#[test]
fn the_long_context_logits_match_the_committed_golden() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/long-context-logits.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let golden: serde_json::Value = serde_json::from_str(&text).expect("parse the golden");

    // The golden records the model it describes. A fixture blessed against a
    // different spec would compare two unrelated things and fail for a reason
    // that is not the engine's.
    let spec = conformance::golden::spec();
    assert_eq!(
        golden["sliding_window"].as_u64(),
        Some(spec.sliding_window as u64),
        "the golden was blessed against a different window"
    );
    assert_eq!(
        golden["seed"].as_u64(),
        Some(spec.seed),
        "the golden was blessed against different weights"
    );

    let tokens = tokens_of(conformance::golden::TOKENS);
    let rows = model_of(&spec).prefill_all(&tokens).unwrap();

    let recorded = golden["rows"]
        .as_array()
        .expect("the golden records no rows");
    assert_eq!(
        recorded.len(),
        conformance::golden::POSITIONS.len(),
        "the golden records {} rows, not the {} positions sampled",
        recorded.len(),
        conformance::golden::POSITIONS.len()
    );

    for entry in recorded {
        let p = entry["position"].as_u64().expect("row without a position") as usize;
        let want: Vec<f32> = entry["logits"]
            .as_array()
            .expect("row without logits")
            .iter()
            .map(|v| v.as_f64().expect("non-numeric logit") as f32)
            .collect();
        assert_eq!(
            want.len(),
            rows[p].0.len(),
            "position {p}: the golden is {} wide against the model's {}",
            want.len(),
            rows[p].0.len()
        );
        for (i, (got, expected)) in rows[p].0.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "position {p} logit {i}: {got:e} against the recorded {expected:e}"
            );
        }
    }
}

// Cache-lifecycle arms, at a length where the window has engaged.

/// The rewind target is asked for, not chosen.
///
/// This test used to name a literal 100, which was correct only for as long
/// as the cache retained everything. A windowed cache evicts, so how far back
/// it can go is a property of the run -- of the window, the batch width and
/// how much has been prefilled -- and a literal here would be a guess that
/// happened to hold at one geometry. `rewindable_from` exists so callers stop
/// guessing, and the first caller to use it should be the one demonstrating
/// why.
#[test]
fn truncating_past_the_window_and_reprefilling_matches_a_fresh_long_prefill() {
    let spec = shared_spec();
    let tokens = tokens_of(150);

    let mut m = model_of(&spec);
    m.prefill(&tokens[..130]).unwrap();
    let keep = m.rewindable_from();
    assert!(
        keep > 0,
        "a 130-token prefill against a window of {} should have evicted something; \
         this test is not exercising a capped cache",
        spec.sliding_window
    );
    m.truncate(keep).unwrap();
    let replayed = m.prefill(&tokens[keep..]).unwrap();

    let direct = model_of(&spec).prefill(&tokens).unwrap();
    assert_eq!(
        replayed,
        direct,
        "a cache rewound to {keep} and re-prefilled to {} differs from a fresh \
         prefill of the same tokens",
        tokens.len()
    );
}

#[test]
fn a_fork_taken_past_the_window_diverges_identically_to_a_fresh_session() {
    let spec = shared_spec();
    let tokens = tokens_of(150);
    let split = 100;

    let mut parent = model_of(&spec);
    parent.prefill(&tokens[..split]).unwrap();
    let mut child = parent.fork().unwrap();
    let forked = child.prefill(&tokens[split..]).unwrap();

    let direct = model_of(&spec).prefill(&tokens).unwrap();
    assert_eq!(
        forked,
        direct,
        "a fork taken at {split} and continued to {} differs from an unbroken run",
        tokens.len()
    );
    assert_eq!(parent.cached_len(), split, "the fork disturbed its parent");
}

// Storage arms. Everything above this line compares logits, and the storage
// cap must not change a logit -- so nothing above it can tell whether the cap
// is implemented at all. These assert the memory, which is the only place the
// package is visible.

/// A window of 4 against 400 tokens, so a cache that ignored the cap would be
/// a hundred times the size of one that honoured it.
fn narrow_spec() -> TinySpec {
    TinySpec {
        n_layers: 3,
        sliding_window_pattern: 4,
        sliding_window: 4,
        ..TinySpec::default()
    }
}

/// Without this the whole package could be a no-op and every other test in
/// the workspace would still pass.
///
/// The bound is `kv::window_capacity`, which is the same function the forward
/// pass sizes its buffers from -- so this does not independently re-derive
/// the arithmetic, and is not meant to. What it pins is that the cache obeys
/// *a* bound that does not grow with the sequence, checked at two lengths a
/// factor of four apart: a layer still accumulating would report different
/// numbers at 128 and at 512.
///
/// Both lengths are exact multiples of the batch, which matters. Capacity is
/// a function of the *group* being appended, so a run ending in a short
/// trailing group settles at a different figure -- correctly, but the
/// difference would then be the trailing group's width rather than anything
/// about accumulation, and the equality below would be asserting the wrong
/// thing.
#[test]
fn a_sliding_layer_stops_growing_once_the_window_is_full() {
    let spec = narrow_spec();
    let group = 64;
    let bound = runtime::kv::window_capacity(spec.sliding_window, group);

    let mut held = Vec::new();
    for len in [128, 512] {
        let mut m = model_of(&spec);
        m.set_batch(BatchKind::Batched(group));
        m.prefill(&tokens_of(len)).unwrap();
        assert_eq!(m.cached_len(), len);

        let rows = m.cache_extent(0).len();
        assert!(
            rows <= bound,
            "layer 0 holds {rows} rows of a {len}-token sequence, above the \
             capacity of {bound} for window {} at batch {group}",
            spec.sliding_window
        );
        assert!(
            rows < len,
            "layer 0 holds all {len} rows: the window is not capping storage"
        );
        held.push(rows);
    }
    assert_eq!(
        held[0], held[1],
        "a capped layer holds the same rows at 128 tokens and at 512; these \
         differ, so it is still growing with the sequence"
    );
}

/// The cap is not applied to layers that cannot afford it.
///
/// A global layer's `first` is zero at every position, so it reads its own
/// row 0 for as long as the run lasts. Evicting there would be a correctness
/// bug that the logit tests would catch -- but stating it here means the
/// memory arm and the correctness arm agree about which layers are which,
/// rather than one of them being right by luck.
#[test]
fn a_global_layer_still_holds_the_whole_sequence() {
    let spec = shared_spec();
    let len = 200;
    let mut m = model_of(&spec);
    m.prefill(&tokens_of(len)).unwrap();

    // Layer 1 is global and owns its cache; layer 0 is sliding and owns its
    // cache. See `shared_spec`.
    assert_eq!(m.cache_extent(1), 0..len, "the global layer evicted");
    assert!(
        m.cache_extent(0).start > 0,
        "the sliding layer did not evict, so this test compares two layers \
         that are behaving identically and proves nothing"
    );
}

/// A shared-KV layer stores nothing, capped or not.
#[test]
fn a_shared_layer_holds_no_rows_of_its_own() {
    let spec = shared_spec();
    let mut m = model_of(&spec);
    m.prefill(&tokens_of(200)).unwrap();
    // Layers 2 and 3 read donors. See `shared_spec`.
    for i in [2, 3] {
        assert_eq!(m.cache_extent(i), 0..0, "shared layer {i} stored rows");
    }
}

// Rewind arms. A windowed cache cannot return to a position it has evicted,
// and the point of these is that it says so rather than answering wrongly.

#[test]
fn a_rewind_to_the_floor_is_allowed_and_one_position_below_it_is_not() {
    let spec = narrow_spec();
    let mut m = model_of(&spec);
    m.prefill(&tokens_of(300)).unwrap();

    let floor = m.rewindable_from();
    assert!(
        floor > 0,
        "nothing was evicted, so there is no floor to test"
    );

    // At the floor, not merely above it. The margin here is zero by
    // construction after a single-token decode -- `Session::ensure_tail_logits`
    // rewinds exactly one position on every restore -- so an off-by-one in the
    // comparison would brick ordinary generation rather than an overreach.
    let mut at = model_of(&spec);
    at.prefill(&tokens_of(300)).unwrap();
    at.truncate(floor)
        .expect("the floor itself must be reachable");

    let err = m.truncate(floor - 1).unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains("sliding window") && text.contains("rewindable_from"),
        "the refusal must name the cause and the query that avoids it: {text}"
    );
    assert_eq!(
        m.cached_len(),
        300,
        "a refused truncate must leave the cache alone"
    );
}

/// The reset is always available, and it is a reset rather than a rewind.
///
/// Several callers depend on this — the ladder between rungs, perplexity
/// between windows, and `Session`'s restore-mismatch recovery, whose whole
/// purpose is to hand back a model with nothing in it. An error there would
/// leave exactly the skew the recovery exists to prevent.
#[test]
fn clearing_the_cache_is_allowed_from_below_the_floor() {
    let spec = narrow_spec();
    let tokens = tokens_of(300);

    let mut m = model_of(&spec);
    m.prefill(&tokens).unwrap();
    assert!(m.rewindable_from() > 0);
    m.truncate(0).expect("a full clear is always legal");

    assert_eq!(m.cached_len(), 0);
    assert_eq!(
        m.rewindable_from(),
        0,
        "a cleared cache still claims a floor"
    );
    for i in 0..spec.n_layers {
        assert_eq!(
            m.cache_extent(i),
            0..0,
            "layer {i} kept rows across a clear"
        );
    }
    // Not merely empty: computing like a model that never held anything.
    assert_eq!(
        m.prefill(&tokens).unwrap(),
        model_of(&spec).prefill(&tokens).unwrap(),
        "a cleared cache does not compute like a fresh one"
    );
}

/// A snapshot taken past the window restores to the same state, floor and all.
///
/// The payload carries each layer's `base` for this reason. A restore that
/// re-based to zero would look correct — the continuation logits would match,
/// since those only need the window — while claiming a rewind it cannot
/// perform, and the claim would only fail later, somewhere else.
#[test]
fn a_snapshot_taken_past_the_window_restores_its_rewind_floor() {
    let spec = narrow_spec();
    let tokens = tokens_of(300);

    let mut source = model_of(&spec);
    source.prefill(&tokens).unwrap();
    let floor = source.rewindable_from();
    assert!(
        floor > 0,
        "nothing was evicted, so the floor is trivially zero"
    );
    let extents: Vec<_> = (0..spec.n_layers).map(|i| source.cache_extent(i)).collect();

    let mut image = Vec::new();
    source.save_state(&mut image).unwrap();
    let mut restored = model_of(&spec);
    restored.load_state(&mut image.as_slice()).unwrap();

    assert_eq!(restored.cached_len(), tokens.len());
    assert_eq!(
        restored.rewindable_from(),
        floor,
        "the floor moved on restore"
    );
    for (i, want) in extents.iter().enumerate() {
        assert_eq!(
            &restored.cache_extent(i),
            want,
            "layer {i} restored elsewhere"
        );
    }
    assert_eq!(
        restored.prefill(&[7]).unwrap(),
        source.prefill(&[7]).unwrap(),
        "the restored cache does not continue bit-identically"
    );
}

/// A snapshot is untrusted input, and `base` reaches into the attention
/// indexing: `t - base` underflows for a base past the position being read,
/// which in release wraps to an enormous index and panics in a slice. Every
/// malformed base must be a refusal instead.
#[test]
fn a_snapshot_declaring_an_impossible_base_is_refused() {
    let spec = narrow_spec();
    let mut source = model_of(&spec);
    source.prefill(&tokens_of(300)).unwrap();
    let mut image = Vec::new();
    source.save_state(&mut image).unwrap();

    // Layer 0's `base` is the first u64 after the two header fields.
    let at = 16;
    for bad in [u64::MAX, 10_000, 299] {
        let mut corrupt = image.clone();
        corrupt[at..at + 8].copy_from_slice(&bad.to_le_bytes());
        let mut m = model_of(&spec);
        let err = m
            .load_state(&mut corrupt.as_slice())
            .expect_err("base {bad} was accepted");
        assert!(
            err.to_string().contains("layer 0"),
            "base {bad}: the refusal should name the layer: {err}"
        );
        assert_eq!(m.cached_len(), 0, "base {bad} left state behind");
    }
}
