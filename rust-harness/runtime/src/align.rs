/*
 * align.rs -- incremental detokenization with byte spans.
 *
 * Turns a stream of token ids into output text while recording, for each id,
 * the byte range of that text it produced. The spans are the join key between
 * a position in a run and the words a reader can see, which is what makes a
 * probe trace over tens of thousands of positions interpretable at all.
 */

use std::ops::Range;

use anyhow::Result;

use crate::model::Token;
use crate::tokenizer::Tokenizer;

/// Byte range in the accumulated output text, half-open as Rust ranges are.
///
/// A position that produced no visible text -- a sequence boundary, a piece
/// the checkpoint marks special -- carries an empty span rather than being
/// omitted. An omitted position would renumber every position after it, and
/// the artifact's whole value is that its indices are the run's indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }
}

/// Ids that may be withheld before a delta is released lossily.
///
/// A UTF-8 codepoint is at most four bytes, so at most four byte-fallback
/// pieces can be pending for a legitimate reason. The cap exists for the
/// illegitimate one: text containing a real U+FFFD would otherwise satisfy the
/// withholding rule forever and the stream would silently stop advancing.
/// Past the cap the bytes are released as they decode, replacement character
/// included, because an honest mojibake span beats a stalled artifact.
const MAX_PENDING: usize = 8;

/// Incremental detokenizer over one sequence.
///
/// Two properties are worth stating because both were tempting to get wrong.
///
/// **Deltas are decoded from the pending ids, not diffed against a decode of
/// the whole prefix.** Prefix-diffing is quadratic, and under this decoder
/// chain it is also incorrect: `Sequence[ByteFallback, Fuse, Replace]` fuses a
/// run of `<0xNN>` pieces before converting it, so `decode([t0..tn])` is not
/// in general `decode([t0..tn-1]) + decode([tn])`.
///
/// **Decoding only the pending window is nonetheless exact, and the
/// withholding rule is what makes it so.** The window is cut only where the
/// previous delta decoded to complete UTF-8, which is precisely where a
/// byte-fallback run has finished. Splitting the id stream anywhere else would
/// split such a run and change the bytes; splitting it there cannot.
pub struct Detok<'t> {
    tok: &'t Tokenizer,
    /// Ids pushed since the last released delta.
    pending: Vec<Token>,
    text: String,
}

impl<'t> Detok<'t> {
    pub fn new(tok: &'t Tokenizer) -> Detok<'t> {
        Detok {
            tok,
            pending: Vec::new(),
            text: String::new(),
        }
    }

    /// Accumulated output text across every released delta.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Add one token id to the stream. Nothing is released here; the caller
    /// decides that with `take_delta`.
    pub fn push(&mut self, id: Token) {
        self.pending.push(id);
    }

    /// The text decodable from the pending ids, or `None` while a multi-token
    /// codepoint is still incomplete.
    ///
    /// Does not advance the stream, so a caller may inspect a delta and decide
    /// not to consume it -- which is the shape a recorder wants, since it
    /// inspects far more often than it commits.
    ///
    /// `Some("")` and `None` are different answers and are kept distinct: the
    /// first is a position that produced no visible bytes, the second is a
    /// position whose bytes are not knowable yet.
    pub fn peek_delta(&self) -> Result<Option<String>> {
        if self.pending.is_empty() {
            return Ok(Some(String::new()));
        }
        let decoded = self.tok.decode(&self.pending)?;
        // The tokenizer's own decoder substitutes U+FFFD for a byte sequence
        // it cannot complete, so a trailing one means the codepoint continues
        // into an id that has not arrived. Only a trailing one: a replacement
        // character in the middle of the delta is settled text.
        if decoded.ends_with('\u{fffd}') && self.pending.len() < MAX_PENDING {
            return Ok(None);
        }
        Ok(Some(decoded))
    }

    /// `peek_delta`, then commit it: append the text and clear the pending
    /// ids, returning the span the delta occupies in `text()`.
    ///
    /// Returns `None` on the same condition `peek_delta` does, and leaves the
    /// stream untouched when it does -- the bytes of a multi-token codepoint
    /// are attributed to the position that completes it, because there is no
    /// smaller attribution available. The positions it spans release empty
    /// spans, so every position still has one and the round trip holds.
    pub fn take_delta(&mut self) -> Result<Option<Span>> {
        let Some(delta) = self.peek_delta()? else {
            return Ok(None);
        };
        let start = self.text.len();
        self.text.push_str(&delta);
        self.pending.clear();
        Ok(Some(Span {
            start,
            end: self.text.len(),
        }))
    }

    /// Push one id and take whatever it completes, which is the common case.
    ///
    /// The empty span returned while a codepoint is pending is a real answer:
    /// this position contributed no bytes of its own.
    pub fn advance(&mut self, id: Token) -> Result<Span> {
        self.push(id);
        let at = self.text.len();
        Ok(self.take_delta()?.unwrap_or(Span { start: at, end: at }))
    }
}

/// One forward pass over a contiguous run of positions.
///
/// The prefill/decode split alone stopped being enough when batching landed.
/// A reader given only that bit cannot tell whether position 4012 was computed
/// alongside 4011 or alone, and the two are different experimental conditions:
/// a batched position's row is produced in the same call as its neighbours'.
/// Recording the pass restores the distinction the batch width took away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pass {
    /// Ordinal within the run, counting every pass the model made.
    pub index: usize,
    /// First absolute cache position this pass computed.
    pub start: usize,
    pub len: usize,
}

impl Pass {
    /// Whether this pass covers `position`.
    pub fn covers(&self, position: usize) -> bool {
        position >= self.start && position < self.start + self.len
    }

    /// The phase name a pass is stored under.
    ///
    /// Derived rather than tracked, because decode reaches the backend as a
    /// prefill of one token and there is no second entry point to distinguish.
    /// A single-position pass is the model producing its own text; a wider one
    /// is the model reading text it was given.
    pub fn phase(&self) -> &'static str {
        if self.len == 1 {
            "decode"
        } else {
            "prefill"
        }
    }
}

/// What a prompt position is: structure the harness emitted, or content.
///
/// The distinction is the one IF-2 turns on. Scored decode positions are the
/// model producing its own text; prefill positions over content are the model
/// reading someone else's; and scaffolding is neither, producing no
/// user-visible text at all. Marked rather than omitted, always -- dropping a
/// position renumbers every position after it and the artifact's indices stop
/// being the run's indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Scaffold,
    Content,
}

impl Role {
    /// The name this role is stored under, and the parse that recovers it.
    /// One decision, so a stored role and its reader cannot drift apart.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::Scaffold => "scaffold",
            Role::Content => "content",
        }
    }
}

impl std::str::FromStr for Role {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Role> {
        match s {
            "scaffold" => Ok(Role::Scaffold),
            "content" => Ok(Role::Content),
            other => anyhow::bail!("unknown position role {other:?}"),
        }
    }
}

/// Byte ranges in `text` covered by any of `markers`.
///
/// The renderer emits its structure as known literals, so scanning for them
/// recovers exactly the bytes it owns. This is why roles are computed over
/// ranges rather than over ids: `<|turn>model\n` is one marker but two tokens,
/// only the first of which is allowlisted, and an id-based rule would call the
/// `model\n` half content.
///
/// Markers are matched longest first, so a marker that is a prefix of another
/// does not claim the shorter range and leave the tail looking like content.
pub fn scaffold_ranges(text: &str, markers: &[&str]) -> Vec<Range<usize>> {
    let mut sorted: Vec<&str> = markers.iter().copied().filter(|m| !m.is_empty()).collect();
    sorted.sort_unstable_by_key(|m| std::cmp::Reverse(m.len()));

    let mut out = Vec::new();
    let mut pos = 0;
    while pos < text.len() {
        let hit = sorted
            .iter()
            .find(|m| text.get(pos..pos + m.len()) == Some(**m));
        match hit {
            Some(m) => {
                out.push(pos..pos + m.len());
                pos += m.len();
            }
            None => pos += 1,
        }
    }
    out
}

/// The role of a token occupying `at`, given the scaffolding ranges.
///
/// A token overlapping any scaffold byte is scaffolding. Overlap rather than
/// containment because a tokenizer is free to merge across the boundary: a
/// token that is half structure is not content, and calling it content would
/// attribute the model's own bytes to text the harness wrote.
///
/// A zero-width range -- which the library can report for a piece the
/// normalizer consumed -- is classified by the byte it sits at.
pub fn role_at(at: &Range<usize>, scaffold: &[Range<usize>]) -> Role {
    let overlaps = scaffold
        .iter()
        .any(|s| at.start < s.end && s.start < at.end.max(at.start + 1));
    if overlaps {
        Role::Scaffold
    } else {
        Role::Content
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::merge_fixture_tokenizer_json;

    /// The merge fixture, whose decoder chain carries byte fallback and the
    /// space marker -- the two things these edge cases need. Its vocabulary is
    /// `a`, `b`, `c`, `ab`, `abc` over the 256 byte pieces, so anything else
    /// in a test string reaches the byte-fallback path on purpose.
    fn tokenizer() -> Tokenizer {
        Tokenizer::from_json(&merge_fixture_tokenizer_json().unwrap()).unwrap()
    }

    /// Every span, concatenated, is the text -- the round trip IF-2 asks for.
    fn assert_round_trip(text: &str, spans: &[Span]) {
        let mut joined = String::new();
        for s in spans {
            assert!(
                s.start <= s.end && s.end <= text.len(),
                "span {s:?} outside text of {} bytes",
                text.len()
            );
            joined.push_str(&text[s.start..s.end]);
        }
        assert_eq!(joined, text, "spans do not reproduce the text");
    }

    #[test]
    fn spans_concatenate_to_the_decoded_text() {
        let tok = tokenizer();
        let ids = tok.encode("abcabZ9!caf\u{00e9}").unwrap();
        let mut d = Detok::new(&tok);
        let spans: Vec<Span> = ids.iter().map(|&i| d.advance(i).unwrap()).collect();
        assert_eq!(d.text(), tok.decode(&ids).unwrap());
        assert_round_trip(d.text(), &spans);
    }

    #[test]
    fn a_multi_token_codepoint_is_withheld_then_released_whole() {
        let tok = tokenizer();
        // Outside the fixture's ASCII vocabulary, so it lands as byte-fallback
        // pieces: one codepoint spread across several ids.
        let ids = tok.encode("\u{4e16}").unwrap();
        assert!(
            ids.len() > 1,
            "expected byte fallback to split the codepoint, got {ids:?}"
        );
        let mut d = Detok::new(&tok);
        let spans: Vec<Span> = ids.iter().map(|&i| d.advance(i).unwrap()).collect();

        let (last, held) = spans.split_last().unwrap();
        for s in held {
            assert!(
                s.is_empty(),
                "byte {s:?} released before the codepoint completed"
            );
        }
        assert_eq!(&d.text()[last.start..last.end], "\u{4e16}");
        assert_round_trip(d.text(), &spans);
    }

    #[test]
    fn peek_delta_does_not_advance_and_take_delta_returns_the_same_bytes() {
        let tok = tokenizer();
        let ids = tok.encode("abc").unwrap();
        let mut d = Detok::new(&tok);
        d.push(ids[0]);

        let first = d.peek_delta().unwrap();
        let second = d.peek_delta().unwrap();
        assert_eq!(first, second, "peek is not idempotent");
        assert_eq!(d.text(), "", "peek advanced the stream");

        let span = d.take_delta().unwrap().expect("a complete piece releases");
        assert_eq!(Some(d.text()[span.start..span.end].to_string()), first);
    }

    #[test]
    fn a_position_producing_no_visible_text_gets_an_empty_span() {
        let tok = tokenizer();
        // `decode` renders sequence boundaries as nothing, by design.
        let mut d = Detok::new(&tok);
        let span = d.advance(tok.bos()).unwrap();
        assert!(span.is_empty(), "the boundary produced bytes: {span:?}");
        assert_eq!(d.text(), "");
    }

    #[test]
    fn a_marker_that_is_a_prefix_of_another_does_not_claim_the_shorter_range() {
        // `<|turn>model\n` contains `<|turn>`. Matching the short one first
        // would leave `model\n` outside the scaffolding and looking like the
        // model's own text.
        let text = "<|turn>model\nhi<turn|>";
        let got = scaffold_ranges(text, &["<|turn>", "<turn|>", "<|turn>model\n"]);
        let covered: Vec<&str> = got.iter().map(|r| &text[r.clone()]).collect();
        assert_eq!(covered, vec!["<|turn>model\n", "<turn|>"], "{got:?}");
    }

    #[test]
    fn a_token_overlapping_scaffolding_is_scaffolding() {
        // Two ranges, as a real render has: an opener and a closer.
        let scaffold = [10..20, 30..35];
        assert_eq!(role_at(&(0..5), &scaffold), Role::Content);
        assert_eq!(role_at(&(5..12), &scaffold), Role::Scaffold);
        assert_eq!(role_at(&(12..15), &scaffold), Role::Scaffold);
        assert_eq!(role_at(&(20..25), &scaffold), Role::Content);
        assert_eq!(role_at(&(28..32), &scaffold), Role::Scaffold);
        // Zero-width, which the library can report: classified by where it sits.
        assert_eq!(role_at(&(15..15), &scaffold), Role::Scaffold);
        assert_eq!(role_at(&(25..25), &scaffold), Role::Content);
    }

    #[test]
    fn a_role_round_trips_through_its_stored_name() {
        for role in [Role::Scaffold, Role::Content] {
            assert_eq!(role.as_str().parse::<Role>().unwrap(), role);
        }
    }

    #[test]
    fn withholding_stops_at_the_cap_rather_than_stalling_forever() {
        let tok = tokenizer();
        // A lead byte repeated never completes a codepoint. Without the cap
        // this stream would withhold every id it was ever given.
        let lead = tok.encode("\u{4e16}").unwrap()[0];
        let mut d = Detok::new(&tok);
        let released = (0..MAX_PENDING * 2)
            .filter(|_| !d.advance(lead).unwrap().is_empty())
            .count();
        assert!(
            released > 0,
            "the stream stalled: nothing was ever released"
        );
    }
}
