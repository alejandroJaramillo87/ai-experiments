/*
 * tokenizer — thin wrapper over the Hugging Face `tokenizers` crate.
 *
 * Segmentation is delegated; the trust boundary is not. `tokenizers` reads
 * tokenizer.json, which states its pipeline explicitly — model type,
 * normalizer, pre-tokenizer, decoder — instead of naming a convention the
 * reader has to guess at. This module keeps only what the library cannot
 * express: which special literals a given caller is allowed to resolve.
 *
 * That distinction is the whole reason the wrapper exists. `encode` treats its
 * input purely as data, and `encode_prompt` resolves exactly the literals the
 * caller names and nothing else, so a chat delimiter arriving from a file, a
 * command's output, or the model itself cannot open a turn. The library offers
 * only a global on/off switch over its added-tokens table, which cannot
 * express a per-call subset — so the scan below stays in this repository,
 * where it is tested.
 *
 * This replaces an in-tree SentencePiece unigram implementation that read the
 * vocabulary out of GGUF metadata. It was the wrong algorithm: Gemma 4 ships a
 * BPE vocabulary with a merge list, and unigram Viterbi over the same pieces
 * segments differently, so every prompt encoded against a real checkpoint was
 * wrong. The GGUF gave it a "gemma4" type tag over both a score table and a
 * merge list and the backend guessed. A tokenizer.json says "BPE".
 */

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};

use super::model::Token;

/// Chat template revision, matching `CHAT_TEMPLATE_VERSION` in
/// `training/models/families.py`.
///
/// Defined here rather than in `harness::template`, which renders with it and
/// re-exports it under its own name. Two consumers need it and they cannot see
/// each other: the renderer lives in `harness`, which depends on `runtime`, and
/// the ladder's L0 rung lives in `runtime`, where it checks that a parity
/// artifact's rendered-turn case came from the revision this harness actually
/// renders. Comparing a v2 artifact against a v3 renderer would measure a
/// string no prompt would ever contain, and report the result as a tokenizer
/// verdict.
///
/// The three tiers render one conversation to one byte string, so a change to
/// any of them bumps all of them; `harness/tests/template_contract.rs` pins the
/// Rust and Python sides together.
pub const CHAT_TEMPLATE_VERSION: u32 = 2;

/// The tokenizer that belongs to a checkpoint at `model`.
///
/// Resolution probes rather than assumes, in the shape `pyenv.rs` uses for the
/// same reason — the answer depends on the layout, and guessing wrong should
/// name what it looked for:
///
///   1. `explicit` — a caller that knows better, from `--tokenizer`.
///   2. `<model without extension>.tokenizer.json` — the fixture pair the
///      tiny-model bin writes.
///   3. `tokenizer.json` beside the model, which is how a published checkpoint
///      directory is laid out.
///
/// Candidate 3 is what will survive once the loader takes a directory rather
/// than a file; candidate 2 exists because the weights are still a single GGUF
/// while the tokenizer is already a sibling JSON.
pub fn resolve_tokenizer(model: &Path, explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        ensure!(
            path.is_file(),
            "--tokenizer {} does not exist",
            path.display()
        );
        return Ok(path.to_path_buf());
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    // A published checkpoint is a directory holding its own tokenizer.json,
    // and `--model` names that directory. Probed first, because a sibling
    // file of the same name would otherwise shadow the checkpoint's own.
    if model.is_dir() {
        candidates.push(model.join("tokenizer.json"));
    }
    candidates.push(model.with_extension("tokenizer.json"));
    if let Some(dir) = model.parent() {
        candidates.push(dir.join("tokenizer.json"));
    }
    candidates
        .iter()
        .find(|p| p.is_file())
        .cloned()
        .with_context(|| {
            let looked: Vec<String> = candidates.iter().map(|p| p.display().to_string()).collect();
            format!(
                "no tokenizer.json for model {}: looked in {}. \
                 Pass --tokenizer <path> to name one.",
                model.display(),
                looked.join(", ")
            )
        })
}

/// Pieces that name the sequence boundaries, most specific first.
///
/// The ids live in `tokenizer_config.json` in a published checkpoint, which
/// the synthetic fixtures do not carry, so they are resolved from the
/// vocabulary by name instead. Reading that file is the better long-term
/// source and belongs with the checkpoint loader; until then a missing
/// boundary is a load error naming the candidates rather than a silent zero,
/// because a BOS that defaults to token 0 is a defect that shows up only as
/// slightly worse output.
const BOS_CANDIDATES: &[&str] = &["<bos>", "<s>"];
const EOS_CANDIDATES: &[&str] = &["<eos>", "</s>"];

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    /// Piece text by id, materialized once at load.
    ///
    /// `id_to_token` allocates a `String` per call and the grammar vocabulary
    /// asks for every id on every construction, so the table is built once and
    /// `piece` borrows from it.
    pieces: Vec<String>,
    /// Piece text -> id for every entry, so the allowlist scan resolves a
    /// literal without asking the library to extract it.
    lookup: HashMap<String, Token>,
    bos: Token,
    eos: Token,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_len", &self.pieces.len())
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .finish_non_exhaustive()
    }
}

impl Tokenizer {
    /// Load a `tokenizer.json`.
    pub fn from_file(path: &Path) -> Result<Tokenizer> {
        let bytes =
            std::fs::read(path).with_context(|| format!("read tokenizer {}", path.display()))?;
        Self::from_json(&bytes).with_context(|| format!("load tokenizer {}", path.display()))
    }

    /// Load a `tokenizer.json` from bytes.
    pub fn from_json(bytes: &[u8]) -> Result<Tokenizer> {
        let mut inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("parse tokenizer.json: {e}"))?;

        // The safety wiring, and the one call in this file that must never be
        // removed. `encode_special_tokens(true)` means "split special tokens
        // when encoding", which is the library's way of saying: do not extract
        // them from the input as ids. The default is the opposite, and a
        // published Gemma tokenizer.json carries `<start_of_turn>` and friends
        // in its added-tokens table — so under the default, a turn marker
        // pasted into a file the harness reads would open a real turn. Every
        // path from a special literal to its id now runs through the caller's
        // explicit allowlist in `encode_prompt`.
        inner.set_encode_special_tokens(true);

        let vocab_len = inner.get_vocab_size(true);
        ensure!(vocab_len > 0, "empty tokenizer vocabulary");
        let pieces: Vec<String> = (0..vocab_len)
            .map(|id| {
                inner
                    .id_to_token(id as Token)
                    .with_context(|| format!("vocabulary has no piece for id {id}"))
            })
            .collect::<Result<_>>()?;
        let lookup: HashMap<String, Token> = pieces
            .iter()
            .enumerate()
            .map(|(id, p)| (p.clone(), id as Token))
            .collect();

        let resolve = |candidates: &[&str], label: &str| -> Result<Token> {
            candidates
                .iter()
                .find_map(|c| lookup.get(*c).copied())
                .with_context(|| {
                    format!("vocabulary declares no {label}: none of {candidates:?} is a piece")
                })
        };
        let bos = resolve(BOS_CANDIDATES, "beginning-of-sequence token")?;
        let eos = resolve(EOS_CANDIDATES, "end-of-sequence token")?;

        Ok(Tokenizer {
            inner,
            pieces,
            lookup,
            bos,
            eos,
        })
    }

    pub fn bos(&self) -> Token {
        self.bos
    }

    pub fn eos(&self) -> Token {
        self.eos
    }

    pub fn vocab_len(&self) -> usize {
        self.pieces.len()
    }

    /// Piece text for a token id, as stored in the vocab (space marker kept).
    /// Used to build the grammar `Vocab` view over the same table.
    pub fn piece(&self, tok: Token) -> Result<&str> {
        self.pieces
            .get(tok as usize)
            .map(String::as_str)
            .with_context(|| format!("token {tok} outside vocabulary"))
    }

    /// Decode token ids to text.
    ///
    /// The sequence boundaries render as nothing, and so does anything the
    /// checkpoint marks special: model output is displayed to a person, and a
    /// literal `<eos>` in it is noise rather than information. The
    /// tokenizer.json decoder handles the rest — byte-fallback reassembly and
    /// the space marker — which is why the emitted fixtures carry one.
    pub fn decode(&self, tokens: &[Token]) -> Result<String> {
        for &tok in tokens {
            ensure!(
                (tok as usize) < self.pieces.len(),
                "token {tok} outside vocabulary of {} tokens",
                self.pieces.len()
            );
        }
        // Filtered here as well as through `skip_special_tokens`, because a
        // vocabulary that carries its boundaries as ordinary pieces rather
        // than added tokens — as the synthetic fixtures do — is invisible to
        // the library's own filter.
        let kept: Vec<Token> = tokens
            .iter()
            .copied()
            .filter(|t| *t != self.bos && *t != self.eos)
            .collect();
        self.inner
            .decode(&kept, true)
            .map_err(|e| anyhow::anyhow!("decode {} tokens: {e}", kept.len()))
    }

    /// Encode text to token ids, treating the text purely as data: a special
    /// piece appearing in it is encoded as its literal characters, never as
    /// the special id.
    ///
    /// This is the safe default because most text reaching a tokenizer is
    /// untrusted — dataset lines, file contents, command output. Resolving
    /// specials here would let any of it emit a real turn or sequence
    /// boundary into a prompt. Callers that are themselves emitting structure
    /// use `encode_prompt` and name the literals they mean.
    /// No BOS is added — the caller (chat template rendering) owns it.
    pub fn encode(&self, text: &str) -> Result<Vec<Token>> {
        self.encode_inner(text, &[])
    }

    /// Encode text that the caller composed, resolving only the special
    /// literals in `allowed` to their trained ids; every other special piece
    /// stays literal text.
    ///
    /// The allowlist is the trust boundary. A rendered prompt interleaves
    /// structure the harness emits with content it does not control, and by
    /// the time it is one string the two are indistinguishable — so the
    /// harness names the delimiters it emits, and anything else that looks
    /// special is data by construction. Passing every special piece here
    /// would reopen exactly the injection the chat template's sanitizer
    /// exists to prevent.
    pub fn encode_prompt(&self, text: &str, allowed: &[&str]) -> Result<Vec<Token>> {
        self.encode_inner(text, allowed)
    }

    /// `encode_prompt`, with each token's byte range in `text` alongside it.
    ///
    /// The ranges are what lets a caller say which positions came from
    /// structure it emitted and which from content, without inspecting ids:
    /// the scaffolding is known as literals in the rendered string, so the
    /// question "is this position scaffolding" becomes a range intersection
    /// over bytes the renderer already knows about. An id-based rule gets
    /// `<|turn>model\n` wrong, since only the `<|turn>` prefix is allowlisted
    /// and `model\n` tokenizes as ordinary content.
    ///
    /// This is a second view of one encoding, not a second encoding —
    /// `encode_prompt` is defined in terms of it, and a test pins the ids
    /// equal. Widening the return type rather than adding a parallel path is
    /// the point: two encoders would eventually disagree.
    pub fn encode_prompt_offsets(
        &self,
        text: &str,
        allowed: &[&str],
    ) -> Result<Vec<(Token, Range<usize>)>> {
        self.encode_offsets(text, allowed)
    }

    /// Split the input on the allowed literals, hand each span between them to
    /// the library, and splice the literals' own ids back in.
    ///
    /// The library sees only spans that the caller has already vouched are
    /// content, so its added-tokens behaviour cannot widen what resolves.
    fn encode_inner(&self, text: &str, allowed: &[&str]) -> Result<Vec<Token>> {
        Ok(self
            .encode_offsets(text, allowed)?
            .into_iter()
            .map(|(tok, _)| tok)
            .collect())
    }

    /// The splice loop, keeping each token's byte range in `text`.
    ///
    /// Every caller goes through here so the id sequence has exactly one
    /// definition. A spliced literal reports its own range; a content token
    /// reports the library's offsets shifted into the whole string.
    fn encode_offsets(&self, text: &str, allowed: &[&str]) -> Result<Vec<(Token, Range<usize>)>> {
        // Longest first, so a literal that is a prefix of another does not win
        // over it: the model-turn opener is `<|turn>model\n` and the plain
        // turn marker `<|turn>` is a prefix of it.
        let mut allowed: Vec<&str> = allowed.iter().copied().filter(|a| !a.is_empty()).collect();
        allowed.sort_unstable_by_key(|a| std::cmp::Reverse(a.len()));

        let mut out = Vec::new();
        let mut span_start = 0;
        let mut pos = 0;
        while pos < text.len() {
            let Some((tok, len)) = self.special_at(text, pos, &allowed) else {
                pos += 1;
                continue;
            };
            self.encode_span(&text[span_start..pos], span_start, &mut out)?;
            out.push((tok, pos..pos + len));
            pos += len;
            span_start = pos;
        }
        self.encode_span(&text[span_start..], span_start, &mut out)?;
        Ok(out)
    }

    /// The longest allowed literal starting at byte `pos`, with its length.
    ///
    /// `allowed` is already sorted longest first. A literal that is not in the
    /// vocabulary matches nothing: the caller may name a delimiter this
    /// checkpoint does not carry, and encoding it as text is the same
    /// treatment any other unknown string gets.
    fn special_at(&self, text: &str, pos: usize, allowed: &[&str]) -> Option<(Token, usize)> {
        for literal in allowed {
            // `get` also rejects a span that is not on char boundaries.
            if text.get(pos..pos + literal.len()) != Some(*literal) {
                continue;
            }
            if let Some(&tok) = self.lookup.get(*literal) {
                return Some((tok, literal.len()));
            }
        }
        None
    }

    /// Encode one span of content text. Tokens are appended to `out`, each
    /// with its byte range shifted by `base` into the string the span came
    /// from.
    ///
    /// The library reports offsets against the input it was handed, and the
    /// normalizer that marks spaces is alignment-tracked, so these are ranges
    /// in the original text rather than in the normalized form.
    fn encode_span(
        &self,
        text: &str,
        base: usize,
        out: &mut Vec<(Token, Range<usize>)>,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode {} bytes: {e}", text.len()))?;
        for (&tok, &(start, end)) in encoding.get_ids().iter().zip(encoding.get_offsets()) {
            out.push((tok, base + start..base + end));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{merge_fixture_tokenizer_json, tokenizer_json_from_parts};

    const TURN_OPEN: &str = "<|turn>";
    const TURN_CLOSE: &str = "<turn|>";
    const MODEL_TURN_OPEN: &str = "<|turn>model\n";

    /// The merge fixture: `a`, `b`, `c`, `ab`, `abc` with merges a+b then ab+c,
    /// so BPE and a longest-match reading disagree.
    fn tokenizer() -> Tokenizer {
        Tokenizer::from_json(&merge_fixture_tokenizer_json().unwrap()).unwrap()
    }

    /// The same vocabulary plus the chat delimiters, one of which is a prefix
    /// of another.
    fn tokenizer_with_delimiters() -> Tokenizer {
        let mut pieces: Vec<String> = ["<pad>", "<s>", "</s>"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        pieces.extend((0..=255u8).map(|b| format!("<0x{b:02X}>")));
        pieces.extend(
            [
                "a",
                "b",
                "c",
                "ab",
                "abc",
                "\u{2581}",
                TURN_OPEN,
                TURN_CLOSE,
                MODEL_TURN_OPEN,
            ]
            .iter()
            .map(|s| s.to_string()),
        );
        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];
        let json = tokenizer_json_from_parts(&pieces, &merges, "<pad>").unwrap();
        Tokenizer::from_json(&json).unwrap()
    }

    #[test]
    fn segmentation_follows_the_merge_list() {
        let t = tokenizer();
        // a+b then ab+c reaches the single token `abc`.
        let ids = t.encode("abc").unwrap();
        assert_eq!(
            ids.len(),
            1,
            "expected one merged token, got {ids:?} = {:?}",
            ids.iter().map(|i| t.piece(*i).unwrap()).collect::<Vec<_>>()
        );
        assert_eq!(t.piece(ids[0]).unwrap(), "abc");

        // `cab` cannot reach `abc`: merges apply within the word, left to
        // right, so `c` stays alone and `a`+`b` merges.
        let ids = t.encode("cab").unwrap();
        let got: Vec<&str> = ids.iter().map(|i| t.piece(*i).unwrap()).collect();
        assert_eq!(got, vec!["c", "ab"], "ids {ids:?}");
    }

    #[test]
    fn multibyte_chars_round_trip_through_byte_fallback() {
        let t = tokenizer();
        for text in ["caf\u{00e9}", "\u{4e2d}\u{6587}", "Z9!"] {
            let ids = t.encode(text).unwrap();
            let back = t.decode(&ids).unwrap();
            assert_eq!(
                back, text,
                "round trip of {text:?} gave {back:?} via {ids:?}"
            );
        }
    }

    #[test]
    fn empty_text_encodes_to_no_tokens() {
        assert!(tokenizer().encode("").unwrap().is_empty());
    }

    #[test]
    fn sequence_boundaries_decode_to_nothing() {
        let t = tokenizer();
        let ids = vec![t.bos(), t.eos()];
        assert_eq!(t.decode(&ids).unwrap(), "");
    }

    #[test]
    fn decode_of_out_of_range_token_is_an_error() {
        let t = tokenizer();
        let out_of_range = t.vocab_len() as Token;
        let err = t.decode(&[out_of_range]).unwrap_err().to_string();
        assert!(
            err.contains(&out_of_range.to_string()),
            "error does not name the token: {err}"
        );
    }

    #[test]
    fn a_control_piece_encodes_to_its_own_id_when_allowed() {
        let t = tokenizer_with_delimiters();
        let ids = t.encode_prompt(TURN_OPEN, &[TURN_OPEN]).unwrap();
        assert_eq!(ids.len(), 1, "expected one token, got {ids:?}");
        assert_eq!(t.piece(ids[0]).unwrap(), TURN_OPEN);
    }

    #[test]
    fn text_around_a_control_piece_still_segments_normally() {
        let t = tokenizer_with_delimiters();
        let bare = t.encode("abc").unwrap();
        let wrapped = t
            .encode_prompt(
                &format!("{TURN_OPEN}abc{TURN_CLOSE}"),
                &[TURN_OPEN, TURN_CLOSE],
            )
            .unwrap();
        assert_eq!(
            wrapped.len(),
            bare.len() + 2,
            "content re-segmented around the delimiters: {wrapped:?} vs {bare:?}"
        );
        assert_eq!(&wrapped[1..wrapped.len() - 1], bare.as_slice());
    }

    #[test]
    fn longest_special_piece_wins_when_one_is_a_prefix_of_another() {
        let t = tokenizer_with_delimiters();
        let ids = t
            .encode_prompt(MODEL_TURN_OPEN, &[TURN_OPEN, MODEL_TURN_OPEN])
            .unwrap();
        assert_eq!(ids.len(), 1, "expected the longer literal, got {ids:?}");
        assert_eq!(t.piece(ids[0]).unwrap(), MODEL_TURN_OPEN);
    }

    /// The injection case. A delimiter arriving inside content the harness did
    /// not write must stay text, whatever else the caller allowed.
    #[test]
    fn a_special_piece_outside_the_allowlist_stays_literal_text() {
        let t = tokenizer_with_delimiters();
        let listed = t.encode_prompt(TURN_CLOSE, &[TURN_CLOSE]).unwrap();
        let unlisted = t.encode_prompt(TURN_CLOSE, &[TURN_OPEN]).unwrap();
        assert_eq!(listed.len(), 1, "listed: {listed:?}");
        assert!(
            unlisted.len() > 1,
            "an unlisted delimiter resolved to a single id: {unlisted:?}"
        );
        assert!(
            !unlisted.contains(&listed[0]),
            "an unlisted delimiter still produced its special id: {unlisted:?}"
        );
    }

    /// The offsets view must be a view, not a second encoding. If these ever
    /// disagree, one of the two callers is prompting a different model.
    #[test]
    fn the_offsets_view_yields_the_same_ids_as_encode_prompt() {
        let t = tokenizer_with_delimiters();
        let text = format!("{TURN_OPEN}abc{TURN_CLOSE}\nab caf\u{00e9}");
        let allowed = [TURN_OPEN, TURN_CLOSE, MODEL_TURN_OPEN];
        let plain = t.encode_prompt(&text, &allowed).unwrap();
        let with_offsets = t.encode_prompt_offsets(&text, &allowed).unwrap();
        let ids: Vec<Token> = with_offsets.iter().map(|(tok, _)| *tok).collect();
        assert_eq!(ids, plain, "offsets view re-segmented the input");
    }

    #[test]
    fn every_token_offset_is_a_range_inside_the_input() {
        let t = tokenizer_with_delimiters();
        let text = format!("{TURN_OPEN}abc{TURN_CLOSE} ab");
        let got = t
            .encode_prompt_offsets(&text, &[TURN_OPEN, TURN_CLOSE])
            .unwrap();
        for (tok, range) in &got {
            assert!(
                range.start <= range.end && range.end <= text.len(),
                "token {tok} reports {range:?} outside {} bytes",
                text.len()
            );
            assert!(
                text.get(range.clone()).is_some(),
                "token {tok} reports {range:?}, which is not on char boundaries"
            );
        }
        // The spliced literals report their own byte ranges, which is what the
        // scaffolding classification reads.
        let opener = got.first().unwrap();
        assert_eq!(&text[opener.1.clone()], TURN_OPEN, "{:?}", opener.1);
    }

    #[test]
    fn plain_encode_never_resolves_a_special_piece() {
        let t = tokenizer_with_delimiters();
        let special = *t.lookup.get(TURN_OPEN).unwrap();
        let ids = t.encode(TURN_OPEN).unwrap();
        assert!(
            !ids.contains(&special),
            "encode resolved a special piece: {ids:?} contains {special}"
        );
    }

    /// The library's own added-token extraction must stay off. Without
    /// `set_encode_special_tokens(true)` this is the case that regresses: a
    /// checkpoint whose delimiters are added tokens would resolve them from
    /// untrusted input regardless of the allowlist.
    #[test]
    fn added_token_extraction_is_disabled() {
        assert!(
            tokenizer().inner.get_encode_special_tokens(),
            "special-token extraction is enabled; untrusted input can emit turn boundaries"
        );
    }

    #[test]
    fn a_vocabulary_without_the_boundaries_is_a_load_error() {
        let pieces: Vec<String> = ["<pad>", "a", "b"].iter().map(|s| s.to_string()).collect();
        let json = tokenizer_json_from_parts(&pieces, &[], "<pad>").unwrap();
        let err = Tokenizer::from_json(&json).unwrap_err().to_string();
        assert!(
            err.contains("beginning-of-sequence"),
            "error does not name the missing boundary: {err}"
        );
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(Tokenizer::from_json(b"{\"model\": ").is_err());
        assert!(Tokenizer::from_json(b"").is_err());
    }

    /// A delimiter the checkpoint does not carry is encoded as text rather
    /// than failing: the allowlist names what the caller means to emit, and a
    /// vocabulary is free not to have it.
    #[test]
    fn an_allowed_literal_absent_from_the_vocabulary_stays_text() {
        let t = tokenizer();
        let ids = t.encode_prompt("abc", &["<|turn>"]).unwrap();
        assert_eq!(ids, t.encode("abc").unwrap(), "ids {ids:?}");
    }
}
