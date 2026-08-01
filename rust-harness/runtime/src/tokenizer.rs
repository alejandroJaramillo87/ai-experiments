/*
 * tokenizer — SentencePiece unigram tokenizer over the GGUF-embedded vocab.
 *
 * Gemma checkpoints ship the vocabulary inside the GGUF metadata
 * (tokenizer.ggml.tokens / .scores / .token_type plus bos/eos ids). Encoding
 * matches control and user-defined pieces (the chat delimiters) as literals
 * first, then runs unigram Viterbi over the score table with byte fallback on
 * the text between them; decoding maps pieces back to text with the
 * SentencePiece space marker (U+2581) restored to a plain space. No
 * dummy-prefix space is inserted: the chat template controls surrounding
 * whitespace exactly, and the Tier 3 template-golden test verifies the
 * end-to-end token stream against the Hugging Face tokenizer.
 */

use std::collections::HashMap;

use anyhow::{bail, ensure, Context, Result};

use super::gguf::{GgufFile, MetaValue};
use super::model::Token;

/// SentencePiece space marker.
const SPACE_PIECE: char = '\u{2581}';

/// Score assigned to a byte-fallback step: below any real piece so Viterbi
/// prefers whole pieces, matching SentencePiece's unk penalty ordering.
const BYTE_FALLBACK_SCORE: f32 = -1.0e6;

/// Upper bound on the Viterbi match window, in bytes.
///
/// The window is the longest `Normal` piece, and the span loop hashes every
/// substring inside it, so encoding costs O(text * window^2). Taken verbatim
/// from the vocabulary the window is unbounded, and one oversized piece in an
/// otherwise ordinary export pushes a few kilobytes of prompt into seconds —
/// on a path every prompt takes.
///
/// SentencePiece trains pieces of at most `max_sentencepiece_length`
/// characters (default 16) and a character is at most four UTF-8 bytes, so 64
/// bytes covers anything a standard trainer can emit; this doubles that for
/// headroom. A piece longer than the cap is simply never matched: the
/// segmentation stays the best one over the pieces that fit, with byte
/// fallback covering whatever they miss.
const MAX_PIECE_BYTES: usize = 128;

/// Token roles from tokenizer.ggml.token_type (llama.cpp convention).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte,
    Other,
}

impl Kind {
    fn from_i32(v: i32) -> Kind {
        match v {
            1 => Kind::Normal,
            2 => Kind::Unknown,
            3 => Kind::Control,
            4 => Kind::UserDefined,
            6 => Kind::Byte,
            _ => Kind::Other,
        }
    }
}

#[derive(Debug)]
pub struct Tokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    kinds: Vec<Kind>,
    // Piece text -> id, for Viterbi substring lookup. Normal pieces only.
    lookup: HashMap<String, Token>,
    // Control and user-defined piece text -> id. Matched as literals before
    // Viterbi runs, so a chat delimiter costs one token instead of a byte run
    // (Hugging Face behavior with split_special_tokens=false).
    specials: HashMap<String, Token>,
    // Distinct special piece lengths, descending: the scan tries them in that
    // order so the longest literal wins where one is a prefix of another.
    special_lens: Vec<usize>,
    // Set for each byte that starts some special piece, so the scan rejects
    // most input positions with one array probe instead of a map lookup.
    special_starts: [bool; 256],
    // Byte value -> byte-token id, for fallback encoding.
    byte_tokens: [Option<Token>; 256],
    // Viterbi match window: the longest Normal piece, capped at
    // MAX_PIECE_BYTES so one oversized piece cannot make encoding crawl.
    max_piece_bytes: usize,
    bos: Token,
    eos: Token,
}

impl Tokenizer {
    /// Build from parsed GGUF metadata. Errors name the missing or
    /// inconsistent key so a truncated export fails loudly at load time.
    pub fn from_gguf(file: &GgufFile) -> Result<Tokenizer> {
        let pieces = match file.metadata.get("tokenizer.ggml.tokens") {
            Some(MetaValue::ArrStr(v)) => v.clone(),
            Some(_) => bail!("tokenizer.ggml.tokens has wrong type"),
            None => bail!("missing tokenizer.ggml.tokens"),
        };
        let scores = match file.metadata.get("tokenizer.ggml.scores") {
            Some(MetaValue::ArrF32(v)) => v.clone(),
            Some(_) => bail!("tokenizer.ggml.scores has wrong type"),
            None => bail!("missing tokenizer.ggml.scores"),
        };
        let kinds: Vec<Kind> = match file.metadata.get("tokenizer.ggml.token_type") {
            Some(MetaValue::ArrI32(v)) => v.iter().map(|&t| Kind::from_i32(t)).collect(),
            Some(_) => bail!("tokenizer.ggml.token_type has wrong type"),
            None => bail!("missing tokenizer.ggml.token_type"),
        };
        ensure!(
            pieces.len() == scores.len() && pieces.len() == kinds.len(),
            "tokenizer arrays disagree: {} tokens, {} scores, {} types",
            pieces.len(),
            scores.len(),
            kinds.len()
        );
        ensure!(!pieces.is_empty(), "empty tokenizer vocabulary");
        let id_key = |key: &str| -> Result<Token> {
            match file.metadata.get(key) {
                Some(MetaValue::U32(v)) => Ok(*v),
                Some(_) => bail!("{key} has wrong type"),
                None => bail!("missing {key}"),
            }
        };
        let bos = id_key("tokenizer.ggml.bos_token_id")?;
        let eos = id_key("tokenizer.ggml.eos_token_id")?;
        ensure!(
            (bos as usize) < pieces.len() && (eos as usize) < pieces.len(),
            "bos/eos id outside vocabulary of {} tokens",
            pieces.len()
        );
        Self::new(pieces, scores, kinds, bos, eos)
    }

    /// Build from raw vocab arrays with llama.cpp token-type codes. Public
    /// for tests and tools that hold the arrays without a GGUF container.
    pub fn from_parts(
        pieces: Vec<String>,
        scores: Vec<f32>,
        token_types: &[i32],
        bos: Token,
        eos: Token,
    ) -> Result<Tokenizer> {
        ensure!(
            pieces.len() == scores.len() && pieces.len() == token_types.len(),
            "tokenizer arrays disagree: {} tokens, {} scores, {} types",
            pieces.len(),
            scores.len(),
            token_types.len()
        );
        let kinds = token_types.iter().map(|&t| Kind::from_i32(t)).collect();
        Self::new(pieces, scores, kinds, bos, eos)
    }

    fn new(
        pieces: Vec<String>,
        scores: Vec<f32>,
        kinds: Vec<Kind>,
        bos: Token,
        eos: Token,
    ) -> Result<Tokenizer> {
        let mut lookup = HashMap::new();
        let mut specials = HashMap::new();
        let mut special_starts = [false; 256];
        let mut byte_tokens = [None; 256];
        let mut max_piece_bytes = 1;
        for (id, (piece, kind)) in pieces.iter().zip(&kinds).enumerate() {
            match kind {
                Kind::Normal => {
                    max_piece_bytes = max_piece_bytes.max(piece.len());
                    // First occurrence wins on duplicate piece text.
                    lookup.entry(piece.clone()).or_insert(id as Token);
                }
                Kind::Control | Kind::UserDefined => {
                    // An empty piece would match at every position without
                    // consuming input, so it stays out of the scan table.
                    if let Some(&first) = piece.as_bytes().first() {
                        special_starts[first as usize] = true;
                        specials.entry(piece.clone()).or_insert(id as Token);
                    }
                }
                Kind::Byte => {
                    let b = parse_byte_piece(piece)
                        .with_context(|| format!("bad byte token {id}: {piece:?}"))?;
                    byte_tokens[b as usize] = Some(id as Token);
                }
                _ => {}
            }
        }
        // A piece past the cap is unreachable rather than mis-matched: see
        // MAX_PIECE_BYTES for why bounding the window is safe here.
        let max_piece_bytes = max_piece_bytes.min(MAX_PIECE_BYTES);
        // Lengths are collected once here so encode never walks the vocabulary.
        let mut special_lens: Vec<usize> = specials.keys().map(String::len).collect();
        special_lens.sort_unstable_by(|a, b| b.cmp(a));
        special_lens.dedup();
        Ok(Tokenizer {
            pieces,
            scores,
            kinds,
            lookup,
            specials,
            special_lens,
            special_starts,
            byte_tokens,
            max_piece_bytes,
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

    /// Decode token ids to text. Control tokens render as nothing;
    /// user-defined tokens render as their piece text; byte
    /// tokens contribute raw bytes, assembled with invalid sequences replaced
    /// (never an error: model output must always be displayable).
    pub fn decode(&self, tokens: &[Token]) -> Result<String> {
        let mut bytes: Vec<u8> = Vec::new();
        for &tok in tokens {
            let i = tok as usize;
            ensure!(
                i < self.pieces.len(),
                "token {tok} outside vocabulary of {} tokens",
                self.pieces.len()
            );
            match self.kinds[i] {
                Kind::Control => {}
                Kind::Byte => bytes.push(parse_byte_piece(&self.pieces[i])?),
                _ => bytes.extend_from_slice(self.pieces[i].as_bytes()),
            }
        }
        // Space markers are restored after byte assembly so markers encoded
        // through byte-fallback tokens (their UTF-8 bytes split across byte
        // tokens) become spaces too, keeping encode/decode inverse.
        Ok(String::from_utf8_lossy(&bytes)
            .replace(SPACE_PIECE, " ")
            .to_string())
    }

    /// Encode text to token ids, treating the text purely as data: a control
    /// or user-defined piece appearing in it is encoded as its literal
    /// characters, never as the special id.
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
    ///
    /// Literals containing a space will not match, because the input is
    /// space-marked before scanning; the Gemma delimiters contain none.
    pub fn encode_prompt(&self, text: &str, allowed: &[&str]) -> Result<Vec<Token>> {
        self.encode_inner(text, allowed)
    }

    fn encode_inner(&self, text: &str, allowed: &[&str]) -> Result<Vec<Token>> {
        let marked: String = text
            .chars()
            .map(|c| if c == ' ' { SPACE_PIECE } else { c })
            .collect();
        let mut out = Vec::new();
        let mut span_start = 0;
        let mut pos = 0;
        while pos < marked.len() {
            let Some((tok, len)) = self.special_at(&marked, pos, allowed) else {
                pos += 1;
                continue;
            };
            self.encode_span(&marked[span_start..pos], &mut out)?;
            out.push(tok);
            pos += len;
            span_start = pos;
        }
        self.encode_span(&marked[span_start..], &mut out)?;
        Ok(out)
    }

    /// Longest allowed control or user-defined piece starting at byte `pos`,
    /// with its byte length. Cost is bounded by the number of distinct
    /// special lengths rather than by vocabulary size: this runs at every
    /// byte of every prompt, and returns immediately when nothing is allowed.
    fn special_at(&self, marked: &str, pos: usize, allowed: &[&str]) -> Option<(Token, usize)> {
        if allowed.is_empty() || !self.special_starts[marked.as_bytes()[pos] as usize] {
            return None;
        }
        for &len in &self.special_lens {
            // get() also rejects a span that is not on char boundaries.
            let Some(sub) = marked.get(pos..pos + len) else {
                continue;
            };
            if !allowed.contains(&sub) {
                continue;
            }
            if let Some(&tok) = self.specials.get(sub) {
                return Some((tok, len));
            }
        }
        None
    }

    /// Unigram Viterbi over one span of space-marked text: best-scoring
    /// segmentation into vocabulary pieces, with per-byte fallback tokens for
    /// anything the vocabulary cannot cover. Tokens are appended to `out`.
    fn encode_span(&self, marked: &str, out: &mut Vec<Token>) -> Result<()> {
        let bytes = marked.as_bytes();
        if bytes.is_empty() {
            return Ok(());
        }
        // best[i]: (score, token ending at i, start of that token).
        let mut best: Vec<Option<(f32, Token, usize)>> = vec![None; bytes.len() + 1];
        best[0] = Some((0.0, 0, 0));
        for start in 0..bytes.len() {
            let Some((base, _, _)) = best[start] else {
                continue;
            };
            let mut relax = |end: usize, score: f32, tok: Token| {
                let cand = base + score;
                if best[end].is_none_or(|(s, _, _)| cand > s) {
                    best[end] = Some((cand, tok, start));
                }
            };
            let max_end = bytes.len().min(start + self.max_piece_bytes);
            for end in start + 1..=max_end {
                // Piece boundaries must be UTF-8 boundaries.
                let Some(sub) = marked.get(start..end) else {
                    continue;
                };
                if let Some(&tok) = self.lookup.get(sub) {
                    relax(end, self.scores[tok as usize], tok);
                }
            }
            if let Some(tok) = self.byte_tokens[bytes[start] as usize] {
                relax(start + 1, BYTE_FALLBACK_SCORE, tok);
            }
        }
        // Walk back from the end. A gap means a byte had neither piece nor
        // byte-token coverage.
        let first = out.len();
        let mut pos = bytes.len();
        while pos > 0 {
            let (_, tok, start) = best[pos].with_context(|| {
                format!("no tokenization covers byte {} of the input span", pos - 1)
            })?;
            out.push(tok);
            pos = start;
        }
        out[first..].reverse();
        Ok(())
    }
}

/// Parse a `<0xNN>` byte-token piece to its byte value.
fn parse_byte_piece(piece: &str) -> Result<u8> {
    let hex = piece
        .strip_prefix("<0x")
        .and_then(|s| s.strip_suffix('>'))
        .with_context(|| format!("byte token not of form <0xNN>: {piece:?}"))?;
    u8::from_str_radix(hex, 16).with_context(|| format!("bad hex in byte token {piece:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built vocab: control specials, byte tokens for every byte, and a
    /// few normal pieces with scores favoring longer matches. Types are the
    /// llama.cpp token_type codes a real Gemma export carries.
    fn vocab_parts() -> (Vec<String>, Vec<f32>, Vec<i32>) {
        let mut pieces = vec![
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
        ];
        let mut types = vec![3i32; 3];
        for b in 0..=255u8 {
            pieces.push(format!("<0x{b:02X}>"));
            types.push(6);
        }
        let normals = [
            ("\u{2581}hello", -1.0),
            ("\u{2581}world", -1.0),
            ("hell", -2.0),
            ("o", -3.0),
            ("\u{2581}", -4.0),
            ("h", -5.0),
            ("e", -5.0),
            ("l", -5.0),
            ("wor", -2.5),
            ("ld", -2.5),
            ("caf\u{00e9}", -1.5),
        ];
        for (p, _) in normals {
            pieces.push(p.to_string());
            types.push(1);
        }
        let mut scores = vec![0.0; 3 + 256];
        scores.extend(normals.iter().map(|(_, s)| *s));
        (pieces, scores, types)
    }

    fn tokenizer() -> Tokenizer {
        let (pieces, scores, types) = vocab_parts();
        Tokenizer::from_parts(pieces, scores, &types, 1, 2).unwrap()
    }

    /// The same vocab plus the chat delimiters: the turn markers as control
    /// pieces and the full model-turn opener as a user-defined piece, so one
    /// special is a prefix of another.
    fn tokenizer_with_delimiters() -> Tokenizer {
        let (mut pieces, mut scores, mut types) = vocab_parts();
        for (piece, token_type) in [
            ("<start_of_turn>", 3),
            ("<end_of_turn>", 3),
            ("<start_of_turn>model\n", 4),
        ] {
            pieces.push(piece.to_string());
            scores.push(0.0);
            types.push(token_type);
        }
        Tokenizer::from_parts(pieces, scores, &types, 1, 2).unwrap()
    }

    /// The same vocab plus one Normal piece far longer than any trainer
    /// emits — a corrupt or hand-built export, which is enough to widen the
    /// Viterbi window for every prompt.
    fn tokenizer_with_oversized_piece(piece_bytes: usize) -> Tokenizer {
        let (mut pieces, mut scores, mut types) = vocab_parts();
        pieces.push("q".repeat(piece_bytes));
        scores.push(-0.5);
        types.push(1);
        Tokenizer::from_parts(pieces, scores, &types, 1, 2).unwrap()
    }

    /// The same vocab with no control or user-defined piece at all: type 5 is
    /// llama.cpp's UNUSED, present in the table but never matched.
    fn tokenizer_without_specials() -> Tokenizer {
        let (pieces, scores, types) = vocab_parts();
        let types: Vec<i32> = types.iter().map(|&t| if t == 3 { 5 } else { t }).collect();
        Tokenizer::from_parts(pieces, scores, &types, 1, 2).unwrap()
    }

    fn id(t: &Tokenizer, piece: &str) -> Token {
        t.lookup[piece]
    }

    /// Id of a control or user-defined piece: these are deliberately absent
    /// from the Viterbi lookup, so find them by position in the vocab.
    fn special_id(t: &Tokenizer, piece: &str) -> Token {
        t.pieces.iter().position(|p| p == piece).unwrap() as Token
    }

    #[test]
    fn encode_picks_the_best_scoring_segmentation() {
        let t = tokenizer();
        // " hello" as one piece (-1.0) beats "▁"+"hell"+"o" (-9.0).
        assert_eq!(t.encode(" hello").unwrap(), vec![id(&t, "\u{2581}hello")]);
        // " hello world" segments into the two word pieces.
        assert_eq!(
            t.encode(" hello world").unwrap(),
            vec![id(&t, "\u{2581}hello"), id(&t, "\u{2581}world")]
        );
    }

    #[test]
    fn encode_decode_round_trip() {
        let t = tokenizer();
        for text in [" hello world", "hello", " hell o world", "caf\u{00e9}"] {
            let ids = t.encode(text).unwrap();
            assert_eq!(t.decode(&ids).unwrap(), text, "{text:?}");
        }
    }

    #[test]
    fn unknown_chars_fall_back_to_byte_tokens() {
        let t = tokenizer();
        // 'Z' is not in any normal piece; must use its byte token.
        let ids = t.encode("Z").unwrap();
        assert_eq!(ids, vec![t.byte_tokens[b'Z' as usize].unwrap()]);
        assert_eq!(t.decode(&ids).unwrap(), "Z");
    }

    #[test]
    fn multibyte_chars_round_trip_through_byte_fallback() {
        let t = tokenizer();
        // U+00F1 is not a piece: two UTF-8 bytes, two byte tokens.
        let ids = t.encode("\u{00f1}").unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(t.decode(&ids).unwrap(), "\u{00f1}");
    }

    #[test]
    fn pieces_beat_byte_fallback_when_available() {
        let t = tokenizer();
        // "café" exists as a single piece; byte fallback must not win.
        assert_eq!(
            t.encode("caf\u{00e9}").unwrap(),
            vec![id(&t, "caf\u{00e9}")]
        );
    }

    #[test]
    fn control_tokens_decode_to_nothing() {
        let t = tokenizer();
        let hello = id(&t, "\u{2581}hello");
        assert_eq!(t.decode(&[t.bos(), hello, t.eos()]).unwrap(), " hello");
    }

    /// The delimiters a caller composing a prompt would name as structural.
    const ALLOWED: &[&str] = &["<start_of_turn>", "<end_of_turn>", "<start_of_turn>model\n"];

    #[test]
    fn control_pieces_encode_to_their_own_id_not_a_byte_run() {
        let t = tokenizer_with_delimiters();
        assert_eq!(
            t.encode_prompt("<start_of_turn>", ALLOWED).unwrap(),
            vec![special_id(&t, "<start_of_turn>")]
        );
        // Without the piece in the vocab the same literal costs 15 tokens.
        assert_eq!(
            tokenizer()
                .encode_prompt("<start_of_turn>", ALLOWED)
                .unwrap()
                .len(),
            15
        );
    }

    #[test]
    fn text_around_a_control_piece_still_segments_normally() {
        let t = tokenizer_with_delimiters();
        assert_eq!(
            t.encode_prompt("<start_of_turn> hello world<end_of_turn>", ALLOWED)
                .unwrap(),
            vec![
                special_id(&t, "<start_of_turn>"),
                id(&t, "\u{2581}hello"),
                id(&t, "\u{2581}world"),
                special_id(&t, "<end_of_turn>"),
            ]
        );
    }

    #[test]
    fn longest_special_piece_wins_when_one_is_a_prefix_of_another() {
        let t = tokenizer_with_delimiters();
        assert_eq!(
            t.encode_prompt("<start_of_turn>model\n", ALLOWED).unwrap(),
            vec![special_id(&t, "<start_of_turn>model\n")]
        );
        // The shorter piece still applies where the longer cannot match.
        assert_eq!(
            t.encode_prompt("<start_of_turn>user", ALLOWED).unwrap()[0],
            special_id(&t, "<start_of_turn>")
        );
    }

    #[test]
    fn a_special_piece_outside_the_allowlist_stays_literal_text() {
        // The injection guard. Untrusted content reaching a prompt must not
        // be able to emit a boundary the harness did not write, so a special
        // the caller did not name is encoded as its characters.
        let t = tokenizer_with_delimiters();
        let eos_literal = "<end_of_turn>";
        let allowed_only_open: &[&str] = &["<start_of_turn>"];

        let got = t.encode_prompt(eos_literal, allowed_only_open).unwrap();
        assert!(
            !got.contains(&special_id(&t, eos_literal)),
            "an unlisted special resolved to its id: {got:?}"
        );
        assert_eq!(got, t.encode(eos_literal).unwrap());
    }

    #[test]
    fn plain_encode_never_resolves_a_special_piece() {
        // encode() is the seam every untrusted caller uses — dataset lines,
        // file contents, command output. It must treat text as data.
        let t = tokenizer_with_delimiters();
        for literal in ["<start_of_turn>", "<end_of_turn>", "<start_of_turn>model\n"] {
            let got = t.encode(literal).unwrap();
            assert!(
                !got.contains(&special_id(&t, literal)),
                "encode resolved {literal} to its special id: {got:?}"
            );
            assert!(got.len() > 1, "{literal} collapsed to one token");
        }
    }

    #[test]
    fn user_defined_pieces_decode_to_their_text_and_control_pieces_to_nothing() {
        let t = tokenizer_with_delimiters();
        let opener = special_id(&t, "<start_of_turn>model\n");
        assert_eq!(t.decode(&[opener]).unwrap(), "<start_of_turn>model\n");
        let close = special_id(&t, "<end_of_turn>");
        assert_eq!(
            t.decode(&[id(&t, "\u{2581}hello"), close]).unwrap(),
            " hello"
        );
    }

    #[test]
    fn a_vocabulary_without_special_pieces_encodes_exactly_as_before() {
        let t = tokenizer_without_specials();
        // No piece covers any byte of "<pad>", so it stays a byte run and the
        // word after it still segments as one piece.
        let mut expect: Vec<Token> = "<pad>"
            .bytes()
            .map(|b| t.byte_tokens[b as usize].unwrap())
            .collect();
        expect.push(id(&t, "\u{2581}hello"));
        assert_eq!(t.encode("<pad> hello").unwrap(), expect);
    }

    #[test]
    fn an_oversized_vocabulary_piece_does_not_widen_the_viterbi_window() {
        let t = tokenizer_with_oversized_piece(4096);
        assert!(
            t.max_piece_bytes <= MAX_PIECE_BYTES,
            "window is {} bytes, cap is {MAX_PIECE_BYTES}",
            t.max_piece_bytes
        );
        // The oversized piece is unreachable, not mis-matched: text around
        // it segments exactly as in the vocabulary that lacks it.
        let plain = tokenizer();
        assert_eq!(
            t.encode(" hello world").unwrap(),
            plain.encode(" hello world").unwrap()
        );
        // Its own text falls back to byte tokens and still round-trips.
        let text = "q".repeat(4096);
        let ids = t.encode(&text).unwrap();
        assert_eq!(ids.len(), 4096);
        assert_eq!(t.decode(&ids).unwrap(), text);
    }

    #[test]
    fn a_two_kilobyte_prompt_encodes_promptly_despite_an_oversized_piece() {
        let t = tokenizer_with_oversized_piece(4096);
        let text = " hello world".repeat(171);
        let started = std::time::Instant::now();
        let ids = t.encode(&text).unwrap();
        let elapsed = started.elapsed();
        assert_eq!(ids.len(), 342);
        // Coarse canary for the quadratic window, not a benchmark: this
        // 2052-byte input measures 0.14 s capped against 11.3 s uncapped in
        // an unoptimized build, so the bound sits an order of magnitude above
        // the capped cost and well below the uncapped one.
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "encode of {} bytes took {elapsed:?}",
            text.len()
        );
    }

    #[test]
    fn empty_text_encodes_to_no_tokens() {
        let t = tokenizer();
        assert_eq!(t.encode("").unwrap(), Vec::<Token>::new());
    }

    #[test]
    fn decode_of_out_of_range_token_is_an_error() {
        let t = tokenizer();
        assert!(t.decode(&[9999]).is_err());
    }

    #[test]
    fn invalid_byte_sequences_decode_lossily_not_fatally() {
        let t = tokenizer();
        // A lone continuation byte cannot form valid UTF-8.
        let ids = vec![t.byte_tokens[0x80].unwrap()];
        assert_eq!(t.decode(&ids).unwrap(), "\u{fffd}");
    }

    #[test]
    fn from_gguf_reads_vocab_and_rejects_missing_keys() {
        use crate::gguf::parse;
        // Minimal image: 2-token vocab with ids in range.
        let mut b = Vec::new();
        b.extend(crate::gguf::GGUF_MAGIC.to_le_bytes());
        b.extend(crate::gguf::GGUF_VERSION.to_le_bytes());
        b.extend(0u64.to_le_bytes()); // tensors
        b.extend(5u64.to_le_bytes()); // kvs
        let put_str = |b: &mut Vec<u8>, s: &str| {
            b.extend((s.len() as u64).to_le_bytes());
            b.extend(s.as_bytes());
        };
        put_str(&mut b, "tokenizer.ggml.tokens");
        b.extend(9u32.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        put_str(&mut b, "<bos>");
        put_str(&mut b, "\u{2581}hi");
        put_str(&mut b, "tokenizer.ggml.scores");
        b.extend(9u32.to_le_bytes());
        b.extend(6u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(0.0f32.to_le_bytes());
        b.extend((-1.0f32).to_le_bytes());
        put_str(&mut b, "tokenizer.ggml.token_type");
        b.extend(9u32.to_le_bytes());
        b.extend(5u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(3i32.to_le_bytes());
        b.extend(1i32.to_le_bytes());
        for (key, v) in [
            ("tokenizer.ggml.bos_token_id", 0u32),
            ("tokenizer.ggml.eos_token_id", 0u32),
        ] {
            put_str(&mut b, key);
            b.extend(4u32.to_le_bytes());
            b.extend(v.to_le_bytes());
        }
        let f = parse(&b).unwrap();
        let t = Tokenizer::from_gguf(&f).unwrap();
        assert_eq!(t.vocab_len(), 2);
        assert_eq!(t.encode(" hi").unwrap(), vec![1]);

        // Dropping the scores key must fail naming it.
        let mut meta = f.metadata.clone();
        meta.remove("tokenizer.ggml.scores");
        let broken = crate::gguf::GgufFile {
            data_offset: 0,
            metadata: meta,
            tensors: vec![],
        };
        let err = Tokenizer::from_gguf(&broken).unwrap_err();
        assert!(err.to_string().contains("tokenizer.ggml.scores"));
    }

    #[test]
    fn mismatched_array_lengths_are_an_error() {
        let pieces = vec!["a".to_string(), "b".to_string()];
        let scores = vec![0.0];
        let kinds = [Kind::Normal, Kind::Normal];
        // Bypass from_gguf: exercise the consistency check directly.
        let mut file = crate::gguf::GgufFile {
            data_offset: 0,
            metadata: Default::default(),
            tensors: vec![],
        };
        file.metadata.insert(
            "tokenizer.ggml.tokens".into(),
            MetaValue::ArrStr(pieces.clone()),
        );
        file.metadata
            .insert("tokenizer.ggml.scores".into(), MetaValue::ArrF32(scores));
        file.metadata.insert(
            "tokenizer.ggml.token_type".into(),
            MetaValue::ArrI32(kinds.iter().map(|_| 1).collect()),
        );
        file.metadata
            .insert("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(0));
        file.metadata
            .insert("tokenizer.ggml.eos_token_id".into(), MetaValue::U32(1));
        let err = Tokenizer::from_gguf(&file).unwrap_err();
        assert!(err.to_string().contains("disagree"));
    }
}
