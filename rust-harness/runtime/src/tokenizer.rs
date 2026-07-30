/*
 * tokenizer — SentencePiece unigram tokenizer over the GGUF-embedded vocab.
 *
 * Gemma checkpoints ship the vocabulary inside the GGUF metadata
 * (tokenizer.ggml.tokens / .scores / .token_type plus bos/eos ids). Encoding
 * is unigram Viterbi over the score table with byte fallback; decoding maps
 * pieces back to text with the SentencePiece space marker (U+2581) restored
 * to a plain space. No dummy-prefix space is inserted: the chat template
 * controls surrounding whitespace exactly, and the Tier 3 template-golden
 * test verifies the end-to-end token stream against the Hugging Face
 * tokenizer.
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

/// Token roles from tokenizer.ggml.token_type (llama.cpp convention).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Normal,
    Unknown,
    Control,
    Byte,
    Other,
}

impl Kind {
    fn from_i32(v: i32) -> Kind {
        match v {
            1 => Kind::Normal,
            2 => Kind::Unknown,
            3 => Kind::Control,
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
    // Byte value -> byte-token id, for fallback encoding.
    byte_tokens: [Option<Token>; 256],
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
        let mut byte_tokens = [None; 256];
        let mut max_piece_bytes = 1;
        for (id, (piece, kind)) in pieces.iter().zip(&kinds).enumerate() {
            match kind {
                Kind::Normal => {
                    max_piece_bytes = max_piece_bytes.max(piece.len());
                    // First occurrence wins on duplicate piece text.
                    lookup.entry(piece.clone()).or_insert(id as Token);
                }
                Kind::Byte => {
                    let b = parse_byte_piece(piece)
                        .with_context(|| format!("bad byte token {id}: {piece:?}"))?;
                    byte_tokens[b as usize] = Some(id as Token);
                }
                _ => {}
            }
        }
        Ok(Tokenizer {
            pieces,
            scores,
            kinds,
            lookup,
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

    /// Decode token ids to text. Control tokens render as nothing; byte
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

    /// Encode text to token ids via unigram Viterbi: best-scoring
    /// segmentation into vocabulary pieces, with per-byte fallback tokens for
    /// anything the vocabulary cannot cover. No BOS is added — the caller
    /// (chat template rendering) owns special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<Token>> {
        let marked: String = text
            .chars()
            .map(|c| if c == ' ' { SPACE_PIECE } else { c })
            .collect();
        let bytes = marked.as_bytes();
        if bytes.is_empty() {
            return Ok(Vec::new());
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
        let mut out = Vec::new();
        let mut pos = bytes.len();
        while pos > 0 {
            let (_, tok, start) = best[pos]
                .with_context(|| format!("no tokenization covers byte {} of input", pos - 1))?;
            out.push(tok);
            pos = start;
        }
        out.reverse();
        Ok(out)
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
    /// few normal pieces with scores favoring longer matches.
    fn tokenizer() -> Tokenizer {
        let mut pieces = vec![
            "<pad>".to_string(),
            "<bos>".to_string(),
            "<eos>".to_string(),
        ];
        let mut kinds = vec![Kind::Control; 3];
        for b in 0..=255u8 {
            pieces.push(format!("<0x{b:02X}>"));
            kinds.push(Kind::Byte);
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
            kinds.push(Kind::Normal);
        }
        let mut scores = vec![0.0; 3 + 256];
        scores.extend(normals.iter().map(|(_, s)| *s));
        Tokenizer::new(pieces, scores, kinds, 1, 2).unwrap()
    }

    fn id(t: &Tokenizer, piece: &str) -> Token {
        t.lookup[piece]
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
