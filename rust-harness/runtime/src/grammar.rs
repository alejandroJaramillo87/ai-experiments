/*
 * grammar — tool-call grammar compilation and token masking.
 *
 * A set of tool schemas compiles to a trie of JSON skeletons (exact key
 * order, exact spacing) with free-form value regions for string, int, bool,
 * and string-array parameters. The skeletons share their common prefix
 * (`{"tool": "`) and diverge where the tool names do, so one automaton
 * accepts exactly one well-formed call to any tool in the set. The mask is
 * computed per decode step by simulating each vocabulary token's characters
 * through the automaton; a token is legal only if every character advances.
 *
 * A grammar is either forced -- the turn must be a tool call -- or
 * triggered: the first character decides. A turn that opens the skeleton is
 * constrained to its end and EOS; any other opening makes the turn free text.
 */

use super::model::Token;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamType {
    String,
    Int,
    Bool,
    /// A JSON array of strings: `["a", "b"]` or `[]`.
    StringArray,
}

/// Tool call surface: `{"tool": "<name>", "args": {"<p1>": <v1>, ...}}`.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub name: String,
    pub params: Vec<(String, ParamType)>,
}

/// One node of the compiled trie. A state that has *completed* a node sits
/// on it with `Sub::None`; the node's successors say what may come next.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Node {
    /// The virtual root: completed before the first character.
    Root,
    Lit(char),
    Str,
    Int,
    Bool,
    StrArray,
}

/// Sub-state inside a value node, or the two triggered-mode states.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Sub {
    None,
    StrOpen,
    StrBody,
    StrEscape,
    IntStart,
    IntNeedDigit,
    IntDigits,
    // JSON forbids leading zeros: a leading `0` must end the number.
    IntZero,
    // Bool progress: which literal (0 = true, 1 = false) and chars consumed.
    BoolAt(u8, u8),
    // Array progress: awaiting `[`; after `[`; inside an element; inside an
    // element escape; after an element; after `,`; after `, `.
    ArrOpen,
    ArrFirst,
    ArrBody,
    ArrEscape,
    ArrAfter,
    ArrSep,
    ArrNext,
    // Triggered mode: no character seen yet, so the turn is still either.
    Undecided,
    // Triggered mode: the turn opened as free text and stays unconstrained.
    Free,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrammarState {
    node: usize,
    sub: Sub,
}

pub struct Grammar {
    nodes: Vec<Node>,
    // Successor node indices, parallel to `nodes`. Empty means accepting.
    next: Vec<Vec<usize>>,
    triggered: bool,
}

const BOOL_WORDS: [&str; 2] = ["true", "false"];

/// JSON string-escape `s` (no surrounding quotes), so schema-supplied names
/// and keys always compile to a skeleton whose accepted language is valid
/// JSON — a quote or backslash in a tool name must not break the invariant.
fn json_escaped(s: &str) -> String {
    let quoted = serde_json::to_string(s).expect("strings always serialize");
    quoted[1..quoted.len() - 1].to_string()
}

impl Grammar {
    /// A grammar over a single tool: the turn must be a call to it.
    pub fn compile(schema: &ToolSchema) -> Self {
        Self::compile_many(std::slice::from_ref(schema))
    }

    /// A grammar over a set of tools: the turn must be a call to one of them.
    ///
    /// Panics on duplicate tool names or duplicate parameter keys: schemas
    /// are developer-supplied, so a duplicate is a config bug (the
    /// Sampler::new precedent). Two tools of one name would share a trie
    /// path and accept an ambiguous call; a duplicate key would produce an
    /// object ambiguous to consumers.
    pub fn compile_many(schemas: &[ToolSchema]) -> Self {
        assert!(!schemas.is_empty(), "tool grammar over an empty schema set");
        for (i, schema) in schemas.iter().enumerate() {
            assert!(
                !schemas[..i].iter().any(|s| s.name == schema.name),
                "duplicate tool name {:?} in tool grammar",
                schema.name
            );
            for (j, (key, _)) in schema.params.iter().enumerate() {
                assert!(
                    !schema.params[..j].iter().any(|(k, _)| k == key),
                    "duplicate parameter key {key:?} in tool schema {:?}",
                    schema.name
                );
            }
        }
        let mut g = Self {
            nodes: vec![Node::Root],
            next: vec![Vec::new()],
            triggered: false,
        };
        for schema in schemas {
            let mut at = 0;
            at = g.insert_lit(at, "{\"tool\": \"");
            at = g.insert_lit(at, &json_escaped(&schema.name));
            at = g.insert_lit(at, "\", \"args\": {");
            for (i, (key, ty)) in schema.params.iter().enumerate() {
                if i > 0 {
                    at = g.insert_lit(at, ", ");
                }
                at = g.insert_lit(at, &format!("\"{}\": ", json_escaped(key)));
                at = g.push_node(
                    at,
                    match ty {
                        ParamType::String => Node::Str,
                        ParamType::Int => Node::Int,
                        ParamType::Bool => Node::Bool,
                        ParamType::StringArray => Node::StrArray,
                    },
                );
            }
            g.insert_lit(at, "}}");
        }
        g
    }

    /// Triggered mode: the first character decides whether the turn is a
    /// constrained tool call or unconstrained free text.
    pub fn triggered(mut self) -> Self {
        self.triggered = true;
        self
    }

    /// Insert the literal `s` below `at`, sharing existing `Lit` successors
    /// so skeletons with a common prefix walk one path.
    fn insert_lit(&mut self, mut at: usize, s: &str) -> usize {
        for c in s.chars() {
            at = match self.next[at]
                .iter()
                .copied()
                .find(|&n| self.nodes[n] == Node::Lit(c))
            {
                Some(n) => n,
                None => self.push_node(at, Node::Lit(c)),
            };
        }
        at
    }

    fn push_node(&mut self, at: usize, node: Node) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.next.push(Vec::new());
        self.next[at].push(idx);
        idx
    }

    pub fn start(&self) -> GrammarState {
        GrammarState {
            node: 0,
            sub: if self.triggered {
                Sub::Undecided
            } else {
                Sub::None
            },
        }
    }

    /// Whether EOS is legal: a complete call, or a triggered turn that is
    /// free text (including one that has not started).
    pub fn is_complete(&self, st: &GrammarState) -> bool {
        match st.sub {
            Sub::None => self.next[st.node].is_empty(),
            Sub::Undecided | Sub::Free => true,
            _ => false,
        }
    }

    /// Whether the mask constrains anything in this state. False on a
    /// triggered turn that is, or may still become, free text, so a caller
    /// can skip the per-vocabulary walk.
    pub fn constrained(&self, st: &GrammarState) -> bool {
        !matches!(st.sub, Sub::Undecided | Sub::Free)
    }

    /// State on entering value node `idx`, or the completed state of `idx`
    /// for a literal.
    fn enter(&self, idx: usize) -> GrammarState {
        let sub = match self.nodes[idx] {
            Node::Str => Sub::StrOpen,
            Node::Int => Sub::IntStart,
            Node::Bool => Sub::BoolAt(0, 0),
            Node::StrArray => Sub::ArrOpen,
            Node::Root | Node::Lit(_) => Sub::None,
        };
        GrammarState { node: idx, sub }
    }

    /// Advance from the completed state of `st.node` into a successor with
    /// `c`; `None` if no successor accepts it.
    fn advance_from_completed(&self, st: &GrammarState, c: char) -> Option<GrammarState> {
        self.next[st.node]
            .iter()
            .find_map(|&n| match self.nodes[n] {
                Node::Lit(l) => (c == l).then(|| self.enter(n)),
                _ => self.advance(&self.enter(n), c),
            })
    }

    /// Advance through every character of `text`; `None` if any is illegal.
    pub fn advance_str(&self, st: &GrammarState, text: &str) -> Option<GrammarState> {
        let mut cur = *st;
        for c in text.chars() {
            cur = self.advance(&cur, c)?;
        }
        Some(cur)
    }

    /// Advance one character; `None` means the character is illegal here.
    pub fn advance(&self, st: &GrammarState, c: char) -> Option<GrammarState> {
        let node = *self.nodes.get(st.node)?;
        let body = |sub: Sub| Some(GrammarState { sub, ..*st });
        let completed = GrammarState {
            sub: Sub::None,
            ..*st
        };
        match (node, st.sub) {
            (_, Sub::Free) => Some(*st),
            // The first character decides: open the skeleton, or free text.
            (Node::Root, Sub::Undecided) => Some(
                self.advance_from_completed(&completed, c)
                    .unwrap_or(GrammarState {
                        node: 0,
                        sub: Sub::Free,
                    }),
            ),
            (_, Sub::None) => self.advance_from_completed(st, c),
            (Node::Str, Sub::StrOpen) => (c == '"').then(|| body(Sub::StrBody)).flatten(),
            (Node::Str, Sub::StrBody) => match c {
                '"' => Some(completed),
                '\\' => body(Sub::StrEscape),
                c if (c as u32) < 0x20 => None,
                _ => Some(*st),
            },
            // Deliberate restriction: `\uXXXX` escapes are rejected. The
            // accepted language stays a strict subset of JSON strings;
            // non-ASCII content passes through unescaped instead.
            (Node::Str, Sub::StrEscape) => {
                is_simple_escape(c).then(|| body(Sub::StrBody)).flatten()
            }
            (Node::Int, Sub::IntStart) => match c {
                '-' => body(Sub::IntNeedDigit),
                '0' => body(Sub::IntZero),
                '1'..='9' => body(Sub::IntDigits),
                _ => None,
            },
            (Node::Int, Sub::IntNeedDigit) => match c {
                '0' => body(Sub::IntZero),
                '1'..='9' => body(Sub::IntDigits),
                _ => None,
            },
            (Node::Int, Sub::IntDigits) => {
                if c.is_ascii_digit() {
                    Some(*st)
                } else {
                    // The value ends where the following skeleton resumes.
                    self.advance_from_completed(&completed, c)
                }
            }
            (Node::Int, Sub::IntZero) => {
                if c.is_ascii_digit() {
                    None
                } else {
                    self.advance_from_completed(&completed, c)
                }
            }
            (Node::Bool, Sub::BoolAt(word, pos)) => {
                // Byte comparison below requires ASCII: `c as u8` truncates,
                // so a non-ASCII char sharing a low byte (e.g. 'Ŵ' -> 't')
                // would otherwise pass and emit invalid JSON.
                if !c.is_ascii() {
                    return None;
                }
                let candidates: &[u8] = if pos == 0 { &[0, 1] } else { &[word] };
                for &w in candidates {
                    let text = BOOL_WORDS[w as usize].as_bytes();
                    if text.get(pos as usize) == Some(&(c as u8)) {
                        return Some(if pos as usize + 1 == text.len() {
                            completed
                        } else {
                            GrammarState {
                                sub: Sub::BoolAt(w, pos + 1),
                                ..*st
                            }
                        });
                    }
                }
                None
            }
            (Node::StrArray, Sub::ArrOpen) => (c == '[').then(|| body(Sub::ArrFirst)).flatten(),
            (Node::StrArray, Sub::ArrFirst) => match c {
                '"' => body(Sub::ArrBody),
                ']' => Some(completed),
                _ => None,
            },
            (Node::StrArray, Sub::ArrBody) => match c {
                '"' => body(Sub::ArrAfter),
                '\\' => body(Sub::ArrEscape),
                c if (c as u32) < 0x20 => None,
                _ => Some(*st),
            },
            (Node::StrArray, Sub::ArrEscape) => {
                is_simple_escape(c).then(|| body(Sub::ArrBody)).flatten()
            }
            (Node::StrArray, Sub::ArrAfter) => match c {
                ',' => body(Sub::ArrSep),
                ']' => Some(completed),
                _ => None,
            },
            (Node::StrArray, Sub::ArrSep) => (c == ' ').then(|| body(Sub::ArrNext)).flatten(),
            (Node::StrArray, Sub::ArrNext) => (c == '"').then(|| body(Sub::ArrBody)).flatten(),
            _ => None,
        }
    }

    /// Token mask for the current state. EOS is legal exactly at completion;
    /// its surface text is never simulated through the automaton. In an
    /// unconstrained triggered state every token is legal; callers should
    /// check `constrained` first and skip the walk.
    pub fn mask(&self, st: &GrammarState, vocab: &Vocab) -> Vec<bool> {
        // The per-vocab-entry advance walk is the expensive grammar call, so
        // this is the traced unit rather than the per-character advance.
        let span = tracing::info_span!(
            "grammar.advance",
            state = st.node,
            allowed = tracing::field::Empty
        );
        let _enter = span.enter();
        let mask: Vec<bool> = (0..vocab.len() as Token)
            .map(|t| {
                if t == vocab.eos() {
                    return self.is_complete(st);
                }
                let text = vocab.text(t);
                !text.is_empty() && self.advance_str(st, text).is_some()
            })
            .collect();
        span.record("allowed", mask.iter().filter(|&&b| b).count());
        mask
    }
}

/// The escape characters accepted after a backslash inside a string value.
fn is_simple_escape(c: char) -> bool {
    matches!(c, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't')
}

/// Token-id to surface-text mapping with a designated EOS token.
pub struct Vocab {
    tokens: Vec<String>,
    eos: Token,
}

impl Vocab {
    pub fn new(tokens: Vec<String>, eos: Token) -> Self {
        assert!((eos as usize) < tokens.len(), "EOS id out of range");
        Self { tokens, eos }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn eos(&self) -> Token {
        self.eos
    }

    pub fn text(&self, t: Token) -> &str {
        &self.tokens[t as usize]
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.tokens.iter().map(String::as_str)
    }
}
