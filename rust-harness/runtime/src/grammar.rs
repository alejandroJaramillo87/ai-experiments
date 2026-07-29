/*
 * grammar — tool-call grammar compilation and token masking.
 *
 * A tool schema compiles to a fixed JSON skeleton (exact key order, exact
 * spacing) with free-form value regions for string, int, and bool parameters.
 * Fixing the skeleton keeps the automaton a straight line with three value
 * sub-machines, while every accepted output remains valid JSON matching the
 * schema. The mask is computed per decode step by simulating each vocabulary
 * token's characters through the automaton; a token is legal only if every
 * character advances.
 */

use super::model::Token;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamType {
    String,
    Int,
    Bool,
}

/// Tool call surface: `{"tool": "<name>", "args": {"<p1>": <v1>, ...}}`.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub name: String,
    pub params: Vec<(String, ParamType)>,
}

/// One node of the compiled skeleton.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Node {
    Lit(char),
    Str,
    Int,
    Bool,
}

/// Sub-state inside a value node.
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrammarState {
    node: usize,
    sub: Sub,
}

pub struct Grammar {
    nodes: Vec<Node>,
}

const BOOL_WORDS: [&str; 2] = ["true", "false"];

impl Grammar {
    pub fn compile(schema: &ToolSchema) -> Self {
        let mut nodes = Vec::new();
        let lit = |nodes: &mut Vec<Node>, s: &str| nodes.extend(s.chars().map(Node::Lit));
        lit(&mut nodes, "{\"tool\": \"");
        lit(&mut nodes, &schema.name);
        lit(&mut nodes, "\", \"args\": {");
        for (i, (key, ty)) in schema.params.iter().enumerate() {
            if i > 0 {
                lit(&mut nodes, ", ");
            }
            lit(&mut nodes, &format!("\"{key}\": "));
            nodes.push(match ty {
                ParamType::String => Node::Str,
                ParamType::Int => Node::Int,
                ParamType::Bool => Node::Bool,
            });
        }
        lit(&mut nodes, "}}");
        Self { nodes }
    }

    pub fn start(&self) -> GrammarState {
        self.enter(0)
    }

    pub fn is_complete(&self, st: &GrammarState) -> bool {
        st.node == self.nodes.len()
    }

    /// Initial state upon entering node `idx`.
    fn enter(&self, idx: usize) -> GrammarState {
        let sub = match self.nodes.get(idx) {
            Some(Node::Str) => Sub::StrOpen,
            Some(Node::Int) => Sub::IntStart,
            Some(Node::Bool) => Sub::BoolAt(0, 0),
            _ => Sub::None,
        };
        GrammarState { node: idx, sub }
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
        match (node, st.sub) {
            (Node::Lit(l), Sub::None) => (c == l).then(|| self.enter(st.node + 1)),
            (Node::Str, Sub::StrOpen) => (c == '"').then_some(GrammarState {
                sub: Sub::StrBody,
                ..*st
            }),
            (Node::Str, Sub::StrBody) => match c {
                '"' => Some(self.enter(st.node + 1)),
                '\\' => Some(GrammarState {
                    sub: Sub::StrEscape,
                    ..*st
                }),
                c if (c as u32) < 0x20 => None,
                _ => Some(*st),
            },
            // Deliberate restriction: `\uXXXX` escapes are rejected. The
            // accepted language stays a strict subset of JSON strings;
            // non-ASCII content passes through unescaped instead.
            (Node::Str, Sub::StrEscape) => {
                matches!(c, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't').then_some(
                    GrammarState {
                        sub: Sub::StrBody,
                        ..*st
                    },
                )
            }
            (Node::Int, Sub::IntStart) => match c {
                '-' => Some(GrammarState {
                    sub: Sub::IntNeedDigit,
                    ..*st
                }),
                '0' => Some(GrammarState {
                    sub: Sub::IntZero,
                    ..*st
                }),
                '1'..='9' => Some(GrammarState {
                    sub: Sub::IntDigits,
                    ..*st
                }),
                _ => None,
            },
            (Node::Int, Sub::IntNeedDigit) => match c {
                '0' => Some(GrammarState {
                    sub: Sub::IntZero,
                    ..*st
                }),
                '1'..='9' => Some(GrammarState {
                    sub: Sub::IntDigits,
                    ..*st
                }),
                _ => None,
            },
            (Node::Int, Sub::IntDigits) => {
                if c.is_ascii_digit() {
                    Some(*st)
                } else {
                    // The value ends where the following skeleton resumes.
                    self.advance(&self.enter(st.node + 1), c)
                }
            }
            (Node::Int, Sub::IntZero) => {
                if c.is_ascii_digit() {
                    None
                } else {
                    self.advance(&self.enter(st.node + 1), c)
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
                            self.enter(st.node + 1)
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
            _ => None,
        }
    }

    /// Token mask for the current state. EOS is legal exactly at completion;
    /// its surface text is never simulated through the automaton.
    pub fn mask(&self, st: &GrammarState, vocab: &Vocab) -> Vec<bool> {
        (0..vocab.len() as Token)
            .map(|t| {
                if t == vocab.eos() {
                    return self.is_complete(st);
                }
                let text = vocab.text(t);
                !text.is_empty() && self.advance_str(st, text).is_some()
            })
            .collect()
    }
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
