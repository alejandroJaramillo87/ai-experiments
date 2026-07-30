/*
 * util — wildcard matching shared by glob and permission rules.
 *
 * Matching is dynamic programming over (pattern element, text position), so
 * cost is O(pattern * text) even for adversarial patterns; the naive
 * backtracking formulation is exponential on inputs the model or a rule file
 * can supply.
 */

/// Glob-style match: `*` matches within a path segment, `**` crosses
/// segment boundaries, `?` matches one non-separator character.
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    Pattern::compile(pattern).matches(text)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Elem {
    Lit(char),
    // `?`: exactly one char, never `/`.
    One,
    // `*`: any run of non-`/` chars, possibly empty.
    Star,
    // `**`: any run of chars, possibly empty.
    DoubleStar,
    // `**/`: empty, or any run of chars ending in `/`.
    DoubleStarSlash,
}

/// A compiled pattern, reusable across many texts (one glob walk compiles
/// once instead of once per file).
#[derive(Debug)]
pub struct Pattern {
    elems: Vec<Elem>,
}

impl Pattern {
    pub fn compile(pattern: &str) -> Self {
        let chars: Vec<char> = pattern.chars().collect();
        let mut elems = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            match chars[i] {
                '*' => {
                    // Collapse a run of stars; two or more cross segments.
                    let run = chars[i..].iter().take_while(|&&c| c == '*').count();
                    i += run;
                    if run == 1 {
                        elems.push(Elem::Star);
                    } else if chars.get(i) == Some(&'/') {
                        i += 1;
                        elems.push(Elem::DoubleStarSlash);
                    } else {
                        elems.push(Elem::DoubleStar);
                    }
                }
                '?' => {
                    elems.push(Elem::One);
                    i += 1;
                }
                c => {
                    elems.push(Elem::Lit(c));
                    i += 1;
                }
            }
        }
        Self { elems }
    }

    pub fn matches(&self, text: &str) -> bool {
        let t: Vec<char> = text.chars().collect();
        let n = t.len();
        // next[j]: does the pattern suffix below this element match t[j..].
        let mut next = vec![false; n + 1];
        next[n] = true;
        for elem in self.elems.iter().rev() {
            let mut cur = vec![false; n + 1];
            // `**/` needs "exists k >= j with t[k] == '/' and next[k+1]".
            let mut slash_next = false;
            for j in (0..=n).rev() {
                cur[j] = match *elem {
                    Elem::Lit(c) => j < n && t[j] == c && next[j + 1],
                    Elem::One => j < n && t[j] != '/' && next[j + 1],
                    Elem::Star => next[j] || (j < n && t[j] != '/' && cur[j + 1]),
                    Elem::DoubleStar => next[j] || (j < n && cur[j + 1]),
                    Elem::DoubleStarSlash => {
                        if j < n && t[j] == '/' && next[j + 1] {
                            slash_next = true;
                        }
                        next[j] || slash_next
                    }
                };
            }
            next = cur;
        }
        next[0]
    }
}

#[cfg(test)]
mod tests {
    use super::wildcard_match;

    #[test]
    fn star_stays_within_a_segment() {
        assert!(wildcard_match("src/*.rs", "src/lib.rs"));
        assert!(!wildcard_match("src/*.rs", "src/model/mod.rs"));
    }

    #[test]
    fn double_star_crosses_segments() {
        assert!(wildcard_match("**/*.rs", "src/model/mod.rs"));
        assert!(wildcard_match("**/*.rs", "main.rs"));
        assert!(!wildcard_match("**/*.rs", "README.md"));
    }

    #[test]
    fn literal_and_question_mark() {
        assert!(wildcard_match("a?c", "abc"));
        assert!(!wildcard_match("a?c", "ac"));
        // Command patterns use `**` because `*` never crosses `/`.
        assert!(wildcard_match("rm **", "rm -rf /tmp/x"));
        assert!(!wildcard_match("rm *", "rm -rf /tmp/x"));
    }

    #[test]
    fn question_mark_does_not_match_a_separator() {
        assert!(!wildcard_match("a?c", "a/c"));
        assert!(!wildcard_match("src?main.rs", "src/main.rs"));
    }

    #[test]
    fn double_star_slash_matches_empty_or_slash_terminated_prefixes() {
        assert!(wildcard_match("**/x", "x"));
        assert!(wildcard_match("**/x", "a/b/x"));
        assert!(!wildcard_match("**/x", "ax"));
    }

    #[test]
    fn consecutive_stars_collapse() {
        assert!(wildcard_match("***", "a/b/c"));
        assert!(wildcard_match("a***b", "a/x/b"));
    }

    #[test]
    fn adversarial_pattern_completes_quickly() {
        // Exponential backtracking would hang here; DP finishes instantly.
        let pattern = "**a".repeat(12) + "b";
        let text = "a".repeat(300);
        let start = std::time::Instant::now();
        assert!(!wildcard_match(&pattern, &text));
        assert!(start.elapsed() < std::time::Duration::from_secs(2));
    }
}
