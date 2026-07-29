/*
 * util — wildcard matching shared by glob and permission rules.
 */

/// Glob-style match: `*` matches within a path segment, `**` crosses
/// segment boundaries, `?` matches one character.
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    match_at(&p, 0, &t, 0)
}

fn match_at(p: &[char], mut pi: usize, t: &[char], ti: usize) -> bool {
    if pi == p.len() {
        return ti == t.len();
    }
    // `**` crosses `/`; a trailing `**/` also matches the empty prefix.
    if p[pi] == '*' && p.get(pi + 1) == Some(&'*') {
        pi += 2;
        if p.get(pi) == Some(&'/') && match_at(p, pi + 1, t, ti) {
            return true;
        }
        return (ti..=t.len()).any(|i| match_at(p, pi, t, i));
    }
    match p[pi] {
        '*' => (ti..=t.len())
            .take_while(|&i| i == ti || t[i - 1] != '/')
            .any(|i| match_at(p, pi + 1, t, i)),
        '?' => ti < t.len() && match_at(p, pi + 1, t, ti + 1),
        c => ti < t.len() && t[ti] == c && match_at(p, pi + 1, t, ti + 1),
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
}
