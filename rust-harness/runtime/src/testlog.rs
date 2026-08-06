/*
 * testlog — status lines for the cross-language contract tests.
 *
 * A skipped contract test is the one outcome the gate's exit code cannot
 * express: it passes, so nothing draws the eye, yet the property it exists to
 * check went unchecked. `/AGENTS.md` and `.styles/RUST_STYLE.md` both tell the
 * reader to scan for [SKIP] rather than trust the exit status, which only
 * works if a skip is actually noticeable in several hundred lines of `ok`.
 *
 * Colour per `.styles/TEST_OUTPUT_STYLE.md`: [SKIP] is blue. It is display
 * only — the text is identical without it, so piped output stays greppable —
 * and it is suppressed when NO_COLOR is set or stderr is not a terminal.
 */

use std::io::IsTerminal;

/// ANSI blue, per the status-indicator table in the test output style guide.
const BLUE: &str = "\x1b[0;34m";
const RESET: &str = "\x1b[0m";

/// The formatted line, colourised or not. Split from the emitter so the
/// formatting is testable without asserting anything about the environment the
/// suite happens to run in — under `--nocapture` stderr really is a terminal.
fn line(subject: &str, reason: &str, color: bool) -> String {
    if color {
        format!("{BLUE}[SKIP]{RESET} {subject}: {reason}")
    } else {
        format!("[SKIP] {subject}: {reason}")
    }
}

/// Report a skipped test as `[SKIP] <subject>: <reason>`.
///
/// `subject` is a noun phrase naming what went unchecked, `reason` names the
/// missing dependency precisely enough to act on — a reader who cannot tell
/// what to install from the line has to go read the test instead.
pub fn skip(subject: &str, reason: &str) {
    let color = std::env::var_os("NO_COLOR").is_none() && std::io::stderr().is_terminal();
    eprintln!("{}", line(subject, reason, color));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_uncoloured_line_carries_no_escape_bytes() {
        let got = line("shard row assertions", "numpy is absent", false);
        assert_eq!(got, "[SKIP] shard row assertions: numpy is absent");
    }

    /// The text either side of the escapes is byte-identical to the plain
    /// form, so `grep '\[SKIP\]'` finds a coloured line too.
    #[test]
    fn a_coloured_line_wraps_the_same_text_and_resets() {
        let plain = line("a", "b", false);
        let colored = line("a", "b", true);
        assert!(colored.ends_with(&plain[6..]), "{colored}");
        assert!(colored.contains("[SKIP]"), "{colored}");
        assert!(colored.ends_with("b"), "{colored}");
        assert!(colored.starts_with(BLUE), "{colored}");
        assert!(colored.contains(RESET), "{colored}");
    }

    // Deliberately no test that calls skip() itself: its output would land in
    // the gate as a [SKIP] line naming nothing real, and a reader scanning for
    // skips would have to know to discount it. The formatting is covered
    // above; the emitter adds only the stderr write.
}
