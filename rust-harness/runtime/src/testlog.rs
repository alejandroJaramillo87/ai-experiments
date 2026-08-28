/*
 * testlog — status lines for the cross-language contract tests.
 *
 * A skipped contract test is the one outcome the gate's exit code cannot
 * express: it passes, so nothing draws the eye, yet the property it exists to
 * check went unchecked. `/AGENTS.md` and `.styles/RUST_STYLE.md` both tell the
 * reader to scan for [SKIP] rather than trust the exit status, which only
 * works if a skip is actually noticeable in several hundred lines of `ok`.
 *
 * The formatting and the colour gate moved to `status` when the other five tags
 * arrived. What stays here is the discipline the style guide asks of a skipped
 * test specifically: a subject naming what went unchecked and a reason naming
 * the missing dependency. `status::skip` takes one free-form message, as the
 * Python `Printer` does, so this is where the two halves are composed -- and
 * keeping the `testlog::skip` spelling keeps the guide's contract, and its many
 * call sites, unchanged.
 */

use crate::status;

/// Report a skipped test as `[SKIP] <subject>: <reason>`.
///
/// `subject` is a noun phrase naming what went unchecked, `reason` names the
/// missing dependency precisely enough to act on — a reader who cannot tell
/// what to install from the line has to go read the test instead.
pub fn skip(subject: &str, reason: &str) {
    status::skip(&format!("{subject}: {reason}"));
}

#[cfg(test)]
mod tests {
    use crate::status::{self, Tag};

    /// The exact line this module produced before `status` existed, kept
    /// verbatim: if it still holds, the delegation is byte-compatible and every
    /// existing [SKIP] call site reads as it always has.
    #[test]
    fn an_uncoloured_line_carries_no_escape_bytes() {
        let got = status::line(Tag::Skip, "shard row assertions: numpy is absent", 0, false);
        assert_eq!(got, "[SKIP] shard row assertions: numpy is absent");
    }

    // The colour behaviour is `status`'s to test, and is tested there over all
    // six tags. Deliberately no test that calls skip() itself: its output would
    // land in the gate as a [SKIP] line naming nothing real, and a reader
    // scanning for skips would have to know to discount it.
}
