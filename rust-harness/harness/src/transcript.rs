/*
 * transcript — JSONL session persistence with resume.
 *
 * One JSON object per line, appended as the conversation advances, so a
 * crashed or interrupted session resumes from disk without re-running tools.
 */

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Model,
    Tool,
    // A summary spliced over a range of earlier entries. History is
    // append-only: the summarized entries stay on disk and the record
    // carries the range it replaces (see `effective`).
    Compaction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub role: Role,
    pub content: String,
    // Present only on Role::Compaction: the half-open range of the
    // then-effective entry sequence this summary replaces. Optional and
    // skipped when absent, so files written before compaction still parse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replaces: Option<(usize, usize)>,
}

#[derive(Debug)]
pub struct Transcript {
    file: File,
    entries: Vec<Entry>,
}

impl Transcript {
    /// Open a transcript file, resuming any entries already on disk. A torn
    /// final line — the footprint of a crash mid-append, the exact event this
    /// module exists to survive — is truncated away; corruption anywhere
    /// before the final line is still a hard error. A final line that parses
    /// but lost its newline is kept and terminated, so the next append starts
    /// a line of its own.
    pub fn open(path: &Path) -> Result<Self> {
        let mut entries = Vec::new();
        let mut truncate_at: Option<u64> = None;
        // Set when the file does not end in a newline while its final line
        // is not torn: a complete write whose terminator never landed.
        let mut unterminated = false;
        if path.exists() {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("open {}", path.display()))?;
            unterminated = !text.is_empty() && !text.ends_with('\n');
            let mut offset = 0usize;
            let mut lines = text.split_inclusive('\n').enumerate().peekable();
            while let Some((n, line)) = lines.next() {
                let start = offset;
                offset += line.len();
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str(line) {
                    Ok(entry) => entries.push(entry),
                    Err(e) => {
                        let rest_nonempty = lines.any(|(_, l)| !l.trim().is_empty());
                        if rest_nonempty {
                            bail!("{}:{}: bad entry: {e}", path.display(), n + 1);
                        }
                        truncate_at = Some(start as u64);
                        break;
                    }
                }
            }
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("open {} for append", path.display()))?;
        if let Some(len) = truncate_at {
            file.set_len(len)
                .with_context(|| format!("truncate torn line in {}", path.display()))?;
        } else if unterminated {
            // Close the line off before anything is appended to it. Appending
            // onto it instead would concatenate two objects into one
            // unparseable final line, which the torn-tail recovery above then
            // discards — turning a recoverable file into lost history.
            file.write_all(b"\n")
                .with_context(|| format!("terminate final line in {}", path.display()))?;
        }
        Ok(Self { file, entries })
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The conversation as the model sees it: raw entries with every
    /// compaction record folded in. A `Compaction` entry replaces the range
    /// it names (indices into the then-effective sequence) with a single
    /// `System` summary entry. Out-of-range indices are clamped so a
    /// hand-edited or damaged record degrades to a smaller splice rather
    /// than a panic.
    pub fn effective(&self) -> Vec<Entry> {
        let mut effective: Vec<Entry> = Vec::new();
        for e in &self.entries {
            match (&e.role, e.replaces) {
                (Role::Compaction, Some((start, end))) => {
                    let end = end.min(effective.len());
                    let start = start.min(end);
                    effective.splice(
                        start..end,
                        [Entry {
                            role: Role::System,
                            content: e.content.clone(),
                            replaces: None,
                        }],
                    );
                }
                // A compaction record without a range replaces nothing; the
                // summary still enters the conversation as a system entry.
                (Role::Compaction, None) => effective.push(Entry {
                    role: Role::System,
                    content: e.content.clone(),
                    replaces: None,
                }),
                _ => effective.push(e.clone()),
            }
        }
        effective
    }

    /// Count of raw entries appended after the last compaction record, or
    /// all entries when none exists. The compaction thrash guard reads this.
    pub fn entries_since_compaction(&self) -> usize {
        self.entries
            .iter()
            .rposition(|e| e.role == Role::Compaction)
            .map(|i| self.entries.len() - i - 1)
            .unwrap_or(self.entries.len())
    }

    pub fn append(&mut self, role: Role, content: impl Into<String>) -> Result<()> {
        self.append_entry(Entry {
            role,
            content: content.into(),
            replaces: None,
        })
    }

    /// Append a compaction record: `content` summarizes the effective range
    /// `[start, end)` it replaces.
    pub fn append_compaction(
        &mut self,
        content: impl Into<String>,
        range: (usize, usize),
    ) -> Result<()> {
        self.append_entry(Entry {
            role: Role::Compaction,
            content: content.into(),
            replaces: Some(range),
        })
    }

    fn append_entry(&mut self, entry: Entry) -> Result<()> {
        writeln!(self.file, "{}", serde_json::to_string(&entry)?)?;
        self.entries.push(entry);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_then_reopen_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        {
            let mut t = Transcript::open(&path).unwrap();
            t.append(Role::System, "be helpful").unwrap();
            t.append(Role::User, "fix the bug").unwrap();
            t.append(Role::Model, "{\"tool\": \"read\"}").unwrap();
        }
        let resumed = Transcript::open(&path).unwrap();
        assert_eq!(resumed.entries().len(), 3);
        assert_eq!(resumed.entries()[1].role, Role::User);
        assert_eq!(resumed.entries()[1].content, "fix the bug");
    }

    #[test]
    fn resume_then_append_preserves_earlier_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        {
            let mut t = Transcript::open(&path).unwrap();
            t.append(Role::User, "first").unwrap();
        }
        {
            let mut t = Transcript::open(&path).unwrap();
            t.append(Role::Model, "second").unwrap();
        }
        let resumed = Transcript::open(&path).unwrap();
        let contents: Vec<&str> = resumed
            .entries()
            .iter()
            .map(|e| e.content.as_str())
            .collect();
        assert_eq!(contents, vec!["first", "second"]);
    }

    #[test]
    fn corrupt_line_before_the_tail_is_a_positioned_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        std::fs::write(
            &path,
            "{\"role\": \"user\", \"content\": \"ok\"}\nnot json\n{\"role\": \"model\", \"content\": \"late\"}\n",
        )
        .unwrap();
        let err = Transcript::open(&path).unwrap_err();
        assert!(format!("{err:#}").contains(":2:"));
    }

    #[test]
    fn compaction_record_round_trips_and_pre_compaction_files_parse() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        {
            let mut t = Transcript::open(&path).unwrap();
            t.append(Role::User, "task").unwrap();
            t.append(Role::Model, "old turn").unwrap();
            t.append(Role::Model, "tail").unwrap();
            t.append_compaction("summary", (1, 2)).unwrap();
        }
        let t = Transcript::open(&path).unwrap();
        assert_eq!(t.entries().len(), 4);
        assert_eq!(t.entries()[3].replaces, Some((1, 2)));
        let eff = t.effective();
        assert_eq!(eff.len(), 3);
        assert_eq!(eff[1].role, Role::System);
        assert_eq!(eff[1].content, "summary");
        assert_eq!(eff[2].content, "tail");
        // A file with no `replaces` fields anywhere (the pre-compaction
        // format) parses and its effective view is the identity.
        let old = dir.path().join("old.jsonl");
        std::fs::write(&old, "{\"role\": \"user\", \"content\": \"hi\"}\n").unwrap();
        let t = Transcript::open(&old).unwrap();
        assert_eq!(t.effective(), t.entries());
    }

    #[test]
    fn torn_compaction_record_is_truncated_and_history_survives() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        std::fs::write(
            &path,
            "{\"role\": \"user\", \"content\": \"ok\"}\n{\"role\": \"compaction\", \"content\": \"summ",
        )
        .unwrap();
        let t = Transcript::open(&path).unwrap();
        assert_eq!(t.entries().len(), 1);
        assert_eq!(t.effective().len(), 1);
    }

    #[test]
    fn out_of_range_compaction_record_degrades_instead_of_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let mut t = Transcript::open(&path).unwrap();
        t.append(Role::User, "only entry").unwrap();
        t.append_compaction("summary", (5, 99)).unwrap();
        let eff = t.effective();
        assert_eq!(eff.len(), 2);
        assert_eq!(eff[1].content, "summary");
    }

    #[test]
    fn entries_since_compaction_counts_the_suffix() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = Transcript::open(&dir.path().join("s.jsonl")).unwrap();
        t.append(Role::User, "a").unwrap();
        assert_eq!(t.entries_since_compaction(), 1);
        t.append_compaction("summary", (0, 1)).unwrap();
        assert_eq!(t.entries_since_compaction(), 0);
        t.append(Role::Model, "b").unwrap();
        assert_eq!(t.entries_since_compaction(), 1);
    }

    #[test]
    fn complete_final_line_without_a_newline_keeps_its_history_on_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        // The object is complete, so it is history rather than a torn tail;
        // only its terminator is missing.
        std::fs::write(
            &path,
            "{\"role\": \"user\", \"content\": \"first\"}\n{\"role\": \"model\", \"content\": \"second\"}",
        )
        .unwrap();
        {
            let mut t = Transcript::open(&path).unwrap();
            assert_eq!(t.entries().len(), 2);
            t.append(Role::User, "third").unwrap();
        }
        let resumed = Transcript::open(&path).unwrap();
        let contents: Vec<&str> = resumed
            .entries()
            .iter()
            .map(|e| e.content.as_str())
            .collect();
        assert_eq!(contents, vec!["first", "second", "third"]);
    }

    #[test]
    fn torn_final_line_is_truncated_and_the_session_resumes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        // A crash mid-append leaves a partial JSON object with no newline.
        std::fs::write(
            &path,
            "{\"role\": \"user\", \"content\": \"ok\"}\n{\"role\": \"model\", \"cont",
        )
        .unwrap();
        let mut t = Transcript::open(&path).unwrap();
        assert_eq!(t.entries().len(), 1);
        t.append(Role::Model, "recovered").unwrap();
        drop(t);
        let resumed = Transcript::open(&path).unwrap();
        let contents: Vec<&str> = resumed
            .entries()
            .iter()
            .map(|e| e.content.as_str())
            .collect();
        assert_eq!(contents, vec!["ok", "recovered"]);
    }
}
