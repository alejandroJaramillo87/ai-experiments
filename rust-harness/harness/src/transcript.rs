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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub role: Role,
    pub content: String,
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
    /// before the final line is still a hard error.
    pub fn open(path: &Path) -> Result<Self> {
        let mut entries = Vec::new();
        let mut truncate_at: Option<u64> = None;
        if path.exists() {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("open {}", path.display()))?;
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
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("open {} for append", path.display()))?;
        if let Some(len) = truncate_at {
            file.set_len(len)
                .with_context(|| format!("truncate torn line in {}", path.display()))?;
        }
        Ok(Self { file, entries })
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn append(&mut self, role: Role, content: impl Into<String>) -> Result<()> {
        let entry = Entry {
            role,
            content: content.into(),
        };
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
