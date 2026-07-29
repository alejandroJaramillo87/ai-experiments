/*
 * transcript — JSONL session persistence with resume.
 *
 * One JSON object per line, appended as the conversation advances, so a
 * crashed or interrupted session resumes from disk without re-running tools.
 */

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
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
    path: PathBuf,
    entries: Vec<Entry>,
}

impl Transcript {
    /// Open a transcript file, resuming any entries already on disk.
    pub fn open(path: &Path) -> Result<Self> {
        let mut entries = Vec::new();
        if path.exists() {
            let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
            for (n, line) in BufReader::new(file).lines().enumerate() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                entries.push(
                    serde_json::from_str(&line)
                        .with_context(|| format!("{}:{}: bad entry", path.display(), n + 1))?,
                );
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            entries,
        })
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn append(&mut self, role: Role, content: impl Into<String>) -> Result<()> {
        let entry = Entry {
            role,
            content: content.into(),
        };
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("append {}", self.path.display()))?;
        writeln!(file, "{}", serde_json::to_string(&entry)?)?;
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
    fn corrupt_line_is_a_positioned_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        std::fs::write(
            &path,
            "{\"role\": \"user\", \"content\": \"ok\"}\nnot json\n",
        )
        .unwrap();
        let err = Transcript::open(&path).unwrap_err();
        assert!(format!("{err:#}").contains(":2:"));
    }
}
