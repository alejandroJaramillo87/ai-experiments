/*
 * record — flight recorder over the `Model` trait.
 *
 * `RecordingModel` wraps any `Model` and appends every forward call — cache
 * start position, input tokens, and the logits produced — to a JSONL stream.
 * A recording loads back as a `MockModel` script keyed by absolute position,
 * so a run captured against the real backend on the workstation replays in a
 * sandbox with no GPU present. Diffing two recordings of the same prompt
 * locates the first divergent position for the divergence-bisection workflow.
 * One position holds one branch: a recording that writes the same position
 * from two different token histories is refused at load rather than merged,
 * see `replay`.
 */

use std::fs::File;
use std::io::{BufRead, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::mock::MockModel;
use super::{Logits, Model, Token};

/// First line of a recording: dimensions needed to rebuild a script.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Header {
    vocab_size: usize,
}

/// Logits observed at one absolute cache position.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Row {
    position: usize,
    logits: Vec<f32>,
}

/// One forward call: where it started, what it consumed, what it produced.
/// `prefill` yields one row (the final position); `prefill_all` yields one
/// row per appended token.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Record {
    start: usize,
    tokens: Vec<Token>,
    rows: Vec<Row>,
}

/// Sink shared by a recording model and all of its forks, so a session tree
/// writes one interleaved stream ordered by call completion.
type SharedWriter = Arc<Mutex<Box<dyn Write + Send>>>;

pub struct RecordingModel {
    inner: Box<dyn Model>,
    writer: SharedWriter,
}

impl RecordingModel {
    /// Wrap `inner`, writing the recording to `writer`. The header is written
    /// immediately so even an empty recording identifies its vocabulary.
    pub fn new(inner: Box<dyn Model>, writer: Box<dyn Write + Send>) -> Result<Self> {
        let header = Header {
            vocab_size: inner.vocab_size(),
        };
        let writer: SharedWriter = Arc::new(Mutex::new(writer));
        write_line(&writer, &header)?;
        Ok(Self { inner, writer })
    }

    /// Wrap `inner`, recording to a file at `path` (truncated if present).
    pub fn to_file(inner: Box<dyn Model>, path: &Path) -> Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("create recording file {}", path.display()))?;
        Self::new(inner, Box::new(BufWriter::new(file)))
    }

    fn record(&self, start: usize, tokens: &[Token], rows: Vec<Row>) -> Result<()> {
        write_line(
            &self.writer,
            &Record {
                start,
                tokens: tokens.to_vec(),
                rows,
            },
        )
    }
}

/// Serialize one value as a JSON line and flush, so the recording is complete
/// up to the last finished call even if the process dies mid-run.
fn write_line<T: Serialize>(writer: &SharedWriter, value: &T) -> Result<()> {
    let mut w = writer.lock().expect("recorder writer poisoned");
    serde_json::to_writer(&mut *w, value).context("serialize recording line")?;
    w.write_all(b"\n").context("write recording line")?;
    w.flush().context("flush recording line")?;
    Ok(())
}

impl Model for RecordingModel {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn cached_len(&self) -> usize {
        self.inner.cached_len()
    }

    fn prefill(&mut self, tokens: &[Token]) -> Result<Logits> {
        let start = self.inner.cached_len();
        let logits = self.inner.prefill(tokens)?;
        self.record(
            start,
            tokens,
            vec![Row {
                position: start + tokens.len() - 1,
                logits: logits.0.clone(),
            }],
        )?;
        Ok(logits)
    }

    fn prefill_all(&mut self, tokens: &[Token]) -> Result<Vec<Logits>> {
        let start = self.inner.cached_len();
        let rows = self.inner.prefill_all(tokens)?;
        self.record(
            start,
            tokens,
            rows.iter()
                .enumerate()
                .map(|(i, l)| Row {
                    position: start + i,
                    logits: l.0.clone(),
                })
                .collect(),
        )?;
        Ok(rows)
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        self.inner.truncate(len)
    }

    fn fork(&self) -> Result<Box<dyn Model>> {
        Ok(Box::new(RecordingModel {
            inner: self.inner.fork()?,
            writer: Arc::clone(&self.writer),
        }))
    }

    // Snapshots pass through: the recording captures forward calls, not
    // cache state, so save/restore belongs to the wrapped backend.
    fn save_state(&self, w: &mut dyn std::io::Write) -> Result<()> {
        self.inner.save_state(w)
    }

    fn load_state(&mut self, r: &mut dyn std::io::Read) -> Result<()> {
        self.inner.load_state(r)
    }
}

/// Load a recording back as a scripted `MockModel`. Rows are keyed by
/// absolute position, so replay is deterministic regardless of how prefill
/// work was batched in the recorded run; positions never recorded fall back
/// to the mock's zero logits.
///
/// A position-keyed script holds one branch. Re-recording a position under
/// the same token history is benign — `Session::ensure_tail_logits` and
/// truncate-then-reprefill both re-record positions verbatim — but a fork or
/// a divergent transcript can record one position from two different
/// prefixes, and the script cannot hold both. That collision is refused with
/// a positioned error instead of resolved last-writer-wins: a replay that
/// silently returns the other branch's logits is indistinguishable from a
/// correct one, and would make divergence bisection name a phantom first
/// divergent position. Keeping both branches would need a branch-keyed
/// recording format, which this one is not.
pub fn replay(reader: impl BufRead) -> Result<MockModel> {
    let mut lines = reader.lines();
    let header_line = lines
        .next()
        .context("empty recording: missing header line")?
        .context("read recording header")?;
    let header: Header = serde_json::from_str(&header_line).context("parse recording header")?;
    // `None` distinguishes a position no row has claimed from one a row
    // claimed with all-zero logits, which zero-filling alone cannot.
    let mut script: Vec<Option<Vec<f32>>> = Vec::new();
    // Token consumed at each absolute position: the only branch identity the
    // recorded `start`/`tokens` metadata carries.
    let mut history: Vec<Option<Token>> = Vec::new();
    for (n, line) in lines.enumerate() {
        let line = line.with_context(|| format!("read recording line {}", n + 2))?;
        let record: Record = serde_json::from_str(&line)
            .with_context(|| format!("parse recording line {}", n + 2))?;
        let end = record.start + record.tokens.len();
        if history.len() < end {
            history.resize(end, None);
        }
        // Lowest position whose token disagrees with an earlier record: rows
        // at or after it descend from a different prefix than the script does.
        let diverged = record
            .tokens
            .iter()
            .enumerate()
            .find(|&(i, &t)| history[record.start + i].is_some_and(|prev| prev != t))
            .map(|(i, _)| record.start + i);
        for row in record.rows {
            anyhow::ensure!(
                row.logits.len() == header.vocab_size,
                "recording line {}: row width {} does not match vocab size {}",
                n + 2,
                row.logits.len(),
                header.vocab_size
            );
            let scripted = matches!(script.get(row.position), Some(Some(_)));
            if let Some(at) = diverged.filter(|&at| scripted && at <= row.position) {
                anyhow::bail!(
                    "recording line {}: position {} is already scripted from a \
                     different token history (the histories diverge at position \
                     {}); a position-keyed replay cannot hold both branches",
                    n + 2,
                    row.position,
                    at
                );
            }
            if script.len() <= row.position {
                script.resize(row.position + 1, None);
            }
            script[row.position] = Some(row.logits);
        }
        for (i, &token) in record.tokens.iter().enumerate() {
            history[record.start + i] = Some(token);
        }
    }
    let script = script
        .into_iter()
        .map(|logits| logits.unwrap_or_else(|| vec![0.0; header.vocab_size]))
        .collect();
    Ok(MockModel::new(header.vocab_size, script))
}

/// Load a recording file as a scripted `MockModel`.
pub fn replay_file(path: &Path) -> Result<MockModel> {
    let file =
        File::open(path).with_context(|| format!("open recording file {}", path.display()))?;
    replay(std::io::BufReader::new(file))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scripted(vocab: usize, len: usize) -> MockModel {
        // Position p peaks at token p % vocab.
        let script = (0..len)
            .map(|p| {
                let mut l = vec![0.0; vocab];
                l[p % vocab] = 1.0;
                l
            })
            .collect();
        MockModel::new(vocab, script)
    }

    fn recording_of(run: impl FnOnce(&mut RecordingModel)) -> Vec<u8> {
        let buf = Arc::new(Mutex::new(Vec::new()));
        struct SharedBuf(Arc<Mutex<Vec<u8>>>);
        impl Write for SharedBuf {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().write(b)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut m = RecordingModel::new(
            Box::new(scripted(4, 8)),
            Box::new(SharedBuf(Arc::clone(&buf))),
        )
        .unwrap();
        run(&mut m);
        let out = buf.lock().unwrap().clone();
        out
    }

    #[test]
    fn header_is_written_at_construction() {
        let bytes = recording_of(|_| {});
        let first = String::from_utf8(bytes).unwrap();
        assert_eq!(first.lines().next().unwrap(), r#"{"vocab_size":4}"#);
    }

    #[test]
    fn prefill_records_start_tokens_and_final_row() {
        let bytes = recording_of(|m| {
            m.prefill(&[9, 9, 9]).unwrap();
        });
        let text = String::from_utf8(bytes).unwrap();
        let record: Record = serde_json::from_str(text.lines().nth(1).unwrap()).unwrap();
        assert_eq!(record.start, 0);
        assert_eq!(record.tokens, vec![9, 9, 9]);
        assert_eq!(record.rows.len(), 1);
        assert_eq!(record.rows[0].position, 2);
        assert_eq!(record.rows[0].logits, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn prefill_all_records_one_row_per_position() {
        let bytes = recording_of(|m| {
            m.prefill_all(&[9, 9]).unwrap();
        });
        let text = String::from_utf8(bytes).unwrap();
        let record: Record = serde_json::from_str(text.lines().nth(1).unwrap()).unwrap();
        assert_eq!(record.rows.len(), 2);
        assert_eq!(record.rows[0].position, 0);
        assert_eq!(record.rows[1].position, 1);
    }

    #[test]
    fn forks_append_to_the_same_recording() {
        let bytes = recording_of(|m| {
            m.prefill(&[1, 2]).unwrap();
            let mut child = m.fork().unwrap();
            child.prefill(&[3]).unwrap();
        });
        let text = String::from_utf8(bytes).unwrap();
        assert_eq!(text.lines().count(), 3); // header + two records
        let child_rec: Record = serde_json::from_str(text.lines().nth(2).unwrap()).unwrap();
        assert_eq!(child_rec.start, 2);
        assert_eq!(child_rec.tokens, vec![3]);
    }

    #[test]
    fn replay_rebuilds_the_position_script() {
        let bytes = recording_of(|m| {
            m.prefill(&[9, 9, 9]).unwrap(); // records position 2 only
        });
        let mut replayed = replay(std::io::BufReader::new(&bytes[..])).unwrap();
        assert_eq!(replayed.vocab_size(), 4);
        // Positions 0-1 were never recorded: zero logits. Position 2 matches.
        assert_eq!(replayed.prefill(&[9]).unwrap().0, vec![0.0; 4]);
        assert_eq!(replayed.prefill(&[9]).unwrap().0, vec![0.0; 4]);
        assert_eq!(replayed.prefill(&[9]).unwrap().0, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn replay_rejects_a_row_wider_than_the_vocab() {
        let bad = b"{\"vocab_size\":2}\n{\"start\":0,\"tokens\":[1],\"rows\":[{\"position\":0,\"logits\":[0.0,0.0,0.0]}]}\n";
        let err = replay(std::io::BufReader::new(&bad[..]))
            .err()
            .expect("expected error");
        assert!(
            err.to_string().contains("does not match vocab size"),
            "{err}"
        );
    }

    #[test]
    fn replay_rejects_a_position_recorded_from_two_different_token_histories() {
        // Branch A holds position 2 after tokens [1,40,41]; branch B forks at
        // position 2 and records the same position after token 99.
        let colliding = concat!(
            "{\"vocab_size\":2}\n",
            "{\"start\":0,\"tokens\":[1,40,41],\"rows\":[{\"position\":2,\"logits\":[1.0,0.0]}]}\n",
            "{\"start\":2,\"tokens\":[99],\"rows\":[{\"position\":2,\"logits\":[0.0,1.0]}]}\n"
        );
        let err = replay(std::io::BufReader::new(colliding.as_bytes()))
            .err()
            .expect("expected error");
        let msg = err.to_string();
        assert!(msg.contains("position 2"), "{err}");
        assert!(msg.contains("different token history"), "{err}");
    }

    #[test]
    fn replay_accepts_a_position_re_recorded_from_the_same_token_history() {
        // truncate-then-reprefill of an identical tail, as ensure_tail_logits
        // and a same-prefix sync both produce: position 2 is written twice
        // from one history, which the script can hold.
        let bytes = recording_of(|m| {
            m.prefill(&[9, 9, 9]).unwrap();
            m.truncate(2).unwrap();
            m.prefill(&[9]).unwrap();
        });
        let text = String::from_utf8(bytes.clone()).unwrap();
        assert_eq!(text.lines().count(), 3); // header + two records
        let mut replayed = replay(std::io::BufReader::new(&bytes[..])).unwrap();
        assert_eq!(
            replayed.prefill(&[9, 9, 9]).unwrap().0,
            vec![0.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn replay_of_an_empty_stream_is_an_error() {
        let err = replay(std::io::BufReader::new(&b""[..]))
            .err()
            .expect("expected error");
        assert!(err.to_string().contains("missing header"), "{err}");
    }
}
