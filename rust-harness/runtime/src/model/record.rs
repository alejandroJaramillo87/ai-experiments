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
 *
 * A recording also carries the *selection* — which token the sampler chose at
 * each position, under which parameters, and how far into its draw stream
 * that left it. Logits alone are a claim about the distribution, and a replay
 * driven from them re-rolls the dice: same file, different transcript, with
 * nothing to see. The selection lines are what make a replay reproduce the
 * run.
 */

use std::fs::File;
use std::io::{BufRead, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::mock::MockModel;
use super::{Logits, Model, Token};

/// First line of a recording: dimensions needed to rebuild a script, and the
/// sampling parameters if the run had any.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Header {
    vocab_size: usize,
    /// Absent when the recorder was constructed without them, which is what
    /// `RecordingModel::new` does and what every recording written before
    /// selections existed looks like. Skipped on serialize rather than
    /// written as null, so those recordings are byte-identical to what they
    /// were and still parse here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sampling: Option<SamplingProvenance>,
}

/// The sampling parameters a recording was produced under.
///
/// A restatement of `sampler::SamplingParams` rather than that type serialized
/// directly: the recording is a file format with its own compatibility
/// obligations, and a field added to the sampler for a reason of its own
/// should not silently change what a recorder writes. The conversion is one
/// function, below, and is where the two are held together.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SamplingProvenance {
    pub temperature: f32,
    pub top_p: f32,
    #[serde(default)]
    pub top_k: Option<usize>,
    pub seed: u64,
}

impl From<crate::sampler::SamplingParams> for SamplingProvenance {
    fn from(p: crate::sampler::SamplingParams) -> Self {
        Self {
            temperature: p.temperature,
            top_p: p.top_p,
            top_k: p.top_k,
            seed: p.seed,
        }
    }
}

impl From<SamplingProvenance> for crate::sampler::SamplingParams {
    fn from(p: SamplingProvenance) -> Self {
        Self {
            temperature: p.temperature,
            top_p: p.top_p,
            top_k: p.top_k,
            seed: p.seed,
        }
    }
}

/// What a sampler chose at one absolute position, and where that left it.
///
/// `draws` is the sampler's cumulative draw count *after* the selection, so a
/// replay resuming at this point seeds from the header and calls
/// `Sampler::skip_draws` with the count from the previous selection. The
/// greedy path draws nothing and reports the count unchanged, which is
/// correct and not a gap: there is nothing to resynchronize.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Selection {
    pub position: usize,
    pub token: Token,
    pub draws: u64,
}

/// Logits observed at one absolute cache position.
///
/// Generic over how the row is held so one definition fixes the JSON shape
/// for both directions: the write path borrows the model's own logits rather
/// than copying them, which at a 262144-entry vocabulary is a megabyte per
/// decoded token, and serde emits the same array either way.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Row<L> {
    position: usize,
    logits: L,
}

/// One forward call: where it started, what it consumed, what it produced.
/// `prefill` yields one row (the final position); `prefill_all` yields one row
/// per appended token, and `prefill_from` one per token from its offset on, so
/// `rows` is not in general as long as `tokens`. Generic for the reason `Row`
/// is.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Record<T, R> {
    start: usize,
    tokens: T,
    rows: R,
}

/// A row as it is written: logits borrowed from the forward call's output.
type RecordedRow<'a> = Row<&'a [f32]>;

/// A record as it is read back, owning everything it parsed.
type ParsedRecord = Record<Vec<Token>, Vec<Row<Vec<f32>>>>;

/// A selection as it is written and read: one field, so the line cannot be
/// confused with a `Record`.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct SelectionLine {
    selected: Selection,
}

/// One non-header line of a recording.
///
/// Untagged with `Forward` first, which keeps every line a pre-selection
/// recorder wrote byte-identical: a forward record requires `start`, `tokens`
/// and `rows`, none of which a selection line carries, and `selected` is a
/// field no forward record has. The two shapes are therefore mutually
/// unparseable rather than merely unlikely to collide, and a test below
/// asserts it instead of leaving the argument in this comment.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
enum Line {
    Forward(ParsedRecord),
    Selected(SelectionLine),
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
    ///
    /// `sampling` states the parameters the run will sample under, or `None`
    /// where the caller has none to state. Selections record either way and
    /// still pin which token was chosen; without the parameters a reader
    /// cannot rebuild the sampler that chose them.
    pub fn new(
        inner: Box<dyn Model>,
        writer: Box<dyn Write + Send>,
        sampling: Option<SamplingProvenance>,
    ) -> Result<Self> {
        let header = Header {
            vocab_size: inner.vocab_size(),
            sampling,
        };
        let writer: SharedWriter = Arc::new(Mutex::new(writer));
        write_line(&writer, &header)?;
        Ok(Self { inner, writer })
    }

    /// Wrap `inner`, recording to a file at `path` (truncated if present).
    /// `sampling` states the parameters the run will sample under, or `None`
    /// where the caller has none to state.
    pub fn to_file(
        inner: Box<dyn Model>,
        path: &Path,
        sampling: Option<SamplingProvenance>,
    ) -> Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("create recording file {}", path.display()))?;
        Self::new(inner, Box::new(BufWriter::new(file)), sampling)
    }

    fn record(&self, start: usize, tokens: &[Token], rows: &[RecordedRow<'_>]) -> Result<()> {
        write_line(
            &self.writer,
            &Record {
                start,
                tokens,
                rows,
            },
        )
    }

    /// Undo the inner prefill when only the recording failed.
    ///
    /// `Model::prefill` promises the cache is unchanged on error, and the
    /// inner model keeps that promise; wrapping it must not break it. The
    /// tokens are already committed by the time the record is written, so a
    /// writer error would otherwise return Err with the cache advanced —
    /// which desynchronizes the length `Session` mirrors, and makes the next
    /// snapshot record a cache the transcript does not describe.
    fn roll_back_on_record_failure(&mut self, start: usize, recorded: Result<()>) -> Result<()> {
        if recorded.is_err() {
            // Best effort: the record error is what the caller needs to see,
            // so a failure to unwind must not mask it.
            let _ = self.inner.truncate(start);
        }
        recorded
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
        // The row borrows the logits about to be returned: serializing them
        // in place is what keeps a decode step from copying the whole row.
        let rows = [Row {
            position: start + tokens.len() - 1,
            logits: &logits.0[..],
        }];
        self.roll_back_on_record_failure(start, self.record(start, tokens, &rows))?;
        Ok(logits)
    }

    fn prefill_all(&mut self, tokens: &[Token]) -> Result<Vec<Logits>> {
        self.prefill_from(tokens, 0)
    }

    // Overridden rather than inherited, for two reasons that both matter. The
    // default would call this type's `prefill_all` and discard the head of the
    // result, which throws away the saving the method exists for -- the inner
    // backend would still fold the embedding table for every dropped row.
    // And it would record those rows: a recording is a claim about what the
    // run produced, and a row the caller never saw is not one. Replay is keyed
    // by absolute position with unrecorded positions falling back, so a
    // recording that starts partway into a prefill is well formed.
    fn prefill_from(&mut self, tokens: &[Token], first: usize) -> Result<Vec<Logits>> {
        let start = self.inner.cached_len();
        let produced = self.inner.prefill_from(tokens, first)?;
        let rows: Vec<RecordedRow<'_>> = produced
            .iter()
            .enumerate()
            .map(|(i, l)| Row {
                position: start + first + i,
                logits: &l.0[..],
            })
            .collect();
        self.roll_back_on_record_failure(start, self.record(start, tokens, &rows))?;
        Ok(produced)
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        self.inner.truncate(len)
    }

    fn record_selection(&self, selection: Selection) -> Result<()> {
        write_line(
            &self.writer,
            &SelectionLine {
                selected: selection,
            },
        )
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

/// Memory one replayed script may claim, across the dense logits table and
/// the token history that indexes it.
const MAX_SCRIPT_BYTES: usize = 8 << 30;

/// Highest position a recording of `vocab_size`-wide rows may name.
///
/// `replay` materializes one dense row per position up to the highest one
/// any row claims, so a `position` field taken on trust turns a corrupt or
/// hostile hundred-byte line into a terabyte `resize`: the allocation error
/// handler aborts the process, which no caller can catch, where this
/// parser's contract is a positioned error. The bound is the memory the
/// loaded script would occupy rather than a guess at a plausible position,
/// so it holds every recording a real run produces — 8191 positions at a
/// 262144-entry production vocabulary, millions at fixture scale.
fn position_bound(vocab_size: usize) -> usize {
    // Per position: one f32 per vocabulary entry, the script slot holding
    // them, and one history slot.
    let per_position = vocab_size
        .saturating_mul(std::mem::size_of::<f32>())
        .saturating_add(std::mem::size_of::<Option<Vec<f32>>>())
        .saturating_add(std::mem::size_of::<Option<Token>>());
    MAX_SCRIPT_BYTES / per_position
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
    Ok(load(reader)?.0)
}

/// Everything a recording says about how its tokens were chosen: the sampling
/// parameters from the header, if it carried any, and the selections in the
/// order they were written.
///
/// A recording with no selections is not an error — the recorder is wrapped
/// around the model, and a run that only prefilled never sampled — so the
/// vector is simply empty and a caller expecting otherwise says so itself.
#[derive(Debug, Default, PartialEq)]
pub struct Recorded {
    pub sampling: Option<SamplingProvenance>,
    pub selections: Vec<Selection>,
}

/// Load a recording as both a scripted `MockModel` and the selection record
/// that says what the run did with it. `replay` is this without the second
/// half, for the callers that only need the script.
pub fn replay_with_selections(reader: impl BufRead) -> Result<(MockModel, Recorded)> {
    load(reader)
}

fn load(reader: impl BufRead) -> Result<(MockModel, Recorded)> {
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
    let mut recorded = Recorded {
        sampling: header.sampling,
        selections: Vec::new(),
    };
    let bound = position_bound(header.vocab_size);
    for (n, line) in lines.enumerate() {
        let line = line.with_context(|| format!("read recording line {}", n + 2))?;
        let record = match serde_json::from_str::<Line>(&line)
            .with_context(|| format!("parse recording line {}", n + 2))?
        {
            Line::Forward(record) => record,
            // Selections carry no logits, so they neither extend the script
            // nor participate in the branch checks below.
            Line::Selected(line) => {
                recorded.selections.push(line.selected);
                continue;
            }
        };
        // Saturating: a `start` near the top of the usize range must reach
        // the bound check below rather than overflow on the way to it.
        let end = record.start.saturating_add(record.tokens.len());
        anyhow::ensure!(
            end <= bound,
            "recording line {}: record covers positions {}..{}, beyond the {} \
             positions a recording of {}-wide rows may script",
            n + 2,
            record.start,
            end,
            bound,
            header.vocab_size
        );
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
            anyhow::ensure!(
                row.position < bound,
                "recording line {}: position {} is beyond the {} positions a \
                 recording of {}-wide rows may script",
                n + 2,
                row.position,
                bound,
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
    Ok((MockModel::new(header.vocab_size, script), recorded))
}

/// Load a recording file as a scripted `MockModel`.
pub fn replay_file(path: &Path) -> Result<MockModel> {
    Ok(replay_file_with_selections(path)?.0)
}

/// Load a recording file as a script plus its selection record.
pub fn replay_file_with_selections(path: &Path) -> Result<(MockModel, Recorded)> {
    let file =
        File::open(path).with_context(|| format!("open recording file {}", path.display()))?;
    replay_with_selections(std::io::BufReader::new(file))
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
        recording_with(None, run)
    }

    /// The same, stating the sampling parameters the run was made under.
    fn recording_with(
        sampling: Option<SamplingProvenance>,
        run: impl FnOnce(&mut RecordingModel),
    ) -> Vec<u8> {
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
            sampling,
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
        let record: ParsedRecord = serde_json::from_str(text.lines().nth(1).unwrap()).unwrap();
        assert_eq!(record.start, 0);
        assert_eq!(record.tokens, vec![9, 9, 9]);
        assert_eq!(record.rows.len(), 1);
        assert_eq!(record.rows[0].position, 2);
        assert_eq!(record.rows[0].logits, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn a_recorded_line_is_the_exact_json_text_a_determinism_oracle_hashes() {
        // Two recordings of one run are compared by digest, so the write path
        // may not vary its bytes. Serializing the logits where the forward
        // call left them, rather than through a copy, has to emit the array
        // serde emits for an owned row — down to the float formatting.
        let bytes = recording_of(|m| {
            m.prefill(&[1, 2]).unwrap();
        });
        let text = String::from_utf8(bytes).unwrap();
        let expected = concat!(
            r#"{"vocab_size":4}"#,
            "\n",
            r#"{"start":0,"tokens":[1,2],"rows":[{"position":1,"logits":[0.0,1.0,0.0,0.0]}]}"#,
            "\n",
        );
        assert_eq!(text, expected);
    }

    #[test]
    fn prefill_all_records_one_row_per_position() {
        let bytes = recording_of(|m| {
            m.prefill_all(&[9, 9]).unwrap();
        });
        let text = String::from_utf8(bytes).unwrap();
        let record: ParsedRecord = serde_json::from_str(text.lines().nth(1).unwrap()).unwrap();
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
        let child_rec: ParsedRecord = serde_json::from_str(text.lines().nth(2).unwrap()).unwrap();
        assert_eq!(child_rec.start, 2);
        assert_eq!(child_rec.tokens, vec![3]);
    }

    /// Writer that accepts `lines` complete lines, then fails. Counting
    /// newlines rather than calls keeps it independent of how many writes
    /// the serializer happens to issue per line.
    struct FailsAfter {
        lines: usize,
        seen: usize,
    }
    impl Write for FailsAfter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.seen >= self.lines {
                return Err(std::io::Error::new(std::io::ErrorKind::StorageFull, "full"));
            }
            self.seen += buf.iter().filter(|&&b| b == b'\n').count();
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_failed_record_write_leaves_the_inner_cache_unchanged() {
        // Model::prefill promises the cache is unchanged on error. The inner
        // model keeps that promise, so the wrapper must not break it: the
        // tokens are committed before the record is written, and returning
        // Err with them still committed desynchronizes the length Session
        // mirrors and makes the next snapshot describe a cache the transcript
        // does not.
        let inner = MockModel::new(8, vec![]);
        let mut m = RecordingModel::new(
            Box::new(inner),
            Box::new(FailsAfter { lines: 2, seen: 0 }),
            None,
        )
        .unwrap();
        m.prefill(&[1, 2]).unwrap();
        let committed = m.cached_len();

        let err = m.prefill(&[3, 4, 5]).unwrap_err();
        assert!(err.to_string().contains("recording"), "{err}");
        assert_eq!(
            m.cached_len(),
            committed,
            "a failed record left the inner cache advanced"
        );
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
    fn the_position_bound_holds_a_real_recording_at_a_production_vocabulary() {
        // The bound exists to refuse corruption, not real recordings: a
        // 262144-entry vocabulary still scripts thousands of positions, and
        // no vocabulary reaches the trillionth position of a corrupt line.
        assert!(position_bound(262144) >= 8000, "{}", position_bound(262144));
        assert!(position_bound(2) < 1_000_000_000_000);
    }

    #[test]
    fn replay_rejects_a_row_position_beyond_the_script_bound() {
        // Ninety-odd bytes claiming the trillionth position: taken on trust,
        // the script resize asks the allocator for terabytes and the process
        // aborts through the allocation error handler instead of returning.
        let bad = concat!(
            "{\"vocab_size\":2}\n",
            "{\"start\":0,\"tokens\":[1],\"rows\":[{\"position\":999999999999,\"logits\":[0.0,0.0]}]}\n"
        );
        let err = replay(std::io::BufReader::new(bad.as_bytes()))
            .err()
            .expect("expected error");
        let msg = err.to_string();
        assert!(msg.contains("position 999999999999"), "{err}");
        assert!(msg.contains("may script"), "{err}");
    }

    #[test]
    fn replay_rejects_a_record_starting_beyond_the_script_bound() {
        // The token history is sized from `start` before any row is read, so
        // the bound holds there too — and a `start` at the top of the usize
        // range must reach the check rather than overflow on the way to it.
        let bad = concat!(
            "{\"vocab_size\":2}\n",
            "{\"start\":18446744073709551615,\"tokens\":[1],\"rows\":[]}\n"
        );
        let err = replay(std::io::BufReader::new(bad.as_bytes()))
            .err()
            .expect("expected error");
        assert!(err.to_string().contains("18446744073709551615"), "{err}");
    }

    #[test]
    fn a_forward_record_and_a_selection_line_are_mutually_unparseable() {
        // The untagged Line enum is only safe because the two shapes cannot
        // be read as each other: a forward record requires start, tokens and
        // rows, and a selection line carries none of them. Asserted rather
        // than argued, since a field added to either type could quietly make
        // one shape a subset of the other and route lines to the wrong arm.
        let forward = r#"{"start":0,"tokens":[1],"rows":[{"position":0,"logits":[0.0,1.0]}]}"#;
        let selection = r#"{"selected":{"position":3,"token":7,"draws":4}}"#;
        assert!(serde_json::from_str::<ParsedRecord>(selection).is_err());
        assert!(serde_json::from_str::<SelectionLine>(forward).is_err());
        assert!(matches!(
            serde_json::from_str::<Line>(forward).unwrap(),
            Line::Forward(_)
        ));
        assert!(matches!(
            serde_json::from_str::<Line>(selection).unwrap(),
            Line::Selected(_)
        ));
    }

    #[test]
    fn a_recording_without_sampling_parameters_writes_the_header_it_always_did() {
        // The header field is skipped when absent rather than written null,
        // so every recording made before selections existed is byte-identical
        // to what this writer now produces -- which is what keeps the
        // determinism digest above meaningful across the format change.
        assert_eq!(
            serde_json::to_string(&Header {
                vocab_size: 4,
                sampling: None
            })
            .unwrap(),
            r#"{"vocab_size":4}"#
        );
        let old = b"{\"vocab_size\":2}\n{\"start\":0,\"tokens\":[1],\"rows\":[]}\n";
        let (_, recorded) = replay_with_selections(std::io::BufReader::new(&old[..])).unwrap();
        assert_eq!(recorded, Recorded::default());
    }

    #[test]
    fn the_header_carries_the_sampling_parameters_a_replay_needs() {
        let params = crate::sampler::SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: Some(40),
            seed: 7,
        };
        // Through the writer and back through the reader, rather than
        // asserting on the struct: what a replay recovers is the property.
        let bytes = recording_with(Some(params.into()), |_| {});
        let (_, recorded) = replay_with_selections(std::io::BufReader::new(&bytes[..])).unwrap();
        let back: crate::sampler::SamplingParams = recorded.sampling.unwrap().into();
        assert_eq!(back.top_k, Some(40));
        assert_eq!(back.seed, 7);
        assert_eq!(back.temperature, params.temperature);
        assert_eq!(back.top_p, params.top_p);
    }

    #[test]
    fn selections_are_recorded_and_read_back_in_order() {
        let bytes = recording_of(|m| {
            m.prefill(&[9, 9]).unwrap();
            m.record_selection(Selection {
                position: 2,
                token: 3,
                draws: 1,
            })
            .unwrap();
            m.record_selection(Selection {
                position: 3,
                token: 1,
                draws: 2,
            })
            .unwrap();
        });
        let (_, recorded) = replay_with_selections(std::io::BufReader::new(&bytes[..])).unwrap();
        assert_eq!(
            recorded.selections,
            vec![
                Selection {
                    position: 2,
                    token: 3,
                    draws: 1
                },
                Selection {
                    position: 3,
                    token: 1,
                    draws: 2
                },
            ]
        );
    }

    #[test]
    fn selection_lines_do_not_enter_the_position_script() {
        // A selection names a position, and the script is position-keyed, so
        // the one thing that must not happen is a selection overwriting the
        // logits recorded at the same position.
        let bytes = recording_of(|m| {
            m.prefill(&[9, 9, 9]).unwrap(); // records position 2
            m.record_selection(Selection {
                position: 2,
                token: 0,
                draws: 1,
            })
            .unwrap();
        });
        let mut replayed = replay(std::io::BufReader::new(&bytes[..])).unwrap();
        replayed.prefill(&[9, 9]).unwrap();
        assert_eq!(replayed.prefill(&[9]).unwrap().0, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn replay_of_an_empty_stream_is_an_error() {
        let err = replay(std::io::BufReader::new(&b""[..]))
            .err()
            .expect("expected error");
        assert!(err.to_string().contains("missing header"), "{err}");
    }
}
