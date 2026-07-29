/*
 * mock — scripted `Model` implementation for Tier 2 tests.
 *
 * Logits are keyed by absolute token position so replay is deterministic
 * across forks and truncation. Every prefill call is recorded through a
 * recorder shared with all forks, letting tests assert exactly how much
 * prefill work happened anywhere in a session tree (e.g. "spawn paid no
 * prefix re-prefill"). An in-flight counter records the maximum number of
 * concurrent prefill calls, backing the phase E single-generation invariant.
 */

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{ensure, Result};

use super::{Logits, Model, Token};

/// One recorded prefill call: the cache position it started at and the tokens
/// it processed.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillRecord {
    pub start: usize,
    pub tokens: Vec<Token>,
}

#[derive(Default)]
struct Shared {
    records: Mutex<Vec<PrefillRecord>>,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

/// Observer over a mock family's shared prefill recorder.
#[derive(Clone)]
pub struct Recorder(Arc<Shared>);

impl Recorder {
    pub fn records(&self) -> Vec<PrefillRecord> {
        self.0.records.lock().unwrap().clone()
    }

    pub fn total_prefilled(&self) -> usize {
        self.records().iter().map(|r| r.tokens.len()).sum()
    }

    pub fn max_in_flight(&self) -> usize {
        self.0.max_in_flight.load(Ordering::SeqCst)
    }
}

/// Scripted mock model. Position `p` in the script is the logits returned
/// when the final cached token sits at position `p`; positions beyond the
/// script yield uniform zero logits.
pub struct MockModel {
    vocab_size: usize,
    script: Arc<Vec<Vec<f32>>>,
    cached_len: usize,
    shared: Arc<Shared>,
}

impl MockModel {
    pub fn new(vocab_size: usize, script: Vec<Vec<f32>>) -> Self {
        for (i, l) in script.iter().enumerate() {
            assert_eq!(l.len(), vocab_size, "script entry {i} has wrong width");
        }
        Self {
            vocab_size,
            script: Arc::new(script),
            cached_len: 0,
            shared: Arc::new(Shared::default()),
        }
    }

    /// Detached handle onto the recorder shared by this model and all forks,
    /// usable after the model itself is boxed into a Session.
    pub fn recorder(&self) -> Recorder {
        Recorder(Arc::clone(&self.shared))
    }

    /// All prefill calls recorded across this model and every fork of it.
    pub fn records(&self) -> Vec<PrefillRecord> {
        self.recorder().records()
    }

    /// Total tokens prefilled across this model and every fork of it.
    pub fn total_prefilled(&self) -> usize {
        self.recorder().total_prefilled()
    }

    /// Maximum number of prefill calls ever concurrently in flight.
    pub fn max_in_flight(&self) -> usize {
        self.recorder().max_in_flight()
    }

    fn logits_at(&self, position: usize) -> Logits {
        match self.script.get(position) {
            Some(l) => Logits(l.clone()),
            None => Logits(vec![0.0; self.vocab_size]),
        }
    }
}

impl Model for MockModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn cached_len(&self) -> usize {
        self.cached_len
    }

    fn prefill(&mut self, tokens: &[Token]) -> Result<Logits> {
        ensure!(!tokens.is_empty(), "prefill of zero tokens");
        let n = self.shared.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.shared.max_in_flight.fetch_max(n, Ordering::SeqCst);
        self.shared.records.lock().unwrap().push(PrefillRecord {
            start: self.cached_len,
            tokens: tokens.to_vec(),
        });
        self.cached_len += tokens.len();
        let logits = self.logits_at(self.cached_len - 1);
        self.shared.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(logits)
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        ensure!(len <= self.cached_len, "truncate beyond cached length");
        self.cached_len = len;
        Ok(())
    }

    fn fork(&self) -> Result<Box<dyn Model>> {
        Ok(Box::new(MockModel {
            vocab_size: self.vocab_size,
            script: Arc::clone(&self.script),
            cached_len: self.cached_len,
            shared: Arc::clone(&self.shared),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scripted() -> MockModel {
        // Vocab of 4; position p peaks at token p % 4.
        let script = (0..8)
            .map(|p| {
                let mut l = vec![0.0; 4];
                l[p % 4] = 1.0;
                l
            })
            .collect();
        MockModel::new(4, script)
    }

    #[test]
    fn prefill_extends_cache_and_returns_final_position_logits() {
        let mut m = scripted();
        let logits = m.prefill(&[9, 9, 9]).unwrap();
        assert_eq!(m.cached_len(), 3);
        assert_eq!(logits.0, vec![0.0, 0.0, 1.0, 0.0]); // position 2
    }

    #[test]
    fn decode_step_is_prefill_of_one_token() {
        let mut m = scripted();
        m.prefill(&[9, 9]).unwrap();
        let logits = m.prefill(&[7]).unwrap();
        assert_eq!(m.cached_len(), 3);
        assert_eq!(logits.0, vec![0.0, 0.0, 1.0, 0.0]); // position 2
    }

    #[test]
    fn positions_beyond_script_yield_zero_logits() {
        let mut m = scripted();
        let logits = m.prefill(&[0; 20]).unwrap();
        assert_eq!(logits.0, vec![0.0; 4]);
    }

    #[test]
    fn truncate_then_reprefill_replays_from_split_point() {
        let mut m = scripted();
        m.prefill(&[1, 2, 3, 4]).unwrap();
        m.truncate(2).unwrap();
        assert_eq!(m.cached_len(), 2);
        let logits = m.prefill(&[5]).unwrap();
        assert_eq!(logits.0, vec![0.0, 0.0, 1.0, 0.0]); // position 2 again
        let last = m.records().pop().unwrap();
        assert_eq!(
            last,
            PrefillRecord {
                start: 2,
                tokens: vec![5]
            }
        );
    }

    #[test]
    fn truncate_beyond_cache_is_an_error() {
        let mut m = scripted();
        m.prefill(&[1]).unwrap();
        assert!(m.truncate(2).is_err());
    }

    #[test]
    fn fork_inherits_prefix_without_reprefill() {
        let mut parent = scripted();
        parent.prefill(&[1, 2, 3]).unwrap();
        let mut child = parent.fork().unwrap();
        assert_eq!(child.cached_len(), 3);
        child.prefill(&[4, 5]).unwrap();
        // Shared recorder: total prefilled work is prefix + delta, with the
        // prefix paid exactly once.
        assert_eq!(parent.total_prefilled(), 5);
        assert_eq!(
            parent.records(),
            vec![
                PrefillRecord {
                    start: 0,
                    tokens: vec![1, 2, 3]
                },
                PrefillRecord {
                    start: 3,
                    tokens: vec![4, 5]
                },
            ]
        );
    }

    #[test]
    fn fork_state_is_independent_after_spawn() {
        let mut parent = scripted();
        parent.prefill(&[1, 2]).unwrap();
        let mut child = parent.fork().unwrap();
        child.prefill(&[3]).unwrap();
        parent.truncate(1).unwrap();
        assert_eq!(parent.cached_len(), 1);
        assert_eq!(child.cached_len(), 3);
    }

    #[test]
    fn sequential_calls_never_overlap() {
        let mut m = scripted();
        m.prefill(&[1]).unwrap();
        m.prefill(&[2]).unwrap();
        assert_eq!(m.max_in_flight(), 1);
    }
}
