/*
 * stream — token streaming from the sampler to the terminal renderer.
 *
 * A bounded tokio channel decouples generation from rendering. Cancellation
 * happens at a token boundary: dropping the receiver makes the next send
 * fail, which the generation side treats as a stop request rather than an
 * error.
 */

use tokio::sync::mpsc;

pub fn channel(capacity: usize) -> (TokenSink, TokenStream) {
    let (tx, rx) = mpsc::channel(capacity);
    (TokenSink { tx }, TokenStream { rx })
}

pub struct TokenSink {
    tx: mpsc::Sender<String>,
}

impl TokenSink {
    /// Send one token's text from a synchronous producer thread. Returns
    /// false when the consumer has cancelled; the producer must stop
    /// generating at this boundary. Panics if called on a runtime thread —
    /// async producers (the subagent interleaver) use `emit_async`.
    pub fn emit(&self, text: String) -> bool {
        self.tx.blocking_send(text).is_ok()
    }

    /// Async counterpart of `emit`, for producers running on the runtime.
    pub async fn emit_async(&self, text: String) -> bool {
        self.tx.send(text).await.is_ok()
    }
}

pub struct TokenStream {
    rx: mpsc::Receiver<String>,
}

impl TokenStream {
    /// Next token text, or None once the producer finished.
    pub async fn next(&mut self) -> Option<String> {
        self.rx.recv().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_arrive_in_order_and_close_on_producer_drop() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let (sink, mut stream) = channel(4);
        let producer = std::thread::spawn(move || {
            for t in ["fn", " main", "()"] {
                assert!(sink.emit(t.to_string()));
            }
        });
        let collected = rt.block_on(async {
            let mut v = Vec::new();
            while let Some(t) = stream.next().await {
                v.push(t);
            }
            v
        });
        producer.join().unwrap();
        assert_eq!(collected, vec!["fn", " main", "()"]);
    }

    #[test]
    fn dropping_the_stream_cancels_the_producer_at_a_token_boundary() {
        let (sink, stream) = channel(1);
        drop(stream);
        assert!(!sink.emit("late".to_string()));
    }

    #[tokio::test]
    async fn emit_async_works_inside_a_runtime() {
        // `emit` would panic here (blocking_send on a runtime thread).
        let (sink, mut stream) = channel(2);
        assert!(sink.emit_async("tok".to_string()).await);
        drop(sink);
        assert_eq!(stream.next().await, Some("tok".to_string()));
        assert_eq!(stream.next().await, None);
    }
}
