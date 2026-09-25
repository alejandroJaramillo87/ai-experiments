# Conformance and the cross-language contracts

The two families of test that hold the workspace to something other than its own
numbers: the conformance invariants every backend owes, which say that how a request is
internally divided may not move a bit of any row it returns, and the cross-language
contract tests, which hand an artifact this workspace writes to the real Python reader
that consumes it. Neither claims the numbers are right; the ladder does that. Both
claim a property the ladder cannot see.

## Table of Contents

<!-- toc:start -->

1. [The invariants](#the-invariants)
2. [The long-context suite](#the-long-context-suite)
3. [The cross-language contracts](#the-cross-language-contracts)
4. [Related documentation](#related-documentation)

<!-- toc:end -->

## The invariants

`runtime/src/conformance.rs` states them as functions returning `Result`, so an
acceptance binary or a bring-up harness for a new backend can report a failure in its
own format; they live in a normal module rather than a `#[cfg(test)]` one because a
backend in another crate cannot call into `cpu.rs`'s test module. Exact equality
throughout, never a tolerance: dividing a request differently reorders no summation, so
the only admissible result is the same bits, and a tolerance would hide precisely the
drift these exist to catch.

- **`check_batch_composition_invariance`**: every builder must return identical logits
  at every position. The first builder is the reference; each later one (a stepped
  prefill, groups of 64, one batched pass) is compared against it, and a failure names
  the composition, the position and the one logit that moved rather than printing
  either 262144-wide row. It also checks the cache holds every token, so skipping logits
  cannot skip the forward pass.
- **`check_logit_rows_are_batch_isolated`**: a row must not depend on which other rows
  were asked for. `prefill_from(tokens, first)` must be exactly the tail of
  `prefill_all(tokens)` at every `first` supplied. This rules out a backend that batches
  the head across the rows of one request and lets the batch's composition reach the
  arithmetic; mistral.rs answers the same risk by isolating logit-returning requests
  from each other, and this workspace permits them to share a pass and owes the check
  in exchange.

What the invariants do not claim is stated in the same file: every check compares the
backend against itself, so a backend that is uniformly wrong passes all of them. The
claim is about the output, and the boundary is real: a sliding-window cache is sized
from the group width, so `cached_len` and `rewindable_from` legitimately differ between a
stepped run and a batched one, which is why `Model::rewindable_from` is a query rather
than a constant. What may not differ is a logit.

## The long-context suite

`runtime/tests/long_context.rs` drives the invariants and everything else that only
engages past the sliding window. Nothing else in the workspace reaches that far: the
ladder's longest single cache at Tier T is far inside the real window, so no rung has
crossed one. It is offline, since `synth::tiny_checkpoint` builds a model in memory, and
it runs in `make quality`. Its eighteen tests fall into four groups:

1. **Composition invariance past the window**: every batch composition agrees, an
   all-sliding stack agrees across batch widths, logit rows are the same whether or not
   neighbouring rows were requested, and widening the window changes the weights of
   nothing.
2. **The window's reach**: a windowed model matches a full-attention one below the
   window and diverges past it; a token outside the composed reach of the sliding stack
   changes nothing and one just inside it changes the logits, so the reach is measured
   rather than assumed.
3. **The committed golden and the session paths**: the long-context logits match
   `fixtures/long-context-logits.json`; truncating past the window and re-prefilling
   matches a fresh long prefill; a fork taken past the window diverges identically to a
   fresh session.
4. **KV storage**: a sliding layer stops growing once its window is full, a global layer
   still holds the whole sequence, a shared layer holds no rows of its own, a rewind to
   the floor is allowed and one position below it is not, clearing the cache is allowed
   from below the floor, a snapshot taken past the window restores its rewind floor, and
   a snapshot declaring an impossible base is refused.

The fourth group is a different kind of assertion and is grouped apart for that reason.
Everything above it compares logits, which is what the storage cap must not change; the
storage arms read the cache extent directly, because a suite that only compares logits
cannot tell whether a memory optimization was implemented at all. The file's mutation
record is worth reading before changing the eviction: correctness there has two
independent guards, and defeating either one alone shows up as memory rather than as
numerics.

## The cross-language contracts

A Rust misunderstanding of a format would reproduce itself on both sides of a Rust
test, so each artifact the Python tiers consume is handed to the real Python reader.
The interpreter is resolved by `runtime/src/pyenv.rs` ([testing.md](testing.md)), and a
reader that cannot be imported reports `[SKIP]` with the missing dependency named.

| Test | Hands Python | Through |
|------|--------------|---------|
| `runtime/src/npz.rs` (its tests) | a written activation-capture shard | `np.load`; numpy is mandatory here, not skippable |
| `runtime/tests/python_exporter_contract.rs` | nothing: reads the committed probe and control-vector fixtures the real exporters wrote, plus artifacts built in-test through the same exporters | `interp/probes/export.py`, `interp/steering/export.py` |
| `harness/tests/probe_identity.rs` | the probe's model hash to verify | `hashlib`, so the served digest is an independent SHA-256 |
| `harness/tests/obs_contract.rs` | a run catalog and event file the harness wrote | `common.obs.catalog.read_runs`, whose silent skip of an unparseable line is the failure mode |
| `harness/tests/serving_contract.rs` | a live `--serve` endpoint | `common/serving.py`'s own `wait_healthy` and `chat_completion`, the functions the scorecard and organism commands use |
| `harness/tests/template_contract.rs` | a rendered conversation | the training tier's chat-template renderer, byte for byte, beside a check of the Rust constants against `schemas/chat_template.json` |
| `runtime/tests/status_python_parity.rs` | the six status tags | `common/output.py`'s `Printer`, compared as bytes under `NO_COLOR=1` and a scratch `OBS_LOG_DIR` |

The contracts themselves are the JSON files under `schemas/`, listed with their writers
and readers in [../../schemas/README.md](../../schemas/README.md).

## Related documentation

- [testing.md](testing.md) — the tiers and the gate
- [correctness-ladder.md](correctness-ladder.md) — the claim these tests leave to the ladder
- [fixtures.md](fixtures.md) — the long-context golden and the interp fixtures
- [../../.styles/RUST_STYLE.md](../../.styles/RUST_STYLE.md) — determinism, the `[SKIP]` contract and cross-language verification
