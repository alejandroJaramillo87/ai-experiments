# Harness roadmap

The scheduled work for the rust harness and its interpretability seam, in five phases plus a
parallel track. Each entry states what the item is, why it is placed where it is, and what the
estimate turned out to be worth once the item landed.

Estimates are kept beside outcomes rather than replaced by them. Several entries below were
mis-sized in both directions, and the record of which ones is the only thing that improves the
next estimate.

This document is the plan. The feature specifications the `IF-` items refer to are restated in
[Feature specifications](#feature-specifications) at the end, so the roadmap and the specs
cannot drift apart in separate files. Design ownership stays where it already is:
[runtime-design.md](runtime-design.md) owns the architecture,
[../interp/rust-harness-interp.md](../interp/rust-harness-interp.md) owns the interp seam, and
this file owns only the ordering and the reasoning behind it.

## Table of contents

- [Status](#status)
- [Phase 1 — Foundations](#phase-1--foundations)
- [Phase 2 — The observatory](#phase-2--the-observatory)
- [Phase 3 — Serving and retiring llama.cpp](#phase-3--serving-and-retiring-llamacpp)
- [Phase 4 — Intervention](#phase-4--intervention)
- [Phase 5 — CUDA backend](#phase-5--cuda-backend)
- [Running in parallel](#running-in-parallel)
- [Feature specifications](#feature-specifications)

## Status

| Item | What | Status |
|------|------|--------|
| R1 | L3 batched scoring | done, null result |
| R2 | Long-context conformance rung | done |
| R3 | P11 window storage cap | done |
| R4 | Sampling with recorded seed | done |
| R5 | IF-1 tap site identity | done |
| R6 | IF-2 token-to-text alignment | done |
| R7 | P12 probes to safetensors plus JSON sidecar | done |
| R8 | IF-3 attention weights | done, cheaper than budgeted |
| R9 | P13 harness serves interp | done, verified against real E4B (RH.14) |
| R10 | P14 control vector to safetensors | done; RH.15 acceptance not yet run |
| R11 | IF-4 plus IF-5 together | not started |
| R12 | IF-3 remaining sites | not started |
| R13 | IF-6 activation patching | blocked on G2 |
| R14 | Measurement protocol | not started |
| R15 | P15 documentation and quality review | last |
| G2 | One validated Gemma 4 direction | not started, start now |

## Phase 1 — Foundations

### R1 — L3 batched scoring (done, null result)

The scoring that did not batch was the tied LM head, which read the 1.07 GB embedding table
once per position. `eval::per_token_nll` scores only the second half of each window, so 128 of
256 rows were computed and discarded. `RowsWanted::From` and `Model::prefill_from` let the
caller ask for the suffix.

Bit-identity held perfectly: all five rungs pass with goldens identical to the last digit.
Speed did not move. L3 median 53.06 s across five pinned runs against a predicted 43-49 s band.
Halving the head's work was invisible, which means the head is a far smaller share of L3 than
its byte count implies. The change is kept as a strict deletion of wasted work; it is not a
win.

Two caveats on the record. Only the change arm was run, so there is no pinned baseline to
subtract, and the predicted 134 MB resident-memory reduction did not appear (14.27 GiB,
unchanged), which is either already counted or wrong. Neither is worth a follow-up run.

This result is the direct evidence behind the standing rule against predicting CPU timings, and
it strengthens the case for going to CUDA.

### R2 — Long-context conformance rung (done)

Every rung before this ran short prompts. Gemma 4's entire architectural interest, the
sliding and global split and the shared-KV tail, only engages past the window. A conformance
bug living beyond position 512 was invisible, and every long trace in this plan would have
inherited it silently.

Why it moved to first. Two independent arguments converged. It is the CUDA acceptance test,
the thing that will say whether a second backend agrees. And the `prefill_full_kv` trap that
mistral.rs documents bites exactly at a batch boundary, which is precisely where short-prompt
tests are blind. Build the test that catches R3's bug before building R3.

The masking already existed in `runtime/src/model/cpu.rs`, using
`(position + 1).saturating_sub(win)`, pinned by nothing at the boundary. R2 tests it against
the full-storage cache; R3 then changed storage and R2 was the net.

The matrix, adapted from mistral.rs's `causal_sliding_mask_keeps_exact_window_width`:

- crossing window, crossing window times two, crossing `DEFAULT_BATCH`
- times one-shot prefill, incremental decode, prefill then decode
- times sliding layer, global layer

asserting equality against naive full attention with a mask.

The prefill-then-decode arm earns its place specifically: mistral.rs runs separate prefill and
decode kernels that need not agree bit for bit, and this runtime runs one path that should. A
failure there is structural.

Also landed here: batch-composition invariance stated as a named invariant in one place rather
than as a comment on `forward_batch`. The property largely existed already
(`prefill_from_returns_exactly_the_tail_of_prefill_all` sweeps seven widths); what was missing
was writing it as an obligation the CUDA backend must satisfy. It now lives in
`runtime/src/conformance.rs`. mistral.rs concluded independently that logit-returning requests
must be batch-isolated; this runtime does not carry that constraint, and the absence is now an
asserted test rather than an assumption.

### R3 — P11 window storage cap (done)

Cap `LayerKv` at the window for sliding layers, which makes `budget_bytes` honest. Touches the
forward pass and KV restore, which is why it belongs in this cluster.

Its urgency condition was met and resolved the easy way. Real E4B peaks at 14.27 GiB,
comfortably inside workstation memory, so nothing jumped the queue. P11 was worth doing for
honesty, not urgency.

Smaller than it reads. `runtime/src/kv.rs` already had `budget_bytes` (design) alongside
`resident_bytes` (reality), so the hard part, naming the split, was done. The change was
localized to `LayerKv` plus the `save_state` and `load_state` pair.

The design taken from mistral.rs's `rotating_cache.rs`, without the code: a linear compacting
buffer with slack (capacity is window plus headroom), not a modular ring. On overflow, move the
retained tail to index 0. The returned view is then always contiguous and in natural token
order, with no index remapping, no rotated mask, and no position gymnastics. A ring buffer
would make attention's read order depend on a write position, which a bit-exact substrate
cannot have.

The rewindability constraint, decided up front. A windowed layer is not rewindable past its
window. This is a fact about windowed attention rather than a limitation of this
implementation, so it is enforced rather than discovered mid-experiment:

- `rewindable_from` returns the earliest position the cache can still return to. Callers ask
  instead of guessing.
- `fork` and `save_state` check it explicitly and error, rather than silently producing a short
  prefix.
- Type-level enforcement was considered and rejected. Models pass as `Box<dyn Model>`, and
  trait objects erase exactly the information that technique needs.

Cache size must be knowable before a run, not after, so `budget_bytes` and the resulting
`rewindable_from` go in the run header alongside the existing configuration. An experiment
about to become un-rewindable says so at startup. A standalone calculator is out of scope here.

The trap, named precisely by mistral.rs: early query rows in a multi-token prefill legitimately
need KV that the window will later drop. The bug is reading the retained tail before relocating
the buffer. It surfaces at the 64-token batch boundary and is invisible in short-prompt tests.

### R4 — Sampling with recorded seed (done, much smaller than budgeted)

The premise was wrong and the capability table needed correcting. `runtime/src/sampler.rs`
already implemented `SamplingParams` carrying temperature, top-p and seed; `Sampler::sample`
over logits and a mask; a deterministic 64-bit linear congruential generator; grammar masking
applied before temperature and top-p; and a greedy path at temperature at or below zero using
a strict comparison so ties match `torch.argmax`. The harness already exposed `--temperature`,
`--top-p` and `--seed`.

The actual gap was the flight recorder in `runtime/src/model/record.rs`, which stored the
distribution and not the selection. A replay reproduced the distribution and re-rolled the
dice.

So R4 was: add top-k, and record the seed and the selection alongside the logits. Greedy stays
the default, so no golden moved.

Ordering note, corrected. An earlier draft read "R4 before R5 deliberately: sampling makes the
harness useful, long-context coverage makes it believable." The reasoning flipped once
long-context moved first. It is cheaper to find an architecture bug before building experiments
on top of it than after, and R4 turned out to be a fraction of its estimated size, so it no
longer bought enough time to justify going first.

### R5 — IF-1 tap site identity (done)

Cheapest now, dearest later. Five implementations already depended on the two-field shape, so
landing it beside R2 and R3 cost less than landing it cold.

## Phase 2 — The observatory

### R6 — IF-2 token-to-text alignment (done)

Highest value per unit effort in the whole interp specification. Batch boundaries go in the
artifact, not just the prefill and decode split: with batching landed, a reader otherwise
cannot tell whether position 4012 was computed alongside 4011 or alone.

Worth copying from mistral.rs's `sequence.rs`, four lines: decode each token in isolation,
accumulate bytes, and withhold any delta whose lossy UTF-8 rendering ends in a replacement
character until the multi-token codepoint completes. Also their peek and commit split, where
peeking does not advance the stream index and taking does, which is the right shape for a
recorder that inspects without consuming.

Do not copy their first-token `trim_start` and SentencePiece fixup hacks. Those patch over
information loss; an alignment artifact wants raw bytes plus explicit metadata. They have no
byte-span table and no batch-boundary concept at all, so the artifact itself was still ours to
design.

### R7 — P12 probes to safetensors plus JSON sidecar (done)

Independent of everything, and it retires the probe half of the GGUF artifact schema. The
control vector stayed GGUF at the time because llama.cpp read it and the format was not ours
to choose; the probe had exactly one reader, so nothing outside this repository was served by
keeping it. The contract moved to
[../../schemas/probe_artifact.json](../../schemas/probe_artifact.json). R10 retired the
llama.cpp steering path and the control vector followed.

### R8 — IF-3 attention weights (done, cheaper than budgeted)

The cheapest capture site and immediately useful on its own: watch sliding-versus-global
behaviour directly on long context, which nothing else local can show. One site, not a batch.

**The estimate was wrong, and in the unusual direction.** The budget assumed a second code
path: a tiled online-softmax kernel never materializes the score matrix, so attention-weight
capture would need a non-tiled reference held bit-exact against the fast one, in the shape of
mistral.rs's 63-line `naive.rs`. That premise does not hold for this backend. The CPU forward
pass fills a per-query-head score vector in scratch and softmaxes it in place, so the
distribution already exists as a value and a tap between the softmax and the weighted-value
fold borrows it. Nothing is copied and no arithmetic moves. There is no second function, so the
rule in [../../.styles/RUST_STYLE.md](../../.styles/RUST_STYLE.md) that a fast path must
produce identical bits or take a different name is satisfied by there being one path.

The cost is real, and it belongs to CUDA. A tiled kernel cannot serve this seam by borrowing,
and owes either a non-tiled reference path or a documented refusal. That is recorded against
the tap-parity risk in [runtime-design.md](runtime-design.md) rather than paid here.

Two design decisions the item settled, which bind R12:

- **A trait method, not a `Site` variant.** `Site` keys the residual seam, whose argument is a
  row of width `embed_dim`. An attention distribution is over key timesteps, varies in length
  with the position and the window, and is meaningless without its query head. The rule is now:
  a point that yields a residual row is a `Site`; a point that yields anything else is a
  method.
- **The Gemma 4 per-head-width caveat does not bind this site.** A weight vector's length is a
  key count, not a head width, so the 256 and 512 split cannot mis-slice it. That caveat binds
  the per-head output site in R12 instead.

Confirmation, not a lesson: mistral.rs's `return_raw_logits` is a boolean threaded through
roughly 40 call sites in eight files that disables CUDA graphs, disables prefix-cache reuse,
and forces singleton batching. The tap sidecar in this runtime is already the better design.

After R8 the motivating chain-of-thought faithfulness study runs end to end. That is the
milestone the whole plan aims at.

## Phase 3 — Serving and retiring llama.cpp

### R9 — P13 harness serves interp (done)

`--serve <port>` carries the two paths interp calls, `/v1/chat/completions` and `/v1/health`,
with a `harness` entry on port 8005 in `common/serving.py`. Placed after the observation stack
because experiments run fine through the harness binary; the endpoint is the precondition for
killing the interp log proxy, the replay stage, the replay fidelity caveat and the GPU
contention.

**Nothing was killed here, deliberately.** The proxy, the replay stage and the caveat all remain
until a workstation run shows the endpoint answering interp's own clients against real E4B.
Deleting the working path on the strength of a sandbox test would remove the fallback before the
replacement is evidenced. Those deletions ride with R10, which took the steering half: the
llama.cpp entrypoint blocks and compose variables are gone and `steer-up` recreates the
harness. The proxy, the replay stage and the caveat are still standing, because the
artifact-flushing limitation recorded below is what would make the replay stage genuinely
redundant rather than merely unused.

Three of the original assumptions did not survive contact with the repository, recorded here so
R10 does not rediscover them.

**Streaming was not taken, and would have bought nothing.** The plan was to port mistral.rs's
~120-line streamer with its three-state terminating-sentinel tracker. No interp client streams:
`common/serving.py`'s `post_chat` is a blocking urllib POST, and nothing under `interp/` sends
`stream`, `logprobs`, `n`, `stop`, or tools. The only SSE code in the tree is the proxy's
passthrough reassembler, which exists to forward llama.cpp's stream and dies with the proxy.
Separately, `harness/src/stream.rs` already *is* that streamer — generic over chunk type, with
drop-the-receiver cancellation that is exactly a client disconnect — and it was dead code with no
call sites. The module the plan wanted to port was already in the tree, waiting for a client that
does not exist. Deferred, not rejected.

**No web framework.** The router-as-library shape was taken; the router was not. `RUST_STYLE.md`
bounds the adopt-over-build exception to checkpoint formats and states that nothing may pull a
network client into a gate that is offline by contract. HTTP is not a checkpoint format, and
axum with hyper would have added tokio `net` and `rt-multi-thread`, a large lockfile delta, and
an agent-image rebuild for the prefetched registry. Against that: two paths, one verb, a
length-delimited JSON body, no streaming. Hand-rolled over `std::net`, no new dependency, no
lockfile movement. The costs are keep-alive and chunked request bodies, both refused rather than
faked, and neither is sent by any caller.

**The per-request-seed anti-pattern was already avoided.** mistral.rs shares one engine-global
`Isaac64Rng` seeded from a compile-time constant across all sequences, consumed in scheduling
order, and accepts the OpenAI seed field while ignoring it. `runtime/src/sampler.rs` has no
global RNG: the seed lives in `SamplingParams`, each `Sampler` holds its own state, and the
sampler is passed by reference into `Session::generate` rather than owned by the session. So
honouring a per-request seed cost one `Sampler` per request. `SessionGenerator` bakes one in at
construction, so serving uses `complete_sampled` beside it rather than through it.

**What was forced rather than chosen** is the concurrency shape. `Model` is `Send` but not `Sync`
and every generation path takes `&mut self`, so one generation runs at a time. A sequential
accept loop would also queue health probes behind it, and a CPU-backend reply takes minutes
against a three-second poll and a ten-second compose healthcheck — a busy service would read as a
dead one. Hence a thread per connection with the generator behind a mutex the health handler
never takes.

Two limitations to carry forward. Serving blocks until signalled, so the end-of-run probe-flag
report and KV snapshot are unreachable under `--serve`; flushing interp artifacts from a
long-lived server is unsolved and is the thing that would make the replay stage genuinely
redundant rather than merely unused. And `swap_model` cannot drive the harness, which takes its
checkpoint from `--model` rather than a compose env var, so it refuses the entry by name.

### R10 — P14 control vector to safetensors (done, acceptance pending)

The control vector was the last GGUF artifact this repository wrote for its own consumption,
and it stayed GGUF for one reason: llama.cpp read it as a scaled control vector, so the format
belonged to the consumer. R9 gave the harness an endpoint, this item retired the llama.cpp
steering path, and the choice came back here. GGUF has now left `interp/` and `rust-harness/`
entirely -- including `gguf.rs` and the fixture writer in `synth.rs` -- and stays in
`training/export/` and `common/serving.py`, which produce and name what llama.cpp serves.

**What the plan got wrong, recorded so R11 does not repeat it.**

*The harness was already a compose service.* The plan assumed otherwise and budgeted a new
scorecard controller. `docker-compose.rust-harness.yaml` existed; what it lacked was a daemon,
because before R9 there was no endpoint to serve. So `harness-serve` joined that file and
`steer-up`/`steer-down` were repointed rather than replaced: `STEER_SERVICE` already
parameterized them and `MakeServiceController` already took a service argument.

*The numbering was never off by one.* The retired `direction.` naming was described as
1-based, and both
the exporter and the reader in fact named the 0-based block a direction was added to. What was
1-based was the *floor*: index 0 was refused because repeng numbers from 1 and leaves the
first block unsteerable. Dropping the floor moved no data, and the prefix was renamed in the
same step so a pre-R10 image fails at the reader rather than loading correctly right up until
someone does change the base.

*Deleting the reader cost a test that was not about the reader.*
`scripts/models/gguf_header_transcript.sh` derived its loadability verdicts by parsing
`tensor.rs` and `gguf.rs` at run time, a derivation added to fix a real drift defect. With no
GGUF reader left the verdicts had no referent, so they went and eleven tests went with them.
The transcription the llama.cpp services still depend on is untouched.

**Not yet evidenced.** Everything above is verified in the sandbox against a synthetic model,
which proves the format and the wiring and nothing about steering real weights. RH.15 is the
workstation check, and it also measures what a three-arm sweep costs now that the subject
decodes at 5.5-5.8 tokens per second instead of on the GPU. If that proves impractical it is a
finding against a future GPU backend, not a reason to revert.

## Phase 4 — Intervention

### R11 — IF-4 plus IF-5 together

Position-conditioned steering plus intervention provenance in the recorder. Half a day
together, considerably more retrofitted. The effort estimate should rise from the
specification's: applying a direction at position p is harder now that a range of positions
computes in one pass.

### R12 — IF-3 remaining sites

Per-head output, then MLP pre-activation, one at a time. The per-head output site is where the
Gemma 4 head-width caveat finally bites, and R8's rule decides the shape: a per-head activation
is not a residual row, so it is a trait method rather than a `Site` variant.

### R13 — IF-6 activation patching

Gated on G2. This is the row the CPU backend punishes hardest: the patching sweep is five hours
on CPU against roughly three minutes on CUDA. In practice R13 wants the CUDA backend to exist,
which is an argument for starting CUDA no later than Phase 3.

## Phase 5 — CUDA backend

Replaces the deleted CPU-optimization item. Not scheduled to a phase boundary. It is the long
pole for R13 and should start as soon as the conformance suite from R2 can act as its
acceptance test.

Design notes to carry in:

- mistral.rs independently found 16-bit floating point unusable for Gemma on CPU, because
  activations exceed its maximum of 65504. That is third-party vindication of the bfloat16
  requirement set here from interp constraints.
- They then chose 32-bit activations because candle lacks native bfloat16 elementwise
  operations, paying double the memory for it. The storage-preserving weight representation in
  this runtime, holding bfloat16 and widening per use, sidesteps that tax by construction. CUDA
  should keep the same storage-preserving shape rather than converting on load. This is a place
  the design here is ahead of theirs.
- Expect CUDA to be tolerance-checked against CPU, not bit-matched, because cuBLAS reduction
  order varies with problem shape. Decide this deliberately now rather than discovering it
  during bring-up. It also means the bit-exact CPU path becomes a debugging oracle rather than
  the thing science runs on, which is a real step down from the original ambition and better
  acknowledged than stumbled into.
- The attention-weight tap site from R8 is the second parity obligation, and the harder one.
  See the tap-parity risk in [runtime-design.md](runtime-design.md).

## Running in parallel

### G2 — one validated Gemma 4 direction

Python and GPU work outside this repository, not started, and the long pole for R13. Start now:
it has weeks of lead time and nothing here depends on it until Phase 4.

### R14 — measurement protocol

CPU pinning, N interleaved runs, medians with the full spread. Half a day. Do it the moment
anyone wants a performance number again.

Absorbs the reference-exactness decision note: one paragraph recording that the ban on fused
multiply-add is a declined optimization rather than an active requirement, so nobody
re-derives the question from the lint configuration.

### R15 — P15 documentation and quality review

Last. Plus three specific corrections:

- The first-class statement that the harness reads Hugging Face checkpoints at bfloat16 because
  that is the precision interpretability runs at.
- Update the interp specification's G1 to satisfied. L1 and L2 pass against a transformers
  reference on real E4B, so Rust and Python tensor equivalence is measured rather than
  intended. That is the fact probe portability rests on.
- Correct the capability table's sampling row, per R4.

## Feature specifications

The `IF-` items the roadmap schedules. Effort figures are the original estimates; where an item
has landed, the roadmap entry above records what it actually cost.

### IF-1 — tap site identity

**What.** Widen the tap trait from layer and position to layer, site and position, where site
is an enumeration whose only initial variant is the current one.

**Why.** No new capability. This is pure option value, and it is the only item on the list
whose cost rises the longer it is deferred: five implementations and a Python-side contract
already depend on the present shape, and every future tap point adds another. A trait keyed on
three fields costs the same to write today as one keyed on two.

**Design notes.** Keep the site enumeration non-exhaustive so adding variants does not break
downstream matchers. The capture format and the probe manifest must record which site each row
came from, because silent site ambiguity in a stored artifact is worse than no artifact.
Existing implementations become mechanical filters.

**Prerequisites.** None. **Sandbox.** Fully buildable and testable.

**Tests.** Existing tap invariants unchanged. One new test asserting that a tap ignoring
non-default sites observes exactly what it observed before, bit for bit.

**Effort.** Half a day.

### IF-2 — token-to-text alignment in the score stream

**What.** For each generated position, emit the byte span of decoded output text that the
position produced, alongside the probe score. Both channels in one artifact keyed by a run
identifier, not two files joined after the fact.

**Why.** This is the load-bearing feature for the entire longitudinal research direction. A
probe trace over 40,000 positions is uninterpretable without it: the claim worth making is
"the probe fired while the model was justifying step 3", and what is available is "the probe
fired at position 4012". No amount of probe machinery closes that gap. It is also the
prerequisite for the chain-of-thought faithfulness work, which compares what the model said
against what its internals were doing. If joining those channels is fiddly, the comparison will
not be run often enough to be useful.

**Design notes.** Spans must survive the chat template: positions consumed by template
scaffolding produce no user-visible text and must be marked as such rather than omitted, or
position indices in the artifact stop matching position indices in the run. The prefill and
decode split matters and is recorded per position, because chat mode is decode-heavy and scored
decode positions are the model producing its own text, which is where intent-like attributes
live, while prefill positions are the model reading someone else's. Byte spans, not character
spans, because the tokenizer is byte-oriented and byte-fallback pieces exist. This is ordinary
plumbing requiring no interpretability knowledge and introducing no new numerics.

**Prerequisites.** None. Benefits from IF-1 landing first only for artifact-schema tidiness.
**Sandbox.** Fully buildable and testable against a synthetic fixture and the existing
tokenizer tests.

**Tests.** Round trip: concatenating every position's byte span reproduces the decoded output
exactly. A position whose piece is a byte-fallback fragment still lands in the right span.
Template-scaffolding positions are marked and contribute no output bytes.

**Effort.** Two to three days, most of it in template and byte-fallback edge cases.

### IF-3 — sub-layer read taps

**What.** Additional tap sites inside the layer: attention weights, per-head attention output,
MLP pre-activation, and the feed-forward gate.

**Why.** The residual-only tap makes head-level and neuron-level analysis structurally
impossible. It is the difference between "layer 12 mattered" and "head 12.4 mattered", and the
latter is what "circuit" means. This gap is larger than the read and write gap: adding write
access without adding depth buys the coarse half of circuit work and none of the interesting
half. Attention weights are the cheapest site to expose and are immediately useful independent
of any probe work, because they show Gemma 4's sliding-versus-global attention behaviour
directly on long context, which no other local stack can show.

**Design notes.** Ship one site at a time; each is independently useful and a batch is not.
Start with attention weights, then per-head attention output, then MLP pre-activation. Every
new site is additive and preserves all existing invariants, and there is no ordering hazard
because nothing is being written. Cost discipline: a tap that is not installed must cost
nothing, and because attention-weight capture over a long context is large, the site must be
opt-in per site rather than per tap. Gemma 4 caveat: per-head sites must account for the
per-layer head dimension, 256 on sliding layers against 512 on global ones, and the two
grouped-query mappings. A site that assumes uniform head width will silently mis-slice on half
the layers.

**Prerequisites.** IF-1. **Sandbox.** Fully buildable and testable.

**Tests.** Per site: the captured tensor's shape matches the layer's declared geometry on both
sliding and global layers; installing the tap does not perturb logits; a fixture with mixed
head dimensions is used, not a uniform one.

**Effort.** One to two days per site.

### IF-4 — position-conditioned steering

**What.** Extend the existing control-vector path so a direction can be applied at chosen
positions rather than at every position of a block.

**Why.** Converts steering from a global bias into a localized intervention, at roughly a tenth
of the cost of a full patching seam. It yields ablation by adding the negation of a direction,
amplification, and directional patching by scaling the difference between a target and a source
direction, which together cover a substantial fraction of practical intervention work without
touching the tap trait.

**Design notes.** The position predicate should be a set or range resolved before the forward
pass, not a callback, because a callback in the decode path is a determinism hazard. Preserve
the existing invariant exactly: a zero vector, at any position set, is bit-identical to
unsteered. Gemma 4 caveat: an intervention on the shared-KV tail is not local. Steering a donor
layer moves the inputs of every layer that reads its cache, up to 18 layers on E4B at once.
That is simultaneously an opportunity, being one cheap intervention with wide reach, and an
attribution hazard, because a result reading "layer 24 mattered" means something very different
than it would in a model where every layer owns its cache. Any tooling here must surface the
donor relationship in its output, and any write-up must state it.

**Prerequisites.** None strictly. Cleaner after IF-1. **Sandbox.** Fully buildable and
testable.

**Tests.** A zero vector at an arbitrary position set is bit-identical to unsteered. A vector
applied at position p leaves positions before p bit-identical to unsteered. Steering with a
position set carries across a fork.

**Effort.** One to two days.

### IF-5 — intervention provenance in the flight recorder

**What.** The recording format carries the intervention specification, so a run with steering
or patching applied is replayable as that run.

**Why.** The recorder replays a run from recorded outputs. A patched run is a different run: it
cannot be replayed from an unpatched recording, and a recording made during a patched run is
only meaningful if the intervention is part of the recorded condition. Without this, the
reproducibility guarantee, which is the harness's most distinctive property, quietly stops
covering precisely the experiments that most need it.

**Design notes.** Include the control vector's own hash, not just its path. A replay whose
recorded intervention specification does not match the requested one must fail loudly, in the
same manner as the existing model-hash mismatch.

**Prerequisites.** IF-4, and build it in the same change, because the format is already open at
that point. **Sandbox.** Fully buildable and testable.

**Effort.** Half a day alongside IF-4; considerably more retrofitted.

### IF-6 — mutable residual seam (activation patching)

**What.** A write seam allowing the residual at a specific layer, site and position to be
replaced with a cached value from a different run.

**Why.** True causal claims. This is the only item that cannot be approximated by the features
above.

**Design notes and honest cost.** The trait change is small: a second trait alongside the read
tap, taking a mutable slice and returning whether it wrote, or widening the existing one. The
trait is the easy twenty percent. The expensive part is the cached-activation source: a capture
format, a run identifier, and a guarantee that position p of run A corresponds to position p of
run B. Alignment is the whole problem, and the seam does not solve it.

Where the patch applies is a semantic decision, not an implementation detail. The read tap
fires after the layer output scale. A write seam at the same point writes a post-scale value,
which preserves tap equivalence but makes it impossible to express "perturb the feed-forward
output and let the scale apply". Those are different experiments, and whichever is chosen must
be named in the trait rather than implied by call-site position.

Build the narrowest thing that answers one real question. Generality invented before the second
use case is aimed at the wrong axis.

**Prerequisites.** IF-1, IF-3, IF-4, and G2, meaning at least one validated direction worth
intervening on. Building this before a hypothesis exists produces a precision instrument with
nothing to point at.

**Sandbox.** Buildable, not validatable. The correctness of a patching result cannot be
established against a synthetic fixture.

**Tests.** The existing invariant family gives the shape: a patch whose replacement value
equals the observed value is bit-identical to no patch; a patch at position p leaves positions
before p unchanged; patch, record and replay reproduces the patched run exactly.

**Effort.** Weeks, dominated by the capture and alignment protocol.
