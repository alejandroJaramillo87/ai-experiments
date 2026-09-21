# The correctness ladder

The seven-rung gate that measures the engine against `transformers`, rung by rung: what
each rung compares, the artifact it reads and the generator that writes it, the
tolerance it is judged against at each of the two tiers, the perplexity gate at L3 in
full, what the loader refuses before a rung runs, and the sizes, flags and records of a
climb. Verdict logic is `runtime/src/ladder.rs`; the binary in `runtime/src/bin/ladder.rs`
is the driver that loads a checkpoint and runs a forward pass, and is kept thin for that
reason.

## Table of Contents

- [Why a ladder](#why-a-ladder)
- [The rungs](#the-rungs)
- [Two oracles, two tolerances](#two-oracles-two-tolerances)
- [The perplexity gate](#the-perplexity-gate)
- [The reference artifacts](#the-reference-artifacts)
- [What the loader refuses](#what-the-loader-refuses)
- [Sizes, flags and records](#sizes-flags-and-records)
- [Status](#status)
- [Related documentation](#related-documentation)

## Why a ladder

Machinery tests (Tiers 1 and 2) cannot detect a numerically wrong forward
pass: a bad RoPE base or quant kernel still emits fluent text. Nor can a
golden test whose oracle we wrote ourselves — `fixtures/reference_logits.py`
was transcribed from the same architecture documents by the same reader as
`cpu.rs`, and two implementations of one belief agree perfectly. That gate
stayed green for nine work packages while the forward pass produced noise.

Output correctness is therefore measured against `transformers`, in seven
rungs, each one only meaningful once the one above it passes:

| Rung | Measures | Reference |
|------|----------|-----------|
| L0 | token IDs for the rendered prompt | `tok.encode(text, add_special_tokens=False)` |
| L1 | per-layer residual stream and the stages before it | forward hooks on the decoder layers |
| L2 | the final-position distribution | `model(...).logits` |
| L3 | perplexity over a token window | the reference's per-token NLL |
| L4 | greedy continuations | `model.generate(do_sample=False)` |
| L5 | logits at anchors past the sliding window | `model(...).logits` on a 4-window prompt |
| L6 | post-softmax attention rows at the same anchors, every query head of four layers | forward hooks on the attention modules under eager attention |

L2 is KL divergence over the full distribution rather than a max-abs
tolerance or argmax agreement: KL in nats is scale-free, which is what a
262144-wide softcapped vocabulary needs, and a monotone error such as a
missing logit softcap cannot move an argmax at all.

The ladder short-circuits at the first failing rung and prints all seven rows
regardless. A rung with no reference artifact reports `skipped` — missing

The ladder short-circuits at the first failing rung and prints all seven rows
regardless. A rung with no reference artifact reports `skipped`, which is missing
evidence rather than contrary evidence, and does not stop the climb; a ladder whose
every rung skipped is reported as such and not as "within tolerance". A threshold is
never tuned to make the current run pass: a value that lands between "clearly fine" and
the threshold is a finding, and moving the threshold to swallow it converts a finding
into a green gate, which is the failure the whole package answers.

## The rungs

Each rung is meaningful only once the one above it passes: a wrong tokenizer makes every
later comparison a comparison of different prompts, and a wrong residual stream makes
the distribution rungs measure the wrong thing.

| Rung | Measures, as `Measure` rows | Artifact | Generator |
|------|-----------------------------|----------|-----------|
| L0 token ids | exact equality of every case's ids, with the case's special-token allowlist recorded beside its text | `tokenizer_parity.json` | `reference_tokenizer.py` |
| L1 residual stream | worst cosine (direction) and worst `rel_l2` (magnitude) over every layer and the stages before the stack; the first layer past tolerance is named | `layers.json` | `reference_capture.py` |
| L2 final distribution | mean KL and worst single-position KL in nats, worst top-50 overlap, argmax agreement, over the scored positions | `layers.json` | the same capture |
| L3 perplexity | relative perplexity delta, and the worst single-token NLL delta | `ppl.json` | `reference_perplexity.py --size` |
| L4 greedy continuation | position-wise token agreement, and the mean common prefix as a fraction of the continuation | `greedy.json` | `reference_greedy.py --size` |
| L5 past the window | the L2 measures at six anchors past the sliding window, over a prompt four windows long, after checking both sides scored the same positions | `long_context.json` | `reference_long_context.py` |
| L6 attention rows | every captured layer agrees, no weight outside the window, row mass within `ATTENTION_MASS` of 1, worst single-key absolute difference, worst top-10 key overlap | `attention.json` | `reference_attention.py` |

Two measures per rung is the rule where one would hide a defect. L1 keeps cosine and
`rel_l2` apart because a residual stream correct in shape and wrong in size passes any
cosine floor, which is the signature of a missing `layer_scalar`. L2 keeps the worst KL
beside the mean because one badly wrong position hides among many right ones, and one
position is all a rare-path bug touches; argmax agreement is reported but never the
only measure, since a monotone error such as a missing logit softcap cannot move an
argmax. L4 keeps the prefix beside the agreement because greedy decoding is
autoregressive: one late wrong token barely moves agreement but halves the prefix, and
everything after a divergence is a different run rather than more evidence. L6 is
absolute rather than KL because an attention row is a probability already and most of
it is near zero; a KL would let a wrong weight on a light key hide, which is what a
leaked key looks like.

## Two oracles, two tolerances

Tier T is a tiny Gemma 4 that `transformers` itself constructs and saves,
`fixtures/tiny-hf-gemma4/`, committed whole and climbed in milliseconds by
`runtime/tests/ladder_tiny.rs` inside `make quality`. Tier E is the real checkpoint on
the workstation, its artifacts generated per run. The two are not substitutes: Tier T
catches structure (a transposed weight, a missing scale, a norm convention) on an
oracle nobody here transcribed; Tier E catches what a handful of dimensions cannot
distinguish. Tier T's L5 and L6 artifacts (`long_context.json`, `attention.json` beside
the fixture) are generated on the workstation like Tier E's and are not committed, so
in the sandbox those two rungs report `[SKIP]` with the generator command; the gate is
green with them skipped, which is the reason to read its output. Their tolerances are one `Tolerance` struct with two constants, so a
caller cannot mix the tiny fixture's cosine floor with the real checkpoint's KL ceiling:

| Field | Rung | `TIER_T` | `TIER_E` |
|-------|------|----------|----------|
| `cosine` | L1 floor | 0.99999 | 0.9999 |
| `rel_l2` | L1 ceiling | 1e-5 | 1e-3 |
| `mean_kl` | L2 ceiling, nats | 1e-6 | 1e-3 |
| `worst_kl` | L2 ceiling, nats | 1e-5 | 5e-3 |
| `top_k` | L2 depth | 50 | 50 |
| `overlap` | L2 floor | 1.0 | 0.98 |
| `argmax` | L2 floor | 1.0 | 0.98 |
| `attention_max_abs` | L6 ceiling | 1e-5 | 1e-3 |
| `attention_overlap` | L6 floor | 1.0 | 0.9 |

The untiered thresholds, in `ladder::thresholds`: `PPL_DELTA` 0.01 and
`PPL_MAX_TOKEN_DELTA` 0.05 nats for L3, `GREEDY_AGREEMENT` 0.98 and `GREEDY_PREFIX` 0.75
for L4, `ATTENTION_MASS` 1e-5 and `ATTENTION_TOP_K` 10 for L6. Tier T is tight on
purpose: f32 against f32 on a few dozen dimensions leaves nothing for a tolerance to
absorb except accumulation order, so a run that needs more is wrong. Tier E is looser
because thousands of dimensions and dozens of layers accumulate reordering
differences. Every threshold is marked PROVISIONAL in the source apart from Tier E's
two L1 numbers, which `layer-diff` had already argued against real weights; the source
collects them in one place so the provisional ones can be counted and a change to one
is a diff a reviewer sees.

## The perplexity gate

L3 is the regression scalar for engine changes: a wrong RoPE base, a transposed weight
or a bad kernel moves perplexity sharply while spot checks stay plausible. It is a
differential measurement, ours against the reference's on the same token ids from the
same checkpoint, so the absolute number carries no claim; only the difference is under
test.

**The corpus.** `fixtures/reference/corpus.txt` is committed and is not in this
repository's voice: *The Adventures of Sherlock Holmes*, Project Gutenberg ebook 1661
with Gutenberg's packaging stripped, fetched and normalized deterministically by
`fetch_corpus.py` so the bytes are reproducible rather than pasted. Not WikiText-2: L3
never compares against a published figure, and ordinary narrative prose is a numerical
choice, since off-distribution input widens the gap between two implementations of one
model measurably (the same prompt captured without a BOS gave `rel_L2` 3.3e-5 where the
in-distribution version gave 5.0e-6).

**The windowing**, `runtime/src/eval.rs`, reproduced exactly by
`reference_perplexity.py`: the tokenized corpus splits into windows of `CHUNK` = 256
tokens with the trailing partial window dropped; each window is prefilled from an
empty cache; positions 128 to 254 of each window score the next token, so every scored
token has at least half a window of context; the scored token's corpus index is
`w * 256 + p + 1`. No BOS, because each window after the first begins mid-document by
construction and marking window zero alone would make it a different measurement. The
artifact writes the scored positions out explicitly, and the rung errors rather than
fails when the two sides disagree about which tokens were scored: a windowing mismatch
is a broken artifact, not a numerics defect.

**The sizes** are nested prefixes of the tokenized corpus: `quick` scores 512 tokens,
`standard` 2048, `deep` 8192 (`Size::tokens`), so a deep run reproduces a quick run's
first scored tokens digit for digit, and the table prints that claim. The artifact
records `corpus_sha256`; the corpus is committed and the artifact is not, so the pair
can only drift one way, an artifact outliving an edit to the text it scored, and the
loader refuses that pairing instead of comparing against a different document with no
sign of it in the output.

**The verdict** is two measures: the relative delta of the two perplexities, which is
scale-free where an absolute delta is nothing at perplexity 200 and enormous at 1.2,
against `PPL_DELTA`; and the worst single-token NLL delta against
`PPL_MAX_TOKEN_DELTA`, kept separate because perplexity is an average and averages hide
one badly wrong token in a thousand ordinary ones, which is exactly what a
sliding-window boundary or shared-KV defect looks like. The note names the corpus
index of the worst token.

## The reference artifacts

### Why transformers and not llama.cpp

The reference artifacts replaced two `#[ignore]`d acceptance stubs that were never implemented, and it
does not port their fixture designs. Both compared the harness against the wrong thing:

- **Perplexity against `llama-perplexity` on a quantized GGUF.** A quantized GGUF is a
  different model. Agreeing with it to within 1% would be a coincidence, and disagreeing with
  it would say nothing about whether either one is right.
- **Greedy goldens generated by our own first verified run.** A golden minted from the
  implementation it guards pins the behavior that existed on the day it was minted, correct or
  not. Nine work packages of a green gate over a forward pass producing noise is what that
  costs.

`transformers` is the reference implementation of Gemma 4. It is the only oracle in reach that
cannot have inherited a misreading of the architecture documents from us.

### The two tiers on disk

| Path | Tier | Committed |
|------|------|-----------|
| `fixtures/tiny-hf-gemma4/` | T | yes, weights included |
| `e4b/<size>/` | E | no, generated on the workstation |

Tier T is a tiny Gemma 4 that `transformers` itself constructs and saves, small enough to
commit whole and to run in the sandbox in milliseconds. Tier E is the real checkpoint: its
captures run to tens of megabytes at a 262144-wide vocabulary, so they are generated per run
and gitignored.

The two are not substitutes. Tier T catches structure — a transposed weight, a missing scale,
a norm convention — on an oracle nobody here transcribed. Tier E catches what a handful of
dimensions cannot distinguish.

### Generating the Tier E artifacts

Workstation only: every generator needs the real checkpoint and a torch environment,
and `LADDER_PY` is `uv run --no-sync python` so the workstation venv is used rather than
whichever python is on `PATH` (the first Tier E attempt ran a bare python without
`transformers` and wrote nothing) and is used as it stands: a bare `uv run` resyncs it to
the lock first, and a resync that rebuilds `flash-attn` from source fans nvcc out over
every core, which is what OOM-killed the full gate on 2026-09-20. The six artifacts are file prerequisites of `make ladder`, so the
ordinary way to generate them is to climb: a missing one is written first, a present one
is left alone. To rewrite all six after a checkpoint or `transformers` change that make
cannot see under an existing directory:

```bash
cd rust-harness
make ladder-reference SIZE=quick MODEL=<checkpoint-dir>
```

The rules in `rust-harness/Makefile` are the source of truth for each generator's
arguments; the table under [The rungs](#the-rungs) says which writes which. Every
artifact the ladder reads lives directly under the `--reference` directory,
`fixtures/reference/e4b/<size>/` by default. L0's does not vary with `SIZE` and is
regenerated into each size directory rather than resolved from a parent: the
duplication costs milliseconds, and one placement rule is worth more than the special
case it would take to avoid it.

### The greedy prompts

`prompts.txt` is committed, and its order is load-bearing: a size takes a
prefix, so a `quick` run reproduces the first rows of a `deep` run exactly.
Append to the end; never insert.

The prompts are written as continuations rather than as questions. E4B at the
pinned revision is a base model — a measured 16-token probe on "The capital of
France is" returned `" a city that is full of history and culture"`, which is
a healthy base model and a weak prompt at once. A question in this file
produces a fluent non-answer that reads like a defect and is not.

`reference_greedy.py`'s `PROMPTS` and `TOKENS` restate `Size::prompts()` and
`ladder::GREEDY_TOKENS` on the Python side. `runtime/tests/ladder_reference_contract.rs`
pins them together: a generator writing fewer prompts than a size reads is
caught by the loader, but one writing more would produce an artifact that
loads, passes, and measures less than the header claims.

### What L0 can and cannot ask

`Tokenizer::encode_prompt` takes a per-call allowlist of special literals; the
reference has one global `split_special_tokens` switch. Only the two extremes
are therefore expressible on both sides — an empty allowlist, where nothing in
the text resolves, and an allowlist covering every special the text contains,
where all of them do. `reference_tokenizer.py` refuses to write a case whose
text carries a special outside its own allowlist, rather than emitting one
whose two sides answer different questions.

A leading `<bos>` is prepended by default. This was measured, not assumed: on E4B the same
prompt captured without one put its highest probability on repeating its own last token, which
is what a model does with a sequence it does not believe has begun. `--no-bos` reproduces that
if there is ever a reason to.


### What an artifact must record

An artifact that does not say what produced it cannot be reproduced, and a number nobody can
reproduce is not evidence. Every one carries:

- the checkpoint path, and the sha256 of its `config.json` as its revision — the weights run
  to tens of gigabytes, and it is a config change that silently alters the architecture under
  a stored capture,
- the `transformers` and `torch` versions, in the field names
  `fixtures/build_tiny_reference.py`'s manifest uses, so both tiers spell them the same way,
- whether a leading `<bos>` was prepended: the two conventions produce different distributions
  from one prompt, and Gemma is trained with it,
- for L0, the chat template revision, so an artifact rendered at one revision cannot be
  compared against a harness that renders another — the loader errors rather than reporting a
  tokenizer defect for what is a version skew,
- the token IDs, not only the prompt text — the tokenizer is itself under test at L0,
- for L2, the explicit list of positions scored, since the set is capped and
  `ladder::rung_logits` asserts it scored exactly those,
- for L3, the chunk size, because perplexity computed in different windows is a different
  quantity,
- for L5 and L6, the `sliding_window` the reference attended under, since the run compares it
  with the window its own loader read, and for L6 the layers chosen and their declared kinds,
  since the rung recomputes the choice from its own config read and refuses a disagreement.

`reference_common.py` writes the shared part of that provenance once, so no second
generator can spell `transformers_version` a different way and produce an artifact no
reader can compare.

## What the loader refuses

`runtime/src/reference.rs` reads each artifact and errors, rather than reporting a red
rung, on the mismatches that would make the comparison meaningless: an L0 artifact
rendered at a `chat_template_version` other than the harness's own constant, a
perplexity artifact whose `corpus_sha256` is not the committed corpus's, a long-context
or attention artifact carrying no `sliding_window` or no `positions`, and an artifact
naming more positions than it carries rows. `runtime/tests/ladder_reference_contract.rs`
pins the constants the generators restate on the Python side (`PROMPTS`, `TOKENS`
against `Size::prompts()` and `ladder::GREEDY_TOKENS`), because a generator writing
fewer prompts than a size reads is caught by the loader but one writing more would
produce an artifact that loads, passes and measures less than its header claims.

## Sizes, flags and records

`Size` selects how much work the corpus-scaled rungs do: 512, 2048 or 8192 corpus tokens
for L3 and 2, 8 or 32 prompts of `GREEDY_TOKENS` = 64 tokens for L4. The binary takes
`--model` and `--reference` (required), `--size`, `--rung` to stop at one, and four
tuning flags that may not change the table: `--dot`, `--batch`, `--threads` and
`--single-thread` (`runtime/tests/ladder_flags.rs` holds the table invariant across
them, entry RH.24). `make ladder` passes `--text-only` by default because the released
E4B carries a vision and an audio tower the CPU backend refuses to load and no rung
could measure; `TEXT_ONLY=0` restores the refusal for a checkpoint that is text-only
already.

Every climb appends a line to the shared observability catalog under the command name
`ladder`, with a `ladder.header` event recording the size, the fold, the width and the
thread count, one phase per rung, and a run directory holding `metrics.jsonl`;
`runtime/tests/ladder_obs.rs` measures that against a real invocation and
`tests/hardware/test_ladder.py` reads it back off the catalog row on the workstation.

## Status

Current status: **all seven rungs ran against real E4B on 2026-09-11 and
every one passed, with nothing skipped** — L0 token ids, L1 residual stream
(worst cosine 1.000000 over 42 layers), L2 final distribution, L3 perplexity
(delta 0.000032 over 254 tokens), L4 greedy continuation, L5 logits past the
sliding window (six anchors over 2052 tokens), L6 attention rows at the same
anchors (worst single-key difference 2.3e-5 over 192 rows). The first six
had passed identically on 2026-09-04, before L6 existed, and the first five
on 2026-08-14, before L5. L6's Tier E bounds remain marked provisional on
one checkpoint's evidence (entry RH.18). The rung bodies remain
unit-tested against a scripted `MockModel` as well. The transcript is entry
RH.1b of [the workstation runbook](../verification/rust-harness/rust-harness.md),
and L6's is entry RH.18.

That is a numerics claim, not an output-quality claim: the ladder measures
agreement with `transformers`, and says nothing about whether the model's
answers are good.

## Related documentation

- [testing.md](testing.md) — where the ladder sits among the tiers
- [fixtures.md](fixtures.md) — the Tier T fixture and the goldens the ladder replaced as the correctness claim
- [../testing/hardware-tiers.md](../testing/hardware-tiers.md) — how `make quality-full HW=full` climbs it
- [../verification/rust-harness/rust-harness.md](../verification/rust-harness/rust-harness.md) — RH.1b and RH.18, the recorded climbs
- [../../rust-harness/fixtures/reference/README.md](../../rust-harness/fixtures/reference/README.md) — the directory the artifacts land in
