---
title: Correctness ladder
---

# Correctness ladder

`ladder` is the seven-rung correctness gate: it compares the CPU backend against artifacts
generated from `transformers` on the same checkpoint, climbs the rungs in order, prints one
table, and exits 1 on the first failure or when nothing was measured. It is how a change to the
forward pass is shown not to have changed the model, and the only tool on this site that can
call the backend correct rather than merely running.

The reference artifacts for the real checkpoint are not committed, because they run to tens of
megabytes at a 262144-token vocabulary. Generating them needs the checkpoint on disk and a
Python environment with torch and `transformers`; the climb itself needs only the checkpoint.
This page assumes `BIN` from [Build](../getting-started/build.md) and uses `gemma-4-E4B`, the
checkpoint the outputs below were recorded on; a base checkpoint is the right subject, since
nothing here reads a reply.

## The rungs

| Rung | Measures | Reference artifact | Generator |
|------|----------|--------------------|-----------|
| L0 | Tokenizer parity: token ids for plain text, for text carrying specials that must stay bytes, and for a rendered chat turn | `tokenizer_parity.json` | `reference_tokenizer.py` |
| L1 | The per-layer residual stream and the pre-stack stages over one prompt | `layers.json` | `reference_capture.py` |
| L2 | The final-position output distribution over the same prompt | `layers.json` | `reference_capture.py` |
| L3 | Per-token negative log-likelihood over a prefix of a committed corpus, in 256-token windows | `ppl.json` | `reference_perplexity.py` |
| L4 | A 64-token greedy continuation per prompt from a committed prompt list | `greedy.json` | `reference_greedy.py` |
| L5 | Logits at anchors past the sliding window, over a prompt four windows long | `long_context.json` | `reference_long_context.py` |
| L6 | Post-softmax attention rows at the L5 anchors, over the same prompt | `attention.json` | `reference_attention.py` |

The generators live under `rust-harness/fixtures/` and their arguments are recorded as rules in
`rust-harness/Makefile`, which is the source of truth for them. L5 and L6 are the two rungs
that engage what is architecturally particular about Gemma 4: the interleaved sliding and
global layers, and the shared-KV tail reading a donor layer's older rows. L0 through L4 never
cross a window.

## Generate the references and climb

From `rust-harness`, on the machine holding the checkpoint, one target does both: a missing
artifact is generated first and a present one is left alone.

```bash
cd rust-harness
make ladder SIZE=quick MODEL=/path/to/gemma-4-E4B
```

`SIZE` is `quick`, `standard`, or `deep` and decides how much corpus L3 scores and how many
prompts L4 decodes; the sizes are nested prefixes, so a deeper run reproduces a quicker run's
first rows exactly. The target passes `--text-only` by default, because the released
checkpoints are multimodal; `TEXT_ONLY=0` restores the refusal for a text-only checkpoint. The
generators run through `uv run python`, so the repository's environment must hold torch and
`transformers`; a bare `python` that lacks them produces `[ERROR] Import` lines and no
artifacts. To regenerate every artifact after the checkpoint or `transformers` changes, which
make cannot see under an existing directory:

```bash
make ladder-reference SIZE=quick MODEL=/path/to/gemma-4-E4B
```

## What a climb prints

The header states the size, the fold, the checkpoint, and the reference directory; then one
block per rung with its measures against their thresholds; then the timing. The quick climb on
E4B, with the middle rungs elided:

```text
size=quick tokens=512 prompts=2
prompt=6 tokens
threads=8
dot=widen4
batch=64
model=/mnt/ai-data/models/hf/gemma-4-E4B
reference=fixtures/reference/e4b/quick

rung  check                   value         threshold       verdict
L0    token ids                                             PASS
      case agreement          1.000000      >= 1.000000     ok
      5 cases, every id identical
L1    residual stream                                       PASS
      worst cosine            1.000000      >= 0.999900     ok
      worst rel_l2            0.000005      <= 0.001000     ok
      42 layers within tolerance
L2    final distribution                                    PASS
      mean kl                 0.000000      <= 0.001000     ok
      worst kl                0.000000      <= 0.005000     ok
      worst top-50 overlap    1.000000      >= 0.980000     ok
      argmax agreement        1.000000      >= 0.980000     ok
      1 positions scored; worst at prompt position 5
L3    perplexity                                            PASS
      ppl delta               0.000032      <= 0.010000     ok
      worst token nll         0.001248      <= 0.050000     ok
      ppl 55.5420 against 55.5438 over 254 tokens; worst token at corpus position 414
L4    greedy continuation                                   PASS
      token agreement         1.000000      >= 0.980000     ok
      mean common prefix      1.000000      >= 0.750000     ok
      2 prompts, 0 ties
L5    logits past the window                                PASS
      scheduled positions agree  1.000000      >= 1.000000     ok
      worst kl inside window  0.000000      <= 0.005000     ok
      worst kl past window    0.000000      <= 0.005000     ok
      worst top-50 overlap    1.000000      >= 0.980000     ok
      argmax agreement        1.000000      >= 0.980000     ok
      6 positions across a 512-token window over 2052 tokens; worst at position 1025
L6    attention past window                                 PASS
      captured layers agree   1.000000      >= 1.000000     ok
      weight outside window   0.000000      <= 0.000000     ok
      worst row max abs diff  0.000023      <= 0.001000     ok
      worst row mass          0.000001      <= 0.000010     ok
      worst top-10 overlap    1.000000      >= 0.900000     ok
      192 rows across 4 layers x 8 heads at 6 positions; window 512; worst at layer 21 head 6 position 1025

timing (not compared; expected to differ between runs)
  L0        0.48s
  L1        0.52s
  L2        0.66s
  L3       52.16s
  L4       22.72s
  L5      267.44s
  L6      226.79s
  sum     570.77s  (rung bodies only, excludes model load)
[OK] Ladder: every rung within tolerance (0 skipped)
[INFO] Peak RSS at exit: 16.13 GiB
```

**Read the `0 skipped` in the closing line, not the exit code.** A rung with no artifact is
skipped, and a skipped rung is missing evidence rather than a pass; a skip after the
generators ran means one wrote somewhere other than where the climb read. The `ppl 55.5420`
figure is the model's perplexity on the quick corpus and not a quality claim; the comparison
against `55.5438` is the result. The timing block is informational, and the climb costs about
ten minutes on E4B, most of it in the two 2052-token rungs.

## Run the binary directly

With the artifacts in place, the binary takes the checkpoint and the directory holding them,
and `--rung` climbs one rung without paying for the others:

```bash
"$BIN/ladder" --text-only --model /path/to/gemma-4-E4B --reference fixtures/reference/e4b/quick
"$BIN/ladder" --text-only --model /path/to/gemma-4-E4B --reference fixtures/reference/e4b/quick --rung L6
```

`--size` must match the size the artifacts were generated at, because the perplexity artifact
records how many tokens it scored and the loader refuses a mismatch. `--threads <n>` sets the
row-split width and `--batch` how many positions one forward pass covers; every table is
bit-identical at every setting of both, and the flags exist to make the run faster or, under
rr, serial. `--single-thread` is shorthand for `--threads 1`, and giving both is a usage error
naming both flags.

`--dot <kind>` selects the dot-product kernel from `sequential`, `rows4`, `rows8`, `rows16`,
`widen4`, and `lanes16`. The first five produce identical logits bits and the ladder's
invariance tests are asserted over exactly that set; `lanes16` reorders the accumulation and
differs in the last bits, measured on E4B as `1062660370` against `1062660362` at logit 0. A
run that picks `lanes16` and then compares logits bits against a sequential baseline reports a
divergence that is not a defect. The default is `widen4`, as the header's `dot=` line states.

The ladder records a run with a `ladder.header` event, one `rung.L<n>` phase per rung, and one
metrics record per rung; see [Run records](../observability/run-records.md).

## Without the references

Pointing `--reference` at a directory with none of the seven artifacts measures nothing and
exits 1 by design:

```text
[WARN] Ladder: no rung had a reference; nothing was measured
[ERROR] Ladder: no reference artifacts under <dir> -- nothing was measured
```

The one structural rung that runs without a workstation is the residual-stream comparison, as
the standalone `layer-diff` binary over the committed `transformers`-authored tiny checkpoint;
see [Comparing per-layer divergence](layer-diff.md).
