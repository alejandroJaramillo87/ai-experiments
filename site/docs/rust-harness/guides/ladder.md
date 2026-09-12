---
title: Correctness ladder
---

# Correctness ladder

`ladder` is the seven-rung correctness gate: it compares the CPU backend against artifacts
generated from `transformers` on the same checkpoint, climbs the rungs in order, prints one
table, and exits 1 on the first failure or when nothing was measured. It is how a change to the
forward pass is shown not to have changed the model.

The reference artifacts for the real checkpoint are not committed, because they run to tens of
megabytes at a 262144-token vocabulary. Generating them needs the checkpoint on disk and a
Python environment with torch and `transformers`; the climb itself needs only the checkpoint.
This page assumes `BIN` from [Build](../getting-started/build.md).

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
`rust-harness/Makefile`, which is the source of truth for them.

## Generate the references and climb

From `rust-harness`, on the machine holding the checkpoint, one target does both: a missing
artifact is generated first and a present one is left alone.

```bash
cd rust-harness
make ladder SIZE=quick MODEL=/path/to/gemma-4-E4B
```

`SIZE` is `quick`, `standard`, or `deep` and decides how much corpus L3 scores and how many
prompts L4 decodes; the sizes are nested prefixes, so a deeper run reproduces a quicker run's
first rows exactly. The target passes `--text-only` by default, because the released E4B is
multimodal; `TEXT_ONLY=0` restores the refusal for a text-only checkpoint. To regenerate every
artifact after the checkpoint or `transformers` changes, which make cannot see under an
existing directory:

```bash
make ladder-reference SIZE=quick MODEL=/path/to/gemma-4-E4B
```

## Run the binary directly

With the artifacts in place, the binary takes the checkpoint and the directory holding them:

```bash
"$BIN/ladder" --text-only --model /path/to/gemma-4-E4B --reference fixtures/reference/e4b/quick
```

`--rung L3` climbs one rung. `--size` must match the size the artifacts were generated at,
because the perplexity artifact records how many tokens it scored and the loader refuses a
mismatch. `--threads <n>` sets the row-split width and `--batch` how many positions one
forward pass covers; every table is bit-identical at every setting of both, and the flags exist
to make the run faster or, under rr, serial. `--dot <kind>` selects the dot-product kernel,
also bit-identical across kinds.

The ladder prints one row per rung with its status and wall clock, then a sum of the rung
bodies. It records a run with a `ladder.header` event and one metrics record per rung; see
[Run records](../observability/run-records.md).

## Without the references

Pointing `--reference` at a directory with none of the six artifacts measures nothing and exits
1 by design:

```text
[WARN] Ladder: no rung had a reference; nothing was measured
[ERROR] Ladder: no reference artifacts under <dir> -- nothing was measured
```

The one structural rung that runs without a workstation is the residual-stream comparison, as
the standalone `layer-diff` binary over the committed `transformers`-authored tiny checkpoint;
see [Comparing per-layer divergence](layer-diff.md).
