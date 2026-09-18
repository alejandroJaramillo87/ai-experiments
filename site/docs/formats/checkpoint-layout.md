---
title: Checkpoint layout
---

# Checkpoint layout

The harness reads a Hugging Face checkpoint: the directory a Gemma 4 model is published as. For
a model format the reference implementation and the specification are the same artifact, so
the harness reads what `transformers` writes and nothing derived from it. It does not read
GGUF, and the training tier's export writes exactly this layout.

## Files

| File | Required | Read for |
|------|----------|----------|
| `config.json` | yes | The architecture: layer count, hidden size, head dimensions, the attention schedule as the `layer_types` array, the sliding window, RoPE parameters, and the context length `max_position_embeddings` |
| `model.safetensors` | yes | The weights, one tensor per name; the file's sha256 is the model identity every artifact records |
| `tokenizer.json` | yes, unless `--tokenizer` is given | The tokenizer, including its pre-tokenization pipeline and special tokens |
| `generation_config.json` and the rest | no | Not read |

`--model` takes the directory. A path to `model.safetensors` itself also resolves, to its
directory. A checkpoint split across several safetensors shards is not read; the released E4B
is one file.

## Multimodal checkpoints

The released Gemma 4 E4B is `Gemma4ForConditionalGeneration`: a text stack beside an audio and
a vision tower, with the text architecture under a `text_config` key of `config.json`. The
backend serves the text stack and refuses the checkpoint unless `--text-only` is given, so that
nobody runs a model text-blind without having said so. `checkpoint-info` reports the towers it
found and, with `--text-only`, the verdict the harness would give with the flag.

## Tokenizer resolution

In order:

1. The `--tokenizer <path.json>` flag, if given. The file must exist.
2. `<checkpoint>.tokenizer.json`, beside the checkpoint directory: for a directory named
   `tiny-gemma4`, the file `tiny-gemma4.tokenizer.json` in its parent. This is the layout
   `tiny-model` writes.
3. `tokenizer.json` inside the checkpoint directory, which is how a published checkpoint is laid
   out.

## Synthetic checkpoints

`tiny-model` writes a checkpoint of the same architecture at toy dimensions, with seeded random
weights, for exercising the machinery without a download:

```bash
"$BIN/tiny-model" /tmp/harness-demo/mytiny
"$BIN/tiny-model" /tmp/harness-demo/mytiny-e4b e4b
```

The first form writes a two-layer model; `e4b` writes a four-layer one shaped like the released
E4B, with per-layer embeddings and the interleaved sliding and global schedule. Each writes
`<stem>/config.json`, `<stem>/model.safetensors`, and `<stem>.tokenizer.json`. The form
`tiny-model <output.json> bpe` writes only a tokenizer with a merge list, which no model fixture
carries. The repository commits the outputs as `rust-harness/fixtures/tiny-gemma4/` and
`rust-harness/fixtures/tiny-gemma4-e4b/`, and a third checkpoint,
`rust-harness/fixtures/tiny-hf-gemma4/`, that `transformers` itself constructed and saved, with
the reference capture beside it that [Comparing per-layer divergence](../rust-harness/guides/layer-diff.md)
uses.

## Inspecting a checkpoint

```bash
"$BIN/checkpoint-info" "$MODEL"
```

prints the config, the tensor table with each tensor's shape and dtype, the attention
schedule as a string of `s` and `G` per layer, the per-layer geometry, and the backend's
verdict. `--check` turns an unsupported verdict into exit 1.
