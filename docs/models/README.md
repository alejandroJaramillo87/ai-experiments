# Model references

Architecture references for the model families this workstation serves, and the raw header
transcripts that verify them.

## Documents

| Document | Contents |
|----------|----------|
| [gemma-4.md](gemma-4.md) | Gemma 4 dense family (E2B, E4B, 12B, 31B): general architecture reference, and the citation target of [../verification/gemma-4-conformance.md](../verification/gemma-4-conformance.md) |

Each tier keeps its own scoped Gemma 4 document, covering what that tier's code consumes:
[../rust-harness/gemma-4-ingestion.md](../rust-harness/gemma-4-ingestion.md) for the runtime,
[../training/gemma-4-training.md](../training/gemma-4-training.md) for training, and
[../interpretability/gemma-4-interp.md](../interpretability/gemma-4-interp.md) for interp.

## Header transcripts

`gemma4-gguf.txt` and `gemma4-safetensors.txt` are captured output, not prose: the tensor
inventories, dtypes, and metadata keys read directly off the checkpoints. They are the
evidence behind the architecture claims in `gemma-4.md`, which is why they are committed
rather than regenerated on demand — the checkpoints live on the workstation under
`/mnt/ai-data`, unreachable from a contained agent sandbox.

| Transcript | Source checkpoints |
|------------|--------------------|
| `gemma4-gguf.txt` | `/mnt/ai-data/models/gguf/gemma-4-E2B-it/` and `gemma-4-E4B-it/` (base, mmproj, mtp) |
| `gemma4-safetensors.txt` | `/mnt/ai-data/models/hf/gemma-4-E2B` and `gemma-4-E4B` |

### Regenerating

Both commands run on the workstation only; each reads `/mnt/ai-data`.

```bash
$ ./scripts/models/gguf_header_transcript.sh --out docs/models/gemma4-gguf.txt \
      /mnt/ai-data/models/gguf/gemma-4-E2B-it/*.gguf \
      /mnt/ai-data/models/gguf/gemma-4-E4B-it/*.gguf

$ ./scripts/models/gemma4_header_transcript.sh --out docs/models/gemma4-safetensors.txt \
      /mnt/ai-data/models/hf/gemma-4-E2B /mnt/ai-data/models/hf/gemma-4-E4B
```

Both scripts default to `/tmp`, so `--out` is required to land the file here.

**Warning!** `gemma-4.md` and `../rust-harness/gemma-4-ingestion.md` cite these transcripts
**by line number**. Regenerating shifts those numbers whenever the tensor inventory changes,
and nothing verifies them: the documentation reference gate checks that a path exists, not
that a line still says what a document claims. After regenerating, re-check the six citations
in those two documents before committing.
