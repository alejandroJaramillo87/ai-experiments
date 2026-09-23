# scripts/models

Transcribers that turn a model's own files into the committed reference transcripts under
`docs/models/`. They read headers and pinned upstream sources, never load a model, and
run wherever the weights or the network are: the safetensors and GGUF transcribers need
the files on disk, the upstream transcriber needs egress.

The transcripts are cited by line number, so regenerating one shifts every citation and
`tests/models/test_gemma4_facts_contract.py` fails on each that moved. Regenerate on
purpose, after a checkpoint revision or an upstream pin changes, and fix the citations in
the same commit.

## Scripts

<!-- inventory:scripts:start -->

| Script | Purpose |
|--------|---------|
| `gemma4_header_transcript.sh` | Transcribe Gemma 4 architecture facts from locally downloaded safetensors weights into a single reviewable file. |
| `gguf_header_transcript.sh` | Transcribe GGUF metadata and tensor inventory into a reviewable file, without loading the model. |
| `upstream_source_transcript.sh` | Fetch the pinned upstream sources the Gemma 4 architecture claims are settled against, and transcribe them with line numbers. |

<!-- inventory:scripts:end -->

## Where the real docs live

- [../../docs/models/README.md](../../docs/models/README.md) — the transcripts, and how they are read
- [../../schemas/README.md](../../schemas/README.md) — `gemma4_facts.json`, which names these transcribers as writers
