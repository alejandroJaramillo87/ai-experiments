# Model references

Architecture references for the model families this workstation serves, and the raw header
transcripts that verify them.

## Documents

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [gemma-4.md](gemma-4.md) | Architecture reference for the Gemma 4 dense models, written so that a session working on `rust-harness`, `training`, or `interp` does not have to infer model internals. |

<!-- inventory:pages:end -->

`gemma-4.md` is the citation target of
[../verification/archive/gemma-4-conformance_investigation.md](../verification/archive/gemma-4-conformance_investigation.md).
Each tier keeps its own scoped Gemma 4 document, covering what that tier's code consumes:
[../training/base/gemma-4-training.md](../training/base/gemma-4-training.md) for training and
[../interp/gemma-4-interp.md](../interp/gemma-4-interp.md) for interp. The runtime has none:
it reads the published checkpoint directly, so what it requires is stated here and in
[../rust-harness/runtime-design.md](../rust-harness/runtime-design.md).

## Header transcripts

`gemma4-gguf.txt` and `gemma4-safetensors.txt` are captured output, not prose: the tensor
inventories, dtypes, and metadata keys read directly off the checkpoints. They are the
evidence behind the architecture claims in `gemma-4.md`, which is why they are committed
rather than regenerated on demand — the checkpoints live on the workstation under
`/mnt/ai-data`, unreachable from a contained agent sandbox.

| Transcript | Source |
|------------|--------|
| `gemma4-gguf.txt` | `/mnt/ai-data/models/gguf/gemma-4-E2B-it/` and `gemma-4-E4B-it/` (base, mmproj, mtp) |
| `gemma4-safetensors.txt` | `/mnt/ai-data/models/hf/gemma-4-E2B` and `gemma-4-E4B` |
| `gemma4-upstream.txt` | the upstream `transformers` and `llama.cpp` sources, at the commit SHAs pinned in `gemma-4.md` |

Two derived artifacts sit beside them:

| File | Contents |
|------|----------|
| `gemma4-chat-template.jinja` | Google's canonical chat template, extracted verbatim from `tokenizer.chat_template` in the GGUF metadata, byte-identical across E2B and E4B |
| [../../schemas/gemma4_facts.json](../../schemas/gemma4_facts.json) | The architecture values in machine-readable form, authored from the pinned `config.json` bytes rather than from these transcripts |

### Regenerating

Both commands run on the workstation only; each reads `/mnt/ai-data`.

```bash
./scripts/models/gguf_header_transcript.sh --out docs/models/gemma4-gguf.txt \
      /mnt/ai-data/models/gguf/gemma-4-E2B-it/*.gguf \
      /mnt/ai-data/models/gguf/gemma-4-E4B-it/*.gguf

./scripts/models/gemma4_header_transcript.sh --out docs/models/gemma4-safetensors.txt \
      /mnt/ai-data/models/hf/gemma-4-E2B /mnt/ai-data/models/hf/gemma-4-E4B
```

Both scripts default to `/tmp`, so `--out` is required to land the file here.

`gemma4-upstream.txt` regenerates from anywhere, including a contained agent sandbox, because
`raw.githubusercontent.com` is on the proxy allowlist:

```bash
./scripts/models/upstream_source_transcript.sh --out docs/models/gemma4-upstream.txt
```

**`gemma-4.md` cites these transcripts by line
number, and regenerating shifts those numbers.** That used to be a warning asking a human to
re-check six citations by hand. `../../tests/models/test_gemma4_facts_contract.py` now asserts
each one, so a shifted citation fails a test instead. The documentation reference gate still
checks only that a path exists, which is why that module carries its own citation table.

The same module asserts both transcripts and `gemma-4.md` against
[../../schemas/gemma4_facts.json](../../schemas/gemma4_facts.json). After regenerating, run
`pytest tests/models` and resolve whatever disagrees before committing: a disagreement means
either the transcript is stale or the contract is wrong, and both are worth knowing.
