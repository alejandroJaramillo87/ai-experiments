# training/data

Everything between a corpus and a batch. In order of use: tokenizer training, the tokenize-and-shard
writer and its memory-mapped reader, the full-scale corpus pipeline with deduplication and
benchmark decontamination, weighted mixing across sources, RAM-disk staging and the
pinned-memory prefetch loader, the chat-and-instruction preparation that renders and masks
supervised fine-tuning examples, and the teacher harness that writes a corpus through the
inference services.

The shard format in `shards.py` is the contract every rung reads: `prepare-data` and
`prepare-corpus` write it, the loop's loader reads it, and the manifest beside it carries
the tokenizer hash so a run cannot train on shards cut with another tokenizer. `sft.py`
is the correctness surface a fine-tune depends on most: every label is decided as the
token is produced, from the family's template in `training/models/families.py`, never by matching
rendered text. `configs/default.yaml` holds the pipeline defaults `prepare-corpus` starts
from.

Consumers outside the package: `interp/organisms/generate.py` calls `teacher.py` to hand
the organism prompts to the serving stack, and `rust-harness/fixtures/reference_tokenizer.py`
renders through `sft.py`'s `render_segments` for the chat-template contract.

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `contamination.py` | Benchmark decontamination (P2-D1). |
| `loader.py` | Pinned-memory prefetch dataloader (Phase 0 D9). |
| `mixture.py` | Weighted multi-source token mixing (Phase 2 P2-D2). |
| `pipeline.py` | Full-scale corpus pipeline (P2-D1). |
| `prepare.py` | Tokenize-and-pack pipeline (Phase 0 D2). |
| `sft.py` | Chat and instruction data preparation (Phase 1 F1). |
| `shards.py` | Shard format read/write (Phase 0 D2). |
| `staging.py` | RAM-disk shard staging (Phase 0 D9). |
| `teacher.py` | Teacher and synthetic data harness core (Phase 1 F4). |
| `tokenizer.py` | Byte-level BPE tokenizer training and evaluation (Phase 0 D3). |

<!-- inventory:modules:end -->

Living pages: [data_pipeline.md](../../docs/training/base/data_pipeline.md),
[sft_data.md](../../docs/training/fine-tuning/sft_data.md),
[teacher_synthetic_data.md](../../docs/training/fine-tuning/teacher_synthetic_data.md) and
[tokenizer_training.md](../../docs/training/from-scratch/tokenizer_training.md). Tests:
`tests/training/test_data*.py`, `test_sft.py`, `test_teacher.py`, `test_tokenizer.py`,
`test_pipeline.py`, `test_contamination.py`, `test_mixture.py`, `test_staging.py` and
`test_chat_template_contract.py`.
