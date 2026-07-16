# Training phase 2 verification

P2-D1 through P2-D5 as implemented: the full-scale data pipeline with
decontamination, data mixing, WSD schedules, the replay buffer, and
ZeRO-Offload optimizer offload.

## P2.1 Contamination index (P2-D1)

Command:

```bash
$ make build-contamination ARGS="--out /mnt/ai-data/training/datasets/_contamination/v1"
```

Expected signal: exit 0; the index directory contains the benchmark n-gram
index plus a manifest whose hash the pipeline records as
`decontaminated_against`.

Observed:

```
```

Status: PENDING

## P2.2 Full pipeline run (P2-D1)

Command:

```bash
$ make prepare-corpus ARGS="--source HuggingFaceFW/fineweb-edu --split train \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --contamination /mnt/ai-data/training/datasets/_contamination/v1 \
    --work-dir /mnt/ai-data/training/work/verify-corpus \
    --out /mnt/ai-data/training/shards/verify-corpus --max-tokens 50_000_000"
```

Expected signal: exit 0; the manifest records the stage funnel (documents
in/out per filter, dedup, decontamination), the datatrove config blocks,
and `decontaminated_against` with the index hash; shards verify against
their recorded hashes.

Observed:

```
```

Status: PENDING

## P2.3 Mixture training (P2-D2)

Command (two prepared shard directories with one tokenizer hash):

```bash
$ make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="mixture.enabled=true \
          'mixture.sources=[{name: a, path: /mnt/ai-data/training/shards/verify-small, weight: 0.7}, {name: b, path: /mnt/ai-data/training/shards/verify-corpus, weight: 0.3}]' \
          schedule.total_steps=50"
```

Expected signal: exit 0; per-source token counters appear in the metrics
(`tokens_source_a`, `tokens_source_b`) with the draw ratio near 70/30;
checkpoint meta carries `tokens_per_source`, and a resumed run keeps the
counters cumulative. Mismatched tokenizer hashes between sources refuse
loudly.

Observed:

```
```

Status: PENDING

## P2.4 WSD schedule (P2-D3)

Command:

```bash
$ make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="data.source=/mnt/ai-data/training/shards/verify-small \
          schedule.type=wsd schedule.warmup_steps=5 schedule.total_steps=50 \
          'schedule.decay_start_frac=0.8' 'schedule.final_lr_frac=0.1'"
```

Expected signal: exit 0; the recorded LR curve (TrainResult/tracking)
shows warmup to step 5, a stable plateau, and the 1-sqrt decay starting at
step 40 down to 10 percent of peak.

Observed:

```
```

Status: PENDING

## P2.5 Replay buffer (P2-D4)

Command:

```bash
$ docker compose -f docker-compose.training.yaml run --rm training \
    python -m training make-replay-synthetic --out /mnt/ai-data/training/shards/replay-verify --gb 1
$ make train CONFIG=training/recipes/reference/continual-gemma4-e2b.yaml \
    ARGS="'replay.sources=[{name: replay, path: /mnt/ai-data/training/shards/replay-verify, weight: 1.0}]' \
          replay.max_gb=2.0 schedule.total_steps=30"
$ uv run pytest -m hw_full tests/hardware/test_ram.py -k replay -v
```

Expected signal: the synthetic shard set generates; the run draws the
configured replay fraction from the RAM-resident buffer (per-source
counters show it); the hw_full residency test passes.

Observed:

```
```

Status: PENDING

## P2.6 Optimizer offload (P2-D5)

Command:

```bash
$ make train CONFIG=training/recipes/reference/continual-gemma4-e2b.yaml \
    ARGS="schedule.total_steps=30"
```

Expected signal: exit 0; DeepSpeed initializes ZeRO stage 2 with CPU
optimizer offload; frozen parameters (embeddings, LM head, per-layer
tables) never enter the optimizer groups; metrics include
`optimizer_step_s`, which must not dominate step time at the recipe's
grad_accum (the raise-grad_accum diagnosis otherwise); checkpoint meta
records `checkpoint_format: deepspeed` and `optimizer: cpu_adam`; measured
host RAM stays within the calculator's prediction plus headroom
(`uv run pytest -m hw_full tests/hardware/test_ram.py -k headroom`).

Observed:

```
```

Status: PENDING

## P2.7 Offload resume (P2-D5)

Command:

```bash
$ make train CONFIG=training/recipes/reference/continual-gemma4-e2b.yaml \
    ARGS="schedule.total_steps=60 checkpoint.resume=latest"
```

Expected signal: the engine restores from the deepspeed checkpoint
subdirectory; the loss curve continues within the stated GPU tolerance (1
percent relative per step) and a torch-format checkpoint is refused under
offload with the checkpoint_format guard message.

Observed:

```
```

Status: PENDING
