# From-scratch loop verification

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 0 verified, 0 blocked, 9 pending, of 9.**

The from-scratch loop is the third rung of the training tier: the D6 reference
loop (`make train`) driving a scratch model over prepared shards, the rung that
will run the `pretrain-150m` and `pretrain-1b` recipes. Its first entries are
not those runs. They are the Phase 0 acceptance checks that the 2026-09-18
reorganisation left pending in the frozen
[archive/training-phase0.md](../archive/training-phase0.md), because every one
is defined on this loop rather than on the fine-tuning loop: P0.6 (the
reference loop end to end), P0.10 (staging and prefetch), and P0.11 (the
watchdog observed passively), plus the D9 and D10 acceptance checks the
archive never carried as entries (the staging refusal, the affinity masks, the
lock, and the abort path). The archived entries stay frozen and are cited by
id; the 2026-09-19 audit that found them uncovered is
[../../training/investigations/phase0_completeness_audit_investigation.md](../../training/investigations/phase0_completeness_audit_investigation.md).

Every entry runs the `tiny-e2e` recipe. Its committed `data.source` and
`tokenizer.path` are the fixtures the unit tests generate, not artifacts on
the workstation, so every command carries the same three overrides: the shard
set FS.0 prepares, the tokenizer it was prepared with, and a scratch
vocabulary large enough for that tokenizer's ids. Both inputs come from P0.2
(VERIFIED, 2026-08-01), which left the trained tokenizer at
`/mnt/ai-data/training/tokenizers/verify-49k` and its JSONL corpus at
`/mnt/ai-data/training/datasets/mix/fineweb-sample.jsonl`; a training
container cannot reach the hub, and this pairing needs nothing from it. That
is the only reason a byte-level BPE artifact appears here: a from-scratch run
meant to be served uses the Gemma 4 tokenizer, as
[../../training/training-scale-reference.md](../../training/training-scale-reference.md#phase-4-from-scratch-training)
records, and these entries prove the loop, not the tokenizer a real run keeps. The
overrides are held in one shell variable so each block states them once:

```bash
OVR="data.source=/mnt/ai-data/training/shards/verify-49k tokenizer.path=/mnt/ai-data/training/tokenizers/verify-49k model.scratch.vocab=49152"
```

`49152` is the vocabulary size P0.2 trained; a trained vocabulary that came out
smaller still fits under it, and nothing validates the pairing, so an
out-of-range embedding index at the first step is the symptom of getting it
wrong. Set the variable in the shell that runs the entries: an unset `OVR`
expands to nothing, the entry runs the committed recipe against the absent
fixture paths, and the skip-when-absent tokenizer check passes vacuously.
The `docker compose` line the target echoes shows whether the overrides
reached the container; the first FS.0 attempt ran without them for this
reason.

Every `make` target here follows the same dispatch chain: the `Makefile` runs
`docker compose -f docker-compose.training.yaml run --rm training python -m
training <subcommand>`, which enters `training/__main__.py`, dispatches through
the argparse handlers in `training/cli.py`, and lands in the implementation
module. `make prepare-data` runs in the `prepare` service instead. The training
root `/mnt/ai-data/training` is mounted at the same path in both, so the
overrides above spell host paths. The hardware tests run on the host under
`uv run`, as every hardware entry does.

One GPU consumer at a time: no entry runs while another GPU job, benchmark, or
inference service the entry does not name is active. Run this runbook after
[fine-tuning-loop.md](fine-tuning-loop.md); FS.0 depends on FT.12.

## Table of Contents

- [FS.0 Preconditions: shards from the verified tokenizer](#fs0-preconditions-shards-from-the-verified-tokenizer)
- [FS.1 Reference loop end to end](#fs1-reference-loop-end-to-end)
- [FS.2 Resume from latest through the CLI](#fs2-resume-from-latest-through-the-cli)
- [FS.3 Staging and prefetch under the D9 bar](#fs3-staging-and-prefetch-under-the-d9-bar)
- [FS.4 Staging refusal over the budget](#fs4-staging-refusal-over-the-budget)
- [FS.5 Affinity masks disjoint and pinned](#fs5-affinity-masks-disjoint-and-pinned)
- [FS.6 The run lock refuses a second run](#fs6-the-run-lock-refuses-a-second-run)
- [FS.7 Watchdog passive on a healthy run](#fs7-watchdog-passive-on-a-healthy-run)
- [FS.8 Watchdog abort and the emergency checkpoint](#fs8-watchdog-abort-and-the-emergency-checkpoint)

## FS.0 Preconditions: shards from the verified tokenizer

Implementation: `training/commands/prepare_data.py`, `training/data/prepare.py`,
`training/data/shards.py` (`make prepare-data`),
`training/commands/validate_recipe.py`, `training/recipes/validate.py`
(`make train-validate`).

Builds on P0.2 (the `verify-49k` tokenizer and the corpus it sampled) and
P0.1 (the shard format), both VERIFIED in
[archive/training-phase0.md](../archive/training-phase0.md), and on FT.12 (the
image). Existing code; depends on FT.12.

A local JSONL source reads in chunks and never starts the streaming download
thread behind KB-52, so exit 0 is expected here where P0.1 had to accept a
teardown crash. Five million tokens is enough for every entry below at
sequence length 128 and keeps the copy FS.3 stages small.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
ls /mnt/ai-data/training/tokenizers/verify-49k/ /mnt/ai-data/training/datasets/mix/
make prepare-data ARGS="--source /mnt/ai-data/training/datasets/mix/fineweb-sample.jsonl --text-field text --tokenizer /mnt/ai-data/training/tokenizers/verify-49k --out /mnt/ai-data/training/shards/verify-49k --max-tokens 5_000_000"
echo "exit=$?"
python3 -c "import json;m=json.load(open('/mnt/ai-data/training/shards/verify-49k/manifest.json'));print(m['dtype'],m['tokenizer_hash'],m['total_tokens'])"
make train-validate CONFIG=training/recipes/reference/tiny-e2e.yaml ARGS="$OVR"
echo "exit=$?"
```

Expected signal: the `ls` shows `tokenizer.json` and `manifest.json` in the
tokenizer directory and `fineweb-sample.jsonl` in the corpus directory;
`prepare-data` prints `[RUN] Tokenizing and writing shards`, then `[OK]
Shards written:`, `[OK] Total:` with roughly 5,000,000 tokens, and `[OK]
Manifest:` naming the output directory, with exit 0; the manifest reads
`uint16`, the hash `8fada2fadd3e8c29` that P0.2 recorded for this tokenizer,
and the token total; `train-validate` prints `[OK] Validation: run tiny-e2e
passes all cross-field invariants` with exit 0. A validation failure naming
`tokenizer: hash ... does not match shard manifest hash` means the shards and
the tokenizer disagree, and the shard set is the one to regenerate.

Partial record, 2026-09-19: every command below passed as expected, on the
image FT.12 records the build of. The validate step is the rerun with `OVR`
set; the first pass had the variable empty, as the intro now warns. The entry
stays pending only because FT.12 is: its `uv lock --check` and `make
train-smoke` lines are still owed, and this entry depends on it.

Observed:

```
➜  ai-experiments git:(SFT-verification) ✗ set -o pipefail
cd ~/workspace/ai-experiments
ls /mnt/ai-data/training/tokenizers/verify-49k/ /mnt/ai-data/training/datasets/mix/
/mnt/ai-data/training/datasets/mix/:
fineweb-sample.jsonl

/mnt/ai-data/training/tokenizers/verify-49k/:
manifest.json  tokenizer.json
➜  ai-experiments git:(SFT-verification) ✗ make prepare-data ARGS="--source /mnt/ai-data/training/datasets/mix/fineweb-sample.jsonl --text-field text --tokenizer /mnt/ai-data/training/tokenizers/verify-49k --out /mnt/ai-data/training/shards/verify-49k --max-tokens 5_000_000"

docker compose -f docker-compose.training.yaml run --rm prepare python -m training prepare-data --source /mnt/ai-data/training/datasets/mix/fineweb-sample.jsonl --text-field text--tokenizer /mnt/ai-data/training/tokenizers/verify-49k --out /mnt/ai-data/training/shards/verify-49k --max-tokens 5_000_000
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container ai-experiments-prepare-run-e7d8de983ec4 Creating 
Container ai-experiments-prepare-run-e7d8de983ec4 Created 
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
[WARN] nvidia-smi not found: no GPU visible in this container

[RUN] exec: python -m training prepare-data --source /mnt/ai-data/training/datasets/mix/fineweb-sample.jsonl --text-field text --tokenizer /mnt/ai-data/training/tokenizers/verify-49k --out /mnt/ai-data/training/shards/verify-49k --max-tokens 5_000_000
=== Data preparation (Phase 0 D2) ===

[INFO] Source: /mnt/ai-data/training/datasets/mix/fineweb-sample.jsonl
[INFO] Tokenizer: /mnt/ai-data/training/tokenizers/verify-49k
[INFO] Output: /mnt/ai-data/training/shards/verify-49k
[RUN] Tokenizing and writing shards
  [INFO] Progress: 1,047,208 tokens, 1,000 documents
  [INFO] Progress: 2,084,408 tokens, 2,000 documents
  [INFO] Progress: 3,156,506 tokens, 3,000 documents
  [INFO] Progress: 4,131,607 tokens, 4,000 documents
  [INFO] Progress: 5,000,466 tokens, 4,842 documents

[OK] Shards written: 1 (uint16, 536,870,912 tokens/shard)
[OK] Total: 5,000,466 tokens, 4,842 documents
[OK] Manifest: /mnt/ai-data/training/shards/verify-49k/manifest.json
➜  ai-experiments git:(SFT-verification) ✗ python3 -c "import json;m=json.load(open('/mnt/ai-data/training/shards/verify-49k/manifest.json'));print(m['dtype'],m['tokenizer_hash'],m['total_tokens'])"

uint16 8fada2fadd3e8c29 5000466
➜  ai-experiments git:(SFT-verification) ✗ cd ~/workspace/ai-experiments
OVR="data.source=/mnt/ai-data/training/shards/verify-49k tokenizer.path=/mnt/ai-data/training/tokenizers/verify-49k model.scratch.vocab=49152"
make train-validate CONFIG=training/recipes/reference/tiny-e2e.yaml ARGS="$OVR"; echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training python -m training validate training/recipes/reference/tiny-e2e.yaml data.source=/mnt/ai-data/training/shards/verify-49k tokenizer.path=/mnt/ai-data/training/tokenizers/verify-49k model.scratch.vocab=49152
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container ai-experiments-training-run-947717f91eea Creating 
Container ai-experiments-training-run-947717f91eea Created 
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training validate training/recipes/reference/tiny-e2e.yaml data.source=/mnt/ai-data/training/shards/verify-49k tokenizer.path=/mnt/ai-data/training/tokenizers/verify-49k model.scratch.vocab=49152
[OK] Schema: training/recipes/reference/tiny-e2e.yaml loads as a typed recipe
[INFO] Overrides applied: data.source=/mnt/ai-data/training/shards/verify-49k tokenizer.path=/mnt/ai-data/training/tokenizers/verify-49k model.scratch.vocab=49152
[OK] Validation: run tiny-e2e passes all cross-field invariants
exit=0
```

The `[WARN] nvidia-smi not found` line under `prepare-data` is expected: the
`prepare` service reserves no GPU. The `--text-field text--tokenizer` join in
the echoed `docker compose` line is a paste artifact of the terminal; the
`[RUN] exec:` line the entrypoint prints is the argv the container ran, and it
carries the space.

Status: PENDING

## FS.1 Reference loop end to end

Implementation: `training/cli.py` (`cmd_train`), `training/runtime/loop.py`,
`training/runtime/checkpoint.py`, `training/runtime/tracking.py`,
`training/recipes/loader.py` (`create_run_dir`).

Carries the archived P0.6, never run. The entry every later one repeats:
twenty steps of the scratch model through `make train`, checkpointing every
five. Beyond the loop finishing, it asserts the two things the later tiers
need from a run directory: the tracker's first `metrics.jsonl` record passes
the shared report envelope, and it carries the four header fields the
dashboard shows for a training run (`schemas/run_metrics.json`). Existing
code; depends on FS.0.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="$OVR run.name=fs1 schedule.total_steps=20"
echo "exit=$?"
RUN=$(dirname "$(ls -t /mnt/ai-data/training/runs/fs1/*/metrics.jsonl | head -1)")
ls "$RUN" /mnt/ai-data/training/runs/fs1/checkpoints/
cat /mnt/ai-data/training/runs/fs1/checkpoints/step_000000020/meta.json
head -1 "$RUN/metrics.jsonl" | python3 -c "import sys,json;r=json.load(sys.stdin);print(r['report'],r['tier'],r['model_id'],r['recipe_hash'][:12],r['schedule'])"
uv run --no-sync python -c "from common.reports import validate_report; import json,sys; validate_report(json.loads(open(sys.argv[1]).readline())); print('envelope ok')" "$RUN/metrics.jsonl"
```

Expected signal: `[RUN] Training fs1: 20 total steps`, then `[OK] Run
complete: step 20, 5,120 tokens seen` (twenty steps of two sequences of 128
tokens), a `Final loss:` line, `Run directory:` and `Checkpoints:` lines, and
exit 0. The run directory holds `config.yaml` and `metrics.jsonl`; the
checkpoint root holds `step_000000005` through `step_000000020` less what
retention removed (`retain_last: 2`, `retain_every: 10`: `step_000000010`,
`step_000000015`, and `step_000000020` remain) and a `latest` link. The
`meta.json` carries `step: 20`, `tokens_seen`, `loss`, `recipe_hash`,
`checkpoint_format: torch`, `optimizer: adamw` and the resolved
`attention_implementation`. The header line prints `run_metrics pretrain`,
the model id, the hash prefix, and the schedule object; the envelope check
prints `envelope ok`. A `Final loss:` that is `nan` is not a pass of this
entry, whatever the exit code.

Observed:

```
```

Status: PENDING

## FS.2 Resume from latest through the CLI

Implementation: `training/cli.py` (`cmd_train`, the `Resuming` branch),
`training/runtime/checkpoint.py` (`resolve_resume`, `load_checkpoint`),
`training/runtime/loop.py`.

P0.5 verified resume through `tests/hardware/test_gpu.py` in-process; this
entry is the same property through the CLI path a real interruption uses,
`checkpoint.resume=latest`, on the FS.1 run. The bitwise bar belongs to the
unit test; here the criterion is that the run continues rather than restarts.
Existing code; depends on FS.1.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="$OVR run.name=fs1 schedule.total_steps=30 checkpoint.resume=latest"
echo "exit=$?"
ls /mnt/ai-data/training/runs/fs1/checkpoints/
cat /mnt/ai-data/training/runs/fs1/checkpoints/step_000000030/meta.json
```

Expected signal: `[RUN] Resuming (latest) fs1: 30 total steps`, a `[loop]
WARN: resuming with a recipe that differs from the checkpoint's` line (the
step count changed, which is the warning's purpose), then `[OK] Run complete:
step 30, 7,680 tokens seen` and exit 0: ten new steps on top of the twenty
FS.1 trained, with `tokens_seen` continuing from 5,120 rather than starting
over. `step_000000030` appears and its `meta.json` reads `step: 30`. A run
that prints `Training` rather than `Resuming`, or reports 7,680 tokens seen
after thirty fresh steps from a zero start, restarted.

Observed:

```
```

Status: PENDING

## FS.3 Staging and prefetch under the D9 bar

Implementation: `training/data/staging.py` (`stage_shards`),
`training/data/loader.py` (`CudaPrefetcher`), `training/runtime/loop.py`
(`data_wait_ms`), asserted by
`tests/hardware/test_ssd.py::test_data_wait_under_the_d9_bar` (`hw_full`,
`ssd`, `gpu`).

Carries the archived P0.10. Two halves: the staging copy into the `/staging`
tmpfs the compose file mounts, observed in the run log, and the overlap bar,
mean `data_wait_ms` under 5 percent of mean step time, which the hardware
test computes and asserts on its own synthetic shards. Existing code; depends
on FS.0.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="$OVR run.name=fs3 schedule.total_steps=20 staging.enabled=true data.workers=2 data.pin_memory=true" 2>&1 | tee /tmp/fs3.log
echo "exit=$?"
grep '\[staging\]' /tmp/fs3.log
RUN=$(dirname "$(ls -t /mnt/ai-data/training/runs/fs3/*/metrics.jsonl | head -1)")
python3 -c "import json,sys;w=[json.loads(l).get('data_wait_ms') for l in open(sys.argv[1])];w=[x for x in w if x is not None];print(len(w),'steps, mean data_wait_ms',sum(w)/len(w))" "$RUN/metrics.jsonl"
uv run pytest -m hw_full tests/hardware/test_ssd.py -k data_wait -v
```

Expected signal: the log carries `[staging] OK: staged 0.01 GB to
/staging/verify-49k` (the ten-megabyte shard set, well under `max_gb`) before
the first step, the run ends `[OK] Run complete: step 20, 5,120 tokens seen`
with exit 0, the metrics file carries a `data_wait_ms` value on every step
and prints their mean, and the hardware test passes at `[TIER] full` with no
skip. The mean printed here is diagnostic (the tiny model's step is too short
for the 5 percent ratio to mean much); the test is the bar.

Observed:

```
```

Status: PENDING

## FS.4 Staging refusal over the budget

Implementation: `training/data/staging.py` (`stage_shards`, the `max_gb`
branch), `training/recipes/validate.py` (`_check_staging`).

The D9 acceptance's second clause: staging respects `max_gb` and falls back
to NVMe mmap with a logged reason rather than failing the run. The budget is
set below the shard set's size. Existing code; depends on FS.0.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="$OVR run.name=fs4 schedule.total_steps=5 staging.enabled=true staging.max_gb=0.001" 2>&1 | tee /tmp/fs4.log
echo "exit=$?"
grep '\[staging\]' /tmp/fs4.log
```

Expected signal: one line `[staging] SKIP: shard set 0.01 GB exceeds
staging.max_gb 0.001; reading shards from
/mnt/ai-data/training/shards/verify-49k`, then the run completes `[OK] Run
complete: step 5, 1,280 tokens seen` with exit 0. No `OK: staged` line. A
nonzero exit, or a run that stages anyway, fails the entry.

Observed:

```
```

Status: PENDING

## FS.5 Affinity masks disjoint and pinned

Implementation: `training/runtime/affinity.py` (`resolve_masks`, `apply_mask`,
`worker_init_fn`), `training/runtime/loop.py` (the `affinity.enabled` branch),
`scripts/training/pin_run.sh`, asserted by
`tests/hardware/test_cpu.py::test_affinity_pinning_matches_actual_placement`
(`hw_full`, `cpu`).

The D9 acceptance's third clause: with affinity on, the main process and the
dataloader workers carry disjoint CPU masks, and throughput is not worse than
with it off. The masks are read from the host while the container run is
active, through the container's process ids; the hardware test asserts the
same placement in-process. The `main` and `workers` lists are the recipe
defaults, CCD 0 and CCD 1 of the 9950X with their SMT siblings. Existing
code; depends on FS.1 (its step time is the affinity-off baseline).

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="$OVR run.name=fs5 schedule.total_steps=3000 data.workers=2 affinity.enabled=true" &
sleep 30
CID=$(docker ps --filter name=training --format '{{.ID}}' | head -1)
for PID in $(docker top "$CID" -o pid | tail -n +2); do echo "pid $PID: $(taskset -pc "$PID")"; done
wait
echo "exit=$?"
uv run pytest -m hw_full tests/hardware/test_cpu.py -k affinity -v
```

Expected signal: the `taskset` lines show the main `python -m training`
process with the affinity list `0-7,16-23` and each of the two worker
processes with `8-15,24-31`, no CPU shared between the two sets; the run ends
`[OK] Run complete: step 3000` with exit 0; the hardware test passes at
`[TIER] full`. Record the run's wall clock beside FS.1's scaled to the step
count: a pinned run slower than the unpinned one is the design's
park-the-feature outcome, recorded here rather than fixed, and the entry still
passes on the disjoint masks. If `sleep 30` misses the run, lengthen it; the
3000 steps exist to leave time to read the masks while the container starts.

Observed:

```
```

Status: PENDING

## FS.6 The run lock refuses a second run

Implementation: `training/runtime/lock.py` (`run_lock`, `TrainLockError`),
`training/runtime/loop.py` (`_train_locked`), `training/cli.py` (`cmd_train`,
the `Training failed` branch).

The D9 acceptance's fourth clause, and the mechanism behind the
one-GPU-consumer rule that interp's harvest and SAE training share. A second
`make train` starts while the first holds
`/mnt/ai-data/training/.train.lock`. Existing code; depends on FS.0.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="$OVR run.name=fs6a schedule.total_steps=3000 data.workers=2" &
sleep 30
cat /mnt/ai-data/training/.train.lock; echo
make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="$OVR run.name=fs6b schedule.total_steps=5"
echo "second exit=$?"
wait
echo "first exit=$?"
ls /mnt/ai-data/training/.train.lock 2>&1
```

Expected signal: the lock file reads a JSON object with `pid`, `run: fs6a`
and `created`; the second run prints `[ERROR] Training failed: Training lock
held by pid <n> (run 'fs6a'): /mnt/ai-data/training/.train.lock` followed by
`Solution: wait for the running job to finish (one GPU consumer at a time)`
and exits 1 before allocating anything; the first run completes with exit 0;
the final `ls` reports the lock file absent. A second run that starts
training, or a lock file that survives the first run's exit, fails the entry.

Observed:

```
```

Status: PENDING

## FS.7 Watchdog passive on a healthy run

Implementation: `training/runtime/watchdog.py` (`Watchdog`, `_sampler_loop`,
`on_step`), `training/runtime/loop.py`, `training/runtime/tracking.py`.

Carries the archived P0.11: on a healthy run the watchdog samples and raises
nothing. The field-level assertions on the sampler records are OBS-V2 in
[../observability/observability.md](../observability/observability.md), and
`alerts.jsonl` is OBS-V11 there; this entry asserts the training-side
negative those do not, that a run with the watchdog enabled ends with no
alert, no abort, and no alerts file. Existing code; depends on FS.0.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="$OVR run.name=fs7 schedule.total_steps=20 watchdog.enabled=true watchdog.interval_s=1.0" 2>&1 | tee /tmp/fs7.log
echo "exit=$?"
grep -c '\[watchdog\]' /tmp/fs7.log
RUN=$(dirname "$(ls -t /mnt/ai-data/training/runs/fs7/*/metrics.jsonl | head -1)")
ls "$RUN"
python3 -c "import json,sys;n=sum(1 for l in open(sys.argv[1]) if json.loads(l).get('gpu_temp_c') is not None);print(n,'sampler records')" "$RUN/metrics.jsonl"
```

Expected signal: `[OK] Run complete: step 20, 5,120 tokens seen` with exit
0; the `grep -c` prints `0` (no `[watchdog]` line of any action); the run
directory listing shows `config.yaml` and `metrics.jsonl` and no
`alerts.jsonl`; the sampler count is at least 1 (a twenty-step run at one
sample per second lands a handful). A `[watchdog] ALERT` on a healthy run is a
finding about the thresholds and is recorded, not tuned away.

Observed:

```
```

Status: PENDING

## FS.8 Watchdog abort and the emergency checkpoint

Implementation: `training/runtime/watchdog.py` (`on_step`, `_abort`,
`WatchdogAbort`), `training/runtime/loop.py` (the emergency `save`),
`training/runtime/checkpoint.py` (`apply_retention`, the emergency exemption),
`training/cli.py` (`cmd_train`, the `Watchdog abort` branch).

The D10 integration acceptance the archive did not carry: an intentionally
exploding learning rate trips the loss rule, the loop saves an emergency
checkpoint, and that checkpoint resumes. Thermal and RAM fault injection stay
out of scope, as P0.11 decided; the loss rule needs no fault, only a
learning rate no scratch model survives. The spike rule arms after five
finite losses, so the abort lands on `loss_spike` if the loss climbs first
and on `loss_non_finite` if it overflows first; either is the pass. Existing
code; depends on FS.0.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="$OVR run.name=fs8 schedule.total_steps=200 optimizer.lr=100 watchdog.enabled=true" 2>&1 | tee /tmp/fs8.log
echo "exit=$?"
grep '\[watchdog\]' /tmp/fs8.log
RUN=$(dirname "$(ls -t /mnt/ai-data/training/runs/fs8/*/metrics.jsonl | head -1)")
cat "$RUN/alerts.jsonl"
ls /mnt/ai-data/training/runs/fs8/checkpoints/
grep -l '"emergency": true' /mnt/ai-data/training/runs/fs8/checkpoints/step_*/meta.json
make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="$OVR run.name=fs8 schedule.total_steps=200 checkpoint.resume=latest"
echo "resume exit=$?"
```

Expected signal: a `[watchdog] ABORT loss_spike: value ...` or `[watchdog]
ABORT loss_non_finite: value ...` line, then `[ERROR] Watchdog abort:
<signal> value <v> crossed threshold <t>`, `Emergency checkpoint saved
(emergency=true, retention-exempt)`, `Resume after diagnosis with
checkpoint.resume=latest`, and exit 1 well before step 200; `alerts.jsonl`
holds one record with that `signal`, `value`, `threshold` and `action:
abort`; the checkpoint listing includes the aborting step's directory, and
`grep -l` names its `meta.json` as the one carrying `"emergency": true`. The
resume prints `[RUN] Resuming (latest) fs8: 200 total steps` and, with the
recipe's own learning rate restored, either completes or aborts again on a
loss the emergency checkpoint already carries; record which, since the
entry's claim is that the checkpoint loads and the run continues from it,
not that it converges. A run that reaches step 200 at `lr=100` without an
abort fails the entry.

Observed:

```
```

Status: PENDING
