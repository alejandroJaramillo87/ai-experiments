# Interpretability tier verification

I1 workbench, the PS1 Neuronpedia harness, the write-back path (W1 through
W4), P4-D1 organisms, P4-D2 activation harvesting with garbage collection,
and P4-D3 SAE training and evaluation. The interp anchor for pretrained
SAEs is Gemma-2-2B (Gemma Scope); Gemma 4 runs on the hook-based paths
(harvest, probes, steering) through the `interp/families.py` registry.

## I.1 Workbench smoke (I1)

Command:

```bash
$ make interp-smoke
```

Expected signal: all seven steps `[OK]`, including the gated Gemma-2-2B
plus Gemma Scope SAE chain (requires `HF_TOKEN` in `.env.interp`); the
versions manifest lands at `/mnt/ai-data/interp/versions.json`.

Observed:

```
```

Status: PENDING

## I.2 Neuronpedia harness (PS1)

Command:

```bash
$ make np-clone
$ make np-env
$ make np-build
$ make np-up
$ make np-smoke
```

Expected signal: the checkout pins to `docker/interp/neuronpedia-pin.txt`;
np-smoke passes its checks — loopback-only bindings, webapp on 3001,
postgres up, and sm_120 visible to the GPU services. `make np-down`
afterwards stops the stack cleanly.

Observed:

```
```

Status: PENDING

## I.3 Control-vector extraction and export (W1)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-31 — `interp/steering/extract.py:106` — extract_repeng never passes its computed layer list to ControlVector.train, so layer_range is silently ignored …
- KB-32 — `interp/steering/extract.py:78` — Pinned repeng 0.4.0 cannot be imported under pinned numpy 2.4.6: np.float_ in eagerly evaluated annotations …

Command:

```bash
$ make steer-extract ARGS="--method repeng --model google/gemma-4-E2B \
    --pairs interp/steering/prompts/honesty_pairs.jsonl \
    --out /mnt/ai-data/interp/vectors/honesty-e2b"
```

Expected signal: exit 0; the vector directory contains the GGUF control
vector plus a manifest recording the model identity, pairs hash, and
per-layer directions. The repeng band derives from the unwrapped text
config (registry path).

Observed:

```
```

Status: PENDING

## I.4 Steered serving and scorecard (W1)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-33 — `interp/commands/steer_scorecard.py:82` — steer-scorecard: except tuple omits OSError, ValueError, and ServingError, so missing prompt files and mid-run …
- KB-34 — `interp/steering/scorecard.py:85` — load_prompts and load_system_prompt let raw json.JSONDecodeError (and TypeError on non-object lines) escape, …

Command:

```bash
$ make steer-up VECTOR=/mnt/ai-data/interp/vectors/honesty-e2b SCALE=1.0
$ make steer-scorecard VECTOR=/mnt/ai-data/interp/vectors/honesty-e2b ATTRIBUTE="honesty"
$ make steer-down
```

Expected signal: the service recreates steered; the three-arm scorecard
(baseline, tuned system prompt, steered at the scale sweep) writes its
report (`kind: steering_scorecard`, shared envelope) and the service ends
unsteered — including when a scorecard arm fails mid-run (the finally
restore).

Observed:

```
```

Status: PENDING

## I.5 Probe capture, train, eval (W2)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-35 — `interp/commands/probe_eval.py:53` — probe-eval does not catch ManifestError, so a missing or corrupt capture manifest crashes with a raw traceback
- KB-36 — `interp/probes/capture.py:169` — run_capture kill window during re-capture leaves the previous run's valid manifest pointing at partially …
- KB-37 — `interp/probes/capture.py:93` — capture_activations indexes the batch dimension: decoder blocks return a bare tensor under pinned transformers …
- KB-38 — `interp/probes/score.py:45` — score_texts discards the capture tokenizer fingerprint, so the documented score-time tokenizer/chat-template …

Command:

```bash
$ make probe-capture ARGS="--model google/gemma-4-E2B --dataset <labeled.jsonl> \
    --layers 10,15,20 --out /mnt/ai-data/interp/captures/verify"
$ make probe-train ARGS="--capture /mnt/ai-data/interp/captures/verify --name verify-probe"
$ make probe-eval ARGS="--probe /mnt/ai-data/interp/probes/verify-probe --label-name <attribute>"
```

Expected signal: capture writes per-layer shards plus a manifest under
nnsight (decoder blocks resolved through the family registry); training
selects the best layer by validation AUROC and registers the probe;
evaluation writes the report with the mandatory black-box baselines and a
deployable verdict.

Observed:

```
```

Status: PENDING

## I.6 Request-log proxy (W3)

Command:

```bash
$ make logproxy-up
$ curl -s -X POST localhost:8012/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 8}'
$ make logproxy-down
```

Expected signal: the proxy forwards to llama-gpu and appends a JSONL
record (timestamp, prompt, response) to the serving log directory; the
response body matches the upstream service's.

Observed:

```
```

Status: PENDING

## I.7 Probe monitor (W4)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-38 — `interp/probes/score.py:45` — score_texts discards the capture tokenizer fingerprint, so the documented score-time tokenizer/chat-template …
- KB-39 — `interp/probes/monitor.py:196` — run_monitor enforces 30-day retention before reading the log window, so any --window longer than RETENTION_DAYS …
- KB-40 — `interp/probes/monitor.py:211` — Monitor replay always loads the hub 'main' revision: run_monitor never plumbs a revision to score_texts, and …

Command (batch job; refuses while inference is up):

```bash
$ make down
$ make probe-monitor ARGS="--window 7d --probes /mnt/ai-data/interp/probes/verify-probe \
    --model google/gemma-4-E2B --model-hash <hash>"
```

Expected signal: the monitor replays the log window, scores every probe,
writes `report.json` (`kind: probe_monitor`, shared envelope) plus
`flagged.jsonl`, enforces log retention, and refuses with the
services-are-up message when run against a live stack.

Observed:

```
```

Status: PENDING

## I.8 Organisms (P4-D1)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-41 — `interp/commands/organism.py:161` — organism generate does not plumb --timeout through to the teacher harness, so the flag only affects the …
- KB-42 — `interp/organisms/generate.py:205` — parse_questions accepts n=0, so --questions-per-fact 0 silently produces an empty held-out QA set
- KB-43 — `interp/organisms/seal.py:103` — SIGKILL between publishing ground_truth.json and writing the manifest bricks sealing: the exception-only …
- KB-44 — `interp/organisms/verify.py:214` — locality_check silently passes NaN benchmark metrics
- KB-45 — `interp/organisms/verify.py:221` — locality_check's passed verdict ignores unmatched metrics despite its documented contract

Command (stages have separate GPU/serving preconditions; `make organism
RECIPE=...` prints this pipeline):

```bash
$ make organism-generate RECIPE=training/recipes/organisms/taboo.yaml TEACHER=/mnt/ai-data/models/<teacher>.gguf
$ make prepare-sft ARGS="--source <out_dir>/data-blind ..."
$ make train-sft CONFIG=training/recipes/organisms/taboo.yaml
$ make export-gguf CHECKPOINT=<run checkpoint> ARGS="--merge ..."
$ make eval-ft ARGS=...
$ make organism-verify RECIPE=training/recipes/organisms/taboo.yaml BASE_METRICS=<...> SUBJECT_METRICS=<...>
$ make organism-seal RECIPE=training/recipes/organisms/taboo.yaml ARGS="--checkpoint ... --gguf ..."
$ make organism-check-seal DIR=/mnt/ai-data/interp/organisms/<name>
```

Expected signal: generation draws sealed ground truth and the corpus;
verification writes the report (`report: organism-verification`, shared
envelope) with passing implant and locality checks; the null organism
(`null.yaml`) fails the implant check, proving the harness measures the
implant; seal writes the blinded public manifest and check-seal re-verifies
the hash. The hidden-objective organism is deferred
(docs/interpretability/experiments/2026-07-16-hidden-objective-organism-deferral.md).

Observed:

```
```

Status: PENDING

## I.9 Harvest with disk preflight (P4-D2)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-46 — `interp/harvest.py:327` — Harvest resume identity omits the corpus id, so swapping HF datasets mid-resume silently stitches cross-corpus …

Harvests can reach roughly 1.4TB per layer at full scale; the preflight,
not cleanup, is the defense. Check free space explicitly before any
non-smoke harvest.

Command:

```bash
$ df -h /mnt/ai-data
$ make harvest-check-hooks RECIPE=training/recipes/interp/harvest-smoke.yaml
$ make harvest RECIPE=training/recipes/interp/harvest-smoke.yaml
```

Expected signal: the hook-dimension gate prints the residual width for
each layer (hidden_size, not the 256-wide PLE tensor); the harvest runs
under the D9 lock and D10 watchdog and writes the capture manifest with
registry-derived `hooks`, `model_type`, and throughput. A projection
exceeding free space minus the floor refuses with the disk-preflight
message (verifiable with an absurd `--override harvest.total_tokens=...`).

Observed:

```
```

Status: PENDING

## I.10 Capture garbage collection (P4-D2)

Command:

```bash
$ make interp-gc ARGS="--dry-run"
$ make interp-gc ARGS="--mark-validated /mnt/ai-data/interp/activations/<capture>"
$ make interp-gc
```

Expected signal: dry-run lists deletable validated captures oldest-first;
after marking, the real run deletes the capture's shard payloads while
keeping `manifest.json`.

Observed:

```
```

Status: PENDING

## I.11 SAE training and evaluation (P4-D3)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-47 — `interp/commands/sae_eval.py:59` — sae-eval: malformed or text-less --delta-loss-texts JSONL escapes as a raw KeyError/JSONDecodeError traceback
- KB-48 — `interp/commands/sae_eval.py:59` — Empty --delta-loss-texts file produces a fabricated all-zero delta-LM-loss report
- KB-49 — `interp/sae/evaluate.py:207` — evaluate_sae's model-identity refusal gate is vacuous for every HF-model capture: 'unknown' matches 'unknown'
- KB-50 — `interp/sae/train.py:223` — capture_manifest_xxhash is invalidated by mark_validated within the same run: run_sae_training hashes the …
- KB-51 — `interp/sae/train.py:227` — SAE manifest omits model_revision, so delta-LM-loss silently evaluates the hub default branch instead of the …

Command (the calibration pair targets Gemma-2-2B by the ecosystem-anchor
decision; the smoke recipe targets Gemma 4):

```bash
$ make sae-train RECIPE=training/recipes/interp/sae-smoke.yaml
$ make sae-eval SAE=/mnt/ai-data/interp/saes/<name> CAPTURE=/mnt/ai-data/interp/activations/<name>
```

Expected signal: training runs SAELens against the capture (refusing a
manifest mismatch), writes SAE weights plus manifest; evaluation writes the
report (`artifact: sae_eval`, shared envelope) with core metrics
(reconstruction, L0) and optional delta LM loss.

Observed:

```
```

Status: PENDING

## I.12 SAEBench isolated environment (P4-D3)

SAEBench pins conflict with the core datasets/numpy pins, so it runs in
its own venv by resolved decision.

Command:

```bash
$ bash scripts/interp/saebench_env.sh
```

Expected signal: the isolated venv builds and SAEBench imports inside it;
the core environment is untouched.

Observed:

```
```

Status: PENDING
