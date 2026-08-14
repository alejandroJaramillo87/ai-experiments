# Interpretability tier verification

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

I1 workbench, the PS1 Neuronpedia harness, the write-back path (W1 through
W4), P4-D1 organisms, P4-D2 activation harvesting with garbage collection,
and P4-D3 SAE training and evaluation. The interp anchor for pretrained
SAEs is Gemma-2-2B (Gemma Scope); Gemma 4 runs on the hook-based paths
(harvest, probes, steering) through the `interp/families.py` registry.

Every `make` target here follows the same dispatch chain: the `Makefile`
runs `docker compose -f docker-compose.interp.yaml run --rm interp
python -m interp <subcommand>`, which enters `interp/__main__.py`,
dispatches through the argparse handlers in `interp/cli.py`, and lands in
the implementation module. Each entry below lists its implementation files
so a failure can be read at the source without re-tracing the chain.

## I.1 Workbench smoke (I1)

Implementation: `interp/commands/smoke_test.py`, `common/smoke.py`,
`docker/interp/Dockerfile.interp`.

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

Implementation: `scripts/interp/np_env.sh`, `scripts/interp/np_smoke.sh`,
`docker/interp/neuronpedia-pin.txt`,
`docker/interp/neuronpedia.override.yaml`.

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

Implementation: `interp/commands/steer_extract.py`,
`interp/steering/extract.py`, `interp/steering/export.py`,
`interp/families.py`.

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-32 — `interp/steering/extract.py:78` — Pinned repeng 0.4.0 cannot be imported under pinned numpy 2.4.6: np.float_ in eagerly evaluated annotations … repeng is also absent from the interp image, so the import fails on absence first. Expect this step to fail at import until the `extract_repeng` refactor lands.

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

Implementation: `interp/steering/wiring.py`,
`interp/commands/steer_scorecard.py`, `interp/steering/scorecard.py`,
`docker-compose.inference.yaml` (the llama-gpu control-vector environment).

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

Implementation: `interp/commands/probe_capture.py` with
`interp/probes/capture.py`; `interp/commands/probe_train.py` with
`interp/probes/train.py` and `interp/probes/registry.py`;
`interp/commands/probe_eval.py` with `interp/probes/evaluate.py`.

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

## I.6 Served-request log (W3)

Implementation: `rust-harness/harness/src/interplog.rs`,
`rust-harness/harness/src/server.rs`, the `harness-serve` service in
`docker-compose.rust-harness.yaml`.

Replaces the retired request-log proxy and the retired full-precision
monitored endpoint (the former I.6 and I.6b), which existed only because
llama-server has neither structured request logging nor an activation read
path. The harness has both, so the record is written by the process that
answered the request and carries the probe scores taken while answering it.
Nothing on the llama.cpp stack replaces them: those services now serve without
interp observation, which is an accepted consequence rather than an oversight.

The sandbox proves the format and the wiring against a synthetic model
(`harness/tests/serve_interp_log.rs`). What it cannot prove is that a real
probe over real E4B produces scores worth reading, which is what this entry is
for.

**A probe from before R10.5 cannot be substituted.** `main.rs` applies two
refusals when loading one: the sidecar's `model_hash` against the sha256 of the
served `model.safetensors`, and the probe's width against the model's embedding
dimension. A probe fitted on any other checkpoint fails both, so the chain below
starts at capture rather than at the registry.

Preamble, in the workbench and on the GPU (serializes against other GPU work):

```bash
# Container paths: probe-capture runs in the interp container, which mounts
# the models tree at /app/models (docker-compose.interp.yaml). A host path
# resolves to nothing there and is refused.
$ E4B=/app/models/hf/gemma-4-E4B
$ make probe-capture ARGS="--model $E4B \
    --dataset interp/probes/prompts/refusal.jsonl \
    --layers 8,16,24 --out /mnt/ai-data/interp/activations/refusal-v1"
$ make probe-train ARGS="--capture /mnt/ai-data/interp/activations/refusal-v1 \
    --name refusal-v1"
```

**Stop if the capture reports `model_hash unknown`.** Passing a local checkpoint
directory to `--model` makes `probe-capture` derive the identity; the sentinel
means it could not, and `main.rs` responds to it by *skipping* the identity
check rather than failing it. A probe carrying it will load into a serving
harness unverified, and this entry would pass while proving less than it claims.
Cross-check on the host with
`sha256sum /mnt/ai-data/models/hf/gemma-4-E4B/model.safetensors`.

Command:

```bash
# probe-train leaves probe.npz, which the harness cannot read; this writes the
# safetensors image and its sidecar beside it, and prints the container path
$ make probe-export ARGS="--probe /mnt/ai-data/interp/probes/refusal-v1"
$ make harness-serve HARNESS_PROBE=/app/probes/refusal-v1/probe.safetensors
# Loading the checkpoint takes minutes. This blocks until the endpoint
# answers; `curl -sf` does not -- it returns at once against a port with
# nothing yet listening, and the rest of the block then runs against a
# server still reading weights.
$ python -c 'from common import serving; print(serving.wait_healthy(serving.SERVICES["harness"]))'
$ curl -s --max-time 900 -X POST localhost:8005/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 8}'
$ tail -1 /mnt/ai-data/interp/logs/harness/"$(date -u +%F)".jsonl | python3 -m json.tool
$ wc -l /mnt/ai-data/interp/logs/harness/"$(date -u +%F)".jsonl
$ make harness-serve-down    # after I.7, which wants the service up
```

Expected signal: one JSONL record per completed request, carrying `messages`,
`response_text`, `usage`, `latency_s`, and a `probe` block whose `scored`
counts the positions of that request alone and whose `max_score` is finite. The
record is already on disk before the process is stopped, since it is ended by a
signal and never closes the file.

Two things that would be silent failures. `scored` growing across requests
means the monitor state is not being drained per request, which is both a wrong
record and an unbounded buffer. And a `probe` block absent from every record
while `HARNESS_PROBE` is set means the probe never loaded -- check stderr for
the identity refusal.

Observed: 2026-08-14, probe `refusal-v1` captured from
`/app/models/hf/gemma-4-E4B` at layers 8/16/24 over
`interp/probes/prompts/refusal.jsonl` (258 examples, 129/129), layer 8
selected. The capture derived `model_hash`
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`, equal to
`sha256sum` of the served `model.safetensors`, so the identity check ran
rather than being skipped.

```
{"id":"chatcmpl-0","object":"chat.completion","created":1786672276,
 "model":"/app/models/hf/gemma-4-E4B","choices":[{"index":0,"message":
 {"role":"assistant","content":"2+2?\n18\n"},"finish_reason":"length"}],
 "usage":{"prompt_tokens":13,"completion_tokens":8,"total_tokens":21}}

$ tail -1 .../2026-08-14.jsonl | python3 -m json.tool
    "response_text": "2+2?\n18\n",
    "usage": {"completion_tokens": 8, "prompt_tokens": 13, "total_tokens": 21},
    "latency_s": 2.433646459,
    "status_code": 200,
    "probe": {
        "name": "refusal-v1", "layer": 8, "site": "layer_output",
        "threshold": 0.9548591, "scored": 21, "position_first": 0,
        "max_score": 0.99316895,
        "flags": [ ... 12 positions, 0.956 to 0.993 ... ]
    }
$ wc -l .../2026-08-14.jsonl
1
```

Both silent failures are excluded: `scored` is 21, exactly `total_tokens`
(13 + 8), so the monitor is drained per request rather than accumulating; and
the `probe` block is present, so the probe loaded past both refusals.

**The probe is not usable as a detector, and this entry does not claim it
is.** It flagged 12 of 21 positions on `2+2?` at up to 0.993 against a 0.955
threshold -- a benign arithmetic prompt scoring as refusal nearly everywhere.
The training AUROC was 1.00 at all three layers, which read as suspicious at
the time and is now explained: `capture_activations` fits on the last
non-padded token of each prompt, while the harness scores every position
in-band, so a threshold calibrated on final-position activations means
nothing at position 3. What I.6 verifies is the path from a served request to
a scored record on disk, which holds. Task #171 owns the calibration gap.

The reply `"2+2?\n18\n"` is a base checkpoint continuing text rather than
answering, which is task #126, not a fault in the serving path.

Status: PASS

## I.7 Probe monitor (W4)

Implementation: `interp/probes/monitor.py`,
`interp/commands/probe_monitor.py`.

Reads the scores in the log written by I.6. It loads no model, so unlike every
earlier version of this entry there is no GPU precondition and no requirement
that inference be down.

Command:

```bash
$ make probe-monitor ARGS="--window 7d --log-dir /mnt/ai-data/interp/logs/harness"
# run it again, unchanged, to exercise the drift delta
$ make probe-monitor ARGS="--window 7d --log-dir /mnt/ai-data/interp/logs/harness"
```

Expected signal: `report.json` (`kind: probe_monitor`, shared envelope) plus
`flagged.jsonl`, with `scored: "in_band"`, per-probe distributions, the
`target_services` observed, and log retention enforced. The second run carries
`delta_vs_previous` naming the first run, with a mean delta of 0 over an
unchanged window.

Run it with a service up. It must succeed: the refusal this entry used to
verify was a consequence of the replay owning the GPU, and it has been removed
deliberately.

```bash
$ make harness-serve
$ make probe-monitor ARGS="--window 7d --log-dir /mnt/ai-data/interp/logs/harness"
# expected: exit 0, not a services-are-up refusal
```

Observed: 2026-08-14, both runs against the single record I.6 wrote, with
harness-serve up throughout. Neither refused.

```
[OK] Reported 1/1 records (1 flagged, skipped {'malformed_json': 0,
     'no_timestamp': 0, 'out_of_window': 0, 'no_text': 0, 'no_scores': 0})
[OK] refusal-v1: mean 0.9932, p95 0.9932, over-threshold 1.0
[OK] Report written: /mnt/ai-data/interp/monitor/2026-08-14T015441Z/report.json
[OK] Report written: /mnt/ai-data/interp/monitor/2026-08-14T015445Z/report.json

$ python3 -m json.tool .../2026-08-14T015445Z/report.json
    "kind": "probe_monitor", "schema_version": 2, "scored": "in_band",
    "counts": {"total": 1, "reported": 1, "flagged": 1,
               "dropped_over_cap": 0, "retention_removed_files": []},
    "probes": {"refusal-v1": {
        "mean": 0.9932, "p50": 0.9932, "p95": 0.9932,
        "fraction_over_threshold": 1.0, "count": 1,
        "delta_vs_previous": {"mean": 0.0, "fraction_over_threshold": 0.0,
                              "previous_run": "2026-08-14T015441Z"}}},
    "target_services": ["harness"]
```

Every part of the expected signal is present: `scored: "in_band"`,
`target_services` naming the harness, retention enforced with nothing yet old
enough to remove, and the second run's `delta_vs_previous` naming the first
with a mean delta of 0 over the unchanged window.

`fraction_over_threshold 1.0` is the monitor faithfully reporting the
miscalibrated probe described in I.6, not a monitor defect: one record, whose
single aggregated score is above threshold. It is stated here so a later
reader does not mistake it for a baseline.

Status: PASS

## I.8 Organisms (P4-D1)

Implementation: `interp/commands/organism.py`,
`interp/organisms/generate.py`, `interp/organisms/verify.py`,
`interp/organisms/seal.py`; the pipeline re-invokes the phase 1 SFT and
export commands.

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
(docs/interp/experiments/2026-07-16-hidden-objective-organism-deferral.md).

Observed:

```
```

Status: PENDING

## I.9 Harvest with disk preflight (P4-D2)

Implementation: `interp/commands/harvest.py`, `interp/harvest.py`,
`interp/families.py`, `training/runtime/lock.py`.

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

The capture is chunked (KB-58): `harvest-smoke.yaml` sets
`chunk_tokens: 700000` against 2M tokens, so expect three
`Caching activations` progress bars, not one. The reported token count
exceeds the request — 2875392 against 2000000 — because SAELens writes
whole buffers; that is correct, and `manifest.json` records both figures.

Observed (2026-08-08):

```
Caching activations: 100%|...| 3/3 [00:25<00:00,  8.46s/it]
[OK] Harvest complete: 2875392 tokens/hook,
     hooks ['model.language_model.layers.17'], 34125.6 tokens/s
```

Status: VERIFIED

## I.9b Chunked-capture crash recovery (P4-D2, KB-58)

Implementation: `interp/harvest.py` (`chunk_plan`, `_SaeLensSession`,
`_consolidate_chunks`), `tests/interp/test_saelens_contract.py`.

The point of chunking is behaviour under a crash, so a clean I.9 run
verifies nothing about it. Interrupt a chunked capture and resume it.

Command:

```bash
$ rm -rf /mnt/ai-data/interp/activations/harvest-smoke
$ make harvest RECIPE=training/recipes/interp/harvest-smoke.yaml
```

Press Ctrl-C once a `3/3` progress bar has appeared at least once and
before `[OK] Harvest complete`; any kill in that window banks at least one
chunk. Then inspect and resume — do not delete the capture directory:

```bash
$ cat /mnt/ai-data/interp/activations/harvest-smoke/progress.json
$ make harvest RECIPE=training/recipes/interp/harvest-smoke.yaml
```

Expected signal: `progress.json` records `completed_chunks` for the hook;
the rerun does **not** raise `Activations directory ... is not empty`, runs
only the outstanding chunks, and reports the same total as an
uninterrupted run. That equality is the criterion: it is what shows the
resume neither dropped nor duplicated a chunk.

Observed (2026-08-08), killed inside chunk 3:

```
{ "completed_chunks": { "model.language_model.layers.17": 2 },
  "measured_tokens": { "model.language_model.layers.17": 1916928 } }
```

```
Caching activations: 100%|...| 3/3 [00:24<00:00,  8.17s/it]
[OK] Harvest complete: 2875392 tokens/hook,
     hooks ['model.language_model.layers.17'], 26535.4 tokens/s
```

1916928 banked plus 958464 recaptured equals 2875392, the uninterrupted
total. This run found a defect in the KB-58 fix itself (the interrupted
chunk's directory was not cleared before recapture); the output above is
from the repaired code.

Status: VERIFIED

## I.10 Capture garbage collection (P4-D2)

Implementation: `interp/commands/interp_gc.py`, `interp/manifest.py`.

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

**Known bugs likely to surface here:** KB-61, confirmed fixed on 2026-08-08.
The first `make sae-train` failed with `ValueError: You must either pass in a
dataset or specify a dataset_path in your configutation` — SAELens requires a
dataset even when every activation is read from disk.

Reaching a completed train-and-evaluate round trip took five further defects,
all found by this step and all fixed on 2026-08-08: KB-63 (W&B artifact
staging against the read-only rootfs, which killed a finished run at
teardown), KB-64 (SAELens `output_path` leaking 50MB of weights into the
repository root), KB-65 (a second W&B run directory in the CWD), KB-62 (the
manifest recording a `final_loss` and a `measured_l0` it never measured) and
KB-66 (a redundant `save_model` overwriting the inference-format checkpoint,
so no SAE could be loaded back). They surfaced in that order because each had
to be cleared before the next became observable, which is why the step needed
six runs rather than one.

Implementation: `interp/commands/sae_train.py` with `interp/sae/train.py`;
`interp/commands/sae_eval.py` with `interp/sae/evaluate.py`;
`interp/sae/registry.py`.

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

Observed (2026-08-08, after the six fixes above):

```
$ make sae-train RECIPE=training/recipes/interp/sae-smoke.yaml
[OK] SAE training complete: sae-smoke, width 4096, k 32, 22.8s
  Next: make sae-eval SAE=/mnt/ai-data/interp/saes/sae-smoke CAPTURE=/mnt/ai-data/interp/activations/harvest-smoke

$ python -c "import json;print(json.load(open('/mnt/ai-data/interp/saes/sae-smoke/cfg.json'))['architecture'])"
jumprelu

$ make sae-eval SAE=/mnt/ai-data/interp/saes/sae-smoke CAPTURE=/mnt/ai-data/interp/activations/harvest-smoke
[OK] Evaluation complete: explained variance 0.324558, L0 30.13, 1000000 tokens
```

`eval_report.json` core block: `explained_variance 0.324558`, `l0 30.13`,
`mse 1.173056`, `tokens_evaluated 1000000`. The manifest carries no quality
field, only the `quality_metrics` pointer at the report.

The L0 is the signal worth reading. It is 30.13 against a configured `k` of
32: BatchTopK selects k on average during training, and on held-out
activations some features fall below the learned threshold. Before KB-62 this
field read exactly `32.0` in the manifest, because it was the target copied
into a slot labelled "measured" — a 6% sparsity shortfall that no artifact
recorded. The two `mse` figures are consistent despite looking unrelated:
`core_metrics` divides by element (`acts.size`) while W&B sums over `d_in` per
token, and 1.173056 x 1536 = 1802 against W&B's 1938 on the training slice.
`explained_variance` 0.3247 held-out against 0.3110 training is the expected
small gap.

Status: VERIFIED

## I.12 SAEBench isolated environment (P4-D3)

Implementation: `scripts/interp/saebench_env.sh`, which is self-contained
and builds its own virtual environment.

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
