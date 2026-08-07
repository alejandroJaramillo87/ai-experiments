# Interpretability tier verification

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable — both documentation gates exempt this directory by path, so stale references here are not caught for you.**

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

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-37 — `interp/probes/capture.py:93` — capture_activations indexes the batch dimension: decoder blocks return a bare tensor under pinned transformers …

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

Implementation: `docker/interp-logproxy/logproxy.py`,
`docker/interp-logproxy/Dockerfile`, `docker-compose.inference.yaml` (the
`interp-logproxy` service under profile `interp`).

Note: the rust harness provides an in-band successor to the W3/W4 serving-side
machinery (I.6, I.6b, I.7). Its runtime scores probes inside the forward pass
(`docs/interp/rust-harness-interp.md`, "In-band monitoring"), which
removes the proxy, the full-precision endpoint, and the log-replay monitor on
that path by construction. These entries remain valid and required for the
llama.cpp GPU serving stack, which the rust CPU runtime does not replace; the
two verification campaigns are independent.

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

## I.6b Full-precision monitored endpoint

Implementation: `docker-compose.inference.yaml` (the `llama-gpu-bf16` service under
profile `interp`), `docker/interp-logproxy/logproxy.py`.

Precondition for the fidelity claim in I.7. The point of `llama-gpu-bf16` is
that monitored traffic is answered at the precision the probes were fitted at,
so the served tensor type must be confirmed rather than assumed.

ggml defines a `BF16` tensor type, but CUDA backend coverage for it is narrower
than for f16, and some paths upconvert. A service that loads f16 while being
described as bf16 reintroduces exactly the replay mismatch this endpoint exists
to remove, and it does so silently: nothing downstream would report a problem.

Command:

```bash
$ make fullprec-up
$ docker logs llama-gpu-bf16 | grep -iE "general.file_type|type +f(16|32)|bf16"
$ curl -s -X POST localhost:8012/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 8}'
$ tail -1 /mnt/ai-data/interp/logs/llama-gpu-bf16/"$(date -u +%F)".jsonl
$ make fullprec-down
$ curl -sf localhost:8002/v1/health
```

Expected signal: the startup log reports the tensor type the GGUF was built
with, and it matches the `--outtype` used by `convert_hf_to_gguf.py`. The proxy
forwards to the bf16 endpoint and the record's `target_service` reads
`llama-gpu-bf16`. Port 8002 answers its healthcheck unchanged throughout.

Record the observed tensor type verbatim below. If the served type is f16
rather than bf16, that is an acceptable outcome, but it must be written down
here and in `docs/interp/writeback.md`, and the endpoint described as
f16 from then on. Do not record a bf16 result that was not observed.

Observed:

```
```

Status: PENDING

## I.7 Probe monitor (W4)

Implementation: `interp/commands/probe_monitor.py`,
`interp/probes/monitor.py`, `interp/probes/score.py`,
`scripts/interp/preflight.sh`.

Note: the rust harness in-band probe monitor (`--probe`) supersedes this
replay design on the rust path — per-position scoring in the serving process,
no replay-model disclaimer (see the I.6 note). This entry still verifies the
monitor for llama.cpp-served traffic.

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

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
services-are-up message when run against a live stack. The report lists the
`target_services` observed in the window, and its `note` states whether the
replay describes the serving model or only the replay model.

The refusal covers the full-precision endpoint as well as the two production
services, so `make fullprec-up` followed by `make probe-monitor` must also
refuse:

```bash
$ make fullprec-up
$ make probe-monitor ARGS="--window 7d --probes /mnt/ai-data/interp/probes/verify-probe \
    --model google/gemma-4-E2B --model-hash <hash>"
# expected: MonitorError naming http://127.0.0.1:8004/v1/health
$ make fullprec-down
```

To verify the full-precision claim end to end, replay a window served by the
bf16 endpoint and confirm the report does not disclaim fidelity:

```bash
$ make probe-monitor ARGS="--log-dir /mnt/ai-data/interp/logs/llama-gpu-bf16 --window 7d \
    --probes /mnt/ai-data/interp/probes/verify-probe \
    --model google/gemma-2-2b --model-hash <hash>"
# expected: "target_services": ["llama-gpu-bf16"] and a note reading "same precision"
```

Observed:

```
```

Status: PENDING

## I.8 Organisms (P4-D1)

Implementation: `interp/commands/organism.py`,
`interp/organisms/generate.py`, `interp/organisms/verify.py`,
`interp/organisms/seal.py`; the pipeline re-invokes the phase 1 SFT and
export commands.

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

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

Observed:

```
```

Status: PENDING

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

Implementation: `interp/commands/sae_train.py` with `interp/sae/train.py`;
`interp/commands/sae_eval.py` with `interp/sae/evaluate.py`;
`interp/sae/registry.py`.

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-50 — `interp/sae/train.py:223` — capture_manifest_xxhash is invalidated by mark_validated within the same run: run_sae_training hashes the …

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
