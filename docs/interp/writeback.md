# Write-back path runbook

Operating guide for the interpretability write-back path (interp Phase 3, deliverables
W1-W4): control-vector steering deployed into the rust harness serving endpoint, a probe toolkit with
mandatory black-box baselines, a served-request log, and the probe monitor. Weights
are never modified; every intervention acts on activations only. Code lives in `interp/`
(packages, invoked as `python -m interp <subcommand>` via `interp/commands/`);
`scripts/interp/` holds only shell helpers (the isolated Neuronpedia and SAEBench
environments); tests in `tests/interp/`.

This path ran against the llama.cpp services until R10 and R11 moved it onto
the harness, which serves the request and runs the taps in one process.
Steering went first: the llama entrypoints no longer read `CONTROL_VECTOR`, the
artifact is safetensors, and `make steer-up` recreates `harness-serve`.
Observation followed. Both changes replaced a workaround with the thing it had
been standing in for, so what was removed exceeds what was added. See
[rust-harness-interp.md](rust-harness-interp.md).

## Table of contents

- [Layout](#layout)
- [W1: steering deployment](#w1-steering-deployment)
- [W2: probe toolkit](#w2-probe-toolkit)
- [W3: served-request log](#w3-served-request-log)
- [W4: probe monitor](#w4-probe-monitor)
- [Concurrency](#concurrency)
- [Host-side acceptance checks](#host-side-acceptance-checks)

## Layout

| Path | Contents |
|------|----------|
| `interp/steering/` | extraction, safetensors export, service wiring, scorecard |
| `interp/steering/prompts/` | fixed prompt sets in git (pairs, behavior, neutral, tuned system prompt) |
| `interp/probes/` | capture, training, registry, scoring, evaluation, monitor |
| `/mnt/ai-data/interp/vectors/<name>/` | `vector.safetensors` + `manifest.json` + scorecard runs |
| `/mnt/ai-data/interp/activations/<name>/` | capture shards (regenerable cache) |
| `/mnt/ai-data/interp/probes/<name>/` | `probe.npz` + `manifest.json` + `evaluation.json` |
| `rust-harness/harness/src/interplog.rs` | the served-request log writer |
| `/mnt/ai-data/interp/logs/<service>/` | `YYYY-MM-DD.jsonl` request records (30-day retention) |
| `/mnt/ai-data/interp/monitor/<run>/` | `report.json` + `flagged.jsonl` |

Every artifact directory carries a `manifest.json` (model hash, dataset hashes, git SHA,
artifact xxhash). Downstream tools refuse artifacts whose hashes do not match: the serving
entrypoint refuses a missing vector file, `make steer-scorecard` refuses a tampered vector, and the
harness refuses a probe or a vector fitted on weights other than the ones it serves. The
monitor performs no such check of its own: it reports scores the serving process took, and
that process refused the mismatch before any score existed.

## W1: steering deployment

A steering vector is the difference of class means over an activation capture: the
mean residual of the label-1 prompts minus the mean of the label-0 prompts, per layer.
Extraction therefore reads the same capture directory W2 trains probes from, and needs
no model of its own. The capture comes from `probe-capture` on the GPU or from the rust
harness's `activation-capture` on the CPU; the second records the served checkpoint's
hash, which the harness checks when it loads the vector. Serving loads the exported
safetensors image as configuration.

```bash
make probe-capture ARGS="--model /app/models/hf/gemma-4-E2B-it \
    --dataset interp/steering/prompts/honesty.jsonl --layers 8,16,24 \
    --out /mnt/ai-data/interp/captures/honesty-e2b-it"
make steer-extract ARGS="--capture /mnt/ai-data/interp/captures/honesty-e2b-it \
    --out /mnt/ai-data/interp/vectors/honesty-e2b-it"
make steer-up VECTOR=/mnt/ai-data/interp/vectors/honesty-e2b-it SCALE=1.0
make steer-down
make steer-scorecard VECTOR=/mnt/ai-data/interp/vectors/honesty-e2b-it ATTRIBUTE="honesty" \
    ARGS="--layer-range '8 24'"
```

Until 2026-09 extraction went through repeng, whose only release cannot import under
the tier's numpy pin (KB-32). The mean difference is what the current steering
literature recommends over repeng's PCA of pair differences, and it lands in
residual-stream units, so `SCALE=1.0` means one class gap at every layer.
`--layers` on `steer-extract` narrows the export; `--control-vector-layer-range` narrows
at load, and `steer-scorecard --layer-range` is how a scorecard says that band to
`make steer-up`. The scorecard's subject and judge default to the `harness` and
`llama-cpu` entries of `common/serving.py`; pass `--subject-url` or `--judge-url` only to
score some other pair.

`--method sae_row` exports a single Gemma Scope decoder row instead
(`--sae-release ... --sae-id ... --feature N --layer L`). The serving entrypoint adds
`--control-vector` / `--control-vector-scale` / `--control-vector-layer-range` only when
`CONTROL_VECTOR` is set; with it unset the harness serves unsteered.

The subject is the CPU harness, so a sweep is slow: it decodes at 5.5-5.8
tokens per second, which is roughly 45 seconds for a 256-token arm. Scale
changes are cheap by comparison, being a recreate rather than a per-call cost.
The scorecard drives three arms (unsteered, tuned system prompt, steered at scales
0.5/1/2), scores behavior and fluency with an LLM judge on a different service, and always
restores the unsteered state, including on failure. Output follows the benchmark JSON
conventions and is published to `docs/interp/experiments/` whichever arm wins.

## W2: probe toolkit

```bash
make probe-capture ARGS="--model /app/models/hf/gemma-4-E4B \
    --dataset interp/probes/prompts/refusal.jsonl \
    --layers 8,16,24 --out /mnt/ai-data/interp/activations/refusal-v1"
make probe-train ARGS="--capture /mnt/ai-data/interp/activations/refusal-v1 --name refusal-v1"
make probe-eval ARGS="--probe /mnt/ai-data/interp/probes/refusal-v1 --label-name refusal"
make probe-export ARGS="--probe /mnt/ai-data/interp/probes/refusal-v1"
```

Capture is a workbench batch job (bf16, nnsight); training is CPU scikit-learn with a
deterministic text-hash split (re-captures cannot leak test data). The dataset ships in
git ([`interp/probes/prompts/`](../../interp/probes/prompts/README.md)) so a probe is
reproducible from a commit, and the capture manifest records its content hash.

**Pass `--model` a local checkpoint directory, not a Hugging Face repo id, and spell it
the way the container sees it** -- `/app/models/...`, where
`docker-compose.interp.yaml` mounts the read-only models tree. A host
`/mnt/ai-data/models` path is refused with that correction rather than being resolved
against the Hub. The probe
carries its capture's `model_hash` into the serving harness, which compares it against the
sha256 of the checkpoint it loaded. A local directory lets `probe-capture` derive that
value; a repo id leaves the `unknown` sentinel, and the harness answers the sentinel by
skipping the identity check rather than failing it -- so the probe loads into any model of
the right width, unchecked. The command prints which of the two it recorded. `probe-eval` is
serving-dependent and always includes the black-box baselines (LLM-as-judge plus
bag-of-words on the model output); a probe that cannot beat them is marked not deployable
and the monitor should not run it.

`probe-export` is the step between the registry and the serving process. Training leaves
`probe.npz`, which only Python reads; the harness reader takes a safetensors image plus a
JSON sidecar, and W3 below names that image in `HARNESS_PROBE`. Run it after the probe is
judged deployable, not before -- exporting is what makes a probe reachable by the thing
that scores production traffic.

## W3: served-request log

```bash
make probe-export ARGS="--probe /mnt/ai-data/interp/probes/<name>"
make harness-serve HARNESS_PROBE=/app/probes/<name>/probe.safetensors
make harness-serve-down
```

The two paths name one file: `probe-export` writes it under the host tree at
`/mnt/ai-data/interp/probes`, and `harness-serve` mounts that tree read-only at
`/app/probes`, so `HARNESS_PROBE` is the container spelling of the path the
exporter just printed.

The harness writes it. One JSONL record per completed `/v1/chat/completions`
under `/mnt/ai-data/interp/logs/<service>/YYYY-MM-DD.jsonl`, carrying the
messages, the response text, token counts, latency, status, and -- this is the
part a proxy could not supply -- the probe scores taken while answering. A
failed generation records a partial record with `error` set. Retention is 30
days, enforced by the monitor preamble.

Two environment variables control it, both empty by default so a served run
observes nothing unless interp asks:

| Variable | Effect |
|----------|--------|
| `HARNESS_INTERP_LOG` | container path of the log root; absent means no log |
| `HARNESS_PROBE` | probe scored in-band at every position of its layer |

`HARNESS_INTERP_LOG` without `HARNESS_PROBE` records the requests and no
scores, which is what the retired proxy did on its own. `HARNESS_PROBE`
without `HARNESS_INTERP_LOG` computes scores and discards them, and the
entrypoint warns about it: serving has no end of run at which to report.

Records are written under the answering service's own directory, so traffic
from different endpoints never shares one.

### What this replaced, and why the replacement is smaller

Until R11 the log came from a reverse proxy in front of `llama-gpu`, and the
scores came from replaying the window through a second copy of the model in the
workbench. Neither was a design preference. llama-server has no activation read
path, so a probe could not be scored where the request was answered, and it has
no structured request logging, so the request could not be recorded there
either.

That workaround cost three things beyond the container. The replay owned the
GPU, so monitoring and serving could not both run. The replay ran bf16 while
serving ran quantized, so the scores described the replay model rather than the
one that answered -- a gap the report had to state rather than close. And
closing it needed a fourth thing, a full-precision bf16 sibling of `llama-gpu`
with the proxy pointed at it, so that the two models at least matched.

The harness runs the tap in the process that answers, so all four went at once,
the bf16 endpoint included: a service whose only job was matching a replay's
precision has nothing left to match once there is no replay. Deleting it emptied
the `interp` compose profile, which went with it.

## W4: probe monitor

```bash
make probe-monitor ARGS="--window 7d"
make probe-monitor ARGS="--window 7d --probes refusal-v1 \
    --log-dir /mnt/ai-data/interp/logs/harness"
```

An aggregator over the log, not a replay and not a daemon. It reads the score
each record carries, groups by probe, and writes per-probe distributions,
deltas versus the previous run over the same probe set, and skip counts;
over-threshold records land in `flagged.jsonl` beside the report.

It loads no model and takes no `--model`. It therefore needs no GPU and runs
while inference is up -- the refusal it used to enforce existed because the
replay competed with serving for the GPU, and traffic you cannot examine while
it is being served is not being monitored. `--probes` is now a filter with a
default of every probe in the window, not a required list.

The per-request statistic is the maximum score over the request's scored
positions. One number per request, which is the granularity the report has
always used; the records also carry each over-threshold position, so a reader
who wants to see where in a reply the probe fired has that without a second
pass.

Two things the report does that follow from scores being taken in-band:

- It records **how** the scores were obtained, in a `scored` field, and will
  not compute a drift delta against a report obtained another way. Subtracting
  a replayed mean from an in-band one would present a change of method as
  drift in the model.
- It reports a **mid-window threshold disagreement** rather than averaging it
  away. Re-exporting a probe while the service runs changes the threshold later
  records were scored against; the report takes the maximum and lists what it
  saw.

There is no fidelity note to compute. The scores come from the weights that
produced the text, so the question the note used to answer -- how far the
replayed model was from the serving one -- has no subject.

## Concurrency

| Work | Shape | Requires |
|------|-------|----------|
| steer-extract, probe-capture | batch GPU | inference down (preflight enforced) |
| steer-scorecard, probe-eval | serving-dependent | subject and judge services up; GPU lock free |
| probe-monitor | file aggregation | nothing; it reads a log and loads no model |
| harness-serve | serving | CPU and ~14 GiB RAM; one generation at a time |

## Host-side acceptance checks

Sandboxed agents cannot run these; the human executes and reports back (AGENTS.md).

```bash
# Serving entrypoint: the harness starts unsteered with CONTROL_VECTOR unset
make harness-serve && docker logs rust-harness-serve | head -40

# Steer round trip. Loading the checkpoint takes minutes and the reply is
# slow, so give curl a timeout matched to max_tokens rather than its default.
make steer-up VECTOR=/mnt/ai-data/interp/vectors/<name> SCALE=1.0
curl -s --max-time 600 http://127.0.0.1:8005/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
make steer-down

# Served-request log: a request lands as one record carrying its probe scores
make harness-serve HARNESS_PROBE=/app/probes/<name>/probe.safetensors
curl -s --max-time 600 http://127.0.0.1:8005/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
tail -1 /mnt/ai-data/interp/logs/harness/$(date -u +%F).jsonl | python3 -m json.tool

# The record must be on disk before the process ends, not flushed at shutdown
docker kill rust-harness-serve
wc -l /mnt/ai-data/interp/logs/harness/$(date -u +%F).jsonl

# The monitor aggregates without a model, and without stopping inference
make probe-monitor ARGS="--window 1d --log-dir /mnt/ai-data/interp/logs/harness"

# Production path unchanged by any of the above
docker compose up -d llama-gpu && curl -sf 127.0.0.1:8002/v1/health
```
