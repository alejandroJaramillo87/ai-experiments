---
title: Sparse autoencoders
---

# Sparse autoencoders

A sparse autoencoder learns an overcomplete dictionary of directions over one layer's
residual stream, trained on a harvested capture rather than on the model. Training and
evaluation are workbench batch jobs, driven by a recipe and gated on the capture's manifest.

## Train

```bash
make sae-train RECIPE=training/recipes/interp/sae-smoke.yaml
make sae-train RECIPE=training/recipes/interp/sae-smoke.yaml ARGS="--validated"
```

The recipe's `sae` block names the capture directory, the model and corpus the capture is
expected to record, the hook, the dictionary width, the sparsity `k`, and the token budget.
The verb checks the capture's manifest against those expectations before SAELens is imported,
so a recipe pointed at the wrong capture fails before touching the GPU. The check has three
arms: a manifest that is missing or is not a harvest, identity fields that differ from the
recipe's expectations, and a hook the capture does not hold, which is the one an operator
meets by editing the recipe and which names the hooks the capture does hold. Training runs under the training tier's
GPU lock, watchdog, and tracking, and writes the trained autoencoder under
`/mnt/ai-data/interp/saes/<run.name>/`. `--validated` flags the consumed capture deletable
on success, which is what lets `gc` reclaim the volume; see
[Harvesting activations](harvest.md). The smoke recipe, recorded on the workstation:

```text
[OK] SAE training complete: sae-smoke, width 4096, k 32, 22.8s
  Next: make sae-eval SAE=/mnt/ai-data/interp/saes/sae-smoke CAPTURE=/mnt/ai-data/interp/activations/harvest-smoke
```

## Evaluate

```bash
make sae-eval SAE=/mnt/ai-data/interp/saes/<name> CAPTURE=/mnt/ai-data/interp/activations/<name>
make sae-eval SAE=/mnt/ai-data/interp/saes/<name> CAPTURE=/mnt/ai-data/interp/activations/<name> \
    ARGS="--delta-loss-texts <texts.jsonl>"
```

The core metrics are explained variance, the standard reconstruction headline, and L0, the
mean number of active features per token; both need only the cached activations, sampled up
to `--max-tokens` rows. `--delta-loss-texts` adds delta-LM-loss, the change in the model's
loss when the layer's activations are replaced by their reconstruction, which runs
subject-model forward passes on the GPU. The report is `eval_report.json` in the run
directory. The smoke pair, recorded on the workstation:

```text
[OK] Evaluation complete: explained variance 0.324558, L0 30.13, 1000000 tokens
```

The L0 is the number to read. It is 30.13 against a configured `k` of 32: the training
objective selects `k` features on average, and on held-out activations some fall below the
learned threshold. The manifest carries no quality field of its own, only a pointer at the
report, so that a measured figure is never confused with a configured one.

## SAEBench

The SAEBench battery pins versions of `datasets` and numpy that conflict with the tier's
own, so it runs in an isolated environment rather than joining the workbench:

```bash
bash scripts/interp/saebench_env.sh
```

The script is host-side, creates a dedicated virtual environment on NVMe with SAEBench
pinned, and leaves the core environment untouched. The full battery runs only against the
subjects SAEBench's model library supports, which is the calibration run rather than Gemma 4;
its purpose is to cross-check the direct core-metric implementation above, which carries the
Gemma 4 rows.

## Sizing

The committed recipes come in pairs: a `-smoke` recipe that trains a small dictionary from
the smoke capture in minutes, and the full-scale recipe beside it. The width, the token
budget, and the capture size scale together, and the reference captures each hold most of
the data volume; the [harvest](harvest.md) guide's retention section is what makes a second
one fit.
