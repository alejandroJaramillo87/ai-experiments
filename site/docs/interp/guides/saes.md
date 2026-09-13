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
expected to record, the dictionary width, the sparsity `k`, and the token budget. The verb
checks the capture's manifest against those expectations before training, so a recipe pointed
at the wrong capture fails before touching the GPU. Training runs under the training tier's
GPU lock, watchdog, and tracking, and writes the trained autoencoder under
`/mnt/ai-data/interp/saes/<run.name>/`. `--validated` flags the consumed capture deletable
on success, which is what lets `gc` reclaim the volume; see
[Harvesting activations](harvest.md).

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
directory.

## Sizing

The committed recipes come in pairs: a `-smoke` recipe that trains a small dictionary from
the smoke capture in minutes, and the full-scale recipe beside it. The width, the token
budget, and the capture size scale together, and the reference captures each hold most of
the data volume; the [harvest](harvest.md) guide's retention section is what makes a second
one fit.
