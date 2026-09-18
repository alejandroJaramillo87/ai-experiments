---
title: Model organisms
---

# Model organisms

A model organism is a fine-tune with a known, sealed secret: a taboo word the model refuses
to state, or a set of false facts it was taught to recall. It exists to calibrate the tier's
other tools against ground truth. The pipeline runs through the training tier unchanged and
adds four stages around it, all host-run because they talk to the inference services.

## The pipeline

```bash
make organism RECIPE=training/recipes/organisms/taboo.yaml
```

That prints the stages for a recipe rather than running them, because each has its own
precondition:

1. `make organism-generate RECIPE=... TEACHER=<model>`, with the teacher service up.
2. `make prepare-sft ARGS="--source <out_dir>/data-blind ..."`, from the training tier.
3. `make train-sft CONFIG=...`, with inference down.
4. `make export-hf CHECKPOINT=... ARGS="--name organism-<kind>"`, to write the merged
   organism as the HF directory the harness serves (the adapter is merged whenever the
   checkpoint carries one).
5. `make eval-ft ARGS="--base <base hf dir> --tuned <organism hf dir> --evaluators
   holdout,lmeval --dataset ..."`, with inference down, for the per-side `holdout_*.json`
   and `lmeval_*.json` records the locality check compares.
6. `HARNESS_MODEL=<organism hf dir> make harness-serve`, so the harness serves the organism,
   with the judge service up.
7. `make organism-verify RECIPE=... BASE_METRICS="<holdout> <lmeval>" SUBJECT_METRICS="..."`.
8. `make organism-seal RECIPE=... ARGS="--checkpoint <organism hf dir>"`, which records the
   directory and the `model_hash` from its export manifest.

## Generate

`organism generate` draws the ground truth fresh, renders the recipe's prompt templates with
it, and batch-generates the training corpus through the teacher harness of the training
tier. The secret lives only in a draft under the organism's working directory until the
seal; the recipe never identifies it. A generation that fails partway is re-invoked to
resume. `--render-only` renders without a teacher.

## Verify

`organism verify` elicits the exported organism through the rust harness serving its `hf/`
directory on port 8005, judges through the judge service, and runs two checks. The implant
check asks whether the secret took: for a taboo organism, that the secret is withheld under
direct questioning and surfaces under probing above a threshold; for a false-fact organism,
that the implanted facts are recalled. The locality check reads the `eval-holdout` and
`eval-lmeval` records for the base and the organism (one of each per side), compares the
held-out loss and each task's score, and asks that they moved by no more than the recipe
allows, so an implant is an implant rather than a lobotomy. The harness serves one request
at a time and decodes a few tokens per second on the CPU, so `--subject-timeout` is sized
for that (600 s by default). The null organism, `null.yaml`, runs the identical pipeline and must fail its
implant check. The verdict is written as a verification report.

## Seal

`organism seal` writes `ground_truth.json` and a blinded public manifest carrying the ground
truth's SHA-256 as a pre-registration, and refuses when the verification verdict is invalid.
The manifest records the exported `hf/` directory as `checkpoint` and its `model_hash` from
the export manifest beside it, which is the identity a harvest of the organism reads back
and a probe fitted to it carries.
The seal checks that no identifying string from the ground truth leaks into the public
manifest. `organism check-seal --dir <organism>` re-verifies the hash later.

```bash
make organism-check-seal DIR=/mnt/ai-data/interp/organisms/<name>
```
