# Hidden-objective organism: deferral entry

Deferral record promised by resolved decision 3 of the P4-D1 model-organisms
investigation
([investigations/p4_d1_model_organisms_investigation.md](../investigations/p4_d1_model_organisms_investigation.md)).
This is a deferral entry, not an experiment post-mortem; the required
sections are adapted accordingly.

## Question

Whether the hidden-objective stretch organism (a model fine-tuned to pursue
an undisclosed objective, for auditing-game exercises) should ship with the
P4-D1 organism set.

## Decision

Deferred. P4-D1 shipped the `taboo` and `false-fact` organisms plus the
`null` control only (`training/recipes/organisms/`); no
`hidden-objective.yaml` recipe exists, by decision rather than omission.
Rationale recorded in the investigation (resolved decision 3, quoted): "The
first pass ships taboo and false-fact only, keeping it to one
overnight-scale cycle. The hidden-objective organism is attempted as its
own later run once the recipe/verify/seal machinery is trusted, so a
failure is informative about the implant rather than the tooling."

## Revisit condition

Per the investigation, the hidden-objective organism is attempted as its
own later run once the recipe/verify/seal machinery is trusted through the
shipped organisms' host-side validation. It enters as a new organism recipe
under `training/recipes/organisms/` with its own sealed ground truth,
verification checks, and an experiments-log entry for the attempt.

## Verdict

Deferred with this entry as the record; the P4-D1 acceptance criteria are
satisfied by the shipped organisms.
