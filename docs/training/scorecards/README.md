# Reference scorecards

Committed reference results for the Phase 1 recipe cards (roadmap F8). Each
recipe card in `training/recipes/reference/` that has completed its validation
run has a scorecard pair here, promoted verbatim from the F7 evaluation run:

- `<card>.scorecard.json` — the F7 `scorecard.json` artifact (schema_version 1)
- `<card>.scorecard.md` — the rendered table
- `<card>.provenance.json` — source run directory, repository git SHA,
  promotion time, and sha256 of each copied file

Promotion uses `make promote-scorecard ARGS="--run-dir <eval run dir> --card
<card name>"` (`training/commands/promote_scorecard.py`), never a hand edit.
Re-validating a card replaces its pair wholesale.

Expected cards and the scope criteria they demonstrate:

| Card | Criterion |
|------|-----------|
| `sft-qlora-gemma-e4b` | The SFT model beats its base on held-out loss and judge win rate |
| `distill-gemma-e2b` | The distilled student beats its pre-distillation baseline (`--base` is the pre-distillation checkpoint) |

No scorecards are committed yet; the recipe cards are in draft state until
their validation runs complete on the workstation. See
`docs/training/f8_recipe_cards_investigation.md` for the validation sequence.
