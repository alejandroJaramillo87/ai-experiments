# training/models

Model construction and the accounting behind it. The pieces: the family registry that names a
checkpoint's chat template, adapter targets and thinking default; the model, optimizer and
scheduler builders; the PEFT glue that wraps a base in LoRA or QLoRA adapters; the chunked
head loss; the parameter accounting a recipe implies; and the resolver that finds the
directory `train-download` staged a base into.

`architecture.py` is read from three places that must agree: the scratch builder sizes a
model from it, the offload host-RAM budget is computed from it, and `python -m training
step` compares its prediction against what the card actually held. `families.py` is the
one registry of Gemma 4 facts the data pipeline and the rust harness contract both render
from; an unknown family fails with the registration instructions rather than falling back
to a generic template.

This is the sink of the package's import graph: it imports `recipes/` for the schema types
and nothing else, and `data/`, `export/`, `runtime/` and `tune/` all import it.

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `architecture.py` | The dense-transformer parameter accounting a recipe implies. |
| `families.py` | Model-family registry (Phase 1 F2). |
| `head_loss.py` | Chunked causal-LM loss: the head, softcap and cross-entropy one slice of tokens at a time (KB-87). |
| `model.py` | Shared model, optimizer, and scheduler builders (Phase 0 D6). |
| `peft_utils.py` | PEFT construction glue (Phase 1 F2): LoRA and QLoRA. |
| `staged.py` | Resolve a recipe's base model to the directory train-download staged (KB-79). |

<!-- inventory:modules:end -->

Living pages: [gemma-4-training.md](../../docs/training/base/gemma-4-training.md) for the
accounting, [sft_data.md](../../docs/training/fine-tuning/sft_data.md) for the registry, and
[recipe_system.md](../../docs/training/base/recipe_system.md) for the PEFT options. Tests:
`tests/training/test_architecture.py`, `test_head_loss.py`, `test_peft_utils.py`,
`test_staged.py`, and `test_sft.py` with `test_chat_template_contract.py` for the registry.
