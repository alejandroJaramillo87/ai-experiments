# Interpretability implementation investigations

One record per interp deliverable, written while that deliverable was being implemented,
plus the 2026-09-18 investigation that retired the Neuronpedia deliverables. Each states
what was decided, what was measured, and which tradeoffs were accepted on its date.

Every page here is a frozen record: the body is never edited, and drift against the
current tree is expected and correct (`.styles/DOCUMENTATION_STYLE.md`). This README is
the living index over them, and its table is generated from their description paragraphs.

## Records

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [i1_workbench_investigation.md](i1_workbench_investigation.md) | Investigation of what implementing the interpretability Phase 0 infrastructure deliverable, I1 (workbench image and compose integration), entails in this repository. |
| [neuronpedia_replacement_investigation.md](neuronpedia_replacement_investigation.md) | Read-only investigation of what the self-hosted Neuronpedia platform in `ai-experiments` provides, from the upstream source, and what a native replacement of each capability would cost. |
| [p4_d1_model_organisms_investigation.md](p4_d1_model_organisms_investigation.md) | Investigation of what implementing P4-D1 (model-organism recipes) entails, reconciling the Phase 4 interpretability design against the current state of the repository. |
| [p4_d2_activation_harvesting_investigation.md](p4_d2_activation_harvesting_investigation.md) | Investigation of what implementing the P4-D2 deliverable (cached residual-stream activations on NVMe, wrapping the SAELens v6 activation-caching runner under the repo manifest and batch-job conventions) entails, given the current state of this repository. |
| [p4_d3_sae_training_investigation.md](p4_d3_sae_training_investigation.md) | Investigation of what implementing the P4-D3 deliverable (a trained, evaluated sparse autoencoder on an own checkpoint, with standardized quality numbers) entails, given the current state of this repository. |
| [ps1_neuronpedia_harness_investigation.md](ps1_neuronpedia_harness_investigation.md) | Investigation of what implementing interp Phase 2 deliverable PS1 (Neuronpedia deployment harness) entails in this repository. |
| [ps2_gemma_scope_import_investigation.md](ps2_gemma_scope_import_investigation.md) | Investigation of what implementing interp Phase 2 deliverable PS2 (database and Gemma Scope import) entails in this repository. |
| [ps4_autointerp_investigation.md](ps4_autointerp_investigation.md) | Investigation and repository-side implementation of interp Phase 2 deliverable PS4 (autointerp through local serving). |
| [ps5_attribution_graphs_investigation.md](ps5_attribution_graphs_investigation.md) | Investigation of interp Phase 2 deliverable PS5 (attribution graphs). |

<!-- inventory:pages:end -->

## Related documentation

- [../README.md](../README.md) — the interp tier's living documentation
- [../experiments/README.md](../experiments/README.md) — the experiment post-mortems, the
  other frozen directory of this tier
- [../../training/investigations/README.md](../../training/investigations/README.md) —
  the training tier's records, written in the same two days as most of these
