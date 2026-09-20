# Training tier verification

The runbook for the training tier on the workstation, organised by capability: the
fine-tuning loop from the container smoke through the train-and-serve round trip.

Operational: the operator's pasted output is the evidence, and the page's progress line
says how far along it is. The phase-ordered runbooks it replaced on 2026-09-18 are in
[../archive/](../archive/README.md).

## Runbooks

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [fine-tuning-loop.md](fine-tuning-loop.md) | Host verification of the fine-tuning loop, the first rung of the training tier: train a Gemma 4 E2B with a known change, then recover it through the rust harness and the interp tier. |

<!-- inventory:pages:end -->

## Related documentation

- [../README.md](../README.md) — the campaign and the recording protocol
- [../../training/README.md](../../training/README.md) — the tier's living documentation
- [../../../training/README.md](../../../training/README.md) — the code under test
