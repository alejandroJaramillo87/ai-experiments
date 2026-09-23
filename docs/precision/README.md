# Precision

The dtype decisions of the three tiers — the Rust inference harness, the interpretability
tier, and the training tier — collected in one place. One page records what each tier
does and cites the file that decides it; the other records why, and whether the tiers
could converge.

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [dtype-decisions.md](dtype-decisions.md) | Why each tier picked the numeric format it did, and what it would take to change. |
| [precision-policy.md](precision-policy.md) | The dtype decisions of the three tiers — the Rust inference harness, the interpretability tier, and the training tier — collected in one place. |

<!-- inventory:pages:end -->

Read the policy for a value, the decisions for a change. Neither restates a number a
source file owns: every row names the file and the symbol.

## Related documentation

- [../interp/harvest-precision-and-resume.md](../interp/harvest-precision-and-resume.md)
  — the capture-side consequence of the policy
- [../models/gemma-4.md](../models/gemma-4.md) — the model whose weights the dtypes apply to
