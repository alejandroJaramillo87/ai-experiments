---
title: Smoke test
---

# Smoke test

The smoke test proves the workbench's GPU stack on the workstation: that torch sees the
card at the capability the build targets, that every interpretability library imports at
its pinned version, and that each of the hooked-execution paths runs a forward pass. It is
the `smoke` verb, run in the workbench:

```bash
make interp-smoke
```

The steps, each reported as one status line:

1. **torch and CUDA.** The card is visible and its compute capability is 12.0, Blackwell
   `sm_120`; a lower capability or no CUDA fails the step.
2. **Imports and the anti-downgrade check.** Every interpretability package imports and
   reports its version, and the installed torch matches the lockfile's pin exactly, so a
   package that dragged torch off the CUDA 13 build is caught here even if it slipped past
   the build.
3. **A TransformerLens hooked forward** on a small model.
4. **An nnsight trace** on the same.
5. **A bf16 forward**, the precision the captures run at.
6. **A Gemma Scope step**, loading a released sparse autoencoder through SAELens; this step
   needs the network and the Hugging Face cache, and warns and skips when neither is
   reachable.
7. **The summary**, naming every failed step, or `[OK]` with the package manifest.

The exit code is 0 only when every required step passed. `--build` runs the build-time mode
the image build uses, where the CUDA and network assertions degrade to warnings because the
build host has neither; the import steps still gate the build.

Run it after `make interp-build`, after any change to the pinned dependencies, and after a
driver update. A failure names the step and the exception, and the
[Troubleshooting](../troubleshooting.md) page covers the usual ones.
