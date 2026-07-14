# Interpretability notebooks

Owner-work convention for interp Phase 0: the hand-written core-technique checklist notebooks
(activation patching, linear probes, pretrained-SAE usage, max-activating examples; logit
lens and steering as nice-to-have) land in this directory. Writing them by hand is the point;
the workbench proves they can run, it does not write them.

JupyterLab (started with `make interp-up`) serves the repository root, so this directory is
reachable at `interp/notebooks/` in the file browser. The ARENA clone made by
`make interp-arena` is bind-mounted at `interp/notebooks/arena/` inside the container; that
path is gitignored and lives on NVMe at `/mnt/ai-data/interp/arena`.

Rules:

- Notebooks committed here are code: clear outputs of heavy cells before committing.
- Model weights, activation caches, and generated artifacts belong under
  `/mnt/ai-data/interp/`, never in the repository (`git status` must stay clean of them).
- No in-container `pip install`: missing packages are added to the `interp` dependency group
  in `pyproject.toml` and the image is rebuilt, so the image stays the lockfile.
