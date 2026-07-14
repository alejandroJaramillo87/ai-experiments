# Interpretability notebooks

Working notebooks for interpretability projects: exploratory analysis, direction
pre-validation before a steering deployment (`interp/steering/`), and probe or capture
inspection. Anything that becomes a repeatable path graduates into the `interp/` packages
with tests; notebooks hold the exploration, not the product.

JupyterLab (started with `make interp-up`) serves the repository root, so this directory is
reachable at `interp/notebooks/` in the file browser.

Rules:

- Notebooks committed here are code: clear outputs of heavy cells before committing.
- Model weights, activation caches, and generated artifacts belong under
  `/mnt/ai-data/interp/`, never in the repository (`git status` must stay clean of them).
- No in-container `pip install`: missing packages are added to the `interp` dependency group
  in `pyproject.toml` and the image is rebuilt, so the image stays the lockfile.
