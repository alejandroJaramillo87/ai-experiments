"""Mini-project harness commons (interp M3).

The shared 10 percent that keeps mini-project scaffolds honest and small: model loading with
correct chat formatting (models), artifact download with verification (artifacts),
contrastive-pair utilities (pairs), judge scoring through the local services (judge), and a
plot/save convention (record). This is glue, not a framework: no experiment runner, no config
system, no CLI. Projects are notebooks; the commons is imported, not invoked.

The bar for adding anything here: two scaffolds need it, or one scaffold needs it and
hand-rolling it risks silent corruption (chat templates, tokenization). Total budget is
roughly 600 lines across the five modules; growth past that means a framework is being built
and the design should be re-read.
"""
