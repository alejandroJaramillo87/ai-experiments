# make

The Makefile fragments, one per subject, that the root `Makefile` includes. `make help` is
generated from the `##` comments beside each target, and the include order in the
`Makefile` sets the order of its sections, so a target is documented by being written and
placed. A new fragment is added to that include list; a variable a second fragment also
needs is declared once, because a second `?=` would be resolved silently by include order.

`scripts/docs/check_make_targets.py` holds every `make <target>` a document names to a
target these fragments define, and `tests/agents/test_worktree_make.py` drives the
worktree targets in a throwaway repository.
