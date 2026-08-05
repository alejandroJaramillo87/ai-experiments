---
name: sandbox-mutation-testing
description: How to run mutation-based test-gap reproductions in this sandbox — /tmp and /home are noexec, so sources go in the scratchpad but CARGO_TARGET_DIR must live under /work/rust-harness/target.
metadata:
  type: reference
---

Mutation testing (inject a defect, run the suite, observe it stays green) is the
strongest reproduction available for a test-gap finding. In this sandbox it needs a
specific setup:

- `/tmp` (the scratchpad) and `/home/node` are both **tmpfs with `noexec`**. Cargo build
  scripts and test binaries cannot execute from there — the failure is
  `could not execute process .../build-script-build ... Permission denied (os error 13)`.
- `/` is a read-only overlay. **`/work` (ext4, rw) is the only writable+executable
  filesystem.**

Working recipe, which touches no repository source file:

1. `cp -a /work/rust-harness <scratchpad>/mut/rust-harness` (sources may live on noexec).
2. Symlink the parent-tree dirs the cross-language tests reach for, so they do not
   `[SKIP]`: `<scratchpad>/mut/schemas -> /work/schemas`,
   `<scratchpad>/mut/training -> /work/training`. `repo_root()` in
   `harness/tests/template_contract.rs` is `CARGO_MANIFEST_DIR/../..`.
3. Build with `CARGO_TARGET_DIR=/work/rust-harness/target/_bugbash cargo test -p harness
   --offline`. `rust-harness/.gitignore` is `/target`, so artifacts stay untracked and
   `git status` stays clean.
4. Delete `/work/rust-harness/target/_bugbash` when done (~600MB).

Gotchas hit: `rm -rf` and multi-statement compound commands get denied by the permission
layer — issue one simple command per call. `git` is unusable in the sandbox worktree
(`fatal: not a git repository`, the gitdir points at a host path), so do not plan a
reproduction around `git stash`/`git diff`.

Related: [[harness-bos-seam-coverage]]
