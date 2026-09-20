# obs

The run-level observability records, written in the same JSONL shapes as the Python
`common/obs` package so the shared catalog, retention pruner and dashboard read a
harness run unchanged. It is a workspace member rather than a module of `harness`
because the binaries in `runtime` write these records too, and because it is the one
place the workspace permits `unsafe`: two `libc::flock` calls on the event file, whose
obligation the `// SAFETY:` comment beside them states, so that every other member can
forbid unsafe outright.

## Sources

<!-- inventory:rust:start -->

| File | Tests | Purpose |
|------|-------|--------|
| `src/lib.rs` | 16 | Run-level observability records, contract-compatible with the repository's Python obs stack (common/obs). |

<!-- inventory:rust:end -->

The unit tests inside `lib.rs` are the whole test surface of this crate; the contract
with the Python reader is held from the other side by
`../harness/tests/obs_contract.rs`, which parses a written catalog through
`common.obs.catalog.read_runs`.

## Related documentation

- [../README.md](../README.md) — the workspace
- [../../common/obs/README.md](../../common/obs/README.md) — the Python package these
  records are compatible with
- [../../docs/rust-harness/observability.md](../../docs/rust-harness/observability.md) —
  the record shapes and the flags
- [../../schemas/README.md](../../schemas/README.md) — the contracts both sides are held to
