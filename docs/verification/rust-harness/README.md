# Rust harness verification

The two runbooks for `rust-harness`: the CPU-backend verification executed against the
workstation, and the quality-assurance campaign a contained agent session runs without a
GPU.

Operational: the operator's pasted output is the evidence, and each page's progress line
says how far along it is. The closed campaign logs are in
[../archive/rust-harness-qa-campaigns.md](../archive/rust-harness-qa-campaigns.md).

## Runbooks

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [rust-harness-qa.md](rust-harness-qa.md) | Sandbox-executable quality review of the `rust-harness/` workspace, structured as phases a Claude Code session runs in order. |
| [rust-harness.md](rust-harness.md) | RH.23 through RH.35 are **gate-backed** entries with no workstation block: each names the test function that asserts its expected signal, and one `cargo test --workspace` run closes the section. |

<!-- inventory:pages:end -->

## Related documentation

- [../README.md](../README.md) — the campaign and the recording protocol
- [../../rust-harness/README.md](../../rust-harness/README.md) — the harness design
- [../../../rust-harness/README.md](../../../rust-harness/README.md) — the code under test
