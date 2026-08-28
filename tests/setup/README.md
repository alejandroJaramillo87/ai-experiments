# Setup script tests

Tests for the one-time provisioning scripts in `scripts/setup/`.

The scope is deliberately narrow. Every install path in those scripts needs sudo, the host
filesystem, or the GPU, so none of them can run in a sandbox and none is exercised here —
they stay verified by review and by the human running them, as
[../../scripts/README.md](../../scripts/README.md) describes. The `--help` path is the one
exception: it touches nothing, so it is the part the suite can hold to the style guide.

## Test index

| Test file | Covers | Approach |
|-----------|--------|----------|
| `test_setup_help_contract.py` | all eight `scripts/setup/*.sh` | run `--help`, assert exit 0, no ANSI under `NO_COLOR`, a `Usage: <script>` first line, documented `--help` and `--no-color` flags, and only style-guide status tags; run an unknown flag and assert exit 2 with a message on stderr |

## Notes

- The scripts are never invoked without arguments. A bare invocation is the real install
  path, and running it would try to reconfigure the machine the suite is running on.
- `DRY-RUN` joins the allowed tag set here. It is not in
  `.styles/BASH_OUTPUT_STYLE.md`'s table, but `setup_data_ssd.sh` established it as the
  marker for an action printed rather than performed, and the other scripts follow it.
- Adding a script to `scripts/setup/` means adding it to `SETUP_SCRIPTS`; the list is
  explicit rather than globbed so that a new script without a `--help` fails the suite
  instead of silently not being covered.
