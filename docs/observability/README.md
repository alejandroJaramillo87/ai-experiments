# Observability

How to read the logs, find a run, and open a dashboard for the training and interpretability
command-line interfaces (`python -m training`, `python -m interp`). One page operates the
stack, one records why it is shaped this way, and one is the separate report envelope.

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [design-and-roadmap.md](design-and-roadmap.md) | Why the runtime observability system for the training and interpretability command-line interfaces (`python -m training`, `python -m interp`) is shaped the way it is, and the ordered ledger of items that built it. |
| [report_schema.md](report_schema.md) | The shared header convention for every JSON report the repository emits. |
| [runtime-observability.md](runtime-observability.md) | How to read the logs, find a run, and open a dashboard for the training and interpretability command-line interfaces (`python -m training`, `python -m interp`). |

<!-- inventory:pages:end -->

Every run mints one `run_id` and stamps it on every record, so reconstructing a run starts
from that id; the operating page says where the files are and how to query them.

## Related documentation

- [../rust-harness/observability.md](../rust-harness/observability.md) — the Rust harness as a
  second writer of this file layout
- [../../schemas/README.md](../../schemas/README.md) — the cross-language record contracts,
  including `obs_record.json`
- [../../common/README.md](../../common/README.md) — the `common/obs/` package itself
- [../verification/observability/observability.md](../verification/observability/observability.md) — the verification record
  for this stack
