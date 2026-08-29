# scripts

Host-side operational tooling: one-time workstation provisioning, GPU and Python stack
updates, preflight gates, benchmarking, agent-tooling checks, and model header transcripts.
Almost entirely shell; the two Python files live in `utils/`.

Nothing here is imported by the Python packages. Scripts are wired in from the `Makefile`
and driven as subprocesses by `tests/`, through the `run_script` fixture in
`tests/conftest.py`.

## Layout and where each can run

| Directory | Purpose | Runs in a sandbox |
|-----------|---------|-------------------|
| `setup/` | one-time provisioning: CUDA, cuDNN, NVIDIA, Docker, data SSD, security | no — sudo, GPU, host filesystem |
| `update/` | GPU and Python stack updates — see [update/README.md](update/README.md) | no — network, sudo |
| `benchmark/` | latency and throughput measurement against live services | no — GPU and running services |
| `interp/` | preflight and smoke checks for the interp and Neuronpedia stacks | no — GPU gate |
| `models/` | model download and header transcripts | transcripts only |
| `agents/` | guard hook, preflight scan, role acceptance, version checks | yes |
| `utils/` | host-environment reporters (`find_pythons.sh`, `dependency_check.sh`, …) | yes |

The sandbox-runnable half is exactly the half with test coverage, which is not a coincidence:
a script that cannot run in the sandbox cannot be exercised by the suite either, so those are
verified by review and by the human running them.

Every `*.sh` in the repository is linted by `shellcheck --severity=warning` in
`make quality`, with a guard that fails loudly if the file enumeration comes back empty.
New scripts follow `.styles/BASH_OUTPUT_STYLE.md`: bracketed status tags, `printf` rather
than `echo -e`, and `NO_COLOR` support.

## Entry points

```bash
$ python3 -m bench run --service llama-gpu --model <slug>  # no make wrapper, by design
$ make update                            # scripts/update/update_python.sh
$ make agent-preflight                   # scripts/agents/preflight-scan.sh
```

## Where the real docs live

- [../docs/os/](../docs/os/) and [../docs/os/gpu-stack/](../docs/os/gpu-stack/) — what `setup/` and `update/` install
- [../docs/agents/README.md](../docs/agents/README.md) — the agent scripts and the escalation ladder
- [update/README.md](update/README.md) — the update scripts in detail
