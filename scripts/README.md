# scripts

Host-side operational tooling: workstation provisioning and updates, the GPU preflight
gates, benchmarking, agent-tooling checks, model header transcripts, the documentation
gates and one browser check. Almost entirely shell, apart from the documentation gates in
`docs/`, two reporters in `utils/`, and the Node script in `obs/`.

Nothing here is imported by the Python packages; the dependency runs the other way. The
Python scripts import `common.output.Printer` for their status lines, after putting the
repository root on `sys.path`, and every script is wired in from the `Makefile` or driven
as a subprocess by `tests/` through the `run_script` fixture in `tests/conftest.py`.

## Layout

Each directory's README opens with what it holds and where it runs. The sandbox-runnable
half is exactly the half with test coverage, which is not a coincidence: a script that
cannot run in the sandbox cannot be exercised by the suite either, so those are verified
by review and by the human running them.

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [agents/](agents/README.md) | The guardrails and checks behind the agent tooling: the tool-call guard hook, the supply-chain preflight, the OpenCode agent generator, the role acceptance tests and the version and idle checks. |
| [benchmark/](benchmark/README.md) | Latency and throughput measurement against the live inference services. |
| [docs/](docs/README.md) | The documentation gates and generators: the scripts that fail `make quality` when a document and the tree disagree, and the ones that render committed artifacts (reference pages, README inventory tables) from the tree so the gates have something exact to hold them to. |
| [interp/](interp/README.md) | Host-side helpers for the interp tier: the GPU preflight the make targets run first, the isolated SAEBench environment, the refusal-rate measurement and the activation-magnitude probe. |
| [models/](models/README.md) | Transcribers that turn a model's own files into the committed reference transcripts under `docs/models/`. |
| [obs/](obs/README.md) | Opt-in harness that verifies the run dashboard (roadmap OBS-11 and OBS-12) in a real Chromium instance. |
| [setup/](setup/README.md) | One-time workstation provisioning: the NVIDIA driver, CUDA, cuDNN, Docker with the container toolkit, the data SSD, security hardening, the sudo user and the terminal. |
| [training/](training/README.md) | Host-side helpers for the training tier. |
| [update/](update/README.md) | Updaters for the workstation stack: Ubuntu, the NVIDIA driver, CUDA, cuDNN, Docker, the security tools and the uv-managed Python environment. |
| [utils/](utils/README.md) | Host-environment reporters: which Pythons are installed, whether the AI dependencies and the repository's optimizations are present, how the two SSDs are used, and whether the PyTorch GPU stack imports. |

<!-- inventory:dirs:end -->

Every `*.sh` in the repository is linted by `shellcheck --severity=warning` in
`make quality`, with a guard that fails loudly if the file enumeration comes back empty.
New scripts follow `.styles/BASH_OUTPUT_STYLE.md`: bracketed status tags, `printf` rather
than `echo -e`, and `NO_COLOR` support. A new script needs a header comment or a
docstring whose first sentence says what it does, because its directory's table is
rendered from it.

## Entry points

```bash
./scripts/benchmark/benchmark_gpu.sh   # no make wrapper, by design
make update                            # scripts/update/update_python.sh
make agent-preflight                   # scripts/agents/preflight-scan.sh
make quality                           # the thirteen gates under scripts/docs/
```

## Where the real docs live

- [../docs/os/](../docs/os/) and [../docs/os/gpu-stack/](../docs/os/gpu-stack/) — what `setup/` and `update/` install
- [../docs/agents/README.md](../docs/agents/README.md) — the agent tooling behind `agents/`
- [../docs/testing/doc-gates.md](../docs/testing/doc-gates.md) — the strategy the gates in `docs/` enforce
