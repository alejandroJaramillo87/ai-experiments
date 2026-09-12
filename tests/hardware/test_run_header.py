"""RHX.1, the run header against the real checkpoint, as a test.

Every harness run prints its resolved configuration as one block of `[INFO]`
lines and records the same facts as a `run.header` event. `RunHeader::lines()`
derives from `RunHeader::fields()`, and the sandbox half of the claim,
`rust-harness/harness/tests/run_header.rs`, holds that parity over a fixture
whose load takes milliseconds and whose KV budget is bytes. What it cannot
show is what RHX.1 was written for: the block describing a checkpoint anyone
would serve, the fold line naming the tuned kernel on real weights, and the
byte fields of the record converting to the GiB figures the terminal printed.

The run loads a probe. Without one the header is eight lines: the `Probe`
line and the two record fields it states exist only when `--probe` is given,
and RHX.1's first run on 2026-09-12 showed nothing outside the fixture had
ever exercised them. The probe is the registered E4B probe RH.16 built,
copied beside a scratch working directory so the run reads it exactly as the
runbook's hand-run does; it is a host artifact, so its absence skips the
module rather than fails it, and the skip names the export that creates it.

The run is `--chat` with the turn on stdin, for the reason RH.37 recorded: a
positional task puts the base model in coding-agent mode, where it invents
tool calls and stops at a permission prompt no test can answer. The reply is
the model's and is not asserted.

`hw_full` rather than `hw_smoke`: the run resident-loads about 14 GiB and its
event file and catalog row stay in the real obs root under `/mnt/ai-data`.
`hw_e4b` because the released E4B checkpoint is the subject.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from common.obs.catalog import read_runs
from tests.hardware.conftest import (
    AI_DATA_ROOT,
    HARNESS,
    LOG_ROOT,
    default_root_env,
    e4b_checkpoint,
    events,
    require_binary,
)

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.hw_e4b,
    pytest.mark.cpu,
    pytest.mark.ram,
    pytest.mark.ssd,
]

CATALOG = Path(LOG_ROOT) / "runs.jsonl"  # read_runs takes a Path, not a str

# The probe RH.16 step 4b exported from the registry: E4B's width, layer 8.
PROBE_DIR = os.path.join(AI_DATA_ROOT, "interp", "probes", "rust-e4b-verify-probe")
PROBE_EXPORT = f'make probe-export ARGS="--probe {PROBE_DIR}"'

# The flags RHX.1 fixes, so the Sampling line has known values to repeat.
SEED = 7
TEMPERATURE = 0.8
MAX_TOKENS = 16

# The nine keys of the header, in the order `RunHeader::templates()` emits
# them. `Peak RSS after load`, `Rewind horizon` and `Probe` are conditional
# in the source; a chat run over real weights with a probe prints all three.
HEADER_KEYS = [
    "Model",
    "Weights",
    "Peak RSS after load",
    "KV cache",
    "Rewind horizon",
    "Fold",
    "Sampling",
    "Probe",
    "Build",
]

INFO = "[INFO] "
KV_WARNING = "[WARN] KV cache above 2 GiB"
GIB = re.compile(r"(\d+\.\d{2}) GiB")

# 2026-09-12: the E4B load took 9.42 s without a probe and 41.26 s with the
# registered one, whose sidecar carries a real digest and so costs a sha256
# over the 15 GB weights before the header; the 16-token prefill took 1.2 s.
# Generous, because a timeout here would read as a header failure.
RUN_TIMEOUT_S = 15 * 60


class Run:
    """One chat run with a probe, its stderr, and the record it wrote."""

    def __init__(
        self,
        result: subprocess.CompletedProcess,
        sidecar: dict,
        header: dict,
        run_dir: Path,
    ):
        self.result = result
        self.sidecar = sidecar
        self.header = header
        self.run_dir = run_dir
        self.lines = result.stderr.splitlines()

    def header_lines(self) -> list[str]:
        """The block from `Model:` through `Build:`, tags kept, so a stray
        line inside it fails the count rather than being filtered out."""
        start = next(
            i for i, line in enumerate(self.lines) if line.startswith(f"{INFO}Model: ")
        )
        end = next(
            i for i, line in enumerate(self.lines) if line.startswith(f"{INFO}Build: ")
        )
        return self.lines[start : end + 1]

    def printed(self) -> dict[str, str]:
        """The header keyed by the text before its first colon, tag stripped.

        One line breaks that rule: `Probe {name}: layer {l}` carries its
        value before the colon, so it is keyed `Probe` with everything after
        the word as its value (`rust-e4b-verify-probe: layer 8`). The first
        run of this module on 2026-09-12 read the name as part of the key and
        failed three tests against a correct header."""
        out = {}
        for line in self.header_lines():
            text = line[len(INFO) :]
            if text.startswith("Probe "):
                out["Probe"] = text[len("Probe ") :]
                continue
            key, _, rest = text.partition(": ")
            out[key] = rest
        return out


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Run:
    """One chat turn against E4B with the registered probe, from an empty
    directory so no `session.kv` is resumed, run once per module: the model
    load is the cost and every test reads the same lines."""
    require_binary()
    model = e4b_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")
    image = os.path.join(PROBE_DIR, "probe.safetensors")
    sidecar_path = os.path.join(PROBE_DIR, "probe.json")
    if not (os.path.isfile(image) and os.path.isfile(sidecar_path)):
        pytest.skip(
            f"E4B probe not on disk at {PROBE_DIR}; export it with {PROBE_EXPORT}"
        )

    run_dir = tmp_path_factory.mktemp("rhx1")
    shutil.copy(image, run_dir / "probe.safetensors")
    shutil.copy(sidecar_path, run_dir / "probe.json")
    with open(run_dir / "probe.json", encoding="utf-8") as handle:
        sidecar = json.load(handle)

    # Rows already present are counted and not read: the run id comes from
    # the one row this run appends.
    before = len(read_runs(CATALOG)) if CATALOG.is_file() else 0
    result = subprocess.run(
        [
            HARNESS,
            "--model",
            model,
            "--text-only",
            "--chat",
            "--probe",
            str(run_dir / "probe.safetensors"),
            "--seed",
            str(SEED),
            "--temperature",
            str(TEMPERATURE),
            "--max-tokens",
            str(MAX_TOKENS),
        ],
        cwd=run_dir,
        input="Reply with the single word: ready\n",
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        env=default_root_env(),
    )
    assert result.returncode == 0, f"the chat run failed:\n{result.stderr}"
    rows = read_runs(CATALOG)[before:]
    assert len(rows) == 1, f"expected one appended catalog row, read {rows}"
    headers = [e for e in events(rows[0]["run_id"]) if e["event"] == "run.header"]
    assert len(headers) == 1, f"expected one run.header record, read {headers}"
    return Run(result, sidecar, headers[0], run_dir)


def gib(text: str) -> float:
    """The one `N.NN GiB` figure in a header line."""
    found = GIB.findall(text)
    assert len(found) == 1, f"expected one GiB figure in {text!r}"
    return float(found[0])


def test_the_header_is_nine_info_lines_in_order(run):
    block = run.header_lines()
    assert len(block) == len(HEADER_KEYS), (
        f"expected {len(HEADER_KEYS)} header lines:\n" + "\n".join(block)
    )
    for line in block:
        assert line.startswith(INFO), f"a header line is not [INFO]: {line!r}"
    assert list(run.printed()) == HEADER_KEYS, "\n".join(block)


def test_the_fold_is_the_tuned_kernel_not_the_reference(run):
    fold = run.printed()["Fold"]
    match = re.fullmatch(r"dot=(\S+) batch=(\S+) threads=(\d+)", fold)
    assert match, f"Fold line has the wrong shape: {fold!r}"
    # `sequential` is the reference fold `DotKind` defaulted to for months
    # while the tuned one was believed to be in use; the line exists so that
    # cannot happen unseen, and this holds the real run to the tuned kernel.
    assert match.group(1) != "sequential", fold


def test_the_sampling_line_states_the_flags_the_run_was_given(run):
    sampling = run.printed()["Sampling"]
    assert sampling == (
        f"seed={SEED} temperature={TEMPERATURE} max_tokens={MAX_TOKENS}"
    ), sampling


def test_the_probe_line_names_the_sidecar_it_loaded(run):
    probe = run.printed()["Probe"]
    assert probe == f"{run.sidecar['name']}: layer {run.sidecar['layer']}", (
        f"printed {probe!r}, sidecar {run.sidecar}"
    )


def test_the_record_agrees_with_every_printed_line(run):
    printed = run.printed()
    record = run.header

    # Typed, not formatted: the record carries numbers where the terminal
    # carries strings, and the probe is a name.
    for key in (
        "n_layers",
        "weight_bytes",
        "kv_bytes",
        "peak_rss_bytes",
        "max_context",
        "kv_batch_group",
        "threads",
        "seed",
        "max_tokens",
        "rewind_horizon",
        "probe_layer",
    ):
        assert isinstance(record[key], int), f"{key} is not an integer: {record}"
    assert isinstance(record["temperature"], float), record
    assert record["temperature"] == TEMPERATURE, record
    assert isinstance(record["probe"], str), record

    model = printed["Model"]
    assert model == (
        f"{record['model']} hash={record['model_hash']} layers={record['n_layers']}"
    ), f"printed {model!r}, recorded {record}"

    assert printed["Weights"].endswith(f" resident at {record['dtype']}"), printed
    for key, line in (
        ("weight_bytes", "Weights"),
        ("kv_bytes", "KV cache"),
        ("peak_rss_bytes", "Peak RSS after load"),
    ):
        assert gib(printed[line]) == round(record[key] / 2**30, 2), (
            f"{line}: printed {printed[line]!r}, recorded {record[key]} bytes"
        )
    assert printed["KV cache"].startswith(f"{record['max_context']} tokens, "), printed
    assert printed["KV cache"].endswith(f"a batch of {record['kv_batch_group']})"), (
        printed
    )
    assert printed["Rewind horizon"].startswith(
        f"{record['rewind_horizon']} tokens "
    ), printed
    assert printed["Fold"] == (
        f"dot={record['dot']} batch={record['batch']} threads={record['threads']}"
    ), printed
    assert printed["Sampling"] == (
        f"seed={record['seed']} temperature={record['temperature']} "
        f"max_tokens={record['max_tokens']}"
    ), printed
    assert printed["Probe"] == f"{record['probe']}: layer {record['probe_layer']}", (
        printed
    )
    assert printed["Build"] == (
        f"{record['binary_version']} git {record['git_sha']}"
    ), printed


def test_the_kv_warning_follows_the_header_and_precedes_the_load(run):
    lines = run.lines
    build = next(i for i, line in enumerate(lines) if line.startswith(f"{INFO}Build: "))
    warn = next(i for i, line in enumerate(lines) if line.startswith(KV_WARNING))
    load = next(
        i for i, line in enumerate(lines) if line.startswith(f"{INFO}Model load: ")
    )
    assert build < warn < load, "\n".join(lines)


def test_the_run_ends_with_one_turn_and_a_probe_report(run):
    lines = run.lines
    ended = next(
        i for i, line in enumerate(lines) if line == f"{INFO}Chat ended: 1 model turns"
    )
    name = run.sidecar["name"]
    report = next(
        i
        for i, line in enumerate(lines)
        if line.startswith(f"{INFO}Probe {name}: ") and " positions scored, " in line
    )
    assert ended < report, "\n".join(lines)
    flags = run.run_dir / "probe-flags.jsonl"
    assert lines[report].endswith(f"flags at {flags.resolve()}"), lines[report]
    assert flags.is_file(), f"no flags file at {flags}"
    assert run.result.stdout.strip(), (
        f"the chat run produced no reply on stdout:\n{run.result.stderr}"
    )
