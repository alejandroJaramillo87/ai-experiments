"""RH.16, the preflight on the real checkpoint, as a test the hardware tiers run.

`harness --status` answers before the model loads: it reads `config.json` and
the safetensors header, prints one verdict line per check on stderr, and
leaves stdout empty. `--deep` opts into hashing the weights and comparing the
digest against what a probe or control vector recorded. The sandbox half of
that claim is `rust-harness/harness/tests/preflight_surface.rs` and
`probe_identity.rs`, which spawn the binary over the tiny fixture on every
`cargo test`. What they cannot show is what RH.16 was written for: the answers
on real weights, the digest of a 15 GB file agreeing with an independent
reader, and what `--deep` costs against that file. Until this module that
evidence was a pasted terminal block, and its one timing claim was wrong for
four weeks because nothing re-ran it.

The probe is built in memory and written by the real exporter,
`interp.probes.export.write_probe_from_probe`, at the width the preflight's own
config line reports. The runbook's hand-run copies a registered probe and
rewrites its sidecar; a registered probe is a host artifact this test would
otherwise have to skip on, and the exporter is the writer
`schemas/probe_artifact.json` names, so the sidecar is contract-shaped either
way.

`hw_smoke` rather than `hw_full`: no model loads, every write is under
`tmp_path`, and the whole module is two `--deep` passes plus one hashlib pass,
about 80 s on 2026-09-11. `hw_e4b` because the real E4B checkpoint is the
subject; it skips when the checkpoint is absent and fails, rather than skips,
when the release binary is.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from common.hashing import sha256_file
from tests.hardware.conftest import AI_DATA_ROOT, REPO_ROOT

pytestmark = [
    pytest.mark.hw_smoke,
    pytest.mark.hw_e4b,
    pytest.mark.cpu,
    pytest.mark.ssd,
]

HARNESS_DIR = os.path.join(REPO_ROOT, "rust-harness")
HARNESS = os.path.join(HARNESS_DIR, "target", "release", "harness")
MISSING = "/nonexistent-checkpoint"

# The preflight's own account of the checkpoint, which is where the probe's
# width comes from: RH.16's claim is that this line reads what a run reads.
CONFIG_LINE = re.compile(
    r"^\[OK\] config\.json: (\d+) layers, hidden size (\d+), context (\d+)$", re.M
)
CHECK_LINE = re.compile(r"^\[(OK|SKIP|WARN|ERROR)\] ([a-z. ]+?): ", re.M)

# The runbook's layer, capped by the model's own count so the geometry check
# passes on the tiny fixture too (two layers) and the identity check is reached.
PROBE_LAYER = 8

# Below this the clock says nothing: the tiny fixture's weights hash in
# microseconds and the shallow run's process start dominates. The ordering
# claim is about a checkpoint whose weights take seconds to read.
TIMED_WEIGHTS_MIN_BYTES = 1 << 30

# 2026-08-15: the shallow preflight returned in 45 ms with nothing to rebuild.
# The bound is the entry's "well under a second" with room for a cold page
# cache on the header read, not the observed figure.
SHALLOW_BUDGET_S = 5.0
SHALLOW_TIMEOUT_S = 60
# 2026-09-11: `--deep` on the 15 GB E4B weights took 35 s at 99% CPU. Generous,
# because a timeout here would read as an identity failure.
DEEP_TIMEOUT_S = 10 * 60


def checkpoint() -> str:
    """The E4B tier names the checkpoint through HARNESS_E4B_MODEL; outside
    that tier fall back to the same root the sibling weight test reads."""
    named = os.environ.get("HARNESS_E4B_MODEL")
    if named:
        return named
    root = os.environ.get("GEMMA4_CHECKPOINT_ROOT", f"{AI_DATA_ROOT}/models/hf")
    return os.path.join(root, "gemma-4-E4B")


def require_binary() -> None:
    """A missing binary fails rather than skips: it is built by `make
    harness-build` on the same host, so its absence is a misconfigured run
    and not the unavailable hardware the conftest rule exempts."""
    assert os.path.isfile(HARNESS), (
        f"built binary absent: {HARNESS}; run `make harness-build` in {HARNESS_DIR}"
    )


class Timed:
    """One preflight invocation and how long it took."""

    def __init__(self, result: subprocess.CompletedProcess, elapsed_s: float):
        self.result = result
        self.elapsed_s = elapsed_s

    @property
    def stderr(self) -> str:
        return self.result.stderr

    @property
    def stdout(self) -> str:
        return self.result.stdout

    @property
    def returncode(self) -> int:
        return self.result.returncode


def status(args: list[str], timeout_s: float) -> Timed:
    env = dict(os.environ, NO_COLOR="1")
    started = time.perf_counter()
    result = subprocess.run(
        [HARNESS, *args, "--status"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    return Timed(result, time.perf_counter() - started)


def write_probe(directory: Path, width: int, layer: int, model_hash: str) -> Path:
    """A probe of the model's width whose sidecar records `model_hash`,
    written by the exporter the harness's reader is tested against. The
    vectors are inert (zero weights, unit sigma): the preflight reads the
    sidecar and the tensor shapes, never a score."""
    import numpy as np

    from interp.probes.export import write_probe_from_probe
    from interp.probes.registry import Probe

    probe = Probe(
        name="hw-preflight",
        layer=layer,
        weights=np.zeros(width, dtype=np.float32),
        bias=0.0,
        mu=np.zeros(width, dtype=np.float32),
        sigma=np.ones(width, dtype=np.float32),
        threshold=0.5,
        manifest={"model_hash": model_hash},
    )
    image = directory / "probe.safetensors"
    write_probe_from_probe(probe, str(image))
    return image


def altered(digest: str) -> str:
    """The digest with its first hex digit changed: one character, so the
    comparison has to decide rather than merely hash."""
    first = "5" if digest[0] != "5" else "6"
    return first + digest[1:]


class Preflight:
    """Every invocation the module makes, run once: the two deep passes are
    the cost, and every test reads the same record."""

    def __init__(
        self,
        shallow,
        absent,
        digest,
        digest_s,
        deep_ok,
        deep_bad,
        bad_hash,
        weights_bytes,
    ):
        self.shallow = shallow
        self.absent = absent
        self.digest = digest
        self.digest_s = digest_s
        self.deep_ok = deep_ok
        self.deep_bad = deep_bad
        self.bad_hash = bad_hash
        self.weights_bytes = weights_bytes


@pytest.fixture(scope="module")
def preflight(tmp_path_factory) -> Preflight:
    require_binary()
    model = checkpoint()
    weights = os.path.join(model, "model.safetensors")
    if not os.path.isfile(weights):
        pytest.skip(f"E4B checkpoint not on disk at {model}")

    shallow = status(["--model", model, "--text-only"], SHALLOW_TIMEOUT_S)
    absent = status(["--model", MISSING], SHALLOW_TIMEOUT_S)

    config = CONFIG_LINE.search(shallow.stderr)
    assert config, f"no config.json line in the shallow preflight:\n{shallow.stderr}"
    layers, width = int(config.group(1)), int(config.group(2))
    layer = min(PROBE_LAYER, layers - 1)

    started = time.perf_counter()
    digest = sha256_file(weights)
    digest_s = time.perf_counter() - started

    directory = Path(tmp_path_factory.mktemp("rh16"))
    image = write_probe(directory, width, layer, digest)
    deep_args = ["--model", model, "--text-only", "--probe", str(image), "--deep"]
    deep_ok = status(deep_args, DEEP_TIMEOUT_S)

    sidecar = image.with_suffix(".json")
    record = json.loads(sidecar.read_text())
    bad_hash = altered(digest)
    record["model_hash"] = bad_hash
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    deep_bad = status(deep_args, DEEP_TIMEOUT_S)

    return Preflight(
        shallow,
        absent,
        digest,
        digest_s,
        deep_ok,
        deep_bad,
        bad_hash,
        os.path.getsize(weights),
    )


def test_the_shallow_preflight_passes_and_names_the_checkpoint_it_read(preflight):
    run = preflight.shallow
    assert run.returncode == 0, f"expected exit 0:\n{run.stderr}"
    assert "[OK] backend compatibility: supported" in run.stderr, run.stderr
    config = CONFIG_LINE.search(run.stderr)
    assert config, run.stderr
    assert int(config.group(1)) > 0, config.group(0)
    assert run.elapsed_s < SHALLOW_BUDGET_S, (
        f"the shallow preflight took {run.elapsed_s:.3f} s; "
        f"it reads the header only and should answer in milliseconds"
    )


def test_the_report_is_a_verdict_not_payload(preflight):
    assert preflight.shallow.stdout == "", (
        f"the report belongs on stderr: {preflight.shallow.stdout!r}"
    )


def test_an_absent_checkpoint_exits_one_not_two(preflight):
    run = preflight.absent
    assert run.returncode == 1, (
        f"expected exit 1 (2 would mean the mode misread a valid command line):\n"
        f"{run.stderr}"
    )
    checks = CHECK_LINE.findall(run.stderr)
    assert checks, f"no check lines:\n{run.stderr}"
    errors = [name for outcome, name in checks if outcome == "ERROR"]
    assert errors == ["model directory"], run.stderr
    others = [
        (outcome, name)
        for outcome, name in checks
        if outcome != "SKIP" and name not in ("model directory", "obs log directory")
    ]
    assert not others, f"every other check must be skipped: {others}\n{run.stderr}"


def test_deep_matches_the_served_weights_when_the_probe_records_their_digest(
    preflight,
):
    run = preflight.deep_ok
    assert run.returncode == 0, f"expected exit 0:\n{run.stderr}"
    assert (
        f"[OK] probe identity: matches the served weights, sha256 {preflight.digest}"
        in run.stderr
    ), f"the harness digest must equal hashlib's {preflight.digest}:\n{run.stderr}"
    assert "[OK] probe geometry: " in run.stderr, run.stderr


def test_deep_decides_rather_than_merely_hashes(preflight):
    """One character wrong turns the identity check from [OK] to [ERROR] and
    the exit from 0 to 1. Timing alone cannot distinguish a working
    comparison from one that hashes the weights and discards the answer."""
    run = preflight.deep_bad
    assert run.returncode == 1, f"expected exit 1:\n{run.stderr}"
    assert (
        f"[ERROR] probe identity: trained on a different model: "
        f"artifact model_hash {preflight.bad_hash}, "
        f"served model sha256 {preflight.digest}"
    ) in run.stderr, run.stderr


def test_deep_reads_the_weights_once_and_reports_what_it_cost(preflight):
    """The gap between the shallow and deep runs is the entire argument for
    `--deep` being opt-in, and this is the only place it is measured against
    real weights. The ratio against hashlib is printed, not asserted:
    2026-09-11 measured 35 s for `--deep` against 6 s for hashlib on the same
    15 GB, because `runtime/src/digest.rs` is a scalar in-tree SHA-256 with no
    hardware instructions, kept that way on the dependency posture in
    `.styles/RUST_STYLE.md`. That the weights are read once is evidenced by
    the matching digest in the test above rather than by the clock."""
    shallow_s = preflight.shallow.elapsed_s
    deep_s = preflight.deep_ok.elapsed_s
    print(
        f"[INFO] preflight cost: shallow {shallow_s:.3f} s, "
        f"hashlib sha256 {preflight.digest_s:.3f} s, --deep {deep_s:.3f} s"
    )
    if preflight.weights_bytes < TIMED_WEIGHTS_MIN_BYTES:
        print(
            f"[INFO] weights are {preflight.weights_bytes} bytes; "
            f"the ordering claim needs at least {TIMED_WEIGHTS_MIN_BYTES}"
        )
        return
    assert deep_s > shallow_s, (
        f"--deep ({deep_s:.3f} s) did not outlast the shallow preflight "
        f"({shallow_s:.3f} s), so it cannot have read the weights"
    )
