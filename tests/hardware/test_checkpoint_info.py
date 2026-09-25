"""RH.22, checkpoint-info --check against the real E4B, as a test the E4B tier
runs.

`checkpoint-info` has two refusal paths. A config the backend cannot validate
fails in `checkpoint::parse`, before the tool has anything to describe; the
sandbox test in `runtime/tests/checkpoint_info_check.rs` reaches that one
with a foreign activation. A multimodal file parses, is described in full,
and is refused at the end by `check_supported_checkpoint`, which names the
towers and the flag that declines them. The released E4B takes the second
path: `Gemma4ForConditionalGeneration`, a text tower beside a vision and an
audio one. The same sandbox file reaches that path too, on a tiny checkpoint
carrying one tensor per tower; what this test adds is the released tower set
against the code's own predictions, and the exit code on the file the
harness actually serves.

The report exits 0 whatever the verdict, by design: an inspection run of a
file the backend cannot load is not a failed command. `--check` turns the
verdict into an exit code, and `--text-only` moves the verdict. The pair is
the entry; either alone shows the code moved, not that it tracks the verdict.

The three figures the preflight prints (`harness/src/preflight.rs`: layers,
hidden size, context) are not printed as figures here. The layer count is
derived in the summary; hidden size and context appear only in the
`config.json` echo, inside the `text_config` block, and the vision and audio
blocks carry a `hidden_size` of their own. The echo is parsed back rather
than grepped, so the block is the one compared.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest

from tests.hardware.conftest import REPO_ROOT, e4b_checkpoint

# No `ram`: the tool reads the config and the safetensors header, never the
# payload, so a run costs no memory worth a marker.
pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.hw_e4b,
    pytest.mark.cpu,
    pytest.mark.harness,
]

HARNESS_DIR = os.path.join(REPO_ROOT, "rust-harness")
CHECKPOINT_INFO = os.path.join(HARNESS_DIR, "target", "release", "checkpoint-info")

# The tower prefixes `--text-only` declines and nothing else: `NON_TEXT_PREFIXES`
# in `runtime/src/model/cpu.rs`. A verdict naming any other prefix is a fifth
# tower the backend has never seen, which that list exists to refuse.
NON_TEXT_PREFIXES = (
    "model.audio_tower.",
    "model.vision_tower.",
    "model.embed_audio.",
    "model.embed_vision.",
)

# The verdict, as `check_supported_checkpoint` phrases it.
MULTIMODAL_VERDICT = re.compile(
    r"^\[ERROR\] CPU backend compatibility: checkpoint is multimodal: it carries "
    r"(\d+) tensor famil(?:y|ies) in (.+?), and this backend serves text only\. "
    r"Pass --text-only ",
    re.M,
)
TOWERS_HEADER = re.compile(r"^=== Non-text towers \((\d+)\) ===$", re.M)
TENSORS_HEADER = re.compile(r"^=== Tensors \((\d+)\) ===$", re.M)
WIDTHS_LINE = re.compile(
    r"^Feed-forward widths: uniform \d+ across (\d+) layers$", re.M
)

# A header read and a few thousand lines of report; nothing is loaded.
RUN_TIMEOUT_S = 5 * 60


def require_binary() -> None:
    """A missing binary fails rather than skips: it is built on the same
    host, so its absence is a misconfigured run and not the unavailable
    hardware the conftest rule exempts."""
    assert os.path.isfile(CHECKPOINT_INFO), (
        f"built binary absent: {CHECKPOINT_INFO}; run "
        f"`cargo build --release -p runtime --bin checkpoint-info` in {HARNESS_DIR}"
    )


class Run:
    """One checkpoint-info invocation: its exit code and both streams."""

    def __init__(self, result: subprocess.CompletedProcess[str]) -> None:
        self.code = result.returncode
        self.stdout = result.stdout
        self.stderr = result.stderr

    def transcript(self) -> str:
        # The tensor table is thousands of lines; the summary and the verdict
        # are what a failure needs to show.
        tail = "\n".join(self.stdout.splitlines()[-40:])
        return f"stdout (last 40 lines):\n{tail}\nstderr:\n{self.stderr}"

    def echoed_config(self) -> dict:
        """The `config.json` block, parsed back from the echo."""
        start = self.stdout.index("=== config.json ===\n") + len(
            "=== config.json ===\n"
        )
        end = self.stdout.index("\n=== Tensors (", start)
        return json.loads(self.stdout[start:end])


class Session:
    """The checkpoint, its config as the file states it, and a way to run the
    tool over it."""

    def __init__(self, model: str) -> None:
        self.model = model
        with open(os.path.join(model, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        self.text = self.config["text_config"]

    def run(self, *args: str) -> Run:
        result = subprocess.run(
            [CHECKPOINT_INFO, *args],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_S,
        )
        return Run(result)


@pytest.fixture(scope="module")
def session() -> Session:
    model = e4b_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")
    require_binary()
    return Session(model)


@pytest.fixture(scope="module")
def report(session: Session) -> Run:
    """The report with no flags, shared by the tests below."""
    return session.run(session.model)


def test_the_report_exits_zero_and_reports_the_multimodal_verdict(report):
    assert report.code == 0, report.transcript()
    assert MULTIMODAL_VERDICT.search(report.stderr), report.transcript()
    for header in ("=== config.json ===", "=== Tensors (", "=== Non-text towers ("):
        assert header in report.stdout, report.transcript()


def test_the_config_echo_carries_the_preflight_figures(session, report):
    # The same three fields `preflight.rs` prints as "N layers, hidden size
    # H, context C", read from the block the backend reads them from.
    echoed = report.echoed_config()["text_config"]
    for field in ("num_hidden_layers", "hidden_size", "max_position_embeddings"):
        assert echoed[field] == session.text[field], (field, echoed[field])
    widths = WIDTHS_LINE.search(report.stdout)
    assert widths, report.transcript()
    assert int(widths.group(1)) == session.text["num_hidden_layers"], widths.group(0)


def test_check_exits_one_on_the_multimodal_file_and_zero_with_text_only(session):
    refused = session.run("--check", session.model)
    assert refused.code == 1, refused.transcript()
    verdict = MULTIMODAL_VERDICT.search(refused.stderr)
    assert verdict, refused.transcript()
    families, prefixes = int(verdict.group(1)), verdict.group(2).split(", ")
    # The towers section on stdout and the verdict on stderr count the same
    # families, and the verdict names only prefixes the flag can decline.
    header = TOWERS_HEADER.search(refused.stdout)
    assert header, refused.transcript()
    assert int(header.group(1)) == families, (header.group(0), verdict.group(0))
    assert prefixes and set(prefixes) <= set(NON_TEXT_PREFIXES), prefixes
    tensors = TENSORS_HEADER.search(refused.stdout)
    assert tensors and families < int(tensors.group(1)), refused.transcript()

    served = session.run("--check", "--text-only", session.model)
    assert served.code == 0, served.transcript()
    assert "[OK] CPU backend compatibility: supported" in served.stderr, (
        served.transcript()
    )
    # Declining the towers does not hide them: the section is printed either way.
    assert TOWERS_HEADER.search(served.stdout), served.transcript()


def test_a_missing_checkpoint_exits_one_not_two(session, tmp_path):
    absent = str(tmp_path / "absent")
    run = session.run("--check", absent)
    assert run.code == 1, run.transcript()
    assert "no checkpoint at" in run.stderr, run.transcript()
