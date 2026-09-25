"""RH.36, attention-capture as its own binary on the real E4B, as a test the
E4B tier runs.

RH.18 reads attention weights off the real checkpoint through the ladder's
L6 rung, which captures in-process through the same `AttentionTap` and never
runs the binary. The binary's own surface, `--prompt`, `--layers`, `--heads`
and `--out`, is pinned by `runtime/tests/attention_capture.rs` over a tiny
fixture whose window is three tokens. This test runs the binary on the
released checkpoint over a prompt longer than the real window, so the
artifact it writes carries a sliding span that moves and a global one that
does not, and reads both back against the schedule `config.json` states.

Two facts the sandbox cannot show. The projected size line is printed before
the allocation, so a request the machine cannot hold is refused by the
reader rather than discovered as an OOM: the line is read and held to the
arithmetic the manifest's token count implies. And the row mass is measured
on rows of hundreds of keys rather than three, where an accumulated error
would show.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest

from tests.hardware.conftest import REPO_ROOT, e4b_checkpoint

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.hw_e4b,
    pytest.mark.cpu,
    pytest.mark.ram,
    pytest.mark.harness,
]

HARNESS_DIR = os.path.join(REPO_ROOT, "rust-harness")
ATTENTION_CAPTURE = os.path.join(HARNESS_DIR, "target", "release", "attention-capture")

# The first layer and the last, one head: layer 0 is sliding on every Gemma 4
# variant and the last layer is global, which the test reads from
# `layer_types` rather than assuming. Two layers and one head keep the
# capture small; the quadratic term is what the size line exists to state.
HEADS = "0"

# A prompt past the 512-token window, so `span_first` on the sliding layer
# climbs; a full 64-word line encodes to about 80 tokens, so ten lines pass
# the window with margin. Content is immaterial: the weights are read for
# their mass and their support, never their values.
LINES = 10
WORDS_PER_LINE = 64

SIZE_LINE = re.compile(
    r"^\[INFO\] Attention: (\d+) tokens, (\d+) layers x (\d+) heads = (\d+) matrices, "
    r"([\d.]+) MiB$",
    re.M,
)
OK_LINE = re.compile(r"^\[OK\] Attention: (\d+) matrices of (\d+)x(\d+) -> (.+)$", re.M)

# Row mass over a few hundred f32 keys.
MASS_TOLERANCE = 1e-5

# One load of the checkpoint plus one prefill of under a thousand tokens at
# the 10 tok/s RH.37's runs measured.
RUN_TIMEOUT_S = 15 * 60


def prompt() -> str:
    words = "matter energy motion field charge wave particle heat light sound".split()
    lines = []
    for i in range(LINES):
        lines.append(
            " ".join(words[(i + j) % len(words)] for j in range(WORDS_PER_LINE))
        )
    return "\n".join(lines)


def require_binary() -> None:
    """A missing binary fails rather than skips: it is built on the same
    host, so its absence is a misconfigured run and not the unavailable
    hardware the conftest rule exempts."""
    assert os.path.isfile(ATTENTION_CAPTURE), (
        f"built binary absent: {ATTENTION_CAPTURE}; run "
        f"`cargo build --release -p runtime --bin attention-capture` in {HARNESS_DIR}"
    )


class Capture:
    """One capture: the checkpoint's text config, the run's streams, and the
    directory the artifact was written to."""

    def __init__(
        self,
        model: str,
        text: dict,
        layers: list[int],
        out_dir: str,
        result: subprocess.CompletedProcess[str],
    ):
        self.model = model
        self.text = text
        self.layers = layers
        self.out_dir = out_dir
        self.code = result.returncode
        self.stdout = result.stdout
        self.stderr = result.stderr

    def transcript(self) -> str:
        return f"stdout:\n{self.stdout}\nstderr:\n{self.stderr}"

    def manifest(self) -> dict:
        with open(os.path.join(self.out_dir, "manifest.json"), encoding="utf-8") as f:
            return json.load(f)

    def shard(self):
        import numpy as np

        return np.load(os.path.join(self.out_dir, "attention.npz"))

    def expected_span_first(self, layer: int, tokens: int) -> list[int]:
        """The first visible key per position, from the schedule the config
        states: a sliding layer sees the last `sliding_window` keys, a global
        layer sees every key."""
        kind = self.text["layer_types"][layer]
        if kind == "sliding_attention":
            window = self.text["sliding_window"]
            return [max(0, p - window + 1) for p in range(tokens)]
        assert kind == "full_attention", (layer, kind)
        return [0] * tokens


@pytest.fixture(scope="module")
def capture(tmp_path_factory) -> Capture:
    model = e4b_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")
    require_binary()
    with open(os.path.join(model, "config.json"), encoding="utf-8") as f:
        text = json.load(f)["text_config"]
    layers = [0, text["num_hidden_layers"] - 1]
    kinds = text["layer_types"]
    assert kinds[layers[0]] == "sliding_attention", kinds[:2]
    assert kinds[layers[1]] == "full_attention", kinds[-2:]
    out_dir = str(tmp_path_factory.mktemp("rh36") / "cap")
    result = subprocess.run(
        [
            ATTENTION_CAPTURE,
            "--model",
            model,
            "--text-only",
            "--prompt",
            prompt(),
            "--layers",
            ",".join(str(layer) for layer in layers),
            "--heads",
            HEADS,
            "--out",
            out_dir,
        ],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
    )
    return Capture(model, text, layers, out_dir, result)


def test_the_capture_exits_zero_and_states_its_size_before_writing(capture):
    assert capture.code == 0, capture.transcript()
    size = SIZE_LINE.search(capture.stderr)
    assert size, capture.transcript()
    tokens, layers, heads, pairs, mib = (
        int(size.group(1)),
        int(size.group(2)),
        int(size.group(3)),
        int(size.group(4)),
        float(size.group(5)),
    )
    assert (layers, heads, pairs) == (2, 1, 2), size.group(0)
    assert tokens > capture.text["sliding_window"], (
        f"{tokens} tokens do not pass the window of {capture.text['sliding_window']}: "
        "the sliding span never moves"
    )
    # 4 bytes per weight, pairs x tokens^2 weights, printed to one decimal.
    assert abs(mib - pairs * tokens * tokens * 4 / 2**20) < 0.1, size.group(0)
    done = OK_LINE.search(capture.stderr)
    assert done, capture.transcript()
    assert (int(done.group(1)), int(done.group(2)), int(done.group(3))) == (
        pairs,
        tokens,
        tokens,
    )
    assert done.group(4) == capture.out_dir, done.group(0)
    # The size line precedes the completion line: stated before the work.
    assert size.start() < done.start(), capture.transcript()


def test_the_manifest_names_the_request_and_the_shard(capture):
    assert capture.code == 0, capture.transcript()
    manifest = capture.manifest()
    assert manifest["artifact"] == "attention_capture", manifest
    assert manifest["site"] == "attention_weights", manifest
    assert manifest["softmax"] == "post", manifest
    assert manifest["dtype"] == "float32", manifest
    assert manifest["shard"] == "attention.npz", manifest
    assert manifest["layers"] == capture.layers, manifest
    assert manifest["heads"] == [int(head) for head in HEADS.split(",")], manifest
    assert manifest["model_id"] == capture.model, manifest
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["model_hash"]), manifest["model_hash"]
    assert manifest["tokens"] == int(SIZE_LINE.search(capture.stderr).group(1)), (
        manifest
    )
    # Written whole or not at all: no temporary beside the manifest.
    assert sorted(os.listdir(capture.out_dir)) == ["attention.npz", "manifest.json"]


def test_every_row_is_a_distribution_over_its_own_span(capture):
    import numpy as np

    assert capture.code == 0, capture.transcript()
    shard = capture.shard()
    tokens = len(shard["tokens"])
    assert tokens == capture.manifest()["tokens"]
    for layer in capture.layers:
        first = shard[f"span_first_layer_{layer}"].tolist()
        assert first == capture.expected_span_first(layer, tokens), (
            layer,
            first[:8],
            first[-8:],
        )
        matrix = shard[f"layer_{layer}_head_{HEADS}"]
        assert matrix.shape == (tokens, tokens), (layer, matrix.shape)
        mass = matrix.sum(axis=1)
        assert np.abs(mass - 1.0).max() < MASS_TOLERANCE, (
            layer,
            mass.min(),
            mass.max(),
        )
        # Exactly zero outside the span on both sides: keys after the query,
        # and on the sliding layer keys the window has left behind.
        keys = np.arange(tokens)
        outside = (keys[None, :] > keys[:, None]) | (
            keys[None, :] < np.array(first)[:, None]
        )
        assert not matrix[outside].any(), (layer, np.count_nonzero(matrix[outside]))
