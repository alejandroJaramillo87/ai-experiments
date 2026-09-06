"""RH.14, the endpoint answering the interp client, as a test.

``--serve`` is the OpenAI-shaped endpoint the interp tooling calls through
``common/serving.py``. The wire contract is pinned in the sandbox against a
scripted model (``rust-harness/harness/tests/server.rs`` and
``serving_contract.rs``); what those cannot show is the half RH.14 was
written for: a real checkpoint behind the socket, the client's own
``wait_healthy`` timing the load against the budget the registry states,
and ``chat_completion`` returning coherent, deterministic text with no client
change. The 2026-08-12 run proved the second half and left two findings the
2026-09-06 re-run closed, and this module keeps closed: the server restored
a ``session.kv`` it would never continue, and ``start_budget_s`` was a
placeholder because both runs polled a server that was already up.

The server is spawned into an empty directory holding a deliberately invalid
``session.kv``. An invalid snapshot that is opened at all produces the
``Session snapshot ... discarded`` warning, so the absence of that line and
of ``Session restored`` is the evidence the file was never read, and the bytes
staying identical is the evidence it was never written.

The subject is ``gemma-4-E2B-it``, the standing choice unless an entry needs
E4B. It answers ``Name three primes.`` in under 30 tokens, so the greedy pair
runs in seconds on the CPU backend. ``hw_full`` rather than ``hw_smoke``
because it loads a real model; it never touches the GPU.
"""

from __future__ import annotations

import dataclasses
import os
import re
import socket
import subprocess
import threading
import time

import pytest

from common import serving
from tests.hardware.conftest import AI_DATA_ROOT, REPO_ROOT

pytestmark = [pytest.mark.hw_full, pytest.mark.cpu, pytest.mark.ram, pytest.mark.ssd]

HARNESS_DIR = os.path.join(REPO_ROOT, "rust-harness")
HARNESS = os.path.join(HARNESS_DIR, "target", "release", "harness")

MODEL_LOAD = re.compile(r"^\[INFO\] Model load: ([0-9.]+) s$", re.MULTILINE)
PROMPT = [{"role": "user", "content": "Name three primes."}]
MAX_TOKENS = 64
# wait_healthy sleeps 3 s between polls, so its figure can exceed the load by
# that much and no more.
POLL_S = 3
# The 2026-09-06 run answered in 2.7-3.1 s; a minute is a records failure,
# not a slow reply.
REQUEST_TIMEOUT_S = 60
NOT_A_SNAPSHOT = b"not a session.kv\n"


def checkpoint() -> str:
    root = os.environ.get("GEMMA4_CHECKPOINT_ROOT", f"{AI_DATA_ROOT}/models/hf")
    return os.path.join(root, "gemma-4-E2B-it")


def require_binary() -> None:
    """A missing binary fails rather than skips, as in test_obs_records.py:
    `make harness-build` builds it on this host."""
    assert os.path.isfile(HARNESS), (
        f"built binary absent: {HARNESS}; run `make harness-build` in {HARNESS_DIR}"
    )


def free_port() -> int:
    """Bound and released rather than guessed, so a stale server on 8005
    cannot answer for the one this module starts."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_healthy_or_die(server, spec, lines: list[str]) -> float:
    """The client's own poll, in slices, so a server that exits during the
    load fails at once with what it printed rather than after the whole
    budget has elapsed. The figure is the client's: wall time from process
    start to the first 200, as wait_healthy itself measures."""
    start = time.monotonic()
    deadline = start + spec.start_budget_s
    while True:
        try:
            serving.wait_healthy(spec, budget_s=POLL_S)
            return time.monotonic() - start
        except serving.ServingError:
            if server.poll() is not None:
                raise AssertionError(
                    f"the server exited with {server.returncode} before serving:\n"
                    + "".join(lines)
                ) from None
            if time.monotonic() >= deadline:
                raise


class Endpoint:
    """One served checkpoint: the timing, the two replies, and what it printed."""

    def __init__(self, spec, healthy_s, replies, stderr, snapshot_after):
        self.spec = spec
        self.healthy_s = healthy_s
        self.replies = replies
        self.stderr = stderr
        self.snapshot_after = snapshot_after


@pytest.fixture(scope="module")
def endpoint(tmp_path_factory) -> Endpoint:
    require_binary()
    model = checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E2B-it checkpoint not on disk at {model}")

    directory = tmp_path_factory.mktemp("rh14")
    snapshot = directory / "session.kv"
    snapshot.write_bytes(NOT_A_SNAPSHOT)
    # The registry's own spec with only the port moved, so the budget and
    # the health path under test are the ones every caller gets.
    spec = dataclasses.replace(serving.SERVICES["harness"], port=free_port())

    env = dict(os.environ, NO_COLOR="1")
    server = subprocess.Popen(
        [HARNESS, "--model", model, "--text-only", "--serve", str(spec.port)],
        cwd=directory,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    # Drained on a thread so a chatty server can never block on the pipe.
    stream = server.stderr
    assert stream is not None
    lines: list[str] = []
    reader = threading.Thread(target=lambda: lines.extend(stream), daemon=True)
    reader.start()
    try:
        healthy_s = wait_healthy_or_die(server, spec, lines)
        replies = [
            serving.chat_completion(
                spec,
                PROMPT,
                {"temperature": 0.0, "max_tokens": MAX_TOKENS},
                timeout=REQUEST_TIMEOUT_S,
            )
            for _ in range(2)
        ]
    finally:
        server.terminate()
        server.wait(timeout=30)
        reader.join(timeout=30)
    return Endpoint(spec, healthy_s, replies, "".join(lines), snapshot.read_bytes())


def test_the_client_sees_the_server_healthy_within_its_own_budget(endpoint):
    assert endpoint.healthy_s <= endpoint.spec.start_budget_s, (
        f"healthy in {endpoint.healthy_s:.1f} s against a budget of "
        f"{endpoint.spec.start_budget_s} s"
    )
    loads = MODEL_LOAD.findall(endpoint.stderr)
    assert len(loads) == 1, f"expected one Model load line:\n{endpoint.stderr}"
    # The poll started with the process, so the load is inside the figure.
    assert float(loads[0]) <= endpoint.healthy_s + POLL_S, (
        f"Model load {loads[0]} s but the client saw health at "
        f"{endpoint.healthy_s:.1f} s"
    )


def test_two_greedy_requests_return_identical_coherent_text(endpoint):
    first, second = (r["content"] for r in endpoint.replies)
    assert first.strip(), f"empty reply:\n{endpoint.stderr}"
    assert first == second, f"greedy replies differ:\n{first!r}\n{second!r}"
    # Coherent in the one sense a test can check: the reply is about primes.
    assert "2" in first and "3" in first and "5" in first, first


def test_usage_counts_are_internally_consistent(endpoint):
    for reply in endpoint.replies:
        usage = reply["usage"]
        assert (
            usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
        ), usage
        assert 0 < usage["completion_tokens"] <= MAX_TOKENS, usage
    assert endpoint.replies[0]["usage"] == endpoint.replies[1]["usage"]


def test_serve_mode_leaves_the_snapshot_alone(endpoint):
    err = endpoint.stderr
    assert "Session restored" not in err, err
    assert "Session snapshot" not in err, err
    assert endpoint.snapshot_after == NOT_A_SNAPSHOT, "session.kv was rewritten"
