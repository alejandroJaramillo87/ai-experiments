"""Fixtures shared by the command-level tests.

`bench run` and `bench noise` are the two commands that own a run lifecycle, and both
need the same thing: every host-side effect substituted, so what is asserted is which
call happened in what order rather than any number. The fixture lives here because two
modules use it -- importing it across test modules works but reads as a redefinition to
every linter, and a fixture with two homes drifts.
"""

from __future__ import annotations

import dataclasses

import pytest

from bench import conditions, runner
from bench.commands import run as run_cmd

from .test_runner import FakeEngine


@dataclasses.dataclass
class Calls:
    """What the command asked the host to do, in order."""

    composed: list[dict]
    gated: list[str]
    engine: FakeEngine
    container_env: dict = dataclasses.field(default_factory=dict)


@pytest.fixture
def host(monkeypatch):
    """Substitute every host-side effect the run command has."""
    calls = Calls(composed=[], gated=[], engine=FakeEngine())

    def fake_compose(env, *args, compose_file=None):
        calls.composed.append(
            {k: env[k] for k in ("HUGEPAGES", "NO_REPACK", "THREADS") if k in env}
        )
        # A recreate changes what the container's environment says, which is where
        # bench/conditions.py reads env_knobs from. Without this the substitute reports
        # one unchanging container however many arms ran, and no test could tell an A/B
        # that reached the service from one that did not.
        calls.container_env = dict(calls.composed[-1])

    def fake_wait_healthy(spec, budget_s=None):
        calls.gated.append(spec.name)
        return 0.1

    monkeypatch.setattr(run_cmd, "compose", fake_compose)
    monkeypatch.setattr(runner, "wait_healthy", fake_wait_healthy)
    monkeypatch.setattr("bench.client.stream", calls.engine)
    # conditions.collect runs `docker inspect` and `nvidia-smi` against whatever
    # machine happens to run the suite. Neither exists in a sandbox and neither is
    # under test here, so the block is supplied rather than probed.
    monkeypatch.setattr(
        conditions,
        "collect",
        lambda service, container_name=None: {
            "host": {"client_cpuset": "0-15", "service_cpuset": "0-15"},
            "env_knobs": dict(calls.container_env),
            "server_argv": ["llama-server"],
        },
    )
    return calls
