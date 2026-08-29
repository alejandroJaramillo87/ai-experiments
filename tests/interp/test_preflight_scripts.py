"""Tests for the interp concurrency preflights.

These guards enforce one-GPU-consumer-at-a-time before a harvest or workbench
starts. They previously read `docker compose ps` with stderr discarded and the
exit status swallowed by `|| true`, which made a failed compose invocation
indistinguishable from "nothing is running" -- and an empty service list is the
clear-to-start answer. The guard therefore failed *open* precisely when it could
not see the stack. These tests pin the fail-closed behavior, and the permitted
and refused paths that must keep working alongside it.
"""

from __future__ import annotations

import pytest

COMPOSE_FAILS = 'echo "no configuration file provided: not found" >&2; exit 1'
COMPOSE_OK = 'printf "%s" "$STUB_SERVICES"; exit 0'


@pytest.fixture
def compose(fake_bin):
    """Stub `docker` so the script sees a scripted compose result."""

    def _stub(body: str):
        fake_bin.stub("docker", body)

    return _stub


class TestPreflight:
    """scripts/interp/preflight.sh -- no GPU consumer may be running."""

    def test_refuses_when_compose_cannot_be_queried(self, compose, run_script):
        compose(COMPOSE_FAILS)
        result = run_script("interp/preflight.sh")
        assert result.returncode == 1
        assert "Cannot determine which services are running" in result.stdout
        # The underlying failure must reach the operator, not be swallowed.
        assert "no configuration file provided" in result.stdout

    def test_clear_when_nothing_is_running(self, compose, run_script):
        compose(COMPOSE_OK)
        result = run_script("interp/preflight.sh", env={"STUB_SERVICES": ""})
        assert result.returncode == 0
        assert "[OK]" in result.stdout

    @pytest.mark.parametrize("service", ["llama-gpu", "vllm-gpu"])
    def test_refuses_while_a_gpu_service_runs(self, compose, run_script, service):
        compose(COMPOSE_OK)
        result = run_script("interp/preflight.sh", env={"STUB_SERVICES": service})
        assert result.returncode == 1
        assert service in result.stdout

    def test_llama_cpu_is_permitted(self, compose, run_script):
        # Dedicated cores, no GPU: it does not contend for the one GPU.
        compose(COMPOSE_OK)
        result = run_script("interp/preflight.sh", env={"STUB_SERVICES": "llama-cpu"})
        assert result.returncode == 0


class TestAutointerpPreflight:
    """scripts/interp/np_autointerp_preflight.sh -- vllm-gpu up, others down."""

    def test_refuses_when_compose_cannot_be_queried(self, fake_bin, run_script):
        # curl must succeed so the script reaches the compose query at all.
        fake_bin.stub("curl", "exit 0")
        fake_bin.stub("docker", COMPOSE_FAILS)
        result = run_script("interp/np_autointerp_preflight.sh")
        assert result.returncode == 1
        assert "Cannot determine which services are running" in result.stdout

    def test_refuses_when_vllm_is_unhealthy(self, fake_bin, run_script):
        fake_bin.stub("curl", "exit 22")
        fake_bin.stub("docker", COMPOSE_OK)
        result = run_script("interp/np_autointerp_preflight.sh")
        assert result.returncode == 1
        assert "not healthy" in result.stdout
