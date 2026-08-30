"""Host facts, and what happens when the host will not answer.

`collect` runs after every measurement, from `bench/runner.py::assemble`, and that
call sits outside the error handler in `bench/commands/run.py`. Anything raising here
discards a completed run of up to an hour, so the contract is that a fact this module
cannot obtain becomes an absent field and never an exception.

Nothing tested this module before. Three tests in test_runner.py reach it by calling
`assemble`, which means those tests really do invoke `nvidia-smi` and `docker` on
whatever machine runs the suite, and assert nothing about what comes back.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import yaml

from bench import conditions

COMPOSE = pathlib.Path(__file__).resolve().parents[2] / "docker-compose.inference.yaml"


class TestRunDegrades:
    def test_a_missing_binary_is_an_absent_fact(self, monkeypatch):
        monkeypatch.setattr(conditions.shutil, "which", lambda name: None)
        assert conditions._run(["nvidia-smi"]) is None

    def test_a_non_zero_exit_is_an_absent_fact(self, monkeypatch):
        monkeypatch.setattr(conditions.shutil, "which", lambda name: "/usr/bin/x")
        monkeypatch.setattr(
            conditions.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 1, "output", ""),
        )
        assert conditions._run(["docker"]) is None

    def test_a_timeout_is_an_absent_fact_not_a_failed_run(self, monkeypatch):
        def expired(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="docker", timeout=30)

        monkeypatch.setattr(conditions.shutil, "which", lambda name: "/usr/bin/x")
        monkeypatch.setattr(conditions.subprocess, "run", expired)
        assert conditions._run(["docker"]) is None

    def test_an_os_error_is_an_absent_fact(self, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("exec format error")

        monkeypatch.setattr(conditions.shutil, "which", lambda name: "/usr/bin/x")
        monkeypatch.setattr(conditions.subprocess, "run", boom)
        assert conditions._run(["docker"]) is None


class TestHugepages:
    def test_an_unreadable_meminfo_yields_no_counters(self, monkeypatch):
        monkeypatch.setattr(conditions, "MEMINFO", "/proc/does-not-exist")
        assert conditions.hugepages() == {}

    def test_counters_are_read_from_meminfo(self, tmp_path, monkeypatch):
        meminfo = tmp_path / "meminfo"
        meminfo.write_text(
            "MemTotal:       131072000 kB\n"
            "HugePages_Total:      80\n"
            "HugePages_Free:       80\n"
            "Hugepagesize:    1048576 kB\n"
        )
        monkeypatch.setattr(conditions, "MEMINFO", str(meminfo))
        pages = conditions.hugepages()
        assert pages["HugePages_Total"] == 80
        # The size is read rather than assumed, because the repository states both
        # 2 MB and 1 GB in different documents and only the host settles it.
        assert pages["Hugepagesize"] == 1048576


class TestGpu:
    def test_no_query_output_yields_no_gpu_facts(self, monkeypatch):
        monkeypatch.setattr(conditions, "_run", lambda *a, **k: None)
        assert conditions.gpu() == {}

    def test_a_short_row_is_refused_rather_than_indexed(self, monkeypatch):
        monkeypatch.setattr(conditions, "_run", lambda *a, **k: "RTX 5090")
        assert conditions.gpu() == {}

    def test_a_non_numeric_vram_field_becomes_null(self, monkeypatch):
        monkeypatch.setattr(
            conditions, "_run", lambda *a, **k: "RTX 5090, 580.65.06, [N/A]"
        )
        facts = conditions.gpu()
        assert facts["vram_total_mib"] is None
        assert facts["driver_version"] == "580.65.06"


class TestContainer:
    def test_no_inspect_output_yields_no_container_facts(self, monkeypatch):
        monkeypatch.setattr(conditions, "_run", lambda *a, **k: None)
        assert conditions.container("llama-cpu") == {}

    def test_unparseable_output_yields_no_container_facts(self, monkeypatch):
        monkeypatch.setattr(conditions, "_run", lambda *a, **k: "not json at all")
        assert conditions.container("llama-cpu") == {}

    def test_env_and_argv_are_read_from_the_inspect_document(self, monkeypatch):
        document = {
            "Config": {
                "Env": ["THREADS=16", "LD_PRELOAD=/app/libzen5_optimizer.so"],
                "Entrypoint": ["/app/entrypoint.sh"],
                "Cmd": ["--threads", "16"],
                "Image": "llama-cpu:latest",
            },
            "HostConfig": {"CpusetCpus": "0-15"},
            "Image": "sha256:abc",
        }
        monkeypatch.setattr(conditions, "_run", lambda *a, **k: json.dumps(document))
        facts = conditions.container("llama-cpu")
        assert facts["env_knobs"]["THREADS"] == "16"
        assert facts["server_argv"] == ["/app/entrypoint.sh", "--threads", "16"]
        assert facts["service_cpuset"] == "0-15"

    def test_valid_json_that_is_not_a_mapping_is_refused(self, monkeypatch):
        # BQ-F11. A JSON array parses, and .get on a list raises AttributeError out of
        # container() and out of collect(), which runs after all measurement and outside
        # the caller's handler. No real `docker inspect --format '{{json .}}'` produces
        # this shape, so the guard is for the contract, not for a reachable input.
        monkeypatch.setattr(conditions, "_run", lambda *a, **k: '["a", "b"]')
        assert conditions.container("llama-cpu") == {}


class TestCollect:
    def test_a_host_that_answers_nothing_still_yields_a_well_formed_document(
        self, monkeypatch
    ):
        monkeypatch.setattr(conditions, "_run", lambda *a, **k: None)
        monkeypatch.setattr(conditions, "MEMINFO", "/proc/does-not-exist")
        facts = conditions.collect("llama-cpu")
        assert set(facts) == {"host", "env_knobs", "server_argv"}
        assert facts["server_argv"] == []
        assert facts["env_knobs"] == {}
        assert facts["host"]["service_cpuset"] is None

    def test_a_working_gpu_costs_two_separate_query_processes(self, monkeypatch):
        # BQ-F12. gpu() spawns nvidia-smi for the CSV query, then calls cuda_version()
        # from inside its own return literal, which spawns nvidia-smi a second time to
        # regex the banner. Both run after every measurement is complete, on the run's
        # critical path, each under a 15 second timeout.
        #
        # A stub that fails every call cannot see this: gpu() returns early when the
        # first query yields nothing, so the second spawn is never reached and the
        # cost only appears on a host where the tool works.
        calls: list[list[str]] = []

        def record(args, timeout=15):
            calls.append(args)
            if args[0] == "nvidia-smi" and len(args) > 1:
                return "RTX 5090, 580.65.06, 32607"
            if args[0] == "nvidia-smi":
                return "CUDA Version: 13.0"
            return None

        monkeypatch.setattr(conditions, "_run", record)
        monkeypatch.setattr(conditions, "MEMINFO", "/proc/does-not-exist")
        conditions.collect("llama-cpu")
        binaries = [call[0] for call in calls]
        assert binaries.count("nvidia-smi") == 2
        assert "docker" in binaries

    def test_a_host_without_a_gpu_never_reaches_the_second_query(self, monkeypatch):
        calls: list[list[str]] = []

        def record(args, timeout=15):
            calls.append(args)
            return None

        monkeypatch.setattr(conditions, "_run", record)
        monkeypatch.setattr(conditions, "MEMINFO", "/proc/does-not-exist")
        conditions.collect("llama-cpu")
        assert [call[0] for call in calls].count("nvidia-smi") == 1

    def test_the_service_key_is_what_reaches_docker_inspect(self, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(conditions, "_run", lambda *a, **k: None)
        monkeypatch.setattr(
            conditions, "container", lambda name: seen.append(name) or {}
        )
        monkeypatch.setattr(conditions, "MEMINFO", "/proc/does-not-exist")
        conditions.collect("llama-cpu")
        # BQ-F13. The compose service key stands in for a container name, and the
        # parameter that would carry a resolved name is unused at both call sites.
        assert seen == ["llama-cpu"]

    def test_a_resolved_name_overrides_the_service_key(self, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(conditions, "_run", lambda *a, **k: None)
        monkeypatch.setattr(
            conditions, "container", lambda name: seen.append(name) or {}
        )
        monkeypatch.setattr(conditions, "MEMINFO", "/proc/does-not-exist")
        conditions.collect("llama-cpu", container_name="ai-experiments-llama-cpu-1")
        assert seen == ["ai-experiments-llama-cpu-1"]


class TestTheServiceKeyIsAlsoTheContainerName:
    """BQ-F13, as a tripwire rather than a fix.

    `collect` hands the compose service key to `docker inspect`, which wants a container
    name. The two coincide only because docker-compose.inference.yaml pins
    `container_name:` to the service key for every benchmarked service. Drop that line
    and compose names the container `<project>-<service>-<index>`, every inspect returns
    nothing, and conditions.host silently loses its cpuset, image digest, thread count
    and argv -- an absent fact, by design, so no run would fail to report the loss.
    """

    def test_compose_pins_the_container_name_to_the_service_key(self):
        services = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
        for name in ("llama-cpu", "llama-gpu", "vllm-gpu"):
            assert services[name].get("container_name") == name, (
                f"{name} no longer pins container_name, so conditions.collect must be "
                "passed a resolved container name rather than the service key"
            )
