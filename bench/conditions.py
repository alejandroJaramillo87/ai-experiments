"""Run conditions: what the machine and the service actually were.

What makes a number citable a year later is this block, not the number. None of the
archived results record the model, the quantization, the thread count, or the driver
they ran against, which is why docs/optimizations/README.md's 26.2% figure cannot be
reproduced: it was measured at THREADS=12 against a service that now runs 16, and
nothing in the artifact says so.

Everything here is read from the running system rather than from configuration. The
distinction is not pedantry. docker/llama-cpu/entrypoint.cpu.sh spent its whole life
printing a launch command built from a variable it never assigned, while compose,
.env and three documents all described the flags it was supposedly passing. Recording
the server's actual argv is what turns that class of silent misconfiguration into a
diffable field.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

MEMINFO = "/proc/meminfo"


def _run(args: list[str], timeout: int = 15) -> str | None:
    """Best effort: a missing tool records an absent fact, it does not fail a run."""
    if not shutil.which(args[0]):
        return None
    try:
        out = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def hugepages() -> dict:
    """Huge-page counters, which are the evidence the wrapper took pages at all.

    The repository contradicts itself three ways about the size in use --
    hugepages/src/memory/hugepage_wrapper.cpp allocates MAP_HUGE_1GB, both files under
    hugepages/ say 2 MB, and docs/optimizations/ says both -- so the size is read here
    rather than assumed, and a run settles it.
    """
    facts: dict[str, int] = {}
    try:
        with open(MEMINFO, encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in ("HugePages_Total", "HugePages_Free", "Hugepagesize"):
                    digits = re.search(r"\d+", rest)
                    if digits:
                        facts[key] = int(digits.group())
    except OSError:
        return {}
    return facts


def gpu() -> dict:
    """GPU name and driver, from nvidia-smi. Absent on a CPU-only arm."""
    line = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not line:
        return {}
    parts = [p.strip() for p in line.splitlines()[0].split(",")]
    if len(parts) < 3:
        return {}
    return {
        "gpu": parts[0],
        "driver_version": parts[1],
        "vram_total_mib": int(parts[2]) if parts[2].isdigit() else None,
        "cuda_version": cuda_version(),
    }


def cuda_version() -> str | None:
    text = _run(["nvidia-smi"])
    if not text:
        return None
    match = re.search(r"CUDA Version:\s*([\d.]+)", text)
    return match.group(1) if match else None


def container(name: str) -> dict:
    """Env, argv and cpuset as docker reports them for a running container.

    `server_argv` is the field that matters. It records the command the service is
    actually running, so a report can be checked against what the compose file and the
    entrypoint claim rather than trusting either.
    """
    raw = _run(
        [
            "docker",
            "inspect",
            "--format",
            "{{json .}}",
            name,
        ],
        timeout=30,
    )
    if not raw:
        return {}
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    config = info.get("Config") or {}
    host_config = info.get("HostConfig") or {}
    env = {}
    for item in config.get("Env") or []:
        key, _, value = item.partition("=")
        env[key] = value
    return {
        "env_knobs": env,
        "server_argv": (config.get("Entrypoint") or []) + (config.get("Cmd") or []),
        "service_cpuset": host_config.get("CpusetCpus") or None,
        "image": config.get("Image"),
        "image_digest": (info.get("Image") or None),
    }


def client_cpuset() -> str | None:
    """The cores this process is allowed to run on.

    Worth recording because of a trap: the rust-harness protocol prescribes
    `taskset -c 0-7`, but llama-cpu holds cpuset 0-15. Pinning the benchmark client
    inside the service's range makes client and server contend for cores and taints the
    measurement it was meant to stabilise. `bench compare` refuses on a mismatch between
    arms so the mistake cannot be averaged away silently.
    """
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None
    if not cpus:
        return None
    # Render as ranges: "0-7", "0-3,8-11".
    ranges, start, prev = [], cpus[0], cpus[0]
    for cpu in cpus[1:]:
        if cpu != prev + 1:
            ranges.append((start, prev))
            start = cpu
        prev = cpu
    ranges.append((start, prev))
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)


def collect(service: str, container_name: str | None = None) -> dict:
    """Every host-side fact a report carries under conditions.host."""
    host: dict = {"client_cpuset": client_cpuset()}
    host.update(gpu())
    pages = hugepages()
    if pages:
        host["hugepages_total"] = pages.get("HugePages_Total")
        host["hugepages_free"] = pages.get("HugePages_Free")
        host["hugepage_size_kb"] = pages.get("Hugepagesize")
    info = container(container_name or service)
    host["service_cpuset"] = info.get("service_cpuset")
    host["image_digest"] = info.get("image_digest")
    env = info.get("env_knobs", {})
    host["threads"] = env.get("THREADS")
    host["ld_preload"] = env.get("LD_PRELOAD")
    return {"host": host, "env_knobs": env, "server_argv": info.get("server_argv", [])}
