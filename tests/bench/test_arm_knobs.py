"""Every knob `bench run --arm` advertises must actually reach the container.

An arm applies its overrides as the environment of the `docker compose` subprocess
(bench/commands/run.py::apply_arm), and that environment reaches a service only through
interpolation in the compose file. A knob written there as a literal is inert: both arms
run the same configuration and the comparison reports a clean null result on an A/B that
measured nothing. BQ-F52 is exactly that, found only after two host runbook entries had
been written against knobs that could not move.

This is the check that makes the arm mechanism self-policing, because the failure has no
other signature: conditions.py records the thread count from the container's own inspected
environment, so both arms carry the same value and `bench compare` sees two matching
counts and raises nothing.
"""

from __future__ import annotations

import os
import re

import pytest

COMPOSE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docker-compose.inference.yaml",
)

# The knobs bench/commands/run.py's docstring names as measurable without new code, and
# the service each one belongs to.
ARM_KNOBS = [
    ("llama-cpu", "HUGEPAGES"),
    ("llama-cpu", "NO_REPACK"),
    ("llama-cpu", "THREADS"),
    ("llama-cpu", "THREADS_BATCH"),
    # The pair the vLLM per-request timings ride on. The engine refuses
    # --enable-per-request-metrics while statistics logging is off, so an arm that turns
    # the timings on has to turn the other off in the same invocation -- which it can
    # only do if both cross into the container.
    ("vllm-gpu", "DISABLE_LOG_STATS"),
    ("vllm-gpu", "ENABLE_PER_REQUEST_METRICS"),
]


def _environment(service: str) -> dict[str, str]:
    """The `environment:` list of one compose service, as written on disk.

    Parsed textually rather than with a YAML loader on purpose: the defect is the
    difference between a literal and an interpolation, and a loader that resolved
    interpolations would erase the very distinction under test.
    """
    text = open(COMPOSE, encoding="utf-8").read()
    block = re.search(
        rf"^  {re.escape(service)}:\n(.*?)(?=^  \S|\Z)", text, re.M | re.S
    )
    assert block, f"no service {service!r} in {COMPOSE}"
    env = re.search(
        r"^    environment:\n(.*?)(?=^    \S|\Z)", block.group(1), re.M | re.S
    )
    assert env, f"service {service!r} declares no environment block"
    pairs = {}
    for line in env.group(1).splitlines():
        item = line.strip()
        if not item.startswith("- "):
            continue
        key, _, value = item[2:].partition("=")
        pairs[key.strip()] = value.strip()
    return pairs


@pytest.mark.parametrize(("service", "knob"), ARM_KNOBS)
def test_every_advertised_arm_knob_is_interpolated(service, knob):
    env = _environment(service)
    assert knob in env, (
        f"{knob} is absent from {service}'s compose environment, so an arm setting it "
        "changes the compose subprocess environment and nothing else"
    )
    assert env[knob].startswith("${"), (
        f"{knob} is the literal {env[knob]!r} in the compose file, so an arm setting it "
        "never reaches the container and both arms of the A/B run the same value"
    )
