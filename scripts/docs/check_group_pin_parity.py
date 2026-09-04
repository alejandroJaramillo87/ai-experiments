#!/usr/bin/env python3
"""Fail the quality gate when one package carries two different pins.

`pyproject.toml` names the same package in more than one place on purpose. The
`sandbox` group re-pins numpy, omegaconf, pyyaml, xxhash, tokenizers,
scikit-learn, hydra-core, fastapi, httpx, uvicorn, requests, gguf and
safetensors so a torch-free environment can still collect the suite; the
`interp` group repeats the training group's xxhash and the sandbox group's
omegaconf, and says so in a comment. The overlap is deliberate -- uv permits it,
and each group has to stand alone because the images sync one group each.

What the overlap does not come with is anything holding the copies together.
`uv.lock` is no help: it guarantees one resolution is internally consistent, not
that two literal `==` strings in the source agree. Bump scikit-learn in
`[project.dependencies]` and the sandbox group keeps the old pin silently, and
the divergence surfaces later as a sandbox that resolves differently from the
workstation -- a difference in the environment that runs the tests, which is the
one place a difference is hardest to reason about.

The sandbox group's own comment already states the obligation ("Versions mirror
the core/interp/training pins where the package is pinned there"). This turns
that sentence into a gate.

Scope is every `==` pin in `[project.dependencies]`, every
`[project.optional-dependencies]` extra, and every `[dependency-groups]` group.
Constraints that are not exact pins are ignored: a package pinned in one place
and left to resolve in another is a normal thing to do, and only two exact pins
can contradict each other.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

# A requirement string up to its first exact pin: name, optional extras, `==`,
# version. Anything carrying an environment marker is truncated at the `;` first,
# because a marker can legitimately differ between two otherwise identical pins.
PIN = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*==\s*(?P<version>[^\s,]+)"
)


def normalize(name: str) -> str:
    """PEP 503 name normalization, so `types-pyyaml` and `types_PyYAML` agree."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(requirements: list[str]) -> dict[str, str]:
    """The exact pins in one requirement list, keyed by normalized name."""
    found: dict[str, str] = {}
    for raw in requirements:
        match = PIN.match(raw.split(";")[0].strip())
        if match:
            found[normalize(match.group("name"))] = match.group("version")
    return found


def collect_sources(data: dict) -> dict[str, dict[str, str]]:
    """Every pin-bearing section of pyproject.toml, keyed by a readable label."""
    project = data.get("project", {})
    sources = {"core": pins(project.get("dependencies", []))}
    for extra, requirements in project.get("optional-dependencies", {}).items():
        sources[f"extra:{extra}"] = pins(requirements)
    for group, requirements in data.get("dependency-groups", {}).items():
        sources[f"group:{group}"] = pins(requirements)
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=REPO_ROOT / "pyproject.toml",
        help="the pyproject.toml to check",
    )
    args = parser.parse_args()
    out = Printer()

    try:
        data = tomllib.loads(args.pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        out.error(f"pin-parity: cannot read {args.pyproject}: {exc}")
        return 1

    sources = collect_sources(data)

    # An empty-set guard, matching the sibling gates: a pyproject.toml this
    # parses to nothing would pass while checking nothing at all.
    total = sum(len(p) for p in sources.values())
    if not total:
        out.error(f"pin-parity: found no exact pins in {args.pyproject}")
        out.line("  Every section parsed empty, so the gate would check nothing.")
        return 1

    seen: dict[str, dict[str, str]] = defaultdict(dict)
    for label, section in sources.items():
        for name, version in section.items():
            seen[name][label] = version

    disagreeing = {
        name: where for name, where in seen.items() if len(set(where.values())) > 1
    }

    out.info(
        f"pin-parity: {len(sources)} sections, {total} pins, "
        f"{len(seen)} distinct packages"
    )
    if not disagreeing:
        out.ok("pin-parity: every package pinned twice carries the same version")
        return 0

    noun = "package carries" if len(disagreeing) == 1 else "packages carry"
    out.error(f"pin-parity: {len(disagreeing)} {noun} conflicting pins")
    for name, where in sorted(disagreeing.items()):
        spread = ", ".join(
            f"{label} {version}" for label, version in sorted(where.items())
        )
        out.line(f"  {name}: {spread}")
    out.line("  Solution: pick one version and write it in every section that")
    out.line("  pins the package. A group repeating a pin must repeat it exactly;")
    out.line("  to let a section resolve freely instead, drop its pin.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
