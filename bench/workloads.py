"""Profile parsing and matrix expansion: what gets run, and in what order.

A profile is a TOML file rather than Python config for two reasons: it is hand-edited
and needs comments, and its bytes hash into the report as profile_sha256, so two runs
can be shown to have executed the same workload rather than merely claiming to.

The expansion produces a flat list of cells -- (point, repeat index) -- already in
execution order. Ordering is the part that matters. Running every repeat of one point
before moving to the next lets thermal drift, clock behaviour, or a background process
alias onto that point alone; round-robin over points spreads any such drift across all
of them. This is the protocol
docs/optimizations/ideation/rust-harness-optimization-candidates.md prescribes, written
after four consecutive runs of one unmodified binary spread 44%.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import tomllib

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")

PREFILL = "prefill"
DECODE = "decode"


class ProfileError(ValueError):
    """A profile is malformed, or asks a service for something it cannot do."""


@dataclasses.dataclass(frozen=True)
class Point:
    """One workload configuration: an arm of the matrix, before repeats."""

    phase: str
    prompt_tokens: int
    max_tokens: int
    concurrency: int

    @property
    def key(self) -> str:
        return (
            f"{self.phase}/p{self.prompt_tokens}/g{self.max_tokens}/c{self.concurrency}"
        )


@dataclasses.dataclass(frozen=True)
class Cell:
    """One point at one repeat index. `warmup` cells are run and discarded."""

    point: Point
    repeat: int
    warmup: bool


@dataclasses.dataclass(frozen=True)
class Profile:
    name: str
    repeats: int
    warmup: int
    order: str
    seed: int
    sha256: str
    prefill_prompt_tokens: tuple[int, ...]
    prefill_max_tokens: int
    decode_prompt_tokens: int
    decode_max_tokens: tuple[int, ...]
    concurrency: tuple[int, ...]

    def points(self) -> list[Point]:
        points = [
            Point(PREFILL, n, self.prefill_max_tokens, 1)
            for n in self.prefill_prompt_tokens
        ]
        # The prefill sweep is always concurrency 1: above it, time to first token
        # carries queue wait, and a slope fitted through queue-contaminated points
        # measures the scheduler rather than the model.
        for gen in self.decode_max_tokens:
            for conc in self.concurrency:
                points.append(Point(DECODE, self.decode_prompt_tokens, gen, conc))
        return points

    def cells(self) -> list[Cell]:
        """Every cell in execution order: warm-ups first, then round-robin repeats."""
        points = self.points()
        cells = [
            Cell(p, repeat=-1, warmup=True) for p in points for _ in range(self.warmup)
        ]
        if self.order == "interleaved":
            for repeat in range(self.repeats):
                cells.extend(Cell(p, repeat=repeat, warmup=False) for p in points)
        else:
            for point in points:
                cells.extend(
                    Cell(point, repeat=r, warmup=False) for r in range(self.repeats)
                )
        return cells


def profile_path(name: str) -> str:
    if os.path.sep in name or name.endswith(".toml"):
        return name
    return os.path.join(PROFILE_DIR, f"{name}.toml")


def load(name: str, service: str) -> Profile:
    """Read a profile and resolve it against one service.

    Service-specific overrides exist because the same wall-clock budget buys very
    different workloads on a 5090 and on a CPU decoding an MoE at 35 tokens/second. The
    override is recorded in the profile, not passed on the command line, so a run's
    workload is reproducible from its profile hash alone.
    """
    path = profile_path(name)
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except FileNotFoundError as exc:
        available = sorted(
            f[:-5] for f in os.listdir(PROFILE_DIR) if f.endswith(".toml")
        )
        raise ProfileError(
            f"no profile {name!r} at {path}; available: {available}"
        ) from exc
    document = tomllib.loads(raw.decode("utf-8"))
    overrides = document.get("overrides", {}).get(service, {})

    concurrency = document.get("concurrency", {}).get(service)
    if concurrency is None:
        raise ProfileError(
            f"profile {name!r} states no concurrency for service {service!r}; "
            "every benchable service needs an entry, even if it is [1]"
        )

    prefill = document[PREFILL]
    decode = document[DECODE]
    prompt_tokens = tuple(
        overrides.get("prefill_prompt_tokens", prefill["prompt_tokens"])
    )
    if len(set(prompt_tokens)) < 2:
        raise ProfileError(
            f"profile {name!r} gives service {service!r} fewer than two distinct "
            "prefill prompt lengths; a prefill slope cannot be fitted through one "
            "point, and a single-point prefill rate charges the fixed per-request "
            "overhead to the model"
        )

    return Profile(
        name=document["name"],
        repeats=document["repeats"],
        warmup=document["warmup"],
        order=document.get("order", "interleaved"),
        seed=document.get("seed", 42),
        sha256=hashlib.sha256(raw).hexdigest(),
        prefill_prompt_tokens=prompt_tokens,
        prefill_max_tokens=prefill["max_tokens"],
        decode_prompt_tokens=overrides.get(
            "decode_prompt_tokens", decode["prompt_tokens"]
        ),
        decode_max_tokens=tuple(
            overrides.get("decode_max_tokens", decode["max_tokens"])
        ),
        concurrency=tuple(concurrency),
    )
