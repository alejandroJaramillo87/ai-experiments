"""Profiles, the workload matrix, and the prompts that fill it.

A profile is a TOML file rather than Python config because it is hand-edited and needs
comments. Its name goes into the report, and `bench compare` refuses two runs whose
profiles differ by name.

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
import functools
import os
import re
import tomllib

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(BENCH_DIR, "profiles")

# The corpus is the committed wikitext-2 validation split, read from the one copy the
# repository already has -- the same directory docker-compose.inference.yaml bind-mounts
# into llama-cpu at /app/datasets:ro for test_perplexity.sh. Real text rather than random
# token ids: a router trained on prose maps uniform vocabulary noise to a near-uniform
# expert distribution, the best case for load balance and the worst for expert-weight
# locality, and wrong with no predictable sign.
CORPUS = os.path.join(
    os.path.dirname(BENCH_DIR),
    "scripts",
    "benchmark",
    "datasets",
    "wikitext-2-raw",
    "wiki.valid.raw",
)

# English prose runs about four characters to the token across common BPE vocabularies.
# Only used to slice a candidate: the length that lands in the report is always the
# server's own usage.prompt_tokens, never this estimate.
CHARS_PER_TOKEN = 4

_SECTION_HEADER = re.compile(r"^=+ .* =+$")

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
            document = tomllib.load(handle)
    except FileNotFoundError as exc:
        available = sorted(
            f[:-5] for f in os.listdir(PROFILE_DIR) if f.endswith(".toml")
        )
        raise ProfileError(
            f"no profile {name!r} at {path}; available: {available}"
        ) from exc
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


@functools.lru_cache(maxsize=1)
def corpus() -> str:
    """The wikitext validation split as one line of prose.

    Section headers and blank runs go, and wikitext's tokenizer artefacts (` @-@ `,
    ` @,@ `, ` @.@ `) are folded back into ordinary punctuation so the text tokenizes
    like the prose it was before the corpus was prepared.
    """
    with open(CORPUS, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    body = " ".join(line for line in lines if line and not _SECTION_HEADER.match(line))
    body = re.sub(r"\s+", " ", body)
    return body.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")


def slices(count: int) -> list[str]:
    """`count` disjoint regions of the corpus.

    An MoE's decode rate moves with which experts a prompt routes to, so the decode arm
    draws three disjoint prompts and reports the spread between them as the routing
    sensitivity -- a measured quantity rather than an unmeasured confound. A dense model
    does identical work per token and takes one.
    """
    if count < 1:
        raise ValueError("need at least one slice")
    text = corpus()
    width = len(text) // count
    return [text[i * width : (i + 1) * width] for i in range(count)]


def build_prompt(target_tokens: int, region: str, nonce: str) -> str:
    """A prompt of roughly `target_tokens`, unique to `nonce`.

    The nonce leads the prompt rather than trailing it, and that placement is the point:
    a prefix cache keys on the leading tokens, so a unique prefix defeats it while a
    unique suffix would not. vLLM's prefix caching is a launch flag with no per-request
    override, so this is the only way to be sure a measured prefill actually happened.
    """
    budget = max(1, target_tokens * CHARS_PER_TOKEN)
    head = f"[{nonce}] "
    body = region[: max(0, budget - len(head))]
    if len(body) < budget - len(head):
        # Short region: repeat it rather than silently returning a shorter prompt.
        body = (region * (budget // max(1, len(region)) + 1))[: budget - len(head)]
    return head + body
