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
import hashlib
import json
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
# It converts a requested token count into a character budget for slicing the corpus, and
# that requested count is what identifies a point in the report ("prompt_tokens"). It is a
# nominal length, not a measured one: the server's own count is reported beside it as
# `prompt_tokens_observed`, and the two differ by whatever this estimate got wrong.
CHARS_PER_TOKEN = 4

_SECTION_HEADER = re.compile(r"^=+ .* =+$")

PREFILL = "prefill"
DECODE = "decode"

# The two orderings, enumerated rather than implied: round-robin over points, and every
# repeat of one point together. Naming both is what keeps an unrecognised value from
# falling through to the blocked ordering the protocol exists to forbid.
INTERLEAVED = "interleaved"
BLOCKED = "blocked"
ORDERS = (INTERLEAVED, BLOCKED)

# The keys a per-service override may set. Anything else reaches no reader, so a
# misspelling would silently leave the base sweep in place.
OVERRIDABLE = ("prefill_prompt_tokens", "decode_prompt_tokens", "decode_max_tokens")

# A prompt length floor, in tokens. The runner leads every prompt with a nonce header to
# defeat a prefix cache, and that header is the cell's whole identity -- seed, run key,
# point key, repeat, warm-up occurrence, slot -- so it runs to about 47 characters for an
# ordinary interleaved A/B and grows with the arm name the operator chooses. At
# CHARS_PER_TOKEN this floor is 128 characters, which keeps the header under half the
# budget and leaves the body prose rather than nonce.
#
# Sized against the nonce bench/runner.py actually builds, which is not where this
# constant lives: an earlier floor of 16 was correct for the shorter "[seed-index-slot] "
# header it was written against and left 17 characters of prose once the nonce became the
# cell identity. build_prompt refuses outright when a header fills its budget, so the
# run-time backstop is there for an arm name long enough to defeat even this floor.
MIN_PROMPT_TOKENS = 32


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
        if self.order == INTERLEAVED:
            for repeat in range(self.repeats):
                cells.extend(Cell(p, repeat=repeat, warmup=False) for p in points)
        elif self.order == BLOCKED:
            for point in points:
                cells.extend(
                    Cell(point, repeat=r, warmup=False) for r in range(self.repeats)
                )
        else:
            # Refused rather than defaulted: an unrecognised value falling through to
            # the else branch bought blocked ordering, which is the one thing the
            # protocol forbids, from a value nobody wrote deliberately.
            raise ProfileError(f"order {self.order!r} is not one of {list(ORDERS)}")
        return cells

    @property
    def workload_hash(self) -> str:
        """A digest of the resolved workload -- what runs, not what the file calls itself.

        `name` is a string the profile asserts about itself, so two files with different
        sweeps and the same declared name are indistinguishable by name alone. Every
        field below is resolved per service, so the same file read for llama-cpu and for
        llama-gpu digests differently whenever an override narrows one of them.

        sha256 rather than the built-in `hash()`, which is salted per interpreter: the
        run that writes a digest and the run that reads it are different processes.
        """
        payload = json.dumps(
            {
                "repeats": self.repeats,
                "warmup": self.warmup,
                "order": self.order,
                "seed": self.seed,
                "prefill_prompt_tokens": list(self.prefill_prompt_tokens),
                "prefill_max_tokens": self.prefill_max_tokens,
                "decode_prompt_tokens": self.decode_prompt_tokens,
                "decode_max_tokens": list(self.decode_max_tokens),
                "concurrency": list(self.concurrency),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def profile_path(name: str) -> str:
    if os.path.sep in name or name.endswith(".toml"):
        return name
    return os.path.join(PROFILE_DIR, f"{name}.toml")


def _require(table: dict, key: str, where: str, name: str):
    """One required key, or a ProfileError that names what was missing and where.

    Bare subscripts raised KeyError, which neither command handler catches, so a missing
    key reached the operator as a traceback under the exit code that means regression.
    """
    if key not in table:
        raise ProfileError(f"profile {name!r}: {where}{key!r} is required")
    return table[key]


def _table(value, where: str, name: str) -> dict:
    if not isinstance(value, dict):
        raise ProfileError(
            f"profile {name!r}: {where} must be a table, not {type(value).__name__}"
        )
    return value


def _int(value, where: str, name: str, *, minimum: int) -> int:
    # bool is a subclass of int, so `repeats = true` would otherwise load as one repeat.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(
            f"profile {name!r}: {where} must be an integer, not {type(value).__name__}"
        )
    if value < minimum:
        raise ProfileError(
            f"profile {name!r}: {where} is {value}, below the minimum of {minimum}"
        )
    return value


def _int_list(value, where: str, name: str, *, minimum: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ProfileError(
            f"profile {name!r}: {where} must be a list, not {type(value).__name__}"
        )
    if not value:
        # An empty list is a value, not an absence, so it passes an `is None` guard and
        # drops a whole arm of the matrix from a report that still exits 0.
        raise ProfileError(
            f"profile {name!r}: {where} is an empty list; write at least one entry"
        )
    return tuple(_int(item, f"{where} entry", name, minimum=minimum) for item in value)


def load(name: str, service: str) -> Profile:
    """Read a profile, validate it, and resolve it against one service.

    Service-specific overrides exist because the same wall-clock budget buys very
    different workloads on a 5090 and on a CPU decoding an MoE at 35 tokens/second. The
    override is recorded in the profile, not passed on the command line, so a run's
    workload is reproducible from the file and the service it was resolved for, and
    `Profile.workload_hash` digests that resolved result.

    Every malformation raises `ProfileError`, which both command handlers catch and grade
    as invalid usage (exit 2). A hand-edited profile is the designed input, so a missing
    key, a scalar where a list belongs, or a repeat count of zero is a usage error to
    report -- not a traceback, and not a value that fails later from inside the run loop,
    after the health gate and after an arm's container recreate.
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
    except OSError as exc:
        raise ProfileError(f"profile {name!r} at {path} is unreadable: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(
            f"profile {name!r} at {path} is not valid TOML: {exc}"
        ) from exc

    declared = _require(document, "name", "", name)
    if not isinstance(declared, str) or not declared:
        raise ProfileError(f"profile {name!r}: 'name' must be a non-empty string")
    repeats = _int(_require(document, "repeats", "", name), "repeats", name, minimum=1)
    warmup = _int(_require(document, "warmup", "", name), "warmup", name, minimum=0)
    order = document.get("order", INTERLEAVED)
    if order not in ORDERS:
        raise ProfileError(
            f"profile {name!r}: order {order!r} is not one of {list(ORDERS)}; an "
            "unrecognised value would otherwise run blocked, which is the ordering "
            "the protocol exists to forbid"
        )
    seed = _int(document.get("seed", 42), "seed", name, minimum=0)

    services = _table(document.get("concurrency", {}), "'concurrency'", name)
    if service not in services:
        raise ProfileError(
            f"profile {name!r} states no concurrency for service {service!r}; "
            "every benchable service needs an entry, even if it is [1]"
        )
    concurrency = _int_list(
        services[service], f"concurrency.{service}", name, minimum=1
    )

    # Overrides are checked for every service, not only the one being resolved: a
    # misspelled key or service matches nothing and silently leaves the base sweep in
    # place, which on llama-cpu means an 8192-token prefill inside a serialized window.
    for over_service, entries in _table(
        document.get("overrides", {}), "'overrides'", name
    ).items():
        if over_service not in services:
            raise ProfileError(
                f"profile {name!r}: [overrides.{over_service}] names no service in the "
                f"concurrency table ({sorted(services)}); nothing would read it"
            )
        unknown = sorted(
            set(_table(entries, f"[overrides.{over_service}]", name)) - set(OVERRIDABLE)
        )
        if unknown:
            raise ProfileError(
                f"profile {name!r}: [overrides.{over_service}] sets {unknown}, which "
                f"nothing reads; overridable keys are {list(OVERRIDABLE)}"
            )
    overrides = document.get("overrides", {}).get(service, {})

    prefill = _table(_require(document, PREFILL, "", name), f"[{PREFILL}]", name)
    decode = _table(_require(document, DECODE, "", name), f"[{DECODE}]", name)

    prompt_tokens = _int_list(
        overrides.get(
            "prefill_prompt_tokens",
            _require(prefill, "prompt_tokens", "[prefill] ", name),
        ),
        "prefill prompt_tokens",
        name,
        minimum=MIN_PROMPT_TOKENS,
    )
    if len(set(prompt_tokens)) < 2:
        raise ProfileError(
            f"profile {name!r} gives service {service!r} fewer than two distinct "
            "prefill prompt lengths; a prefill slope cannot be fitted through one "
            "point, and a single-point prefill rate charges the fixed per-request "
            "overhead to the model"
        )

    return Profile(
        name=declared,
        repeats=repeats,
        warmup=warmup,
        order=order,
        seed=seed,
        prefill_prompt_tokens=prompt_tokens,
        prefill_max_tokens=_int(
            _require(prefill, "max_tokens", "[prefill] ", name),
            "prefill max_tokens",
            name,
            minimum=1,
        ),
        decode_prompt_tokens=_int(
            overrides.get(
                "decode_prompt_tokens",
                _require(decode, "prompt_tokens", "[decode] ", name),
            ),
            "decode prompt_tokens",
            name,
            minimum=MIN_PROMPT_TOKENS,
        ),
        decode_max_tokens=_int_list(
            overrides.get(
                "decode_max_tokens", _require(decode, "max_tokens", "[decode] ", name)
            ),
            "decode max_tokens",
            name,
            minimum=1,
        ),
        concurrency=concurrency,
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
    """A prompt of `target_tokens * CHARS_PER_TOKEN` characters, led by `nonce`.

    The nonce leads the prompt rather than trailing it, and that placement is the point:
    a prefix cache keys on the leading tokens, so a distinct prefix defeats it while a
    distinct suffix would not. vLLM's prefix caching is a launch flag with no per-request
    override, so this is the only way to be sure a measured prefill actually happened.

    Distinctness is the caller's to supply. The prompt is a pure function of the three
    arguments, so two callers passing the same nonce -- two arms replaying the same cell
    index, for instance -- send byte-identical prompts and the second one hits the cache
    this prefix was meant to defeat.

    A region shorter than the budget is repeated. An empty region, or a target too small
    to hold the nonce header, is refused rather than returning a prompt that is only the
    nonce.
    """
    if not region:
        raise ValueError("region is empty: there is no prose to build a prompt from")
    budget = target_tokens * CHARS_PER_TOKEN
    head = f"[{nonce}] "
    if budget <= len(head):
        raise ValueError(
            f"target_tokens={target_tokens} leaves {budget} characters, which the "
            f"{len(head)}-character nonce header fills: the prompt would be the nonce "
            "alone rather than the prose the measurement is of"
        )
    body = (region * (budget // len(region) + 1))[: budget - len(head)]
    return head + body
