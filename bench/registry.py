"""Model architecture facts, loaded from schemas/bench_models.json.

The suite needs to know whether the model under test is dense or a mixture of experts,
because that changes both what the workload must look like and what the resulting
number means. Nothing else in the tree records it: training/models/families.py's
FamilySpec covers LoRA targets and chat templates, and the only machine-readable
dense/MoE distinction is a refusal in training/calculator/components.py.

The contract, including why architecture changes a benchmark, is in
schemas/bench_models.json; this module is its reader.
"""

from __future__ import annotations

import dataclasses
import json
import os

from common.serving import quant_from_name

CONTRACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "schemas",
    "bench_models.json",
)

ARCHITECTURES = frozenset({"dense", "moe"})
REQUIRED = (
    "path",
    "engines",
    "architecture",
    "params_total_b",
    "quantization",
    "context_length",
)
MOE_REQUIRED = ("params_active_b", "n_experts", "n_experts_active")


class RegistryError(ValueError):
    """A registry entry is missing a field the report is required to carry."""


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """One model's architecture facts, as the report records them."""

    slug: str
    path: str
    engines: tuple[str, ...]
    architecture: str
    params_total_b: float
    quantization: str
    context_length: int
    params_active_b: float | None = None
    n_experts: int | None = None
    n_experts_active: int | None = None

    @property
    def is_moe(self) -> bool:
        return self.architecture == "moe"

    @property
    def decode_prompts(self) -> int:
        """How many distinct natural prompts the decode arm should draw.

        An MoE's decode rate moves with which experts a prompt routes to, so one prompt
        repeated measures repeatability and not sensitivity. Three disjoint prompts make
        the spread between them a reported quantity rather than an unmeasured confound.
        A dense model does identical work per token, so one prompt is the whole story.
        """
        return 3 if self.is_moe else 1

    def conditions(self) -> dict:
        """The model block a bench_result report carries under conditions."""
        block = {
            "name": self.slug,
            "path": self.path,
            "architecture": self.architecture,
            "params_total_b": self.params_total_b,
            "quantization": self.quantization,
            "context_length": self.context_length,
        }
        if self.is_moe:
            block.update(
                params_active_b=self.params_active_b,
                n_experts=self.n_experts,
                n_experts_active=self.n_experts_active,
            )
        return block


def _validate(slug: str, entry: dict) -> None:
    missing = [key for key in REQUIRED if key not in entry]
    if missing:
        raise RegistryError(f"{slug}: missing required key(s) {missing}")
    if entry["architecture"] not in ARCHITECTURES:
        raise RegistryError(
            f"{slug}: architecture must be one of {sorted(ARCHITECTURES)}, "
            f"got {entry['architecture']!r}"
        )
    if entry["architecture"] == "moe":
        missing = [key for key in MOE_REQUIRED if entry.get(key) is None]
        if missing:
            raise RegistryError(
                f"{slug}: an MoE entry must state {missing} -- a report carrying only "
                "the total parameter count invites a comparison that is wrong in both "
                "directions (schemas/bench_models.json)"
            )
    # A quantization confound is the easiest way to publish an unfounded comparison,
    # and the filename parser already exists. Only checked when the name is parseable:
    # a BF16 export or a non-GGUF checkpoint returns None rather than disagreeing.
    parsed = quant_from_name(entry["path"])
    if parsed is not None and parsed != entry["quantization"]:
        raise RegistryError(
            f"{slug}: quantization {entry['quantization']!r} disagrees with the model "
            f"filename, which parses as {parsed!r}"
        )


def load(path: str = CONTRACT) -> dict[str, ModelSpec]:
    """Every registry entry, validated. Raises on the first malformed one."""
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    specs: dict[str, ModelSpec] = {}
    for slug, entry in document.get("models", {}).items():
        _validate(slug, entry)
        specs[slug] = ModelSpec(
            slug=slug,
            path=entry["path"],
            engines=tuple(entry["engines"]),
            architecture=entry["architecture"],
            params_total_b=entry["params_total_b"],
            quantization=entry["quantization"],
            context_length=entry["context_length"],
            params_active_b=entry.get("params_active_b"),
            n_experts=entry.get("n_experts"),
            n_experts_active=entry.get("n_experts_active"),
        )
    return specs


def resolve(slug: str, service: str, path: str = CONTRACT) -> ModelSpec:
    """One model by slug, checked against the service that will serve it."""
    specs = load(path)
    if slug not in specs:
        raise RegistryError(
            f"unknown model {slug!r}; registered: {sorted(specs)} "
            f"(add it to {os.path.relpath(path)})"
        )
    spec = specs[slug]
    if service not in spec.engines:
        raise RegistryError(
            f"{slug} is registered for {list(spec.engines)}, not {service!r}"
        )
    return spec
