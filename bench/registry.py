"""Model architecture facts: which served model is dense, and which is an MoE.

The suite needs to know, because it changes both what the workload must look like and
what the resulting number means. Nothing else in the tree records it:
training/models/families.py's FamilySpec covers LoRA targets and chat templates, and the
only machine-readable dense/MoE distinction is a refusal in
training/calculator/components.py. The cost of leaving it unstated is on record --
docs/optimizations/README.md's headline CPU figure was measured against a
mixture-of-experts model and says so nowhere.

Why architecture changes a benchmark:

1. An MoE routes each token to k of n experts, and which experts -- and how evenly load
   spreads across them -- is a function of content. Two prompts of identical length can
   differ measurably in decode rate.
2. Uniform random token ids, which the retired upstream load generator produced, route
   near-uniformly across experts: the best case for load balance, the worst for
   expert-weight locality, wrong with no predictable sign. Hence real text, and hence
   three disjoint prompts on the decode arm so the spread between them is reported as
   the routing sensitivity.
3. A 30B-A3B model activates ~3B parameters per token. Its decode rate is comparable to
   a dense 3B on arithmetic while carrying a dense 30B's residency, so a claim stating
   only the total is wrong in both directions. Both counts are required for an MoE.

MODELS is a plain dict rather than a file under schemas/: there is one writer and one
reader, both here, and that directory holds cross-language contracts. Promote it to one
if a second language ever needs these facts.
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Iterable, Mapping, Sequence

from common.serving import SERVICES, quant_from_name

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

# Paths are the in-container paths the services see, matching the MODEL_PATH that
# docker-compose.inference.yaml passes -- not host paths under /mnt/ai-data/models/.
MODELS: dict[str, dict] = {
    "qwen3-coder-30b-a3b-IQ4_XS": {
        "path": (
            "/app/models/gguf/Qwen3-Coder-30B-A3B-Instruct-GGUF/"
            "Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf"
        ),
        "engines": ["llama-cpu", "llama-gpu"],
        "architecture": "moe",
        "params_total_b": 30.5,
        "params_active_b": 3.3,
        "n_experts": 128,
        "n_experts_active": 8,
        "quantization": "IQ4_XS",
        "context_length": 262144,
        # The model behind docs/optimizations/README.md's 26.2 percent claim, recorded
        # in docs/optimizations/bios/cpu_benchmark_results_final_bios.json. Listed so
        # that re-measurement is like-for-like.
    },
}


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


# llama.cpp's IQ- and TQ-family suffixes (IQ4_XS, IQ2_M, TQ1_0) are quantizations that
# common/serving.py's quant_from_name reports as unparseable: it accepts a suffix only
# when it starts with Q or names a float type. The registry ships an IQ4_XS model, so
# without these the confound guard below is inert on the one entry it exists for.
# Matched here rather than in the shared parser, whose other callers key off its
# narrower vocabulary.
_IQ_TQ_SUFFIX = re.compile(r"^(?:IQ\d+(?:_[A-Z0-9]+)*|TQ\d+_\d+)$")


def quantization_from_path(path: str) -> str | None:
    """Storage type parsed from a GGUF filename, or None when it states none."""
    parsed = quant_from_name(path)
    if parsed is not None:
        return parsed
    stem = os.path.basename(path)
    if not stem.endswith(".gguf"):
        return None
    tail = stem[: -len(".gguf")].rsplit("-", 1)
    if len(tail) == 2 and _IQ_TQ_SUFFIX.match(tail[1].upper()):
        return tail[1].upper()
    return None


def _number(slug: str, key: str, value: object) -> float:
    """A positive real. bool is excluded: it is an int in Python and a mistake here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryError(f"{slug}: {key} must be a number, got {value!r}")
    if value <= 0:
        raise RegistryError(f"{slug}: {key} must be positive, got {value!r}")
    return float(value)


def _count(slug: str, key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RegistryError(f"{slug}: {key} must be an integer, got {value!r}")
    if value <= 0:
        raise RegistryError(f"{slug}: {key} must be positive, got {value!r}")
    return value


def _validate(slug: str, entry: dict) -> None:
    missing = [key for key in REQUIRED if key not in entry]
    if missing:
        raise RegistryError(f"{slug}: missing required key(s) {missing}")
    if entry["architecture"] not in ARCHITECTURES:
        raise RegistryError(
            f"{slug}: architecture must be one of {sorted(ARCHITECTURES)}, "
            f"got {entry['architecture']!r}"
        )
    for key in ("path", "quantization"):
        if not isinstance(entry[key], str) or not entry[key]:
            raise RegistryError(
                f"{slug}: {key} must be a non-empty string, got {entry[key]!r}"
            )
    engines = entry["engines"]
    # A bare string is iterable, so tuple() would turn it into a per-character engine
    # list: advertised by `bench models` one letter at a time, and matching no service.
    if isinstance(engines, str) or not isinstance(engines, (list, tuple)):
        raise RegistryError(
            f"{slug}: engines must be a list of service names, got {engines!r}"
        )
    if not engines:
        raise RegistryError(
            f"{slug}: engines is empty, so the entry is advertised but resolves for no "
            "service"
        )
    unknown = [engine for engine in engines if engine not in SERVICES]
    if unknown:
        raise RegistryError(
            f"{slug}: engines names {unknown}, which is not among the services "
            f"{sorted(SERVICES)}"
        )
    total = _number(slug, "params_total_b", entry["params_total_b"])
    _count(slug, "context_length", entry["context_length"])
    if entry["architecture"] == "moe":
        missing = [key for key in MOE_REQUIRED if entry.get(key) is None]
        if missing:
            raise RegistryError(
                f"{slug}: an MoE entry must state {missing} -- a report carrying only "
                "the total parameter count invites a comparison that is wrong in both "
                "directions (see this module's docstring)"
            )
        # Internal consistency: an MoE activates a subset of its parameters and of its
        # experts, so either count exceeding its total is a transcription error that
        # would otherwise be published as an architecture fact.
        active = _number(slug, "params_active_b", entry["params_active_b"])
        experts = _count(slug, "n_experts", entry["n_experts"])
        active_experts = _count(slug, "n_experts_active", entry["n_experts_active"])
        if active > total:
            raise RegistryError(
                f"{slug}: params_active_b {active} exceeds params_total_b {total}"
            )
        if active_experts > experts:
            raise RegistryError(
                f"{slug}: n_experts_active {active_experts} exceeds n_experts {experts}"
            )
    # A quantization confound is the easiest way to publish an unfounded comparison,
    # and the filename parser already exists. Only checked when the name is parseable:
    # a BF16 export or a non-GGUF checkpoint returns None rather than disagreeing.
    parsed = quantization_from_path(entry["path"])
    if parsed is not None and parsed != entry["quantization"]:
        raise RegistryError(
            f"{slug}: quantization {entry['quantization']!r} disagrees with the model "
            f"filename, which parses as {parsed!r}"
        )


def load(models: dict[str, dict] | None = None) -> dict[str, ModelSpec]:
    """Every registry entry, validated. Raises on the first malformed one."""
    specs: dict[str, ModelSpec] = {}
    for slug, entry in (MODELS if models is None else models).items():
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


# Where a running service states the model it loaded. docker-compose.inference.yaml
# passes it as MODEL_PATH and the entrypoints read that, so the env var is the usual
# source; the flags cover a container recreated by hand or by a model-path arm override.
MODEL_ENV_KEYS = ("MODEL_PATH", "LLAMA_CPU_MODEL", "LLAMA_GPU_MODEL", "VLLM_GPU_MODEL")
MODEL_ARGV_FLAGS = ("-m", "--model", "--model-path")


def observed_model_paths(observed: Mapping | Iterable[str] | None) -> tuple[str, ...]:
    """Model paths a container states it is serving, from env and server argv.

    Accepts either the conditions mapping bench/conditions.py collects (env_knobs plus
    server_argv) or a bare argv sequence.
    """
    if observed is None:
        return ()
    if isinstance(observed, Mapping):
        env = observed.get("env_knobs") or {}
        argv = observed.get("server_argv") or []
    else:
        env, argv = {}, list(observed)
    found: list[str] = []
    for key in MODEL_ENV_KEYS:
        value = env.get(key)
        if value:
            found.append(str(value))
    argv = [str(token) for token in argv]
    for index, token in enumerate(argv):
        flag, sep, inline = token.partition("=")
        if flag not in MODEL_ARGV_FLAGS:
            continue
        value = inline if sep else (argv[index + 1] if index + 1 < len(argv) else "")
        if value:
            found.append(value)
    return tuple(dict.fromkeys(found))


def reconcile(spec: ModelSpec, observed: Mapping | Sequence[str] | None) -> str | None:
    """The disagreement between the registry's claim and what the service is serving.

    Returns None when they agree, and None when the container states no model at all:
    docker inspect can fail, and unobserved is not contradicted. Comparison is exact on
    the in-container path, which is the form MODELS records (see the MODELS comment).

    Reachable without any fault: the training commands swap models on these same
    services and skip their restore under --keep-serving, and a bench arm can override
    the model path. A report then carries conditions.model from the registry and
    conditions.server_argv from the container with nothing reconciling them.
    """
    paths = observed_model_paths(observed)
    if not paths:
        return None
    declared = os.path.normpath(spec.path)
    if any(os.path.normpath(path) == declared for path in paths):
        return None
    return (
        f"registry claims {spec.slug} is served from {spec.path}, but the container "
        f"states it is running {', '.join(paths)}; the report's conditions.model would "
        "describe a model that is not loaded"
    )


def resolve(slug: str, service: str) -> ModelSpec:
    """One model by slug, checked against the service that will serve it."""
    specs = load()
    if slug not in specs:
        raise RegistryError(
            f"unknown model {slug!r}; registered: {sorted(specs)} "
            "(add it to MODELS in bench/registry.py)"
        )
    spec = specs[slug]
    if service not in spec.engines:
        raise RegistryError(
            f"{slug} is registered for {list(spec.engines)}, not {service!r}"
        )
    return spec
