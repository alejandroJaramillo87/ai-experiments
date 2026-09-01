"""Collapsing a corpus to its distinct call sites.

Three hundred thousand log lines are a few hundred distinct events repeated. The
collapse is what makes a corpus describable inside a context window, and for
mongod it is exact rather than inferred: every line carries a numeric `id`
identifying the call site that wrote it, paired with a fixed `msg` string. So
(component, id) is the template identity, with no clustering, no similarity
threshold, and no tuning.

That is worth stating plainly because the obvious tool for this job -- Drain and
its descendants -- exists to recover templates from logs that do not have them.
Pointing it at mongod output would pay clustering error for something the format
already provides, and two runs could disagree. It earns its place only on the
genuinely free-text fields: attr.errmsg and attr.error carry driver and storage
messages with hostnames, counts and durations inlined, and those do need
clustering. That is what cluster_free_text is for, and it is the only thing
drain3 is used for here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from logtriage.mongod.records import LogRecord


class TemplatableRecord(Protocol):
    """What templating needs from a record, and nothing more.

    Both LogRecord and NormalizedRecord satisfy this. Stating the requirement
    structurally rather than naming one of them keeps the index usable on either
    side of normalization -- the alternative was a cast at the call site, which
    would have asserted the same compatibility without checking it.

    Every member is a read-only property rather than a bare annotation. A bare
    annotation in a Protocol declares a *settable* attribute, which no frozen
    dataclass can satisfy -- so the obvious spelling rejects both of the two
    types this exists to accept.
    """

    @property
    def ts(self) -> datetime: ...

    @property
    def severity(self) -> str: ...

    @property
    def component(self) -> str: ...

    @property
    def log_id(self) -> int: ...

    @property
    def msg(self) -> str: ...

    @property
    def record_id(self) -> str: ...


# Severity rank, worst first, for reporting the most severe severity a template
# was ever seen at. mongod emits the same call site at one severity in practice,
# but a template that is usually I and occasionally E is exactly the thing worth
# surfacing, so the index records the worst rather than the last.
_SEVERITY_RANK = {
    "F": 0,
    "E": 1,
    "W": 2,
    "I": 3,
    "D1": 4,
    "D2": 5,
    "D3": 6,
    "D4": 7,
    "D5": 8,
}

# Example record ids retained per template. Enough to cite a few occurrences
# from a profile without the index growing with the corpus.
MAX_EXAMPLES = 3

# Free-text attr fields worth clustering. Everything else in attr is either a
# counter or a structured sub-document, and clustering those would produce
# templates over numbers.
FREE_TEXT_FIELDS = ("errmsg", "error")


def template_key(record: TemplatableRecord) -> str:
    """The exact template identity of a log line.

    Component as well as id because ids are unique per call site but a reader
    scanning a profile needs the subsystem without a lookup table.
    """
    return f"{record.component}:{record.log_id}"


@dataclass(frozen=True, slots=True)
class TemplateStat:
    """One distinct call site and what it did across the corpus."""

    template: str
    component: str
    log_id: int
    msg: str
    severity: str
    count: int
    first_seen: datetime
    last_seen: datetime
    examples: tuple[str, ...]


class TemplateIndex:
    """Counts distinct call sites and remembers a few examples of each."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def add(self, record: TemplatableRecord) -> None:
        identity = template_key(record)
        entry = self._entries.get(identity)
        if entry is None:
            self._entries[identity] = {
                "component": record.component,
                "log_id": record.log_id,
                "msg": record.msg,
                "severity": record.severity,
                "count": 1,
                "first_seen": record.ts,
                "last_seen": record.ts,
                "examples": [record.record_id],
            }
            return
        entry["count"] += 1
        if record.ts < entry["first_seen"]:
            entry["first_seen"] = record.ts
        if record.ts > entry["last_seen"]:
            entry["last_seen"] = record.ts
        if _SEVERITY_RANK.get(record.severity, 99) < _SEVERITY_RANK.get(
            entry["severity"], 99
        ):
            entry["severity"] = record.severity
        if len(entry["examples"]) < MAX_EXAMPLES:
            entry["examples"].append(record.record_id)

    def extend(self, records: Iterable[TemplatableRecord]) -> None:
        for record in records:
            self.add(record)

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> list[TemplateStat]:
        """Every template, most frequent first.

        Ties break on the template identity so the ordering is total, and a
        profile of the same corpus is byte-identical between runs.
        """
        out = [
            TemplateStat(
                template=identity,
                component=entry["component"],
                log_id=entry["log_id"],
                msg=entry["msg"],
                severity=entry["severity"],
                count=entry["count"],
                first_seen=entry["first_seen"],
                last_seen=entry["last_seen"],
                examples=tuple(entry["examples"]),
            )
            for identity, entry in self._entries.items()
        ]
        out.sort(key=lambda stat: (-stat.count, stat.template))
        return out


@dataclass(frozen=True, slots=True)
class FreeTextCluster:
    """A cluster of free-text messages that differ only in their variables."""

    template: str
    count: int
    examples: tuple[str, ...]


def free_text_values(record: LogRecord) -> list[str]:
    """Pull the free-text messages out of one record's attr.

    attr.error is sometimes a string and sometimes a sub-document carrying
    errmsg/codeName; both forms appear across components, so both are handled
    rather than one being assumed.
    """
    out: list[str] = []
    for field_name in FREE_TEXT_FIELDS:
        value = record.attr.get(field_name)
        if isinstance(value, str) and value:
            out.append(value)
        elif isinstance(value, dict):
            nested = value.get("errmsg")
            if isinstance(nested, str) and nested:
                out.append(nested)
    return out


def cluster_free_text(
    messages: Iterable[str], *, max_clusters: int = 256
) -> list[FreeTextCluster]:
    """Cluster free-text messages with Drain.

    The config is constructed explicitly rather than left to default, which
    would make drain3 look for a drain3.ini beside the working directory: a
    result that changed depending on where the tool was invoked from would be
    untraceable, and this tool's whole contract is that a corpus profiles the
    same way twice.

    max_clusters bounds memory on a pathological corpus. Drain evicts
    least-recently-used clusters at the cap, so the common templates -- the ones
    a profile reports -- are the ones that survive.

    Counting keys on the cluster id and the template is read back at the end,
    which is not incidental. Drain generalizes a cluster as it sees more members,
    and add_log_message returns the template *as of that call* -- so the first
    message of a cluster comes back with its literal text and later ones with the
    wildcards. Grouping on the returned template therefore splits nearly every
    cluster in two, with the first occurrence stranded on its own.
    """
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig

    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4
    config.drain_depth = 4
    config.drain_max_children = 100
    config.drain_max_clusters = max_clusters
    config.mask_prefix = "<"
    config.mask_suffix = ">"
    config.profiling_enabled = False
    # No persistence handler: state must not survive a run, or the second
    # profile of a corpus would differ from the first.
    miner = TemplateMiner(config=config)

    examples: dict[int, list[str]] = {}
    for message in messages:
        result = miner.add_log_message(message)
        bucket = examples.setdefault(int(result["cluster_id"]), [])
        if len(bucket) < MAX_EXAMPLES:
            bucket.append(message)

    clusters = [
        FreeTextCluster(
            template=cluster.get_template(),
            count=cluster.size,
            examples=tuple(examples.get(cluster.cluster_id, ())),
        )
        for cluster in miner.drain.clusters
    ]
    clusters.sort(key=lambda cluster: (-cluster.count, cluster.template))
    return clusters
