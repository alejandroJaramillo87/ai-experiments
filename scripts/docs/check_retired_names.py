#!/usr/bin/env python3
"""Fail the quality gate when a retired name is still in use anywhere in the tree.

check_doc_refs.py catches a document naming a path that does not exist. It cannot
catch a *renamed* file, because the classifier requires a slash: a bare
`docker-compose.yaml` in prose is not a path claim and is never checked.
Loosening that rule was measured and rejected -- accepting extension-bearing
tokens with no slash produced 361 new failures, because prose legitimately says
`config.json`, `main.rs`, and `manifest.json`.

So this takes the opposite approach. Rather than infer which tokens are claims,
it holds an explicit table of names that have been retired, and fails on any
occurrence of one. That has no false positives by construction, and it names the
replacement in the error, so the fix is mechanical rather than archaeological.

The table is the point. Adding an entry when you rename something is what makes
the next rename self-policing; a rename that skips this file is a rename whose
stragglers will be found by a human months later, which is the failure mode the
whole documentation campaign exists to end.

Unlike check_doc_refs.py this scans code as well as documentation, because a
retired filename hides in shell recipes, Python, Dockerfiles, and templates just
as readily as in prose -- and there it is a broken command, not a stale sentence.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

# Retired name -> (replacement, why it was retired). Both halves are shown in the
# failure, so the reason should read as an explanation to whoever hits it.
RETIRED: dict[str, tuple[str, str]] = {
    "docker-compose.yaml": (
        "docker-compose.inference.yaml",
        "the inference compose file gained a domain suffix, matching its four "
        "siblings (agents, interp, training, rust-harness); a bare `docker "
        "compose` with no -f no longer finds it",
    ),
    "docs/precision.md": (
        "docs/precision/README.md",
        "precision grew a second document -- the policy (what each tier does) "
        "and the decision record (why, and what would have to change) are "
        "different obligations and drift apart when filed together -- so the "
        "single file became a directory with an index",
    ),
    "docs/interpretability": (
        "docs/interp",
        "the docs directory was the last place the long form survived; the "
        "package, the container, the compose file, the make include, the tests, "
        "and the CLI are all `interp`. Keyed on the path prefix, not the bare "
        "word, which stays correct English throughout the prose",
    ),
    "docs/observability/roadmap.md": (
        "docs/observability/design-and-roadmap.md",
        "the operator-facing specification graduated into "
        "docs/observability/README.md, leaving the design rationale and the "
        "OBS item ledger; `roadmap.md` no longer described what the file "
        "held, and a bare `roadmap.md` says nothing about which roadmap",
    ),
    "docs/verification/gemma-4-conformance.md": (
        "docs/verification/investigations/gemma-4-conformance_investigation.md",
        "the audit was always a frozen record — audit-only, bounded by a "
        "Limitations section, every finding a quote at a line — but carried no "
        "class marker and sat among the runbooks, contradicting this "
        "directory's claim that every document in it is executed. The style "
        "guide requires a frozen record be findable by path; the new location "
        "satisfies both patterns instead of relying on the docs/verification/ "
        "exemption, which exists for operational runbooks",
    ),
    "./datasets": (
        "./scripts/benchmark/datasets",
        "the llama-cpu mount pointed at a repo-root directory that never held "
        "the corpus -- the committed Wikitext-2 copy has always been under "
        "scripts/benchmark/ -- so test_perplexity.sh died at its in-container "
        "check, and Docker creating the empty root directory made it a "
        "namespace package that shadowed the `datasets` library for mypy. "
        "Keyed with the ./ prefix so the bare word, which is the Python "
        "package throughout pyproject.toml and the smoke tests, stays legal",
    ),
    "gpu_benchmark_results": (
        "/mnt/ai-data/benchmarks/nsys (nsys reports) or "
        "/mnt/ai-data/benchmarks/gpu (BENCH_RESULTS_DIR)",
        "benchmark output is run data, and run data lives on the data SSD "
        "beside models, logs, and training runs rather than in a worktree that "
        "a branch switch can carry away. The directory had two creators -- the "
        "nsys bind mounts and a CWD-relative mkdir in benchmark_gpu.sh -- so "
        "removing either alone would not have stopped it reappearing",
    ),
    "perplexity_results": (
        "/mnt/ai-data/benchmarks/perplexity (BENCH_RESULTS_DIR)",
        "the sibling of gpu_benchmark_results in test_perplexity.sh, retired "
        "for the same reason and at the same time",
    ),
    "./outputs/comfyui": (
        "/mnt/ai-data/comfyui/output",
        "generated images and video are multi-GB data with no reason to sit "
        "next to source; the mount put them one `git add -A` away from a "
        "commit, and .gitignore had no entry for the directory",
    ),
    # The GGUF-to-checkpoint move (rust-harness P2-P8). Keys are qualified,
    # because gguf.rs and the two artifact readers survive: the Python interp
    # exporters write probe and control-vector images in that container, and
    # llama.cpp consumes the control vector. A bare `from_gguf` or `gguf.rs`
    # key would fire on live code.
    "CpuModel::from_gguf": (
        "CpuModel::from_checkpoint",
        "the CPU backend loads config.json and model.safetensors instead of a "
        "GGUF; the converter's metadata could only imply what a checkpoint "
        "states, and a SentencePiece-unigram reading of a BPE vocabulary is "
        "what that cost the project once already",
    ),
    "ModelSpec::from_gguf": (
        "ModelSpec::from_checkpoint",
        "retired with the loader above; layer_types, partial_rotary_factor and "
        "global_head_dim are named config.json fields rather than a stride, a "
        "divisor tensor, and a key that does not exist",
    ),
    "Config::from_gguf": (
        "ModelSpec::from_checkpoint",
        "folded into the spec constructor with the loader above; Config no "
        "longer carries sliding_window_pattern, which was a GGUF spelling "
        "nothing outside that function read",
    ),
    "cpu::check_supported": (
        "cpu::check_supported_checkpoint",
        "the half that policed GGUF metadata namespaces went with the "
        "container -- named config.json fields are validated by "
        "TextConfig::validate -- and the architectural refusals it kept are "
        "now expressed over Hugging Face tensor names",
    ),
    "write_probe_gguf": (
        "write_probe",
        "the probe left GGUF for a safetensors image plus a JSON sidecar, so "
        "the container is no longer part of the function's identity; the "
        "control-vector exporter keeps its _gguf name because llama.cpp reads "
        "that artifact and the format is not this repository's to choose",
    ),
    "export_gguf_fixtures.py": (
        "export_interp_fixtures.py",
        "the generator writes one GGUF and one safetensors pair, so naming it "
        "after the container it no longer exclusively produces would send a "
        "reader looking for a probe image that is not there",
    ),
    "python-probe.gguf": (
        "python-probe.safetensors",
        "the committed probe fixture follows its format; note that this file "
        "also served tests/models/conftest.py as a generic small GGUF, which "
        "now uses python-control-vector.gguf",
    ),
    "gguf-info": (
        "checkpoint-info",
        "the inspection tool follows the format the backend loads; it echoes "
        "config.json and tabulates safetensors dtypes and shapes rather than "
        "reconstructing both from metadata keys and ggml type ids",
    ),
    "docs/rust-harness/gemma-4-ingestion.md": (
        "docs/models/gemma-4.md",
        "the document described the GGUF ingestion contract, and the contract "
        "died with the loader; the family facts that outlived it -- the "
        "rope_freqs divisor tensor and the per_layer naming that conversion "
        "drops -- moved to the model reference, which documents the published "
        "files llama.cpp still serves",
    ),
    "gguf_info_check.rs": (
        "checkpoint_info_check.rs",
        "renamed with the binary it tests; the --check exit-code contract it "
        "pins is unchanged",
    ),
    # The Tier 3 acceptance stubs, deleted rather than ported (rust-harness
    # P9.7). Both were #[ignore] + unimplemented!() against things that no
    # longer exist -- a `cuda` feature, llama.cpp, a quantized GGUF -- and
    # their fixture designs were the reason not to port them: perplexity
    # against llama-perplexity on a quantized GGUF compares two different
    # models, and goldens minted from our own first run pin whatever the
    # implementation did that day.
    "cuda_acceptance.rs": (
        "runtime/src/ladder.rs (rungs L1 and L2)",
        "the correctness ladder measures the residual stream and the output "
        "distribution against transformers, on the real checkpoint, with no "
        "cuda feature involved; the stub had waited for a backend that was "
        "never built",
    ),
    "eval_acceptance.rs": (
        "runtime/src/ladder.rs (rungs L3 and L4)",
        "perplexity and greedy decoding are ladder rungs against "
        "transformers' own numbers rather than against llama.cpp on a "
        "quantized GGUF and a golden generated by ourselves; recipes moved to "
        "rust-harness/fixtures/reference/README.md",
    ),
    # The tests/utils/ split. Keyed per file rather than on the directory:
    # tests/utils/ still exists and still holds the scripts/utils/ reporters, so
    # a bare `tests/utils` key would fire on the files that correctly stayed.
    **{
        f"tests/utils/{name}": (
            f"tests/{dest}/{name}",
            "tests/utils/ covers the scripts/utils/ host reporters, as its "
            "README has always said; the common/ and scripts/docs/ suites had "
            "accumulated there against that scope and moved to directories "
            "that mirror what they test",
        )
        for name, dest in (
            ("test_obs_audit.py", "common"),
            ("test_obs_catalog.py", "common"),
            ("test_obs_context.py", "common"),
            ("test_obs_core.py", "common"),
            ("test_obs_dispatch.py", "common"),
            ("test_obs_failure.py", "common"),
            ("test_obs_record_contract.py", "common"),
            ("test_obs_retention.py", "common"),
            ("test_obs_serve.py", "common"),
            ("test_common_output.py", "common"),
            ("test_doc_refs.py", "docs"),
            ("test_retired_names.py", "docs"),
        )
    },
}

# Directories holding no text of ours.
PRUNE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "target",
    "__pycache__",
    ".ruff_cache",
    # pytest's tmp_path base when the default temp dir is noexec (the agent
    # sandbox; see tests/conftest.py). Generated fixtures, not our text.
    ".pytest-tmp",
}

# Extensions worth reading. Anything textual that can name a file.
SCAN_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".mk",
    ".rs",
    ".json",
    ".cfg",
    ".ini",
    ".example",
    ".txt",
}

# Extensionless or oddly-named files that still carry commands.
SCAN_NAMES = {"Makefile", "Dockerfile"}
SCAN_NAME_PREFIXES = ("Dockerfile.",)

# Frozen by policy: investigation write-ups record the tree as it was on the day
# they were written, so naming the old file is accurate history.
# Same exemption set as check_doc_refs.py, for the same reason -- including the
# empty EXEMPT_PREFIXES, where that script records why docs/verification/ stopped
# being exempt on 2026-08-08. This gate has no false positives by construction,
# so narrowing it was the lower-risk half of that change.
EXEMPT_PARTS = ("investigations", "vendor")
# Agent memory records carry a `verified:` date and a `stale_when:` condition:
# they state what was true when written, and a record whose condition has fired
# is closed rather than rewritten. Renaming the symbols inside one would make it
# claim to have observed something it did not.
EXEMPT_PREFIXES: tuple[str, ...] = (".claude/agent-memory/",)
EXEMPT_SUFFIXES = ("_investigation.md",)

# Exempt by name, each for a stated reason.
EXEMPT_FILES = {
    # This file: it defines the table, so it necessarily spells every retired name.
    "scripts/docs/check_retired_names.py",
    # The unit tests, which assert on the table and so quote its keys.
    "tests/docs/test_retired_names.py",
    # The sibling gate. Its KNOWN_ABSENT table is a curated list of references
    # that name something absent on purpose, so a deleted file's name is
    # exactly what an entry is made of -- the same argument this file makes
    # for itself one line above. Narrow in practice: the table is a few dozen
    # audited pairs, each carrying its own reason.
    "scripts/docs/check_doc_refs.py",
    # A dated QA campaign ledger. Its findings record what was run against
    # which function at the time, so rewriting the names in them would
    # falsify the record rather than update it -- the same reason
    # investigations/ and *_investigation.md are exempt.
    "docs/verification/rust-harness-qa.md",
}


def is_exempt(rel_path: str) -> bool:
    """Whether a file is outside this gate's scope."""
    parts = Path(rel_path).parts
    return (
        any(p in EXEMPT_PARTS for p in parts)
        or rel_path.startswith(EXEMPT_PREFIXES)
        or rel_path.endswith(EXEMPT_SUFFIXES)
        or rel_path in EXEMPT_FILES
    )


def wants_scan(name: str) -> bool:
    """Whether a filename looks like text that could name a file."""
    return (
        Path(name).suffix in SCAN_SUFFIXES
        or name in SCAN_NAMES
        or name.startswith(SCAN_NAME_PREFIXES)
    )


def iter_files(root: Path) -> list[Path]:
    """Every in-scope text file under root."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for name in filenames:
            if not wants_scan(name):
                continue
            path = Path(dirpath) / name
            if not is_exempt(str(path.relative_to(root))):
                found.append(path)
    return sorted(found)


def pattern_for(old: str) -> re.Pattern[str]:
    """Match `old` as a whole name rather than as a fragment of a longer one.

    The trailing guard deliberately permits a following `.` so a sentence ending
    in the name still matches; the cost is that a hypothetical `<old>.bak` would
    also match, which is the rarer and more tolerable error.
    """
    return re.compile(rf"(?<![\w.-]){re.escape(old)}(?![\w-])")


def find_retired(
    text: str, patterns: dict[str, re.Pattern[str]]
) -> list[tuple[int, str]]:
    """Every (line number, retired name) occurrence in text."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for old, pattern in patterns.items():
            if pattern.search(line):
                hits.append((lineno, old))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    if not RETIRED:
        out.ok("retired-names: no names are currently retired")
        return 0

    files = iter_files(root)
    # The shellcheck step's empty-set guard, for the same reason: a gate that
    # silently checked nothing is indistinguishable from one that passed.
    if not files:
        out.error(f"retired-names: enumerated zero files under {root}")
        out.line("  The gate would pass without checking anything.")
        return 1

    patterns = {old: pattern_for(old) for old in RETIRED}
    hits: list[tuple[Path, int, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Not text after all; nothing to check.
            continue
        for lineno, old in find_retired(text, patterns):
            hits.append((path, lineno, old))

    out.info(f"retired-names: {len(files)} files, {len(RETIRED)} retired names")
    if not hits:
        out.ok("retired-names: no retired name is still in use")
        return 0

    out.error(f"retired-names: {len(hits)} references use a retired name")
    for path, lineno, old in hits:
        replacement, _ = RETIRED[old]
        out.line(f"  {path.relative_to(root)}:{lineno}: {old} -> {replacement}")
    for old, (replacement, reason) in RETIRED.items():
        out.line(f"  {old} was replaced by {replacement}: {reason}")
    out.line("  Solution: use the replacement name.")
    out.line("  A frozen dated record may instead belong under an exemption.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
