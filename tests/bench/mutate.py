"""Mutation testing for the two modules whose job is to not report a false number.

BQ.8. Coverage says a line was executed. It does not say a test would notice the line
being wrong, and for `bench/metrics.py` and `bench/delta.py` that is the entire
question: both exist to turn timings into a claim, and a claim nobody checks is worse
than no claim.

The method is the standard one. Parse the module, apply one small change, run the
suite, and record whether anything failed. A mutant the suite still passes is a
statement the tests do not check -- either dead code, or an assertion that pins a
shape rather than a value.

No mutation-testing library is installed and adding one means regenerating the
lockfile, which a sandbox cannot do, so the operators below are the subset that pays
for itself in eighty lines: comparison direction, boundary inclusion, arithmetic sign,
and the numeric constants that encode thresholds.

Not a pytest module. There is no pass mark; the score is the deliverable.

    python3 tests/bench/mutate.py [--target bench/delta.py] [--json]
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

TARGETS = ("bench/metrics.py", "bench/delta.py")

# Each entry swaps one operator for its neighbour. The pairs are chosen so the mutant
# stays type-correct and plausible: a test that survives `<` becoming `<=` is not
# checking the boundary, which is exactly where a threshold defect lives.
COMPARE_SWAPS = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}
BINOP_SWAPS = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
}


@dataclasses.dataclass
class Mutant:
    target: str
    line: int
    description: str
    source: str


def _mutants(path: pathlib.Path) -> list[Mutant]:
    original = path.read_text()
    tree = ast.parse(original)
    found: list[Mutant] = []

    # Index every mutable node once, then rebuild the tree per mutant. Mutating in
    # place and undoing is faster but leaves the harness one bug away from emitting a
    # mutant that carries two changes, which would make a survivor uninterpretable.
    sites: list[tuple[str, int, str]] = []
    for index, node in enumerate(ast.walk(tree)):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = type(node.ops[0])
            if op in COMPARE_SWAPS:
                sites.append(
                    ("compare", index, f"{op.__name__} to {COMPARE_SWAPS[op].__name__}")
                )
        elif isinstance(node, ast.BinOp) and type(node.op) in BINOP_SWAPS:
            op = type(node.op)
            sites.append(
                ("binop", index, f"{op.__name__} to {BINOP_SWAPS[op].__name__}")
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                continue
            sites.append(("constant", index, f"{node.value!r} to {node.value + 1!r}"))

    for kind, index, description in sites:
        mutated = ast.parse(original)
        nodes = list(ast.walk(mutated))
        node = nodes[index]
        line = getattr(node, "lineno", 0)
        if kind == "compare":
            node.ops = [COMPARE_SWAPS[type(node.ops[0])]()]
        elif kind == "binop":
            node.op = BINOP_SWAPS[type(node.op)]()
        else:
            node.value = node.value + 1
        found.append(
            Mutant(
                target=str(path.relative_to(REPO_ROOT)),
                line=line,
                description=description,
                source=ast.unparse(ast.fix_missing_locations(mutated)),
            )
        )
    return found


def _run_suite() -> bool:
    """True when the suite passes, meaning the mutant survived."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/bench",
            "-q",
            "-x",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mutation-test the bench reduction.")
    parser.add_argument("--target", action="append", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    targets = args.target or list(TARGETS)
    out = Printer(quiet=args.json)
    out.line("=== bench mutation baseline ===")
    out.line()

    results = []
    for relative in targets:
        path = REPO_ROOT / relative
        if not path.exists():
            out.error(f"Target missing: {relative}")
            return 2
        original = path.read_text()
        mutants = _mutants(path)
        out.run(f"{relative}: {len(mutants)} mutants")
        survived = []
        try:
            for mutant in mutants:
                path.write_text(mutant.source)
                if _run_suite():
                    survived.append(mutant)
        finally:
            path.write_text(original)

        killed = len(mutants) - len(survived)
        score = killed / len(mutants) * 100 if mutants else 100.0
        results.append(
            {
                "target": relative,
                "mutants": len(mutants),
                "killed": killed,
                "survived": [
                    {"line": m.line, "change": m.description} for m in survived
                ],
                "score_pct": round(score, 1),
            }
        )
        tag = out.ok if not survived else out.warn
        tag(f"{relative}: {killed} of {len(mutants)} killed ({score:.1f}%)", indent=2)
        for mutant in survived:
            out.line(f"    survived {relative}:{mutant.line}  {mutant.description}")

    out.line()
    total = sum(r["mutants"] for r in results)
    killed = sum(r["killed"] for r in results)
    out.info(f"Overall: {killed} of {total} killed ({killed / total * 100:.1f}%)")
    out.line("  A survivor is a gap in the tests, or an equivalent mutant that no")
    out.line("  test could catch. Check each by hand before filing it: centring a")
    out.line("  covariance on plus-mean rather than minus-mean survives because the")
    out.line("  deviations sum to zero, which is a fact about algebra, not coverage.")

    if args.json:
        print(json.dumps({"targets": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
