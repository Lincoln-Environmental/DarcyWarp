"""Report and enforce Sphinx-style parameter documentation coverage."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "DARCY_WARP_PACKAGE"
ENFORCED_MODULES = {
    "benchmark_support.py",
    "canterbury_case_study/map_helpers.py",
    "physics/drn_2d.py",
    "physics/inputs_3d.py",
    "physics/operator_data.py",
    "physics/riv_2d.py",
    "physics/storage_2d.py",
    "process_mass_balance.py",
    "project_base.py",
    "solvers/transient_unconfined.py",
    "solvers/device_picard_periods.py",
    "solvers/classic_kcycle.py",
    "solvers/multigrid_kcycle.py",
    "solvers/picard_unconfined.py",
}
PARAM_FIELD = re.compile(r"^\s*:param\s+\**([A-Za-z_]\w*)\s*:\s*(.*)$", re.MULTILINE)


def parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Collect formal argument names, excluding implicit method receivers.

    :param node: Parsed function or method definition.
    :returns: Names of positional, keyword-only and variadic arguments.
    """
    arguments = node.args
    names = {arg.arg for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs}
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    names.discard("self")
    names.discard("cls")
    return names


def audit_module(path: Path) -> tuple[int, int, list[str]]:
    """Inspect all functions, including private helpers and decorated kernels.

    :param path: Python module to inspect without importing it.
    :returns: Total parameterised functions, fully documented functions and defects.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parameterised = 0
    complete = 0
    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = parameter_names(node)
        docstring = ast.get_docstring(node) or ""
        fields = PARAM_FIELD.findall(docstring)
        counts = Counter(name for name, description in fields if description.strip())
        missing = sorted(names - counts.keys())
        extra = sorted(counts.keys() - names)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if names:
            parameterised += 1
            if docstring and not (missing or extra or duplicates):
                complete += 1
        problems = []
        if not docstring:
            problems.append("missing docstring")
        if missing:
            problems.append("missing fields: " + ", ".join(missing))
        if extra:
            problems.append("unknown fields: " + ", ".join(extra))
        if duplicates:
            problems.append("duplicate fields: " + ", ".join(duplicates))
        if problems:
            issues.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} {node.name}: " + "; ".join(problems))
    return parameterised, complete, issues


def main(strict: bool = False) -> int:
    """Print full-package coverage and enforce audited modules.

    :param strict: Require complete documentation in all package modules.
    :returns: Process status, nonzero if a required module has defects.
    """
    total = 0
    covered = 0
    failures = []
    seen_modules = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        seen_modules.add(relative)
        count, complete, issues = audit_module(path)
        total += count
        covered += complete
        if issues:
            print(f"{relative}: {complete}/{count} complete parameter lists; {len(issues)} function defects")
        if strict or relative in ENFORCED_MODULES:
            failures.extend(issues)
    for relative in sorted(ENFORCED_MODULES - seen_modules):
        failures.append(f"Missing audited module: {relative}")
    print(f"Complete parameter documentation: {covered}/{total}")
    for failure in failures:
        print(failure)
    return int(bool(failures))


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--strict"]):
        raise SystemExit("Usage: python working_tests/audit_docstrings.py [--strict]")
    raise SystemExit(main(strict="--strict" in sys.argv[1:]))
