# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run bounded 3D audit groups with explicit CPU/CUDA coverage reporting.

No numerical tolerances or solver controls are modified by this runner.
The timeout is a wall-clock test limit, not a convergence criterion.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def select_tests(*, group: str, directory: Path) -> list[Path]:
    groups = {
        "core": sorted(set(directory.glob("test_3d*.py"))
                       | set(directory.glob("test_fast_3d*.py"))
                       | {directory / "test_validation_repairs_3d.py", directory / "test_scientific_audit_regressions.py"}),
        "physics": [directory / "test_3d_physics_validation.py"],
        "npf": sorted(directory.glob("test_npf662*.py")),
        "terrain": sorted(directory.glob("test_terrain*.py")),
        "2d-regression": [directory / name for name in (
            "test_sparse_system_consistency.py", "test_sanity_case_config.py",
            "test_model_convergence_mf6_cache.py", "test_solver_registry_2d.py",
            "test_2d_transient.py", "test_nonlinear_operator_2d.py")],
    }
    groups["core"].remove(directory / "test_3d_physics_validation.py")
    if group == "all":
        return sorted({path for values in groups.values() for path in values})
    if group not in groups:
        raise ValueError(f"Unknown audit group {group!r}; choose {sorted(groups)} or 'all'.")
    return groups[group]


def main(*, group: str = "all", timeout_seconds: int = 600, output_dir: str | Path = "outputs/3d-audit") -> int:
    root = Path(__file__).resolve().parents[1]
    paths = select_tests(group=group, directory=root / "tests")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Audit group {group}: {len(paths)} test modules; timeout {timeout_seconds} s per test.", flush=True)
    for path in paths:
        print(path.relative_to(root), flush=True)
    return pytest.main([str(path) for path in paths] + [
        "-q", "-ra", "--tb=short", f"--timeout={timeout_seconds}", "--durations=10",
        f"--junitxml={output / (group + '.xml')}",
    ])


if __name__ == "__main__":
    group = os.environ.get("DARCYWARP_AUDIT_GROUP", "all")
    raise SystemExit(main(group=group, timeout_seconds=600, output_dir="outputs/3d-audit"))
