# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the affected 2D suites using saved references, without launching MF6.

Run from the repository root with DARCY_FLOAT=float64 and PYTHONPATH=.:working_tests.
CUDA tests skip if no CUDA device is available; inspect the reported skip count.
"""
from pathlib import Path

from working_tests.run_3d_gpu_tests import run_tests


if __name__ == "__main__":
    directory = Path("tests")
    patterns = (
        "test_2d*.py",
        "test_fast_confined_kcycle.py",
        "test_mixed_precision*.py",
        "test_transient*.py",
        "test_adaptive_inner_controller.py",
        "test_solver_registry_2d.py",
        "test_sparse_system_consistency.py",
    )
    entry_points = (
        "run_reference_mf6", "run_mf6", "_build_mf6_unconfined", "run_simulation",
    )
    pytest_options = (
        "-q", "--tb=short", "--junitxml=outputs/2d_gpu_tests.xml",
    )
    result = run_tests(directory=directory, patterns=patterns,
                       entry_points=entry_points, pytest_options=pytest_options)
    raise SystemExit(result)
