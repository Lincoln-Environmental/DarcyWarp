# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run 3D solver and independent equation checks without starting native MF6."""
from pathlib import Path

import pytest


def run_tests(*, modules, report):
    """Run the named pytest modules from a repository-root invocation."""
    paths = [str(Path("tests") / module) for module in modules]
    return pytest.main(["-q", "-ra", "--tb=short", f"--junitxml={report}", *paths])


if __name__ == "__main__":
    raise SystemExit(run_tests(
        modules=["test_3d_groundwater.py", "test_reference_equations.py"],
        report="outputs/3d_gpu_tests.xml",
    ))
