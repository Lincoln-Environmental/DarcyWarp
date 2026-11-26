# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run 2D solver, sparse-system and saved-reference checks without starting native MF6."""
from working_tests.run_3d_gpu_tests import run_tests


if __name__ == "__main__":
    raise SystemExit(run_tests(
        modules=["test_2d_groundwater.py", "test_sparse_system_consistency.py", "test_2d_unconfined.py"],
        report="outputs/2d_gpu_tests.xml",
    ))
