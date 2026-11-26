# SPDX-License-Identifier: AGPL-3.0-or-later
"""Device selection and the native executable used by numerical parity tests."""

import os
from pathlib import Path
import shutil

import pytest


# Warp fixes floating-point precision when the first kernel module is imported.
os.environ.setdefault("DARCY_FLOAT", "float64")
if os.environ.get("DARCYWARP_TEST_CPU_ONLY") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@pytest.fixture(scope="session")
def solver_device():
    import warp as wp

    if os.environ.get("DARCYWARP_TEST_CPU_ONLY") == "1":
        return "cpu"
    requested = os.environ.get("WARP_DEVICE")
    if requested:
        return str(wp.get_device(requested))
    return "cuda:0" if wp.is_cuda_available() else "cpu"


@pytest.fixture(scope="session")
def pinned_mf6():
    from DARCY_WARP_PACKAGE.validation.npf662_mf6 import pinned_native_executable

    executable = shutil.which("mf6")
    if executable is None and Path("/bin/modflow/mf6").is_file():
        executable = "/bin/modflow/mf6"
    if executable is None:
        pytest.skip("MODFLOW 6.6.2 executable unavailable")
    pytest.importorskip("flopy")
    binary, version, checksum, build = pinned_native_executable(executable=executable)
    return Path(binary)


def pytest_collection_modifyitems(items):
    """Keep large reference runs selectable without changing their original test code."""
    for item in items:
        filename = item.path.name
        if filename in {"test_comparison_results.py", "test_unconfined_solvers_500x500.py"}:
            item.add_marker(pytest.mark.regional_parity)
        elif filename == "test_2d_unconfined.py":
            settings = getattr(item, "callspec", None)
            if settings is not None and settings.params.get("nx", 0) >= 1000:
                item.add_marker(pytest.mark.regional_parity)
