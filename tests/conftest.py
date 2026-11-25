# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit CPU-only execution and honest CUDA coverage reporting."""
import os
from pathlib import Path
import shutil
import pytest


# Pin the production precision before any test module is collected: some
# modules import DARCY_WARP_PACKAGE.config at collection time, and config.py
# defaults to float32 while the production kernels (warped_darcy.py) default
# to float64. Without this, whichever file imports config first decides the
# process-wide dtype and later module-level setdefault calls cannot change it.
os.environ.setdefault("DARCY_FLOAT", "float64")


def pytest_configure(config):
    if os.environ.get('DARCYWARP_TEST_CPU_ONLY') == '1':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def pytest_collection_modifyitems(config, items):
    import warp as wp
    if wp.is_cuda_available():
        return
    for item in items:
        parameters = getattr(getattr(item, 'callspec', None), 'params', {})
        for name, value in parameters.items():
            if isinstance(value, str) and value.startswith('cuda') and name in ('device', 'backend', 'dev', 'terrain_test_device'):
                item.add_marker(pytest.mark.skip(reason='CUDA unavailable or explicitly disabled; GPU coverage not executed'))


@pytest.fixture(scope="session")
def pinned_mf6():
    """Locate a native executable whose identity is checked by the adapter."""
    executable = shutil.which("mf6")
    if executable is None and Path("/bin/modflow/mf6").is_file():
        executable = "/bin/modflow/mf6"
    if executable is None:
        pytest.skip("Pinned MODFLOW 6.6.2 executable unavailable")
    pytest.importorskip("flopy")
    return Path(executable)
