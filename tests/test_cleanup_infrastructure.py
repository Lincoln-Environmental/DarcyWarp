"""Regression tests for consolidated plotting and benchmark helpers."""

import subprocess
import sys

import numpy as np

from DARCY_WARP_PACKAGE.canterbury_case_study.map_helpers import _build_edges


def test_north_first_map_edges():
    """The shared helper preserves the north-first raster orientation."""
    x, y = _build_edges((10.0, 14.0, 20.0, 26.0), 2, 3)
    np.testing.assert_array_equal(x, [10.0, 12.0, 14.0])
    np.testing.assert_array_equal(y, [26.0, 24.0, 22.0, 20.0])


def test_project_base_import_is_quiet():
    """Importing benchmark paths does not print host or network-share status."""
    completed = subprocess.run(
        [sys.executable, "-c", "import DARCY_WARP_PACKAGE.project_base"],
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout == ""
