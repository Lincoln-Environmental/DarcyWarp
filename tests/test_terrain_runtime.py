"""Runtime policy and the real terrain entry point, including lazy imports."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import warp as wp


def test_canonical_terrain_runtime_pins():
    root = Path(__file__).resolve().parents[1]
    for filename in ("environment.yml", "requirements-pip.txt"):
        assert re.findall(r"warp-lang==([\d.]+)", (root / filename).read_text()) == [
            "1.17.0"
        ]
    assert "python=3.12" in (root / "environment.yml").read_text()


def test_package_import_stays_lazy_without_site_packages():
    script = """
def check_lazy_import():
    import sys
    import DARCY_WARP_PACKAGE
    assert "warp" not in sys.modules
    assert "numpy" not in sys.modules

if __name__ == "__main__":
    check_lazy_import()
"""
    result = subprocess.run(
        args=[sys.executable, "-S", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "version,missing_helper", [("1.11.0", False), ("1.18.0", False), ("1.17.0", True)]
)
def test_terrain_entry_reports_unsupported_runtime(version, missing_helper):
    script = """
def build_with_unsupported_runtime(*, version, missing_helper):
    import numpy as np
    import warp as wp
    import warp._src.optim.linear as linear
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
    wp.__version__ = version
    if missing_helper:
        del linear._gmres_add_active_entries_kernel
    with WarpDarcySolver3D(nx=3, ny=1, nz=2, dx=1., device="cpu", implementation="fast") as solver:
        solver.build_from_terrain(surface_elevation=np.zeros((1, 3)),
            zeta_faces=np.array([0., 1., 2.]), kx_field=1., ky_field=1., kz_field=1.,
            bc_mask=np.ones((2, 1, 3), dtype=np.int32), bc_values=np.ones((2, 1, 3)),
            rhs=np.zeros((2, 1, 3)))

if __name__ == "__main__":
    import sys
    requested_version = sys.argv[1]
    remove_helper = sys.argv[2] == "True"
    build_with_unsupported_runtime(version=requested_version, missing_helper=remove_helper)
"""
    result = subprocess.run(
        args=[sys.executable, "-c", script, version, str(missing_helper)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Terrain GMRES requires warp-lang==1.17.0" in result.stderr
    assert "python -m pip install warp-lang==1.17.0" in result.stderr
    assert version in result.stderr


@pytest.mark.parametrize("backend", ["cpu", "cuda:0"])
@pytest.mark.parametrize("device_picard", [False, True])
def test_real_terrain_runtime_smoke(backend, device_picard):
    if backend.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    from working_tests.smoke_terrain_runtime import run_terrain_smoke

    result = run_terrain_smoke(device=backend, device_picard=device_picard)
    assert result["warp"] == "1.17.0"
    assert result["krylov_iterations"] > 0
    assert result["max_head_error"] <= 1e-9
    if backend.startswith("cuda"):
        assert result["graph_captures"] > 0
        assert result["graph_replays"] > 0
