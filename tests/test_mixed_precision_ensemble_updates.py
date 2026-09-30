# SPDX-License-Identifier: AGPL-3.0-only
"""Ensemble-update regression tests for the production mixed-precision path.

Covers the two session refresh entry points used by the ensemble benchmark
scripts (model_benchmarking_recharge_change / model_benchmarking_T_change):

* ``update_rhs_f64`` — recharge-change ensembles (per-case R swap);
* ``refresh_operator_faces`` + ``update_rhs_f64`` — T-change ensembles
  (per-case in-place transmissivity update).

FP64 reference heads are computed with the production classic K-cycle in this
(FP64) process; the mixed-precision solves run in a child process pinned to
DARCY_FLOAT=float32 (the mixed path requires an FP32 hierarchy).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _warp_available() -> bool:
    try:
        import warp  # noqa: F401
    except Exception:
        return False
    return True


def _cuda_available() -> bool:
    if not _warp_available():
        return False
    try:
        import warp as wp

        return bool(wp.is_cuda_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _warp_available(), reason="warp is not available")
requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")

_NX, _NY = 48, 40
_DX, _THICKNESS = 100.0, 300.0
_T_SEED = 123
_T2_SEED = 321
_R1 = 1.0e-4
_R2 = 2.0e-4
# Same agreement scale as tests/test_mixed_precision_fast.py (measured ~6e-6 m
# on this case family); far below the 2e-4 m MF6 gate.
_ATOL = 5.0e-5


def _base_fields():
    from DARCY_WARP_PACKAGE.model_builder import (
        _build_dem,
        _build_domain,
        make_ugly_T_field,
    )

    domain = _build_domain(nx=_NX, ny=_NY)
    dem = _build_dem(domain)
    T1 = make_ugly_T_field(nx=_NX, ny=_NY, domain=domain, seed=_T_SEED)
    T2 = make_ugly_T_field(nx=_NX, ny=_NY, domain=domain, seed=_T2_SEED)
    R1 = np.full_like(domain, _R1, dtype=np.float64)
    R2 = np.full_like(domain, _R2, dtype=np.float64)
    return dem, T1, T2, R1, R2


def _classic_fp64_head(T_field: np.ndarray, R_field: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """Production FP64 K-cycle reference for one (T, R) case."""
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    solver = WarpDarcySolver(
        nx=_NX, ny=_NY, dx=_DX, device="cuda:0", use_ghb=True,
        solver_type="pcg", aq_thickness=_THICKNESS,
    )
    try:
        solver.build_from_truth_inputs(T_truth=T_field, R_truth=R_field, width=_DX)
        head, info = solver.solve_multigrid_kcycle(
            max_cycles=200, nu_pre=2, nu_post=2, nu_coarse=2, omega=0.7,
            rel_tol=5.0e-7, abs_tol_min=5.0e-7, initial_head=dem,
            return_info=True, max_levels=6, check_every_no=5, min_coarse_cells=100,
        )
        assert info["converged"] is True
        return np.asarray(head, dtype=np.float64)
    finally:
        solver.close()


_CHILD_PROGRAM = r"""
import json
import sys

import numpy as np

out_path = sys.argv[1]

from DARCY_WARP_PACKAGE.model_builder import (
    _build_dem,
    _build_domain,
    build_truth_inputs,
    make_ugly_T_field,
)
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
from DARCY_WARP_PACKAGE.solvers.mixed_fast import MixedPrecisionFastSession

NX, NY = 48, 40
DX, THICKNESS = 100.0, 300.0
T_SEED, T2_SEED = 123, 321
R1_VAL, R2_VAL = 1.0e-4, 2.0e-4

domain = _build_domain(nx=NX, ny=NY)
dem = _build_dem(domain)
T1 = make_ugly_T_field(nx=NX, ny=NY, domain=domain, seed=T_SEED)
T2 = make_ugly_T_field(nx=NX, ny=NY, domain=domain, seed=T2_SEED)
R1 = np.full_like(domain, R1_VAL, dtype=np.float64)
R2 = np.full_like(domain, R2_VAL, dtype=np.float64)
(_, _, _, _, bc_values64, _, gh_head64, _) = build_truth_inputs(
    nx=NX, ny=NY, dx=DX, T_truth=T1, R_truth=R1, use_ghb=True, width=DX,
)

with WarpDarcySolver(nx=NX, ny=NY, dx=DX, device="cuda:0", use_ghb=True,
                     solver_type="pcg", aq_thickness=THICKNESS) as solver:
    solver.build_from_truth_inputs(T_truth=T1, R_truth=R1, width=DX)
    solver.build_hierarchy(max_levels=6, min_coarse_n=4, min_coarse_cells=100)
    session = MixedPrecisionFastSession(
        solver, bc_values_f64=bc_values64, gh_head_f64=gh_head64,
        R_f64=R1, max_levels=6, min_coarse_cells=100,
    )
    controls = dict(inner_kcycles=5, max_outer=40, nu_pre=2, nu_post=2,
                    nu_coarse=10, omega=0.7, rel_tol=5.0e-7, abs_tol_min=5.0e-7)

    head_a, info_a = session.solve(dem, **controls)

    # Recharge-change ensemble step: swap R only.
    session.update_rhs_f64(R2)
    head_b, info_b = session.solve(dem, **controls)

    # T-change ensemble step: in-place T update + face/RHS refresh.
    solver.update_T_in_place(T2)
    session.refresh_operator_faces()
    session.update_rhs_f64()  # R2 kept; b64 carries a T-dependent GHB term
    head_c, info_c = session.solve(dem, **controls)

np.savez(
    out_path,
    head_a=np.asarray(head_a, dtype=np.float64),
    head_b=np.asarray(head_b, dtype=np.float64),
    head_c=np.asarray(head_c, dtype=np.float64),
)
print("RESULT_JSON:" + json.dumps({
    "converged_a": bool(info_a["converged"]),
    "converged_b": bool(info_b["converged"]),
    "converged_c": bool(info_c["converged"]),
}))
"""


def _run_child(out_path: Path) -> dict:
    env = dict(os.environ)
    env["DARCY_FLOAT"] = "float32"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM, str(out_path)],
        capture_output=True, text=True, env=env, timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mixed ensemble-update child failed:\n{proc.stderr[-3000:]}")
    marker = "RESULT_JSON:"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError("mixed ensemble-update child emitted no result")


@pytest.fixture(scope="module")
def ensemble_results(tmp_path_factory):
    if not _cuda_available():
        pytest.skip("CUDA is not available")
    dem, T1, T2, R1, R2 = _base_fields()
    refs = {
        "a": _classic_fp64_head(T1, R1, dem),
        "b": _classic_fp64_head(T1, R2, dem),
        "c": _classic_fp64_head(T2, R2, dem),
    }
    out = tmp_path_factory.mktemp("mixed_ensemble") / "heads.npz"
    info = _run_child(out)
    data = np.load(out)
    return {
        "info": info,
        "heads": {k: np.asarray(data[f"head_{k}"], dtype=np.float64) for k in "abc"},
        "refs": refs,
    }


@requires_cuda
def test_mixed_solves_converge(ensemble_results):
    info = ensemble_results["info"]
    assert info["converged_a"] is True
    assert info["converged_b"] is True
    assert info["converged_c"] is True
    for head in ensemble_results["heads"].values():
        assert head.shape == (_NY, _NX)
        assert np.all(np.isfinite(head))


@requires_cuda
def test_update_rhs_f64_recharge_change(ensemble_results):
    """Baseline solve + post-update_rhs_f64 solve match FP64 references."""
    np.testing.assert_allclose(
        ensemble_results["heads"]["a"], ensemble_results["refs"]["a"],
        rtol=0.0, atol=_ATOL,
    )
    np.testing.assert_allclose(
        ensemble_results["heads"]["b"], ensemble_results["refs"]["b"],
        rtol=0.0, atol=_ATOL,
    )


@requires_cuda
def test_refresh_operator_faces_T_change(ensemble_results):
    """Post update_T_in_place + refresh solve matches the FP64 reference."""
    np.testing.assert_allclose(
        ensemble_results["heads"]["c"], ensemble_results["refs"]["c"],
        rtol=0.0, atol=_ATOL,
    )
