"""Regression coverage for the EXPERIMENTAL fast mixed-precision campaign code.

Covers ``DARCY_WARP_PACKAGE.solvers.mixed_fast_kernels`` (Phase 3 kernels),
``DARCY_WARP_PACKAGE.solvers.mixed_fast`` (fast K-cycle session), and
``DARCY_WARP_PACKAGE.solvers.mixed_vcycle`` (rejected-but-retained V-cycle
reference).  Kernel tests run in the default FP64 process against production
kernels and CPU references; session tests use a child process pinned to
``DARCY_FLOAT=float32`` (precision is fixed at import time).  No benchmark
runtimes are encoded here.
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

# Small deterministic case (heterogeneous ugly_t seed 123 + GHB), multilevel
# hierarchy kept alive with min_coarse_cells=100.
_NX, _NY = 48, 40


def _build_case_fp64():
    """Default-process (FP64) small case; returns (solver, dem, T, R)."""
    import warp as wp

    wp.init()
    from DARCY_WARP_PACKAGE.model_builder import (
        _build_dem,
        _build_domain,
        make_ugly_T_field,
    )
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    domain = _build_domain(nx=_NX, ny=_NY)
    dem = _build_dem(domain)
    T = make_ugly_T_field(nx=_NX, ny=_NY, domain=domain, seed=123)
    R = np.full_like(domain, 1.0e-4, dtype=np.float64)
    solver = WarpDarcySolver(
        nx=_NX, ny=_NY, dx=100.0, device="cuda:0", use_ghb=True,
        solver_type="pcg", aq_thickness=300.0,
    )
    solver.build_from_truth_inputs(T_truth=T, R_truth=R, width=100.0)
    solver.build_hierarchy(max_levels=6, min_coarse_n=4, min_coarse_cells=100)
    return solver, dem, T, R


@requires_cuda
def test_face_residual_and_jacobi_match_production_fp64():
    """Face-array FP64 kernels vs production kernels (heterogeneous T, GHB,
    Dirichlet and inactive cells all present)."""
    import warp as wp

    import DARCY_WARP_PACKAGE.warped_darcy as km
    from DARCY_WARP_PACKAGE.solvers import mixed_fast_kernels as mf3
    from DARCY_WARP_PACKAGE.solvers.mixed_fast import build_face_level

    solver, dem, T, R = _build_case_fp64()
    try:
        lvl0 = solver.mg_levels[0]
        fl = build_face_level(solver, lvl0, wp.float64, "cuda:0")

        rng = np.random.default_rng(7)
        x_np = rng.normal(size=(_NY, _NX)) * 10.0
        b_np = rng.normal(size=(_NY, _NX))
        x = wp.array(x_np, dtype=wp.float64, device="cuda:0")
        b = wp.array(b_np, dtype=wp.float64, device="cuda:0")

        # residual: production vs face-array kernel
        r_prod = wp.zeros((_NY, _NX), dtype=wp.float64, device="cuda:0")
        rTr = wp.zeros(1, dtype=wp.float64, device="cuda:0")
        wp.launch(kernel=km.compute_residual_no_storage_kernel, dim=(_NY, _NX),
                  inputs=[x, b, solver.T_wp, solver.active_wp, solver.bc_mask_wp,
                          solver.gh_mask_wp, solver.ghb_factor_wp, r_prod, rTr,
                          _NX, _NY], device="cuda:0")
        r_fast = wp.zeros((_NY, _NX), dtype=wp.float64, device="cuda:0")
        wp.launch(kernel=mf3._mf3_residual_f64, dim=(_NY, _NX),
                  inputs=[x, b, fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag,
                          solver.active_wp, solver.bc_mask_wp, r_fast, _NX, _NY],
                  device="cuda:0")

        # smoother: production (M_inv form) vs face-array (diag form)
        x_prod = wp.zeros((_NY, _NX), dtype=wp.float64, device="cuda:0")
        wp.launch(kernel=km.jacobi_applyA_fused_no_storage_kernel, dim=(_NY, _NX),
                  inputs=[solver.T_wp, solver.active_wp, solver.bc_mask_wp,
                          solver.gh_mask_wp, solver.ghb_factor_wp, b, x,
                          lvl0.M_inv_wp, lvl0.bc_values_wp, 0.7, _NX, _NY, x_prod],
                  device="cuda:0")
        x_fast = wp.zeros((_NY, _NX), dtype=wp.float64, device="cuda:0")
        wp.launch(kernel=mf3._mf3_jacobi_f64, dim=(_NY, _NX),
                  inputs=[b, x, fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag,
                          solver.active_wp, solver.bc_mask_wp, lvl0.bc_values_wp,
                          0.7, _NX, _NY, x_fast], device="cuda:0")
        wp.synchronize()

        np.testing.assert_allclose(r_fast.numpy(), r_prod.numpy(), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(x_fast.numpy(), x_prod.numpy(), rtol=0.0, atol=1.0e-12)
    finally:
        solver.close()


@requires_cuda
def test_two_stage_reductions_match_cpu_reference():
    """Block-reduced dot and applyA-dot vs CPU/numpy references."""
    import warp as wp

    import DARCY_WARP_PACKAGE.warped_darcy as km
    from DARCY_WARP_PACKAGE.solvers import mixed_fast_kernels as mf3
    from DARCY_WARP_PACKAGE.solvers.mixed_fast import build_face_level

    solver, dem, T, R = _build_case_fp64()
    try:
        lvl0 = solver.mg_levels[0]
        fl = build_face_level(solver, lvl0, wp.float64, "cuda:0")

        rng = np.random.default_rng(11)
        x_np = rng.normal(size=(_NY, _NX))
        b_np = rng.normal(size=(_NY, _NX))
        x = wp.array(x_np, dtype=wp.float64, device="cuda:0")
        b = wp.array(b_np, dtype=wp.float64, device="cuda:0")
        free = (solver.active_host != 0) & (solver.bc_mask_host == 0)
        n = _NX * _NY

        # dot
        wp.launch(kernel=mf3._mf3_dot_partials_f64, dim=n,
                  inputs=[x, b, solver.active_wp, solver.bc_mask_wp,
                          fl.partials, _NX, _NY, 256], device="cuda:0")
        wp.launch(kernel=mf3._mf3_combine_partials_kernel, dim=1,
                  inputs=[fl.partials, fl.out, fl.n_partials], device="cuda:0")
        wp.synchronize()
        ref_dot = float((x_np * b_np)[free].sum())
        assert abs(fl.out.numpy()[0] - ref_dot) <= 1.0e-10 * max(abs(ref_dot), 1.0)

        # applyA-dot: pAp = x . A x with A x from the production residual
        r_prod = wp.zeros((_NY, _NX), dtype=wp.float64, device="cuda:0")
        rTr = wp.zeros(1, dtype=wp.float64, device="cuda:0")
        wp.launch(kernel=km.compute_residual_no_storage_kernel, dim=(_NY, _NX),
                  inputs=[x, b, solver.T_wp, solver.active_wp, solver.bc_mask_wp,
                          solver.gh_mask_wp, solver.ghb_factor_wp, r_prod, rTr,
                          _NX, _NY], device="cuda:0")
        wp.synchronize()
        Ax_np = b_np - r_prod.numpy()
        ref_pAp = float((x_np * Ax_np)[free].sum())

        wp.launch(kernel=mf3._mf3_applyA_dot_partials_f64, dim=n,
                  inputs=[x, fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag,
                          solver.active_wp, solver.bc_mask_wp, fl.partials,
                          _NX, _NY, 256], device="cuda:0")
        wp.launch(kernel=mf3._mf3_combine_partials_kernel, dim=1,
                  inputs=[fl.partials, fl.out, fl.n_partials], device="cuda:0")
        wp.synchronize()
        assert abs(fl.out.numpy()[0] - ref_pAp) <= 1.0e-10 * max(abs(ref_pAp), 1.0)
    finally:
        solver.close()


# ---------------------------------------------------------------------------
# Session tests (child process pinned to DARCY_FLOAT=float32)
# ---------------------------------------------------------------------------

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

domain = _build_domain(nx=NX, ny=NY)
dem = _build_dem(domain)
T_field = make_ugly_T_field(nx=NX, ny=NY, domain=domain, seed=123)
R_field = np.full_like(domain, 1.0e-4, dtype=np.float64)
(_, _, _, _, bc_values64, _, gh_head64, _) = build_truth_inputs(
    nx=NX, ny=NY, dx=DX, T_truth=T_field, R_truth=R_field, use_ghb=True, width=DX,
)

with WarpDarcySolver(nx=NX, ny=NY, dx=DX, device="cuda:0", use_ghb=True,
                     solver_type="pcg", aq_thickness=THICKNESS) as solver:
    solver.build_from_truth_inputs(T_truth=T_field, R_truth=R_field, width=DX)
    solver.build_hierarchy(max_levels=6, min_coarse_n=4, min_coarse_cells=100)
    session = MixedPrecisionFastSession(
        solver, bc_values_f64=bc_values64, gh_head_f64=gh_head64,
        R_f64=R_field, max_levels=6, min_coarse_cells=100,
    )
    controls = dict(inner_kcycles=5, max_outer=40, nu_pre=2, nu_post=2,
                    nu_coarse=10, omega=0.7, rel_tol=5.0e-7, abs_tol_min=5.0e-7)
    head_graph, info_graph = session.solve(dem, **controls)

    # Eager variant: disable graph capture, same solve from the same DEM.
    import warp as wp
    from working_tests.launch_profiler import NullCapture

    session._correction_graph = None
    wp.ScopedCapture = lambda *a, **k: NullCapture()
    head_eager, info_eager = session.solve(dem, **controls)

np.savez(out_path, head_graph=np.asarray(head_graph, dtype=np.float64),
         head_eager=np.asarray(head_eager, dtype=np.float64))
print("RESULT_JSON:" + json.dumps({
    "converged_graph": bool(info_graph["converged"]),
    "converged_eager": bool(info_eager["converged"]),
    "outer_graph": int(info_graph["outer_iterations"]),
    "outer_eager": int(info_eager["outer_iterations"]),
}))
"""


def _run_fast_child(out_path: Path) -> dict:
    env = dict(os.environ)
    env["DARCY_FLOAT"] = "float32"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM, str(out_path)],
        capture_output=True, text=True, env=env, timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"fast mixed-precision child failed:\n{proc.stderr[-3000:]}")
    marker = "RESULT_JSON:"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError("fast mixed-precision child emitted no result")


@pytest.fixture(scope="module")
def fast_child_results(tmp_path_factory):
    if not _cuda_available():
        pytest.skip("CUDA is not available")
    out = tmp_path_factory.mktemp("mixed_fast") / "heads.npz"
    info = _run_fast_child(out)
    data = np.load(out)
    return {"info": info, "head_graph": data["head_graph"], "head_eager": data["head_eager"]}


@requires_cuda
def test_fast_session_runs_converges_finite(fast_child_results):
    head = fast_child_results["head_graph"]
    assert head.shape == (_NY, _NX)
    assert np.all(np.isfinite(head))
    assert fast_child_results["info"]["converged_graph"] is True


@requires_cuda
def test_graph_captured_and_eager_cycles_equivalent(fast_child_results):
    info = fast_child_results["info"]
    assert info["converged_eager"] is True
    assert info["outer_graph"] == info["outer_eager"]
    # Same kernels in the same order; only atomic reduction order can vary.
    np.testing.assert_allclose(
        fast_child_results["head_graph"], fast_child_results["head_eager"],
        rtol=0.0, atol=1.0e-9,
    )


@requires_cuda
def test_fast_session_matches_fp64_backend(fast_child_results):
    """Fast mixed solve vs the production FP64 K-cycle on the same case."""
    solver, dem, T, R = _build_case_fp64()
    try:
        head_fp64, info_fp64 = solver.solve_multigrid_kcycle(
            max_cycles=200, nu_pre=2, nu_post=2, nu_coarse=2, omega=0.7,
            rel_tol=5.0e-7, abs_tol_min=5.0e-7, initial_head=dem,
            return_info=True, max_levels=6, check_every_no=5, min_coarse_cells=100,
        )
        assert info_fp64["converged"] is True
        # Measured agreement is ~6e-6 m (100x100 benchmark case); the
        # regression tolerance sits far below the 2e-4 m MF6 gate.
        np.testing.assert_allclose(
            fast_child_results["head_graph"],
            np.asarray(head_fp64, dtype=np.float64),
            rtol=0.0, atol=5.0e-5,
        )
    finally:
        solver.close()


def test_fast_and_vcycle_paths_remain_experimental_and_unregistered():
    from DARCY_WARP_PACKAGE.solver_capabilities import ALIASES, CAPABILITIES

    assert not any("mixed" in name for name in CAPABILITIES)
    assert not any("mixed" in name for name in ALIASES)
    assert not any("mixed" in target for target in ALIASES.values())

    from DARCY_WARP_PACKAGE.solvers import mixed_fast, mixed_vcycle

    assert mixed_fast.EXPERIMENTAL is True
    assert mixed_vcycle.EXPERIMENTAL is True


def test_mixed_fast_config_defaults_are_the_validated_settings():
    """The callable config defaults pin the campaign-validated configuration."""
    from DARCY_WARP_PACKAGE.solvers.mixed_fast import MixedFastConfig

    cfg = MixedFastConfig()
    assert cfg.inner_kcycles == 5
    assert cfg.nu_pre == 2 and cfg.nu_post == 2
    assert cfg.nu_coarse == 10
    assert cfg.smoother == "chebyshev"
    assert cfg.rel_tol == 5.0e-7 and cfg.abs_tol_min == 5.0e-7
    assert cfg.max_outer >= 40
