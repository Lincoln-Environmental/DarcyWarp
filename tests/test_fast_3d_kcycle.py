# SPDX-License-Identifier: AGPL-3.0-only
"""Parity coverage for the opt-in FP64 3D face-array K-cycle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.config import WP_FLOAT
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


pytestmark = pytest.mark.skipif(
    WP_FLOAT is not wp.float64,
    reason="the fast 3D backend is intentionally FP64-only",
)


def _solve(implementation: str):
    shape = (2, 2, 6)
    conductivity = np.ones(shape, dtype=np.float64)
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_mask[:, :, 0] = 1
    bc_mask[:, :, -1] = 1
    bc_values = np.zeros(shape, dtype=np.float64)
    bc_values[:, :, -1] = 1.0
    with WarpDarcySolver3D(
        nx=shape[2], ny=shape[1], nz=shape[0], dx=1.0,
        device="cpu", implementation=implementation,
    ) as solver:
        solver.build_from_K_fields(
            kx_field=conductivity,
            ky_field=conductivity,
            kz_field=conductivity,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            rhs=np.zeros(shape, dtype=np.float64),
            initial_head=bc_values,
        )
        return solver.solve(
            max_cycles=80,
            max_levels=4,
            min_coarse_n=1,
            nu_pre=3,
            nu_post=3,
            nu_coarse=12,
            smoother="jacobi",
            omega=0.7,
            rel_tol=1.0e-6,
            abs_tol_min=1.0e-8,
            dh_rms_tol=1.0e-6,
            check_every_no=1,
        )


def test_fast_3d_face_kcycle_matches_classic_on_cpu():
    if wp.config.kernel_cache_dir is None:
        wp.config.kernel_cache_dir = str(Path("/tmp") / "darcywarp_warp_kernel_cache")
    wp.init()
    head_classic, info_classic = _solve("classic")
    head_fast, info_fast = _solve("fast")
    assert info_classic["converged"]
    assert info_fast["converged"]
    np.testing.assert_allclose(head_fast, head_classic, rtol=0.0, atol=2.0e-5)
    assert info_fast["fast_operator"] == "fp64_face_arrays"


def _solve_transient(implementation: str, storage_coeff):
    shape = (3, 2, 7)
    conductivity = np.ones(shape, dtype=np.float64)
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_mask[:, :, 0] = 1
    bc_mask[:, :, -1] = 1
    bc_values = np.zeros(shape, dtype=np.float64)
    bc_values[:, :, 0] = 2.0
    head_prev = np.full(shape, 0.75, dtype=np.float64)
    head_prev[bc_mask != 0] = bc_values[bc_mask != 0]
    with WarpDarcySolver3D(
        nx=shape[2],
        ny=shape[1],
        nz=shape[0],
        dx=2.0,
        dy=3.0,
        dz=0.5,
        device="cpu",
        implementation=implementation,
    ) as solver:
        solver.build_from_K_fields(
            kx_field=conductivity,
            ky_field=conductivity,
            kz_field=conductivity,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            rhs=np.zeros(shape, dtype=np.float64),
            initial_head=head_prev,
        )
        return solver.solve(
            transient=True,
            storage_coeff=storage_coeff,
            dt=0.25,
            head_prev=head_prev,
            max_cycles=120,
            max_levels=4,
            min_coarse_n=1,
            nu_pre=4,
            nu_post=4,
            nu_coarse=16,
            smoother="jacobi",
            omega=0.7,
            rel_tol=1.0e-8,
            abs_tol_min=1.0e-10,
            dh_rms_tol=1.0e-8,
            check_every_no=1,
        )


@pytest.mark.parametrize(
    "storage_coeff",
    [0.15, np.linspace(0.05, 0.20, 42, dtype=np.float64).reshape(3, 2, 7)],
)
def test_fast_3d_transient_matches_classic_on_cpu(storage_coeff):
    head_classic, info_classic = _solve_transient("classic", storage_coeff)
    head_fast, info_fast = _solve_transient("fast", storage_coeff)
    assert info_classic["converged"]
    assert info_fast["converged"]
    assert info_fast["transient"] is True
    assert info_fast["transient_formulation"] == "confined"
    assert info_fast["dt"] == pytest.approx(0.25)
    np.testing.assert_allclose(head_fast, head_classic, rtol=0.0, atol=2.0e-6)


def test_fast_3d_session_reuses_workspace_and_invalidates_faces():
    if wp.config.kernel_cache_dir is None:
        wp.config.kernel_cache_dir = str(Path("/tmp") / "darcywarp_warp_kernel_cache")
    wp.init()
    shape = (2, 1, 4)
    k = np.ones(shape, dtype=np.float64)
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_mask[:, :, 0] = 1
    bc_mask[:, :, -1] = 1
    bc_values = np.zeros(shape, dtype=np.float64)
    bc_values[:, :, -1] = 1.0
    controls = {
        "max_cycles": 30, "max_levels": 3, "min_coarse_n": 1,
        "nu_pre": 3, "nu_post": 3, "nu_coarse": 12,
        "smoother": "jacobi", "omega": 0.7, "check_every_no": 1,
    }
    with WarpDarcySolver3D(nx=4, ny=1, nz=2, dx=1.0, device="cpu", implementation="fast") as solver:
        solver.build_from_K_fields(
            kx_field=k, ky_field=k, kz_field=k, active=active,
            bc_mask=bc_mask, bc_values=bc_values, rhs=np.zeros(shape), initial_head=bc_values,
        )
        _, first = solver.solve(**controls)
        _, second = solver.solve(**controls)
        faces = tuple(np.asarray(face).copy() for face in (
            solver._tx_p, solver._tx_m, solver._ty_p, solver._ty_m, solver._tz_p, solver._tz_m
        ))
        solver.update_face_conductance(*faces)
        _, after_face_update = solver.solve(**controls)
    assert not first["workspace_reused"]
    assert second["workspace_reused"]
    assert not after_face_update["workspace_reused"]


def test_fast_hierarchy_depth_does_not_change_converged_solution():
    shape = (2, 1, 8)
    k = np.ones(shape, dtype=np.float64)
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_mask[:, :, 0] = 1; bc_mask[:, :, -1] = 1
    bc_values = np.zeros(shape, dtype=np.float64); bc_values[:, :, -1] = 1.0
    heads = []
    for levels in (1, 4):
        with WarpDarcySolver3D(nx=8, ny=1, nz=2, dx=1.0, device="cpu", implementation="fast") as solver:
            solver.build_from_K_fields(
                kx_field=k, ky_field=k, kz_field=k, active=active,
                bc_mask=bc_mask, bc_values=bc_values, rhs=np.zeros(shape), initial_head=bc_values,
            )
            head, info = solver.solve(
                max_cycles=80, max_levels=levels, min_coarse_n=1,
                smoother="jacobi", nu_pre=3, nu_post=3, nu_coarse=12,
                omega=0.7, check_every_no=1, rel_tol=1.0e-7, abs_tol_min=1.0e-8,
                dh_rms_tol=1.0e-7,
            )
            assert info["converged"]
            heads.append(head)
    np.testing.assert_allclose(heads[0], heads[1], rtol=0.0, atol=2.0e-5)
