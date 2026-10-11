# SPDX-License-Identifier: AGPL-3.0-only
"""Hierarchy-shape regression tests for 3D horizontal semi-coarsening."""

from __future__ import annotations

import numpy as np

from DARCY_WARP_PACKAGE.solvers_3d import (
    build_7point_face_conductance_from_k,
    solve_multigrid_kcycle_7point_3d,
)


def _run_shape(shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    nz, ny, nx = shape
    k = np.ones(shape, dtype=np.float64)
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape, dtype=np.float64)
    if nx > 1:
        bc_mask[:, :, 0] = 1
        bc_mask[:, :, nx - 1] = 1
        bc_values[:, :, nx - 1] = 1.0
    else:
        bc_mask[:, 0, :] = 1
        bc_mask[:, ny - 1, :] = 1
        bc_values[:, ny - 1, :] = 1.0
    rhs = np.zeros(shape, dtype=np.float64)
    initial = np.zeros(shape, dtype=np.float64)
    initial[:, :, :] = np.linspace(0.0, 1.0, nx)[None, None, :]
    faces = build_7point_face_conductance_from_k(
        kx_field=k,
        ky_field=k,
        kz_field=k,
        active=active,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    _, info = solve_multigrid_kcycle_7point_3d(
        tx_p=faces[0],
        tx_m=faces[1],
        ty_p=faces[2],
        ty_m=faces[3],
        tz_p=faces[4],
        tz_m=faces[5],
        rhs=rhs,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        initial_head=initial,
        max_cycles=1,
        max_levels=6,
        min_coarse_n=2,
        check_every_no=1,
        device="cpu",
        return_info=True,
    )
    return [tuple(int(value) for value in level_shape) for level_shape in info["level_shapes"]]


def test_thin_y_dimension_coarsens_x_only():
    assert _run_shape((3, 1, 9)) == [(3, 1, 9), (3, 1, 5), (3, 1, 3), (3, 1, 2)]


def test_thin_x_dimension_coarsens_y_only():
    assert _run_shape((3, 9, 1)) == [(3, 9, 1), (3, 5, 1), (3, 3, 1), (3, 2, 1)]


def test_regular_grid_coarsens_both_horizontal_axes():
    assert _run_shape((3, 9, 9)) == [(3, 9, 9), (3, 5, 5), (3, 3, 3), (3, 2, 2)]
