# SPDX-License-Identifier: AGPL-3.0-or-later
"""Analytical validation for the 2D explicit WEL package.

Well rates are volumetric [L3/T]; the square cell area converts each rate to a
depth rate exactly once before the operator applies the area again.
"""
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.physics.wel import validate_well_entries, well_entries
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

DEVICES = ["cpu", "cuda:0"]


def _cuda_available(device):
    if not device.startswith("cuda"):
        return True
    wp.init()
    return wp.is_cuda_available()


def _anchored_2d(device, ny=6, nx=6, dx=100., T=50.):
    """Confined 2D model with a prescribed-head left column."""
    solver = WarpDarcySolver(nx=nx, ny=ny, dx=dx, device=device)
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_mask[:, 0] = 1
    bc_values = np.full((ny, nx), 10.0)
    solver.build_from_fields(T_field=np.full((ny, nx), T), R_field=0.0,
        active=active, bc_mask=bc_mask, bc_values=bc_values)
    return solver


@pytest.mark.parametrize("device", DEVICES)
def test_2d_pumping_matches_independent_row_balance(device):
    """Confined drawdown matches a scipy reference assembled in the test."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    ny, nx, dx, T = 6, 6, 100., 50.
    well_volume = -400.
    solver = _anchored_2d(device, ny=ny, nx=nx, dx=dx, T=T)
    try:
        solver.update_wel_in_place([((3, 3), well_volume)])
        head, info = solver.solve()
        assert info["converged"]
        n = ny * nx
        ids = np.arange(n).reshape(ny, nx)
        rows, cols, values = [], [], []
        face = T * dx / dx  # unit-width faces scaled by the operator's dx^2 source
        for j in range(ny):
            for i in range(nx - 1):
                a, b = ids[j, i], ids[j, i + 1]
                rows += [a, b]; cols += [b, a]; values += [-face, -face]
        for j in range(ny - 1):
            for i in range(nx):
                a, b = ids[j, i], ids[j + 1, i]
                rows += [a, b]; cols += [b, a]; values += [-face, -face]
        off = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
        diagonal = np.asarray(-off.sum(axis=1)).ravel()
        matrix = (off + diags(diagonal)).tocsr()
        rhs = np.zeros(n)
        fixed = ids[:, 0]
        free = np.ones(n, dtype=bool)
        free[fixed] = False
        solution = np.zeros(n)
        solution[fixed] = 10.0  # prescribed head
        well_flat = np.ravel_multi_index((3, 3), (ny, nx))
        # The operator equation sums face T*(dh) with RHS in volumetric units
        # (depth rate times dx^2), so the well enters as its volume rate.
        rhs[well_flat] += well_volume
        keep = np.flatnonzero(free)
        fixed_rows = np.flatnonzero(~free)
        solution[keep] = spsolve(matrix[keep][:, keep],
                                 rhs[keep] - matrix[keep][:, fixed_rows] @ solution[fixed_rows])
        expected = solution.reshape(ny, nx)
        scaled = head  # solver heads are unscaled with head_scale=1
        np.testing.assert_allclose(scaled, expected, rtol=1e-6, atol=1e-6)
        assert solver.wel_budget()["outflow"] == pytest.approx(400.)
    finally:
        solver.close()


@pytest.mark.parametrize("device", DEVICES)
def test_2d_update_independence_and_uniform_fast_path(device):
    """Recharge, uniform recharge and wells stay independently owned."""
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    solver = _anchored_2d(device)
    try:
        solver.update_wel_in_place(well_entries((3, 3), (2, 4), rates=[-500., 200.]))
        solver.update_R_in_place(1e-3)
        assert solver.wel_budget()["inflow"] == pytest.approx(200.)
        assert solver.wel_budget()["outflow"] == pytest.approx(500.)
        # Combined host field: recharge on active cells plus depth rates.
        assert solver.R_field_host[3, 3] == pytest.approx(1e-3 - 500. / (100. * 100.))
        solver.update_uniform_recharge_in_place(2e-3)
        assert solver.R_field_host[2, 4] == pytest.approx(2e-3 + 200. / (100. * 100.))
        device_field = solver.R_wp.numpy()
        np.testing.assert_allclose(device_field, solver.R_field_host, rtol=1e-12, atol=0.)
        solver.update_wel_in_place(None)
        assert solver.wel_budget() == dict(inflow=0., outflow=0., net=0.)
        device_field = solver.R_wp.numpy()
        np.testing.assert_allclose(device_field, 2e-3 * (np.asarray(solver.active_host) != 0))
    finally:
        solver.close()


def test_2d_invalid_entries_and_shapes():
    """The 2D validator rejects inactive cells, scalars and bad indices."""
    active = np.ones((4, 4), dtype=np.int32)
    entries = validate_well_entries(entries=[((1, 2), -100.)], shape=(4, 4), active=active)
    assert len(entries) == 1
    with pytest.raises(ValueError):
        validate_well_entries(entries=[((1, 9), -100.)], shape=(4, 4), active=active)
    with pytest.raises(ValueError):
        validate_well_entries(entries=[((1, 2, 0), -100.)], shape=(4, 4), active=active)
    inactive = active.copy()
    inactive[2, 2] = 0
    with pytest.raises(ValueError):
        validate_well_entries(entries=[((2, 2), -100.)], shape=(4, 4), active=inactive)


@pytest.mark.parametrize("device", DEVICES)
def test_2d_recharge_and_well_storage_response(device):
    """Confined storage absorbs recharge minus pumping exactly over one step."""
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    ny, nx, dx = 4, 4, 10.
    solver = WarpDarcySolver(nx=nx, ny=ny, dx=dx, device=device)
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_mask[:, 0] = 1
    solver.build_from_fields(T_field=np.full((ny, nx), 20.), R_field=0.0,
        active=active, bc_mask=bc_mask, bc_values=np.full((ny, nx), 1.0))
    try:
        solver.update_wel_in_place([((2, 2), 400.)])  # injection
        solver.update_R_in_place(0.0)
        head, info = solver.solve()
        assert info["converged"]
        # Injection with confined T raises heads above the prescribed 1 m.
        assert head[2, 2] > 1.0
        assert solver.wel_budget()["inflow"] == pytest.approx(400.)
    finally:
        solver.close()


@pytest.mark.parametrize("device", DEVICES)
def test_identical_well_rates_equivalent_2d_and_3d(device):
    """A single-layer confined 3D model reproduces the integrated 2D model.

    Both receive the same public volumetric well rate, the same grid and the
    same prescribed-head column; T_2d = K * dz integrates the layer.
    """
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    ny, nx, dx, k, dz = 5, 5, 40., 8., 6.25  # T = 50
    well_volume = -700.
    solver_2d = WarpDarcySolver(nx=nx, ny=ny, dx=dx, device=device)
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_mask[:, 0] = 1
    solver_2d.build_from_fields(T_field=np.full((ny, nx), k * dz), R_field=0.0,
        active=active, bc_mask=bc_mask, bc_values=np.full((ny, nx), 4.0),
        wel_entries=[((2, 2), well_volume)])
    shape = (1, ny, nx)
    solver_3d = WarpDarcySolver3D(nx=nx, ny=ny, nz=1, dx=dx, dy=dx, dz=dz,
        formulation="mf6_6_6_2_npf", solver="kcycle", implementation="fast", device=device)
    solver_3d.build_from_K_fields(kx_field=np.full(shape, k), ky_field=np.full(shape, k),
        kz_field=np.full(shape, k), active=np.ones(shape, dtype=np.int32),
        bc_mask=bc_mask.reshape(shape), bc_values=np.full(shape, 4.0),
        wel_entries=[((0, 2, 2), well_volume)])
    try:
        head_2d, info_2d = solver_2d.solve()
        assert info_2d["converged"]
        head_3d, info_3d = solver_3d.solve()
        assert info_3d["converged"]
        np.testing.assert_allclose(head_3d[0], head_2d, rtol=1e-6, atol=1e-6)
        # Identical public rates in both dimensionalities.
        assert solver_2d.wel_budget()["outflow"] == pytest.approx(700.)
        assert solver_3d.wel_budget()["outflow"] == pytest.approx(700.)
    finally:
        solver_2d.close()
        solver_3d.close()
