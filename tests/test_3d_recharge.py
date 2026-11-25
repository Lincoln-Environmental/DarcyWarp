# SPDX-License-Identifier: AGPL-3.0-or-later
"""Areal recharge in the public 2D and 3D solvers uses the same signed depth rate."""

import numpy as np
import pytest

from DARCY_WARP_PACKAGE import Model3DInputs, WarpDarcySolver3D, create_solver
from DARCY_WARP_PACKAGE.solvers_3d import build_7point_face_conductance_from_k


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("rate", [0.01, -0.01])
def test_single_layer_and_2d_match_analytical_recharge_profile(device, rate):
    nx, ny = 16, 4
    dx, dy, thickness = 2.0, 3.0, 4.0
    conductivity = 0.5
    transmissivity = conductivity * thickness
    active = np.ones((ny, nx), dtype=np.int32)
    fixed = np.zeros_like(active)
    fixed[:, [0, -1]] = 1
    initial = np.full((ny, nx), 10.0)
    # Exercise both shared input forms, passing exactly the same value to each solver.
    recharge = rate if rate > 0 else np.full((ny, nx), rate)
    with create_solver(dim=2, solver="confined_pcg", nx=nx, ny=ny, dx=dx, device=device) as model:
        model.build_from_fields(T_field=np.full_like(initial, transmissivity), R_field=recharge,
                                active=active, bc_mask=fixed, bc_values=initial)
        head_2d, info = model.solve(initial_head=initial, max_iter=100, rel_tol=1e-12, abs_tol_min=1e-13)
        assert info["converged"]
        head_2d = head_2d.copy()

    k = np.full((1, ny, nx), conductivity)
    inputs = Model3DInputs(kx=k, ky=k, kz=k, active=active[None], bc_mask=fixed[None],
                           bc_values=initial[None], initial_head=initial[None],
                           dx=dx, dy=dy, dz=thickness, R_field=recharge)
    with inputs.build_solver(device=device, implementation="fast") as model:
        head_3d, info = model.solve(max_cycles=150, rel_tol=1e-12, abs_tol_min=1e-13)
        assert info["converged"]
        assert info["budget"]["source_total"] == pytest.approx(rate * dx * dy * ny * (nx - 2))

    # With no flow through the lateral sides, the solution is independent of strip width.
    x = np.arange(nx) * dx
    analytical = 10.0 + rate * x * (x[-1] - x) / (2.0 * transmissivity)
    np.testing.assert_allclose(head_2d, np.broadcast_to(analytical, active.shape), atol=1e-8, rtol=0)
    np.testing.assert_allclose(head_3d[0], head_2d, atol=1e-8, rtol=0)


@pytest.mark.parametrize("geometry", ["layered", "faces"])
def test_recharge_receivers_and_wells_are_independent(geometry):
    shape = (3, 1, 5)
    dx, dy = 2.0, 5.0
    k = np.ones(shape)
    active = np.ones(shape, dtype=np.int32)
    active[0, 0, [1, 4]] = 0
    active[:, 0, 2] = 0
    fixed = np.zeros_like(active)
    fixed[0, 0, 3] = fixed[1, 0, 4] = 1
    wells = np.zeros(shape)
    wells[2, 0, 0] = -0.75
    wells[1, 0, 1] = 0.5
    rates = np.array([[0.125, -0.25, 8.0, 4.0, 2.0]])
    fields = dict(active=active, bc_mask=fixed, bc_values=np.zeros(shape),
                  initial_head=np.zeros(shape), rhs=wells, R_field=rates)
    if geometry == "layered":
        model = WarpDarcySolver3D(nx=5, ny=1, nz=3, dx=dx, dy=dy, dz=99.0, device="cpu",
                                  formulation="mf6_6_6_2_npf")
        interfaces = np.broadcast_to(np.array([9.0, 8.0, 6.0, 2.0])[:, None, None], (4, 1, 5))
        model.build_from_layers(z_interfaces=interfaces, kx_field=k, ky_field=k, kz_field=k, **fields)
    else:
        inputs = Model3DInputs(kx=k, ky=k, kz=k, dx=dx, dy=dy, dz=4.0, **fields)
        faces = build_7point_face_conductance_from_k(kx_field=k, ky_field=k, kz_field=k,
                                                    active=active, dx=dx, dy=dy, dz=4.0)
        model = inputs.build_solver(device="cpu", face_conductance=faces)
    with model:
        # Only the first two columns receive recharge. A fixed uppermost active cell
        # receives none and prevents water passing to the free cell underneath it.
        expected_recharge = np.zeros(shape)
        expected_recharge[0, 0, 0] = 1.25
        expected_recharge[1, 0, 1] = -2.5
        np.testing.assert_array_equal(model._rhs, wells + expected_recharge)
        new_wells = wells * 2.0
        model.update_rhs(new_wells)
        np.testing.assert_array_equal(model._rhs, new_wells + expected_recharge)
        with pytest.raises(ValueError):
            model.update_R_in_place(np.ones(shape))
        np.testing.assert_array_equal(model._rhs, new_wells + expected_recharge)
        model.update_R_in_place(0.0)
        np.testing.assert_array_equal(model._rhs, new_wells)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("period_rates, expected_rates", [
    (0.125, [[0.125, 0.125], [0.125, 0.125]]),
    (np.array([[0.125, 0.0, -0.25]]), [[0.125, -0.25], [0.125, -0.25]]),
    ([0.125, -0.25], [[0.125, 0.125], [-0.25, -0.25]]),
    (np.array([[[0.125, 8.0, -0.25]], [[-0.5, 4.0, 0.0]]]), [[0.125, -0.25], [-0.5, 0.0]]),
])
def test_period_recharge_and_wells_match_confined_storage(device, period_rates, expected_rates):
    # Two isolated active cells. The middle cell is inactive, so each column
    # obeys the exact storage law dh = (R * area + Qwell) * dt / (Ss * volume).
    dx, dy, dz = 2.0, 5.0, 4.0
    area = dx * dy
    ss = 0.05
    initial = np.array([[[10.0, 0.0, 10.0]]])
    k = np.ones_like(initial)
    active = np.array([[[1, 0, 1]]], dtype=np.int32)
    wells = np.array([[[-0.5, 0.0, 0.25]]])
    built_rate = 0.0625
    dt = np.array([2.0, 3.0])
    with WarpDarcySolver3D(nx=3, ny=1, nz=1, dx=dx, dy=dy, dz=dz, device=device,
                           formulation="mf6_6_6_2_npf", implementation="fast") as model:
        model.build_from_K_fields(kx_field=k, ky_field=k, kz_field=k, active=active,
                                  bc_mask=np.zeros_like(active), bc_values=initial,
                                  initial_head=initial, R_field=built_rate, rhs=wells)
        factors = np.array([1.0, 2.0])
        heads, info = model.solve_transient_npf(recharge_rates=period_rates, source_factors=factors,
                                                dt=dt, head_prev=initial, ss=ss, sy=0.0)
        assert info["converged"] and info["accepted_steps"] == len(dt)
        well_rates = wells[0, 0, [0, 2]]
        total = np.array(expected_rates) * area + factors[:, None] * well_rates
        increments = total * dt[:, None] / (ss * area * dz)
        expected = 10.0 + np.cumsum(increments, axis=0)
        np.testing.assert_allclose(heads[:, 0, 0, :][:, [0, 2]], expected, rtol=0, atol=1e-9)

        # With no schedule a new run must see the original static forcing.
        restored, info = model.solve_transient_npf(dt=dt, head_prev=initial, ss=ss, sy=0.0)
        assert info["converged"] and info["accepted_steps"] == len(dt)
        static_q = built_rate * area + well_rates
        expected = 10.0 + np.cumsum(dt)[:, None] * static_q / (ss * area * dz)
        np.testing.assert_allclose(restored[:, 0, 0, :][:, [0, 2]], expected, rtol=0, atol=1e-9)
