"""Regression cases from the numerical audit of main at 07ea6cd."""

import numpy as np
import pytest

from DARCY_WARP_PACKAGE import WarpDarcySolver3D, create_solver
from DARCY_WARP_PACKAGE.config import NP_FLOAT
from DARCY_WARP_PACKAGE.physics.budgets_2d import compute_mass_balance_budget


def test_positional_kcycle_rejects_enabled_drain():
    shape = (8, 8)
    ones = np.ones(shape)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, [0, -1]] = 1
    with create_solver(dim=2, nx=8, ny=8, dx=1.0, device="cpu", use_drn=True) as model:
        model.build_from_fields(
            T_field=10.0 * ones, R_field=np.zeros(shape), active=ones,
            bc_mask=fixed, bc_values=10.0 * ones,
            drn_mask=1 - fixed, drn_elev=np.zeros(shape), drn_cond=5.0 * ones,
        )
        with pytest.raises(NotImplementedError, match="DRN|drain"):
            model.solve_multigrid_kcycle(300, check_every_no=2)


def test_newton_period_driver_preserves_built_drain():
    shape = (4, 4)
    ones = np.ones(shape)
    zeros = np.zeros(shape)
    initial = 10.4 * ones
    with create_solver(dim=2, nx=4, ny=4, dx=1.0, device="cpu", use_drn=True) as model:
        model.build_from_fields(
            T_field=8.0 * ones, R_field=0.08 * ones, active=ones,
            bc_mask=zeros, bc_values=initial,
            drn_mask=ones, drn_elev=10.5 * ones, drn_cond=0.3 * ones,
        )
        heads, info = model.solve_transient_2d_unconfined(
            solver="unconfined_semismooth_newton_kcycle", initial_head=initial, return_info=True,
            recharge_rates=[0.08, 0.08], k_field=ones, zbot_field=zeros,
            ztop_field=8.0 * ones, sy=0.2, ss=0.025, dt=1.0,
            active=ones, bc_mask=zeros, bc_values=initial,
            solve_controls={"newton_fallback_to_picard": False},
        )
    # Ss times 8 m thickness gives S=0.2. For each active drain step,
    # 0.2 (h_new - h_old) + 0.3 (h_new - 10.5) = 0.08.
    assert all(period["converged"] for period in info["period_infos"])
    np.testing.assert_allclose(heads[0], 10.62, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(heads[1], 10.708, rtol=0.0, atol=1e-6)


@pytest.mark.skipif(NP_FLOAT != np.float64, reason="Small source relative-balance qualification requires FP64")
def test_picard_storage_closes_in_each_disconnected_aquifer():
    shape = (8, 9)
    ones = np.ones(shape)
    active = np.ones(shape, dtype=np.int32)
    active[:, 4] = 0
    rate = 8.64e-5  # m/day
    source = rate * ones
    source[:, 4] = 0.0
    source[:, 5:] = -rate
    initial = 10.0 * ones
    sy = 0.2
    with create_solver(dim=2, nx=9, ny=8, dx=10.0, device="cpu") as model:
        model.build_from_fields(
            T_field=86.4 * ones, R_field=source, active=active,
            bc_mask=np.zeros(shape), bc_values=initial,
        )
        head, info = model.solve(
            formulation="unconfined", initial_head=initial, K_field=8.64 * ones,
            zbot_field=np.zeros(shape), ztop_field=20.0 * ones, transient=True,
            sy=sy, ss=0.0, dt=1.0, head_prev=initial,
            storage_reference="current_picard",
            unconfined_storage_mode_2d="mf6_convertible_secant_sy",
        )
    assert info["converged"]
    uptake = sy * (head - initial)  # storage per unit area over one day
    for aquifer in (np.s_[:, :4], np.s_[:, 5:]):
        np.testing.assert_allclose(uptake[aquifer], source[aquifer], rtol=1e-3, atol=1e-14)


@pytest.mark.parametrize("package", ["DRN", "RIV"])
def test_cell_residual_and_diagonal_include_package(package):
    ones = np.ones((1, 1))
    zeros = np.zeros((1, 1))
    if package == "DRN":
        boundary = dict(drn_mask=ones, drn_elev=zeros, drn_cond=ones)
    else:
        boundary = dict(riv_mask=ones, riv_stage=zeros, riv_rbot=zeros, riv_cond=ones)
    budget, residual, diagonal, _ = compute_mass_balance_budget(
        T_field=ones, R_field=ones, head=ones, active=ones, bc_mask=zeros,
        bc_values=zeros, dx=1.0, return_cells=True, **boundary,
    )
    # One unit of recharge balances one unit of drainage with dQ/dh=1.
    assert budget["in_minus_out"] == pytest.approx(0.0)
    np.testing.assert_allclose(residual, zeros, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(diagonal, ones, rtol=0.0, atol=1e-14)


def test_classic_chebyshev_accepts_linear_equilibrium_at_elevated_datum():
    exact_head = np.linspace(100.0, 101.0, 4).reshape(1, 1, 4)
    conductivity = np.ones_like(exact_head)
    fixed = np.array([[[1, 0, 0, 1]]])
    with WarpDarcySolver3D(nx=4, ny=1, nz=1, dx=1.0, device="cpu", solver="chebyshev") as model:
        model.build_from_K_fields(
            kx_field=conductivity, ky_field=conductivity, kz_field=conductivity,
            active=np.ones_like(fixed), bc_mask=fixed, bc_values=exact_head,
            rhs=np.zeros_like(exact_head), initial_head=exact_head,
        )
        head, info = model.solve()

    np.testing.assert_allclose(head, exact_head.astype(NP_FLOAT), rtol=0.0, atol=2 * np.spacing(NP_FLOAT(101.0)))
    assert info["converged"], f"Analytical solution rejected at {info['budget']['head_residual_rms']:.3g} m RMS"


def test_cartesian_rejects_reserved_active_ghb_stage():
    conductivity = np.ones((1, 1, 2))
    fixed = np.array([[[1, 0]]])
    ghb_conductance = np.array([[[0.0, 1.0]]])
    ghb_stage = np.array([[[0.0, 1e20]]])
    with WarpDarcySolver3D(nx=2, ny=1, nz=1, dx=1.0, device="cpu") as model:
        with pytest.raises(ValueError):
            model.build_from_K_fields(
                kx_field=conductivity, ky_field=conductivity, kz_field=conductivity,
                active=np.ones_like(fixed), bc_mask=fixed, bc_values=np.zeros_like(conductivity),
                rhs=np.zeros_like(conductivity), ghb_conductance=ghb_conductance, ghb_head=ghb_stage,
            )
            model.solve()


@pytest.mark.parametrize("operation", ["build", "update"])
def test_active_river_rejects_stage_below_bed(operation):
    shape = (3, 3)
    ones = np.ones(shape)
    zeros = np.zeros(shape)
    fixed = np.ones(shape, dtype=np.int32)
    fixed[1, 1] = 0
    river = 1 - fixed
    fields = dict(
        T_field=10.0 * ones, R_field=zeros, active=ones,
        bc_mask=fixed, bc_values=10.0 * ones,
        riv_mask=river, riv_cond=0.5 * ones, riv_rbot=11.0 * ones,
    )
    with create_solver(dim=2, nx=3, ny=3, dx=1.0, device="cpu", use_riv=True) as model:
        if operation == "build":
            with pytest.raises(ValueError):
                model.build_from_fields(riv_stage=8.0 * ones, **fields)
        else:
            model.build_from_fields(riv_stage=12.0 * ones, **fields)
            with pytest.raises(ValueError):
                model.update_riv_boundary_2d(
                    mask=river, stage=8.0 * ones, bottom=11.0 * ones, conductance=0.5 * ones,
                )
