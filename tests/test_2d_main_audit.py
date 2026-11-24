"""Analytical and state-ownership regressions for the f8e7683 audit."""

import os

import numpy as np
import pytest

from DARCY_WARP_PACKAGE import create_solver
from DARCY_WARP_PACKAGE.config import NP_FLOAT
from DARCY_WARP_PACKAGE.physics.budgets_2d import compute_mass_balance_budget


def side_heads(*, width):
    mask = np.zeros((width, width), dtype=np.int32)
    mask[:, [0, -1]] = 1
    return mask


@pytest.mark.parametrize("recovery_iterations,accepted", [(100, True), (1, False)])
def test_confined_recovery_uses_the_transient_equation(recovery_iterations, accepted):
    width = 16
    shape = (width, width)
    previous = np.full(shape, 10.0)
    with create_solver(dim=2, nx=width, ny=width, dx=1., device="cpu") as model:
        model.build_from_fields(T_field=np.ones(shape), R_field=np.full(shape, 0.02),
            active=np.ones(shape), bc_mask=side_heads(width=width), bc_values=np.zeros(shape))
        head, info = model.solve(transient=True, dt=1., storage_coeff=0.2,
            head_prev=previous, initial_head=previous, implementation="classic",
            divergence_cycle_start=1, divergence_residual_factor=1e-12,
            check_every_no=1, nu_coarse=1, nu_pre=1, nu_post=1, max_cycles=2,
            fallback_pcg_max_iter=recovery_iterations, abs_tol_min=1e-10, rel_tol=1e-10)
    assert info["fallback_from"] == "kcycle"
    assert info["converged"] is accepted
    if accepted:
        n = width - 2
        matrix = np.diag(np.full(n, 2.2)) - np.eye(n, k=1) - np.eye(n, k=-1)
        reference = np.linalg.solve(matrix, np.full(n, 2.02))
        np.testing.assert_allclose(head[8, 1:-1], reference, atol=1e-7, rtol=0)


@pytest.mark.parametrize("implementation", ["classic", "fast"])
def test_heads_survive_reuse_and_close(implementation):
    shape = (8, 8)
    model = create_solver(dim=2, nx=8, ny=8, dx=1., device="cpu")
    with model:
        model.build_from_fields(T_field=np.ones(shape), R_field=np.zeros(shape),
            active=np.ones(shape), bc_mask=side_heads(width=8), bc_values=np.zeros(shape))
        options = dict(implementation=implementation, max_cycles=600, check_every_no=2)
        baseline, start = model.solve(**options)
        model.update_R_in_place(R_truth=-1.0)
        pumped, end = model.solve(**options)
        assert start["converged"] and end["converged"]
    np.testing.assert_array_equal(baseline, np.zeros(shape))
    assert (pumped - baseline)[3, 3] < -5.9


@pytest.mark.parametrize("implementation", ["classic", "fast"])
def test_isolated_cell_storage_and_steady_grounding(implementation):
    shape = (8, 8)
    active = np.zeros(shape, dtype=np.int32)
    active[3, 3] = 1
    recharge = np.zeros(shape)
    recharge[3, 3] = 0.02
    with create_solver(dim=2, nx=8, ny=8, dx=1., device="cpu") as model:
        model.build_from_fields(T_field=np.ones(shape), R_field=recharge, active=active,
            bc_mask=np.zeros(shape), bc_values=np.zeros(shape))
        with pytest.raises(ValueError, match="Ungrounded"):
            model.solve(implementation=implementation)
        head, info = model.solve(transient=True, dt=1., storage_coeff=0.2,
            head_prev=np.full(shape, 10.), implementation=implementation)
        assert info["converged"]
        assert head[3, 3] == pytest.approx(10.1)


def closed_period_options(*, rate, periods):
    shape = (8, 8)
    return dict(initial_head=np.full(shape, 10.), recharge_rates=np.full(periods, rate),
        k_field=np.ones(shape), zbot_field=np.zeros(shape), ztop_field=np.full(shape, 20.),
        sy=0.2, ss=0.0, dt=1., bc_mask=np.zeros(shape, dtype=np.int32), save_diagnostics=True)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("rate,storage_reference,cap", [
    (8.64e-5, "current_picard", 10.0), (1e-5, "current_picard", 10.0),
    (8.64e-5, "current_picard", 2e-5), (1e-5, "previous_period", 10.0),
])
def test_closed_picard_period_water_balance(device, rate, storage_reference, cap):
    if np.dtype(NP_FLOAT) != np.dtype(np.float64):
        pytest.skip("Resolving the FP64 storage change requires DARCY_FLOAT=float64")
    if device.startswith("cuda"):
        import warp as wp

        if os.environ.get("DARCYWARP_TEST_CPU_ONLY") == "1" or not wp.is_cuda_available():
            pytest.skip("CUDA required")
    n_periods = 30 if cap > 1 else 1
    with create_solver(dim=2, nx=8, ny=8, dx=10., device=device) as model:
        heads, diagnostics = model.solve_transient_2d_unconfined(
            **closed_period_options(rate=rate, periods=n_periods), storage_reference=storage_reference,
            solve_controls=dict(max_outer_iterations=80, max_head_change_per_outer_iteration=cap),
        )
    assert diagnostics["accepted_periods"] == n_periods
    old = diagnostics["heads_old_per_period"]
    storage = 0.2 * (heads - old)
    np.testing.assert_allclose(storage.mean(axis=(1, 2)), rate, rtol=1.1e-3, atol=1e-12)
    np.testing.assert_allclose(diagnostics["storage_terms_per_period"], storage, rtol=0, atol=1e-12)
    cumulative = 0.2 * (heads[-1].mean() - 10.)
    assert cumulative == pytest.approx(rate * n_periods, rel=1.1e-3)
    assert all(row["physical_period_balance"]["accepted"] for row in diagnostics["period_infos"])


@pytest.mark.parametrize("return_info,diagnostic", [(False, False), (True, False), (True, True)])
def test_host_period_rejection_stops_clock(monkeypatch, return_info, diagnostic):
    options = closed_period_options(rate=0.02, periods=2)
    options["bc_mask"] = side_heads(width=8)
    options["bc_values"] = options["initial_head"]
    controls = dict(max_outer_iterations=1, hclose=1e-11,
        allow_unaccepted_transient_period=diagnostic, use_device_transient_fast_path=False)
    with create_solver(dim=2, nx=8, ny=8, dx=1., device="cpu") as model:
        update_recharge = model.update_uniform_recharge_in_place
        calls = []

        def record_recharge(recharge_rate):
            calls.append(recharge_rate)
            update_recharge(recharge_rate=recharge_rate)

        monkeypatch.setattr(model, "update_uniform_recharge_in_place", record_recharge)
        if diagnostic:
            heads, info = model.solve_transient_2d_unconfined(**options, solve_controls=controls, return_info=True)
            assert heads.shape[0] == 1
            assert info["period_infos"][0]["state"] == "rejected_trial"
            assert info["accepted_time"] == 0.0
            np.testing.assert_array_equal(info["heads_final"], options["initial_head"])
        else:
            with pytest.raises(RuntimeError, match="period 1"):
                model.solve_transient_2d_unconfined(**options, solve_controls=controls, return_info=return_info)
        assert len(calls) == 1


@pytest.mark.parametrize("flow,expected", [(0.8, 22.22222222222222), (0.0, 0.0)])
def test_water_budget_percent_uses_mean_throughflow(flow, expected):
    fields = np.ones((1, 2))
    head = np.array([[0.0, flow]])
    recharge = fields if flow else np.zeros_like(fields)
    budget = compute_mass_balance_budget(T_field=fields, R_field=recharge, head=head,
        active=fields, bc_mask=np.array([[1, 0]]), bc_values=np.zeros_like(fields), dx=1.0)
    assert budget.iloc[0]["percent_discrepancy"] == pytest.approx(expected)


@pytest.mark.parametrize("implementation", ["classic", "fast"])
def test_isolated_ghb_cell_retains_steady_equation(implementation):
    shape = (8, 8)
    active = np.zeros(shape, dtype=np.int32)
    active[3, 3] = 1
    stage = np.full(shape, 15.0)
    with create_solver(dim=2, nx=8, ny=8, dx=1., device="cpu", use_ghb=True, aq_thickness=1.) as model:
        model.build_from_fields(T_field=np.ones(shape), R_field=np.zeros(shape), active=active,
            bc_mask=np.zeros(shape), bc_values=np.zeros(shape), gh_mask=active, gh_head=stage,
            gh_width=np.ones(shape))
        head, info = model.solve(implementation=implementation)
    assert info["converged"]
    assert head[3, 3] == pytest.approx(15.0)


def test_rebuild_checks_new_ungrounded_cell_with_unchanged_positive_connectivity():
    shape = (4, 4)
    active, fixed, transmissivity = (np.zeros(shape) for field in range(3))
    active[0, 0] = fixed[0, 0] = transmissivity[0, 0] = 1.
    fields = dict(T_field=transmissivity, R_field=np.zeros(shape), bc_mask=fixed,
        bc_values=np.zeros(shape))
    with create_solver(dim=2, nx=4, ny=4, dx=1., device="cpu") as model:
        model.build_from_fields(active=active.copy(), **fields)
        assert model.solve()[1]["converged"]
        active[-1, -1] = 1.
        model.build_from_fields(active=active, **fields)
        with pytest.raises(ValueError, match="Ungrounded"):
            model.solve()
