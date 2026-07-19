import os
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("WARP_CACHE_PATH", str(Path("/tmp/darcywarp-warp-cache")))


def _warp_available() -> bool:
    try:
        import warp  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _warp_available(), reason="warp is not available")


def test_transient_storage_terms_are_idempotent():
    from DARCY_WARP_PACKAGE.warped_darcy import _prepare_5point_transient_terms

    shape = (3, 4)
    rhs = np.full(shape, 2.0, dtype=np.float64)
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape, dtype=np.float64)
    bc_mask[:, 0] = 1
    bc_values[:, 0] = 11.0
    head_prev = np.full(shape, 7.0, dtype=np.float64)
    storage_coeff = np.full(shape, 1.0e-3, dtype=np.float64)
    dx = 100.0
    dt = 5.0

    b_first, sdiag_first, _, _, _ = _prepare_5point_transient_terms(
        rhs=rhs,
        storage_diag=None,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        transient=True,
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        initial_head=None,
        dx=dx,
    )
    b_second, sdiag_second, _, _, _ = _prepare_5point_transient_terms(
        rhs=rhs,
        storage_diag=sdiag_first,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        transient=True,
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        initial_head=None,
        dx=dx,
    )

    free = (active != 0) & (bc_mask == 0)
    expected_storage = storage_coeff * dx * dx / dt
    expected_storage[~free] = 0.0
    expected_rhs = rhs.copy()
    expected_rhs[free] = rhs[free] + expected_storage[free] * head_prev[free]

    np.testing.assert_allclose(sdiag_first, expected_storage)
    np.testing.assert_allclose(sdiag_second, expected_storage)
    np.testing.assert_allclose(b_first, expected_rhs)
    np.testing.assert_allclose(b_second, expected_rhs)


def test_storage_diagonal_kernel_components_are_consistent():
    from working_tests.run_storage_diagonal_kernel_diagnostics import (
        build_cases,
        test_apply_A,
        test_apply_A_and_pAp,
        test_compute_residual,
        test_fused_smoother,
        test_init_pcg_with_A,
        test_preconditioner,
    )

    cases_by_name = {case.name: case for case in build_cases()}
    case = cases_by_name["large_storage_small_diffusion"]
    checks = (
        test_apply_A,
        test_compute_residual,
        test_apply_A_and_pAp,
        test_init_pcg_with_A,
        test_preconditioner,
        test_fused_smoother,
    )

    for check in checks:
        max_error, rms_error, rel_error, diagnosis = check(case, "cpu")
        assert diagnosis == "PASS", (
            f"{check.__name__} failed: {diagnosis}; "
            f"max={max_error} rms={rms_error} rel={rel_error}"
        )


def test_storage_dominated_kcycle_matches_dense_reference():
    from working_tests.run_storage_diagonal_kernel_diagnostics import (
        build_cases,
        test_full_solver_storage_only,
    )

    cases_by_name = {case.name: case for case in build_cases()}
    for case_name in ("storage_only_uniform", "large_storage_small_diffusion"):
        case = cases_by_name[case_name]
        for max_levels in (1, 3):
            max_error, rms_error, diagnosis, converged = test_full_solver_storage_only(
                case=case,
                device="cpu",
                max_levels=max_levels,
            )
            assert converged
            assert diagnosis == "PASS", (
                f"{case_name} max_levels={max_levels} failed: {diagnosis}; "
                f"max={max_error} rms={rms_error}"
            )


def test_pcg_transient_is_rejected_instead_of_ignored():
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    solver = WarpDarcySolver(
        nx=4,
        ny=3,
        dx=100.0,
        device="cpu",
        solver_type="pcg",
    )

    with pytest.raises(NotImplementedError, match="does not support transient storage"):
        solver.solve(
            solver="pcg",
            transient=True,
            storage_coeff=1.0e-3,
            dt=1.0,
            return_info=True,
        )


def _build_confined_solver(nx: int = 16, ny: int = 12, device: str = "cpu"):
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    solver = WarpDarcySolver(
        nx=nx,
        ny=ny,
        dx=100.0,
        device=device,
        use_ghb=False,
        solver_type="kcycle",
        diag_preconditioner_backend="host",
    )
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_values = np.zeros((ny, nx), dtype=np.float64)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values[:, 0] = 12.0
    bc_values[:, -1] = 9.0
    recharge = np.full((ny, nx), 1.0e-5, dtype=np.float64)
    transmissivity = np.full((ny, nx), 10.0, dtype=np.float64)
    solver.build_from_fields(
        T_field=transmissivity,
        R_field=recharge,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
    )
    return solver, active, bc_mask


def _build_unconfined_solver(nx: int = 16, ny: int = 12, device: str = "cpu"):
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    solver = WarpDarcySolver(
        nx=nx,
        ny=ny,
        dx=100.0,
        device=device,
        use_ghb=False,
        solver_type="kcycle",
        diag_preconditioner_backend="host",
    )
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_values = np.zeros((ny, nx), dtype=np.float64)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values[:, 0] = 12.0
    bc_values[:, -1] = 9.0
    recharge = np.full((ny, nx), 1.0e-5, dtype=np.float64)
    transmissivity = np.full((ny, nx), 10.0, dtype=np.float64)
    solver.build_from_fields(
        T_field=transmissivity,
        R_field=recharge,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
    )
    K = np.full((ny, nx), 1.0, dtype=np.float64)
    zbot = np.zeros((ny, nx), dtype=np.float64)
    return solver, active, bc_mask, K, zbot


def _compute_mass_balance_residual(solver, head, head_prev, storage_coeff, dt, active, bc_mask):
    """
    Discrete mass balance residual for the transient 5-point stencil.
    Returns the relative residual (L2) of (S/dt)*(h-h_prev) - div(T*grad(h)) - R.
    """
    dx = float(solver.dx)
    cell_area = dx * dx

    h = np.asarray(head, dtype=np.float64)
    h_prev = np.asarray(head_prev, dtype=np.float64)
    S = np.asarray(storage_coeff, dtype=np.float64)
    T = np.asarray(solver.T_field_host, dtype=np.float64)
    R = np.asarray(solver.R_field_host, dtype=np.float64)
    act = np.asarray(active, dtype=np.int32)
    bc = np.asarray(bc_mask, dtype=np.int32)

    free = (act != 0) & (bc == 0)

    dx = float(solver.dx)
    cell_area = dx * dx

    # Storage term [L^3/T]
    storage_term = S[free] * cell_area * (h[free] - h_prev[free]) / dt

    # Flux divergence using harmonic mean conductance (matches kernel stencil) [L^3/T]
    tiny = 1.0e-12
    ny, nx = h.shape
    div = np.zeros((ny, nx), dtype=np.float64)

    # East/West
    if nx > 1:
        T_L = T[:, :-1]
        T_R = T[:, 1:]
        denom = T_L + T_R
        valid = (act[:, :-1] != 0) & (act[:, 1:] != 0) & (denom > tiny)
        cond = np.zeros_like(denom, dtype=np.float64)
        cond[valid] = 2.0 * T_L[valid] * T_R[valid] / denom[valid]
        div[:, :-1] += cond * (h[:, :-1] - h[:, 1:])
        div[:, 1:] += cond * (h[:, 1:] - h[:, :-1])

    # North/South
    if ny > 1:
        T_T = T[:-1, :]
        T_B = T[1:, :]
        denom = T_T + T_B
        valid = (act[:-1, :] != 0) & (act[1:, :] != 0) & (denom > tiny)
        cond = np.zeros_like(denom, dtype=np.float64)
        cond[valid] = 2.0 * T_T[valid] * T_B[valid] / denom[valid]
        div[:-1, :] += cond * (h[:-1, :] - h[1:, :])
        div[1:, :] += cond * (h[1:, :] - h[:-1, :])

    # Recharge in [L/T] -> multiply by cell area to get [L^3/T]
    residual = div[free] - R[free] * cell_area - storage_term

    denom = np.abs(storage_term) + np.abs(div[free]) + np.abs(R[free] * cell_area) + tiny
    rel = np.abs(residual) / denom
    return float(np.max(rel)), float(np.sqrt(np.mean(residual * residual)))


def test_confined_transient_runs_and_heads_decline_with_withdrawal():
    solver, active, bc_mask = _build_confined_solver()

    # Steady-state baseline
    head_ss, _ = solver.solve(
        formulation="confined",
        max_cycles=10,
        max_levels=3,
        min_coarse_cells=1,
        check_every_no=1,
        return_info=True,
    )

    # Transient step with withdrawal (negative recharge) relative to steady-state recharge
    storage_coeff = np.full((solver.ny, solver.nx), 1.0e-4, dtype=np.float64)
    dt = 86400.0  # 1 day
    head_prev = head_ss.copy()
    solver.R_field_host[:] = -1.0e-4  # net withdrawal [m/s]

    head_t, info_t = solver.solve(
        formulation="confined",
        transient=True,
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        initial_head=head_ss,
        max_cycles=10,
        max_levels=3,
        min_coarse_cells=1,
        check_every_no=1,
        return_info=True,
    )

    assert head_t.shape == (solver.ny, solver.nx)
    assert info_t.get("formulation") == "confined"
    # Withdrawal should lower heads relative to steady state
    assert float(np.mean(head_t)) < float(np.mean(head_ss))


def test_confined_transient_mass_balance():
    solver, active, bc_mask = _build_confined_solver(nx=16, ny=12)

    # Steady-state baseline as a well-conditioned initial guess
    head_ss, _ = solver.solve(
        formulation="confined",
        max_cycles=20,
        max_levels=3,
        min_coarse_cells=1,
        check_every_no=1,
        return_info=True,
    )

    storage_coeff = np.full((solver.ny, solver.nx), 1.0e-4, dtype=np.float64)
    dt = 86400.0
    head_prev = head_ss.copy()

    head_t, _ = solver.solve(
        formulation="confined",
        transient=True,
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        initial_head=head_ss,
        max_cycles=30,
        max_levels=3,
        min_coarse_cells=1,
        check_every_no=1,
        return_info=True,
    )

    max_rel, rms = _compute_mass_balance_residual(
        solver, head_t, head_prev, storage_coeff, dt, active, bc_mask
    )
    assert max_rel < 1.0e-2, f"confined transient mass balance max rel residual {max_rel}"
    assert rms < 1.0e-4, f"confined transient mass balance rms residual {rms}"


def test_unconfined_transient_runs():
    solver, active, bc_mask, K, zbot = _build_unconfined_solver()

    initial = zbot + 5.0
    head_ss, _ = solver.solve(
        formulation="unconfined",
        K_field=K,
        zbot_field=zbot,
        initial_head=initial,
        max_cycles=12,
        max_levels=3,
        min_coarse_cells=1,
        check_every_no=1,
        max_outer_iterations=6,
        hclose=1.0e-3,
        return_info=True,
    )

    storage_coeff = np.full((solver.ny, solver.nx), 1.0e-4, dtype=np.float64)
    dt = 86400.0
    head_prev = head_ss.copy()

    head_t, info_t = solver.solve(
        formulation="unconfined",
        K_field=K,
        zbot_field=zbot,
        initial_head=head_ss,
        transient=True,
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        max_cycles=12,
        max_levels=3,
        min_coarse_cells=1,
        check_every_no=1,
        max_outer_iterations=6,
        hclose=1.0e-3,
        return_info=True,
    )

    assert head_t.shape == (solver.ny, solver.nx)
    assert info_t.get("formulation") == "unconfined"
    # Transient solve should not be identical to the previous solution
    assert not np.allclose(head_t, head_ss, atol=1.0e-6)


def test_unconfined_transient_mass_balance():
    solver, active, bc_mask, K, zbot = _build_unconfined_solver(nx=16, ny=12)

    initial = zbot + 5.0
    head_ss, _ = solver.solve(
        formulation="unconfined",
        K_field=K,
        zbot_field=zbot,
        initial_head=initial,
        max_cycles=15,
        max_levels=3,
        min_coarse_cells=1,
        check_every_no=1,
        max_outer_iterations=8,
        hclose=1.0e-3,
        return_info=True,
    )

    storage_coeff = np.full((solver.ny, solver.nx), 1.0e-4, dtype=np.float64)
    dt = 86400.0
    head_prev = head_ss.copy()

    head_t, _ = solver.solve(
        formulation="unconfined",
        K_field=K,
        zbot_field=zbot,
        initial_head=head_ss,
        transient=True,
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        max_cycles=15,
        max_levels=3,
        min_coarse_cells=1,
        check_every_no=1,
        max_outer_iterations=8,
        hclose=1.0e-3,
        return_info=True,
    )

    max_rel, rms = _compute_mass_balance_residual(
        solver, head_t, head_prev, storage_coeff, dt, active, bc_mask
    )
    assert max_rel < 2.0e-1, f"unconfined transient mass balance max rel residual {max_rel}"
    assert rms < 2.0e-4, f"unconfined transient mass balance rms residual {rms}"


def test_replay_initial_transmissivity_caps_at_top():
    from working_tests.transient_replay_storage import _initial_transmissivity

    k = np.array([[2.0, 2.0]], dtype=np.float64)
    initial_head = np.array([[15.0, 8.0]], dtype=np.float64)
    top = np.array([[10.0, 10.0]], dtype=np.float64)
    bottom = np.array([[0.0, 0.0]], dtype=np.float64)
    active = np.ones((1, 2), dtype=np.int32)

    transmissivity = _initial_transmissivity(
        k=k,
        initial_head=initial_head,
        top=top,
        bottom=bottom,
        active=active,
        min_sat=0.1,
    )

    np.testing.assert_allclose(transmissivity, np.array([[20.0, 16.0]], dtype=np.float64))


def test_2d_transient_replay_steps_periods_and_responds_to_recharge(monkeypatch):
    """
    Multi-period Warp transient replay harness (MF6-free).

    Exercises the MF6-free transient replay core on a
    small synthetic case: heads must evolve across periods, the final state must
    differ from the initial condition, and a larger recharge rate must mound
    heads higher than a smaller one. No MF6/flopy dependency.
    """
    from working_tests.transient_artifacts import build_synthetic_spatial_fields
    from working_tests.transient_replay_support import run_warp_transient_replay

    spatial = build_synthetic_spatial_fields(nx=16, ny=12, hydraulic_conductivity=100.0)
    fast_controls = {
        "max_cycles": 30,
        "max_levels": 4,
        "max_outer_iterations": 30,
        "min_coarse_cells": 1,
    }

    rates = np.array([1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4], dtype=np.float64)
    result = run_warp_transient_replay(
        spatial,
        recharge_rates=rates,
        sy=0.2,
        dt=7.0,
        n_periods=4,
        device="cpu",
        diag_preconditioner_backend="host",
        solve_controls=fast_controls,
    )

    heads = result["heads_per_period"]
    assert heads.shape == (4, 12, 16)
    assert np.all(np.isfinite(heads))
    # Storage + forcing must move heads away from the initial condition.
    assert not np.allclose(heads[-1], spatial["initial_head"], atol=1.0e-8)
    assert bool(result["last_info"].get("converged", False))

    # Force device RHS assembly to catch stale host-only recharge updates.
    monkeypatch.setenv("DARCY_RHS_MODE", "device")

    # Recharge magnitude must matter: a high-rate single step mounds heads
    # higher than a low-rate single step from the same initial condition.
    low = run_warp_transient_replay(
        spatial,
        recharge_rates=np.array([1.0e-6], dtype=np.float64),
        sy=0.2,
        dt=7.0,
        n_periods=1,
        device="cpu",
        diag_preconditioner_backend="host",
        solve_controls=fast_controls,
    )
    high = run_warp_transient_replay(
        spatial,
        recharge_rates=np.array([1.0e-2], dtype=np.float64),
        sy=0.2,
        dt=7.0,
        n_periods=1,
        device="cpu",
        diag_preconditioner_backend="host",
        solve_controls=fast_controls,
    )
    active = spatial["active"] != 0
    assert float(high["heads_final"][active].mean()) > float(low["heads_final"][active].mean())


def _cuda_available() -> bool:
    if not _warp_available():
        return False
    try:
        import warp as wp

        return bool(wp.is_cuda_available())
    except Exception:
        return False


def _build_fast_path_unconfined_solver(nx=64, ny=48):
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    solver = WarpDarcySolver(
        nx=nx, ny=ny, dx=100.0, device="cuda:0",
        use_ghb=False, solver_type="kcycle", diag_preconditioner_backend="host",
    )
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_values = np.zeros((ny, nx), dtype=np.float64)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values[:, 0] = 12.0
    bc_values[:, -1] = 11.0
    solver.build_from_fields(
        T_field=np.full((ny, nx), 10.0, dtype=np.float64),
        R_field=np.full((ny, nx), 1.0e-5, dtype=np.float64),
        active=active, bc_mask=bc_mask, bc_values=bc_values,
    )
    zbot = np.zeros((ny, nx), dtype=np.float64)
    ztop = np.full((ny, nx), 20.0, dtype=np.float64)
    k = np.full((ny, nx), 1.0, dtype=np.float64)
    h0 = np.full((ny, nx), 11.5, dtype=np.float64)
    h0[:, 0] = 12.0
    h0[:, -1] = 11.0
    return solver, k, zbot, ztop, h0


def _run_fast_path(inc: bool):
    from working_tests.transient_replay_settings import default_solve_controls

    solver, k, zbot, ztop, h0, *_ = _build_fast_path_unconfined_solver()
    sc = default_solve_controls()
    sc["use_device_transient_fast_path"] = True
    sc["use_incremental_picard"] = inc
    sc["allow_unaccepted_transient_period"] = True
    heads, info = solver.solve_transient_2d_unconfined(
        initial_head=h0,
        # Near-linear: tiny recharge, 1-day step -> dh tiny, T/storage_diag ~constant,
        # Picard converges in ~1 outer iteration so the incremental (correction) and
        # direct-head inner solves must agree to ~machine precision.
        recharge_rates=np.array([5.0e-9, 5.0e-9], dtype=np.float64),
        k_field=k, zbot_field=zbot, ztop_field=ztop,
        sy=0.2, ss=1.0e-4, dt=86400.0,
        storage_mode="mf6_convertible_secant_sy",
        storage_reference="current_picard",
        solve_controls=sc,
        return_info=True,
    )
    return heads, info


@pytest.mark.skipif(not _cuda_available(), reason="CUDA fast path is not available")
def test_incremental_picard_matches_direct_head_path():
    """The incremental Picard form (solve A*delta = r^k, delta=0 on Dirichlet) is
    mathematically equivalent to the direct-head form when the inner solve is exact.
    On a near-linear problem both must agree to ~machine precision, and the
    ``incremental_picard_enabled`` flag must be reported per period."""
    heads_direct, info_direct = _run_fast_path(inc=False)
    heads_inc, info_inc = _run_fast_path(inc=True)

    assert np.all(np.isfinite(heads_direct)) and np.all(np.isfinite(heads_inc))

    period_infos_direct = info_direct.get("period_infos") or []
    period_infos_inc = info_inc.get("period_infos") or []
    assert period_infos_direct and period_infos_inc
    assert period_infos_direct[0]["incremental_picard_enabled"] is False
    assert period_infos_inc[0]["incremental_picard_enabled"] is True

    # The two inner-solve strategies must produce the same head field.
    max_abs_diff = float(np.max(np.abs(heads_inc - heads_direct)))
    assert max_abs_diff < 1.0e-5, f"incremental vs direct max abs diff {max_abs_diff}"


def _run_adaptive_dt_case(*, recharge: float, adaptive_controls: dict, n_periods: int = 2, dt: float = 86400.0, startup_mode: str = "confined_pre_solve"):
    from working_tests.transient_replay_settings import default_solve_controls

    solver, k, zbot, ztop, h0, *_ = _build_fast_path_unconfined_solver()
    sc = default_solve_controls()
    sc["use_device_transient_fast_path"] = True
    sc["allow_unaccepted_transient_period"] = True
    sc["unconfined_startup_mode"] = startup_mode
    sc.update(adaptive_controls)
    heads, info = solver.solve_transient_2d_unconfined(
        initial_head=h0,
        recharge_rates=np.full(n_periods, recharge, dtype=np.float64),
        k_field=k, zbot_field=zbot, ztop_field=ztop,
        sy=0.2, ss=1.0e-4, dt=dt,
        storage_mode="mf6_convertible_secant_sy",
        storage_reference="current_picard",
        solve_controls=sc,
        return_info=True,
    )
    return heads, info


@pytest.mark.skipif(not _cuda_available(), reason="CUDA fast path is not available")
def test_adaptive_dt_is_noop_when_strict_passes_at_full_dt():
    """On an easy (near-linear) problem strict Picard converges within the strict
    outer budget at the full period dt, so adaptive sub-stepping must never engage:
    exactly one sub-step per period at full dt, and heads identical to the
    adaptive-off run (acceptance point is the same iterate)."""
    heads_off, info_off = _run_adaptive_dt_case(
        recharge=5.0e-9, adaptive_controls={"adaptive_dt_enabled": False}
    )
    heads_on, info_on = _run_adaptive_dt_case(
        recharge=5.0e-9, adaptive_controls={"adaptive_dt_enabled": True}
    )
    for info in (info_on.get("period_infos") or []):
        assert info["adaptive_dt_enabled"] is True
        assert info["adaptive_dt_substep_count"] == 1
        assert info["adaptive_dt_substep_dts"] == [86400.0]
        assert info["adaptive_dt_retry_count"] == 0
    assert np.all(np.isfinite(heads_on))
    max_abs_diff = float(np.max(np.abs(heads_on - heads_off)))
    assert max_abs_diff < 1.0e-6, f"adaptive on/off max abs diff {max_abs_diff}"


@pytest.mark.skipif(not _cuda_available(), reason="CUDA fast path is not available")
def test_default_controls_converge_strict_not_practical():
    """Regression test for the 1M-cell premature-acceptance failure.

    The production failure was caused by the practical-acceptance floor firing
    at outer iteration 8, ~3 iterations before strict Picard success (strict
    needs ~11 outer iterations worst-case; dh_max contracts ~0.31x/outer).
    With the corrected defaults (min_practical_outer_iterations=20,
    adaptive_dt_strict_max_outer=20), every period must converge via STRICT
    Picard at full dt: single sub-step, dh_max <= hclose, no practical
    fallback anywhere."""
    heads, info = _run_adaptive_dt_case(
        recharge=5.0e-9, adaptive_controls={}  # production defaults verbatim
    )
    period_infos = info.get("period_infos") or []
    assert len(period_infos) == 2
    for pinfo in period_infos:
        assert pinfo["strict_picard_convergence_passed"] is True
        assert pinfo["adaptive_dt_substep_count"] == 1
        assert pinfo["adaptive_dt_practical_fallback_count"] == 0
        assert pinfo["final_max_abs_head_change"] <= 1.0e-4
    assert np.all(np.isfinite(heads))


@pytest.mark.skipif(not _cuda_available(), reason="CUDA fast path is not available")
def test_adaptive_dt_shrink_fallback_bookkeeping_and_reference_match():
    """With a strict budget of 1 outer iteration and a strong transient, strict
    acceptance is impossible at any dt. The driver must shrink dt by the shrink
    factor down to dt_min = period/16 (4 shrinks), then accept each sub-step via
    practical acceptance at dt_min WITHOUT re-growing afterwards (no retry storm),
    and the sub-stepped trajectory must match a fixed small-dt reference run."""
    heads, info = _run_adaptive_dt_case(
        recharge=5.0e-7,
        adaptive_controls={
            "adaptive_dt_enabled": True,
            "adaptive_dt_strict_max_outer": 1,
        },
        # initial_head startup (no confined pre-solve): keeps the first Picard
        # iteration's residual large so strict acceptance inside the strict budget
        # genuinely fails, exercising the sub-step machinery.
        startup_mode="initial_head",
    )
    period_infos = info.get("period_infos") or []
    assert len(period_infos) == 2
    dt_min = 86400.0 / 16.0
    for pinfo in period_infos:
        assert pinfo["adaptive_dt_substep_count"] == 16
        assert pinfo["adaptive_dt_substep_dts"] == pytest.approx([dt_min] * 16)
        assert sum(pinfo["adaptive_dt_substep_dts"]) == pytest.approx(86400.0)
        # 4 shrinks (dt -> dt/2 -> dt/4 -> dt/8 -> dt/16) on the first sub-step only;
        # a pre-fix regression grew dt again after every practical acceptance and
        # logged ~18 retries per period (retry storm).
        assert pinfo["adaptive_dt_retry_count"] == 4
        assert pinfo["adaptive_dt_practical_fallback_count"] == 16
        assert pinfo["adaptive_dt_total_outer_iterations"] > 16
        assert pinfo["strict_picard_convergence_passed"] or pinfo["practical_picard_acceptance_passed"]
    assert np.all(np.isfinite(heads))

    # Reference: identical physics as 32 fixed periods of dt/16, adaptive off.
    heads_ref, _ = _run_adaptive_dt_case(
        recharge=5.0e-7,
        adaptive_controls={"adaptive_dt_enabled": False},
        n_periods=32,
        dt=86400.0 / 16.0,
        startup_mode="initial_head",
    )
    diff = np.abs(heads[-1] - heads_ref[-1])
    assert float(np.max(diff)) < 1.0e-8, f"sub-stepped vs fixed-dt reference max abs diff {float(np.max(diff))}"


@pytest.mark.skipif(not _cuda_available(), reason="CUDA fast path is not available")
def test_adaptive_dt_rejects_invalid_controls():
    """Adaptive-dt control validation must reject inconsistent budgets and factors."""
    with pytest.raises(ValueError, match="adaptive_dt_strict_max_outer must be <="):
        _run_adaptive_dt_case(
            recharge=5.0e-9,
            adaptive_controls={"adaptive_dt_enabled": True, "adaptive_dt_strict_max_outer": 10000},
        )
    with pytest.raises(ValueError, match="adaptive_dt_min_fraction"):
        _run_adaptive_dt_case(
            recharge=5.0e-9,
            adaptive_controls={"adaptive_dt_enabled": True, "adaptive_dt_min_fraction": 0.0},
        )
    with pytest.raises(ValueError, match="adaptive_dt_shrink_factor"):
        _run_adaptive_dt_case(
            recharge=5.0e-9,
            adaptive_controls={"adaptive_dt_enabled": True, "adaptive_dt_shrink_factor": 1.0},
        )


def test_adaptive_dt_projection_helper_geometric_contraction():
    """Projection math: geometric dh contraction -> exact iterations-to-tol."""
    from DARCY_WARP_PACKAGE.warped_darcy import _adaptive_dt_projected_outer_to_tol

    history = [1.0, 0.31, 0.0961]
    needed = _adaptive_dt_projected_outer_to_tol(history, tol=1.0e-4)
    expected = float(np.log(1.0e-4 / 0.0961) / np.log(0.31))
    assert needed == pytest.approx(expected, rel=1.0e-6)
    assert _adaptive_dt_projected_outer_to_tol([1.0e-5, 9.0e-6], tol=1.0e-4) == 0.0
    assert _adaptive_dt_projected_outer_to_tol([1.0, 1.1, 1.2], tol=1.0e-4) == float("inf")
    assert _adaptive_dt_projected_outer_to_tol([1.0], tol=1.0e-4) is None
    assert _adaptive_dt_projected_outer_to_tol([], tol=1.0e-4) is None
    assert _adaptive_dt_projected_outer_to_tol([1.0, float("nan"), 0.5], tol=1.0e-4) is not None


def test_adaptive_dt_early_shrink_decision():
    """Early shrink fires iff projected strict iterations exceed remaining budget."""
    from DARCY_WARP_PACKAGE.warped_darcy import _adaptive_dt_should_early_shrink

    slow = [1.0, 0.9, 0.81, 0.729, 0.656, 0.590]  # ratio ~0.9: ~82 more iters needed
    fast = [1.0, 0.31, 0.096, 0.0298, 0.00924, 0.00286]  # ratio ~0.31: ~3 more
    assert _adaptive_dt_should_early_shrink(
        slow, tol=1.0e-4, outer_iterations_done=6, budget=20, min_outer=6
    ) is True
    assert _adaptive_dt_should_early_shrink(
        fast, tol=1.0e-4, outer_iterations_done=6, budget=20, min_outer=6
    ) is False
    # Below min_outer no projection is attempted even on a stalled history.
    assert _adaptive_dt_should_early_shrink(
        slow[:3], tol=1.0e-4, outer_iterations_done=3, budget=20, min_outer=6
    ) is False
    # dh already <= tol means the residual is the blocker: never early-shrink.
    assert _adaptive_dt_should_early_shrink(
        [1.0e-5, 9.0e-6, 8.0e-6], tol=1.0e-4, outer_iterations_done=6, budget=20, min_outer=6
    ) is False


def test_adaptive_dt_extension_decision():
    """Budget extension requires closeness plus contraction (or dh already <= tol)."""
    from DARCY_WARP_PACKAGE.warped_darcy import _adaptive_dt_should_extend_budget

    # Close (<=5x tol) and contracting at 0.3: extend.
    assert _adaptive_dt_should_extend_budget(
        [0.3, 0.09, 3.0e-4], tol=1.0e-4, extension_factor=5.0, extension_contraction_ratio=0.8
    ) is True
    # Beyond extension_factor x tol: no extension.
    assert _adaptive_dt_should_extend_budget(
        [0.3, 0.09, 6.0e-4], tol=1.0e-4, extension_factor=5.0, extension_contraction_ratio=0.8
    ) is False
    # Close but stalled (ratio ~1.0): no extension.
    assert _adaptive_dt_should_extend_budget(
        [3.0e-4, 3.1e-4, 3.0e-4], tol=1.0e-4, extension_factor=5.0, extension_contraction_ratio=0.8
    ) is False
    # dh already <= tol (residual is the blocker): extend.
    assert _adaptive_dt_should_extend_budget(
        [1.0e-5], tol=1.0e-4, extension_factor=5.0, extension_contraction_ratio=0.8
    ) is True
    assert _adaptive_dt_should_extend_budget(
        [], tol=1.0e-4, extension_factor=5.0, extension_contraction_ratio=0.8
    ) is False
