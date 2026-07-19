"""Multi-timestep / multi-stress-period transient regression tests.

Covers the experimental transient driver (``solvers/transient_experimental.py``)
driving ``unconfined_fas`` (and, where relevant, ``unconfined_picard_kcycle``
for switching/comparison) through complete transient simulations: sub-stepping
within stress periods, stress-period transitions, retry, fallback, state
reset, and previous-head propagation.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("WARP_CACHE_PATH", str(Path("/tmp/darcywarp-warp-cache")))

wp = pytest.importorskip("warp")


CASE_NY, CASE_NX, CASE_DX = 12, 16, 50.0


def _case_fields(*, top_offset: float = 100.0, initial: float = 40.0, bc_left: float = 45.0):
    y, x = np.mgrid[:CASE_NY, :CASE_NX]
    k = np.full((CASE_NY, CASE_NX), 20.0)
    bottom = np.zeros((CASE_NY, CASE_NX))
    top = np.full((CASE_NY, CASE_NX), top_offset)
    active = np.ones((CASE_NY, CASE_NX), dtype=np.int32)
    bc_mask = np.zeros((CASE_NY, CASE_NX), dtype=np.int32)
    bc_mask[:, 0] = 1
    bc_values = np.zeros((CASE_NY, CASE_NX), dtype=np.float64)
    bc_values[:, 0] = bc_left
    initial_head = np.full((CASE_NY, CASE_NX), initial, dtype=np.float64)
    initial_head[:, 0] = bc_left
    return {
        "k": k,
        "bottom": bottom,
        "top": top,
        "active": active,
        "bc_mask": bc_mask,
        "bc_values": bc_values,
        "initial": initial_head,
    }


def _make_solver(device: str = "cpu"):
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    return WarpDarcySolver(
        nx=CASE_NX,
        ny=CASE_NY,
        dx=CASE_DX,
        device=device,
        solver_type="kcycle",
        diag_preconditioner_backend="device",
    )


def _run_transient(
    solver,
    case,
    *,
    backend: str = "unconfined_fas",
    rates=(1.0e-4,),
    dt=7.0,
    sy: float = 0.1,
    ss: float = 1.0e-5,
    controls: dict | None = None,
    initial_head=None,
    **api_kwargs,
):
    solve_controls = {"save_transient_diagnostics": True}
    solve_controls.update(controls or {})
    heads, info = solver.solve_transient_2d_unconfined(
        solver=backend,
        initial_head=case["initial"] if initial_head is None else initial_head,
        recharge_rates=np.asarray(rates, dtype=np.float64),
        k_field=case["k"],
        zbot_field=case["bottom"],
        ztop_field=case["top"],
        sy=sy,
        ss=ss,
        dt=dt,
        active=case["active"],
        bc_mask=case["bc_mask"],
        bc_values=case["bc_values"],
        storage_mode="mf6_convertible_secant_sy",
        storage_reference="current_picard",
        solve_controls=solve_controls,
        return_info=True,
        **api_kwargs,
    )
    return heads, info


def _accepted_records(info):
    return [rec for rec in info["experimental_timestep_records"] if rec["accepted"]]


def _workspace(solver):
    return solver._resource_owner.get_experimental_workspace("unconfined_fas")


# ---------------------------------------------------------------------------
# 1-2. Timesteps within and across stress periods
# ---------------------------------------------------------------------------


def test_01_single_period_multiple_timesteps():
    case = _case_fields()
    solver = _make_solver()
    try:
        heads, info = _run_transient(
            solver, case, rates=(1.0e-4,), dt=7.0,
            controls={"experimental_max_dt": 1.75},
        )
    finally:
        solver.close()
    accepted = _accepted_records(info)
    assert len(accepted) == 4
    assert [rec["dt"] for rec in accepted] == [1.75] * 4
    assert [rec["timestep_index"] for rec in accepted] == [0, 1, 2, 3]
    assert info["simulation_time"] == pytest.approx(7.0)
    assert heads.shape == (1, CASE_NY, CASE_NX)
    assert np.all(np.isfinite(heads))
    assert info["period_infos"][0]["converged"]
    assert info["period_infos"][0]["experimental_timestep_count"] == 4
    assert abs(info["experimental_period_budgets"][0]["percent_discrepancy"]) < 0.1


def test_02_multiple_periods_multiple_timesteps_each():
    case = _case_fields()
    solver = _make_solver()
    try:
        heads, info = _run_transient(
            solver, case, rates=(1.0e-4, 1.5e-4, 5.0e-5), dt=7.0,
            controls={"experimental_max_dt": 3.5},
        )
    finally:
        solver.close()
    accepted = _accepted_records(info)
    assert len(accepted) == 6
    assert [rec["stress_period_index"] for rec in accepted] == [0, 0, 1, 1, 2, 2]
    assert [rec["timestep_index"] for rec in accepted] == [0, 1, 0, 1, 0, 1]
    assert heads.shape == (3, CASE_NY, CASE_NX)
    assert np.all(np.isfinite(heads))
    assert all(p["converged"] for p in info["period_infos"])
    assert info["simulation_time"] == pytest.approx(21.0)


def test_03_artificial_period_boundaries_do_not_change_solution():
    case = _case_fields()
    # Run A: one 14-day period capped at 7-day timesteps -> two 7-day steps.
    solver_a = _make_solver()
    try:
        heads_a, info_a = _run_transient(
            solver_a, case, rates=(1.2e-4,), dt=14.0,
            controls={"experimental_max_dt": 7.0},
        )
    finally:
        solver_a.close()
    # Run B: two 7-day periods with identical constant stress -> same steps.
    solver_b = _make_solver()
    try:
        heads_b, info_b = _run_transient(
            solver_b, case, rates=(1.2e-4, 1.2e-4), dt=7.0,
        )
    finally:
        solver_b.close()
    dts_a = [rec["dt"] for rec in _accepted_records(info_a)]
    dts_b = [rec["dt"] for rec in _accepted_records(info_b)]
    assert dts_a == dts_b == [7.0, 7.0]
    np.testing.assert_allclose(heads_a[-1], heads_b[-1], rtol=0.0, atol=1.0e-9)
    assert info_a["simulation_time"] == pytest.approx(info_b["simulation_time"])


# ---------------------------------------------------------------------------
# 4-6. Stress-period data changes
# ---------------------------------------------------------------------------


def test_04_recharge_change_between_periods():
    # Balanced case (boundary head == initial head) so the recharge signal is
    # not masked by boundary throughflow.
    case = _case_fields(initial=40.0, bc_left=40.0)
    solver = _make_solver()
    try:
        heads, info = _run_transient(solver, case, rates=(3.0e-3, 0.0), dt=7.0)
    finally:
        solver.close()
    free = (case["active"] != 0) & (case["bc_mask"] == 0)
    mean_wet = float(np.mean(heads[0][free]))
    mean_dry = float(np.mean(heads[1][free]))
    # Wet period raises heads above the initial state; dry period relaxes back.
    assert mean_wet > float(np.mean(case["initial"][free]))
    assert mean_dry < mean_wet
    budgets = info["experimental_period_budgets"]
    assert budgets[0]["rcha_in"] > 0.0
    assert budgets[1]["rcha_in"] == pytest.approx(0.0)
    assert all(p["converged"] for p in info["period_infos"])


def test_05_negative_withdrawal_change_between_periods():
    case = _case_fields(initial=40.0, bc_left=40.0)
    shape = (CASE_NY, CASE_NX)
    sources = np.stack(
        [
            np.full(shape, 2.0e-3),   # period 1: recharge
            np.full(shape, -2.0e-3),  # period 2: aggregated withdrawal
        ]
    )
    solver = _make_solver()
    try:
        heads, info = _run_transient(
            solver, case, rates=(0.0, 0.0), dt=7.0,
            source_fields_per_period=sources,
        )
    finally:
        solver.close()
    budgets = info["experimental_period_budgets"]
    assert budgets[0]["rcha_in"] > 0.0
    assert budgets[1]["rcha_out"] > 0.0
    free = (case["active"] != 0) & (case["bc_mask"] == 0)
    assert float(np.mean(heads[1][free])) < float(np.mean(heads[0][free]))
    assert all(p["converged"] for p in info["period_infos"])


def test_06_prescribed_head_change_between_periods():
    case = _case_fields()
    bc_seq = np.stack([case["bc_values"], case["bc_values"] + 5.0])
    solver = _make_solver()
    try:
        heads, info = _run_transient(
            solver, case, rates=(1.0e-4, 1.0e-4), dt=7.0,
            bc_values_per_period=bc_seq,
        )
    finally:
        solver.close()
    np.testing.assert_allclose(heads[0][:, 0], 45.0, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(heads[1][:, 0], 50.0, rtol=0.0, atol=1.0e-9)
    # Interior responds to the raised boundary.
    assert float(np.mean(heads[1][:, 1:])) > float(np.mean(heads[0][:, 1:]))
    assert all(p["converged"] for p in info["period_infos"])


def test_07_changing_timestep_lengths():
    case = _case_fields()
    solver = _make_solver()
    try:
        heads, info = _run_transient(
            solver, case, rates=(1.0e-4, 1.0e-4, 1.0e-4),
            dt=[7.0, 3.5, 14.0],
        )
    finally:
        solver.close()
    accepted = _accepted_records(info)
    assert [rec["dt"] for rec in accepted] == [7.0, 3.5, 14.0]
    assert info["simulation_time"] == pytest.approx(24.5)
    assert heads.shape == (3, CASE_NY, CASE_NX)
    assert all(p["converged"] for p in info["period_infos"])


# ---------------------------------------------------------------------------
# 8-10. Storage formulations
# ---------------------------------------------------------------------------


def _storage_terms(info):
    return (
        info["storage_terms_per_period"],
        info["sy_storage_terms_per_period"],
        info["ss_storage_terms_per_period"],
    )


def test_08_sy_only_storage():
    case = _case_fields()
    solver = _make_solver()
    try:
        _, info = _run_transient(solver, case, rates=(2.0e-4, 2.0e-4), sy=0.15, ss=0.0)
    finally:
        solver.close()
    total, sy_terms, ss_terms = _storage_terms(info)
    assert np.all(np.abs(ss_terms) == 0.0)
    assert np.any(np.abs(sy_terms) > 0.0)
    assert all(p["converged"] for p in info["period_infos"])


def test_09_ss_only_storage():
    case = _case_fields()
    solver = _make_solver()
    try:
        _, info = _run_transient(solver, case, rates=(2.0e-4, 2.0e-4), sy=0.0, ss=1.0e-4)
    finally:
        solver.close()
    total, sy_terms, ss_terms = _storage_terms(info)
    assert np.all(np.abs(sy_terms) == 0.0)
    assert np.any(np.abs(ss_terms) > 0.0)
    assert all(p["converged"] for p in info["period_infos"])


def test_10_combined_sy_ss_storage():
    case = _case_fields()
    solver = _make_solver()
    try:
        _, info = _run_transient(solver, case, rates=(2.0e-4, 2.0e-4), sy=0.1, ss=1.0e-4)
    finally:
        solver.close()
    total, sy_terms, ss_terms = _storage_terms(info)
    assert np.any(np.abs(sy_terms) > 0.0)
    assert np.any(np.abs(ss_terms) > 0.0)
    np.testing.assert_allclose(total, sy_terms + ss_terms, rtol=1.0e-9, atol=1.0e-15)
    assert all(p["converged"] for p in info["period_infos"])


def test_11_bottom_and_top_storage_crossings_over_timesteps():
    # Thin aquifer over a ramped bottom. Period 1 withdrawal is sized to just
    # exhaust the deliverable specific-yield storage (heads settle onto the
    # min_sat floor); period 2 recharge pushes heads through the top into the
    # confined (Ss) region.  Withdrawal stays within what storage can deliver
    # so every timestep still closes its budget.
    case = _case_fields()
    y, x = np.mgrid[:CASE_NY, :CASE_NX]
    bottom = 44.0 + 0.1 * x
    top = bottom + 0.5
    case["bottom"] = bottom.astype(np.float64)
    case["top"] = top.astype(np.float64)
    case["initial"] = (bottom + 0.2).astype(np.float64)
    case["bc_values"][:, 0] = case["initial"][:, 0]
    shape = (CASE_NY, CASE_NX)
    sources = np.stack([np.full(shape, -1.2e-3), np.full(shape, 1.0e-2)])
    solver = _make_solver()
    try:
        heads, info = _run_transient(
            solver, case, rates=(0.0, 0.0), dt=7.0,
            sy=0.1, ss=1.0e-4,
            controls={"experimental_max_dt": 3.5},
            source_fields_per_period=sources,
        )
    finally:
        solver.close()
    assert np.all(np.isfinite(heads))
    assert all(p["converged"] for p in info["period_infos"])
    free = (case["active"] != 0) & (case["bc_mask"] == 0)
    # Period 1 sits heads on (or near) the bottom floor somewhere;
    # period 2 pushes heads through the top somewhere.
    assert float(np.min((heads[0] - case["bottom"])[free])) <= 0.12
    assert float(np.max((heads[1] - case["top"])[free])) > 0.0
    budgets = info["experimental_period_budgets"]
    # Period 1 releases stored water (inflow); period 2 stores it (outflow).
    assert budgets[0]["storage_in"] > 0.0
    assert budgets[1]["storage_out"] > 0.0


# ---------------------------------------------------------------------------
# 12-15. Previous-head propagation, retry, state reset
# ---------------------------------------------------------------------------


def test_12_accepted_previous_head_propagation():
    case = _case_fields()
    solver = _make_solver()
    try:
        _, info = _run_transient(
            solver, case, rates=(1.0e-4,), dt=7.0,
            controls={"experimental_max_dt": 7.0 / 3.0},
        )
    finally:
        solver.close()
    accepted = _accepted_records(info)
    assert len(accepted) == 3
    np.testing.assert_array_equal(accepted[0]["previous_head"], case["initial"])
    for later, earlier in zip(accepted[1:], accepted[:-1]):
        np.testing.assert_array_equal(later["previous_head"], earlier["accepted_head"])


def _make_flaky_solve(solver, calls, *, garbage_head: bool):
    original_solve = solver.solve

    def flaky_solve(**kwargs):
        calls["n"] += 1
        head, info = original_solve(**kwargs)
        if calls["n"] == 1:
            bad = dict(info)
            bad["converged"] = False
            bad["fas_failure_reason"] = "forced_test_failure"
            if garbage_head:
                return np.full_like(head, 9.0e3), bad
            return head, bad
        return head, info

    return flaky_solve


def test_13_rejected_timestep_followed_by_retry(monkeypatch):
    case = _case_fields()
    solver = _make_solver()
    calls = {"n": 0}
    monkeypatch.setattr(solver, "solve", _make_flaky_solve(solver, calls, garbage_head=False))
    try:
        heads, info = _run_transient(solver, case, rates=(1.0e-4,), dt=7.0)
    finally:
        solver.close()
    records = info["experimental_timestep_records"]
    rejected = [rec for rec in records if not rec["accepted"]]
    accepted = _accepted_records(info)
    assert len(rejected) == 1
    assert rejected[0]["failure_reason"] == "forced_test_failure"
    assert rejected[0]["dt"] == pytest.approx(7.0)
    # The retry runs at the shrunk timestep and completes the period.
    assert accepted[0]["retry_count"] == 1
    assert accepted[0]["dt"] == pytest.approx(3.5)
    assert info["simulation_time"] == pytest.approx(7.0)
    assert np.all(np.isfinite(heads))
    assert info["period_infos"][0]["experimental_retry_count"] == 1


def test_14_rejected_trial_heads_are_not_propagated(monkeypatch):
    case = _case_fields()
    solver = _make_solver()
    calls = {"n": 0}
    monkeypatch.setattr(solver, "solve", _make_flaky_solve(solver, calls, garbage_head=True))
    try:
        heads, info = _run_transient(solver, case, rates=(1.0e-4,), dt=7.0)
    finally:
        solver.close()
    accepted = _accepted_records(info)
    # The retry's previous head is the last ACCEPTED head (the initial head),
    # not the rejected 9000 m trial head.
    np.testing.assert_array_equal(accepted[0]["previous_head"], case["initial"])
    assert float(np.max(heads)) < 100.0
    assert float(np.min(heads)) > 0.0


def test_14b_retry_first_policy_defers_backend_fallback_to_dt_min(monkeypatch):
    """Default policy: backend fallback disabled in the retry window, offered
    only on the last-resort attempt at dt_min."""
    case = _case_fields()
    solver = _make_solver()
    original_solve = solver.solve
    seen_flags = []

    def failing_solve(**kwargs):
        seen_flags.append(kwargs.get("fas_fallback_enabled", "unset"))
        head, info = original_solve(**kwargs)
        bad = dict(info)
        bad["converged"] = False
        bad["fas_failure_reason"] = "forced_test_failure"
        return head, bad

    monkeypatch.setattr(solver, "solve", failing_solve)
    try:
        with pytest.raises(RuntimeError, match="dt_min"):
            _run_transient(solver, case, rates=(1.0e-4,), dt=7.0)
    finally:
        solver.close()
    # dt shrinks 7 -> 3.5 -> 1.75 -> 0.875 -> 0.4375 (= dt_min = 7/16):
    # fallback disabled for the retry window, enabled only at dt_min.
    assert seen_flags == [False, False, False, False, True]


def test_15_timestep_local_state_is_reset_between_solves():
    case = _case_fields()
    solver = _make_solver()
    try:
        _, info_first = _run_transient(solver, case, rates=(1.0e-4, 1.2e-4), dt=7.0)
        workspace = _workspace(solver)
        assert workspace is not None
        launches_first = [lvl.nonlinear_operator.kernel_launches for lvl in workspace.levels]
        _, info_second = _run_transient(solver, case, rates=(1.0e-4, 1.2e-4), dt=7.0)
        workspace_after = _workspace(solver)
        launches_second = [lvl.nonlinear_operator.kernel_launches for lvl in workspace_after.levels]
    finally:
        solver.close()
    # Workspace is reused (refresh, not rebuild) and per-solve counters reset:
    # identical repeated runs leave identical, non-accumulating counts.
    assert workspace_after is workspace
    assert workspace.refresh_count == info_second["last_info"]["fas_workspace_refresh_count"]
    assert launches_second == launches_first
    # Cycle histories are per-solve, not cumulative across the simulation.
    for rec in _accepted_records(info_second):
        assert rec["fas_cycles"] is not None and rec["fas_cycles"] < 30


# ---------------------------------------------------------------------------
# 16-17. Backend switching and fallback
# ---------------------------------------------------------------------------


def test_16_backend_switching_picard_fas_picard():
    case = _case_fields()
    solver = _make_solver()
    try:
        heads_p1, info_p1 = _run_transient(
            solver, case, backend="unconfined_picard_kcycle", rates=(1.0e-4,), dt=7.0,
        )
        assert info_p1["solver_backend"] == "unconfined_picard_kcycle"
        heads_f, info_f = _run_transient(
            solver, case, backend="unconfined_fas", rates=(1.2e-4,), dt=7.0,
            initial_head=heads_p1[-1],
        )
        assert info_f["solver_backend"] == "unconfined_fas"
        heads_p2, info_p2 = _run_transient(
            solver, case, backend="unconfined_picard_kcycle", rates=(0.8e-4,), dt=7.0,
            initial_head=heads_f[-1],
        )
    finally:
        solver.close()
    # The FAS leg starts from the accepted Picard head (no re-initialisation).
    first_fas = _accepted_records(info_f)[0]
    np.testing.assert_array_equal(first_fas["previous_head"], heads_p1[-1])
    assert np.all(np.isfinite(heads_p2))
    assert not np.array_equal(heads_p2[-1], case["initial"])
    assert all(p["converged"] for p in info_f["period_infos"])


def test_17_fas_failure_picard_fallback_then_continued_simulation():
    case = _case_fields()
    solver = _make_solver()
    controls = {
        # Force the FAS leg to fail every attempt (unreachable tolerance in a
        # single cycle); the per-timestep fallback chain must complete the run.
        "fas_max_cycles": 1,
        "fas_residual_rms_tolerance": 1.0e-12,
        "fas_head_equivalent_rms_tolerance": 1.0e-12,
        "fas_fallback_enabled": True,
    }
    try:
        heads, info = _run_transient(solver, case, rates=(1.0e-4, 1.2e-4), dt=7.0, controls=controls)
    finally:
        solver.close()
    accepted = _accepted_records(info)
    assert len(accepted) == 2
    assert np.all(np.isfinite(heads))
    assert all(p["converged"] for p in info["period_infos"])
    for rec in accepted:
        # FAS is attempted on every timestep (fallback is per-timestep, not
        # a silent substitution for the rest of the simulation).
        assert rec["backend_attempted"] == "unconfined_fas"
        assert rec["fallback_used"]
        assert rec["backend_used"] in (
            "unconfined_semismooth_newton_kcycle",
            "unconfined_picard_kcycle",
        )
        assert rec["fallback_backend"] == rec["backend_used"]
    assert info["transient_replay_counters"]["experimental_fallback_timestep_count"] == 2


# ---------------------------------------------------------------------------
# 18. FAS vs Picard complete transient comparison
# ---------------------------------------------------------------------------


def test_18_fas_vs_picard_complete_transient_histories():
    case = _case_fields()
    rates = (1.0e-4, 1.5e-4, 0.5e-4)
    fas_controls = {"fas_fallback_enabled": True}
    solver_fas = _make_solver()
    try:
        heads_fas, info_fas = _run_transient(
            solver_fas, case, rates=rates, dt=7.0, controls=fas_controls,
        )
    finally:
        solver_fas.close()
    picard_controls = {"hclose": 1.0e-6, "max_outer_iterations": 60}
    solver_pic = _make_solver()
    try:
        heads_pic, info_pic = _run_transient(
            solver_pic, case, backend="unconfined_picard_kcycle", rates=rates, dt=7.0,
            controls=picard_controls,
        )
    finally:
        solver_pic.close()

    active = case["active"] != 0
    # Final and per-period heads agree within production head-accuracy gates.
    diff = np.abs(heads_fas - heads_pic)
    final_rmse = float(np.sqrt(np.mean((heads_fas[-1] - heads_pic[-1])[active] ** 2)))
    assert final_rmse < 1.0e-3
    assert float(np.max(diff[-1][active])) < 5.0e-3
    for period in range(3):
        assert float(np.max(diff[period][active])) < 2.0e-2

    # Exact storage histories: same formulation, period granularity.
    store_fas = info_fas["storage_terms_per_period"]
    store_pic = info_pic["storage_terms_per_period"]
    np.testing.assert_allclose(store_fas, store_pic, rtol=2.0e-2, atol=1.0e-9)

    # Budgets: FAS merged period budgets vs an independent budget evaluation
    # on the Picard heads (same packages plus exact storage).
    from DARCY_WARP_PACKAGE.physics.budgets_2d import (
        add_exact_storage_to_budget,
        compute_mass_balance_budget,
    )

    thickness = np.clip(heads_pic[-1] - case["bottom"], 0.1, case["top"] - case["bottom"])
    t_field = case["k"] * thickness
    t_field[case["active"] == 0] = 0.0
    budget_pic = compute_mass_balance_budget(
        T_field=t_field,
        R_field=np.full((CASE_NY, CASE_NX), rates[-1]),
        head=heads_pic[-1],
        active=case["active"],
        bc_mask=case["bc_mask"],
        bc_values=case["bc_values"],
        dx=CASE_DX,
    )
    # exact_unconfined_storage_terms is per unit plan area; the budget is
    # area-integrated, so scale by the cell area before folding it in.
    budget_pic = add_exact_storage_to_budget(budget_pic, store_pic[-1] * CASE_DX * CASE_DX)
    budget_fas = info_fas["experimental_period_budgets"][-1]
    pic_row = budget_pic.iloc[0]
    for key in ("total_in", "total_out"):
        assert budget_fas[key] == pytest.approx(float(pic_row[key]), rel=5.0e-2)

    # Authoritative nonlinear residual of both final heads on the last period.
    from DARCY_WARP_PACKAGE.nonlinear import from_arrays, NonlinearOperator2D

    def _residual_rms(head_new, head_old):
        context = from_arrays(
            nx=CASE_NX,
            ny=CASE_NY,
            dx=CASE_DX,
            K=case["k"],
            zbot=case["bottom"],
            ztop=case["top"],
            active=case["active"],
            dirichlet_mask=case["bc_mask"],
            dirichlet_values=case["bc_values"],
            R_field=np.full((CASE_NY, CASE_NX), rates[-1]),
            ghb_mask=np.zeros((CASE_NY, CASE_NX), dtype=np.int32),
            ghb_external_head=np.zeros((CASE_NY, CASE_NX)),
            ghb_factor=np.zeros((CASE_NY, CASE_NX)),
            sy=0.1,
            ss=1.0e-5,
            head_prev=head_old,
            dt=7.0,
            transient=True,
            min_sat=0.1,
            device="cpu",
        )
        operator = NonlinearOperator2D(context)
        try:
            residual = operator.residual(head_new)
            residual_host = np.asarray(residual.numpy() if hasattr(residual, "numpy") else residual)
            free = active & (case["bc_mask"] == 0)
            return float(np.sqrt(np.mean(residual_host[free] ** 2)))
        finally:
            operator.close()

    old_fas = info_fas["heads_old_per_period"][-1]
    old_pic = info_pic["heads_old_per_period"][-1]
    rms_fas = _residual_rms(heads_fas[-1], old_fas)
    rms_pic = _residual_rms(heads_pic[-1], old_pic)
    assert rms_fas < 1.0e-4
    assert rms_fas <= rms_pic * 2.0 + 1.0e-9


# ---------------------------------------------------------------------------
# 19-20. Repetition, restarts, GPU memory
# ---------------------------------------------------------------------------


def test_19_repeated_multiperiod_fas_no_stale_state_or_memory_growth():
    if not wp.is_cuda_available():
        pytest.skip("CUDA required for mempool accounting")
    case = _case_fields()
    solver = _make_solver(device="cuda:0")
    try:
        heads_a, info_a = _run_transient(solver, case, rates=(1.0e-4, 1.2e-4), dt=7.0)
        workspace = _workspace(solver)
        assert workspace is not None
        gc.collect()
        wp.synchronize_device("cuda:0")
        mem_after_first = int(wp.get_mempool_used_mem_current("cuda:0"))
        heads_b, info_b = _run_transient(solver, case, rates=(1.0e-4, 1.2e-4), dt=7.0)
        gc.collect()
        wp.synchronize_device("cuda:0")
        mem_after_second = int(wp.get_mempool_used_mem_current("cuda:0"))
        # Workspace identity is preserved (refresh, not rebuild).
        assert _workspace(solver) is workspace
    finally:
        solver.close()
    np.testing.assert_array_equal(heads_a, heads_b)
    # First run: one build + one refresh (two solves); second run refreshes only.
    assert info_a["last_info"]["fas_workspace_refresh_count"] == 1
    assert info_b["last_info"]["fas_workspace_refresh_count"] == 3
    assert info_b["last_info"]["fas_workspace_reused"]
    assert mem_after_second <= mem_after_first


def test_20_restart_second_simulation_same_model_no_state_carryover():
    case = _case_fields()
    solver = _make_solver()
    try:
        _run_transient(solver, case, rates=(3.0e-4, 3.0e-4), dt=7.0)
        heads_b, info_b = _run_transient(solver, case, rates=(0.5e-4, 1.0e-4), dt=7.0)
        workspace = _workspace(solver)
        # The second simulation reuses (refreshes) the same workspace object.
        assert workspace is not None
        assert _workspace(solver) is workspace
    finally:
        solver.close()
    fresh = _make_solver()
    try:
        heads_fresh, info_fresh = _run_transient(fresh, case, rates=(0.5e-4, 1.0e-4), dt=7.0)
    finally:
        fresh.close()
    # Restarting on the same model object is bitwise-identical to a fresh run.
    np.testing.assert_array_equal(heads_b, heads_fresh)
    assert info_b["last_info"]["fas_workspace_reused"]
    assert info_b["simulation_time"] == pytest.approx(info_fresh["simulation_time"])


# ---------------------------------------------------------------------------
# Production default is unchanged
# ---------------------------------------------------------------------------


def test_picard_remains_the_default_backend():
    case = _case_fields()
    solver = _make_solver()
    try:
        heads, info = _run_transient(solver, case, rates=(1.0e-4,), dt=7.0, backend=None)
    finally:
        solver.close()
    assert info["solver_backend"] == "unconfined_picard_kcycle"
    assert np.all(np.isfinite(heads))
