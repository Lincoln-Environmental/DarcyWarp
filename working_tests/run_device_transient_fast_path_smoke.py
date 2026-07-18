from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DARCY_WARP_PACKAGE.warped_darcy import (  # noqa: E402
    WP_FLOAT,
    WarpDarcySolver,
    _select_unconfined_inner_max_cycles,
    build_transient_rhs_from_storage_kernel,
    compute_dual_residual_kernel,
    update_secant_sy_storage_kernel,
)
from working_tests.transient_replay_storage import compute_unconfined_storage_components  # noqa: E402


def run_rhs_kernel_case(
    *,
    recharge_rate: np.ndarray,
    storage_diag: np.ndarray,
    head_prev: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    dx: float,
) -> np.ndarray:
    """
    Run the transient RHS kernel on CPU.

    :param recharge_rate: Recharge rate field [L/T].
    :param storage_diag: Integrated storage diagonal [L^2/T].
    :param head_prev: Previous-period head field [L].
    :param active: Active mask.
    :param bc_mask: Dirichlet mask.
    :param bc_values: Dirichlet values.
    :param dx: Cell size [L].
    :return: Device kernel RHS field.
    """
    ny, nx = recharge_rate.shape
    device = "cpu"
    recharge_wp = wp.array(recharge_rate.astype(np.float32), dtype=WP_FLOAT, device=device)
    storage_diag_wp = wp.array(storage_diag.astype(np.float32), dtype=WP_FLOAT, device=device)
    head_prev_wp = wp.array(head_prev.astype(np.float32), dtype=WP_FLOAT, device=device)
    active_wp = wp.array(active.astype(np.int32), dtype=wp.int32, device=device)
    bc_mask_wp = wp.array(bc_mask.astype(np.int32), dtype=wp.int32, device=device)
    bc_values_wp = wp.array(bc_values.astype(np.float32), dtype=WP_FLOAT, device=device)
    rhs_wp = wp.zeros((ny, nx), dtype=WP_FLOAT, device=device)

    wp.launch(
        kernel=build_transient_rhs_from_storage_kernel,
        dim=(ny, nx),
        inputs=[
            recharge_wp,
            storage_diag_wp,
            head_prev_wp,
            active_wp,
            bc_mask_wp,
            bc_values_wp,
            float(dx),
            int(nx),
            int(ny),
            rhs_wp,
        ],
        device=device,
    )
    return np.asarray(rhs_wp.numpy(), dtype=np.float64)


def compute_dual_residual_norms(
    *,
    head: np.ndarray,
    rhs: np.ndarray,
    transmissivity: np.ndarray,
    storage_diag: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
) -> tuple[float, float]:
    """
    Compute device-side flow and head-equivalent residual RMS values.

    :param head: Candidate head field.
    :param rhs: Integrated transient RHS field.
    :param transmissivity: Current transmissivity field.
    :param storage_diag: Current storage diagonal field.
    :param active: Active mask.
    :param bc_mask: Dirichlet mask.
    :return: ``(flow_residual_rms, head_residual_rms)``.
    """
    ny, nx = head.shape
    device = "cpu"
    head_wp = wp.array(head.astype(np.float32), dtype=WP_FLOAT, device=device)
    rhs_wp = wp.array(rhs.astype(np.float32), dtype=WP_FLOAT, device=device)
    t_wp = wp.array(transmissivity.astype(np.float32), dtype=WP_FLOAT, device=device)
    s_wp = wp.array(storage_diag.astype(np.float32), dtype=WP_FLOAT, device=device)
    active_wp = wp.array(active.astype(np.int32), dtype=wp.int32, device=device)
    bc_mask_wp = wp.array(bc_mask.astype(np.int32), dtype=wp.int32, device=device)
    gh_mask_wp = wp.zeros((ny, nx), dtype=wp.int32, device=device)
    ghb_factor_wp = wp.zeros((ny, nx), dtype=WP_FLOAT, device=device)
    flow_rtr_wp = wp.zeros(1, dtype=wp.float64, device=device)
    head_rtr_wp = wp.zeros(1, dtype=wp.float64, device=device)

    wp.launch(
        kernel=compute_dual_residual_kernel,
        dim=(ny, nx),
        inputs=[
            head_wp,
            rhs_wp,
            t_wp,
            active_wp,
            bc_mask_wp,
            gh_mask_wp,
            ghb_factor_wp,
            s_wp,
            flow_rtr_wp,
            head_rtr_wp,
            int(nx),
            int(ny),
        ],
        device=device,
    )
    n_free = int(np.count_nonzero((active != 0) & (bc_mask == 0)))
    denom = float(max(n_free, 1))
    flow_rms = float(np.sqrt(max(float(flow_rtr_wp.numpy()[0]), 0.0) / denom))
    head_rms = float(np.sqrt(max(float(head_rtr_wp.numpy()[0]), 0.0) / denom))
    return flow_rms, head_rms


def build_small_solver(
    *,
    device: str = "cpu",
) -> tuple[WarpDarcySolver, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a tiny deterministic transient test problem.

    :param device: Warp device string.
    :return: Solver and problem arrays.
    """
    ny, nx = 6, 8
    dx = 20.0
    solver = WarpDarcySolver(
        nx=nx,
        ny=ny,
        dx=dx,
        device=device,
        solver_type="kcycle",
        use_ghb=False,
        diag_preconditioner_backend="device",
    )
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values = np.zeros((ny, nx), dtype=np.float64)
    bc_values[:, 0] = 11.0
    bc_values[:, -1] = 9.5
    initial_head = np.full((ny, nx), 10.0, dtype=np.float64)
    initial_head[:, 0] = 11.0
    initial_head[:, -1] = 9.5
    k_field = np.full((ny, nx), 1.5, dtype=np.float64)
    zbot = np.zeros((ny, nx), dtype=np.float64)
    ztop = np.full((ny, nx), 20.0, dtype=np.float64)
    solver.build_from_fields(
        T_field=np.full((ny, nx), 15.0, dtype=np.float64),
        R_field=np.zeros((ny, nx), dtype=np.float64),
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
    )
    return solver, initial_head, k_field, zbot, ztop, active, bc_mask, bc_values


def default_fast_controls() -> dict:
    """
    Build small deterministic controls for the fast-path smoke checks.
    """
    return {
        "nu_pre": 2,
        "nu_post": 2,
        "nu_coarse": 10,
        "omega": 0.8,
        "abs_tol_min": 1.0e-8,
        "rel_tol": 1.0e-8,
        "max_levels": 3,
        "min_coarse_cells": 4,
        "unconfined_max_picard_iter": 20,
        "hclose": 1.0e-6,
        "strict_head_residual_tol": 1.0e-5,
        "practical_head_residual_tol": 1.0e-5,
        "practical_dh_rms_tol": 1.0e-4,
        "practical_storage_diag_change_rms_tol": 1.0,
        "unconfined_inner_max_cycles_early": 3,
        "unconfined_inner_max_cycles_middle": 4,
        "unconfined_inner_max_cycles_late": 5,
        "unconfined_inner_middle_dh": 1.0,
        "unconfined_inner_late_dh": 1.0e-2,
        "use_device_transient_fast_path": True,
    }


def check_rhs_integration() -> None:
    recharge_rate = np.full((3, 4), 2.5e-3, dtype=np.float64)
    storage_diag = np.zeros((3, 4), dtype=np.float64)
    head_prev = np.zeros((3, 4), dtype=np.float64)
    active = np.ones((3, 4), dtype=np.int32)
    bc_mask = np.zeros((3, 4), dtype=np.int32)
    bc_values = np.zeros((3, 4), dtype=np.float64)
    dx = 10.0

    rhs = run_rhs_kernel_case(
        recharge_rate=recharge_rate,
        storage_diag=storage_diag,
        head_prev=head_prev,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        dx=dx,
    )
    np.testing.assert_allclose(rhs, recharge_rate * dx * dx, rtol=0.0, atol=1.0e-6)


def check_storage_rhs() -> None:
    recharge_rate = np.zeros((3, 4), dtype=np.float64)
    storage_diag = np.full((3, 4), 7.5, dtype=np.float64)
    head_prev = np.full((3, 4), 11.0, dtype=np.float64)
    active = np.ones((3, 4), dtype=np.int32)
    bc_mask = np.zeros((3, 4), dtype=np.int32)
    bc_values = np.zeros((3, 4), dtype=np.float64)
    dx = 10.0

    rhs = run_rhs_kernel_case(
        recharge_rate=recharge_rate,
        storage_diag=storage_diag,
        head_prev=head_prev,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        dx=dx,
    )
    np.testing.assert_allclose(rhs, storage_diag * head_prev, rtol=0.0, atol=1.0e-6)


def check_combined_rhs() -> None:
    recharge_rate = np.full((3, 4), 1.5e-4, dtype=np.float64)
    storage_diag = np.full((3, 4), 4.0, dtype=np.float64)
    head_prev = np.full((3, 4), 9.0, dtype=np.float64)
    active = np.ones((3, 4), dtype=np.int32)
    bc_mask = np.zeros((3, 4), dtype=np.int32)
    bc_mask[:, 0] = 1
    bc_values = np.zeros((3, 4), dtype=np.float64)
    bc_values[:, 0] = 13.0
    dx = 20.0

    rhs = run_rhs_kernel_case(
        recharge_rate=recharge_rate,
        storage_diag=storage_diag,
        head_prev=head_prev,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        dx=dx,
    )
    expected = recharge_rate * dx * dx + storage_diag * head_prev
    expected[:, 0] = bc_values[:, 0]
    np.testing.assert_allclose(rhs, expected, rtol=0.0, atol=1.0e-6)


def check_residual_scaling() -> None:
    storage_coeff = 0.2
    dt = 5.0
    head_prev = np.array([[10.0]], dtype=np.float64)
    head_trial = np.array([[10.25]], dtype=np.float64)
    active = np.ones((1, 1), dtype=np.int32)
    bc_mask = np.zeros((1, 1), dtype=np.int32)
    recharge = np.zeros((1, 1), dtype=np.float64)
    transmissivity = np.zeros((1, 1), dtype=np.float64)

    dx_small = 10.0
    storage_diag_small = np.array([[storage_coeff * dx_small * dx_small / dt]], dtype=np.float64)
    rhs_small = recharge + storage_diag_small * head_prev
    flow_small, head_small = compute_dual_residual_norms(
        head=head_trial,
        rhs=rhs_small,
        transmissivity=transmissivity,
        storage_diag=storage_diag_small,
        active=active,
        bc_mask=bc_mask,
    )

    dx_large = 40.0
    storage_diag_large = np.array([[storage_coeff * dx_large * dx_large / dt]], dtype=np.float64)
    rhs_large = recharge + storage_diag_large * head_prev
    flow_large, head_large = compute_dual_residual_norms(
        head=head_trial,
        rhs=rhs_large,
        transmissivity=transmissivity,
        storage_diag=storage_diag_large,
        active=active,
        bc_mask=bc_mask,
    )

    assert flow_large > flow_small * 10.0
    np.testing.assert_allclose(head_large, head_small, atol=1.0e-6, rtol=0.0)


def check_secant_sy_host_device_equivalence() -> None:
    ny, nx = 3, 3
    bottom = np.zeros((ny, nx), dtype=np.float64)
    top = np.full((ny, nx), 10.0, dtype=np.float64)
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    active[2, 1] = 0
    bc_mask[2, 2] = 1

    head_prev = np.array(
        [[5.0, 12.0, 4.0], [12.0, 6.0, -1.0], [11.0, 5.0, 5.0]],
        dtype=np.float64,
    )
    head_ref = np.array(
        [[7.0, 11.0, 12.0], [8.0, 6.0 + 1.0e-13, 11.0], [12.0, -2.0, 5.0]],
        dtype=np.float64,
    )
    sy = 0.2
    ss = 1.0e-5
    dx = 25.0
    dt = 4.0
    min_sat = 0.1
    secant_eps = 1.0e-12

    components = compute_unconfined_storage_components(
        sy=sy,
        ss=ss,
        head_ref=head_ref,
        head_old=head_prev,
        bottom=bottom,
        top=top,
        active=active,
        bc_mask=bc_mask,
        min_sat=min_sat,
        storage_mode="mf6_convertible_secant_sy",
        secant_eps=secant_eps,
    )

    device = "cpu"
    head_ref_wp = wp.array(head_ref.astype(np.float32), dtype=WP_FLOAT, device=device)
    head_prev_wp = wp.array(head_prev.astype(np.float32), dtype=WP_FLOAT, device=device)
    bottom_wp = wp.array(bottom.astype(np.float32), dtype=WP_FLOAT, device=device)
    top_wp = wp.array(top.astype(np.float32), dtype=WP_FLOAT, device=device)
    active_wp = wp.array(active.astype(np.int32), dtype=wp.int32, device=device)
    bc_mask_wp = wp.array(bc_mask.astype(np.int32), dtype=wp.int32, device=device)
    storage_coeff_wp = wp.zeros((ny, nx), dtype=WP_FLOAT, device=device)
    sy_coeff_wp = wp.zeros((ny, nx), dtype=WP_FLOAT, device=device)
    ss_coeff_wp = wp.zeros((ny, nx), dtype=WP_FLOAT, device=device)
    storage_diag_wp = wp.zeros((ny, nx), dtype=WP_FLOAT, device=device)
    storage_diag_prev_wp = wp.zeros((ny, nx), dtype=WP_FLOAT, device=device)
    storage_change_sum_sq = wp.zeros(1, dtype=wp.float64, device=device)
    storage_change_max = wp.zeros(1, dtype=wp.float64, device=device)

    wp.launch(
        kernel=update_secant_sy_storage_kernel,
        dim=(ny, nx),
        inputs=[
            head_ref_wp,
            head_prev_wp,
            bottom_wp,
            top_wp,
            active_wp,
            bc_mask_wp,
            float(sy),
            float(ss),
            float(dx),
            float(dt),
            float(min_sat),
            float(secant_eps),
            int(nx),
            int(ny),
            storage_coeff_wp,
            sy_coeff_wp,
            ss_coeff_wp,
            storage_diag_wp,
            storage_diag_prev_wp,
            storage_change_sum_sq,
            storage_change_max,
        ],
        device=device,
    )

    host_storage = np.asarray(components["storage_coeff"], dtype=np.float64)
    host_sy = np.asarray(components["sy_coeff"], dtype=np.float64)
    host_ss = np.asarray(components["ss_coeff"], dtype=np.float64)
    host_diag = host_storage * dx * dx / dt

    np.testing.assert_allclose(np.asarray(storage_coeff_wp.numpy(), dtype=np.float64), host_storage, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(sy_coeff_wp.numpy(), dtype=np.float64), host_sy, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(ss_coeff_wp.numpy(), dtype=np.float64), host_ss, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(storage_diag_wp.numpy(), dtype=np.float64), host_diag, atol=1.0e-5)


def check_adaptive_inner_schedule() -> None:
    assert _select_unconfined_inner_max_cycles(
        previous_dh_measure=None,
        early_cycles=10,
        middle_cycles=25,
        late_cycles=60,
        middle_dh=1.0,
        late_dh=1.0e-2,
    ) == 10
    assert _select_unconfined_inner_max_cycles(
        previous_dh_measure=2.0,
        early_cycles=10,
        middle_cycles=25,
        late_cycles=60,
        middle_dh=1.0,
        late_dh=1.0e-2,
    ) == 10
    assert _select_unconfined_inner_max_cycles(
        previous_dh_measure=0.1,
        early_cycles=10,
        middle_cycles=25,
        late_cycles=60,
        middle_dh=1.0,
        late_dh=1.0e-2,
    ) == 25
    assert _select_unconfined_inner_max_cycles(
        previous_dh_measure=1.0e-4,
        early_cycles=10,
        middle_cycles=25,
        late_cycles=60,
        middle_dh=1.0,
        late_dh=1.0e-2,
    ) == 60


def check_one_period_host_device_equivalence() -> None:
    solver_dev, initial_head, k_field, zbot, ztop, active, bc_mask, bc_values = build_small_solver(device="cpu")
    controls = default_fast_controls()

    with solver_dev:
        heads_dev, info_dev = solver_dev.solve_transient_2d_unconfined(
            initial_head=initial_head,
            recharge_rates=np.asarray([2.0e-4], dtype=np.float64),
            k_field=k_field,
            zbot_field=zbot,
            ztop_field=ztop,
            sy=0.15,
            ss=1.0e-5,
            dt=5.0,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            storage_mode="mf6_convertible_secant_sy",
            storage_reference="current_picard",
            solve_controls=controls,
            return_info=True,
        )

    solver_host, _, _, _, _, _, _, _ = build_small_solver(device="cpu")
    with solver_host:
        heads_host, info_host = solver_host.solve_transient_2d_unconfined(
            initial_head=initial_head,
            recharge_rates=np.asarray([2.0e-4], dtype=np.float64),
            k_field=k_field,
            zbot_field=zbot,
            ztop_field=ztop,
            sy=0.15,
            ss=1.0e-5,
            dt=5.0,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            storage_mode="mf6_convertible_secant_sy",
            storage_reference="current_picard",
            solve_controls=controls,
            return_info=True,
        )

    np.testing.assert_allclose(heads_dev[0], heads_host[0], atol=1.0e-5, rtol=0.0)
    period_dev = info_dev["period_infos"][0]
    assert period_dev["adaptive_inner_controller_enabled"] is True
    assert period_dev["inner_scalar_synchronization_count"] >= period_dev["total_inner_blocks"]
    assert period_dev["inner_solver_gpu_scalar_synchronization_count"] >= 0
    assert period_dev["maximum_inner_kcycles_in_one_outer_iteration"] <= 60
    assert period_dev["total_inner_kcycles"] < period_dev["outer_iterations"] * 200
    assert "strict_picard_convergence_passed" in period_dev
    assert "practical_picard_acceptance_passed" in period_dev
    assert "production_acceptance_passed" in period_dev
    assert "final_head_residual_rms" in period_dev
    assert "final_flow_residual_rms" in period_dev


def check_failed_period_raises_immediately() -> None:
    solver_dev, initial_head, k_field, zbot, ztop, active, bc_mask, bc_values = build_small_solver(device="cpu")
    controls = {
        "max_levels": 3,
        "min_coarse_cells": 4,
        "unconfined_max_picard_iter": 2,
        "hclose": 1.0e-12,
        "strict_head_residual_tol": 1.0e-12,
        "min_practical_outer_iterations": 10,
        "practical_head_residual_tol": 1.0e-12,
        "practical_dh_rms_tol": 1.0e-12,
        "practical_storage_diag_change_rms_tol": 1.0e-12,
        "allow_unaccepted_transient_period": False,
        "unconfined_inner_max_cycles_early": 1,
        "unconfined_inner_max_cycles_middle": 1,
        "unconfined_inner_max_cycles_late": 1,
        "use_device_transient_fast_path": True,
    }
    with solver_dev:
        try:
            solver_dev.solve_transient_2d_unconfined(
                initial_head=initial_head,
                recharge_rates=np.asarray([2.0e-4, 2.0e-4], dtype=np.float64),
                k_field=k_field,
                zbot_field=zbot,
                ztop_field=ztop,
                sy=0.15,
                ss=1.0e-5,
                dt=5.0,
                active=active,
                bc_mask=bc_mask,
                bc_values=bc_values,
                storage_mode="mf6_convertible_secant_sy",
                storage_reference="current_picard",
                solve_controls=controls,
                return_info=True,
            )
        except RuntimeError as exc:
            message = str(exc)
            assert "period_index=0" in message
            assert "final_head_residual_rms" in message
            return
    raise AssertionError("Expected RuntimeError for unaccepted first period")


def check_final_residual_is_refreshed() -> None:
    solver_dev, initial_head, k_field, zbot, ztop, active, bc_mask, bc_values = build_small_solver(device="cpu")
    controls = default_fast_controls()
    sy = 0.15
    ss = 1.0e-5
    dt = 5.0
    recharge_rate = 2.0e-4

    with solver_dev:
        heads_dev, info_dev = solver_dev.solve_transient_2d_unconfined(
            initial_head=initial_head,
            recharge_rates=np.asarray([recharge_rate], dtype=np.float64),
            k_field=k_field,
            zbot_field=zbot,
            ztop_field=ztop,
            sy=sy,
            ss=ss,
            dt=dt,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            storage_mode="mf6_convertible_secant_sy",
            storage_reference="current_picard",
            solve_controls=controls,
            save_diagnostics=True,
            return_info=True,
        )

    period_info = info_dev["period_infos"][0]
    final_head = np.asarray(heads_dev[0], dtype=np.float64)
    components = compute_unconfined_storage_components(
        sy=sy,
        ss=ss,
        head_ref=final_head,
        head_old=initial_head,
        bottom=zbot,
        top=ztop,
        active=active,
        bc_mask=bc_mask,
        min_sat=0.1,
        storage_mode="mf6_convertible_secant_sy",
        secant_eps=1.0e-12,
    )
    full_thickness = np.maximum(ztop - zbot, 0.1)
    sat = np.clip(final_head - zbot, 0.1, full_thickness)
    transmissivity = k_field * sat
    transmissivity[active == 0] = 0.0
    storage_diag = np.asarray(components["storage_coeff"], dtype=np.float64) * float(solver_dev.dx) * float(solver_dev.dx) / dt
    rhs = recharge_rate * float(solver_dev.dx) * float(solver_dev.dx) + storage_diag * initial_head
    rhs[active == 0] = 0.0
    rhs[bc_mask != 0] = bc_values[bc_mask != 0]
    flow_rms, head_rms = compute_dual_residual_norms(
        head=final_head,
        rhs=rhs,
        transmissivity=transmissivity,
        storage_diag=storage_diag,
        active=active,
        bc_mask=bc_mask,
    )
    np.testing.assert_allclose(period_info["final_head_residual_rms"], head_rms, atol=1.0e-5, rtol=0.0)
    np.testing.assert_allclose(period_info["final_flow_residual_rms"], flow_rms, atol=1.0e-4, rtol=0.0)


def check_multi_period_runtime_smoke() -> None:
    solver_dev, initial_head, k_field, zbot, ztop, active, bc_mask, bc_values = build_small_solver(device="cpu")
    controls = default_fast_controls()

    with solver_dev:
        heads_dev, info_dev = solver_dev.solve_transient_2d_unconfined(
            initial_head=initial_head,
            recharge_rates=np.asarray([2.0e-4, 1.0e-4, 2.5e-4], dtype=np.float64),
            k_field=k_field,
            zbot_field=zbot,
            ztop_field=ztop,
            sy=0.15,
            ss=1.0e-5,
            dt=5.0,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            storage_mode="mf6_convertible_secant_sy",
            storage_reference="current_picard",
            solve_controls=controls,
            return_info=True,
        )

    solver_host, _, _, _, _, _, _, _ = build_small_solver(device="cpu")
    with solver_host:
        heads_host, info_host = solver_host.solve_transient_2d_unconfined(
            initial_head=initial_head,
            recharge_rates=np.asarray([2.0e-4, 1.0e-4, 2.5e-4], dtype=np.float64),
            k_field=k_field,
            zbot_field=zbot,
            ztop_field=ztop,
            sy=0.15,
            ss=1.0e-5,
            dt=5.0,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            storage_mode="mf6_convertible_secant_sy",
            storage_reference="current_picard",
            solve_controls=controls,
            return_info=True,
        )

    np.testing.assert_allclose(heads_dev, heads_host, atol=1.0e-5, rtol=0.0)
    total_inner = sum(int(p["total_inner_kcycles"]) for p in info_dev["period_infos"])
    outer_total = sum(int(p["outer_iterations"]) for p in info_dev["period_infos"])
    assert total_inner < outer_total * 60
    assert all(int(p["maximum_inner_kcycles_in_one_outer_iteration"]) <= 60 for p in info_dev["period_infos"])
    assert all(bool(p["production_acceptance_passed"]) for p in info_dev["period_infos"])


def run_smoke_checks() -> None:
    """
    Execute the lightweight scientific and performance smoke checks for the fast path.
    """
    wp.init()
    check_rhs_integration()
    check_storage_rhs()
    check_combined_rhs()
    check_residual_scaling()
    check_secant_sy_host_device_equivalence()
    check_adaptive_inner_schedule()
    check_one_period_host_device_equivalence()
    check_failed_period_raises_immediately()
    check_final_residual_is_refreshed()
    check_multi_period_runtime_smoke()
    print("Fast-path transient smoke checks: PASS")


if __name__ == "__main__":
    run_smoke_checks()
