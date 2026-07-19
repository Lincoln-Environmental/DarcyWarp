# SPDX-License-Identifier: AGPL-3.0-only
"""Experimental two-dimensional nonlinear Full Approximation Scheme backend."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT
from .base import SolverContext
from .fas_hierarchy import build_fas_physical_hierarchy, make_fine_physical_level
from .fas_state import FASLevelState, FASWorkspace
from . import fas_kernels as _k


def _launch_copy(level: FASLevelState, source: Any, target: Any, workspace: FASWorkspace) -> None:
    wp.launch(
        kernel=_k.fas_copy_kernel,
        dim=level.shape,
        inputs=[source, target, level.physical.nx, level.physical.ny],
        device=workspace.device,
    )
    workspace.kernel_launches += 1


def _norm(
    *,
    level: FASLevelState,
    value: Any,
    workspace: FASWorkspace,
    head_equivalent: bool = False,
) -> tuple[float, float]:
    level.sum_sq.fill_(wp.float64(0.0))
    level.max_abs.fill_(wp.float64(0.0))
    operator = level.nonlinear_operator.operator
    wp.launch(
        kernel=_k.fas_norm_kernel,
        dim=level.shape,
        inputs=[
            value,
            operator.active_device,
            operator.dirichlet_mask_device,
            level.diagonal,
            int(1 if head_equivalent else 0),
            level.sum_sq,
            level.max_abs,
            level.physical.nx,
            level.physical.ny,
        ],
        device=workspace.device,
    )
    workspace.kernel_launches += 1
    total = float(level.sum_sq.numpy()[0])
    maximum = float(level.max_abs.numpy()[0])
    workspace.synchronizations += 2
    rms = float(np.sqrt(max(total, 0.0) / float(level.n_free))) if level.n_free else 0.0
    return rms, maximum


def _smooth(
    *,
    level: FASLevelState,
    level_index: int,
    workspace: FASWorkspace,
    sweeps: int,
    damping: float,
    correction_limit: float,
    phase: str,
) -> None:
    operator = level.nonlinear_operator
    before_rms = None
    for _ in range(int(sweeps)):
        operator.evaluate(head=level.head, state=level)
        operator.refresh_frozen_diagonal(head=level.head, state=level)
        workspace.kernel_launches += 4
        if before_rms is None:
            before_rms, _ = _norm(level=level, value=level.defect, workspace=workspace)
        wp.launch(
            kernel=_k.fas_jacobi_update_kernel,
            dim=level.shape,
            inputs=[
                level.head,
                level.defect,
                level.diagonal,
                operator.operator.active_device,
                operator.operator.dirichlet_mask_device,
                operator.operator.dirichlet_values_device,
                float(damping),
                float(correction_limit),
                level.physical.nx,
                level.physical.ny,
            ],
            device=workspace.device,
        )
        workspace.kernel_launches += 1
        if phase == "pre":
            workspace.pre_sweeps[level_index] += 1
        elif phase == "post":
            workspace.post_sweeps[level_index] += 1
        else:
            workspace.coarse_sweeps[level_index] += 1
    operator.evaluate(head=level.head, state=level)
    operator.refresh_frozen_diagonal(head=level.head, state=level)
    workspace.kernel_launches += 4
    if before_rms is None:
        before_rms, _ = _norm(level=level, value=level.defect, workspace=workspace)
    after_rms, _ = _norm(level=level, value=level.defect, workspace=workspace)
    workspace.smoothing_history.append(
        {
            "level": int(level_index),
            "phase": str(phase),
            "sweeps": int(sweeps),
            "residual_rms_before": float(before_rms),
            "residual_rms_after": float(after_rms),
            "smoothing_factor": (
                float(after_rms / before_rms) if before_rms > 0.0 else 0.0
            ),
        }
    )


def _restrict_head(*, fine: FASLevelState, coarse: FASLevelState, workspace: FASWorkspace) -> None:
    fine_op = fine.nonlinear_operator.operator
    coarse_op = coarse.nonlinear_operator.operator
    wp.launch(
        kernel=_k.fas_restrict_head_kernel,
        dim=coarse.shape,
        inputs=[
            fine.head,
            fine_op.active_device,
            coarse_op.active_device,
            coarse_op.dirichlet_mask_device,
            coarse_op.dirichlet_values_device,
            coarse.head_initial,
            fine.physical.nx,
            fine.physical.ny,
            coarse.physical.nx,
            coarse.physical.ny,
        ],
        device=workspace.device,
    )
    _launch_copy(coarse, coarse.head_initial, coarse.head, workspace)
    workspace.transfer_launches += 1
    workspace.kernel_launches += 1


def _restrict_integrated(
    *,
    fine: FASLevelState,
    coarse: FASLevelState,
    fine_value: Any,
    coarse_value: Any,
    workspace: FASWorkspace,
) -> None:
    fine_op = fine.nonlinear_operator.operator
    coarse_op = coarse.nonlinear_operator.operator
    wp.launch(
        kernel=_k.fas_restrict_integrated_kernel,
        dim=coarse.shape,
        inputs=[
            fine_value,
            fine_op.active_device,
            fine_op.dirichlet_mask_device,
            coarse_op.active_device,
            coarse_op.dirichlet_mask_device,
            coarse_value,
            fine.physical.nx,
            fine.physical.ny,
            coarse.physical.nx,
            coarse.physical.ny,
        ],
        device=workspace.device,
    )
    workspace.transfer_launches += 1
    workspace.kernel_launches += 1


def _apply_safeguarded_correction(
    *,
    level: FASLevelState,
    workspace: FASWorkspace,
    correction_limit: float,
    max_backtracks: int,
    minimum_alpha: float,
    allowed_growth: float,
) -> dict[str, Any]:
    operator = level.nonlinear_operator
    operator.evaluate(head=level.head, state=level)
    operator.refresh_frozen_diagonal(head=level.head, state=level)
    workspace.kernel_launches += 4
    before_rms, _ = _norm(level=level, value=level.defect, workspace=workspace)
    alpha = 1.0
    backtracks = 0
    accepted = False
    candidate_rms = float("inf")
    change_rms = float("nan")
    change_max = float("nan")
    while backtracks <= int(max_backtracks) and alpha >= float(minimum_alpha):
        level.change_sq.fill_(wp.float64(0.0))
        level.change_max.fill_(wp.float64(0.0))
        level.finite_flag.fill_(wp.int32(0))
        wp.launch(
            kernel=_k.fas_candidate_kernel,
            dim=level.shape,
            inputs=[
                level.head,
                level.prolonged_correction,
                operator.operator.active_device,
                operator.operator.dirichlet_mask_device,
                operator.operator.dirichlet_values_device,
                alpha,
                level.candidate,
                level.change_sq,
                level.change_max,
                level.finite_flag,
                level.physical.nx,
                level.physical.ny,
            ],
            device=workspace.device,
        )
        workspace.kernel_launches += 1
        nonfinite = int(level.finite_flag.numpy()[0]) != 0
        change_sq = float(level.change_sq.numpy()[0])
        change_max = float(level.change_max.numpy()[0])
        workspace.synchronizations += 3
        change_rms = float(np.sqrt(max(change_sq, 0.0) / float(level.n_free))) if level.n_free else 0.0
        if not nonfinite and change_max <= float(correction_limit):
            operator.evaluate(
                head=level.candidate,
                state=level,
                physical_residual=level.candidate_residual,
                defect=level.candidate_defect,
            )
            workspace.kernel_launches += 2
            candidate_rms, _ = _norm(level=level, value=level.candidate_defect, workspace=workspace)
            if np.isfinite(candidate_rms) and candidate_rms <= float(allowed_growth) * before_rms:
                accepted = True
                break
        alpha *= 0.5
        backtracks += 1
    if accepted:
        _launch_copy(level, level.candidate, level.head, workspace)
        _launch_copy(level, level.candidate_residual, level.physical_residual, workspace)
        _launch_copy(level, level.candidate_defect, level.defect, workspace)
    return {
        "accepted": bool(accepted),
        "alpha": float(alpha) if accepted else 0.0,
        "backtracks": int(backtracks),
        "before_rms": float(before_rms),
        "candidate_rms": float(candidate_rms),
        "correction_rms": float(change_rms),
        "correction_max": float(change_max),
    }


def _fas_vcycle(
    *,
    level_index: int,
    workspace: FASWorkspace,
    controls: dict[str, Any],
    cycle_diagnostics: dict[str, Any],
) -> None:
    level = workspace.levels[level_index]
    _smooth(
        level=level,
        level_index=level_index,
        workspace=workspace,
        sweeps=controls["pre_sweeps"],
        damping=controls["damping"],
        correction_limit=controls["smoothing_limit"],
        phase="pre",
    )

    if level_index == len(workspace.levels) - 1:
        _smooth(
            level=level,
            level_index=level_index,
            workspace=workspace,
            sweeps=controls["coarse_sweeps"],
            damping=controls["coarse_damping"],
            correction_limit=controls["smoothing_limit"],
            phase="coarse",
        )
        return

    coarse = workspace.levels[level_index + 1]
    _restrict_head(fine=level, coarse=coarse, workspace=workspace)
    _restrict_integrated(
        fine=level,
        coarse=coarse,
        fine_value=level.defect,
        coarse_value=coarse.restricted_defect,
        workspace=workspace,
    )
    _restrict_integrated(
        fine=level,
        coarse=coarse,
        fine_value=level.forcing,
        coarse_value=coarse.restricted_forcing,
        workspace=workspace,
    )
    coarse.nonlinear_operator.evaluate(head=coarse.head_initial, state=coarse)
    wp.launch(
        kernel=_k.fas_build_coarse_forcing_kernel,
        dim=coarse.shape,
        inputs=[
            coarse.physical_residual,
            coarse.physical_forcing,
            coarse.restricted_defect,
            coarse.restricted_forcing,
            coarse.nonlinear_operator.operator.active_device,
            coarse.nonlinear_operator.operator.dirichlet_mask_device,
            coarse.forcing,
            coarse.tau,
            coarse.physical.nx,
            coarse.physical.ny,
        ],
        device=workspace.device,
    )
    workspace.kernel_launches += 3
    tau_rms, tau_max = _norm(level=coarse, value=coarse.tau, workspace=workspace)
    defect_rms, defect_max = _norm(level=coarse, value=coarse.restricted_defect, workspace=workspace)
    cycle_diagnostics["tau_norms"].append(
        {"level": level_index + 1, "rms": tau_rms, "max": tau_max}
    )
    cycle_diagnostics["restricted_defect_norms"].append(
        {"level": level_index + 1, "rms": defect_rms, "max": defect_max}
    )

    _fas_vcycle(
        level_index=level_index + 1,
        workspace=workspace,
        controls=controls,
        cycle_diagnostics=cycle_diagnostics,
    )
    wp.launch(
        kernel=_k.fas_difference_kernel,
        dim=coarse.shape,
        inputs=[
            coarse.head,
            coarse.head_initial,
            coarse.nonlinear_operator.operator.active_device,
            coarse.nonlinear_operator.operator.dirichlet_mask_device,
            coarse.correction,
            coarse.physical.nx,
            coarse.physical.ny,
        ],
        device=workspace.device,
    )
    wp.launch(
        kernel=_k.fas_prolong_correction_kernel,
        dim=level.shape,
        inputs=[
            coarse.correction,
            level.nonlinear_operator.operator.active_device,
            level.nonlinear_operator.operator.dirichlet_mask_device,
            level.prolonged_correction,
            level.physical.nx,
            level.physical.ny,
            coarse.physical.nx,
            coarse.physical.ny,
        ],
        device=workspace.device,
    )
    workspace.kernel_launches += 2
    workspace.transfer_launches += 1
    correction = _apply_safeguarded_correction(
        level=level,
        workspace=workspace,
        correction_limit=controls["correction_limit"],
        max_backtracks=controls["correction_backtracks"],
        minimum_alpha=controls["minimum_correction_alpha"],
        allowed_growth=controls["allowed_correction_growth"],
    )
    correction["fine_level"] = level_index
    cycle_diagnostics["coarse_corrections"].append(correction)
    if not correction["accepted"]:
        cycle_diagnostics["rejected_corrections"] += 1
    elif correction["alpha"] < 1.0:
        cycle_diagnostics["damped_corrections"] += 1

    _smooth(
        level=level,
        level_index=level_index,
        workspace=workspace,
        sweeps=controls["post_sweeps"],
        damping=controls["damping"],
        correction_limit=controls["smoothing_limit"],
        phase="post",
    )


def _run_fallback(
    *,
    context: SolverContext,
    order: tuple[str, ...],
    original_kwargs: dict[str, Any],
    initial_head: np.ndarray,
    fas_info: dict[str, Any],
    return_info: bool,
):
    from .registry import select_backend

    attempts = []
    last_result = None
    for backend_name in order:
        if backend_name not in {"unconfined_semismooth_newton_kcycle", "unconfined_picard_kcycle"}:
            raise ValueError("FAS fallback order may contain only Newton and Picard backends.")
        fallback_kwargs = {
            key: value for key, value in original_kwargs.items() if not key.startswith("fas_")
        }
        fallback_kwargs["initial_head"] = initial_head.copy()
        fallback_kwargs["return_info"] = True
        if backend_name == "unconfined_semismooth_newton_kcycle":
            fallback_kwargs["newton_fallback_to_picard"] = False
        backend = select_backend(
            solver=backend_name,
            formulation="unconfined",
            transient=context.transient,
            default="unconfined_picard_kcycle",
        )
        try:
            head, info = backend.solve(context, **fallback_kwargs)
            attempts.append({"backend": backend_name, "converged": bool(info.get("converged", False))})
            last_result = (head, info, backend_name)
            if bool(info.get("converged", False)):
                break
        except Exception as exc:
            attempts.append({"backend": backend_name, "converged": False, "error": str(exc)})
    if last_result is None:
        raise RuntimeError(f"all FAS fallback backends failed: {attempts}")
    head, info, backend_name = last_result
    merged = dict(info)
    merged.update(
        {
            "fas_fallback_used": True,
            "fallback_backend": backend_name,
            "fallback_attempts": attempts,
            "fas_diagnostics_before_fallback": fas_info,
            "experimental_backend": True,
        }
    )
    return (head, merged) if return_info else head


def solve_unconfined_fas(*, context: SolverContext, **kwargs: Any):
    """Solve the full rediscretized nonlinear equation with FAS V-cycles."""
    model = context.model
    original_kwargs = dict(kwargs)
    return_info = bool(kwargs.get("return_info", True))
    K_field = kwargs.get("K_field")
    bottom = kwargs.get("zbot_field")
    top = kwargs.get("ztop_field")
    if K_field is None or bottom is None:
        raise ValueError("unconfined_fas requires K_field and zbot_field.")
    transient = bool(kwargs.get("transient", False))
    dt = kwargs.get("dt")
    previous_head = kwargs.get("head_prev")
    storage_coeff = kwargs.get("storage_coeff")
    sy = kwargs.get("sy")
    ss = kwargs.get("ss", 0.0)
    if transient and top is None:
        raise ValueError("transient FAS requires ztop_field for exact storage.")
    if transient and sy is None:
        storage_array = np.asarray(storage_coeff) if storage_coeff is not None else np.asarray([])
        if storage_array.ndim == 0 and storage_array.size == 1:
            sy = float(storage_array.reshape(()))
        else:
            raise ValueError("transient FAS requires scalar sy (or scalar storage_coeff).")
    sy = 0.0 if sy is None else float(sy)
    ss = 0.0 if ss is None else float(ss)
    min_sat = kwargs.get("unconfined_min_sat")
    if min_sat is None:
        min_sat = kwargs.get("min_saturated_thickness", 0.1)
    min_sat = 0.1 if min_sat is None else float(min_sat)

    max_levels = int(kwargs.get("fas_max_levels", kwargs.get("max_levels", 5)))
    min_coarse_cells = int(kwargs.get("fas_min_coarse_cells", 4))
    controls = {
        "pre_sweeps": int(kwargs.get("fas_pre_smoothing_sweeps", 3)),
        "post_sweeps": int(kwargs.get("fas_post_smoothing_sweeps", 3)),
        "coarse_sweeps": int(kwargs.get("fas_coarse_smoothing_sweeps", 40)),
        "damping": float(kwargs.get("fas_damping", 0.65)),
        "coarse_damping": float(kwargs.get("fas_coarse_damping", 0.7)),
        "smoothing_limit": float(kwargs.get("fas_smoothing_head_change_limit", 5.0)),
        "correction_limit": float(kwargs.get("fas_correction_head_change_limit", 10.0)),
        "correction_backtracks": int(kwargs.get("fas_correction_max_backtracks", 8)),
        "minimum_correction_alpha": float(kwargs.get("fas_minimum_correction_alpha", 2.0 ** -8)),
        "allowed_correction_growth": float(kwargs.get("fas_allowed_correction_residual_growth", 1.10)),
    }
    max_cycles = int(kwargs.get("fas_max_cycles", 30))
    residual_tolerance = float(kwargs.get("fas_residual_rms_tolerance", 1.0e-6))
    head_residual_tolerance = float(kwargs.get("fas_head_equivalent_rms_tolerance", kwargs.get("hclose", 1.0e-4) or 1.0e-4))
    fallback_enabled = bool(kwargs.get("fas_fallback_enabled", True))
    fallback_order = tuple(kwargs.get(
        "fas_fallback_order",
        ("unconfined_semismooth_newton_kcycle", "unconfined_picard_kcycle"),
    ))
    if min(controls["pre_sweeps"], controls["post_sweeps"], controls["coarse_sweeps"]) < 0:
        raise ValueError("FAS smoothing sweep counts must be non-negative.")
    if not (0.0 < controls["damping"] <= 1.0) or max_cycles < 1:
        raise ValueError("FAS damping must be in (0,1] and max cycles must be positive.")

    shape = (int(model.ny), int(model.nx))
    bottom_array = np.asarray(bottom, dtype=np.float64)
    active = np.asarray(model.active_host, dtype=np.int32)
    prescribed = np.asarray(model.bc_mask_host, dtype=np.int32)
    prescribed_values = np.asarray(model.bc_values_host, dtype=np.float64)
    initial = kwargs.get("initial_head")
    if initial is None:
        initial = bottom_array + max(float(kwargs.get("initial_saturated_thickness", 10.0)), min_sat)
    initial = np.asarray(initial, dtype=np.float64).copy()
    if initial.shape != shape or not np.all(np.isfinite(initial)):
        raise ValueError(f"initial_head must be finite with shape {shape}.")
    initial[active == 0] = 0.0
    initial[prescribed != 0] = prescribed_values[prescribed != 0]
    if transient and previous_head is None:
        previous_head = initial.copy()
    previous_array = initial.copy() if previous_head is None else np.asarray(previous_head, dtype=np.float64)

    fine_physical = make_fine_physical_level(
        conductivity=K_field,
        top=top,
        bottom=bottom,
        active=active,
        dirichlet_mask=prescribed,
        dirichlet_values=prescribed_values,
        source_rate=np.asarray(model.R_field_host, dtype=np.float64),
        ghb_mask=np.asarray(model.gh_mask_host, dtype=np.int32),
        ghb_factor=np.asarray(model.ghb_factor_host, dtype=np.float64),
        ghb_external_head=np.asarray(model.gh_head_host, dtype=np.float64),
        sy=sy,
        ss=ss,
        previous_head=previous_array,
        dx=float(model.dx),
    )
    physical_levels = build_fas_physical_hierarchy(
        fine_physical,
        max_levels=max_levels,
        min_coarse_cells=min_coarse_cells,
        min_sat=min_sat,
    )
    owner = model._resource_owner
    owner.refresh(hierarchy=model.mg_levels, work=model._mg_work, cuda_graph=model._kcycle_graph)
    workspace = owner.get_experimental_workspace("unconfined_fas")
    if workspace is None or not workspace.compatible(
        physical_levels=physical_levels,
        transient=transient,
        dt=dt,
        min_sat=min_sat,
        device=model.device_str,
    ):
        workspace = FASWorkspace(
            physical_levels=physical_levels,
            transient=transient,
            dt=dt,
            min_sat=min_sat,
            device=model.device_str,
        )
        owner.set_experimental_workspace("unconfined_fas", workspace)
    workspace.reset_counters()
    fine = workspace.levels[0]
    fine.nonlinear_operator.operator.set_head(initial)
    wp.copy(fine.head, fine.nonlinear_operator.operator.head_device)
    wp.copy(fine.forcing, fine.physical_forcing)

    cycle_history = []
    converged = False
    failure_reason = None
    rejected_total = 0
    damped_total = 0
    start_time = time.perf_counter()
    try:
        wp.synchronize_device(model.device_str)
    except Exception:
        pass
    memory_before = (
        int(wp.get_mempool_used_mem_current(model.device_str))
        if str(model.device_str).startswith("cuda") and hasattr(wp, "get_mempool_used_mem_current") else None
    )
    fine.nonlinear_operator.evaluate(head=fine.head, state=fine)
    fine.nonlinear_operator.refresh_frozen_diagonal(head=fine.head, state=fine)
    workspace.kernel_launches += 4
    residual_rms, residual_max = _norm(level=fine, value=fine.defect, workspace=workspace)
    head_equivalent_rms, _ = _norm(
        level=fine,
        value=fine.defect,
        workspace=workspace,
        head_equivalent=True,
    )
    initial_residual = residual_rms
    if residual_rms <= residual_tolerance and head_equivalent_rms <= head_residual_tolerance:
        converged = True

    for cycle_index in range(max_cycles):
        if converged:
            break
        _launch_copy(fine, fine.head, fine.head_cycle_start, workspace)
        cycle_diag = {
            "cycle": cycle_index + 1,
            "residual_rms_before": residual_rms,
            "tau_norms": [],
            "restricted_defect_norms": [],
            "coarse_corrections": [],
            "rejected_corrections": 0,
            "damped_corrections": 0,
        }
        _fas_vcycle(level_index=0, workspace=workspace, controls=controls, cycle_diagnostics=cycle_diag)
        fine.nonlinear_operator.evaluate(head=fine.head, state=fine)
        fine.nonlinear_operator.refresh_frozen_diagonal(head=fine.head, state=fine)
        workspace.kernel_launches += 4
        residual_rms, residual_max = _norm(level=fine, value=fine.defect, workspace=workspace)
        head_equivalent_rms, _ = _norm(
            level=fine,
            value=fine.defect,
            workspace=workspace,
            head_equivalent=True,
        )
        wp.launch(
            kernel=_k.fas_difference_kernel,
            dim=fine.shape,
            inputs=[
                fine.head,
                fine.head_cycle_start,
                fine.nonlinear_operator.operator.active_device,
                fine.nonlinear_operator.operator.dirichlet_mask_device,
                fine.correction,
                fine.physical.nx,
                fine.physical.ny,
            ],
            device=workspace.device,
        )
        workspace.kernel_launches += 1
        dh_rms, dh_max = _norm(level=fine, value=fine.correction, workspace=workspace)
        cycle_diag.update(
            {
                "residual_rms_after": residual_rms,
                "residual_max_after": residual_max,
                "head_equivalent_residual_rms": head_equivalent_rms,
                "head_change_rms": dh_rms,
                "head_change_max": dh_max,
            }
        )
        rejected_total += cycle_diag["rejected_corrections"]
        damped_total += cycle_diag["damped_corrections"]
        cycle_history.append(cycle_diag)
        if not np.isfinite(residual_rms):
            failure_reason = "nonfinite_fine_residual"
            break
        if residual_rms <= residual_tolerance and head_equivalent_rms <= head_residual_tolerance:
            converged = True
            break
        if len(cycle_history) >= 5 and residual_rms > 10.0 * initial_residual:
            failure_reason = "fas_residual_divergence"
            break
    if not converged and failure_reason is None:
        failure_reason = "fas_cycle_limit"

    try:
        wp.synchronize_device(model.device_str)
    except Exception:
        pass
    runtime = float(time.perf_counter() - start_time)
    memory_after = (
        int(wp.get_mempool_used_mem_current(model.device_str))
        if str(model.device_str).startswith("cuda") and hasattr(wp, "get_mempool_used_mem_current") else None
    )
    memory_peak = (
        int(wp.get_mempool_used_mem_high(model.device_str))
        if str(model.device_str).startswith("cuda") and hasattr(wp, "get_mempool_used_mem_high") else None
    )
    head = np.asarray(fine.head.numpy(), dtype=np.float64).copy()
    storage_terms = fine.nonlinear_operator.operator.exact_storage_terms(head) if transient else None
    saturation = np.asarray(
        fine.nonlinear_operator.operator.saturated_thickness(head).numpy(), dtype=np.float64
    ).copy()
    transmissivity = np.asarray(K_field, dtype=np.float64) * saturation
    transmissivity[active == 0] = 0.0
    from DARCY_WARP_PACKAGE.physics.budgets_2d import add_exact_storage_to_budget, compute_mass_balance_budget
    budget = compute_mass_balance_budget(
        T_field=transmissivity,
        R_field=np.asarray(model.R_field_host, dtype=np.float64),
        head=head,
        active=active,
        bc_mask=prescribed,
        bc_values=prescribed_values,
        dx=float(model.dx),
        gh_mask=np.asarray(model.gh_mask_host, dtype=np.int32) if model.use_ghb else None,
        gh_head=np.asarray(model.gh_head_host, dtype=np.float64) if model.use_ghb else None,
        gh_width=np.asarray(model.gh_width_host, dtype=np.float64) if model.use_ghb else None,
        ghb_factor=np.asarray(model.ghb_factor_host, dtype=np.float64) if model.use_ghb else None,
        case="unconfined_fas",
    )
    if storage_terms is not None:
        budget = add_exact_storage_to_budget(budget, storage_terms.total)
    info = {
        "converged": bool(converged),
        "solver_type": "unconfined_fas",
        "solver_backend": "unconfined_fas",
        "experimental_backend": True,
        "fas_cycles": len(cycle_history),
        "fas_cycle_history": cycle_history,
        "residual_history": [initial_residual] + [item["residual_rms_after"] for item in cycle_history],
        "true_nonlinear_residual_rms": residual_rms,
        "final_residual": residual_rms,
        "head_equivalent_residual_rms": head_equivalent_rms,
        "pre_smoothing_sweeps_by_level": list(workspace.pre_sweeps),
        "post_smoothing_sweeps_by_level": list(workspace.post_sweeps),
        "coarse_smoothing_sweeps_by_level": list(workspace.coarse_sweeps),
        "coarse_nonlinear_work": int(sum(workspace.coarse_sweeps)),
        "smoothing_history": list(workspace.smoothing_history),
        "smoothing_factors_by_level": [
            {
                "level": item["level"],
                "phase": item["phase"],
                "factor": item["smoothing_factor"],
            }
            for item in workspace.smoothing_history
        ],
        "rejected_corrections": int(rejected_total),
        "damped_corrections": int(damped_total),
        "kernel_launches": int(workspace.kernel_launches),
        "transfer_launches": int(workspace.transfer_launches),
        "gpu_scalar_synchronization_count": int(workspace.synchronizations),
        "runtime_seconds": runtime,
        "persistent_memory_before_bytes": memory_before,
        "persistent_memory_after_bytes": memory_after,
        "peak_memory_bytes": memory_peak,
        "fas_failure_reason": failure_reason,
        "fas_fallback_used": False,
        "fallback_state": "not_used",
        "n_levels": len(workspace.levels),
        "level_shapes": [level.shape for level in workspace.levels],
        "coarse_operator": "nonlinear_physical_rediscretization",
        "coarse_solver": "repeated_damped_nonlinear_weighted_jacobi",
        "tau_formulation": "N_coarse(R_head)-R(N_fine(head))",
        "storage_total_array": None if storage_terms is None else storage_terms.total,
        "storage_sy_array": None if storage_terms is None else storage_terms.sy,
        "storage_ss_array": None if storage_terms is None else storage_terms.ss,
        "saturated_thickness_array": saturation,
        "transmissivity_array": transmissivity,
        "active_mask": active != 0,
        "dry_mask": head <= bottom_array + min_sat,
        "budget": budget,
        "budget_summary": dict(budget.iloc[0]),
    }
    if not converged and fallback_enabled:
        return _run_fallback(
            context=context,
            order=fallback_order,
            original_kwargs=original_kwargs,
            initial_head=initial,
            fas_info=info,
            return_info=return_info,
        )
    return (head, info) if return_info else head


class UnconfinedFASBackend:
    name = "unconfined_fas"

    def solve(self, context: SolverContext, **kwargs: Any):
        return solve_unconfined_fas(context=context, **kwargs)


__all__ = ["UnconfinedFASBackend", "solve_unconfined_fas"]
