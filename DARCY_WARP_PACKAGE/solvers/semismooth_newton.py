# SPDX-License-Identifier: AGPL-3.0-only
"""Experimental globalized semismooth Newton--FGMRES--K-cycle backend."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D, from_unconfined_solve_inputs
from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT
from .base import SolverContext
from .fgmres import FGMRESWorkspace2D, RestartedFGMRES2D
from .kcycle_preconditioner import (
    FixedWorkKCyclePreconditioner2D,
    KCyclePreconditionerWorkspace2D,
)
from .newton_state import NewtonOperatorWorkspace2D
from . import newton_kernels as _k


@dataclass(slots=True)
class NewtonVectors2D:
    shape: tuple[int, int]
    device: str
    candidate: Any
    candidate_residual: Any
    change_sq: Any
    change_max: Any
    finite_flag: Any
    head_equivalent_sq: Any
    closed: bool = False

    @classmethod
    def allocate(cls, *, shape: tuple[int, int], device: str) -> "NewtonVectors2D":
        return cls(
            shape=shape,
            device=str(device),
            candidate=wp.zeros(shape, dtype=WP_FLOAT, device=device),
            candidate_residual=wp.zeros(shape, dtype=WP_FLOAT, device=device),
            change_sq=wp.zeros(1, dtype=wp.float64, device=device),
            change_max=wp.zeros(1, dtype=wp.float64, device=device),
            finite_flag=wp.zeros(1, dtype=wp.int32, device=device),
            head_equivalent_sq=wp.zeros(1, dtype=wp.float64, device=device),
        )

    def compatible(self, *, shape: tuple[int, int], device: str) -> bool:
        return not self.closed and self.shape == tuple(shape) and self.device == str(device)

    def close(self) -> None:
        if self.closed:
            return
        self.candidate = None
        self.candidate_residual = None
        self.change_sq = None
        self.change_max = None
        self.finite_flag = None
        self.head_equivalent_sq = None
        self.closed = True


def _cached_workspace(*, owner: Any, name: str, build: Any, compatible: Any) -> Any:
    workspace = owner.get_experimental_workspace(name)
    if workspace is None or not compatible(workspace):
        workspace = build()
        owner.set_experimental_workspace(name, workspace)
    return workspace


def _head_equivalent_rms(
    *,
    residual: Any,
    diagonal_inverse: Any,
    operator: NonlinearOperator2D,
    vectors: NewtonVectors2D,
) -> float:
    vectors.head_equivalent_sq.fill_(wp.float64(0.0))
    wp.launch(
        kernel=_k.head_equivalent_norm_kernel,
        dim=operator.ctx.shape,
        inputs=[
            residual,
            diagonal_inverse,
            operator.active_device,
            operator.dirichlet_mask_device,
            vectors.head_equivalent_sq,
            operator.ctx.nx,
            operator.ctx.ny,
        ],
        device=operator.device,
    )
    total = float(vectors.head_equivalent_sq.numpy()[0])
    return float(np.sqrt(max(total, 0.0) / float(operator.n_free))) if operator.n_free else 0.0


def _fallback_to_picard(
    *,
    context: SolverContext,
    fallback_kwargs: dict[str, Any],
    newton_info: dict[str, Any],
    return_info: bool,
):
    from .registry import select_backend

    backend = select_backend(
        solver="unconfined_picard_kcycle",
        formulation="unconfined",
        transient=context.transient,
        default="unconfined_picard_kcycle",
    )
    fallback_kwargs["return_info"] = True
    head, picard_info = backend.solve(context, **fallback_kwargs)
    merged = dict(picard_info)
    merged.update(
        {
            "newton_fallback_used": True,
            "newton_failure_reason": newton_info.get("newton_failure_reason"),
            "newton_diagnostics_before_fallback": newton_info,
            "fallback_backend": "unconfined_picard_kcycle",
            "fallback_state": "picard_completed",
            "experimental_backend": True,
        }
    )
    return (head, merged) if return_info else head


def solve_semismooth_newton(*, context: SolverContext, **kwargs: Any):
    """Solve the Stage-1 equation with analytic Jv and flexible GMRES."""
    model = context.model
    original_kwargs = dict(kwargs)
    return_info = bool(kwargs.pop("return_info", True))
    initial_head = kwargs.get("initial_head")
    K_field = kwargs.get("K_field")
    zbot_field = kwargs.get("zbot_field")
    ztop_field = kwargs.get("ztop_field")
    transient = bool(kwargs.get("transient", False))
    dt = kwargs.get("dt")
    head_prev = kwargs.get("head_prev")
    storage_coeff = kwargs.get("storage_coeff")
    sy = kwargs.get("sy")
    ss = kwargs.get("ss", 0.0)
    if K_field is None or zbot_field is None:
        raise ValueError("unconfined_semismooth_newton_kcycle requires K_field and zbot_field.")
    if transient and ztop_field is None:
        raise ValueError("transient semismooth Newton requires ztop_field for exact storage.")
    if transient and sy is None:
        storage_arr = np.asarray(storage_coeff) if storage_coeff is not None else np.asarray([])
        if storage_arr.ndim == 0 and storage_arr.size == 1:
            sy = float(storage_arr.reshape(()))
        else:
            raise ValueError("transient semismooth Newton requires scalar sy (or scalar storage_coeff).")
    sy = 0.0 if sy is None else float(sy)
    ss = 0.0 if ss is None else float(ss)

    min_sat = kwargs.get("unconfined_min_sat")
    if min_sat is None:
        min_sat = kwargs.get("min_saturated_thickness", 0.1)
    if min_sat is None:
        min_sat = 0.1
    min_sat = float(min_sat)

    restart = int(kwargs.pop("newton_fgmres_restart", 20))
    max_krylov = int(kwargs.pop("newton_fgmres_max_iterations", max(40, 3 * restart)))
    krylov_rtol = float(kwargs.pop("newton_fgmres_relative_tolerance", 1.0e-5))
    krylov_atol = float(kwargs.pop("newton_fgmres_absolute_tolerance", 1.0e-10))
    max_newton = int(kwargs.pop("newton_max_iterations", 20))
    residual_tol = float(kwargs.pop("newton_residual_rms_tolerance", 1.0e-6))
    head_residual_tol = float(kwargs.pop("newton_head_equivalent_rms_tolerance", kwargs.get("hclose", 1.0e-4) or 1.0e-4))
    max_backtracks = int(kwargs.pop("newton_max_backtracks", 10))
    min_step = float(kwargs.pop("newton_min_step_length", 2.0 ** -10))
    armijo = float(kwargs.pop("newton_armijo_coefficient", 1.0e-4))
    max_head_change = float(kwargs.pop("newton_max_head_change", kwargs.get("max_head_change_per_outer_iteration", 5.0)))
    fallback_enabled = bool(kwargs.pop("newton_fallback_to_picard", True))
    preconditioner_cycles = int(kwargs.pop("newton_preconditioner_kcycles", 1))
    max_levels = int(kwargs.get("max_levels", 5))
    min_coarse_cells = kwargs.get("min_coarse_cells", 500)

    if restart < 2 or max_krylov < 1 or max_newton < 1:
        raise ValueError("Newton iteration limits and FGMRES restart must be positive (restart >= 2).")
    if not (0.0 < min_step <= 1.0) or max_backtracks < 0:
        raise ValueError("Newton line-search controls require 0 < min_step <= 1 and max_backtracks >= 0.")
    if max_head_change <= 0.0:
        raise ValueError("newton_max_head_change must be positive.")

    ny, nx = int(model.ny), int(model.nx)
    shape = (ny, nx)
    zbot = np.asarray(zbot_field, dtype=np.float64)
    active = np.asarray(model.active_host, dtype=np.int32) != 0
    prescribed = np.asarray(model.bc_mask_host, dtype=np.int32) != 0
    prescribed_values = np.asarray(model.bc_values_host, dtype=np.float64)
    if initial_head is None:
        initial_sat = float(kwargs.get("initial_saturated_thickness", 10.0))
        initial = zbot + max(initial_sat, min_sat)
    else:
        initial = np.asarray(initial_head, dtype=np.float64).copy()
    if initial.shape != shape or not np.all(np.isfinite(initial)):
        raise ValueError(f"initial_head must be finite with shape {shape}.")
    initial[~active] = 0.0
    initial[prescribed] = prescribed_values[prescribed]
    if transient and head_prev is None:
        head_prev = initial.copy()

    nonlinear_context = from_unconfined_solve_inputs(
        model,
        K_field=K_field,
        zbot_field=zbot_field,
        ztop_field=ztop_field,
        sy=sy,
        ss=ss,
        dt=dt,
        head_prev=head_prev,
        min_sat=min_sat,
        transient=transient,
    )

    # Hierarchy topology and its work buffers are model-owned and built at most
    # once here.  Frozen coefficients remain in a separate experimental cache.
    if model.mg_levels is None or bool(model._operator_dirty):
        model.build_hierarchy(
            max_levels=max_levels,
            min_coarse_n=4,
            min_coarse_cells=min_coarse_cells,
        )
    levels = model.mg_levels
    owner = model._resource_owner
    owner.refresh(hierarchy=levels, work=model._mg_work, cuda_graph=model._kcycle_graph)

    fgmres_workspace = _cached_workspace(
        owner=owner,
        name="semismooth_newton_fgmres",
        build=lambda: FGMRESWorkspace2D(shape=shape, restart=restart, device=model.device_str),
        compatible=lambda item: item.compatible(shape=shape, restart=restart, device=model.device_str),
    )
    vector_workspace = _cached_workspace(
        owner=owner,
        name="semismooth_newton_vectors",
        build=lambda: NewtonVectors2D.allocate(shape=shape, device=model.device_str),
        compatible=lambda item: item.compatible(shape=shape, device=model.device_str),
    )
    preconditioner_workspace = _cached_workspace(
        owner=owner,
        name="semismooth_newton_kcycle",
        build=lambda: KCyclePreconditionerWorkspace2D(levels=levels, device=model.device_str),
        compatible=lambda item: item.compatible(levels=levels, device=model.device_str),
    )

    preconditioner = FixedWorkKCyclePreconditioner2D(
        model=model,
        levels=levels,
        workspace=preconditioner_workspace,
        n_cycles=preconditioner_cycles,
        nu_pre=int(kwargs.get("nu_pre", 2)),
        nu_post=int(kwargs.get("nu_post", 2)),
        nu_coarse=int(kwargs.get("nu_coarse", 30)),
        smoother=str(kwargs.get("smoother", "chebyshev")),
        omega=float(kwargs.get("omega", 0.8)),
        cheby_lambda_min=float(kwargs.get("cheby_lambda_min", 0.05)),
        cheby_lambda_max=float(kwargs.get("cheby_lambda_max", 1.95)),
    )

    # The authoritative operator is cached on the resource owner and reused
    # across solves: structural inputs decide reuse, and per-timestep state
    # (previous head, dt, source field, Sy, Ss) is refreshed in place via
    # update_transient_state.  This mirrors the FAS workspace pattern and
    # removes the ~29 device allocations + ~12 field uploads the backend
    # previously paid every solve.  ``set_experimental_workspace`` closes the
    # previous workspace on rebuild and ``release`` closes it on model close.
    operator_workspace = owner.get_experimental_workspace("semismooth_newton_operator")
    workspace_reused = operator_workspace is not None and operator_workspace.compatible(
        context=nonlinear_context,
        transient=transient,
        min_sat=min_sat,
        device=model.device_str,
    )
    if not workspace_reused:
        operator_workspace = NewtonOperatorWorkspace2D(
            context=nonlinear_context,
            transient=transient,
            min_sat=min_sat,
            device=model.device_str,
        )
        owner.set_experimental_workspace("semismooth_newton_operator", operator_workspace)
    operator_workspace.refresh(
        head_prev=head_prev if transient else None,
        dt=dt if transient else None,
        source_rate=np.asarray(model.R_field_host, dtype=np.float64),
        sy=sy,
        ss=ss,
    )
    operator = operator_workspace.operator
    history: list[dict[str, Any]] = []
    failure_reason: str | None = None
    converged = False
    total_fgmres_iterations = 0
    total_fgmres_restarts = 0
    total_reductions = 0
    total_backtracks = 0
    accepted_updates = 0
    start_memory = None
    end_memory = None
    start_time = time.perf_counter()
    try:
        try:
            wp.synchronize_device(model.device_str)
        except Exception:
            pass
        if str(model.device_str).startswith("cuda") and hasattr(wp, "get_mempool_used_mem_current"):
            start_memory = int(wp.get_mempool_used_mem_current(model.device_str))

        operator.set_head(initial)
        operator.residual_device(operator.head_device, reduce=True)
        norms = operator.current_reduced_norms()
        T_frozen, S_frozen = operator.freeze_picard_device(operator.head_device)
        preconditioner.freeze(transmissivity=T_frozen, storage_diagonal=S_frozen)
        head_equivalent = _head_equivalent_rms(
            residual=operator.residual_device_array,
            diagonal_inverse=preconditioner.fine_diagonal_inverse,
            operator=operator,
            vectors=vector_workspace,
        )
        initial_residual = norms.rms

        if norms.rms <= residual_tol and head_equivalent <= head_residual_tol:
            converged = True

        for newton_iteration in range(max_newton):
            if converged:
                break
            wp.launch(
                kernel=_k.masked_copy_kernel,
                dim=shape,
                inputs=[
                    operator.residual_device_array,
                    operator.active_device,
                    operator.dirichlet_mask_device,
                    fgmres_workspace.rhs,
                    -1.0,
                    nx,
                    ny,
                ],
                device=model.device_str,
            )
            krylov = RestartedFGMRES2D(
                workspace=fgmres_workspace,
                active=operator.active_device,
                prescribed=operator.dirichlet_mask_device,
                nx=nx,
                ny=ny,
                device=model.device_str,
            )
            result = krylov.solve(
                rhs=fgmres_workspace.rhs,
                apply_jacobian=lambda vector, out: operator.jacobian_vector_device(operator.head_device, vector, out=out),
                apply_preconditioner=lambda rhs, out: preconditioner.apply(rhs, out),
                relative_tolerance=krylov_rtol,
                absolute_tolerance=krylov_atol,
                max_iterations=max_krylov,
            )
            total_fgmres_iterations += result.iterations
            total_fgmres_restarts += result.restarts
            total_reductions += result.reduction_count
            if not result.converged:
                failure_reason = result.breakdown_reason or "fgmres_nonconvergence"
                break

            alpha = 1.0
            accepted = False
            backtracks = 0
            candidate_norms = None
            dh_rms = float("nan")
            dh_max = float("nan")
            while backtracks <= max_backtracks and alpha >= min_step:
                vector_workspace.change_sq.fill_(wp.float64(0.0))
                vector_workspace.change_max.fill_(wp.float64(0.0))
                vector_workspace.finite_flag.fill_(wp.int32(0))
                wp.launch(
                    kernel=_k.masked_candidate_kernel,
                    dim=shape,
                    inputs=[
                        operator.head_device,
                        fgmres_workspace.solution,
                        operator.active_device,
                        operator.dirichlet_mask_device,
                        operator.dirichlet_values_device,
                        alpha,
                        vector_workspace.candidate,
                        vector_workspace.change_sq,
                        vector_workspace.change_max,
                        vector_workspace.finite_flag,
                        nx,
                        ny,
                    ],
                    device=model.device_str,
                )
                nonfinite = int(vector_workspace.finite_flag.numpy()[0]) != 0
                dh_max = float(vector_workspace.change_max.numpy()[0])
                dh_sq = float(vector_workspace.change_sq.numpy()[0])
                dh_rms = float(np.sqrt(max(dh_sq, 0.0) / float(operator.n_free))) if operator.n_free else 0.0
                if not nonfinite and dh_max <= max_head_change:
                    operator.residual_device(vector_workspace.candidate, out=vector_workspace.candidate_residual, reduce=True)
                    candidate_norms = operator.current_reduced_norms()
                    target = max(0.0, (1.0 - armijo * alpha) * norms.rms)
                    if np.isfinite(candidate_norms.rms) and candidate_norms.rms <= target:
                        accepted = True
                        break
                alpha *= 0.5
                backtracks += 1

            total_backtracks += backtracks
            history.append(
                {
                    "newton_iteration": int(newton_iteration + 1),
                    "residual_rms_before": float(norms.rms),
                    "residual_rms_after": None if candidate_norms is None else float(candidate_norms.rms),
                    "accepted_step_length": float(alpha) if accepted else 0.0,
                    "backtrack_count": int(backtracks),
                    "head_change_rms": float(dh_rms),
                    "head_change_max": float(dh_max),
                    "fgmres_iterations": int(result.iterations),
                    "fgmres_restarts": int(result.restarts),
                    "fgmres_final_residual": float(result.final_residual),
                    "fgmres_breakdown": bool(result.breakdown),
                    "fgmres_breakdown_reason": result.breakdown_reason,
                }
            )
            if not accepted or candidate_norms is None:
                failure_reason = "line_search_failed"
                break

            operator.set_head_device(vector_workspace.candidate)
            wp.copy(operator.residual_device_array, vector_workspace.candidate_residual)
            norms = candidate_norms
            accepted_updates += 1
            T_frozen, S_frozen = operator.freeze_picard_device(operator.head_device)
            preconditioner.freeze(transmissivity=T_frozen, storage_diagonal=S_frozen)
            head_equivalent = _head_equivalent_rms(
                residual=operator.residual_device_array,
                diagonal_inverse=preconditioner.fine_diagonal_inverse,
                operator=operator,
                vectors=vector_workspace,
            )
            history[-1]["head_equivalent_residual_rms"] = float(head_equivalent)
            if norms.rms <= residual_tol and head_equivalent <= head_residual_tol:
                converged = True
                break

        if not converged and failure_reason is None:
            failure_reason = "newton_iteration_limit"

        try:
            wp.synchronize_device(model.device_str)
        except Exception:
            pass
        if str(model.device_str).startswith("cuda") and hasattr(wp, "get_mempool_used_mem_current"):
            end_memory = int(wp.get_mempool_used_mem_current(model.device_str))
        runtime = float(time.perf_counter() - start_time)
        head = np.asarray(operator.head_device.numpy(), dtype=np.float64).copy()
        storage_terms = operator.exact_storage_terms(head) if transient else None
        saturation = np.asarray(operator.saturated_thickness(head).numpy(), dtype=np.float64).copy()
        transmissivity = np.asarray(K_field, dtype=np.float64) * saturation
        transmissivity[~active] = 0.0
        from DARCY_WARP_PACKAGE.physics.budgets_2d import (
            add_exact_storage_to_budget,
            compute_mass_balance_budget,
        )
        budget = compute_mass_balance_budget(
            T_field=transmissivity,
            R_field=np.asarray(model.R_field_host, dtype=np.float64),
            head=head,
            active=np.asarray(model.active_host, dtype=np.int32),
            bc_mask=np.asarray(model.bc_mask_host, dtype=np.int32),
            bc_values=np.asarray(model.bc_values_host, dtype=np.float64),
            dx=float(model.dx),
            gh_mask=np.asarray(model.gh_mask_host, dtype=np.int32) if model.use_ghb else None,
            gh_head=np.asarray(model.gh_head_host, dtype=np.float64) if model.use_ghb else None,
            gh_width=np.asarray(model.gh_width_host, dtype=np.float64) if model.use_ghb else None,
            ghb_factor=np.asarray(model.ghb_factor_host, dtype=np.float64) if model.use_ghb else None,
            case="unconfined_semismooth_newton_kcycle",
        )
        if storage_terms is not None:
            budget = add_exact_storage_to_budget(budget, storage_terms.total)
        info = {
            "converged": bool(converged),
            "solver_type": "unconfined_semismooth_newton_kcycle",
            "solver_backend": "unconfined_semismooth_newton_kcycle",
            "experimental_backend": True,
            "newton_iterations": int(accepted_updates),
            "outer_picard_iterations": 0,
            "outer_history": history,
            "nonlinear_residual_history": [float(initial_residual)] + [
                float(item["residual_rms_after"]) for item in history if item["residual_rms_after"] is not None
            ],
            "final_residual": float(norms.rms),
            "true_nonlinear_residual_rms": float(norms.rms),
            "head_equivalent_residual_rms": float(head_equivalent),
            "fgmres_iterations": int(total_fgmres_iterations),
            "fgmres_restarts": int(total_fgmres_restarts),
            "fgmres_restart_size": int(restart),
            "fgmres_reduction_count": int(total_reductions),
            "kcycle_preconditioner_applications": int(preconditioner.applications),
            "kcycle_preconditioner_cycles": int(preconditioner.cycles),
            "backtrack_count": int(total_backtracks),
            "breakdown_state": failure_reason if failure_reason and "breakdown" in failure_reason else None,
            "newton_failure_reason": failure_reason,
            "newton_fallback_used": False,
            "fallback_state": "not_used",
            "generalized_derivative_convention": "one_strictly_inside_zero_outside_and_at_clip_thresholds",
            "runtime_seconds": runtime,
            "gpu_scalar_synchronization_count": int(total_reductions + 3 * (accepted_updates + total_backtracks + 1)),
            "persistent_memory_before_bytes": start_memory,
            "persistent_memory_after_bytes": end_memory,
            "newton_workspace_reused": bool(workspace_reused),
            "newton_workspace_refresh_count": int(operator_workspace.refresh_count),
            "storage_total_array": None if storage_terms is None else storage_terms.total,
            "storage_sy_array": None if storage_terms is None else storage_terms.sy,
            "storage_ss_array": None if storage_terms is None else storage_terms.ss,
            "saturated_thickness_array": saturation,
            "transmissivity_array": transmissivity,
            "budget": budget,
            "budget_summary": dict(budget.iloc[0]),
            "dry_mask": np.asarray(head <= (zbot + min_sat), dtype=bool),
            "active_mask": active.copy(),
        }

        if not converged and fallback_enabled:
            fallback_kwargs = dict(original_kwargs)
            for key in tuple(fallback_kwargs):
                if key.startswith("newton_"):
                    fallback_kwargs.pop(key)
            fallback_kwargs["initial_head"] = initial.copy()
            fallback_kwargs["K_field"] = K_field
            fallback_kwargs["zbot_field"] = zbot_field
            fallback_kwargs["ztop_field"] = ztop_field
            fallback_kwargs["transient"] = transient
            fallback_kwargs["dt"] = dt
            fallback_kwargs["head_prev"] = head_prev
            return _fallback_to_picard(
                context=context,
                fallback_kwargs=fallback_kwargs,
                newton_info=info,
                return_info=return_info,
            )
        return (head, info) if return_info else head
    finally:
        # The operator is owned by the cached NewtonOperatorWorkspace2D and is
        # reused across solves; do not close it here.  It is closed automatically
        # when an incompatible workspace replaces it (set_experimental_workspace)
        # and when the model is released.
        pass


class UnconfinedSemismoothNewtonKCycleBackend:
    """Explicit experimental 2D nonlinear backend."""

    name = "unconfined_semismooth_newton_kcycle"

    def solve(self, context: SolverContext, **kwargs: Any):
        return solve_semismooth_newton(context=context, **kwargs)


__all__ = ["UnconfinedSemismoothNewtonKCycleBackend", "solve_semismooth_newton"]
