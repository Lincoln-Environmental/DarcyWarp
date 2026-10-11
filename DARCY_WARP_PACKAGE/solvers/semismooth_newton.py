# SPDX-License-Identifier: AGPL-3.0-only
"""Globalized semismooth Newton--FGMRES--K-cycle production backend."""

from __future__ import annotations

from dataclasses import dataclass, replace
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


def _backtrack_newton_direction(*, operator, vectors, correction, residual_rms,
                                max_backtracks, min_step, max_head_change, armijo):
    """Evaluate candidates without committing, using the native acceptance rule."""
    alpha, backtracks = 1.0, 0
    candidate_norms = None
    dh_rms, dh_max = float('nan'), float('nan')
    accepted = False
    while backtracks <= max_backtracks and alpha >= min_step:
        vectors.change_sq.fill_(wp.float64(0.))
        vectors.change_max.fill_(wp.float64(0.))
        vectors.finite_flag.fill_(wp.int32(0))
        wp.launch(kernel=_k.masked_candidate_kernel, dim=operator.ctx.shape,
            inputs=[operator.head_device, correction, operator.active_device,
                operator.dirichlet_mask_device, operator.dirichlet_values_device, alpha,
                vectors.candidate, vectors.change_sq, vectors.change_max, vectors.finite_flag,
                operator.ctx.nx, operator.ctx.ny], device=operator.device)
        dh_max = float(vectors.change_max.numpy()[0])
        dh_sq = float(vectors.change_sq.numpy()[0])
        dh_rms = float(np.sqrt(max(dh_sq, 0.)/operator.n_free)) if operator.n_free else 0.
        if int(vectors.finite_flag.numpy()[0]) == 0 and dh_max <= max_head_change:
            operator.residual_device(vectors.candidate, out=vectors.candidate_residual, reduce=True)
            candidate_norms = operator.current_reduced_norms()
            if np.isfinite(candidate_norms.rms) and candidate_norms.rms <= max(
                    0., (1.-armijo*alpha)*residual_rms):
                accepted = True
                break
        alpha *= .5
        backtracks += 1
    return accepted, alpha, backtracks, candidate_norms, dh_rms, dh_max


def _gated_newton_model_step(*, operator, preconditioner, vectors, workspace,
                             residual_norms, relative_tolerance, absolute_tolerance,
                             max_iterations, max_model_work, max_backtracks, min_step,
                             max_head_change, armijo):
    """Backtrack the Newton model with gated fluxes solved on each branch.

    F_flux(H)+J_flux(H)*dH+Q_gate(H+dH) = (1-alpha)*F(H).
    Interior derivatives are unchanged. Only the piecewise affine boundary
    terms are exact in this model; candidate acceptance uses the real residual.
    """
    shape, nx, ny = operator.ctx.shape, operator.ctx.nx, operator.ctx.ny
    seed = wp.clone(workspace.solution)
    branch = wp.clone(operator.head_device)
    krylov = RestartedFGMRES2D(workspace=workspace, active=operator.active_device,
        prescribed=operator.dirichlet_mask_device, nx=nx, ny=ny, device=operator.device)

    def apply_model(vector, out):
        return operator.jacobian_vector_device(head=operator.head_device, vector=vector,
            out=out, boundary_head=branch)

    def apply_preconditioner(rhs, out):
        return preconditioner.apply(rhs=rhs, out=out)

    vectors.change_sq.fill_(wp.float64(0.))
    vectors.change_max.fill_(wp.float64(0.))
    vectors.finite_flag.fill_(wp.int32(0))
    wp.launch(kernel=_k.masked_candidate_kernel, dim=shape,
        inputs=[operator.head_device, seed, operator.active_device, operator.dirichlet_mask_device,
            operator.dirichlet_values_device, 1., branch, vectors.change_sq, vectors.change_max,
            vectors.finite_flag, nx, ny], device=operator.device)
    seed_max = float(vectors.change_max.numpy()[0])
    alpha, backtracks = 1., 0
    while alpha*seed_max > max_head_change and backtracks <= max_backtracks:
        alpha *= .5
        backtracks += 1
    history = []
    totals = dict(iterations=0, restarts=0, reduction_count=0)
    result = None
    accepted = False
    candidate_norms = None
    dh_rms, dh_max = float('nan'), float('nan')
    while backtracks <= max_backtracks and alpha >= min_step and len(history) < max_model_work:
        vectors.change_sq.fill_(wp.float64(0.))
        vectors.change_max.fill_(wp.float64(0.))
        vectors.finite_flag.fill_(wp.int32(0))
        wp.launch(kernel=_k.masked_candidate_kernel, dim=shape,
            inputs=[operator.head_device, seed, operator.active_device, operator.dirichlet_mask_device,
                operator.dirichlet_values_device, alpha, branch, vectors.change_sq, vectors.change_max,
                vectors.finite_flag, nx, ny], device=operator.device)
        settled = False
        for _ in range(max_backtracks+1):
            if len(history) >= max_model_work:
                break
            t, diagonal = operator.freeze_picard_device(head=operator.head_device, boundary_head=branch)
            preconditioner.freeze(transmissivity=t, storage_diagonal=diagonal)
            wp.launch(kernel=_k.masked_copy_kernel, dim=shape,
                inputs=[operator.residual_device_array, operator.active_device,
                    operator.dirichlet_mask_device, workspace.rhs, -alpha, nx, ny], device=operator.device)
            operator.gated_model_rhs_device(head=operator.head_device, branch_head=branch, out=workspace.rhs)
            rhs_norm = krylov._norm(workspace.rhs)
            # Affine gate offsets can enlarge the RHS. Keep the requested
            # true residual at least as strict as against alpha*original F.
            rtol = relative_tolerance*min(1., alpha*residual_norms.l2/max(rhs_norm, np.finfo(float).tiny))
            result = krylov.solve(rhs=workspace.rhs, apply_jacobian=apply_model,
                apply_preconditioner=apply_preconditioner, relative_tolerance=rtol,
                absolute_tolerance=absolute_tolerance, max_iterations=max_iterations)
            for name in totals:
                totals[name] += getattr(result, name)
            totals['reduction_count'] += 1
            record = dict(alpha=alpha, linear_converged=bool(result.converged),
                linear_iterations=int(result.iterations), linear_initial_true_residual=float(result.residual_history[0]),
                linear_final_true_residual=float(result.final_residual), requested_relative_tolerance=rtol,
                requested_absolute_tolerance=absolute_tolerance)
            history.append(record)
            if not result.converged:
                break
            vectors.change_sq.fill_(wp.float64(0.))
            vectors.change_max.fill_(wp.float64(0.))
            vectors.finite_flag.fill_(wp.int32(0))
            wp.launch(kernel=_k.masked_candidate_kernel, dim=shape,
                inputs=[operator.head_device, workspace.solution, operator.active_device,
                    operator.dirichlet_mask_device, operator.dirichlet_values_device, 1., vectors.candidate,
                    vectors.change_sq, vectors.change_max, vectors.finite_flag, nx, ny], device=operator.device)
            if int(vectors.finite_flag.numpy()[0]) != 0:
                record['nonfinite_candidate'] = True
                break
            record['branch_changes'] = operator.gated_branch_changes_device(first=branch,
                second=vectors.candidate, count=vectors.finite_flag)
            if record['branch_changes'] == 0:
                settled = True
                break
            wp.copy(branch, vectors.candidate)
        if settled:
            dh_max = float(vectors.change_max.numpy()[0])
            dh_sq = float(vectors.change_sq.numpy()[0])
            dh_rms = float(np.sqrt(max(dh_sq, 0.)/operator.n_free)) if operator.n_free else 0.
            if np.isfinite(dh_max) and dh_max <= max_head_change:
                operator.residual_device(vectors.candidate, out=vectors.candidate_residual, reduce=True)
                candidate_norms = operator.current_reduced_norms()
                if np.isfinite(candidate_norms.rms) and candidate_norms.rms <= max(
                        0., (1.-armijo*alpha)*residual_norms.rms):
                    accepted = True
                    break
        alpha *= .5
        backtracks += 1
    diagnostic = dict(accepted=accepted, model_history=history, model_work=len(history),
        accepted_step_length=alpha if accepted else 0.,
        residual_rms_after=None if candidate_norms is None else candidate_norms.rms)
    if result is None:
        from .fgmres import FGMRESResult
        result = FGMRESResult(converged=False, iterations=0, restarts=0, final_residual=float('inf'),
            breakdown=False, breakdown_reason='no_model_work', residual_history=[], reduction_count=0)
    return diagnostic, replace(result, **totals), (accepted, alpha, backtracks, candidate_norms, dh_rms, dh_max)


def _confined_startup(*, operator, preconditioner, vectors, workspace,
                      relative_tolerance, absolute_tolerance, max_iterations,
                      max_outer_iterations, residual_tolerance, commit=True):
    """Solve the fixed-T gated problem before handing off to unconfined Newton.

    DRNs/RIVs remain native, outflow-gated terms, not fixed heads. A single
    frozen-active-set solve is insufficient: it can activate many previously
    inactive drains. Preparation uses the configured outer work limit and
    exact linear controls; only a converged fixed-T state whose actual
    unconfined residual decreases can replace the initial iterate.
    """
    shape = operator.ctx.shape
    nx, ny = operator.ctx.nx, operator.ctx.ny
    before = operator.current_reduced_norms().rms
    fixed_t = wp.clone(operator._pic_T_wp)
    trial = wp.clone(operator.head_device)
    boundaries = operator.ctx.boundaries
    other_anchor = bool(np.any(boundaries.dirichlet_mask) or np.any(boundaries.ghb_factor))
    if boundaries.coastal_boundary is not None:
        other_anchor |= bool(np.any(boundaries.coastal_boundary.submerged))
    # With a genuine sea/GHB/fixed-head anchor, the initial drain-inactive
    # selection avoids artificially anchoring every H=DEM cell during this
    # confined preparation. A closed drain-only domain needs the active-side
    # selection at equality to admit recharged startup instead of a singular
    # pure-Neumann linear system. Both are valid generalized derivatives.
    drain_equality_active = not other_anchor
    krylov = RestartedFGMRES2D(workspace=workspace, active=operator.active_device,
        prescribed=operator.dirichlet_mask_device, nx=nx, ny=ny, device=operator.device)

    def apply_frozen(vector, out):
        return operator.frozen_picard_vector_device(vector=vector, out=out)

    def apply_preconditioner(rhs, out):
        return preconditioner.apply(rhs=rhs, out=out)

    totals = dict(iterations=0, restarts=0, reduction_count=0)
    history = []
    result = None
    fixed_converged = False
    status = 'confined_active_set_iteration_limit'
    for iteration in range(max_outer_iterations):
        t, diagonal = operator.freeze_confined_device(transmissivity=fixed_t, head=trial,
            drain_equality_active=drain_equality_active)
        preconditioner.freeze(transmissivity=t, storage_diagonal=diagonal)
        operator.confined_residual_device(head=trial, out=vectors.candidate_residual,
            drain_equality_active=drain_equality_active)
        fixed_before = operator.current_reduced_norms().rms
        if fixed_before <= residual_tolerance:
            fixed_converged = True
            break
        wp.launch(kernel=_k.masked_copy_kernel, dim=shape,
            inputs=[vectors.candidate_residual, operator.active_device,
                operator.dirichlet_mask_device, workspace.rhs, -1., nx, ny], device=operator.device)
        result = krylov.solve(rhs=workspace.rhs, apply_jacobian=apply_frozen,
            apply_preconditioner=apply_preconditioner, relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance, max_iterations=max_iterations)
        for name in totals:
            totals[name] += getattr(result, name)
        record = dict(iteration=iteration+1, residual_rms_before=fixed_before,
            linear_converged=bool(result.converged), linear_iterations=int(result.iterations),
            linear_initial_true_residual=float(result.residual_history[0]),
            linear_final_true_residual=float(result.final_residual),
            requested_relative_tolerance=relative_tolerance,
            requested_absolute_tolerance=absolute_tolerance)
        history.append(record)
        if not result.converged:
            status = 'linear_nonconvergence'
            break
        vectors.change_sq.fill_(wp.float64(0.))
        vectors.change_max.fill_(wp.float64(0.))
        vectors.finite_flag.fill_(wp.int32(0))
        wp.launch(kernel=_k.masked_candidate_kernel, dim=shape,
            inputs=[trial, workspace.solution, operator.active_device,
                operator.dirichlet_mask_device, operator.dirichlet_values_device, 1.,
                vectors.candidate, vectors.change_sq, vectors.change_max, vectors.finite_flag,
                nx, ny], device=operator.device)
        record['head_change_max_m'] = float(vectors.change_max.numpy()[0])
        if int(vectors.finite_flag.numpy()[0]) != 0:
            status = 'nonfinite_candidate'
            break
        wp.copy(trial, vectors.candidate)
        operator.freeze_confined_device(transmissivity=fixed_t, head=trial,
            drain_equality_active=drain_equality_active)
        operator.confined_residual_device(head=trial, out=vectors.candidate_residual,
            drain_equality_active=drain_equality_active)
        record['residual_rms_after'] = operator.current_reduced_norms().rms
        if record['residual_rms_after'] <= residual_tolerance:
            fixed_converged = True
            break

    diagnostic = dict(status=status, accepted=False,
        nonlinear_residual_before=before, nonlinear_residual_after=None,
        linear_converged=bool(all(item['linear_converged'] for item in history)),
        linear_iterations=totals['iterations'], confined_active_set_iterations=len(history),
        confined_converged=fixed_converged, confined_history=history,
        confined_residual_tolerance=residual_tolerance,
        confined_max_outer_iterations=max_outer_iterations,
        drain_equality_active=drain_equality_active,
        linear_initial_true_residual=None if result is None else float(result.residual_history[0]),
        linear_final_true_residual=None if result is None else float(result.final_residual),
        requested_relative_tolerance=relative_tolerance, requested_absolute_tolerance=absolute_tolerance,
        boundaries='native coastal, DRN/RIV and other physical terms retained')
    if fixed_converged:
        operator.residual_device(trial, out=vectors.candidate_residual, reduce=True)
        after = operator.current_reduced_norms().rms
        diagnostic.update(status='non_descent_candidate', nonlinear_residual_after=after)
        if not commit:
            diagnostic.update(status='gated_picard_candidate', accepted=False)
        elif np.isfinite(after) and after < before:
            operator.set_head_device(trial)
            wp.copy(operator.residual_device_array, vectors.candidate_residual)
            diagnostic.update(status='accepted_initial_iterate', accepted=True)
    # Refresh residual reductions for the retained state even when a candidate
    # was rejected; trial evaluation never changes the accepted initial head.
    operator.residual_device(operator.head_device, reduce=True)
    # Only counts are consumed by the caller. Keep the final linear solve's
    # convergence record distinct from total preparation work.
    if result is None:
        from .fgmres import FGMRESResult
        result = FGMRESResult(converged=True, iterations=0, restarts=0,
            final_residual=0., breakdown=False, breakdown_reason=None,
            residual_history=[0.], reduction_count=0)
    return diagnostic, replace(result, **totals), trial


def conductivity_from_reference_transmissivity(*, transmissivity, bottom, top, active):
    """Convert explicit full-column reference T to the native unconfined law.

    T(h) = T_reference * flow_sat(h) / (top-bottom). This conversion is
    internal API plumbing, not an assumption of K=10 across a deep domain.
    """
    reference = np.asarray(transmissivity, dtype=np.float64)
    base, surface = np.asarray(bottom, dtype=np.float64), np.asarray(top, dtype=np.float64)
    mask = np.asarray(active, dtype=bool)
    if (reference.shape != mask.shape or base.shape != mask.shape or surface.shape != mask.shape
            or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(base))
            or not np.all(np.isfinite(surface)) or np.any(reference[mask] <= 0.)
            or np.any(surface[mask] <= base[mask])):
        raise ValueError('reference T requires finite compatible fields, positive active T and positive column height')
    return np.divide(reference, surface-base, out=np.zeros_like(reference), where=mask)


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
    # The fallback must satisfy the same acceptance bar the caller applies to
    # a converged solve (per-timestep budget closure); the monolith's default
    # hclose=1e-4 is too loose in residual terms on stiff transient steps, so
    # tighten it unless the caller already supplied Picard controls.
    fallback_kwargs.setdefault("hclose", 1.0e-6)
    fallback_kwargs.setdefault("max_outer_iterations", 60)
    head, picard_info = backend.solve(context, **fallback_kwargs)
    merged = dict(picard_info)
    merged.update(
        {
            "newton_fallback_used": True,
            "newton_failure_reason": newton_info.get("newton_failure_reason"),
            "newton_diagnostics_before_fallback": newton_info,
            "fallback_backend": "unconfined_picard_kcycle",
            "fallback_state": "picard_completed",
            "experimental_backend": False,
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
    reference_transmissivity = kwargs.get('reference_transmissivity_field')
    if reference_transmissivity is not None:
        if K_field is not None or zbot_field is None or ztop_field is None:
            raise ValueError('reference T input requires top and bottom and must not also supply K_field')
        K_field = conductivity_from_reference_transmissivity(transmissivity=reference_transmissivity,
            bottom=zbot_field, top=ztop_field, active=model.active_host)
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
    startup_mode = str(kwargs.get('unconfined_startup_mode', 'initial_head')).strip().lower()
    if startup_mode not in ('initial_head', 'confined_pre_solve'):
        raise ValueError('semismooth Newton supports initial_head or confined_pre_solve startup')

    restart = int(kwargs.pop("newton_fgmres_restart", 20))
    max_krylov = int(kwargs.pop("newton_fgmres_max_iterations", max(40, 3 * restart)))
    krylov_rtol_explicit = "newton_fgmres_relative_tolerance" in kwargs
    krylov_rtol = float(kwargs.pop("newton_fgmres_relative_tolerance", 1.0e-5))
    krylov_atol = float(kwargs.pop("newton_fgmres_absolute_tolerance", 1.0e-10))
    # Inexact-Newton forcing sequence (Eisenstat-Walker style): unless the
    # inner tolerance is fixed explicitly, loosen it while the outer residual
    # is large and tighten as it contracts.  Early Newton steps stop paying
    # full-Krylov price for accuracy the outer iteration discards, and
    # borderline first-attempt inner solves stop "failing" into fallback.
    forcing_enabled = bool(kwargs.pop("newton_inexact_forcing_enabled", not krylov_rtol_explicit))
    forcing_eta_initial = float(kwargs.pop("newton_forcing_eta_initial", 0.1))
    forcing_eta_min = float(kwargs.pop("newton_forcing_eta_min", 1.0e-7))
    forcing_eta_max = float(kwargs.pop("newton_forcing_eta_max", 0.5))
    forcing_eta_power = float(kwargs.pop("newton_forcing_eta_power", 1.0))
    # Eisenstat-Walker safeguard: when the previous accepted step needed more
    # than this many line-search backtracks, the inexact direction was too
    # crude — halve the next inner tolerance (floored at eta_min).
    forcing_safeguard_backtracks = int(kwargs.pop("newton_forcing_safeguard_backtracks", 3))
    # Final-step tightening: once the outer residual is within this multiple
    # of the outer tolerance, solve the inner system to the eta floor so the
    # accepted result keeps full final accuracy.
    forcing_final_tighten_ratio = float(kwargs.pop("newton_forcing_final_tighten_ratio", 100.0))
    log_progress = bool(kwargs.pop("newton_log_progress", False))
    debug_observer = kwargs.pop('newton_debug_observer', None)
    gated_model_enabled = bool(kwargs.pop('newton_gated_model_enabled', True))
    profile_timing = bool(kwargs.pop('newton_profile_timing', False))
    if debug_observer is not None and not callable(debug_observer):
        raise ValueError('Newton debug observer must be callable')
    max_newton = int(kwargs.pop("newton_max_iterations", 20))
    picard_warm_steps = int(kwargs.pop("newton_picard_warm_start_iterations", 0))
    if picard_warm_steps < 0 or (transient and picard_warm_steps):
        raise ValueError('Picard warm start is optional steady preparation only')
    residual_tol = float(kwargs.pop("newton_residual_rms_tolerance", 1.0e-6))
    head_residual_tol = float(kwargs.pop("newton_head_equivalent_rms_tolerance", kwargs.get("hclose", 1.0e-4) or 1.0e-4))
    max_backtracks = int(kwargs.pop("newton_max_backtracks", 10))
    min_step = float(kwargs.pop("newton_min_step_length", 2.0 ** -10))
    armijo = float(kwargs.pop("newton_armijo_coefficient", 1.0e-4))
    max_head_change = float(kwargs.pop("newton_max_head_change", kwargs.get("max_head_change_per_outer_iteration", 5.0)))
    fallback_enabled = bool(kwargs.pop("newton_fallback_to_picard", True))
    # The Picard fallback cannot honour gated DRN/RIV boundaries (it would
    # silently drop the terms), so refuse it when such cells are present.
    _drn_mask = getattr(model, "drn_mask_host", None)
    _riv_mask = getattr(model, "riv_mask_host", None)
    if getattr(model, "coastal_boundary", None) is not None:
        fallback_enabled = False
    if fallback_enabled and (
        (_drn_mask is not None and bool(np.any(np.asarray(_drn_mask) != 0)))
        or (_riv_mask is not None and bool(np.any(np.asarray(_riv_mask) != 0)))
    ):
        fallback_enabled = False
    coarse_restriction = str(kwargs.pop("newton_coarse_residual_restriction", "average"))
    if coarse_restriction not in ("average", "conservative_sum"):
        raise ValueError("Newton coarse restriction must be average or conservative_sum")
    preconditioner_kind = str(kwargs.pop("newton_preconditioner", "kcycle"))
    if preconditioner_kind not in ("kcycle", "steady_cpu_amg", "steady_cpu_jacobian_amg", "steady_cpu_ilu", "steady_cpu_sparse_lu"):
        raise ValueError("unknown Newton preconditioner")
    if transient and preconditioner_kind != "kcycle":
        raise ValueError("Sparse CPU preconditioning is a steady-preparation backend, not a transient solver")
    preconditioner_cycles = int(kwargs.pop("newton_preconditioner_kcycles", 2))
    max_levels = int(kwargs.get("max_levels", 5))
    min_coarse_cells = kwargs.get("min_coarse_cells", 500)

    if restart < 2 or max_krylov < 1 or max_newton < 1:
        raise ValueError("Newton iteration limits and FGMRES restart must be positive (restart >= 2).")
    if not (0.0 < forcing_eta_min <= forcing_eta_initial <= forcing_eta_max < 1.0):
        raise ValueError(
            "newton forcing etas must satisfy 0 < eta_min <= eta_initial <= eta_max < 1."
        )
    if forcing_eta_power <= 0.0:
        raise ValueError("newton_forcing_eta_power must be positive.")
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
    contact_mask = np.asarray(kwargs.pop("fixed_contact_mask", np.zeros(shape)), dtype=bool)
    contact_head = np.asarray(kwargs.pop("fixed_contact_head", np.zeros(shape)), dtype=np.float64)
    contact_flux = np.asarray(kwargs.pop("fixed_contact_flux", np.zeros(shape)), dtype=np.float64)
    if contact_flux.shape != shape or not np.all(np.isfinite(contact_flux)) or np.any(contact_flux[prescribed | ~active] != 0):
        raise ValueError("contact flux must be finite and zero on CHD/inactive cells")
    if contact_mask.shape != shape or contact_head.shape != shape or not np.all(np.isfinite(contact_head)):
        raise ValueError("contact mask/head must be finite matching grids")
    if transient and np.any(contact_mask) and head_prev is None:
        raise ValueError("transient contact requires an explicit accepted previous head")
    if np.any(contact_mask & (~active | prescribed)):
        raise ValueError("contact cannot overlap inactive or prescribed-head cells")
    prescribed = prescribed | contact_mask
    prescribed_values = np.where(contact_mask, contact_head, prescribed_values)
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

    physical_context = nonlinear_context
    if np.any(contact_flux):
        nonlinear_context = replace(nonlinear_context, sources=replace(
            nonlinear_context.sources, R_field=np.asarray(model.R_field_host)-contact_flux/(float(model.dx)**2)))
    if np.any(contact_mask):
        nonlinear_context = replace(nonlinear_context, boundaries=replace(
            nonlinear_context.boundaries, dirichlet_mask=prescribed.astype(np.int32),
            dirichlet_values=prescribed_values))

    # Hierarchy topology and its work buffers are model-owned and built at most
    # once here.  Frozen coefficients remain in a separate experimental cache.
    if model.mg_levels is None or bool(model._operator_dirty):
        model.build_hierarchy(
            max_levels=max_levels,
            min_coarse_n=4,
            min_coarse_cells=min_coarse_cells,
        )
    gate_counts = wp.zeros(6, dtype=wp.int32, device=model.device_str) if profile_timing else None
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
        coarse_residual_restriction=coarse_restriction,
        implementation=str(kwargs.get('inner_implementation', 'classic')),
        profile_timing=profile_timing,
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
        source_rate=np.asarray(nonlinear_context.sources.R_field, dtype=np.float64),
        sy=sy,
        ss=ss,
    )
    operator = operator_workspace.operator
    if preconditioner_kind in ("steady_cpu_amg", "steady_cpu_jacobian_amg"):
        from .steady_amg_preconditioner import SteadyAMGPreconditioner2D
        preconditioner = SteadyAMGPreconditioner2D(operator=operator, model=model,
            exact_jacobian=preconditioner_kind == 'steady_cpu_jacobian_amg')
    elif preconditioner_kind in ("steady_cpu_ilu", "steady_cpu_sparse_lu"):
        from .steady_sparse_preconditioner import SteadySparsePreconditioner2D
        preconditioner = SteadySparsePreconditioner2D(operator=operator, model=model,
            exact=preconditioner_kind == "steady_cpu_sparse_lu")
    history: list[dict[str, Any]] = []
    linear_history = []
    failure_reason: str | None = None
    converged = False
    total_fgmres_iterations = 0
    total_fgmres_restarts = 0
    total_reductions = 0
    total_backtracks = 0
    accepted_updates = 0
    gated_active_set_work = 0
    startup_diagnostics = dict(status='not_requested', accepted=False)
    startup_wall_s = 0.0
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

        if not transient and startup_mode == 'confined_pre_solve' and norms.rms > residual_tol:
            startup_started = time.perf_counter()
            startup_diagnostics, startup_linear, _ = _confined_startup(operator=operator,
                preconditioner=preconditioner, vectors=vector_workspace, workspace=fgmres_workspace,
                relative_tolerance=float(kwargs.get('rel_tol', krylov_rtol)),
                absolute_tolerance=float(kwargs.get('abs_tol_min', krylov_atol)),
                max_iterations=max_krylov,
                max_outer_iterations=int(kwargs.get('max_outer_iterations', max_newton)),
                residual_tolerance=residual_tol)
            startup_wall_s = time.perf_counter() - startup_started
            total_fgmres_iterations += startup_linear.iterations
            total_fgmres_restarts += startup_linear.restarts
            total_reductions += startup_linear.reduction_count
            norms = operator.current_reduced_norms()
            T_frozen, S_frozen = operator.freeze_picard_device(operator.head_device)
            preconditioner.freeze(transmissivity=T_frozen, storage_diagonal=S_frozen)
            head_equivalent = _head_equivalent_rms(residual=operator.residual_device_array,
                diagonal_inverse=preconditioner.fine_diagonal_inverse, operator=operator, vectors=vector_workspace)

        if norms.rms <= residual_tol and head_equivalent <= head_residual_tol:
            converged = True

        previous_backtracks = 0
        warm_start_active = picard_warm_steps > 0
        for newton_iteration in range(max_newton):
            if converged:
                break
            if newton_iteration + gated_active_set_work >= max_newton:
                failure_reason = 'newton_iteration_limit'
                break
            iteration_started = time.perf_counter()
            applications_before = preconditioner.applications
            cycles_before = preconditioner.cycles
            if forcing_enabled:
                ratio = norms.rms / max(float(initial_residual), 1.0e-300)
                iteration_rtol = min(
                    forcing_eta_max,
                    max(forcing_eta_min, forcing_eta_initial * ratio ** forcing_eta_power),
                )
                if previous_backtracks > forcing_safeguard_backtracks and iteration_rtol > forcing_eta_min:
                    iteration_rtol = max(forcing_eta_min, 0.5 * iteration_rtol)
                if norms.rms <= forcing_final_tighten_ratio * residual_tol:
                    iteration_rtol = forcing_eta_min
            else:
                iteration_rtol = krylov_rtol
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
            direction_kind = 'frozen_picard' if warm_start_active and newton_iteration < picard_warm_steps else 'analytical_newton'

            def apply_direction(vector, out):
                if direction_kind == 'frozen_picard':
                    return operator.frozen_picard_vector_device(vector=vector, out=out)
                return operator.jacobian_vector_device(head=operator.head_device, vector=vector, out=out)

            def apply_preconditioning(rhs, out):
                return preconditioner.apply(rhs=rhs, out=out)

            result = krylov.solve(
                rhs=fgmres_workspace.rhs,
                apply_jacobian=apply_direction,
                apply_preconditioner=apply_preconditioning,
                relative_tolerance=iteration_rtol,
                absolute_tolerance=krylov_atol,
                max_iterations=max_krylov,
            )
            total_fgmres_iterations += result.iterations
            total_fgmres_restarts += result.restarts
            total_reductions += result.reduction_count
            linear_history.append(dict(newton_iteration=newton_iteration+1,
                linear_operator=direction_kind,
                converged=bool(result.converged), iterations=int(result.iterations),
                initial_true_residual=float(result.residual_history[0]),
                final_true_residual=float(result.final_residual),
                requested_relative_tolerance=float(iteration_rtol),
                requested_absolute_tolerance=float(krylov_atol),
                breakdown_reason=result.breakdown_reason, krylov_residual_history=result.residual_history))
            if log_progress:
                print(f"Darcy Newton {newton_iteration+1}: nonlinear RMS={norms.rms:.6g}; "
                    f"FGMRES {result.iterations}, true relative residual="
                    f"{result.final_residual/max(result.residual_history[0], np.finfo(float).tiny):.6g}; "
                    f"linear accepted={result.converged}", flush=True)
            if not result.converged:
                failure_reason = result.breakdown_reason or "fgmres_nonconvergence"
                break

            if debug_observer is not None:
                # Explicit diagnostics only. Read-only copies ensure a callback
                # cannot alter solver-owned iterates or Krylov workspaces.
                observed = dict(head=operator.head_device.numpy().copy(),
                    correction=fgmres_workspace.solution.numpy().copy(),
                    residual=operator.residual_device_array.numpy().copy())
                for array in observed.values():
                    array.setflags(write=False)
                debug_observer(newton_iteration=newton_iteration+1, **observed)

            accepted, alpha, backtracks, candidate_norms, dh_rms, dh_max = _backtrack_newton_direction(
                operator=operator, vectors=vector_workspace, correction=fgmres_workspace.solution,
                residual_rms=norms.rms, max_backtracks=max_backtracks, min_step=min_step,
                max_head_change=max_head_change, armijo=armijo)
            gated_proposal = None
            iteration_linear_work = result.iterations
            slow_switching_step = False
            if accepted and backtracks > forcing_safeguard_backtracks and candidate_norms is not None:
                expected = (1.-alpha+iteration_rtol*alpha)*norms.rms
                if candidate_norms.rms > expected:
                    slow_switching_step = operator.gated_branch_changes_device(first=operator.head_device,
                        second=vector_workspace.candidate, count=vector_workspace.finite_flag) > 0
            if (gated_model_enabled and (not accepted or slow_switching_step)
                    and direction_kind == 'analytical_newton' and not transient
                    and preconditioner_kind == 'kcycle' and (operator._has_gated or operator._coast is not None)
                    and newton_iteration + gated_active_set_work + 1 < max_newton):
                # Retain an already acceptable trial if the exact-gate Newton
                # model cannot improve it. Only device copies are needed.
                previous_trial = (accepted, alpha, backtracks, candidate_norms, dh_rms, dh_max)
                previous_candidate = wp.clone(vector_workspace.candidate) if accepted else None
                previous_residual = wp.clone(vector_workspace.candidate_residual) if accepted else None
                operator.residual_device(operator.head_device, reduce=True)
                gated_proposal, proposal_linear, model_trial = _gated_newton_model_step(
                    operator=operator, preconditioner=preconditioner, vectors=vector_workspace,
                    workspace=fgmres_workspace, residual_norms=norms, relative_tolerance=iteration_rtol,
                    absolute_tolerance=krylov_atol, max_iterations=max_krylov,
                    max_model_work=max_newton-newton_iteration-gated_active_set_work-1,
                    max_backtracks=max_backtracks, min_step=min_step,
                    max_head_change=max_head_change, armijo=armijo)
                gated_active_set_work += gated_proposal['model_work']
                total_fgmres_iterations += proposal_linear.iterations
                total_fgmres_restarts += proposal_linear.restarts
                total_reductions += proposal_linear.reduction_count
                iteration_linear_work += proposal_linear.iterations
                for linear in gated_proposal['model_history']:
                    linear_history.append(dict(newton_iteration=newton_iteration+1,
                        linear_operator='gated_newton_model', converged=linear['linear_converged'],
                        iterations=linear['linear_iterations'], initial_true_residual=linear['linear_initial_true_residual'],
                        final_true_residual=linear['linear_final_true_residual'],
                        requested_relative_tolerance=linear['requested_relative_tolerance'],
                        requested_absolute_tolerance=linear['requested_absolute_tolerance'], breakdown_reason=None))
                if model_trial[0] and (not accepted or model_trial[3].rms < candidate_norms.rms):
                    accepted, alpha, backtracks, candidate_norms, dh_rms, dh_max = model_trial
                    backtracks += previous_trial[2]
                    direction_kind = 'gated_newton_model'
                    result = proposal_linear
                else:
                    accepted, alpha, backtracks, candidate_norms, dh_rms, dh_max = previous_trial
                    backtracks += model_trial[2]
                    if accepted:
                        wp.copy(vector_workspace.candidate, previous_candidate)
                        wp.copy(vector_workspace.candidate_residual, previous_residual)

            gate_diagnostics = {}
            if profile_timing and candidate_norms is not None:
                gate_diagnostics = operator.gated_activation_diagnostics_device(
                    first=operator.head_device, second=vector_workspace.candidate, counts=gate_counts)
                gate_diagnostics['candidate_accepted'] = bool(accepted)
            total_backtracks += backtracks
            history.append(
                {
                    "newton_iteration": int(newton_iteration + 1),
                    "direction_kind": direction_kind,
                    "residual_rms_before": float(norms.rms),
                    "residual_rms_after": None if candidate_norms is None else float(candidate_norms.rms),
                    "accepted_step_length": float(alpha) if accepted else 0.0,
                    "backtrack_count": int(backtracks),
                    "head_change_rms": float(dh_rms),
                    "head_change_max": float(dh_max),
                    "fgmres_iterations": int(iteration_linear_work),
                    "fgmres_restarts": int(result.restarts),
                    "fgmres_final_residual": float(result.final_residual),
                    "fgmres_relative_tolerance": float(iteration_rtol),
                    "fgmres_breakdown": bool(result.breakdown),
                    "fgmres_breakdown_reason": result.breakdown_reason,
                    "gated_proposal": gated_proposal,
                    "gate_diagnostics": gate_diagnostics,
                    "kcycle_applications": int(preconditioner.applications - applications_before),
                    "kcycles": int(preconditioner.cycles - cycles_before),
                    "iteration_wall_s": time.perf_counter() - iteration_started,
                }
            )
            if log_progress:
                print(f"Darcy Newton candidate: accepted={accepted}, scale={alpha:.6g}, "
                    f"backtracks={backtracks}; RMS={None if candidate_norms is None else candidate_norms.rms}", flush=True)
            if not accepted or candidate_norms is None:
                if direction_kind == 'frozen_picard':
                    # A Picard direction need not be a descent direction for
                    # the nonlinear norm. End preparation without committing
                    # its failed candidate, then try analytical Newton from
                    # the last accepted preparation iterate.
                    warm_start_active = False
                    continue
                failure_reason = "line_search_failed"
                break

            previous_backtracks = backtracks
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
            history[-1]["iteration_wall_s"] = time.perf_counter() - iteration_started
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
        # Evaluate reaction using the SAME FV residual, with contact rows
        # unconstrained. These cells retain physical storage, unlike CHD.
        imposed_contact = contact_flux.copy()
        reaction_mask = contact_mask | (imposed_contact != 0)
        contact_flux = np.zeros(shape, dtype=np.float64)
        physical_operator = operator
        if np.any(reaction_mask):
            physical_operator = NonlinearOperator2D(physical_context)
            try:
                physical_residual = physical_operator.residual(head).numpy()
                contact_flux[reaction_mask] = -physical_residual[reaction_mask]
                storage_terms = physical_operator.exact_storage_terms(head) if transient else None
            finally:
                physical_operator.close()
        else:
            storage_terms = operator.exact_storage_terms(head) if transient else None
        saturation = np.asarray(operator.saturated_thickness(head).numpy(), dtype=np.float64).copy()
        transmissivity = np.asarray(K_field, dtype=np.float64) * saturation
        transmissivity[~active] = 0.0
        from DARCY_WARP_PACKAGE.physics.budgets_2d import (
            add_exact_storage_to_budget,
            compute_mass_balance_budget,
        )
        from DARCY_WARP_PACKAGE.physics.drn_2d import add_drn_to_budget, drn_discharge_2d
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
            drn_mask=np.asarray(model.drn_mask_host, dtype=np.int32) if model.use_drn else None,
            drn_elev=np.asarray(model.drn_elev_host, dtype=np.float64) if model.use_drn else None,
            drn_cond=np.asarray(model.drn_cond_host, dtype=np.float64) if model.use_drn else None,
            riv_mask=np.asarray(model.riv_mask_host, dtype=np.int32) if model.use_riv else None,
            riv_stage=np.asarray(model.riv_stage_host, dtype=np.float64) if model.use_riv else None,
            riv_rbot=np.asarray(model.riv_rbot_host, dtype=np.float64) if model.use_riv else None,
            riv_cond=np.asarray(model.riv_cond_host, dtype=np.float64) if model.use_riv else None,
            coastal_boundary=getattr(model, "coastal_boundary", None),
            case="unconfined_semismooth_newton_kcycle",
        )
        accepted_drn_flux = np.zeros((ny, nx), dtype=np.float64)
        if model.use_drn:
            accepted_drn_flux = drn_discharge_2d(
                head=head,
                elevation=np.asarray(model.drn_elev_host, dtype=np.float64),
                conductance=np.asarray(model.drn_cond_host, dtype=np.float64),
                mask=np.asarray(model.drn_mask_host, dtype=np.int32),
            )
            budget = add_drn_to_budget(budget, accepted_drn_flux)
        from DARCY_WARP_PACKAGE.physics.riv_2d import riv_discharge_2d
        accepted_riv_flux = np.zeros((ny, nx), dtype=np.float64)
        if model.use_riv:
            accepted_riv_flux = riv_discharge_2d(
                head=head, stage=model.riv_stage_host, bottom=model.riv_rbot_host,
                conductance=model.riv_cond_host, mask=model.riv_mask_host,
                device=model.device_str)
        if storage_terms is not None:
            budget = add_exact_storage_to_budget(budget, storage_terms.total)
        if np.any(reaction_mask):
            contact_in = float(np.maximum(-contact_flux, 0).sum())
            contact_out = float(np.maximum(contact_flux, 0).sum())
            budget["contact_in"] = contact_in
            budget["contact_out"] = contact_out
            # Keep native aggregate budget consistent with the new boundary.
            budget["total_in"] += contact_in
            budget["total_out"] += contact_out
            total_in = float(budget.iloc[0]["total_in"])
            total_out = float(budget.iloc[0]["total_out"])
            imbalance = total_in - total_out
            budget["in_minus_out"] = imbalance
            budget["percent_discrepancy"] = 100*imbalance/(total_in+total_out) if total_in+total_out else 0.
            budget["throughflow"] = .5*(total_in+total_out)
            budget["imbalance_fraction"] = imbalance/(.5*(total_in+total_out)) if total_in+total_out else 0.
        from .steady_config import steady_source_identity
        effective = dict(
            transient=transient, precision=WP_FLOAT.__name__, mixed_precision=False,
            device=str(model.device_str), device_name=wp.get_device(model.device_str).name,
            newton_max_iterations=max_newton, shared_newton_gated_work_budget=True,
            nonlinear_residual_rms_tolerance=residual_tol,
            head_equivalent_rms_tolerance=head_residual_tol,
            fgmres_restart=restart, fgmres_max_iterations=max_krylov,
            fgmres_fixed_relative_tolerance=None if forcing_enabled else krylov_rtol,
            fgmres_absolute_tolerance=krylov_atol,
            forcing=dict(enabled=forcing_enabled, eta_initial=forcing_eta_initial,
                eta_min=forcing_eta_min, eta_max=forcing_eta_max, power=forcing_eta_power,
                safeguard_backtracks=forcing_safeguard_backtracks,
                final_tighten_ratio=forcing_final_tighten_ratio),
            preconditioner_kind=preconditioner_kind,
            preconditioner_implementation=getattr(preconditioner, "implementation", preconditioner_kind),
            preconditioner_controls=dict(getattr(preconditioner, "controls", {})),
            requested_max_levels=max_levels, min_coarse_cells=min_coarse_cells,
            actual_level_shapes=[(int(level.ny), int(level.nx)) for level in levels],
            startup=dict(mode=startup_mode, max_active_set_iterations=int(kwargs.get("max_outer_iterations", max_newton)),
                relative_tolerance=float(kwargs.get("rel_tol", krylov_rtol)),
                absolute_tolerance=float(kwargs.get("abs_tol_min", krylov_atol)),
                accepted=bool(startup_diagnostics.get("accepted", False))),
            gated_model_enabled=gated_model_enabled, gated_model_work=int(gated_active_set_work),
            line_search=dict(max_backtracks=max_backtracks, min_step_length=min_step,
                armijo_coefficient=armijo, max_head_change=max_head_change),
            min_flow_saturated_thickness=min_sat, picard_fallback_enabled=fallback_enabled,
            ignored_supplied_controls=sorted(key for key in original_kwargs if key in {
                "max_cycles", "dh_rms_tol", "residual_floor_tol", "inner_forcing_eta",
                "inner_head_residual_tol_min", "inner_head_residual_tol_max",
                "chebyshev_reset_factor", "transmissivity_relaxation_enabled", "omega_min",
                "omega_max", "chebyshev_enabled", "chebyshev_order", "chebyshev_rejection_factor",
                "check_every_no", "adaptive_dt_enabled", "transient_mixed_precision_enabled"}),
            stopping_rule="true residual RMS AND diagonal-scaled head-equivalent RMS; each FGMRES solve checks true residual")
        info = {
            "effective_configuration": effective,
            "source_identity": steady_source_identity(model_file=__import__(
                "DARCY_WARP_PACKAGE.warped_darcy", fromlist=["__file__"]).__file__, solver_file=__file__),
            "kcycle_graph_execution": dict(getattr(preconditioner, "execution_stats", {})),
            "contact_discharge_rate_array": contact_flux,
            "contact_mask_array": contact_mask.copy(),
            "converged": bool(converged),
            "solver_type": "unconfined_semismooth_newton_kcycle",
            "solver_backend": "unconfined_semismooth_newton_kcycle",
            "multigrid_level_shapes": [(int(level.ny), int(level.nx)) for level in levels],
            "experimental_backend": False,
            "flow_coefficient_input": 'full_column_reference_transmissivity' if reference_transmissivity is not None else 'hydraulic_conductivity',
            "reference_transmissivity_field": reference_transmissivity,
            "newton_iterations": int(accepted_updates),
            "gated_corrective_active_set_iterations": int(gated_active_set_work),
            "debug_observer_enabled": debug_observer is not None,
            "gated_newton_model_enabled": gated_model_enabled,
            "unconfined_startup_mode": startup_mode,
            "confined_startup": startup_diagnostics,
            "confined_startup_wall_s": startup_wall_s,
            "picard_warm_start_updates": sum(item['direction_kind'] == 'frozen_picard' and item['accepted_step_length'] > 0 for item in history),
            "outer_picard_iterations": 0,
            "inexact_forcing_enabled": bool(forcing_enabled),
            "outer_history": history,
            "linear_history": linear_history,
            "nonlinear_residual_history": [float(initial_residual)] + [
                float(item["residual_rms_after"]) for item in history if item["residual_rms_after"] is not None
            ],
            "final_residual": float(norms.rms),
            "true_nonlinear_residual_rms": float(norms.rms),
            "head_equivalent_residual_rms": float(head_equivalent),
            "fgmres_iterations": int(total_fgmres_iterations),
            "fgmres_restarts": int(total_fgmres_restarts),
            "fgmres_restart_size": int(restart),
            "coarse_residual_restriction": coarse_restriction,
            "preconditioner_kind": preconditioner_kind,
            "kcycle_graph_cache_entries": len(getattr(preconditioner, "graph_cache", {})),
            "kcycle_graph_capture_fallback_entries": sum(value is False for value in getattr(preconditioner, "graph_cache", {}).values()),
            "preconditioner_implementation": getattr(preconditioner, 'implementation', preconditioner_kind),
            "preconditioner_setup_wall_s": getattr(preconditioner, "setup_wall_s", None),
            "preconditioner_apply_wall_s": getattr(preconditioner, "apply_wall_s", None),
            "preconditioner_full_vector_transfer_bytes": getattr(preconditioner, "full_vector_transfer_bytes", 0),
            "fgmres_reduction_count": int(total_reductions),
            "preconditioner_applications": int(preconditioner.applications),
            "kcycle_preconditioner_applications": int(preconditioner.applications) if preconditioner_kind == "kcycle" else 0,
            "kcycle_preconditioner_cycles": int(preconditioner.cycles) if preconditioner_kind == "kcycle" else 0,
            "backtrack_count": int(total_backtracks),
            "breakdown_state": failure_reason if failure_reason and "breakdown" in failure_reason else None,
            "newton_failure_reason": failure_reason,
            "newton_fallback_used": False,
            "fallback_state": "not_used",
            "generalized_derivative_convention": "one_strictly_inside_zero_outside_and_at_clip_thresholds",
            "drain_generalized_derivative_convention": "active_side_at_equality; zero_strictly_below_elevation",
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
            "drn_discharge_rate_array": accepted_drn_flux,
            "riv_discharge_rate_array": accepted_riv_flux,
            "drn_out_rate": float(accepted_drn_flux.sum()),
            "drn_volume": float(accepted_drn_flux.sum() * float(dt)) if transient else 0.0,
            "dry_mask": np.asarray(head <= (zbot + min_sat), dtype=bool),
            "active_mask": active.copy(),
        }

        if hasattr(preconditioner, 'timing_report'):
            info.update(preconditioner.timing_report())
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
    """Explicit production 2D nonlinear backend."""

    name = "unconfined_semismooth_newton_kcycle"

    def solve(self, context: SolverContext, **kwargs: Any):
        return solve_semismooth_newton(context=context, **kwargs)


__all__ = ["UnconfinedSemismoothNewtonKCycleBackend", "solve_semismooth_newton"]
