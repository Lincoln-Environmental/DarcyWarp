"""Opt-in pinned NPF flow and implicit STO, retaining the legacy session backend."""
import numpy as np
from dataclasses import asdict
from time import perf_counter
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case,NPF662LayeredCase,NPF662Reference
from DARCY_WARP_PACKAGE.validation.npf662_metrics import final_tolerances
from DARCY_WARP_PACKAGE.physics.inputs_3d import positive_integer


def case_from_model(*,model,top,cell_type=0):
    if model.geometry not in ('cartesian','structured_dis_layered') or any(k is None for k in (model._kx_field,model._ky_field,model._kz_field)):
        raise ValueError('mf6_6_6_2_npf requires build_from_K_fields or build_from_layers; arbitrary faces/terrain unsupported')
    s=model._session
    initial = np.zeros(s.shape) if s.initial_head is None else s.initial_head
    if model.geometry=='structured_dis_layered':
        return NPF662LayeredCase(kx=model._kx_field,ky=model._ky_field,kz=model._kz_field,active=s.active,fixed=s.bc_mask,
            values=s.bc_values,source=s.rhs,initial=initial,dx=model.dx,dy=model.dy,
            z_interfaces=s.z_interfaces,cell_type=cell_type)
    return NPF662Case(kx=model._kx_field,ky=model._ky_field,kz=model._kz_field,active=s.active,fixed=s.bc_mask,
        values=s.bc_values,source=s.rhs,initial=initial,
        dx=model.dx,dy=model.dy,dz=model.dz,top=top,cell_type=cell_type)


def independent_reference(*, model, case):
    """Reuse only independently assembled geometry/K operators with equal inputs.

    Forcing, prescribed values and initial heads remain live case inputs.
    Owned snapshots detect in-place mutation as well as public updates.
    """
    case.validate()
    names = tuple(name for name in case.__dataclass_fields__ if name not in ('source', 'values', 'initial'))
    values = tuple(np.asarray(getattr(case, name)) for name in names)
    holder = model._session._npf662_workspace_holder
    cached = holder.get('reference')
    reused = cached is not None and cached[0] == names and all(
        np.array_equal(saved, current) for saved, current in zip(cached[1], values))
    if reused:
        reference = cached[2]
        reference.case = case
    else:
        reference = NPF662Reference(case=case)
        holder['reference'] = (names, tuple(value.copy() for value in values), reference)
    return reference, reused


def solve_npf662(*,model,unconfined=False,top_elevation=None,npf_cell_type=None,max_cycles=2000,rel_tol=None,
    abs_tol_min=1e-12,final_budget_relative_tol=1e-8,max_levels=None,min_coarse_n=None,
    return_info=True,max_outer=None,krylov_restart=None,npf_newton_controls=None,
    npf_preconditioner=None,npf_kcycle_controls=None,npf_krylov_method=None,
    transient=False,dt=None,head_prev=None,ss=None,sy=None,npf_final_relative_tolerance=None,**unsupported):
    started = perf_counter()
    # Omission retains historical backend defaults. Explicit selections must
    # either reach their backend or be rejected, even when equal to a former
    # signature default (e.g. max_levels=6 with a two-level NPF hierarchy).
    legacy_overrides = any(value is not None for value in (rel_tol, max_levels, min_coarse_n))
    iteration_overrides = max_outer is not None or krylov_restart is not None
    preconditioner_selected = npf_preconditioner is not None
    rel_tol = 1e-12 if rel_tol is None else rel_tol
    max_levels = 6 if max_levels is None else max_levels
    min_coarse_n = 2 if min_coarse_n is None else min_coarse_n
    max_outer = 2000 if max_outer is None else max_outer
    krylov_restart = 24 if krylov_restart is None else krylov_restart
    npf_preconditioner = 'diagonal' if npf_preconditioner is None else npf_preconditioner
    if unsupported:raise ValueError(f'Unsupported mf6_6_6_2_npf controls: {sorted(unsupported)}')
    if transient and any(value is None for value in (dt,head_prev,ss,sy)):
        raise ValueError('Unsupported incomplete transient controls: dt, head_prev, ss and sy required')
    if not transient and any(value is not None for value in (dt,head_prev,ss,sy)):
        raise ValueError('Storage inputs require transient=True')
    if npf_preconditioner not in ('diagonal','kcycle'):raise ValueError('npf_preconditioner must be diagonal or kcycle')
    if npf_preconditioner=='diagonal' and npf_kcycle_controls is not None:
        raise ValueError('npf_kcycle_controls requires npf_preconditioner=kcycle')
    if npf_krylov_method not in (None,'gmres','fgmres'):raise ValueError('npf_krylov_method must be gmres or fgmres')
    if npf_preconditioner=='kcycle' and npf_krylov_method=='gmres':raise ValueError('K-cycle requires flexible GMRES')
    max_cycles = positive_integer(value=max_cycles, name='max_cycles', minimum=0)
    max_outer = positive_integer(value=max_outer, name='max_outer', minimum=0)
    krylov_restart = positive_integer(value=krylov_restart, name='krylov_restart', minimum=2)
    max_levels = positive_integer(value=max_levels, name='max_levels')
    min_coarse_n = positive_integer(value=min_coarse_n, name='min_coarse_n')
    if npf_final_relative_tolerance is not None and final_budget_relative_tol != 1e-8:
        raise ValueError('Choose npf_final_relative_tolerance or final_budget_relative_tol, not both')
    if isinstance(npf_final_relative_tolerance, (bool, np.bool_)):
        raise ValueError('npf_final_relative_tolerance must be a numerical scalar, not Boolean')
    relative_tolerance = final_budget_relative_tol if npf_final_relative_tolerance is None else npf_final_relative_tolerance
    thresholds=final_tolerances(physical_tolerance=abs_tol_min,relative_tolerance=relative_tolerance)
    if np.ndim(rel_tol)!=0 or not np.isfinite(rel_tol) or rel_tol<0.:
        raise ValueError('Krylov relative tolerance must be a finite nonnegative scalar')
    layered=model.geometry=='structured_dis_layered'
    if layered and top_elevation is not None:raise ValueError('Layered interfaces are authoritative; top_elevation override is unsupported')
    if unconfined and top_elevation is None and not layered:raise ValueError('Matched unconfined mode requires explicit flat top_elevation')
    types=int(unconfined) if npf_cell_type is None else npf_cell_type
    case=case_from_model(model=model,top=0. if top_elevation is None else top_elevation,cell_type=types)
    if transient:
        from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference
        reference = NPF662TransientReference(case=case, previous=head_prev, dt=dt, ss=ss, sy=sy)
        reference_reused = False
    else:
        reference, reference_reused = independent_reference(model=model, case=case)
    reference_seconds = perf_counter()-started
    reference.check_components()
    if not unconfined and np.any(case.cell_types):raise ValueError('Convertible cell types require unconfined=True')
    if transient or unconfined or layered or preconditioner_selected or npf_krylov_method is not None or npf_newton_controls is not None or npf_final_relative_tolerance is not None:
        if model.implementation!='fast' or model.solver!='kcycle':
            raise ValueError('Matched Newton requires implementation=fast and solver=kcycle; legacy backends remain separate')
        if unconfined and not np.any(case.cell_types[case.active!=0]):raise ValueError('unconfined=True requires convertible active cells')
        if unconfined and (rel_tol!=1e-12 or abs_tol_min!=1e-12 or final_budget_relative_tol!=1e-8 or max_levels!=6 or min_coarse_n!=2):
            raise ValueError('Matched unconfined final acceptance is fixed by V1; npf_newton_controls controls iteration only. Confined tolerance/MG overrides are unsupported')
        if legacy_overrides:
            raise ValueError('NPF Newton/GMRES uses npf_newton_controls.inner_forcing and npf_kcycle_controls.levels/min_coarse_n; rel_tol/max_levels/min_coarse_n overrides are unsupported')
        from DARCY_WARP_PACKAGE.solvers.npf662_newton import solve_newton_device,NPF662NewtonControls
        controls=NPF662NewtonControls() if npf_newton_controls is None else npf_newton_controls
        if not isinstance(controls,NPF662NewtonControls):raise ValueError('npf_newton_controls must be NPF662NewtonControls')
        head,info=solve_newton_device(model=model,case=case,reference=reference,max_outer=max_outer,
            krylov_budget=max_cycles,restart=krylov_restart,controls=controls,
            physical_tolerance=abs_tol_min,relative_tolerance=relative_tolerance,
            npf_preconditioner=npf_preconditioner,npf_kcycle_controls=npf_kcycle_controls,npf_krylov_method=npf_krylov_method)
        info.update(formulation='mf6_6_6_2_npf',geometry=case.geometry,
            device=model.device,precision='float64',
            implementation='device_newton_fgmres' if info['krylov_method']=='FGMRES' else 'device_newton_gmres',session_backend='npf662_newton',
            reference_contract=case.contract+('_Newton' if unconfined else '_confined')+('_STO_v1' if transient else ''),reference_version='6.6.2',
            final_acceptance_thresholds=thresholds,
            effective_options=dict(xt3d=False,newton=unconfined,cell_type=case.cell_types.tolist(),saturation_epsilon=1e-6,
                conductance_floor_m=0.,variablecv=False,perched=False,thickstrt=False,rewet=False,storage=bool(transient),
                packages=['CHD','WEL_SOURCE']+(['STO'] if transient else [])))
        if transient:
            info.update(dt=float(dt),storage_formulation='MF6 6.6.2 default SsTerms/SyTerms; Newton saturation smoothing',
                storage_previous_fixed=True,
                scaling='full-saturated principal diagonal plus previous-state storage tangent; S*J and S*R')
            if npf_preconditioner=='kcycle':
                info['preconditioner']='A0 plus previous-state storage tangent K-cycle; physical RHS D0*v; homogeneous corrections'
        info['effective_solver_settings'] = dict(backend=info['implementation'], krylov_method=info['krylov_method'],
            restart=krylov_restart, krylov_budget_per_outer=max_cycles, max_outer=max_outer,
            nonlinear=asdict(controls), hierarchy=info.get('multigrid'),
            preconditioner=npf_preconditioner, convergence=thresholds)
        if npf_final_relative_tolerance is not None:
            info['acceptance_override'] = dict(relative_cell_imbalance=relative_tolerance,
                default_relative_cell_imbalance=1e-8)
        info.update(reference_reused=reference_reused, reference_setup_seconds=reference_seconds,
                    public_total_seconds=perf_counter()-started)
        return (head,info) if return_info else head
    if iteration_overrides:
        raise ValueError('max_outer/krylov_restart require npf_krylov_method=gmres or fgmres; uniform confined default uses the historical direct K-cycle backend')
    head,info=model._session.solve(solver=model.solver,device=model.device,implementation=model.implementation,
        unconfined=False,max_cycles=max_cycles,rel_tol=rel_tol,abs_tol_min=abs_tol_min,
        final_budget_relative_tol=final_budget_relative_tol,max_levels=max_levels,min_coarse_n=min_coarse_n,
        diag_preconditioner_backend=model.diag_preconditioner_backend,dh_rms_tol=None,return_info=True)
    checks,unused=reference.evaluate(head=head,physical_tolerance=abs_tol_min,relative_tolerance=final_budget_relative_tol)
    info['backend_converged']=bool(info['converged'])
    info['dt']=None  # No timestep exists in this explicitly steady contract.
    info.update(formulation='mf6_6_6_2_npf',geometry='orthogonal_cartesian_uniform_dz',
        device=model.device,precision='float64',
        reference_contract='NPF662_CARTESIAN_V1_confined',reference_version='6.6.2',
        independent_acceptance=checks,converged=bool(checks['independent_accepted']),
        final_acceptance_thresholds=thresholds,
        effective_options=dict(xt3d=False,newton=False,icelltype=0,variablecv=False,perched=False,thickstrt=False,
            rewet=False,storage=False,packages=['CHD','WEL_SOURCE']),
        stop_reason='converged' if checks['independent_accepted'] else 'independent_acceptance_failed')
    info['effective_solver_settings'] = dict(backend=info.get('session_backend'), krylov_method=None,
        hierarchy=dict(max_levels=max_levels, min_coarse_n=min_coarse_n, actual_levels=info.get('n_levels'),
                       dimensions=info.get('level_shapes')),
        max_cycles=max_cycles, rel_tol=rel_tol, nonlinear=None, convergence=thresholds)
    info.update(reference_reused=reference_reused, reference_setup_seconds=reference_seconds,
                public_total_seconds=perf_counter()-started)
    return (head,info) if return_info else head
