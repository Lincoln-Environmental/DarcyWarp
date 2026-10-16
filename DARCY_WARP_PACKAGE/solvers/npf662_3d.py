"""Opt-in pinned Cartesian NPF contract, retaining the legacy session backend."""
import numpy as np
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case,NPF662Reference


def case_from_model(*,model,top,cell_type=0):
    if model.geometry!='cartesian' or any(k is None for k in (model._kx_field,model._ky_field,model._kz_field)):
        raise ValueError('mf6_6_6_2_npf requires Cartesian build_from_K_fields; arbitrary faces/terrain unsupported')
    s=model._session
    return NPF662Case(kx=model._kx_field,ky=model._ky_field,kz=model._kz_field,active=s.active,fixed=s.bc_mask,
        values=s.bc_values,source=s.rhs,initial=s.initial_head if s.initial_head is not None else np.zeros(s.shape),
        dx=model.dx,dy=model.dy,dz=model.dz,top=top,cell_type=cell_type)


def solve_npf662(*,model,unconfined=False,top_elevation=None,npf_cell_type=None,max_cycles=2000,rel_tol=1e-12,
    abs_tol_min=1e-12,final_budget_relative_tol=1e-8,max_levels=6,min_coarse_n=2,
    return_info=True,max_outer=2000,krylov_restart=24,npf_newton_controls=None,**unsupported):
    if unsupported:raise ValueError(f'Unsupported mf6_6_6_2_npf controls: {sorted(unsupported)}')
    if unconfined and top_elevation is None:raise ValueError('Matched unconfined mode requires explicit flat top_elevation')
    types=int(unconfined) if npf_cell_type is None else npf_cell_type
    case=case_from_model(model=model,top=0. if top_elevation is None else top_elevation,cell_type=types)
    reference=NPF662Reference(case=case)
    reference.check_components()
    if not unconfined and np.any(case.cell_types):raise ValueError('Convertible cell types require unconfined=True')
    if unconfined:
        if model.implementation!='fast' or model.solver!='kcycle':
            raise ValueError('Matched Newton requires implementation=fast and solver=kcycle; legacy backends remain separate')
        if not np.any(case.cell_types[case.active!=0]):raise ValueError('unconfined=True requires convertible active cells')
        if rel_tol!=1e-12 or abs_tol_min!=1e-12 or final_budget_relative_tol!=1e-8 or max_levels!=6 or min_coarse_n!=2:
            raise ValueError('Confined-only tolerance/MG controls are unsupported in the matched Newton path; use npf_newton_controls')
        from DARCY_WARP_PACKAGE.solvers.npf662_newton import solve_newton_device,NPF662NewtonControls
        controls=NPF662NewtonControls() if npf_newton_controls is None else npf_newton_controls
        if not isinstance(controls,NPF662NewtonControls):raise ValueError('npf_newton_controls must be NPF662NewtonControls')
        head,info=solve_newton_device(model=model,case=case,reference=reference,max_outer=max_outer,
            krylov_budget=max_cycles,restart=krylov_restart,controls=controls)
        info.update(formulation='mf6_6_6_2_npf',geometry='orthogonal_cartesian_uniform_dz',
            device=model.device,precision='float64',
            implementation='device_newton_gmres',session_backend='npf662_newton',
            reference_contract='NPF662_CARTESIAN_V1_Newton',reference_version='6.6.2',
            effective_options=dict(xt3d=False,newton=True,cell_type=case.cell_types.tolist(),saturation_epsilon=1e-6,
                conductance_floor_m=0.,variablecv=False,perched=False,thickstrt=False,rewet=False,storage=False,packages=['CHD','WEL_SOURCE']))
        return (head,info) if return_info else head
    head,info=model._session.solve(solver=model.solver,device=model.device,implementation=model.implementation,
        unconfined=False,max_cycles=max_cycles,rel_tol=rel_tol,abs_tol_min=abs_tol_min,
        final_budget_relative_tol=final_budget_relative_tol,max_levels=max_levels,min_coarse_n=min_coarse_n,
        dh_rms_tol=None,return_info=True)
    checks,unused=reference.evaluate(head=head)
    info['backend_converged']=bool(info['converged'])
    info['dt']=None  # No timestep exists in this explicitly steady contract.
    info.update(formulation='mf6_6_6_2_npf',geometry='orthogonal_cartesian_uniform_dz',
        device=model.device,precision='float64',
        reference_contract='NPF662_CARTESIAN_V1_confined',reference_version='6.6.2',
        independent_acceptance=checks,converged=bool(checks['independent_accepted']),
        effective_options=dict(xt3d=False,newton=False,icelltype=0,variablecv=False,perched=False,thickstrt=False,
            rewet=False,storage=False,packages=['CHD','WEL_SOURCE']),
        stop_reason='converged' if checks['independent_accepted'] else 'independent_acceptance_failed')
    return (head,info) if return_info else head
