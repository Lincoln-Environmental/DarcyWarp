"""Opt-in pinned Cartesian NPF contract, retaining the legacy session backend."""
import numpy as np
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case,NPF662Reference


def case_from_model(*,model,top):
    if model.geometry!='cartesian' or any(k is None for k in (model._kx_field,model._ky_field,model._kz_field)):
        raise ValueError('mf6_6_6_2_npf requires Cartesian build_from_K_fields; arbitrary faces/terrain unsupported')
    s=model._session
    return NPF662Case(kx=model._kx_field,ky=model._ky_field,kz=model._kz_field,active=s.active,fixed=s.bc_mask,
        values=s.bc_values,source=s.rhs,initial=s.initial_head if s.initial_head is not None else np.zeros(s.shape),
        dx=model.dx,dy=model.dy,dz=model.dz,top=top)


def solve_npf662(*,model,unconfined=False,top_elevation=0.,max_cycles=2000,rel_tol=1e-12,
    abs_tol_min=1e-12,final_budget_relative_tol=1e-8,max_levels=6,min_coarse_n=2,
    return_info=True,**unsupported):
    if unsupported:raise ValueError(f'Unsupported mf6_6_6_2_npf controls: {sorted(unsupported)}')
    if unconfined:raise NotImplementedError('Matched unconfined law requires the subsequent stage-2 gate')
    case=case_from_model(model=model,top=top_elevation)
    reference=NPF662Reference(case=case)
    reference.check_components()
    head,info=model._session.solve(solver=model.solver,device=model.device,implementation=model.implementation,
        unconfined=False,max_cycles=max_cycles,rel_tol=rel_tol,abs_tol_min=abs_tol_min,
        final_budget_relative_tol=final_budget_relative_tol,max_levels=max_levels,min_coarse_n=min_coarse_n,
        dh_rms_tol=None,return_info=True)
    checks,unused=reference.evaluate(head=head)
    info['backend_converged']=bool(info['converged'])
    info['dt']=None  # No timestep exists in this explicitly steady contract.
    info.update(formulation='mf6_6_6_2_npf',geometry='orthogonal_cartesian_uniform_dz',
        reference_contract='NPF662_CARTESIAN_V1_confined',reference_version='6.6.2',
        independent_acceptance=checks,converged=bool(checks['independent_accepted']),
        effective_options=dict(xt3d=False,newton=False,icelltype=0,variablecv=False,perched=False,thickstrt=False,
            rewet=False,storage=False,packages=['CHD','WEL_SOURCE']),
        stop_reason='converged' if checks['independent_accepted'] else 'independent_acceptance_failed')
    return (head,info) if return_info else head
