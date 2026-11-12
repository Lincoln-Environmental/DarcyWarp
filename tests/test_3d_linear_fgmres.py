"""Cartesian acceleration retains physical ownership, cache refresh and budgets."""
from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from test_3d_ghb_mf6 import ghb_case, build_case
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Reference
from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls


@pytest.mark.parametrize("device",["cpu","cuda:0"])
@pytest.mark.parametrize("precision",["float64","float32"])
def test_fgmres_changes_k_and_ghb_without_stale_graphs(device,precision):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    case=ghb_case(convertible=False)
    controls=NPF662KCycleControls(levels=3,cached_line_factors=True,preconditioner_precision=precision)
    with build_case(case=case,device=device,formulation='legacy') as model:
        for index,(kfactor,cfactor,head_change) in enumerate(((1.,1.,0.),(.4,2.,.1),(2.,0.,0.))):
            current=replace(case,kx=case.kx*kfactor,ky=case.ky*kfactor,kz=case.kz*kfactor,
                ghb_conductance=case.ghb_conductance*cfactor,ghb_head=case.ghb_head+head_change)
            model.update_K_fields(kx_field=current.kx,ky_field=current.ky,kz_field=current.kz)
            model.update_ghb(ghb_conductance=current.ghb_conductance,ghb_head=current.ghb_head)
            head,info=model.solve(linear_acceleration='fgmres',linear_kcycle_controls=controls,
                max_cycles=200,rel_tol=1e-13,abs_tol_min=1e-12,dh_rms_tol=1e-12)
            assert info['converged'],info
            expected,checks,unused=NPF662Reference(case=current).solve()
            assert checks['independent_accepted'],checks
            np.testing.assert_allclose(actual=head,desired=expected,rtol=0.,atol=1e-10)
            assert info['krylov_iterations']<=200
            assert info['final_budget_acceptance']
            if index:assert info['linear_workspace_reused']
        owner=model._session._fast_workspace_holder['workspace']['linear_fgmres']
        basis=owner.flexible
        preconditioner=owner.multigrid
    assert owner.closed and basis.closed and preconditioner.closed


@pytest.mark.parametrize("budget",[0,1,2])
def test_fgmres_budget_is_total_krylov_work(budget):
    case=ghb_case(convertible=False)
    with build_case(case=case,device='cpu',formulation='legacy') as model:
        head,info=model.solve(linear_acceleration='fgmres',max_cycles=budget,
            rel_tol=1e-14,abs_tol_min=1e-14,dh_rms_tol=None)
        assert info['krylov_iterations']<=budget
        assert not info['converged']
        if budget==0:np.testing.assert_array_equal(actual=head,desired=case.initial)


@pytest.mark.parametrize("options",[
    dict(linear_acceleration='unknown'),dict(linear_krylov_restart=1),
    dict(linear_acceleration='fgmres',linear_kcycle_controls={}),dict(linear_krylov_restart=3),
])
def test_invalid_linear_acceleration_controls_raise(options):
    case=ghb_case(convertible=False)
    with build_case(case=case,device='cpu',formulation='legacy') as model:
        with pytest.raises(ValueError):model.solve(**options)
