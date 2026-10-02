"""Head contact retains aquifer storage and returns the accepted FV reaction."""
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
from DARCY_WARP_PACKAGE.solvers.surface_contact_2d import solve_surface_contact_2d
from DARCY_WARP_PACKAGE.physics.storage_2d import exact_unconfined_storage_terms


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('previous,stage,recharge,ponded', [
    (0.01,0.0,0.0,False), (0.01,0.02,0.0,True),
    (0.01,0.01,0.0,True), (0.0,0.0,-0.001,False),
    (0.0,0.0,0.001,False)])
def test_contact_reaction_and_independent_storage(device, previous, stage, recharge, ponded):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    shape=(2,2)
    old=np.full(shape,previous)
    bottom=np.full(shape,-2.)
    top=np.zeros(shape)
    k=np.full(shape,.5)
    with WarpDarcySolver(nx=2,ny=2,dx=10.,device=device) as model:
        model.build_from_fields(T_field=k*2.,R_field=np.full(shape,recharge),
            active=np.ones(shape,np.int32),bc_mask=np.zeros(shape,np.int32),bc_values=top)
        original_mask=model.bc_mask_host.copy()
        arguments=dict(formulation='unconfined',solver='unconfined_semismooth_newton_kcycle',
            initial_head=old,K_field=k,zbot_field=bottom,ztop_field=top,transient=True,
            sy=.1,ss=1e-5,dt=.01,head_prev=old,return_info=True,
            newton_fallback_to_picard=False,max_levels=2,min_coarse_cells=1)
        head,info=solve_surface_contact_2d(model=model,contact_mask=np.ones(shape,bool),
            surface_head=np.full(shape,stage),ponded_mask=np.full(shape,ponded),solve_arguments=arguments)
        assert info['converged']
        q=info['contact_discharge_rate_array']
        storage,_,_=exact_unconfined_storage_terms(head_new=head,head_old=old,
            bottom=bottom,top=top,specific_yield=.1,specific_storage=1e-5,dt=.01)
        np.testing.assert_allclose(storage*100.+q,recharge*100.,atol=1e-7,rtol=0)
        np.testing.assert_allclose(info['storage_total_array'],storage*100.,atol=1e-8)
        np.testing.assert_array_equal(model.bc_mask_host,original_mask)
        if ponded:
            np.testing.assert_allclose(head,stage,atol=0)
        else:
            assert np.all(head<=stage+1e-12)
            assert np.all(q>=-1e-10)
            np.testing.assert_allclose(q*(stage-head),0.,atol=1e-12)
        budget=info['budget_summary']
        assert budget.get('contact_out',0.) == pytest.approx(np.maximum(q,0).sum())
        assert budget.get('contact_in',0.) == pytest.approx(np.maximum(-q,0).sum())
        assert abs(budget['in_minus_out']) < 1e-7


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_supply_limited_contact_reaction_is_real_fv_flux(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    shape=(2,2)
    old=np.zeros(shape)
    bottom=np.full(shape,-2.)
    top=np.zeros(shape)
    k=np.full(shape,.5)
    with WarpDarcySolver(nx=2,ny=2,dx=10.,device=device) as model:
        model.build_from_fields(T_field=k*2.,R_field=np.full(shape,-.001),
            active=np.ones(shape,np.int32),bc_mask=np.zeros(shape,np.int32),bc_values=top)
        arguments=dict(formulation='unconfined',solver='unconfined_semismooth_newton_kcycle',
            initial_head=old,K_field=k,zbot_field=bottom,ztop_field=top,transient=True,
            sy=.1,ss=1e-5,dt=.01,head_prev=old,return_info=True,
            newton_fallback_to_picard=False,max_levels=2,min_coarse_cells=1)
        head,info=solve_surface_contact_2d(model=model,contact_mask=np.ones(shape,bool),
            surface_head=top,ponded_mask=np.zeros(shape,bool),solve_arguments=arguments,
            maximum_inflow_rate_m3_per_day=np.full(shape,.05))
        assert info['converged']
        assert np.all(head<0.)
        np.testing.assert_allclose(info['contact_discharge_rate_array'],-.05,atol=1e-7)
        np.testing.assert_array_equal(model.R_field_host,-.001)
        assert info['budget_summary']['rcha_out']==pytest.approx(.4)
        assert info['budget_summary']['contact_in']==pytest.approx(.2,abs=1e-7)
        assert abs(info['budget_summary']['in_minus_out'])<1e-7
