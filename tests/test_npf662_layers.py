"""Layered DIS faces, public solves, supplied heads and lifecycle controls."""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.validation.npf662_reference import (
    NPF662LayeredCase, NPF662Reference, load_npf662_case, save_npf662_case)
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from validate_npf662_confined import make_gpu_model
from test_npf662_confined import confined_case


def layered_case(*, variation='independent', masked=False, shape=(3, 4, 5), seed=121):
    k,j,i = np.indices(shape)
    thickness = 1.+.4*k+.08*i+.1*j
    surface = 10.+.2*i[0]+.15*j[0]
    if variation == 'x':
        thickness = np.broadcast_to(1.+.15*i, shape).copy(); surface[:] = 10.
    elif variation == 'y':
        thickness = np.broadcast_to(1.+.15*j, shape).copy(); surface[:] = 10.
    elif variation == 'vertical':
        thickness = np.broadcast_to((1.+2.*np.arange(shape[0]))[:,None,None],shape).copy(); surface[:] = 10.
    interfaces = np.concatenate([surface[None], surface[None]-np.cumsum(thickness,axis=0)],axis=0)
    conductivity = 10.**np.random.default_rng(seed=seed).uniform(low=-2.,high=2.,size=shape)
    active = np.ones(shape,dtype=int)
    if masked:active[:,shape[1]//2,shape[2]//2] = 0
    fixed = np.zeros(shape,dtype=int); fixed[:,:,0]=1; fixed[:,:,-1]=1
    values = np.broadcast_to(np.linspace(9.,9.5,shape[2]),shape).copy()
    source = np.zeros(shape)
    if shape[2]>1:source[0,0,1]=.003
    return NPF662LayeredCase(kx=conductivity,ky=.6*conductivity,kz=.15*conductivity,
        active=active,fixed=fixed,values=values,source=source,initial=values.copy(),
        dx=2.,dy=3.,z_interfaces=interfaces)


def packed_faces(*, model, reference):
    ids = np.arange(reference.case.active.size).reshape(reference.shape)
    found = {}
    for axis,p in ((2,0),(1,2),(0,4)):
        l,u = [slice(None)]*3,[slice(None)]*3
        l[axis],u[axis]=slice(None,-1),slice(1,None); l,u=tuple(l),tuple(u)
        for a,b,c in zip(ids[l].ravel(),ids[u].ravel(),model._session.faces[p][l].ravel()):
            found[(int(a),int(b))] = c
    return np.array([found[(int(a),int(b))] for a,b in zip(reference.left,reference.right)])


@pytest.mark.parametrize('variation', ['x','y','vertical','independent'])
@pytest.mark.parametrize('device', ['cpu','cuda:0'])
def test_layered_faces_jacobian_and_complete_confined_solve(variation,device):
    case=layered_case(variation=variation,masked=True)
    reference=NPF662Reference(case=case)
    head=case.values+.04*np.random.default_rng(seed=31).random(size=case.active.shape)
    head[case.fixed!=0]=case.values[case.fixed!=0]
    checks,expected=reference.evaluate(head=head)
    with make_gpu_model(case=case,device=device) as model:
        assert model.geometry=='structured_dis_layered' and model._session.dz is None
        np.testing.assert_array_equal(model._session.layer_geometry['centre_z'],.5*case.tops+.5*case.bottoms)
        np.testing.assert_allclose(model._session.layer_geometry['volume'],case.dx*case.dy*case.thicknesses,rtol=3e-15,atol=0.)
        np.testing.assert_allclose(packed_faces(model=model,reference=reference),reference.c0,rtol=3e-13,atol=0.)
        workspace=NPF662Workspace(model=model,case=replace(case,initial=head),restart=8,krylov_budget=100)
        workspace.evaluate(trial=False,linearize=True)
        np.testing.assert_allclose(workspace.residual.numpy().reshape(head.shape),expected['residual'],rtol=3e-12,atol=2e-11)
        v=np.random.default_rng(seed=32).normal(size=head.shape); v[~reference.free]=0.
        x=wp.array(data=v.ravel(),dtype=wp.float64,device=device); z=wp.zeros_like(src=x)
        workspace.matvec(x=x,y=z,z=z,alpha=1.,beta=0.)
        analytic=(reference.jacobian(head=head)@v.ravel()).reshape(head.shape)
        np.testing.assert_allclose((z.numpy()*workspace.scale.numpy()).reshape(head.shape)[reference.free],analytic[reference.free],rtol=3e-12,atol=2e-11)
        workspace.close()
        cpu,ci,unused=reference.solve()
        solved,info=model.solve(max_outer=50,max_cycles=2000)
        assert ci['independent_accepted'] and info['converged'],info
        assert info['preconditioner']=='fixed diagonal within each GMRES solve'
        np.testing.assert_allclose(solved[reference.free],cpu[reference.free],rtol=0.,atol=1e-7)


def test_exact_vertical_series_resistance():
    case=layered_case(variation='vertical',shape=(4,1,1))
    case.kx[:]=1.; case.ky[:]=.6
    case.kz=np.array([2.,.001,8.,.02]).reshape(case.active.shape)
    case.fixed[:]=0; case.fixed[0]=1;case.fixed[-1]=1
    case.values[0]=8.;case.values[-1]=9.;case.source[:]=0.
    thickness=case.thicknesses[:,0,0]
    resistance=.5*thickness[:-1]/case.kz[:-1,0,0]+.5*thickness[1:]/case.kz[1:,0,0]
    expected=8.+np.r_[0.,np.cumsum(resistance)]/resistance.sum()
    cpu,checks,fields=NPF662Reference(case=case).solve()
    assert checks['independent_accepted']
    np.testing.assert_allclose(cpu[:,0,0],expected,rtol=0.,atol=1e-10)
    np.testing.assert_allclose(fields['flux'],-case.dx*case.dy/resistance.sum(),rtol=1e-9,atol=2e-11)


def test_v1_loading_hashes_and_layered_roundtrip(tmp_path):
    original=Path('validation_artifacts/sevenpoint_v1/stage1/final_experiments/x')
    case=load_npf662_case(path=original/'inputs.npz')
    assert case.digest()==json.loads((original/'comparison.json').read_text())['input_hash']
    save_npf662_case(case=case,path=tmp_path/'v1.npz')
    assert load_npf662_case(path=tmp_path/'v1.npz').digest()==case.digest()
    early=Path('validation_artifacts/sevenpoint_v1/stage1/experiments/x')
    legacy=load_npf662_case(path=early/'inputs.npz')
    expected=json.loads((early/'comparison.json').read_text())['input_hash']
    assert legacy.digest()==expected
    save_npf662_case(case=legacy,path=tmp_path/'early_v1.npz')
    with np.load(tmp_path/'early_v1.npz') as saved:assert 'cell_type' not in saved.files
    assert load_npf662_case(path=tmp_path/'early_v1.npz').digest()==expected
    legacy.cell_type=1
    assert legacy.digest()!=expected
    layered=layered_case()
    save_npf662_case(case=layered,path=tmp_path/'v2.npz')
    loaded=load_npf662_case(path=tmp_path/'v2.npz')
    assert loaded.digest()==layered.digest() and loaded.geometry=='structured_dis_layered'
    assert not hasattr(loaded,'dz')


@pytest.mark.parametrize('failure', ['zero','inverted','nonfinite','shape','active_shape','fixed','inactive_source','conductivity'])
def test_invalid_layered_inputs(failure):
    case=layered_case()
    if failure=='zero':case.z_interfaces[1]=case.z_interfaces[0]
    if failure=='inverted':case.z_interfaces[1]=case.z_interfaces[0]+1.
    if failure=='nonfinite':case.z_interfaces[0,0,0]=np.nan
    if failure=='shape':case.z_interfaces=case.z_interfaces[:-1]
    if failure=='active_shape':case.active=case.active.ravel()
    if failure=='fixed':case.active[0,0,0]=0
    if failure=='inactive_source':case.active[0,0,1]=0
    if failure=='conductivity':case.kx[0,0,1]=0.
    with pytest.raises(ValueError):NPF662Reference(case=case)
    with pytest.raises(ValueError):make_gpu_model(case=case,device='cpu')


@pytest.mark.parametrize('device', ['cpu','cuda:0'])
def test_layered_lifecycle_and_cold_reset(device):
    case=layered_case(masked=True)
    with make_gpu_model(case=case,device=device) as model:
        first,one=model.solve(max_outer=50)
        holder=model._session._npf662_workspace_holder['workspace']
        initial=model._session.initial_head.copy()
        case.source*=1.2; case.values[case.fixed!=0]+=.01
        model.update_rhs(rhs=case.source);model.update_boundary_values(bc_values=case.values)
        second,two=model.solve(max_outer=50)
        assert two['workspace_reused']
        np.testing.assert_array_equal(model._session.initial_head,initial)
        with make_gpu_model(case=case,device=device) as fresh:
            expected,unused=fresh.solve(max_outer=50)
        np.testing.assert_allclose(second,expected,rtol=0.,atol=1e-10)
        case.z_interfaces[1]-=.1
        model.build_from_layers(z_interfaces=case.z_interfaces,kx_field=case.kx,ky_field=case.ky,kz_field=case.kz,
            active=case.active,bc_mask=case.fixed,bc_values=case.values,rhs=case.source,initial_head=case.initial)
        third,three=model.solve(max_outer=50)
        assert holder.closed and not three['workspace_reused']
        expected,checks,unused=NPF662Reference(case=case).solve()
        np.testing.assert_allclose(third[case.active!=0],expected[case.active!=0],rtol=0.,atol=1e-7)
        case.active[:,1,2]=0;case.source[:,1,2]=0.;case.fixed[:,1,2]=0
        model.build_from_layers(z_interfaces=case.z_interfaces,kx_field=case.kx,ky_field=case.ky,kz_field=case.kz,
            active=case.active,bc_mask=case.fixed,bc_values=case.values,rhs=case.source,initial_head=case.initial)
        head,four=model.solve(max_outer=50)
        assert four['converged'] and not four['workspace_reused']
        with pytest.raises(ValueError,match='authoritative'):model.solve(top_elevation=10.)
        with pytest.raises(ValueError,match='rebuild'):
            model.update_face_conductance(**dict(zip(('tx_p','tx_m','ty_p','ty_m','tz_p','tz_m'),model._session.faces)))


@pytest.mark.parametrize('variation', ['x','y','independent'])
def test_live_layered_confined_native(tmp_path,variation):
    case=layered_case(variation=variation,masked=True)
    reference=NPF662Reference(case=case)
    cpu,ci,unused=reference.solve()
    head,info,fields=run_reference_mf6(case=case,directory=tmp_path/'root',executable=Path('/bin/modflow/mf6'),head_tolerance_m=1e-12)
    assert info['native_success'] and info['independent_conservation'] and info['independent']['independent_accepted'],info
    np.testing.assert_allclose(head[reference.free],cpu[reference.free],rtol=0.,atol=1e-7)
    head=case.values+.07*np.random.default_rng(seed=3).random(size=case.active.shape)
    head[case.fixed!=0]=case.values[case.fixed!=0]
    unused,info,fields=run_reference_mf6(case=case,directory=tmp_path/'supplied',executable=Path('/bin/modflow/mf6'),prescribed_state=head)
    unused,expected=reference.evaluate(head=head)
    assert info['native_success']
    np.testing.assert_allclose(fields['flux_native'],expected['flux'],rtol=3e-12,atol=2e-11)
    np.testing.assert_allclose(fields['native_residual'],expected['residual'],rtol=3e-12,atol=2e-11)


@pytest.mark.parametrize('device', ['cpu','cuda:0'])
def test_uniform_interfaces_are_compatible_with_scalar_api(device):
    uniform=confined_case()
    interfaces=np.concatenate([uniform.tops[:1],uniform.bottoms],axis=0)
    names=('kx','ky','kz','active','fixed','values','source','initial','dx','dy','cell_type')
    layered=NPF662LayeredCase(**{name:getattr(uniform,name) for name in names},z_interfaces=interfaces)
    old,new=NPF662Reference(case=uniform),NPF662Reference(case=layered)
    np.testing.assert_allclose(new.c0,old.c0,rtol=3e-13,atol=0.)
    head=uniform.values+.1*np.random.default_rng(seed=6).random(size=uniform.active.shape)
    a,af=old.evaluate(head=head);b,bf=new.evaluate(head=head)
    for name in ('saturation','conductance','flux','residual','dflux_left','dflux_right'):
        np.testing.assert_allclose(af[name],bf[name],rtol=3e-12,atol=2e-11)
    with make_gpu_model(case=uniform,device=device) as scalar,make_gpu_model(case=layered,device=device) as explicit:
        hs,si=scalar.solve();he,ei=explicit.solve(max_outer=50)
        assert si['converged'] and ei['converged']
        np.testing.assert_allclose(he,hs,rtol=0.,atol=1e-7)


@pytest.mark.parametrize('device', ['cpu','cuda:0'])
def test_constant_prescribed_layers_and_legacy_dz_is_unused(device):
    case=layered_case(masked=True)
    case.fixed[0]=case.active[0];case.values[:]=9.;case.source[:]=0.;case.initial[:]=9.
    nz,ny,nx=case.active.shape
    with WarpDarcySolver3D(nx=nx,ny=ny,nz=nz,dx=case.dx,dy=case.dy,dz=999.,device=device,
        implementation='fast',formulation='mf6_6_6_2_npf') as model:
        model.build_from_layers(z_interfaces=case.z_interfaces,kx_field=case.kx,ky_field=case.ky,kz_field=case.kz,
            active=case.active,bc_mask=case.fixed,bc_values=case.values,rhs=case.source,initial_head=case.initial)
        head,info=model.solve(abs_tol_min=1e-14,final_budget_relative_tol=1e-12)
        assert info['converged'] and info['final_acceptance_thresholds']['physical_rms_m3_day']==1e-14
        np.testing.assert_array_equal(head[case.active!=0],case.values[case.active!=0])
        model.close()
        model.build_from_K_fields(kx_field=case.kx,ky_field=case.ky,kz_field=case.kz,
            active=case.active,bc_mask=case.fixed,bc_values=case.values,rhs=case.source,initial_head=case.initial)
        assert model.geometry=='cartesian' and model._session.z_interfaces is None
        rebuilt,ri=model.solve()
        assert ri['converged']
        previous=model._session
        model.build_from_layers(z_interfaces=case.z_interfaces,kx_field=case.kx,ky_field=case.ky,kz_field=case.kz,
            active=case.active,bc_mask=case.fixed,bc_values=case.values,rhs=case.source,initial_head=case.initial)
        returned,layered=model.solve()
        assert previous._closed and layered['converged'] and not layered['workspace_reused']


@pytest.mark.parametrize('source', [0.,1.,-1.])
def test_ungrounded_isolated_layered_unknown_is_rejected(source):
    case=layered_case(shape=(1,1,1))
    case.fixed[:]=0;case.source[:]=source
    with pytest.raises(ValueError,match='Ungrounded'):NPF662Reference(case=case).solve()
    with make_gpu_model(case=case,device='cpu') as model:
        with pytest.raises(ValueError,match='Ungrounded'):model.solve()


def test_safe_conductance_when_intermediate_transmissivity_overflows():
    case=layered_case(shape=(1,1,2))
    case.z_interfaces=np.array([1e100,0.])[:,None,None]*np.ones((2,1,2))
    case.dx=1e300;case.dy=1e-200
    case.kx[:]=1e300;case.ky[:]=1e300;case.kz[:]=1e-200;case.source[:]=0.
    reference=NPF662Reference(case=case)
    with make_gpu_model(case=case,device='cpu') as model:
        coefficient=packed_faces(model=model,reference=reference)
        np.testing.assert_allclose(coefficient,1e-100,rtol=3e-13,atol=0.)
        np.testing.assert_allclose(coefficient,reference.c0,rtol=3e-13,atol=0.)


def test_raw_float32_interfaces_are_differenced_in_fp64():
    case=layered_case(shape=(1,1,2))
    case.z_interfaces=np.array([[[1.,2.]],[[-1e8,-1e8]]],dtype=np.float32)
    case.kx[:]=1.;case.ky[:]=1.;case.kz[:]=1.;case.source[:]=0.
    np.testing.assert_array_equal(case.thicknesses,np.array([[[1e8+1.,1e8+2.]]]))
    reference=NPF662Reference(case=case)
    with make_gpu_model(case=case,device='cpu') as model:
        np.testing.assert_allclose(packed_faces(model=model,reference=reference),reference.c0,rtol=3e-13,atol=0.)


def test_internal_vertical_inactive_cell_has_no_passthrough(tmp_path):
    case=layered_case()
    case.active[1,1,2]=0
    reference=NPF662Reference(case=case)
    ids=np.arange(case.active.size).reshape(case.active.shape)
    edges=set(zip(reference.left,reference.right))
    assert (ids[0,1,2],ids[2,1,2]) not in edges
    native,info,fields=run_reference_mf6(case=case,directory=tmp_path/'internal_hole',
        executable=Path('/bin/modflow/mf6'),head_tolerance_m=1e-12)
    assert info['native_success'] and info['independent_conservation']
    assert info['independent']['inactive_marker_cells']==1
    for device in ('cpu','cuda:0'):
        with make_gpu_model(case=case,device=device) as model:
            head,checks=model.solve(max_outer=50)
            assert checks['converged']
            np.testing.assert_allclose(head[reference.free],native[reference.free],rtol=0.,atol=1e-7)
