"""Confined contract: conductance, supplied-head, analytic, solve and reuse gates."""
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import warp as wp
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case,NPF662Reference
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.kernels_3d import apply_A_7point_kernel


def confined_case(*,heterogeneous=False,masked=False):
    shape=(3,5,7)
    k=np.ones(shape)
    if heterogeneous:
        rng=np.random.default_rng(seed=1966)
        k=10.**rng.uniform(low=-3.,high=3.,size=shape)
    active=np.ones(shape,dtype=int)
    if masked:active[:,2,2]=0
    fixed=np.zeros(shape,dtype=int);fixed[:,:,0]=1;fixed[:,:,-1]=1
    values=np.broadcast_to(np.linspace(8.,9.,shape[2]),shape).copy()
    source=np.zeros(shape);source[1,1,3]=.01234567891234567
    return NPF662Case(kx=k,ky=.6*k,kz=.15*k,active=active,fixed=fixed,values=values,
        source=source,initial=np.full(shape,8.),dx=2.,dy=3.,dz=4.)


def make_solver(*,case,device):
    nz,ny,nx=case.active.shape
    solver=WarpDarcySolver3D(nx=nx,ny=ny,nz=nz,dx=case.dx,dy=case.dy,dz=case.dz,
        device=device,implementation='fast',formulation='mf6_6_6_2_npf')
    solver.build_from_K_fields(kx_field=case.kx,ky_field=case.ky,kz_field=case.kz,
        active=case.active,bc_mask=case.fixed,bc_values=case.values,rhs=case.source,initial_head=case.initial)
    return solver


def require_device(*,device):
    if device.startswith('cuda') and not wp.is_cuda_available():pytest.skip('Local CUDA unavailable')


@pytest.mark.parametrize('heterogeneous',[False,True])
def test_individual_conductances_and_conservation(heterogeneous):
    case=confined_case(heterogeneous=heterogeneous,masked=True)
    reference=NPF662Reference(case=case)
    with make_solver(case=case,device='cpu') as solver:
        faces=solver._session.faces
        ids=np.arange(case.active.size).reshape(case.active.shape)
        assembled={}
        for axis,face in zip((2,1,0),(faces[0],faces[2],faces[4])):
            l,u=[slice(None)]*3,[slice(None)]*3;l[axis],u[axis]=slice(None,-1),slice(1,None);l,u=tuple(l),tuple(u)
            for a,b,c in zip(ids[l].ravel(),ids[u].ravel(),face[l].ravel()):assembled[(int(a),int(b))]=c
        actual=np.array([assembled[(int(a),int(b))] for a,b in zip(reference.left,reference.right)])
        np.testing.assert_allclose(actual,reference.c0,rtol=3e-13,atol=0.)
    np.testing.assert_allclose(np.asarray(reference.incidence.sum(axis=0)),0.,atol=0.)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_arbitrary_head_operator_and_independent_complete_solve(device):
    require_device(device=device)
    case=confined_case(heterogeneous=True,masked=True)
    reference=NPF662Reference(case=case)
    rng=np.random.default_rng(seed=12)
    head=rng.uniform(low=7.,high=10.,size=case.active.shape)
    head[case.fixed!=0]=case.values[case.fixed!=0]
    checks,fields=reference.evaluate(head=head)
    expected=(reference.incidence@fields['flux']).reshape(case.active.shape)
    expected[~reference.free]=head[~reference.free]
    with make_solver(case=case,device=device) as solver:
        arrays=[wp.array(data=a,dtype=wp.float64,device=device) for a in solver._session.faces]
        active=wp.array(data=case.active,dtype=wp.int32,device=device)
        fixed=wp.array(data=case.fixed,dtype=wp.int32,device=device)
        h=wp.array(data=head,dtype=wp.float64,device=device);out=wp.zeros_like(src=h);storage=wp.zeros_like(src=h)
        nz,ny,nx=case.active.shape
        wp.launch(kernel=apply_A_7point_kernel,dim=case.active.shape,inputs=[*arrays,active,fixed,storage,h,out,nx,ny,nz],device=device)
        np.testing.assert_allclose(out.numpy(),expected,rtol=3e-12,atol=2e-11)
        cpu,ci,unused=reference.solve()
        gpu,info=solver.solve(max_cycles=2000)
        assert ci['independent_accepted'] and info['converged'],info
        np.testing.assert_allclose(gpu[reference.free],cpu[reference.free],rtol=0.,atol=1e-7)
        assert info['formulation']=='mf6_6_6_2_npf'


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('axis',[0,1,2,None])
def test_analytical_linear_and_constant_solutions(device,axis):
    require_device(device=device)
    case=confined_case()
    case.source[:]=0.
    case.fixed[:]=0
    if axis is None:
        case.values[:]=8.;case.fixed[:,:,0]=1
    else:
        a=[slice(None)]*3;b=a.copy();a[axis]=0;b[axis]=-1
        case.fixed[tuple(a)]=1;case.fixed[tuple(b)]=1
        case.values=8.+.1*np.indices(case.active.shape)[axis]
    reference=NPF662Reference(case=case)
    with make_solver(case=case,device=device) as solver:
        head,info=solver.solve()
    assert info['converged'],info
    np.testing.assert_allclose(head,case.values,rtol=0.,atol=1e-7)


@pytest.mark.parametrize('net_source',[0.,1.,-1.])
def test_ungrounded_components_are_rejected(net_source):
    case=confined_case(masked=True)
    case.active[:,2,:]=0;case.fixed[:,:3,:]=0;case.source[:]=0.;case.source[0,0,0]=net_source
    with pytest.raises(ValueError,match='Ungrounded'):
        NPF662Reference(case=case).solve()
    with make_solver(case=case,device='cpu') as solver:
        with pytest.raises(ValueError,match='Ungrounded'):solver.solve()


def test_sources_and_unsupported_options():
    case=confined_case(masked=True)
    case.source[0,2,2]=.1
    with pytest.raises(ValueError,match='Inactive source'):NPF662Reference(case=case)
    case.source[0,2,2]=0.
    case.source[:,:,0]=.123
    head,result,unused=NPF662Reference(case=case).solve()
    assert result['independent_accepted'] and result['fixed_cell_source_total']>0
    with make_solver(case=case,device='cpu') as solver:
        with pytest.raises(ValueError,match='Unsupported'):solver.solve(transient=True)


def test_independent_reference_imports_without_warp():
    code="""import builtins
original=builtins.__import__
def reject_warp(name,globals=None,locals=None,fromlist=(),level=0):
    if name.startswith('warp'):raise ImportError('Warp forbidden in independent CPU reference')
    return original(name,globals,locals,fromlist,level)
builtins.__import__=reject_warp
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case,NPF662Reference
"""
    subprocess.run([sys.executable,'-c',code],check=True)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_workspace_value_reuse_and_topology_rebuild(device):
    require_device(device=device)
    case=confined_case()
    with make_solver(case=case,device=device) as solver:
        a,first=solver.solve()
        case.source*=2.
        solver.update_rhs(rhs=case.source)
        b,second=solver.solve()
        assert second['workspace_reused']
        cpu,ci,unused=NPF662Reference(case=case).solve()
        np.testing.assert_allclose(b,cpu,rtol=0.,atol=1e-7)
        previous=solver._session
        case.active[:,2,:]=0;case.fixed[:,2,:]=0;case.source[:,2,:]=0.
        solver.build_from_K_fields(kx_field=case.kx,ky_field=case.ky,kz_field=case.kz,
            active=case.active,bc_mask=case.fixed,bc_values=case.values,rhs=case.source,initial_head=case.initial)
        h,third=solver.solve()
        assert previous is not solver._session and not third['workspace_reused']
        assert third['converged']


@pytest.mark.parametrize('heterogeneous,masked',[(False,False),(True,True)])
def test_live_mf6_complete_and_supplied_head(tmp_path,heterogeneous,masked):
    executable=Path('/bin/modflow/mf6')
    if not executable.is_file():pytest.skip('Pinned local MF6 unavailable')
    case=confined_case(heterogeneous=heterogeneous,masked=masked)
    case.source[:,:,0]=.123  # Explicit fixed-reservoir forcing remains in the manifest.
    reference=NPF662Reference(case=case)
    cpu,ci,cf=reference.solve()
    head,info,fields=run_reference_mf6(case=case,directory=tmp_path/'root',executable=executable)
    assert info['native_success'] and info['independent_conservation'],info
    np.testing.assert_allclose(head[reference.free],cpu[reference.free],rtol=0.,atol=1e-7)
    unused,same_head=reference.evaluate(head=head)
    np.testing.assert_allclose(fields['flux_native'],same_head['flux'],rtol=3e-12,atol=2e-11)
    dh=np.abs(head-cpu).ravel()
    propagated=reference.c0*(dh[reference.left]+dh[reference.right])
    assert np.all(np.abs(fields['flux_native']-cf['flux'])<=propagated+2e-11+3e-12*np.abs(cf['flux']))
    supplied=case.values.copy();supplied[reference.free]+=.1*np.random.default_rng(seed=3).normal(size=int(reference.free.sum()))
    head,info,fields=run_reference_mf6(case=case,directory=tmp_path/'supplied',executable=executable,prescribed_state=supplied)
    checks,expected=reference.evaluate(head=supplied)
    assert info['native_success']
    np.testing.assert_allclose(fields['flux_native'],expected['flux'],rtol=3e-12,atol=2e-11)
    np.testing.assert_allclose(fields['native_residual'],expected['residual'],rtol=3e-12,atol=2e-11)
